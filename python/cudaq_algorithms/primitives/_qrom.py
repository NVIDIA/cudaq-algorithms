# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coherent classical-table lookup with SELECT, SELECT-SWAP and SELECT-COPY.

``QROM(data, address_bits, output_bits).kernel`` is a device kernel with
signature ``(address: qview, ladder: qview, output: qview)``. It applies
``|k>|0>|y> -> |k>|0>|y XOR data[k]>``, using little-endian registers.
Addresses beyond the table read zero. By default all ``num_ladder``
ancillas start clean. ``dirty_banks=True`` permits an arbitrary bank
suffix while requiring the walk prefix to start clean. Arbitrary output
states and external controls are supported in both modes.
The views need not be contiguous, but must be disjoint; the address view
is exactly ``address_bits`` wide, the ladder and output views at least
``num_ladder`` and ``output_bits`` wide. Reapplying the kernel uncomputes
a lookup on its required sector.
No involution guarantee is made for dirty walk ancillas.

Constructions and register layout
---------------------------------

``select`` uses one unary-iteration walk (Babbush et al.,
`arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_). Its ladder has
``address_bits`` clean lines.

``select_swap`` uses the SELECT-SWAP construction (Low, Kliuchnikov and
Schaeffer, `arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`_).
For ``B = block_size``, the low ``l = log2(B)`` address bits select an
entry within a block and the high ``h = address_bits - l`` bits select
one of ``ceil(N/B)`` blocks. The ladder contains ``h`` walk lines, then
``B`` registers of ``b = output_bits`` bits. One block walk writes the
bank, a controlled swap network routes the selected entry to slot zero,
CNOTs copy it into the output, and routing and writes are undone.

``select_copy`` adapts the multiplexed-copy and sequential-packet ideas
of Motlagh and Pocrnic,
`arXiv:2605.20334 <https://arxiv.org/abs/2605.20334>`_, Sections II.B-D,
to clean ancillas and unitary uncomputation. Entry zero is written
directly into the output; the bank stores each other entry XOR entry
zero. A low-address walk copies the selected difference into the output,
then a high-address walk clears the bank. The two walks share a pool of
``max(h, l)`` clean lines. ``alpha`` balanced packets (widest first)
reuse ``(B - 1) * ceil(b/alpha)`` bank qubits after that pool. Adjacent
packet unwrite/write walks are fused using XOR differences, so there
are ``alpha + 1`` high-address walks. ``alpha`` need not divide ``b``.

``auto`` enumerates all variants, valid power-of-two block sizes, and
packet counts ``1 <= alpha <= b``. Candidates exceeding
``max_ancillas`` are excluded. Lowest Toffoli count wins; ties prefer
SELECT, then SELECT-COPY, then SELECT-SWAP, then smaller ``alpha`` and
smaller blocks. The budget counts the whole ladder, including the bank,
but excludes address and output registers. Forced choices are validated
against the same budget.

``auto`` prices Toffolis alone, so it may return a blocked construction
with more workspace than SELECT or SELECT-SWAP would use; bound it with
``max_ancillas``. At one output bit the copy walk ``W(l, B) = 3B/2 - 5``
outgrows the routing it replaces, ``b (B - 1)``, above ``B = 8`` (tie at
8), so SELECT-SWAP is strictly cheaper there.

Logical resource model
----------------------

Let ``W(n, m)`` be the exact coherent unary-walk Toffoli count (including
partial trees), ``N = len(data)``, ``B = 2**l``, and ``h = A - l``:

* SELECT: ``W(A, N)``; ``A`` clean ancillas.
* SELECT-SWAP: ``2 W(h, ceil(N/B)) + 2 b (B - 1)``;
  ``h + B b`` clean ancillas.
* SELECT-COPY: ``(alpha + 1) W(h, ceil(N/B)) + alpha W(l, B)
  + b (B - 1)``; ``max(h, l) + (B - 1) ceil(b/alpha)`` clean ancillas.

These are logical circuit counts, checked against the compiler before
any added external control, rather than hardware costs. At fixed block
size, increasing ``alpha`` never reduces the count; budget constraints
can make narrower packets and a larger block optimal. For example,
``N=1024, b=4, max_ancillas=24`` selects ``B=16, alpha=4`` at 591
Toffolis, versus 253 for unconstrained ``B=32, alpha=1``.

Borrowed banks
--------------

``dirty_banks=True`` requires an explicit ``select_swap`` or
``select_copy`` variant. The ladder is ``[num_clean walk lines |
num_dirty borrowed bank lines]``. The bank may be entangled with other
registers and is restored exactly. SELECT-SWAP adds a second
route/copy/unroute after bank restoration, cancelling its original value
in the output: cost ``2 W(h,ceil(N/B)) + 4 b (B-1)``. SELECT-COPY adds
one low-address walk copying the restored bank into every packet:
cost ``(alpha+1) W(h,ceil(N/B)) + (alpha+1) W(l,B) + 2 b (B-1)``.
These echoes add no workspace. The budget still bounds total
``num_ladder``, and candidates are priced using the requested mode.

The original paper's headline costs additionally use measurement-assisted
temporary-AND disposal. Its ``Sel`` adjoint restores the dirty bank; the
final Restore copy removes repeated dirty offsets from the output. This
implementation uses a coherent echo, without the paper's cached offsets
or temporary-AND fanout optimization. Balanced packets also differ from
Appendix A's fixed packet width plus final remainder. See the QROM guide
for these boundaries and a runnable controlled-rotation chain.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._unary_iteration import (_OP_BODY_X, _OP_CCX, _OP_CCX_ADDR_ADDR,
                               _OP_CCX_CTRL, _OP_CCX_LADDER_LADDER_TARGET,
                               _OP_CX_ADDR_ADDR, _OP_CX_ADDR_LADDER,
                               _OP_CX_LADDER_LADDER, _OP_CX_LADDER_TARGET,
                               _OP_DESCRIPTIONS, _OP_X_ADDR, _TOFFOLI_OPCODES,
                               _emit_walk, _mint_interpreter,
                               _walk_toffoli_count, unary_iteration_kernels)

__all__ = ["QROM"]

_VARIANTS = ("auto", "select", "select_swap", "select_copy")
_BLOCK_VARIANTS = ("select_swap", "select_copy")

# Walk opcodes whose (a, b) operands index the address register, used to
# shift the block-index walk onto the high address bits.
_ADDRESS_OPERANDS = {
    _OP_X_ADDR: (0, ),
    _OP_CX_ADDR_LADDER: (0, ),
    _OP_CCX: (1, ),
    _OP_CCX_CTRL: (1, ),
    _OP_CX_ADDR_ADDR: (0, 1),
    _OP_CCX_ADDR_ADDR: (0, 1),
}


def _select_swap_cost(address_bits: int,
                      num_entries: int,
                      output_bits: int,
                      block_size: int,
                      dirty_banks: bool = False) -> int:
    """Documented Toffoli price of the select_swap variant."""
    high_bits = address_bits - (block_size.bit_length() - 1)
    num_blocks = -(-num_entries // block_size)
    copies = 4 if dirty_banks else 2
    return (2 * _walk_toffoli_count(high_bits, num_blocks) +
            copies * output_bits * (block_size - 1))


def _select_copy_cost(address_bits: int,
                      num_entries: int,
                      output_bits: int,
                      block_size: int,
                      alpha: int = 1,
                      dirty_banks: bool = False) -> int:
    """Documented Toffoli price of select_copy with ``alpha`` packets."""
    low_bits = block_size.bit_length() - 1
    high_bits = address_bits - low_bits
    num_blocks = -(-num_entries // block_size)
    echo = int(dirty_banks)
    return ((alpha + 1) * _walk_toffoli_count(high_bits, num_blocks) +
            (alpha + echo) * _walk_toffoli_count(low_bits, block_size) +
            (1 + echo) * output_bits * (block_size - 1))


def _select_swap_ancillas(address_bits: int, output_bits: int,
                          block_size: int) -> int:
    """Clean ancillas (num_ladder) of the select_swap variant."""
    low_bits = block_size.bit_length() - 1
    return address_bits - low_bits + block_size * output_bits


def _select_copy_ancillas(address_bits: int,
                          output_bits: int,
                          block_size: int,
                          alpha: int = 1) -> int:
    """Clean ancillas (num_ladder) of select_copy with ``alpha`` packets."""
    low_bits = block_size.bit_length() - 1
    packet = -(-output_bits // alpha)
    return max(address_bits - low_bits, low_bits) + (block_size - 1) * packet


def _normalize_table(data, address_bits, output_bits):
    """Validate original inputs once, before any XOR preprocessing."""
    if int(address_bits) != address_bits or address_bits < 1:
        raise ValueError("address_bits must be a positive integer")
    if int(output_bits) != output_bits or output_bits < 1:
        raise ValueError("output_bits must be a positive integer")
    address_bits = int(address_bits)
    output_bits = int(output_bits)
    # Materialize first: a single-pass iterable must not be consumed
    # by one check and leave the others looking at an empty iterator.
    raw = list(data)
    if len(raw) == 0:
        raise ValueError("data must be non-empty")
    if any(int(v) != v for v in raw):
        raise ValueError("data entries must be integers")
    entries = [int(v) for v in raw]
    if len(entries) > (1 << address_bits):
        raise ValueError(f"data has {len(entries)} entries but address_bits="
                         f"{address_bits} addresses only {1 << address_bits}")
    if any(v < 0 for v in entries):
        raise ValueError("data entries must be non-negative")
    if max(entries) >= (1 << output_bits):
        raise ValueError(
            f"data entry {max(entries)} does not fit in output_bits="
            f"{output_bits} bits (max representable "
            f"{(1 << output_bits) - 1})")
    return entries, address_bits, output_bits


class QROM:
    """Coherent lookup of a classical integer table (see module docstring).

    Parameters
    ----------
    data
        The table: non-negative integers, one per address, each fitting
        in ``output_bits`` bits.
    address_bits
        Address register width (``len(data) <= 2^address_bits``).
    output_bits
        Output register width.
    variant
        ``"auto"`` (default), ``"select"``, ``"select_swap"`` or
        ``"select_copy"`` — see the module docstring for the
        constructions and their exact prices. ``"auto"`` minimises the
        logical Toffoli count alone and may choose more workspace than
        an explicit SELECT or SELECT-SWAP; see ``max_ancillas``.
    block_size
        SELECT-SWAP / SELECT-COPY block size: a power of two in
        ``[2, 2^(address_bits - 1)]``. Only valid with
        ``variant="select_swap"`` or ``variant="select_copy"``; ``None``
        there picks the power of two minimizing the documented Toffoli
        count.
    alpha
        Number of sequential bit packets (`arXiv:2605.20334`, Sec. II.D)
        in ``[1, output_bits]``; need not divide ``output_bits`` (the
        slices are balanced, widest first). Only valid with
        ``variant="select_copy"``; ``None`` there enumerates every
        ``alpha`` alongside ``block_size`` (with unconstrained ancillas
        the enumeration provably lands on ``alpha = 1`` — packets win
        only under ``max_ancillas``, see the module docstring).
    max_ancillas
        Optional bound on ``num_ladder``: candidates needing more total
        workspace qubits (clean plus borrowed) are dropped, and a
        ``ValueError`` is raised when nothing fits (or when explicitly
        forced parameters exceed the bound).
    dirty_banks
        Explicit opt-in for borrowed bank qubits with ``select_swap``
        or ``select_copy``. The first ``num_clean`` ladder lines must
        start zero; the remaining ``num_dirty`` lines can hold arbitrary
        states, including entanglement, and are restored. Defaults to
        ``False`` (all workspace clean). See the module's echo costs.

    The three views of the minted kernel may live anywhere (no contiguity
    requirement between them), which is what lets callers weave the QROM
    into their own ancilla layouts. Whatever the variant, applying the
    kernel again uncomputes the lookup (see the module docstring for the
    exact inverse claims).
    """

    def __init__(self,
                 data: Sequence[int],
                 address_bits: int,
                 output_bits: int,
                 *,
                 variant: str = "auto",
                 block_size: int | None = None,
                 alpha: int | None = None,
                 max_ancillas: int | None = None,
                 dirty_banks: bool = False) -> None:
        entries, address_bits, output_bits = _normalize_table(
            data, address_bits, output_bits)
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS}, got {variant!r}")
        if not isinstance(dirty_banks, bool):
            raise ValueError("dirty_banks must be a bool")
        if dirty_banks and variant not in _BLOCK_VARIANTS:
            raise ValueError("dirty_banks=True requires variant='select_swap' "
                             "or variant='select_copy'")
        if variant in _BLOCK_VARIANTS and address_bits < 2:
            raise ValueError(
                f"variant={variant!r} requires address_bits >= 2 (no "
                "block-index bit exists at address_bits=1); use "
                "variant='select' or 'auto'")
        if block_size is not None:
            if variant not in _BLOCK_VARIANTS:
                raise ValueError(
                    "block_size is only valid with variant='select_swap' or "
                    f"variant='select_copy', got variant={variant!r}")
            if (isinstance(block_size, bool) or int(block_size) != block_size
                    or block_size < 2
                    or int(block_size) & (int(block_size) - 1)):
                raise ValueError("block_size must be a power of two >= 2, got "
                                 f"{block_size}")
            block_size = int(block_size)
            if block_size > (1 << (address_bits - 1)):
                raise ValueError(
                    f"block_size {block_size} needs at least one block-"
                    f"index address bit: max is 2^(address_bits - 1) = "
                    f"{1 << (address_bits - 1)}")
        if alpha is not None:
            if variant != "select_copy":
                raise ValueError(
                    "alpha (bit packets) is only valid with "
                    f"variant='select_copy', got variant={variant!r}")
            if (isinstance(alpha, bool) or int(alpha) != alpha
                    or not 1 <= alpha <= output_bits):
                raise ValueError(
                    f"alpha must be an integer in [1, output_bits = "
                    f"{output_bits}], got {alpha}")
            alpha = int(alpha)
        if max_ancillas is not None:
            if (isinstance(max_ancillas, bool)
                    or int(max_ancillas) != max_ancillas or max_ancillas < 1):
                raise ValueError("max_ancillas must be a positive integer, "
                                 f"got {max_ancillas}")
            max_ancillas = int(max_ancillas)

        # Enumerate the strategy space (variant, block_size, alpha). The
        # ancilla formulas are O(1), so candidates over the budget are
        # dropped before anything is priced; the prices are closed forms
        # too (see _walk_toffoli_count). Ties: select (no bank
        # allocation), then select_copy over select_swap (no routing-CNOT
        # storm), then smaller alpha, then smaller blocks.
        block_candidates = [1 << s for s in range(1, address_bits)
                            ] if block_size is None else [block_size]
        alpha_candidates = list(range(1, output_bits +
                                      1)) if alpha is None else [alpha]
        candidates = []  # (cost, rank, alpha, block, variant, ancillas)
        enumerated = []  # every candidate's ancilla count, for the error

        def admit(ancillas: int) -> bool:
            enumerated.append(ancillas)
            return max_ancillas is None or ancillas <= max_ancillas

        if variant in ("auto", "select") and admit(address_bits):
            candidates.append((_walk_toffoli_count(address_bits, len(entries)),
                               0, 0, 0, "select", address_bits))
        if variant in ("auto", "select_copy"):
            for size in block_candidates:
                for a in alpha_candidates:
                    ancillas = _select_copy_ancillas(address_bits, output_bits,
                                                     size, a)
                    if admit(ancillas):
                        candidates.append(
                            (_select_copy_cost(address_bits, len(entries),
                                               output_bits, size, a,
                                               dirty_banks), 1, a, size,
                             "select_copy", ancillas))
        if variant in ("auto", "select_swap"):
            for size in block_candidates:
                ancillas = _select_swap_ancillas(address_bits, output_bits,
                                                 size)
                if admit(ancillas):
                    candidates.append(
                        (_select_swap_cost(address_bits, len(entries),
                                           output_bits, size, dirty_banks), 2,
                         0, size, "select_swap", ancillas))
        if not candidates:
            requested = "" if variant == "auto" else f"{variant!r} "
            raise ValueError(f"no {requested}construction fits max_ancillas="
                             f"{max_ancillas}: the smallest candidate needs "
                             f"{min(enumerated)} workspace qubits")
        (best_cost, _, best_alpha, best_size, variant,
         best_ancillas) = min(candidates)

        self._data = tuple(entries)
        self._num_address = address_bits
        self._output_bits = output_bits
        self._variant = variant
        self._block_size = best_size if variant in _BLOCK_VARIANTS else None
        self._alpha = best_alpha if variant == "select_copy" else None
        self._dirty_banks = dirty_banks
        if variant == "select":
            self._build_select()
        elif variant == "select_swap":
            self._build_select_swap()
        else:
            self._build_select_copy()
        if (self._toffoli_count, self._num_ladder) != (best_cost,
                                                       best_ancillas):
            raise RuntimeError(  # pricer/builder drift is a library bug
                f"internal error: priced {best_cost} Toffolis on "
                f"{best_ancillas} workspace qubits but built "
                f"{self._toffoli_count} on {self._num_ladder}")

    def _build_select(self) -> None:
        entries = self._data
        output_bits = self._output_bits

        def body(k: int) -> list[tuple[str, int]]:
            return [("x", t) for t in range(output_bits)
                    if (entries[k] >> t) & 1]

        # On the clean-ladder sector, XOR lookup is its own inverse.
        walk = unary_iteration_kernels(self._num_address,
                                       len(entries),
                                       body,
                                       include_adjoint=False)
        self._kernel = walk.kernel
        self._num_ladder = walk.num_ladder
        self._ops = tuple(walk.ops)
        self._toffoli_count = walk.toffoli_count

    def _build_select_swap(self) -> None:
        entries = self._data
        b = self._output_bits
        size = self._block_size
        low_bits = size.bit_length() - 1
        high_bits = self._num_address - low_bits
        num_blocks = -(-len(entries) // size)

        def block_write(j: int) -> list[tuple[str, int]]:
            gates = []
            for i in range(size):
                k = j * size + i
                word = entries[k] if k < len(entries) else 0
                gates.extend(
                    ("x", i * b + t) for t in range(b) if (word >> t) & 1)
            return gates

        walk_ops = _emit_walk(high_bits, num_blocks, False, block_write)
        # Rebase the walk onto the combined kernel: block-index address
        # wires sit above the low routing bits, and the walk's "target"
        # (the block registers) lives in the ladder view after the walk
        # lines, so the leaf-controlled body X becomes a ladder-to-ladder
        # CNOT.
        ops = []
        for op in walk_ops:
            opcode, a, bb, c = op
            if opcode == _OP_BODY_X:
                ops.append((_OP_CX_LADDER_LADDER, a, high_bits + bb, 0))
                continue
            operands = [a, bb, c]
            for position in _ADDRESS_OPERANDS.get(opcode, ()):
                operands[position] += low_bits
            ops.append((opcode, *operands))
        walk_ops = ops

        # Binary routing network: after stage s (controlled on
        # address[s]), block slot i (i = 0 mod 2^(s+1)) holds the entry
        # at low-address offset (i + a_s 2^s + ... + a_0); slot 0 ends
        # holding the selected entry. Each controlled register swap is b
        # Fredkins: cswap(c; u, v) = cx(v, u) ccx(c, u, v) cx(v, u).
        swap_ops = []
        for s in range(low_bits):
            for i in range(0, size, 1 << (s + 1)):
                for t in range(b):
                    u = high_bits + i * b + t
                    v = high_bits + (i + (1 << s)) * b + t
                    swap_ops.append((_OP_CX_LADDER_LADDER, v, u, 0))
                    swap_ops.append((_OP_CCX, u, s, v))
                    swap_ops.append((_OP_CX_LADDER_LADDER, v, u, 0))
        copy_ops = [(_OP_CX_LADDER_TARGET, high_bits + t, t, 0)
                    for t in range(b)]

        # W S C S^-1 W: write, route, copy, unroute, unwrite — clean
        # ancillas and a self-inverse lookup on the clean-ladder sector.
        ops = (walk_ops + swap_ops + copy_ops + list(reversed(swap_ops)) +
               walk_ops)
        if self._dirty_banks:
            # The bank is now its original arbitrary state D. Copy its
            # selected word once more to cancel D's first contribution.
            ops += swap_ops + copy_ops + list(reversed(swap_ops))
        self._kernel = _mint_interpreter(ops, controlled=False, has_work=False)
        self._num_ladder = high_bits + size * b
        self._ops = tuple(ops)
        self._toffoli_count = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)

    def _build_select_copy(self) -> None:
        entries = self._data
        b = self._output_bits
        size = self._block_size
        alpha = self._alpha
        low_bits = size.bit_length() - 1
        high_bits = self._num_address - low_bits
        num_blocks = -(-len(entries) // size)
        # The block-index and low-address walks run sequentially and both
        # restore their ladder lines, so they share one line pool; the
        # B - 1 block registers (block 0 rides the direct output write)
        # sit after it, one packet wide.
        pool = max(high_bits, low_bits)
        # Balanced packet slices, widest first (alpha need not divide b);
        # the shared register bank is one packet stride per block.
        widths = [
            b // alpha + (1 if s < b % alpha else 0) for s in range(alpha)
        ]
        offsets = [sum(widths[:s]) for s in range(alpha)]
        packet = widths[0]

        def packet_word(s: int, k: int) -> int:
            word = entries[k] if k < len(entries) else 0
            return (word >> offsets[s]) & ((1 << widths[s]) - 1)

        def block_body(prev: int | None, nxt: int | None):
            # Body for one block-index walk: direct-write the next
            # packet's entry (j, 0) into its output slice (tokens < b)
            # and XOR-transition the register bank from packet ``prev``'s
            # contents to packet ``nxt``'s (tokens >= b): the paper's
            # Sec. II.C sequential fusion, unwrite and write sharing one
            # walk. prev=None is the first write, nxt=None the final
            # unwrite. Block register i - 1 holds packet s of entry (j, i)
            # XOR'd against entry (j, 0) (the Sel1 refinement, Sec. II.B).
            def body(j: int) -> list[tuple[str, int]]:
                gates = []
                base_prev = 0 if prev is None else packet_word(prev, j * size)
                base_nxt = 0 if nxt is None else packet_word(nxt, j * size)
                if nxt is not None:
                    gates.extend(("x", offsets[nxt] + t)
                                 for t in range(widths[nxt])
                                 if (base_nxt >> t) & 1)
                for i in range(1, size):
                    word = 0
                    if prev is not None:
                        word ^= packet_word(prev, j * size + i) ^ base_prev
                    if nxt is not None:
                        word ^= packet_word(nxt, j * size + i) ^ base_nxt
                    if word:
                        gates.extend(("x", b + (i - 1) * packet + t)
                                     for t in range(packet) if (word >> t) & 1)
                return gates

            return body

        def block_walk(prev: int | None, nxt: int | None) -> list:
            # Rebase the block-index walk exactly as select_swap does
            # (walk lines lead the ladder view, block-index address wires
            # sit above the low bits); direct output writes stay
            # leaf-controlled body X's, register writes become
            # ladder-to-ladder CNOTs.
            ops = []
            for op in _emit_walk(high_bits, num_blocks, False,
                                 block_body(prev, nxt)):
                opcode, a, bb, c = op
                if opcode == _OP_BODY_X:
                    if bb < b:
                        ops.append((_OP_BODY_X, a, bb, 0))
                    else:
                        ops.append((_OP_CX_LADDER_LADDER, a, pool + bb - b, 0))
                    continue
                operands = [a, bb, c]
                for position in _ADDRESS_OPERANDS.get(opcode, ()):
                    operands[position] += low_bits
                ops.append((opcode, *operands))
            return ops

        def copy_walk(s: int) -> list:
            # The multiplexed copy for packet s: one walk over the low
            # address bits (its wires are address[0 .. low_bits) already —
            # no rebase) whose body at r >= 1 Toffoli-copies block
            # register r - 1 into output slice s; the r = 0 body is empty
            # (the direct write covered it).
            def copy_body(r: int) -> list[tuple[str, int]]:
                if r == 0:
                    return []
                return [("x", (r - 1) * packet + t) for t in range(widths[s])]

            ops = []
            for op in _emit_walk(low_bits, size, False, copy_body):
                opcode, a, bb, c = op
                if opcode == _OP_BODY_X:
                    ops.append((_OP_CCX_LADDER_LADDER_TARGET, a, pool + bb,
                                offsets[s] + bb % packet))
                else:
                    ops.append(op)
            return ops

        # FW_0 C_0 FW_1 C_1 ... C_{alpha-1} FW_alpha: write packet 0,
        # copy it out, XOR-transition to packet 1, copy it out, ...,
        # unwrite packet alpha - 1 — alpha + 1 block walks for alpha
        # packets. At alpha = 1 this is exactly the W1 C W2 sandwich.
        ops = block_walk(None, 0)
        for s in range(alpha):
            ops += copy_walk(s)
            ops += block_walk(s, s + 1 if s + 1 < alpha else None)
        if self._dirty_banks:
            # The original bank D is restored. Each output packet has
            # inherited the same low packet-width bits of selected D_r.
            # One low-address walk cancels all those copies coherently.
            copies = [(pool + (r - 1) * packet + t, offsets[s] + t)
                      for r in range(1, size) for s in range(alpha)
                      for t in range(widths[s])]

            def echo_body(r: int) -> list[tuple[str, int]]:
                if r == 0:
                    return []
                return [("x", (r - 1) * b + t) for t in range(b)]

            for opcode, a, bb, c in _emit_walk(low_bits, size, False,
                                               echo_body):
                if opcode == _OP_BODY_X:
                    bank, output = copies[bb]
                    ops.append((_OP_CCX_LADDER_LADDER_TARGET, a, bank, output))
                else:
                    ops.append((opcode, a, bb, c))
        self._kernel = _mint_interpreter(ops, controlled=False, has_work=False)
        self._num_ladder = pool + (size - 1) * packet
        self._ops = tuple(ops)
        self._toffoli_count = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    @property
    def num_address(self) -> int:
        return self._num_address

    @property
    def num_ladder(self) -> int:
        """Total ladder width, with a clean prefix and optional borrowed bank.

        ``address_bits`` walk lines for ``"select"``; walk lines plus the
        ``block_size * output_bits`` block registers for
        ``"select_swap"``; a ``max(high_bits, log2(block_size))`` shared
        walk-line pool plus ``(block_size - 1) * ceil(output_bits /
        alpha)`` block registers for ``"select_copy"``.
        ``num_clean`` and ``num_dirty`` specify the initialization split.
        """
        return self._num_ladder

    @property
    def num_output(self) -> int:
        return self._output_bits

    @property
    def dirty_banks(self) -> bool:
        """Whether the bank suffix may hold arbitrary borrowed qubits."""
        return self._dirty_banks

    @property
    def num_clean(self) -> int:
        """Required zero-in/zero-out prefix of the ladder register."""
        if not self._dirty_banks:
            return self._num_ladder
        low = self._block_size.bit_length() - 1
        high = self._num_address - low
        return high if self._variant == "select_swap" else max(high, low)

    @property
    def num_dirty(self) -> int:
        """Borrowable ladder suffix, restored even when entangled."""
        return self._num_ladder - self.num_clean

    @property
    def variant(self) -> str:
        """The minted construction: 'select', 'select_swap' or 'select_copy'."""
        return self._variant

    @property
    def block_size(self) -> int | None:
        """SELECT-SWAP / SELECT-COPY block size (``None`` for select)."""
        return self._block_size

    @property
    def alpha(self) -> int | None:
        """Number of sequential bit packets (``None`` unless select_copy)."""
        return self._alpha

    @property
    def toffoli_count(self) -> int:
        return self._toffoli_count

    @property
    def kernel(self):
        """The lookup kernel ``(address, ladder, output)``.

        Applying twice is the identity with the required ``num_clean``
        prefix initialized to zero; the bank suffix may be borrowed when
        explicitly enabled.
        """
        return self._kernel

    def describe(self) -> str:
        """Decode the minted lookup into a human-readable gate listing.

        A header stating the chosen construction and its price, then one
        line per gate in execution order — exactly what the interpreter
        kernel replays; lines tagged ``# Toffoli`` are the gates
        ``toffoli_count`` counts. This is the intended way to *read* a
        minted lookup (the kernel source is a generic interpreter).
        """
        lines = [
            _OP_DESCRIPTIONS[op].format(a=a, b=b, c=c)
            for op, a, b, c in self._ops
        ]
        return "\n".join([f"QROM lookup: {self!r}"] + lines)

    def __repr__(self) -> str:
        return (f"QROM(entries={len(self._data)}, "
                f"address_bits={self.num_address}, "
                f"output_bits={self.num_output}, "
                f"variant={self._variant!r}, "
                f"block_size={self._block_size}, "
                f"alpha={self._alpha}, "
                f"dirty_banks={self._dirty_banks}, "
                f"ancillas={self._num_ladder}, "
                f"toffolis={self.toffoli_count})")
