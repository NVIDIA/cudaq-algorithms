# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical-table XOR lookup with measured clean-tree uncomputation.

This separate class opts into dynamic circuits; existing ``QROM`` and its
consumers remain unitary. Both SELECT and clean SELECT-SWAP are provided.
For ``n`` address bits, ``m`` entries and external control flag ``c``, let
``C(n,m,c) = m + n - 2 + c - popcount(m-1)``. SELECT costs ``C`` Toffolis
and measurements with ``n`` clean ancillas. SELECT-SWAP, using ``B`` entries
per block, ``h = n-log2(B)`` and ``M = ceil(m/B)``, costs
``2*C(h,M,c) + 2*output_bits*(B-1)`` Toffolis, ``2*C(h,M,c)`` measurements,
and ``h + B*output_bits`` clean ancillas by default. The two measured tree
traversals write and erase the cache; routing and unrouting stay coherent.
With ``dirty_banks=True`` and explicit ``variant="select_swap"``, the final
routing/copy/unrouting echo restores a borrowed bank in an arbitrary joint
state. Its cost is ``2*C(h,M,c) + 4*output_bits*(B-1)`` Toffolis, with ``h``
clean lines and ``B*output_bits`` dirty lines. Measurement-erased caches
with separate phase-correction oracles are not implemented.

Resource properties describe emitted logical operations, not physical
hardware cost or compiled T gates. A 4T temporary-AND model applies only
to ``tree_and_count`` after a compatible synthesis; the implementation
emits ordinary Toffolis. Tree lines must always start clean. The routing
network's targets are bank lines, so a bank need not be clean for the
circuit to be well defined, but only ``dirty_banks=True`` restores a
non-clean bank and keeps its word out of the output; with the default, a
non-clean bank is restored while its selected word leaks into the output.

Construction follows Babbush et al., arXiv:1805.03662, Sec. III.A,
Gidney, arXiv:1709.06648, Fig. 3, and the clean SELECT-SWAP composition
of Low, Kliuchnikov and Schaeffer, arXiv:1812.00954. See
``docs/sphinx/guide/measured_qrom.rst`` for execution examples and boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._qrom import _ADDRESS_OPERANDS
from ._unary_iteration import (_OP_BODY_X, _OP_CCX, _OP_CCX_CTRL,
                               _OP_CX_LADDER_LADDER, _OP_CX_LADDER_TARGET,
                               _OP_DESCRIPTIONS, _OP_MEAS_UNCOMPUTE,
                               _OP_MEAS_UNCOMPUTE_CTRL, _TOFFOLI_OPCODES,
                               _mint_interpreter)
from ._unary_iteration_measured import (_MEASURED_OPCODES, _emit_measured_walk,
                                        _measured_walk_toffoli_count,
                                        measured_unary_iteration_kernels)

__all__ = ["MeasuredQROM"]

_VARIANTS = ("auto", "select", "select_swap")

_MEASURED_ADDRESS_OPERANDS = {
    **_ADDRESS_OPERANDS,
    _OP_MEAS_UNCOMPUTE: (2, ),
    _OP_MEAS_UNCOMPUTE_CTRL: (2, ),
}


def _select_swap_cost(address_bits: int,
                      num_entries: int,
                      output_bits: int,
                      block_size: int,
                      controlled: bool = False,
                      dirty_banks: bool = False) -> int:
    """Documented Toffoli price of the select_swap variant."""
    high_bits = address_bits - (block_size.bit_length() - 1)
    num_blocks = -(-num_entries // block_size)
    routing = (4 if dirty_banks else 2) * output_bits * (block_size - 1)
    return 2 * _measured_walk_toffoli_count(high_bits, num_blocks,
                                            controlled) + routing


class MeasuredQROM:
    """Explicitly opt in to measurement-based tree uncomputation.

    With default clean workspace, ``kernel`` maps ``|k>|0...0>|y>`` to
    ``|k>|0...0>|y XOR data[k]>``; missing entries are zero. Registers are little-endian and disjoint.
    ``kernel(address, ladder, output)`` uses a workspace view containing
    ``num_clean`` clean qubits followed by ``num_dirty`` borrowed qubits;
    both are restored. The default is all clean. ``controlled=True`` adds
    a leading one-qubit control view. The logical XOR is self-inverse on
    this specified workspace subspace. Automatic
    ``cudaq.control`` and ``cudaq.adjoint`` are outside this contract.

    Parameters follow :class:`QROM`'s SELECT and SELECT-SWAP options:
    nonempty nonnegative integer ``data``, positive ``address_bits`` and
    ``output_bits``, ``variant`` in ``{"auto", "select", "select_swap"}``,
    and optional power-of-two ``block_size`` for SELECT-SWAP. The coherent
    class's SELECT-COPY variant, packet counts and ancilla budget have no
    measured counterpart. A block leaves at least one high address bit.
    ``auto`` minimizes the emitted Toffoli count, preferring SELECT on
    ties, then the smaller block. ``controlled`` and ``dirty_banks`` must
    be plain ``bool`` values. ``controlled`` constructs the control into the
    tree instead of controlling measurements. ``dirty_banks=True`` is
    supported only for explicit SELECT-SWAP and adds a restored-bank copy
    echo; the borrowed bank may be entangled with any other input. Default
    ``False`` requires a clean bank. ``describe()`` lists the emitted tape.

    See the module and measured-QROM guide for executor requirements,
    resource conventions, and the clean-workspace restriction.
    """

    def __init__(self,
                 data: Sequence[int],
                 address_bits: int,
                 output_bits: int,
                 *,
                 variant: str = "auto",
                 block_size: int | None = None,
                 controlled: bool = False,
                 dirty_banks: bool = False) -> None:
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
            raise ValueError(
                f"data has {len(entries)} entries but address_bits="
                f"{address_bits} addresses only {1 << address_bits}")
        if any(v < 0 for v in entries):
            raise ValueError("data entries must be non-negative")
        if max(entries) >= (1 << output_bits):
            raise ValueError(
                f"data entry {max(entries)} does not fit in output_bits="
                f"{output_bits} bits (max representable "
                f"{(1 << output_bits) - 1})")
        if variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {_VARIANTS}, got {variant!r}")
        if not isinstance(controlled, bool):
            raise ValueError("controlled must be a bool (pass bool(flag) for "
                             "numpy booleans)")
        if not isinstance(dirty_banks, bool):
            raise ValueError("dirty_banks must be a bool (pass bool(flag) for "
                             "numpy booleans)")
        if dirty_banks and variant != "select_swap":
            raise ValueError("dirty_banks=True requires variant='select_swap'")
        if variant == "select_swap" and address_bits < 2:
            raise ValueError(
                "variant='select_swap' requires address_bits >= 2 (no "
                "block-index bit exists at address_bits=1); use "
                "variant='select' or 'auto'")
        if block_size is not None:
            if variant != "select_swap":
                raise ValueError(
                    "block_size is only valid with variant='select_swap', "
                    f"got variant={variant!r}")
            if (int(block_size) != block_size or block_size < 2
                    or int(block_size) & (int(block_size) - 1) != 0):
                raise ValueError("block_size must be a power of two >= 2, got "
                                 f"{block_size}")
            block_size = int(block_size)
            if block_size > (1 << (address_bits - 1)):
                raise ValueError(
                    f"block_size {block_size} needs at least one block-"
                    f"index address bit: max is 2^(address_bits - 1) = "
                    f"{1 << (address_bits - 1)}")

        select_cost = _measured_walk_toffoli_count(address_bits, len(entries),
                                                   controlled)
        swap_candidates = [1 << s for s in range(1, address_bits)
                           ] if block_size is None else [block_size]
        swap_costs = {
            size:
            _select_swap_cost(address_bits, len(entries), output_bits, size,
                              controlled, dirty_banks)
            for size in swap_candidates
        }
        if variant == "auto":
            best_size = min(swap_costs,
                            key=lambda size: (swap_costs[size], size),
                            default=None)
            if best_size is None or select_cost <= swap_costs[best_size]:
                variant = "select"
                block_size = None
            else:
                variant = "select_swap"
                block_size = best_size
        elif variant == "select_swap":
            block_size = min(swap_costs,
                             key=lambda size: (swap_costs[size], size))

        self._dirty_banks = dirty_banks
        self._controlled = controlled
        self._data = tuple(entries)
        self._num_address = address_bits
        self._output_bits = output_bits
        self._variant = variant
        self._block_size = block_size
        if variant == "select":
            self._build_select()
        else:
            self._build_select_swap()

    def _build_select(self) -> None:
        entries = self._data
        output_bits = self._output_bits

        def body(k: int) -> list[tuple[str, int]]:
            return [("x", t) for t in range(output_bits)
                    if (entries[k] >> t) & 1]

        # X-only body: the walk is an involution, skip the adjoint mint.
        walk = measured_unary_iteration_kernels(self._num_address,
                                                len(entries),
                                                body,
                                                controlled=self._controlled,
                                                include_adjoint=False)
        self._kernel = walk.kernel
        self._num_ladder = walk.num_ladder
        self._toffoli_count = walk.toffoli_count
        self._num_measurements = walk.num_measurements
        self._ops = tuple(walk.ops)
        self._tree_and_count = walk.tree_and_count

    def _build_select_swap(self) -> None:
        entries = self._data
        b = self._output_bits
        size = self._block_size
        low_bits = size.bit_length() - 1
        high_bits = self._num_address - low_bits
        num_blocks = (len(entries) + size - 1) // size  # ceil division

        def block_write(j: int) -> list[tuple[str, int]]:
            gates = []
            for i in range(size):
                k = j * size + i
                word = entries[k] if k < len(entries) else 0
                gates.extend(
                    ("x", i * b + t) for t in range(b) if (word >> t) & 1)
            return gates

        bodies = [block_write(j) for j in range(num_blocks)]
        walk_ops = _emit_measured_walk(high_bits, num_blocks, self._controlled,
                                       bodies, False)
        # Rebase high address bits and store each block in the clean
        # ladder view after the tree lines. Fixup CZ address operands
        # must be rebased too: their phase matters on superpositions.
        ops = []
        for op in walk_ops:
            opcode, a, bb, c = op
            if opcode == _OP_BODY_X:
                ops.append((_OP_CX_LADDER_LADDER, a, high_bits + bb, 0))
                continue
            operands = [a, bb, c]
            for position in _MEASURED_ADDRESS_OPERANDS.get(opcode, ()):
                operands[position] += low_bits
            ops.append((opcode, *operands))
        walk_ops = ops
        tree_ands = sum(1 for op in walk_ops
                        if op[0] in (_OP_CCX, _OP_CCX_CTRL))

        # Binary routing network: after stage s (controlled on
        # address[s]), block slot i (i = 0 mod 2^(s+1)) holds the entry
        # at low-address offset (i + a_s 2^s + ... + a_0); slot 0 ends
        # holding the selected entry. Each controlled register swap is b
        # Fredkins: cswap(c; u, v) = cx(v, u) ccx(c, u, v) cx(v, u).
        # u and v are block-register wires (which live in the ladder
        # view — see the rebase note above), so the outer CNOTs of each
        # Fredkin appear as _OP_CX_LADDER_LADDER: CNOTs purely between
        # the data blocks, exactly the routing network's own gates.
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

        # W S C S^-1 W: write, route, copy, unroute, unwrite. The default
        # clean bank gives the logical XOR directly.
        echo = swap_ops + copy_ops + list(reversed(swap_ops))
        ops = walk_ops + echo + walk_ops
        if self._dirty_banks:
            # W (S C S^-1) W restores the bank but copies its original
            # selected word along with the data; a second S C S^-1 cancels
            # exactly that borrowed contribution, even for entangled banks
            # and coherent controls (W S C S^-1 W S C S^-1 in circuit order).
            ops += echo
        self._kernel = _mint_interpreter(ops,
                                         controlled=self._controlled,
                                         has_work=False)
        self._num_ladder = high_bits + size * b
        self._toffoli_count = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)
        self._num_measurements = sum(1 for op in ops
                                     if op[0] in _MEASURED_OPCODES)
        self._ops = tuple(ops)
        self._tree_and_count = 2 * tree_ands

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    @property
    def num_address(self) -> int:
        return self._num_address

    @property
    def num_ladder(self) -> int:
        """Workspace width: ``num_clean`` clean lines, then ``num_dirty``
        borrowed lines. Both portions are restored after the lookup.
        """
        return self._num_ladder

    @property
    def num_output(self) -> int:
        return self._output_bits

    @property
    def variant(self) -> str:
        """The construction actually minted: 'select' or 'select_swap'."""
        return self._variant

    @property
    def block_size(self) -> int | None:
        """SELECT-SWAP block size (``None`` for the select variant)."""
        return self._block_size

    @property
    def toffoli_count(self) -> int:
        return self._toffoli_count

    @property
    def dirty_banks(self) -> bool:
        """Whether SELECT-SWAP borrows its bank in an arbitrary state."""
        return self._dirty_banks

    @property
    def num_dirty(self) -> int:
        """Borrowed bank width; its arbitrary joint state is restored."""
        if self._dirty_banks:
            return self._block_size * self._output_bits
        return 0

    @property
    def num_clean(self) -> int:
        """Clean prefix width in the workspace view (zero in and out)."""
        return self.num_ladder - self.num_dirty

    @property
    def controlled(self) -> bool:
        """Whether the kernel takes a leading one-qubit control view."""
        return self._controlled

    @property
    def tree_and_count(self) -> int:
        """Temporary clean-tree ANDs, counted from the walk tape (opcodes
        ``_OP_CCX`` and ``_OP_CCX_CTRL``); excludes coherent routing
        Toffolis and is independent of ``num_measurements``."""
        return self._tree_and_count

    @property
    def num_measurements(self) -> int:
        """Unconditional X-basis tree measurements per application."""
        return self._num_measurements

    @property
    def kernel(self):
        """Lookup ``([control,] address, ladder, output)`` with the declared
        clean-prefix/borrowed-suffix workspace contract.
        """
        return self._kernel

    def describe(self) -> str:
        """Decode the minted lookup into a gate listing, one line per
        instruction in execution order. Lines tagged ``# Toffoli`` are what
        ``toffoli_count`` counts; lines tagged ``# measured uncompute`` are
        the gadgets ``num_measurements`` counts."""
        lines = [
            _OP_DESCRIPTIONS[op].format(a=a, b=b, c=c)
            for op, a, b, c in self._ops
        ]
        return "\n".join([f"MeasuredQROM lookup: {self!r}"] + lines)

    def __repr__(self) -> str:
        return (f"MeasuredQROM(entries={len(self._data)}, "
                f"address_bits={self.num_address}, "
                f"output_bits={self.num_output}, "
                f"variant={self._variant!r}, "
                f"block_size={self._block_size}, "
                f"controlled={self.controlled}, "
                f"dirty_banks={self.dirty_banks}, "
                f"toffolis={self.toffoli_count}, "
                f"measurements={self.num_measurements})")
