# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QROM: coherent classical-table lookup, two constructions in one surface.

``QROM(data, address_bits, output_bits)`` mints the lookup
``|k>|y> -> |k>|y XOR data[k]>`` behind one kernel signature
``(address: qview, ladder: qview, output: qview)`` — with a leading
one-qubit ``control`` view when ``controlled=True`` — all little-endian
(qubit 0 = LSB, ``docs/conventions.md``), ``ladder`` being ``num_ladder``
clean ancillas (``|0>`` in, ``|0>`` out). Addresses ``k >= len(data)`` read as
zero. Two constructions sit behind ``variant=``:

- ``"select"`` — the plain unary-iteration QROM (Babbush et al.,
  `arXiv:1805.03662`): one fused tree walk over the address register
  whose per-address body CNOTs the set bits of ``data[k]`` into the
  output. ``num_ladder = address_bits``.
- ``"select_swap"`` — SELECT-SWAP / QROAM (Low, Kliuchnikov, Schaeffer,
  `arXiv:1812.00954`): the table is split into ``ceil(N / B)`` blocks of
  ``B = block_size`` entries; one unary-iteration walk over the *block
  index* (the high ``address_bits - log2(B)`` address bits) writes the
  visited block's ``B`` entries into ``B`` clean ``output_bits``-wide
  block registers, a swap network controlled on the low ``log2(B)``
  address bits routes the selected entry's register into block slot 0,
  and CNOTs copy slot 0 into the output. The routing and the writes are
  then undone unitarily. ``num_ladder = (address_bits - log2(B)) +
  B * output_bits`` (walk lines first, then the block registers).
- ``"auto"`` (default) — price both variants (and every power-of-two
  ``block_size``) with the exact Toffoli model below and take the
  cheapest, preferring ``"select"`` on ties (fewer ancillas).

Toffoli accounting (documented contract, pinned by the resource tests;
``W(n, m)`` is the fused walk count from
:mod:`cudaq_algorithms.primitives._unary_iteration`, ``3 * 2^n / 2 - 5``
on full uncontrolled trees):

- ``"select"``: ``W(address_bits, len(data))``.
- ``"select_swap"``: ``2 * W(address_bits - log2(B), ceil(len(data)/B))
  + 2 * output_bits * (B - 1)``. Each controlled register swap is
  ``output_bits`` Fredkins, each Fredkin one Toffoli plus two CNOTs; the
  binary routing network is ``B - 1`` register swaps per direction, and
  both the routing and the block writes run twice (compute + unitary
  uncompute).
- ``controlled=True`` adds an external control qubit and replaces ``W``
  by its controlled count (``_walk_toffoli_count(n, m, controlled)``: the
  walk's root gadgets become Toffolis and its root crossings lose the
  unguarded shortcut). Only the walk passes carry the control: the walk
  sub-kernel is minted ``controlled=True`` and the parent passes the
  control view to both walk passes, while the routing network and the
  copy are the plain kernels. At control ``|0>`` a controlled walk on a
  clean ladder writes nothing, so the block registers stay ``|0>`` and
  the routing Fredkins, the copy CNOTs and the un-route/un-write act on
  ``|0>`` as the identity. The lookup is then the identity at control
  ``|0>`` and the plain lookup at ``|1>``, with the control as a separate
  one-qubit view first in the signature, as ``unary_iteration_kernels``
  does — and at a fraction of the price of ``cudaq.control`` over the
  composition, which would add a control to every gate: for eleven 3-bit
  entries over four address bits the native lookup compiles to 19 / 26 /
  28 Toffolis (select, select_swap with ``B = 2`` and ``B = 4``) against
  15 / 18 / 22 plain, where the wrapped composition costs two to three
  times the native count, widest on select_swap because the wrapper also
  controls the Fredkins and the copy CNOTs. The identity at
  ``|0>`` is a clean-ladder statement (the clean ladder is the kernel's
  precondition in any case): on a dirty ladder the deeper walk Toffolis
  are gated by ladder lines, not by the control.

The papers' headline counts (``N - 1`` lookup Toffolis for SELECT,
``ceil(N/B) + output_bits * (B - 1)`` for QROAM) price ancilla
uncomputation at zero via measurement-and-fixup. These primitives stay
strictly unitary (house rules: statevector-testable, inverse-composable,
no mid-circuit measurement), so the coherent counts above are what the
minted kernels actually cost; the default ``block_size`` optimizes the
coherent model (``~ sqrt(3 N / (2 output_bits))`` rather than the
papers' ``~ sqrt(N / output_bits)``).

Both variants are exactly their own inverse: the write phase is X-only,
so the ``"select"`` walk XORs the same table twice, and the
``"select_swap"`` sandwich ``U = W S C S^-1 W`` squares to identity
(``W`` and ``C`` are involutions and the copy commutes with itself
through the routing). Apply ``kernel()`` again to uncompute; the
property is pinned by ``tests/python/test_primitives_qrom.py``.

Implementation notes: register layout and the ``ladder`` view
------------------------------------------------------------------------

Every minted kernel has the same signature, ``(address, ladder,
output)`` (``(control, address, ladder, output)`` when ``controlled``),
and ``ladder`` is simply **the clean-ancilla register** —
its contents differ per variant, and for ``select_swap`` the name is
looser than the physics:

- ``"select"``: ``ladder`` really is the walk's ladder —
  ``address_bits`` lines, one per tree level.
- ``"select_swap"``: ``ladder`` bundles two conceptually different
  things, walk lines first, block registers after::

      ladder[0 .. h)                 the block-index walk's ladder lines
                                     (h = address_bits - log2(B))
      ladder[h + i*b .. h + (i+1)*b) block register i, i = 0 .. B-1
                                     (the "cache line" slots)

  ``num_ladder = h + B * b``. Both groups are clean (|0> in, |0> out),
  which is why they share one register; the price is that ``ladder``
  qubits above ``h`` are cache-line storage, not ladder lines.

The address register is likewise split by convention, low bits first:
``address[0 .. low)`` are the routing bits (low = log2(B)),
``address[low ..)`` the block index the walk iterates.

``select_swap`` is built by **composition**: a parent kernel carves the
honest sub-views out of ``address`` and ``ladder`` with qview slices and
calls three sub-kernels, ``W S C S^-1 W``:

- ``W`` — the block-index walk, a plain
  :func:`~cudaq_algorithms.primitives.unary_iteration_kernels` mint over
  the high address bits whose *target view is the block registers*: the
  per-block body X's write the visited block's ``B`` entries.
- ``S`` — a dedicated routing kernel over ``(low address bits,
  blocks)``: ``low`` rounds of Fredkins, each controlled register swap
  being ``b`` Fredkins decomposed ``cswap(c; u, v) = cx(v,u) ccx(u,c,v)
  cx(v,u)``. ``S^-1`` is the same Fredkin list reversed (each Fredkin
  is self-inverse).
- ``C`` — ``b`` CNOTs copying block slot 0 onto ``output``, inline in
  the parent.

End-to-end, ``select_swap`` is the five-step program (with
``h + low = address_bits``, ``B = 2^low`` entries per block):

1. walk the block index (high bits) on ``ladder[0..h)``, writing each
   visited block's ``B`` entries into the block registers;
2. route: ``low`` rounds of Fredkins controlled on the low address
   bits bring the selected entry's register into block slot 0;
3. copy block slot 0 into ``output`` (plain CNOTs);
4. un-route (the same Fredkins, reversed order);
5. un-write (the X-only walk again — XOR is self-inverse).

Steps 1/2/4/5 touch only ``address`` + ``ladder``; step 3 is the only
place ``output`` is written, which is what makes the sandwich exactly
self-inverse and the ancillas exactly clean.
"""

from __future__ import annotations

from collections.abc import Sequence

import cudaq

from ._unary_iteration import (_retain, _walk_toffoli_count,
                               unary_iteration_kernels)

__all__ = ["QROM"]

_VARIANTS = ("auto", "select", "select_swap")


def _mint_route(swap_c: list, swap_u: list, swap_v: list):
    """Mint the routing kernel: Fredkin ``i`` swaps ``blocks[swap_u[i]]``
    and ``blocks[swap_v[i]]`` controlled on ``low[swap_c[i]]``.

    The inverse network is this same mint over the reversed lists (each
    Fredkin is self-inverse).
    """
    num_swaps = len(swap_c)

    @cudaq.kernel
    def primitives_qrom_route(low: cudaq.qview, blocks: cudaq.qview):
        for i in range(num_swaps):
            c = swap_c[i]
            u = swap_u[i]
            v = swap_v[i]
            # cswap(low[c]; blocks[u], blocks[v]) via one Toffoli.
            cx(blocks[v], blocks[u])
            x.ctrl(blocks[u], low[c], blocks[v])
            cx(blocks[v], blocks[u])

    _retain(primitives_qrom_route)
    return primitives_qrom_route


def _select_swap_cost(address_bits: int,
                      num_entries: int,
                      output_bits: int,
                      block_size: int,
                      controlled: bool = False) -> int:
    """Documented Toffoli price of the select_swap variant."""
    high_bits = address_bits - (block_size.bit_length() - 1)
    num_blocks = -(-num_entries // block_size)
    return (2 * _walk_toffoli_count(high_bits, num_blocks, controlled) +
            2 * output_bits * (block_size - 1))


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
        ``"auto"`` (default), ``"select"`` or ``"select_swap"`` — see the
        module docstring for the constructions and their exact prices.
    block_size
        SELECT-SWAP block size: a power of two in
        ``[2, 2^(address_bits - 1)]``. Only valid with
        ``variant="select_swap"``; ``None`` there picks the power of two
        minimizing the documented Toffoli count.
    controlled
        ``True`` mints the externally controlled lookup: the kernel takes
        a leading one-qubit ``control`` view, is the identity at control
        ``|0>`` and the plain lookup at ``|1>`` (module docstring). The
        identity at ``|0>`` holds on the clean ladder the kernel requires
        anyway; a dirty ladder is outside the contract either way.
        ``variant="auto"`` prices the controlled candidates against each
        other on Toffolis alone, and the optimum can sit at a larger block
        than the plain lookup's, with more block registers: size the
        ladder from this instance's ``num_ladder`` rather than assume the
        plain and controlled lookups share a layout (seven one-bit entries
        over three address bits take 5 ladder qubits controlled, 4 plain).

    The views of the minted kernel may live anywhere (no contiguity
    requirement between them), which is what lets callers weave the QROM
    into their own ancilla layouts. Whatever the variant, the kernel is
    exactly self-inverse — apply it again to uncompute.
    """

    def __init__(self,
                 data: Sequence[int],
                 address_bits: int,
                 output_bits: int,
                 *,
                 variant: str = "auto",
                 block_size: int | None = None,
                 controlled: bool = False) -> None:
        if not isinstance(controlled, bool):
            raise ValueError("controlled must be a bool (pass bool(flag) "
                             "for numpy booleans)")
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
                    or block_size & (block_size - 1) != 0):
                raise ValueError("block_size must be a power of two >= 2, got "
                                 f"{block_size}")
            block_size = int(block_size)
            if block_size > (1 << (address_bits - 1)):
                raise ValueError(
                    f"block_size {block_size} needs at least one block-"
                    f"index address bit: max is 2^(address_bits - 1) = "
                    f"{1 << (address_bits - 1)}")

        select_cost = _walk_toffoli_count(address_bits, len(entries),
                                          controlled)
        swap_candidates = [1 << s for s in range(1, address_bits)
                           ] if block_size is None else [block_size]
        swap_costs = {
            size:
            _select_swap_cost(address_bits, len(entries), output_bits, size,
                              controlled)
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

        self._data = tuple(entries)
        self._num_address = address_bits
        self._output_bits = output_bits
        self._variant = variant
        self._block_size = block_size
        self._controlled = controlled
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
        walk = unary_iteration_kernels(self._num_address,
                                       len(entries),
                                       body,
                                       controlled=self._controlled,
                                       include_adjoint=False)
        self._kernel = walk.kernel
        self._num_ladder = walk.num_ladder
        self._toffoli_count = walk.toffoli_count

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

        # This IS the textbook SELECT-SWAP: a unary-iteration walk over
        # the high (block-index) bits whose target view is the block
        # registers (the leaf action writes each visited block), then a
        # low-bit-controlled swap network routes the selected entry.
        # X-only body: the walk is an involution, skip the adjoint mint.
        # Only the walk passes carry the external control (module
        # docstring): at control |0> nothing is written, so the routing
        # and copy below act on |0> blocks as the identity.
        walk = unary_iteration_kernels(high_bits,
                                       num_blocks,
                                       block_write,
                                       controlled=self._controlled,
                                       include_adjoint=False)
        walk_kernel = walk.kernel

        # Binary routing network: after stage s (controlled on
        # address[s]), block slot i (i = 0 mod 2^(s+1)) holds the entry
        # at low-address offset (i + a_s 2^s + ... + a_0); slot 0 ends
        # holding the selected entry. Each controlled register swap is b
        # Fredkins (one Toffoli each).
        swap_c: list[int] = []
        swap_u: list[int] = []
        swap_v: list[int] = []
        for s in range(low_bits):
            for i in range(0, size, 1 << (s + 1)):
                for t in range(b):
                    swap_c.append(s)
                    swap_u.append(i * b + t)
                    swap_v.append((i + (1 << s)) * b + t)
        route_kernel = _mint_route(swap_c, swap_u, swap_v)
        unroute_kernel = _mint_route(list(reversed(swap_c)),
                                     list(reversed(swap_u)),
                                     list(reversed(swap_v)))

        # W S C S^-1 W: write, route, copy, unroute, unwrite — clean
        # ancillas and an exactly self-inverse lookup. The parent carves
        # the documented sub-views out of the public (address, ladder,
        # output) signature and composes the sub-kernels. Two parents on
        # purpose: they differ only in the walk calls and share the route
        # and copy sub-kernels; the plain one keeps the three-view
        # signature of the composition contract, and a dummy control would
        # cost every plain caller a qubit.
        if self._controlled:

            @cudaq.kernel
            def primitives_qrom_select_swap_ctrl(control: cudaq.qview,
                                                 address: cudaq.qview,
                                                 ladder: cudaq.qview,
                                                 output: cudaq.qview):
                low = address[0:low_bits]
                high = address[low_bits:]
                lines = ladder[0:high_bits]
                blocks = ladder[high_bits:]
                walk_kernel(control, high, lines, blocks)
                route_kernel(low, blocks)
                for t in range(b):
                    cx(blocks[t], output[t])
                unroute_kernel(low, blocks)
                walk_kernel(control, high, lines, blocks)

            _retain(primitives_qrom_select_swap_ctrl)
            self._kernel = primitives_qrom_select_swap_ctrl
        else:

            @cudaq.kernel
            def primitives_qrom_select_swap(address: cudaq.qview,
                                            ladder: cudaq.qview,
                                            output: cudaq.qview):
                low = address[0:low_bits]
                high = address[low_bits:]
                lines = ladder[0:high_bits]
                blocks = ladder[high_bits:]
                walk_kernel(high, lines, blocks)
                route_kernel(low, blocks)
                for t in range(b):
                    cx(blocks[t], output[t])
                unroute_kernel(low, blocks)
                walk_kernel(high, lines, blocks)

            _retain(primitives_qrom_select_swap)
            self._kernel = primitives_qrom_select_swap
        self._num_ladder = high_bits + size * b
        self._toffoli_count = 2 * walk.toffoli_count + 2 * len(swap_c)

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    @property
    def num_address(self) -> int:
        return self._num_address

    @property
    def num_ladder(self) -> int:
        """Clean ancillas the kernel needs (``|0>`` in, ``|0>`` out).

        ``address_bits`` walk lines for ``"select"``; walk lines plus the
        ``block_size * output_bits`` block registers for
        ``"select_swap"``.
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
    def controlled(self) -> bool:
        """Whether the kernel takes a leading one-qubit ``control`` view."""
        return self._controlled

    @property
    def kernel(self):
        """The lookup kernel — self-inverse. ``(address, ladder, output)``,
        or ``(control, address, ladder, output)`` when ``controlled``."""
        return self._kernel

    def __repr__(self) -> str:
        return (f"QROM(entries={len(self._data)}, "
                f"address_bits={self.num_address}, "
                f"output_bits={self.num_output}, "
                f"variant={self._variant!r}, "
                f"block_size={self._block_size}, "
                f"controlled={self._controlled}, "
                f"toffolis={self.toffoli_count})")
