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
clean lines and ``B*output_bits`` dirty lines. For clean banks, ``cache_uncompute="measure"`` replaces the final write
with X-basis cache erasure and an outcome-dependent phase correction.
``cache_uncompute="auto"`` selects it only for a strict Toffoli saving; the
default ``"traverse"`` preserves the original selection and operation counts.

Resource properties describe emitted logical operations, not physical
hardware cost or compiled T gates. A 4T temporary-AND model applies to
``resources.temporary_and_count`` (tree and one-hot ANDs) after compatible
synthesis; the implementation emits ordinary Toffolis. Tree lines must
always start clean. The routing
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

from ._qrom_common import validate_qrom_data
from ._qrom_plan import QROMResources
from ._qrom_measured_plan import _plan_measured_qrom

from ._qrom import _ADDRESS_OPERANDS
from ._unary_iteration import _OP_DESCRIPTIONS, _mint_interpreter
from ._unary_iteration_measured import _measured_walk_toffoli_count

__all__ = ["MeasuredQROM"]

_VARIANTS = ("auto", "select", "select_swap")

_MEASURED_ADDRESS_OPERANDS = _ADDRESS_OPERANDS


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

    ``cache_uncompute="measure"`` requires explicit clean SELECT-SWAP.
    ``phase_block_size`` optionally selects its power-of-two one-hot width;
    otherwise the planner minimizes correction Toffolis. ``"auto"`` admits
    measured cleanup and favors traversal on equal Toffoli costs. Persistent
    outcomes are fresh local invocation state; compiled kernels are reusable.

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
                 dirty_banks: bool = False,
                 cache_uncompute: str = "traverse",
                 phase_block_size: int | None = None) -> None:
        # Materialize single-pass inputs once before planning and retaining data.
        entries, address_bits, output_bits = validate_qrom_data(
            data, address_bits, output_bits)
        selection = _plan_measured_qrom(entries,
                                        address_bits,
                                        output_bits,
                                        variant=variant,
                                        block_size=block_size,
                                        controlled=controlled,
                                        dirty_banks=dirty_banks,
                                        cache_uncompute=cache_uncompute,
                                        phase_block_size=phase_block_size)
        plan = selection.plan
        self._data = entries
        self._num_address = address_bits
        self._output_bits = output_bits
        self._controlled = controlled
        self._dirty_banks = dirty_banks
        self._variant = selection.variant
        self._block_size = selection.block_size
        self._cache_uncompute = selection.cache_uncompute
        self._phase_block_size = selection.phase_block_size
        self._resources = plan.resources
        self._num_ladder = plan.resources.num_ladder
        self._ops = plan.ops
        self._kernel = _mint_interpreter(plan.ops,
                                         controlled=controlled,
                                         has_work=False,
                                         num_results=plan.num_results,
                                         parity_masks=plan.parity_masks)

    @property
    def cache_uncompute(self) -> str:
        """Resolved cache cleanup: none, traverse, or measure."""
        return self._cache_uncompute

    @property
    def phase_block_size(self) -> int | None:
        """Resolved one-hot phase block width, or None without cache erasure."""
        return self._phase_block_size

    @property
    def resources(self) -> QROMResources:
        """Logical operation counts by stage, not compiled hardware costs."""
        return self._resources

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
        return self._resources.toffoli_count

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
        return self._resources.tree_and_count

    @property
    def num_measurements(self) -> int:
        """Unconditional X-basis tree, cache, and one-hot measurements."""
        return self._resources.num_measurements

    @property
    def kernel(self):
        """Lookup ``([control,] address, ladder, output)`` with the declared
        clean-prefix/borrowed-suffix workspace contract.
        """
        return self._kernel

    def describe(self) -> str:
        """Decode the minted lookup into a gate listing, one line per
        instruction in execution order. Lines tagged ``# Toffoli`` contribute
        to ``toffoli_count``; measured uncompute and cache result lines
        contribute to ``num_measurements``."""
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
                f"cache_uncompute={self.cache_uncompute!r}, "
                f"phase_block_size={self.phase_block_size}, "
                f"toffolis={self.toffoli_count}, "
                f"measurements={self.num_measurements})")
