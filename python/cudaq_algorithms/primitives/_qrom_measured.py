# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical-table XOR lookup with measured clean-tree uncomputation.

This separate class opts into dynamic circuits; existing ``QROM`` and its
consumers remain unitary. SELECT, SELECT-SWAP and packet SELECT-COPY are provided.
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
default ``"traverse"`` retains traversal cleanup for SELECT-SWAP; automatic
selection can also choose SELECT-COPY.

Resource properties describe emitted logical operations, not physical
hardware cost or compiled T gates. A 4T temporary-AND model applies only
to ``temporary_and_count`` after a compatible synthesis; the implementation
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

from ._qrom_common import validate_qrom_data
from ._qrom_plan import QROMPlan, QROMResources
from ._qrom_cache_erasure import _plan_cache_erasure, _plan_cache_measurement
from ._select_copy_plan import _plan_select_copy_phase
from ._qrom_measured_plan import _plan_measured_qrom

from ._qrom import _ADDRESS_OPERANDS
from ._unary_iteration import _OP_DESCRIPTIONS, _mint_interpreter
from ._unary_iteration_measured import _measured_walk_toffoli_count

__all__ = ["MeasuredQROM"]

_VARIANTS = ("auto", "select", "select_swap", "select_copy")

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

    Parameters extend :class:`QROM` with adaptive execution:
    nonempty nonnegative integer ``data``, positive ``address_bits`` and
    ``output_bits``, and ``variant`` in ``{"auto", "select", "select_swap",
    "select_copy"}``. An explicit blocked variant accepts power-of-two
    ``block_size`` leaving at least one high address bit. SELECT-COPY accepts
    balanced ``alpha`` packets or fixed ``packet_width`` with a final remainder,
    but not both. ``max_ancillas`` bounds total clean plus borrowed workspace.
    ``auto`` minimizes the emitted Toffoli count, preferring SELECT on
    ties, then the smaller block. ``controlled`` and ``dirty_banks`` must
    be plain ``bool`` values. ``controlled`` constructs the control into the
    tree instead of controlling measurements. ``dirty_banks=True`` is
    supported for blocked variants and automatic selection; the borrowed bank may be entangled with any other input. Default
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
                 phase_block_size: int | None = None,
                 alpha: int | None = None,
                 packet_width: int | None = None,
                 max_ancillas: int | None = None) -> None:
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
                                        phase_block_size=phase_block_size,
                                        alpha=alpha,
                                        packet_width=packet_width,
                                        max_ancillas=max_ancillas)
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
        self._plan = plan
        self._packets = selection.packets
        self._resources = plan.resources
        self._num_ladder = plan.resources.num_ladder
        self._ops = plan.ops
        self._kernel = _mint_interpreter(plan.ops,
                                         controlled=controlled,
                                         has_work=False,
                                         num_results=plan.num_results,
                                         parity_masks=plan.parity_masks)

    @property
    def plan(self):
        return self._plan

    @property
    def packets(self):
        return self._packets

    @property
    def temporary_and_count(self):
        return self.resources.temporary_and_count

    def _resolve_erase(self):
        if hasattr(self, '_erase_plan'):
            return
        if self.variant == 'select_copy':
            low = self.block_size.bit_length() - 1
            bank = max(low, self.num_address - low)
            prefix = _plan_cache_measurement(self.num_output,
                                             cache_offset=0,
                                             cache_register='output')
            rows = tuple(
                tuple(t for t in range(self.num_output) if word >> t & 1)
                for word in self.data)
            phase = _plan_select_copy_phase(rows,
                                            self.num_address,
                                            controlled=self.controlled,
                                            block_size=self.block_size,
                                            bank_offset=bank,
                                            tree_offset=0)
            resources = QROMResources(
                prefix.resources.stages + phase.resources.stages,
                phase.resources.num_clean, phase.resources.num_dirty,
                self.num_output)
            plan = QROMPlan(prefix.ops + phase.ops, phase.parity_masks,
                            self.num_output, resources)
        else:
            plan = _plan_cache_erasure(self.data,
                                       self.num_address,
                                       self.num_output,
                                       controlled=self.controlled,
                                       phase_block_size=1,
                                       cache_offset=0,
                                       tree_offset=0,
                                       cache_register='output',
                                       unary_offset=0)
        self._erase_plan = plan

    @property
    def erase_resources(self):
        """Independent image-erasure workspace and operation ledger."""
        self._resolve_erase()
        return self._erase_plan.resources

    @property
    def num_erase_clean(self):
        return self.erase_resources.num_clean

    @property
    def num_erase_dirty(self):
        return self.erase_resources.num_dirty

    @property
    def num_erase_ladder(self):
        return self.erase_resources.num_ladder

    @property
    def erase_kernel(self):
        """Erase a known clean-copy image, not an arbitrary output state.

        Signature matches ``kernel``, but allocate the ladder from the erase
        properties: its clean prefix and borrowed suffix may differ from load.
        The inactive control branch requires zero output. SELECT-COPY erasure
        borrows its phase bank even when the load uses a clean bank.
        """
        self._resolve_erase()
        if not hasattr(self, '_erase_kernel'):
            plan = self._erase_plan
            self._erase_kernel = _mint_interpreter(
                plan.ops,
                self.controlled,
                False,
                num_results=plan.num_results,
                parity_masks=plan.parity_masks)
        return self._erase_kernel

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
        """The selected SELECT, SELECT-SWAP or SELECT-COPY construction."""
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
        """Whether the requested bank mode allows arbitrary borrowed states."""
        return self._dirty_banks

    @property
    def num_dirty(self) -> int:
        """Borrowed bank width; its arbitrary joint state is restored."""
        return self._resources.num_dirty

    @property
    def num_clean(self) -> int:
        """Clean prefix width in the workspace view (zero in and out)."""
        return self._resources.num_clean

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
