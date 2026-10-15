# SPDX-License-Identifier: Apache-2.0
"""Pure measured-QROM selection: price geometry first, emit one chosen tape.

Selection minimizes logical Toffolis. Ties prefer SELECT, traversal cleanup,
smaller B, then smaller K among legacy candidates. A SELECT-COPY tie keeps
the legacy winner; copy-only ties prefer total workspace, B, then packet widths.
Budgets filter candidates before ranking. No runtime kernels are built.
Tape and masks use O(m*b + n + B*b) storage: each populated data bit adds at
most a constant number of operations/mask indices; trees and routing are linear.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from ._select_copy_plan import PacketLayout, packet_layout, select_copy_resources, plan_select_copy

from . import _unary_iteration_ir as ir
from ._qrom_common import BlockGeometry, rebase_block_walk, validate_qrom_data
from ._qrom_plan import QROMPlan, QROMResources, QROMStageResources
from ._qrom_cache_erasure import _plan_cache_erasure


@dataclass(frozen=True)
class MeasuredQROMSelection:
    plan: QROMPlan
    variant: str
    block_size: int | None
    cache_uncompute: str
    phase_block_size: int | None
    packets: PacketLayout | None = None


@dataclass(frozen=True)
class _Candidate:
    cost: int
    variant: str
    block_size: int | None
    cache_uncompute: str
    phase_block_size: int | None = None

    @property
    def key(self):
        return (self.cost, self.variant
                != 'select', self.cache_uncompute == 'measure', self.block_size
                or 0, self.phase_block_size or 0)


def _phase_sizes(high_bits, cache_bits):
    return (1 << s for s in range(high_bits) if 1 << s <= cache_bits)


def _price_candidates(n, m, b, *, variant, block_size, controlled, dirty_banks,
                      cache_uncompute, phase_block_size):
    """Yield scalar geometry prices without building bodies, masks, or tapes."""
    count = ir._measured_walk_toffoli_count
    if variant != 'select_swap':
        yield _Candidate(count(n, m, controlled), 'select', None, 'none')
    if variant == 'select':
        return
    sizes = (block_size, ) if block_size is not None else (
        1 << s for s in range(1, n))
    for size in sizes:
        geometry = BlockGeometry(n, b, m, size)
        h, M = geometry.high_bits, geometry.num_blocks
        L = count(h, M, controlled)
        routing = (4 if dirty_banks else 2) * b * (size - 1)
        if cache_uncompute != 'measure':
            yield _Candidate(2 * L + routing, 'select_swap', size, 'traverse')
        if cache_uncompute != 'traverse' and not dirty_banks:
            ks = (phase_block_size,
                  ) if phase_block_size is not None else _phase_sizes(
                      h, size * b)
            for k in ks:
                # An explicit K may disqualify some B values; it must be valid
                # for the selected geometry, not silently clamped.
                if k > min(size * b, 1 << (h - 1)):
                    continue
                P = count(h - (k.bit_length() - 1), (M + k - 1) // k,
                          controlled)
                yield _Candidate(L + routing + P + k - 1, 'select_swap', size,
                                 'measure', k)


def _stage(name, ops, *, tree=False):
    ccx = sum(op[0] in ir._TOFFOLI_OPCODES for op in ops)
    measurements = sum(op[0] in ir._MEASURED_OPCODES for op in ops)
    return QROMStageResources(name, ccx, ccx if tree else 0,
                              ccx if tree else 0, measurements)


def _emit_selection(entries, n, b, candidate, controlled, dirty_banks):
    if candidate.variant == 'select':
        bodies = tuple(
            tuple(('x', t) for t in range(b) if word >> t & 1)
            for word in entries)
        ops = tuple(
            ir._emit_measured_walk(n, len(entries), controlled, bodies, False))
        return QROMPlan(
            ops, (), 0,
            QROMResources((_stage('select', ops, tree=True), ), n, 0))
    B = candidate.block_size
    geometry = BlockGeometry(n, b, len(entries), B)
    l, h, M = geometry.low_bits, geometry.high_bits, geometry.num_blocks
    words = tuple(
        sum(entries[j * B + i] << (i * b)
            for i in range(min(B,
                               len(entries) - j * B))) for j in range(M))
    bodies = tuple(
        tuple(('x', t) for t in range(B * b) if word >> t & 1)
        for word in words)
    write = rebase_block_walk(ir._emit_measured_walk(h, M, controlled, bodies,
                                                     False),
                              address_offset=l,
                              output_bits=0,
                              bank_offset=h)
    route = []
    for s in range(l):
        for i in range(0, B, 1 << (s + 1)):
            for t in range(b):
                u, v = h + i * b + t, h + (i + (1 << s)) * b + t
                route.extend(
                    ((ir._OP_CX_LADDER_LADDER, v, u, 0), (ir._OP_CCX, u, s, v),
                     (ir._OP_CX_LADDER_LADDER, v, u, 0)))
    route = tuple(route)
    copy = tuple((ir._OP_CX_LADDER_TARGET, h + t, t, 0) for t in range(b))
    unroute = tuple(reversed(route))
    echo = route + copy + unroute
    stages = (_stage('write', write, tree=True), _stage('route', route),
              _stage('copy', copy), _stage('unroute', unroute))
    if candidate.cache_uncompute == 'measure':
        erase = _plan_cache_erasure(
            words,
            h,
            B * b,
            controlled=controlled,
            phase_block_size=candidate.phase_block_size,
            cache_offset=h,
            tree_offset=0,
            address_offset=l)
        return QROMPlan(
            write + echo + erase.ops, erase.parity_masks, erase.num_results,
            QROMResources(stages + erase.resources.stages, h + B * b, 0,
                          B * b))
    ops = write + echo + write
    stages += (_stage('unwrite', write, tree=True), )
    if dirty_banks:
        ops += echo
        stages += (_stage('echo_route', route), _stage('echo_copy', copy),
                   _stage('echo_unroute', unroute))
    dirty = B * b if dirty_banks else 0
    return QROMPlan(ops, (), 0, QROMResources(stages, h + B * b - dirty,
                                              dirty))


def _plan_measured_qrom(
        data: Iterable[int],
        address_bits: int,
        output_bits: int,
        *,
        variant: str = 'auto',
        block_size: int | None = None,
        controlled: bool = False,
        dirty_banks: bool = False,
        cache_uncompute: str = 'traverse',
        phase_block_size: int | None = None,
        alpha: int | None = None,
        packet_width: int | None = None,
        max_ancillas: int | None = None) -> MeasuredQROMSelection:
    """Validate a single-pass table, select geometry, and emit its immutable plan."""
    entries, n, b = validate_qrom_data(data, address_bits, output_bits)
    if variant not in ('auto', 'select', 'select_swap', 'select_copy'):
        raise ValueError(
            "variant must be one of auto, select, select_swap, or select_copy")
    if not isinstance(controlled, bool):
        raise ValueError(
            'controlled must be a bool (pass bool(flag) for numpy booleans)')
    if not isinstance(dirty_banks, bool):
        raise ValueError(
            'dirty_banks must be a bool (pass bool(flag) for numpy booleans)')
    if dirty_banks and variant == 'select':
        raise ValueError("dirty_banks=True requires a blocked variant or auto")
    if variant in ('select_swap', 'select_copy') and n < 2:
        raise ValueError("a blocked variant requires address_bits >= 2")
    if block_size is not None:
        if variant not in ('select_swap', 'select_copy'):
            raise ValueError(
                'block_size is only valid with an explicit blocked variant')
        if int(block_size) != block_size or block_size < 2 or int(
                block_size) & (int(block_size) - 1):
            raise ValueError('block_size must be a power of two >= 2')
        block_size = int(block_size)
        if block_size > 1 << (n - 1):
            raise ValueError(
                'block_size needs at least one block-index address bit')
    if cache_uncompute not in ('traverse', 'measure', 'auto'):
        raise ValueError('cache_uncompute must be traverse, measure, or auto')
    if cache_uncompute == 'measure' and (variant != 'select_swap'
                                         or dirty_banks):
        raise ValueError(
            "cache_uncompute='measure' requires explicit select_swap and clean banks"
        )
    if phase_block_size is not None:
        if variant != 'select_swap' or cache_uncompute != 'measure':
            raise ValueError(
                "phase_block_size requires explicit select_swap and cache_uncompute='measure'"
            )
        if (int(phase_block_size) != phase_block_size or phase_block_size < 1
                or int(phase_block_size) & (int(phase_block_size) - 1)):
            raise ValueError(
                'phase_block_size must be an integral power of two >= 1')
        phase_block_size = int(phase_block_size)
    if max_ancillas is not None and (isinstance(max_ancillas, bool)
                                     or not isinstance(max_ancillas, Integral)
                                     or max_ancillas < 0):
        raise ValueError('max_ancillas must be a nonnegative integer budget')
    if (alpha is not None
            or packet_width is not None) and variant != 'select_copy':
        raise ValueError('packet options require explicit select_copy')
    explicit_packets = (packet_layout(
        b, alpha=alpha, packet_width=packet_width) if alpha is not None
                        or packet_width is not None else None)
    # Preserve the legacy SELECT / SELECT-SWAP ordering exactly. Workspace is
    # only a feasibility filter for those candidates, never a new tie-break.
    legacy = []
    if variant != 'select_copy':
        for candidate in _price_candidates(n,
                                           len(entries),
                                           b,
                                           variant=variant,
                                           block_size=block_size,
                                           controlled=controlled,
                                           dirty_banks=dirty_banks,
                                           cache_uncompute=cache_uncompute,
                                           phase_block_size=phase_block_size):
            workspace = (n if candidate.variant == 'select' else n -
                         (candidate.block_size.bit_length() - 1) +
                         candidate.block_size * b)
            if max_ancillas is None or workspace <= max_ancillas:
                legacy.append(candidate)
    best = min(legacy, key=lambda candidate: candidate.key, default=None)
    copies = []
    if variant in ('auto', 'select_copy'):
        layouts = ((explicit_packets, ) if explicit_packets is not None else
                   set(packet_layout(b, alpha=i)
                       for i in range(1, b + 1)) | set(
                           packet_layout(b, packet_width=i)
                           for i in range(1, b + 1)))
        sizes = (block_size, ) if block_size is not None else (
            1 << low for low in range(1, n))
        for size in sizes:
            for packets in layouts:
                resources = select_copy_resources(len(entries),
                                                  n,
                                                  b,
                                                  block_size=size,
                                                  packets=packets,
                                                  controlled=controlled,
                                                  dirty_banks=dirty_banks)
                if max_ancillas is None or resources.num_ladder <= max_ancillas:
                    copies.append(
                        (resources.toffoli_count, resources.num_ladder, size,
                         packets.widths, packets))
    copy = min(copies, key=lambda item: item[:4], default=None)
    # Equal logical cost keeps the legacy winner. Within SELECT-COPY, minimize
    # total clean+dirty workspace, then B, then lexicographic packet widths.
    if copy is not None and (best is None or copy[0] < best.cost):
        _, _, size, _, packets = copy
        plan = plan_select_copy(entries,
                                n,
                                b,
                                block_size=size,
                                packets=packets,
                                controlled=controlled,
                                dirty_banks=dirty_banks)
        return MeasuredQROMSelection(plan, 'select_copy', size, 'traverse',
                                     None, packets)
    if best is None:
        raise ValueError(
            'no configuration fits the ancilla budget and phase geometry')
    plan = _emit_selection(entries, n, b, best, controlled, dirty_banks)
    return MeasuredQROMSelection(plan, best.variant, best.block_size,
                                 best.cache_uncompute, best.phase_block_size)
