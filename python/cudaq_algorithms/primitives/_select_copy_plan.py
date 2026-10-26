# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure host packet geometry for SELECT-COPY QROM planning."""
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class PacketLayout:
    """Packet boundaries and output fanout of each local bank-bit position."""

    widths: tuple[int, ...]
    offsets: tuple[int, ...]
    stride: int
    fanout: tuple[tuple[int, ...], ...]


def packet_layout(output_bits: int,
                  *,
                  alpha: int | None = None,
                  packet_width: int | None = None) -> PacketLayout:
    """Partition a word into balanced packets or fixed-width packets.

    ``alpha`` counts balanced, widest-first packets; ``packet_width`` sets
    their maximum width with a final remainder. With neither option, use one
    packet. Every bank-bit position fans out to that position in each packet
    which contains it.
    """

    def positive_integer(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f'{name} must be a positive integer')
        if value < 1:
            raise ValueError(f'{name} must be a positive integer')
        return int(value)

    output_bits = positive_integer(output_bits, 'output_bits')
    if alpha is not None and packet_width is not None:
        raise ValueError('alpha and packet_width are mutually exclusive')
    if packet_width is not None:
        packet_width = positive_integer(packet_width, 'packet_width')
        if packet_width > output_bits:
            raise ValueError('packet_width must not exceed output_bits')
        full, remainder = divmod(output_bits, packet_width)
        widths = (packet_width, ) * full + ((remainder, ) if remainder else ())
    else:
        alpha = 1 if alpha is None else positive_integer(alpha, 'alpha')
        if alpha > output_bits:
            raise ValueError('alpha must not exceed output_bits')
        width, remainder = divmod(output_bits, alpha)
        widths = (width + 1, ) * remainder + (width, ) * (alpha - remainder)
    offsets = []
    offset = 0
    for width in widths:
        offsets.append(offset)
        offset += width
    stride = max(widths)
    fanout = tuple(
        tuple(start + bit for start, width in zip(offsets, widths)
              if bit < width) for bit in range(stride))
    return PacketLayout(widths, tuple(offsets), stride, fanout)


from ._qrom_common import BlockGeometry, rebase_block_walk, validate_qrom_data
from ._qrom_plan import QROMPlan, QROMResources, QROMStageResources
from ._unary_iteration_ir import (
    _emit_measured_walk,
    _OP_BODY_X,
    _OP_CCX,
    _OP_CCX_CTRL,
    _OP_CCX_LADDER_LADDER_TARGET,
    _OP_CCX_LADDER_LADDER,
    _OP_MEAS_UNCOMPUTE_LADDER,
    _OP_MEAS_UNCOMPUTE,
    _OP_MEAS_UNCOMPUTE_CTRL,
)


def _high_walk(bodies, *, high_bits, num_blocks, controlled, low_bits,
               output_bits, bank_offset):
    """Shared high-address transition, with output/bank body tokens."""
    return rebase_block_walk(_emit_measured_walk(high_bits, num_blocks,
                                                 controlled, bodies, False),
                             address_offset=low_bits,
                             output_bits=output_bits,
                             bank_offset=bank_offset)


def _low_copy(*,
              low_bits,
              block_size,
              controlled,
              stride,
              width,
              bank_offset,
              output_offset=0,
              cache_offset=None):
    """Copy selected bank to output or a persistent clean cache.

    Cache-target CCX gates are generic coherent gates: opcode28 alone does
    not prove temporary-AND eligibility when its target is a live cache.
    """
    if not low_bits:
        return ()
    bodies = [()] + [
        tuple(('x', (i - 1) * stride + t) for t in range(width))
        for i in range(1, block_size)
    ]
    tape = []
    for op, a, b, c in _emit_measured_walk(low_bits, block_size, controlled,
                                           bodies, False):
        if op == _OP_BODY_X:
            opcode = (_OP_CCX_LADDER_LADDER_TARGET
                      if cache_offset is None else _OP_CCX_LADDER_LADDER)
            target = (
                (output_offset if cache_offset is None else cache_offset) +
                b % stride)
            tape.append((opcode, a, bank_offset + b, target))
        else:
            tape.append((op, a, b, c))
    return tuple(tape)


def plan_select_copy(data: tuple[int,
                                 ...], address_bits: int, output_bits: int, *,
                     block_size: int, packets: PacketLayout, controlled: bool,
                     dirty_banks: bool) -> QROMPlan:
    """Compile packet transitions, copies, unload and serialized Restore.

    Tree lines occupy a reusable clean prefix of max(high_bits, low_bits).
    Dirty Restore has a separate single clean scratch immediately after it;
    banks follow the clean work (and are themselves clean in clean-bank mode).
    Every measured AND uses exact CCX compute and H/mz/CZ/reset disposal.
    """
    data, address_bits, output_bits = validate_qrom_data(
        data, address_bits, output_bits)
    if (isinstance(block_size, bool) or not isinstance(block_size, Integral)
            or block_size < 2 or block_size & (block_size - 1)
            or block_size >= 1 << address_bits):
        raise ValueError(
            'block_size must be a power of two >= 2 with a high address bit')
    block_size = int(block_size)
    if not isinstance(controlled, bool) or not isinstance(dirty_banks, bool):
        raise ValueError('controlled and dirty_banks must be bools')
    if (not isinstance(packets, PacketLayout) or not packets.widths
            or any(type(w) is not int or w < 1 for w in packets.widths)
            or sum(packets.widths) != output_bits or packets.offsets != tuple(
                sum(packets.widths[:i]) for i in range(len(packets.widths)))
            or packets.stride != max(packets.widths)
            or packets.fanout != tuple(
                tuple(o + t for o, w in zip(packets.offsets, packets.widths)
                      if t < w) for t in range(packets.stride))):
        raise ValueError(
            'packets must be a consistent partition of output_bits')
    geometry = BlockGeometry(address_bits, output_bits, len(data), block_size)
    low, high = geometry.low_bits, geometry.high_bits
    tree_width = max(low, high)
    bank_width = (block_size - 1) * packets.stride
    scratch_needed = bool(dirty_banks and bank_width)
    scratch = tree_width
    bank_offset = tree_width + int(scratch_needed)
    num_clean = bank_offset + (0 if dirty_banks else bank_width)
    num_dirty = bank_width if dirty_banks else 0
    ops, stages = [], []

    def stage(name, tape):
        tape = tuple(tape)
        trees = sum(op in (_OP_CCX, _OP_CCX_CTRL) for op, *_ in tape)
        restore = sum(op == _OP_CCX_LADDER_LADDER for op, *_ in tape)
        generic = sum(op == _OP_CCX_LADDER_LADDER_TARGET for op, *_ in tape)
        measured = sum(op in (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL,
                              _OP_MEAS_UNCOMPUTE_LADDER) for op, *_ in tape)
        stages.append(
            QROMStageResources(name, trees + restore + generic, trees,
                               trees + restore, measured))
        ops.extend(tape)

    previous = [[0] * block_size for _ in range(geometry.num_blocks)]
    for packet, (width,
                 offset) in enumerate(zip(packets.widths, packets.offsets)):
        bodies, current = [], []
        for q in range(geometry.num_blocks):
            words = [((data[q * block_size + i] if q * block_size +
                       i < len(data) else 0) >> offset) & ((1 << width) - 1)
                     for i in range(block_size)]
            differences = [v ^ words[0] for v in words]
            body = [('x', offset + t) for t in range(width)
                    if words[0] >> t & 1]
            for i in range(1, block_size):
                delta = previous[q][i] ^ differences[i]
                body.extend(('x', output_bits + (i - 1) * packets.stride + t)
                            for t in range(packets.stride) if delta >> t & 1)
            bodies.append(body)
            current.append(differences)
        tape = _high_walk(bodies,
                          high_bits=high,
                          num_blocks=geometry.num_blocks,
                          controlled=controlled,
                          low_bits=low,
                          output_bits=output_bits,
                          bank_offset=bank_offset)
        stage(f'high_transition_{packet}', tape)
        previous = current
        if low:
            stage(
                f'low_copy_{packet}',
                _low_copy(low_bits=low,
                          block_size=block_size,
                          controlled=controlled,
                          stride=packets.stride,
                          width=width,
                          bank_offset=bank_offset,
                          output_offset=offset))
    bodies = []
    for row in previous:
        bodies.append(
            tuple(('x', output_bits + (i - 1) * packets.stride + t)
                  for i in range(1, block_size) for t in range(packets.stride)
                  if row[i] >> t & 1))
    stage(
        'bank_unload',
        _high_walk(bodies,
                   high_bits=high,
                   num_blocks=geometry.num_blocks,
                   controlled=controlled,
                   low_bits=low,
                   output_bits=output_bits,
                   bank_offset=bank_offset))
    if scratch_needed:
        bodies = [()] + [
            tuple(('x', (i - 1) * packets.stride + t)
                  for t in range(packets.stride))
            for i in range(1, block_size)
        ]
        tape = []
        for op, a, b, c in _emit_measured_walk(low, block_size, controlled,
                                               bodies, False):
            if op == _OP_BODY_X:
                bank = bank_offset + b
                tape.append((_OP_CCX_LADDER_LADDER, a, bank, scratch))
                tape.extend((_OP_BODY_X, scratch, target, 0)
                            for target in packets.fanout[b % packets.stride])
                tape.append((_OP_MEAS_UNCOMPUTE_LADDER, scratch, a, bank))
            else:
                tape.append((op, a, b, c))
        stage('restore', tape)
    # Walk/Restore outcomes are consumed immediately; no persistent slots.
    return QROMPlan(tuple(ops), (), 0,
                    QROMResources(tuple(stages), num_clean, num_dirty))


def _plan_select_copy_phase(parity_masks: tuple[tuple[int, ...], ...],
                            address_bits: int,
                            *,
                            controlled: bool,
                            block_size: int,
                            bank_offset: int,
                            tree_offset: int,
                            address_offset: int = 0) -> QROMPlan:
    """Apply runtime parity phases while restoring an arbitrary borrowed bank.

    Rows contain outcome-slot indices, with repeated indices cancelling modulo
    two. No outcome values enter this static plan. The clean tree prefix is
    reused between high/low walks; the borrowed suffix has B-1 one-bit banks.
    It must follow that prefix, including any caller-requested offset gaps.

    Loading c*(p(q,r) XOR p(q,0)), copying its phase, unloading and copying
    again cancels the original dirty phase. The base phase supplies c*p(q,0).
    Both low walks use the same coherent control, so the resulting factor is
    (-1)**(c*p(q,r)), including on an entangled inactive control branch.
    """
    from . import _unary_iteration_ir as ir

    for name, value, minimum in (('address_bits', address_bits,
                                  1), ('block_size', block_size,
                                       2), ('bank_offset', bank_offset, 0),
                                 ('tree_offset', tree_offset,
                                  0), ('address_offset', address_offset, 0)):
        if isinstance(
                value,
                bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    address_bits, block_size, bank_offset, tree_offset, address_offset = map(
        int,
        (address_bits, block_size, bank_offset, tree_offset, address_offset))
    if not isinstance(controlled, bool):
        raise ValueError('controlled must be a bool')
    if block_size & (block_size - 1) or block_size >= 1 << address_bits:
        raise ValueError(
            'block_size must be a power of two leaving a high address bit')
    low = block_size.bit_length() - 1
    high = address_bits - low
    if bank_offset < tree_offset + max(low, high):
        raise ValueError(
            'borrowed bank must follow the disjoint clean tree prefix')
    supplied = tuple(tuple(mask) for mask in parity_masks)
    if not 1 <= len(supplied) <= 1 << address_bits:
        raise ValueError('row count outside address capacity')
    rows = []
    for mask in supplied:
        active = set()
        for slot in mask:
            if isinstance(slot,
                          bool) or not isinstance(slot, Integral) or slot < 0:
                raise ValueError('mask slots must be nonnegative integers')
            active.symmetric_difference_update((int(slot), ))
        rows.append(tuple(sorted(active)))
    num_results = max((int(t) + 1 for mask in supplied for t in mask),
                      default=0)
    num_blocks = (len(rows) + block_size - 1) // block_size
    rows.extend([()] * (num_blocks * block_size - len(rows)))
    masks, mask_ids = [], {}

    def mask_id(mask):
        if mask not in mask_ids:
            mask_ids[mask] = len(masks)
            masks.append(mask)
        return mask_ids[mask]

    load_bodies = []
    for q in range(num_blocks):
        base = rows[q * block_size]
        changes = []
        for r in range(1, block_size):
            delta = tuple(sorted(set(base) ^ set(rows[q * block_size + r])))
            if delta:
                # Body tokens are static (bank index, parity-mask index) metadata.
                changes.append((r - 1, mask_id(delta)))
        load_bodies.append((base, changes))

    def walk(bits, bodies, address_shift, lower):
        tape = []
        for code, a, b, c in ir._emit_measured_walk(bits, len(bodies),
                                                    controlled, bodies, False):
            if code == ir._OP_BODY_X:
                tape.extend(lower(a + tree_offset, b))
            else:
                operands = [a, b, c]
                for pos in ir._ADDRESS_OPERANDS.get(code, ()):
                    operands[pos] += address_offset + address_shift
                for pos in ir._LADDER_OPERANDS.get(code, ()):
                    operands[pos] += tree_offset
                tape.append((code, *operands))
        return tuple(tape)

    high_bodies = tuple((('x', q), ) for q in range(num_blocks))

    def high_body(leaf, q, include_base):
        base, changes = load_bodies[q]
        result = []
        if include_base and base:
            result.append((ir._OP_PARITY_Z_LADDER, leaf, mask_id(base), 0))
        result.extend(
            (ir._OP_PARITY_CX_LADDER_LADDER, leaf, bank_offset + r, m)
            for r, m in changes)
        return result

    load = walk(high, high_bodies, low,
                lambda leaf, q: high_body(leaf, q, True))
    unload = walk(high, high_bodies, low,
                  lambda leaf, q: high_body(leaf, q, False))
    low_bodies = ((), ) + tuple((('x', r - 1), ) for r in range(1, block_size))
    copy = walk(
        low, low_bodies, 0, lambda leaf, r:
        ((ir._OP_CZ_LADDER_LADDER, leaf, bank_offset + r, 0), ))
    stages, ops = [], []
    for name, tape in (('phase_high_load', load), ('phase_low_copy', copy),
                       ('phase_high_unload', unload), ('phase_dirty_echo',
                                                       copy)):
        trees = sum(code in (ir._OP_CCX, ir._OP_CCX_CTRL) for code, *_ in tape)
        measurements = sum(code in ir._MEASURED_OPCODES for code, *_ in tape)
        stages.append(
            QROMStageResources(name, trees, trees, trees, measurements))
        ops.extend(tape)
    resources = QROMResources(tuple(stages), bank_offset, block_size - 1,
                              num_results)
    ir.validate_feedback_ops(ops,
                             parity_masks=masks,
                             num_results=num_results,
                             num_address=address_offset + address_bits,
                             num_ladder=resources.num_ladder,
                             num_output=0,
                             controlled=controlled,
                             initialized_results=range(num_results))
    return QROMPlan(tuple(ops), tuple(masks), num_results, resources)


def select_copy_resources(num_entries, address_bits, output_bits, *,
                          block_size, packets, controlled, dirty_banks):
    """Price the proved schedule without materializing a tape or table bodies.

    With p packets, high tree cost H and low tree cost L, the trees cost
    (p+1)H + (p+d)L, generic copies cost (B-1)b and Restore costs d(B-1)w.
    Here d is dirty-bank mode and w is the widest packet. Restore and all
    tree computes are temporary ANDs; generic copies are exact Toffolis.
    """
    from ._unary_iteration_ir import _measured_walk_toffoli_count as count
    geometry = BlockGeometry(address_bits, output_bits, num_entries,
                             block_size)
    low, high = geometry.low_bits, geometry.high_bits
    H = count(high, geometry.num_blocks, controlled)
    L = count(low, block_size, controlled)
    stages = []
    for j, width in enumerate(packets.widths):
        stages.append(QROMStageResources(f'high_transition_{j}', H, H, H, H))
        stages.append(
            QROMStageResources(f'low_copy_{j}', L + (block_size - 1) * width,
                               L, L, L))
    stages.append(QROMStageResources('bank_unload', H, H, H, H))
    bank = (block_size - 1) * packets.stride
    if dirty_banks:
        stages.append(
            QROMStageResources('restore', L + bank, L, L + bank, L + bank))
    clean = max(low, high) + (1 if dirty_banks else bank)
    return QROMResources(tuple(stages), clean, bank if dirty_banks else 0)
