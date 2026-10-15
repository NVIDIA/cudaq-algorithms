# SPDX-License-Identifier: Apache-2.0
"""Pure cache X-erasure and outcome-dependent phase-correction plans.

Masks retain individual slot indices, including slots above 64. They are not
outcomes: each invocation supplies its own persistent measurement results.
"""
from typing import Literal, Sequence
from . import _unary_iteration_ir as ir
from ._qrom_plan import QROMPlan, QROMResources, QROMStageResources


def _integer(value, name, minimum=0):
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _stage(name, ops, *, tree=False, temporary=False):
    toffolis = sum(op[0] in ir._TOFFOLI_OPCODES for op in ops)
    measured = sum(op[0] in (*ir._MEASURED_OPCODES, 24, 25) for op in ops)
    return QROMStageResources(name, toffolis, toffolis if tree else 0,
                              toffolis if tree or temporary else 0, measured)


def _plan_cache_measurement(
        cache_bits: int,
        *,
        cache_offset: int,
        cache_register: Literal['ladder', 'output'] = 'ladder') -> QROMPlan:
    """Measure/reset a known cache and bind its bits to slots 0..w-1."""
    _integer(cache_bits, 'cache_bits', 1)
    _integer(cache_offset, 'cache_offset')
    if cache_register not in ('ladder', 'output'):
        raise ValueError('cache_register must be ladder or output')
    opcode = ir._OP_MEASURE_RESET if cache_register == 'ladder' else ir._OP_MEASURE_RESET_OUTPUT
    ops = tuple((opcode, cache_offset + t, t, 0) for t in range(cache_bits))
    clean = cache_offset + cache_bits if cache_register == 'ladder' else 0
    resources = QROMResources((_stage('cache_measure', ops), ), clean, 0,
                              cache_bits)
    return QROMPlan(ops, (), cache_bits, resources)


def _plan_phase_correction(parity_masks: tuple[tuple[int, ...], ...],
                           address_bits: int,
                           *,
                           controlled: bool,
                           phase_block_size: int,
                           unary_offset: int,
                           tree_offset: int,
                           address_offset: int = 0) -> QROMPlan:
    """Correct cache phases with incoming results and restored clean scratch.

    Input masks are indexed by cache row. The returned mask registry is
    deduplicated and its operations address that registry. Its required result
    capacity is inferred from used slots; the enclosing cache prefix can retain
    additional unused slots without allocating a second buffer.
    """
    _integer(address_bits, 'address_bits', 1)
    _integer(phase_block_size, 'phase_block_size', 1)
    for name, value in [('unary_offset', unary_offset),
                        ('tree_offset', tree_offset),
                        ('address_offset', address_offset)]:
        _integer(value, name)
    if not isinstance(controlled, bool):
        raise ValueError('controlled must be a bool')
    k = phase_block_size
    if k & (k - 1) or k > 1 << (address_bits - 1):
        raise ValueError(
            'phase_block_size must be a power of two leaving a nonempty high address'
        )
    rows = tuple(tuple(mask) for mask in parity_masks)
    if not 1 <= len(rows) <= 1 << address_bits:
        raise ValueError('row count outside address capacity')
    for mask in rows:
        if any(not isinstance(t, int) or t < 0
               for t in mask) or len(set(mask)) != len(mask):
            raise ValueError('mask slots must be unique nonnegative integers')
    rows = tuple(tuple(sorted(mask)) for mask in rows)
    masks = tuple(dict.fromkeys(rows))
    mask_ids = {mask: i for i, mask in enumerate(masks)}
    num_results = max((t + 1 for mask in masks for t in mask), default=0)
    s = k.bit_length() - 1
    high = address_bits - s
    if k > 1 and max(tree_offset, unary_offset) < min(tree_offset + high,
                                                      unary_offset + k):
        raise ValueError('tree and one-hot workspace must be disjoint')
    prepare, erase, splits = [], [], []
    if k > 1:
        prepare.append((ir._OP_X_LADDER, unary_offset, 0, 0))
        active = [0]
        for bit in reversed(range(s)):
            next_active = []
            for parent in active:
                child = parent + (1 << bit)
                p, a, q = unary_offset + parent, address_offset + bit, unary_offset + child
                prepare.extend(((ir._OP_CCX, p, a, q),
                                (ir._OP_CX_LADDER_LADDER, q, p, 0)))
                splits.append((p, a, q))
                next_active.extend((parent, child))
            active = next_active
        for p, a, q in reversed(splits):
            erase.extend(((ir._OP_CX_LADDER_LADDER, q, p, 0),
                          (ir._OP_MEAS_UNCOMPUTE, q, p, a)))
        erase.append((ir._OP_X_LADDER, unary_offset, 0, 0))
    # Body Z is a row token until lowered here to a parity-controlled phase.
    bodies = tuple(
        tuple(('z', row) for row in range(u * k, min((u + 1) * k, len(rows)))
              if rows[row]) for u in range((len(rows) + k - 1) // k))
    raw = ir._emit_measured_walk(high, len(bodies), controlled, bodies, False)
    walk = []
    for opcode, a, b, c in raw:
        if opcode == ir._OP_BODY_Z:
            leaf, row = a + tree_offset, b
            mask = mask_ids[rows[row]]
            walk.append((ir._OP_PARITY_Z_LADDER, leaf, mask,
                         0) if k == 1 else (ir._OP_PARITY_CZ_LADDER, leaf,
                                            unary_offset + row % k, mask))
        else:
            operands = [a, b, c]
            for pos in ir._ADDRESS_OPERANDS.get(opcode, ()):
                operands[pos] += address_offset + s
            for pos in ir._LADDER_OPERANDS.get(opcode, ()):
                operands[pos] += tree_offset
            walk.append((opcode, *operands))
    ops = tuple(prepare + walk + erase)
    clean = max(tree_offset + high, unary_offset + k if k > 1 else 0)
    stages = (_stage('phase_unary_prepare', prepare,
                     temporary=True), _stage('phase_walk', walk, tree=True),
              _stage('phase_unary_erase', erase))
    resources = QROMResources(stages, clean, 0, num_results)
    ir.validate_feedback_ops(ops,
                             parity_masks=masks,
                             num_results=num_results,
                             num_address=address_offset + address_bits,
                             num_ladder=clean,
                             num_output=0,
                             controlled=controlled,
                             initialized_results=range(num_results))
    return QROMPlan(ops, masks, num_results, resources)


def _plan_cache_erasure(cache_words: Sequence[int],
                        address_bits: int,
                        cache_bits: int,
                        *,
                        controlled: bool,
                        phase_block_size: int,
                        cache_offset: int,
                        tree_offset: int,
                        address_offset: int = 0,
                        cache_register: Literal['ladder', 'output'] = 'ladder',
                        unary_offset: int | None = None) -> QROMPlan:
    """Compose the reusable measurement prefix and measured phase provider."""
    prefix = _plan_cache_measurement(cache_bits,
                                     cache_offset=cache_offset,
                                     cache_register=cache_register)
    if unary_offset is None:
        if cache_register == 'output':
            raise ValueError(
                'output cache erasure requires explicit clean unary_offset')
        unary_offset = cache_offset
    if phase_block_size > cache_bits:
        raise ValueError('phase_block_size cannot exceed cache width')
    words = tuple(cache_words)
    if any(not isinstance(word, int) or not 0 <= word < 1 << cache_bits
           for word in words):
        raise ValueError('cache word outside cache width')
    rows = tuple(
        tuple(t for t in range(cache_bits) if word & (1 << t))
        for word in words)
    phase = _plan_phase_correction(rows,
                                   address_bits,
                                   controlled=controlled,
                                   phase_block_size=phase_block_size,
                                   unary_offset=unary_offset,
                                   tree_offset=tree_offset,
                                   address_offset=address_offset)
    ops = prefix.ops + phase.ops
    resources = QROMResources(
        prefix.resources.stages + phase.resources.stages,
        max(prefix.resources.num_clean, phase.resources.num_clean), 0,
        cache_bits)
    ir.validate_feedback_ops(ops,
                             parity_masks=phase.parity_masks,
                             num_results=cache_bits,
                             num_address=address_offset + address_bits,
                             num_ladder=resources.num_clean,
                             num_output=cache_offset +
                             cache_bits if cache_register == 'output' else 0,
                             controlled=controlled)
    return QROMPlan(ops, phase.parity_masks, cache_bits, resources)
