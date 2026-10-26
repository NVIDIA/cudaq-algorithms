# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure static plans for explicit retained-bank QROM epochs.

During an epoch the address and logical control are immutable. Callers may act
on output and other data, including gates conditioned on address/control, but
must leave ladder (bank, cache and temporary work) untouched. Close with end
before changing address/control; no runtime state tracking enforces this rule.
"""
from dataclasses import dataclass
from numbers import Integral
from ._qrom_common import BlockGeometry, validate_qrom_data
from ._qrom_plan import QROMPlan, QROMResources, QROMStageResources
from ._select_copy_plan import _high_walk, _low_copy
from ._unary_iteration_ir import (
    _OP_BODY_X,
    _OP_CCX,
    _OP_CCX_CTRL,
    _OP_CCX_LADDER_LADDER_TARGET,
    _OP_CCX_LADDER_LADDER,
    _OP_MEAS_UNCOMPUTE,
    _OP_MEAS_UNCOMPUTE_CTRL,
)


@dataclass(frozen=True)
class RetainedQROMPlan:
    begin: QROMPlan
    steps: tuple[QROMPlan, ...]
    end: QROMPlan
    num_address: int
    num_output: int
    num_cache: int
    resources: QROMResources


def _validated_tables(tables, address_bits, output_bits):
    tables = tuple(tables)
    if not tables:
        raise ValueError('tables must be non-empty')
    checked = []
    for table in tables:
        data, address_bits, output_bits = validate_qrom_data(
            table, address_bits, output_bits)
        checked.append(data)
    return tuple(checked), address_bits, output_bits


def difference_tables(tables, address_bits: int, output_bits: int):
    """Return m+1 padded XOR differences for m user tables.

    Starting from zero output and with word-preserving interleavings, successive
    prefixes give each table and finally zero. Arbitrary output operations are
    still legal, but do not imply those stronger output-value statements.
    """
    tables, _, _ = _validated_tables(tables, address_bits, output_bits)
    width = max(map(len, tables))
    previous = (0, ) * width
    differences = []
    for table in tables:
        current = table + (0, ) * (width - len(table))
        differences.append(tuple(a ^ b for a, b in zip(previous, current)))
        previous = current
    differences.append(previous)
    return tuple(differences)


def plan_retained_xor(tables: tuple[tuple[int, ...], ...], address_bits: int,
                      output_bits: int, *, block_size: int, controlled: bool,
                      dirty_banks: bool) -> RetainedQROMPlan:
    """Plan raw XOR lookups sharing a full-word bank and clean offset cache.

    Layout: reusable walk lines, full-word clean cache, then full-word banks.
    Dirty banks are borrowed without a restriction on joint entanglement. Begin
    copies c*D_r into the cache; each step fuses the previous/current bank
    differences, copies the selected bank and cancels its cached offset. End
    restores the original bank before uncomputing the cache. All kernels share
    one chain-wide layout and all endpoint instructions count in resources.
    """
    tables, address_bits, output_bits = _validated_tables(
        tables, address_bits, output_bits)
    if (isinstance(block_size, bool) or not isinstance(block_size, Integral)
            or block_size < 2 or block_size & (block_size - 1)
            or block_size >= 1 << address_bits):
        raise ValueError(
            'block_size must be a power of two >= 2 with a high address bit')
    if not isinstance(controlled, bool) or not isinstance(dirty_banks, bool):
        raise ValueError('controlled and dirty_banks must be bools')
    block_size = int(block_size)
    geometry = BlockGeometry(address_bits, output_bits, max(map(len, tables)),
                             block_size)
    low, high = geometry.low_bits, geometry.high_bits
    cache_offset = max(low, high)
    bank_offset = cache_offset + output_bits
    bank_width = (block_size - 1) * output_bits
    num_clean = bank_offset + (0 if dirty_banks else bank_width)
    num_dirty = bank_width if dirty_banks else 0

    def stage(name, tape):
        tape = tuple(tape)
        trees = sum(op in (_OP_CCX, _OP_CCX_CTRL) for op, *_ in tape)
        generic = sum(op in (_OP_CCX_LADDER_LADDER_TARGET,
                             _OP_CCX_LADDER_LADDER) for op, *_ in tape)
        measured = sum(op in (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)
                       for op, *_ in tape)
        resources = QROMResources((QROMStageResources(
            name, trees + generic, trees, trees, measured), ), num_clean,
                                  num_dirty)
        return QROMPlan(tape, (), 0, resources)

    def high_walk(bodies):
        return _high_walk(bodies,
                          high_bits=high,
                          num_blocks=geometry.num_blocks,
                          controlled=controlled,
                          low_bits=low,
                          output_bits=output_bits,
                          bank_offset=bank_offset)

    def copy(cache=False):
        return _low_copy(low_bits=low,
                         block_size=block_size,
                         controlled=controlled,
                         stride=output_bits,
                         width=output_bits,
                         bank_offset=bank_offset,
                         cache_offset=cache_offset if cache else None)

    begin = stage('begin_cache', copy(cache=True))
    previous = tuple((0, ) * block_size for _ in range(geometry.num_blocks))
    steps = []
    for j, data in enumerate(tables):
        bodies, current = [], []
        for q in range(geometry.num_blocks):
            words = tuple(data[q * block_size + i] if q * block_size +
                          i < len(data) else 0 for i in range(block_size))
            differences = tuple(word ^ words[0] for word in words)
            body = [('x', t) for t in range(output_bits) if words[0] >> t & 1]
            for i in range(1, block_size):
                delta = previous[q][i] ^ differences[i]
                body.extend(('x', output_bits + (i - 1) * output_bits + t)
                            for t in range(output_bits) if delta >> t & 1)
            bodies.append(body)
            current.append(differences)
        tape = high_walk(bodies) + copy()
        tape += tuple(
            (_OP_BODY_X, cache_offset + t, t, 0) for t in range(output_bits))
        steps.append(stage(f'xor_step_{j}', tape))
        previous = tuple(current)
    unload = [
        tuple(('x', output_bits + (i - 1) * output_bits + t)
              for i in range(1, block_size) for t in range(output_bits)
              if row[i] >> t & 1) for row in previous
    ]
    end = stage('end_unload_cache', high_walk(unload) + copy(cache=True))
    resources = QROMResources(
        tuple(s for p in (begin, *steps, end) for s in p.resources.stages),
        num_clean, num_dirty)
    return RetainedQROMPlan(begin, tuple(steps), end, address_bits,
                            output_bits, output_bits, resources)


def retained_resources(num_loads, num_entries, address_bits, output_bits, *,
                       block_size, controlled, dirty_banks):
    """Price raw s-load epochs including both endpoints and the clean cache.

    Trees: (s+1)H + (s+2)L. Generic Toffolis: (s+2)(B-1)b.
    A public m-table difference chain has s=m+1, not s=m.
    Cache-target CCX computes remain generic, never temporary ANDs.
    """
    from ._unary_iteration_ir import _measured_walk_toffoli_count as count
    geometry = BlockGeometry(address_bits, output_bits, num_entries,
                             block_size)
    low, high = geometry.low_bits, geometry.high_bits
    H, L = count(high, geometry.num_blocks,
                 controlled), count(low, block_size, controlled)
    bank = (block_size - 1) * output_bits

    def stage(name, high_count):
        trees = high_count + L
        return QROMStageResources(name, trees + bank, trees, trees, trees)

    stages = (stage('begin_cache', 0), ) + tuple(
        stage(f'xor_step_{j}', H)
        for j in range(num_loads)) + (stage('end_unload_cache', H), )
    clean = max(low, high) + output_bits + (0 if dirty_banks else bank)
    return QROMResources(stages, clean, bank if dirty_banks else 0)


@dataclass(frozen=True)
class RetainedQROMSelection:
    block_size: int
    plan: RetainedQROMPlan


def select_retained_qrom(tables,
                         address_bits,
                         output_bits,
                         *,
                         block_size=None,
                         controlled=False,
                         dirty_banks=True,
                         max_ancillas=None):
    """Select by Toffolis, total workspace, then smaller B; emit only the winner."""
    tables, a, b = _validated_tables(tables, address_bits, output_bits)
    if not isinstance(controlled, bool) or not isinstance(dirty_banks, bool):
        raise ValueError('controlled and dirty_banks must be bools')
    if max_ancillas is not None and (isinstance(max_ancillas, bool)
                                     or not isinstance(max_ancillas, Integral)
                                     or max_ancillas < 0):
        raise ValueError('max_ancillas must be a nonnegative integer budget')
    if block_size is not None and (isinstance(block_size, bool)
                                   or not isinstance(block_size, Integral)
                                   or block_size < 2 or block_size &
                                   (block_size - 1) or block_size >= 1 << a):
        raise ValueError(
            'block_size must be a power of two >= 2 with a high address bit')
    candidates = []
    for B in ((block_size, ) if block_size is not None else
              (1 << low for low in range(1, a))):
        resources = retained_resources(len(tables),
                                       max(map(len, tables)),
                                       a,
                                       b,
                                       block_size=B,
                                       controlled=controlled,
                                       dirty_banks=dirty_banks)
        if max_ancillas is None or resources.num_ladder <= max_ancillas:
            candidates.append(
                (resources.toffoli_count, resources.num_ladder, B))
    if not candidates:
        raise ValueError(
            'no retained configuration fits the ancilla budget and geometry')
    B = min(candidates)[2]
    return RetainedQROMSelection(
        B,
        plan_retained_xor(tables,
                          a,
                          b,
                          block_size=B,
                          controlled=controlled,
                          dirty_banks=dirty_banks))
