"""Independent formula enumeration and pure measured strategy contracts."""
from dataclasses import FrozenInstanceError
import pytest
from qrom_host_loader import load_qrom_host_module


def planner():
    return load_qrom_host_module('_qrom_measured_plan')._plan_measured_qrom


def count(n, m, c):
    return m + n - 2 + int(c) - (m - 1).bit_count()


def candidates(n,
               m,
               b,
               c,
               variant='auto',
               cleanup='auto',
               block=None,
               phase=None):
    result = []
    if variant != 'select_swap':
        result.append((count(n, m,
                             c), 0, 0, 0, 0, 'select', None, 'none', None))
    if variant != 'select':
        for B in ([block] if block else [2**s for s in range(1, n)]):
            h = n - (B.bit_length() - 1)
            M = (m + B - 1) // B
            L = count(h, M, c)
            R = 2 * b * (B - 1)
            if cleanup != 'measure':
                result.append((2 * L + R, 1, 0, B, 0, 'select_swap', B,
                               'traverse', None))
            if cleanup != 'traverse':
                for K in ([phase] if phase else
                          [2**s for s in range(h) if 2**s <= B * b]):
                    if K > min(B * b, 2**(h - 1)):
                        continue
                    P = count(h - (K.bit_length() - 1), (M + K - 1) // K, c)
                    result.append((L + R + P + K - 1, 1, 1, B, K,
                                   'select_swap', B, 'measure', K))
    return result


@pytest.mark.parametrize('n', range(1, 7))
@pytest.mark.parametrize('b', [1, 2, 8])
@pytest.mark.parametrize('c', [False, True])
def test_auto_matches_independent_candidates(n, b, c):
    for m in sorted({1, max(1, 2**n - 3), 2**n}):
        for cleanup in ['auto', 'traverse']:
            expected = min(candidates(n, m, b, c, cleanup=cleanup))
            got = planner()([i % 2**b for i in range(m)],
                            n,
                            b,
                            controlled=c,
                            cache_uncompute=cleanup)
            assert (got.plan.resources.toffoli_count, got.variant,
                    got.block_size, got.cache_uncompute,
                    got.phase_block_size) == (expected[0], *expected[5:])


@pytest.mark.parametrize('kwargs', [
    {
        'variant': 'select',
        'cache_uncompute': 'measure'
    },
    {
        'cache_uncompute': 'measure'
    },
    {
        'variant': 'select_swap',
        'dirty_banks': True,
        'cache_uncompute': 'measure'
    },
    {
        'variant': 'auto',
        'dirty_banks': True
    },
    {
        'variant': 'select',
        'dirty_banks': True
    },
    {
        'cache_uncompute': 'bad'
    },
    {
        'variant': 'bad'
    },
    {
        'controlled': 1
    },
    {
        'dirty_banks': 1
    },
    {
        'phase_block_size': 1
    },
    {
        'variant': 'select_swap',
        'phase_block_size': 1
    },
    {
        'variant': 'select_swap',
        'cache_uncompute': 'auto',
        'phase_block_size': 1
    },
    *[{
        'variant': 'select_swap',
        'block_size': 2,
        'cache_uncompute': 'measure',
        'phase_block_size': k
    } for k in [0, -1, 3, 1.5, 8]],
    {
        'variant': 'select',
        'block_size': 2
    },
    *[{
        'variant': 'select_swap',
        'block_size': k
    } for k in [0, 1, 3, 16]],
])
def test_invalid_combinations(kwargs):
    with pytest.raises(ValueError):
        planner()([1], 4, 1, **kwargs)


def test_regression_counts_and_immutability():
    got = planner()(range(256),
                    8,
                    8,
                    variant='select_swap',
                    block_size=4,
                    cache_uncompute='measure',
                    phase_block_size=8)
    r = got.plan.resources
    assert (r.toffoli_count, r.num_measurements, r.tree_and_count,
            r.temporary_and_count, r.num_clean,
            r.classical_bits) == (123, 107, 68, 75, 38, 32)
    with pytest.raises(FrozenInstanceError):
        got.variant = 'select'
    target = planner()((i % 4 for i in range(256)),
                       8,
                       2,
                       variant='select_swap',
                       block_size=4,
                       cache_uncompute='measure',
                       phase_block_size=8)
    assert target.plan.resources.toffoli_count == 87
    assert target.plan.resources.num_measurements == 83


def test_one_high_bit_single_entry_and_dirty_auto():
    for m in [1, 8]:
        got = planner()([1] * m,
                        3,
                        1,
                        variant='select_swap',
                        block_size=4,
                        cache_uncompute='measure')
        assert got.phase_block_size == 1
    with pytest.raises(ValueError):
        planner()([1], 1, 1, variant='select_swap')
    for c in [False, True]:
        base = planner()([1] * 19,
                         5,
                         1,
                         variant='select_swap',
                         dirty_banks=True,
                         controlled=c)
        auto = planner()([1] * 19,
                         5,
                         1,
                         variant='select_swap',
                         dirty_banks=True,
                         controlled=c,
                         cache_uncompute='auto')
        assert base == auto


def test_default_tapes_match_captured_baseline():
    import json
    from pathlib import Path
    cases = json.loads(
        Path(__file__).with_name('qrom_constructor_baseline.json').read_text())
    for case in cases:
        if case['filename'] != '_qrom_measured.py':
            continue
        got = planner()(case['data'], case['n'], case['b'], **case['kwargs'])
        assert got.plan.ops == tuple(map(tuple, case['ops']))
        assert (got.plan.resources.toffoli_count,
                got.plan.resources.num_ladder, got.variant,
                got.block_size) == tuple(case['counts'])


@pytest.mark.parametrize('b', [1, 2, 8])
@pytest.mark.parametrize('c', [False, True])
def test_every_small_explicit_geometry_and_counts(b, c):
    ir = load_qrom_host_module('_unary_iteration_ir')
    for n in range(2, 7):
        for m in sorted({1, max(1, 2**n - 3), 2**n}):
            data = [(3 * i + 1) % 2**b for i in range(m)]
            for candidate in candidates(n,
                                        m,
                                        b,
                                        c,
                                        variant='select_swap',
                                        cleanup='measure'):
                expected, _, _, B, K, *_ = candidate
                got = planner()(iter(data),
                                n,
                                b,
                                variant='select_swap',
                                block_size=B,
                                controlled=c,
                                cache_uncompute='measure',
                                phase_block_size=K)
                r = got.plan.resources
                h = n - (B.bit_length() - 1)
                M = (m + B - 1) // B
                tree = count(h, M, c) + count(h - (K.bit_length() - 1),
                                              (M + K - 1) // K, c)
                assert r.toffoli_count == expected
                assert r.tree_and_count == tree
                assert r.temporary_and_count == tree + K - 1
                assert r.num_measurements == tree + K - 1 + B * b
                assert r.num_clean == h + B * b and r.num_dirty == 0
                assert sum(op[0] in ir._TOFFOLI_OPCODES
                           for op in got.plan.ops) == expected
                assert sum(op[0] in (*ir._MEASURED_OPCODES, 24, 25)
                           for op in got.plan.ops) == r.num_measurements


def test_prices_do_not_emit_and_selection_emits_once(monkeypatch):
    module = load_qrom_host_module('_qrom_measured_plan')
    original = module._emit_selection
    calls = []

    def emit(*args):
        calls.append(args[3])
        return original(*args)

    monkeypatch.setattr(module, '_emit_selection', emit)
    prices = list(
        module._price_candidates(14,
                                 16381,
                                 8,
                                 variant='auto',
                                 block_size=None,
                                 controlled=False,
                                 dirty_banks=False,
                                 cache_uncompute='auto',
                                 phase_block_size=None))
    assert len(prices) > 50 and not calls
    planner()(iter([1] * 253), 8, 2, cache_uncompute='auto')
    assert len(calls) == 1


@pytest.mark.parametrize('n', [10, 12, 14])
@pytest.mark.parametrize('b', [1, 8])
@pytest.mark.parametrize('B', [2, 8, 32])
def test_linear_plan_storage(n, b, B, record_property):
    import time
    import sys
    m = 2**n - 3
    start = time.perf_counter()
    got = planner()((i % 2**b for i in range(m)),
                    n,
                    b,
                    variant='select_swap',
                    block_size=B,
                    cache_uncompute='measure')
    elapsed = time.perf_counter() - start
    plan = got.plan
    # These stored records contain no history axis; each tree visit/data bit
    # contributes O(1), and one-hot/routing contribute O(B*b + n).
    stored = len(plan.ops) + sum(len(mask) for mask in plan.parity_masks)
    assert stored <= 32 * (m * b + n + B * b)
    shallow_bytes = (sys.getsizeof(plan.ops) +
                     sum(map(sys.getsizeof, plan.ops)) +
                     sys.getsizeof(plan.parity_masks) +
                     sum(map(sys.getsizeof, plan.parity_masks)))
    record_property('build_seconds', elapsed)
    record_property('record_storage_bytes', shallow_bytes)
    print(
        f'n={n} b={b} B={B}: {elapsed:.4f}s, {stored} records/indices, {shallow_bytes} record bytes'
    )
    assert plan.num_results == B * b


def test_wide_masks_unequal_objects_and_explicit_phase_without_block():
    a = planner()([255] * 125,
                  7,
                  8,
                  variant='select_swap',
                  block_size=16,
                  cache_uncompute='measure',
                  phase_block_size=1)
    frozen = a.plan
    b = planner()([3] * 125,
                  7,
                  2,
                  variant='select_swap',
                  block_size=2,
                  cache_uncompute='measure')
    assert a.plan == frozen and a.plan.num_results == 128 and b.plan.num_results == 4
    assert max(t for mask in a.plan.parity_masks for t in mask) == 127
    expected = min(
        candidates(7,
                   125,
                   2,
                   False,
                   variant='select_swap',
                   cleanup='measure',
                   phase=2))
    got = planner()([3] * 125,
                    7,
                    2,
                    variant='select_swap',
                    cache_uncompute='measure',
                    phase_block_size=2)
    assert got.plan.resources.toffoli_count == expected[0]


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('k', [1, 2])
def test_selected_plan_matches_independent_all_history_channel(controlled, k):
    from test_measured_cache_math import check_emitted_plan
    data = (1, 0, 0, 1, 1)
    got = planner()(data,
                    3,
                    1,
                    variant='select_swap',
                    block_size=2,
                    controlled=controlled,
                    cache_uncompute='measure',
                    phase_block_size=k)
    check_emitted_plan(got.plan, data, controlled, twice=True)


def test_explicit_measure_executes_tie_while_auto_prefers_traverse():
    kwargs = dict(variant='select_swap', block_size=2)
    explicit = planner()([1], 2, 1, cache_uncompute='measure', **kwargs)
    auto = planner()([1], 2, 1, cache_uncompute='auto', **kwargs)
    assert explicit.plan.resources.toffoli_count == auto.plan.resources.toffoli_count
    assert explicit.cache_uncompute == 'measure'
    assert auto.cache_uncompute == 'traverse'
    assert explicit.phase_block_size == 1
