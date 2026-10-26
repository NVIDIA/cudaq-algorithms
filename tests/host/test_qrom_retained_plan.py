"""Independent sequence oracles for explicit retained-bank epochs."""
import itertools
from pathlib import Path
import numpy as np
import pytest
from qrom_host_loader import load_qrom_host_module
from qrom_channel_reference import apply_gate
from test_select_copy_plan import run_tape


def module():
    path = Path(__file__).resolve(
    ).parents[2] / 'python/cudaq_algorithms/primitives/_qrom_retained_plan.py'
    assert path.exists(), 'retained lifecycle planner is not implemented'
    return load_qrom_host_module('_qrom_retained_plan')


def plan(tables=((1, 3, 2), (2, )),
         *,
         controlled=False,
         dirty=True,
         b=2,
         a=2,
         block=2):
    return module().plan_retained_xor(tables,
                                      a,
                                      b,
                                      block_size=block,
                                      controlled=controlled,
                                      dirty_banks=dirty)


def initial(p, controlled, seed=108):
    logical = int(
        controlled) + p.num_address + p.num_output + p.resources.num_dirty
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(1 << logical,
                           2)) + 1j * rng.normal(size=(1 << logical, 2))
    psi /= np.linalg.norm(psi)
    state = np.zeros((len(psi) << p.resources.num_clean, 2), complex)
    state[:len(psi)] = psi
    return state


def xor(state, data, a, b, controlled):
    result = np.zeros_like(state)
    for source in range(len(state)):
        c = source & 1 if controlled else 1
        x = source >> int(controlled) & ((1 << a) - 1)
        value = data[x] if x < len(data) else 0
        result[source ^ ((c * value) << (int(controlled) + a))] = state[source]
    return result


def execute(stage, state, p, controlled, outcome=0):
    m = stage.resources.num_measurements
    history = (outcome, ) * m
    return run_tape(stage, state, p.num_address, p.num_output, controlled,
                    history) * 2**(m / 2)


def caller(state, p, controlled):
    # Noncommuting H on output, then output-controlled operation on reference.
    out = int(controlled) + p.num_address
    result = apply_gate(state, 'h', out)
    selected = (np.arange(len(result)) >> out) & 1 != 0
    result[selected] = result[selected, ::-1]
    return result


@pytest.mark.parametrize('controlled,dirty,outcome',
                         tuple(itertools.product((False, True), repeat=3)))
def test_retained_raw_steps_match_independent_xor_sequence(
        controlled, dirty, outcome):
    tables = ((1, 3, 2), (2, ))
    p = plan(tables, controlled=controlled, dirty=dirty)
    state = expected = initial(p, controlled)
    state = execute(p.begin, state, p, controlled, outcome)
    for data, stage in zip(tables, p.steps):
        state = execute(stage, state, p, controlled, outcome)
        expected = xor(expected, data, 2, 2, controlled)
        state, expected = caller(state, p,
                                 controlled), caller(expected, p, controlled)
    state = execute(p.end, state, p, controlled, outcome)
    np.testing.assert_allclose(state, expected, atol=3e-13, rtol=0)


def test_retained_chain_difference_prefixes_and_endpoints():
    tables = ((1, 3, 2), (2, ))
    # Build differences independently, padding each user table.
    padded = [
        tuple(t[x] if x < len(t) else 0 for x in range(max(map(len, tables))))
        for t in tables
    ]
    expected_differences = (padded[0], tuple(x ^ y for x, y in zip(*padded)),
                            padded[-1])
    differences = module().difference_tables(tables, 2, 2)
    assert differences == expected_differences
    p = plan(differences)
    state = original = initial(p, False)
    state = execute(p.begin, state, p, False)
    for j, stage in enumerate(p.steps):
        state = execute(stage, state, p, False)
        # Close a copy to independently inspect each logical prefix.
        prefix = module().plan_retained_xor(differences[:j + 1],
                                            2,
                                            2,
                                            block_size=2,
                                            controlled=False,
                                            dirty_banks=True)
        closed = execute(prefix.end, state, prefix, False)
        target = xor(original, padded[j], 2, 2,
                     False) if j < len(tables) else original
        np.testing.assert_allclose(closed, target, atol=3e-13, rtol=0)
    state = execute(p.end, state, p, False)
    np.testing.assert_allclose(state, original, atol=3e-13, rtol=0)


def test_retained_output_hadamard_preserves_sequence_equivalence():
    p = plan(((1, 0, 1), (1, )), b=1)
    state = expected = initial(p, False)
    state = execute(p.begin, state, p, False)
    for stage, data in zip(p.steps, ((1, 0, 1), (1, ))):
        state = execute(stage, state, p, False)
        expected = xor(expected, data, 2, 1, False)
        state, expected = apply_gate(state, 'h',
                                     2), apply_gate(expected, 'h', 2)
    state = execute(p.end, state, p, False)
    np.testing.assert_allclose(state, expected, atol=3e-13, rtol=0)


def test_retained_address_mutation_requires_closed_epoch():
    p = plan(((1, 0, 1), ), b=1)
    original = initial(p, False)
    live = execute(p.begin, original, p, False)
    live = apply_gate(live, 'x', 0)
    live = execute(p.steps[0], live, p, False)
    live = execute(p.end, live, p, False)
    expected = xor(apply_gate(original, 'x', 0), (1, 0, 1), 2, 1, False)
    assert np.linalg.norm(live - expected) > .1
    # A completed epoch followed by mutation and a fresh epoch is legal.
    closed = execute(p.begin, original, p, False)
    closed = execute(p.steps[0], closed, p, False)
    closed = execute(p.end, closed, p, False)
    closed = apply_gate(closed, 'x', 0)
    expected = xor(apply_gate(xor(original, (1, 0, 1), 2, 1, False), 'x', 0),
                   (1, 0, 1), 2, 1, False)
    closed = execute(p.begin, closed, p, False)
    closed = execute(p.steps[0], closed, p, False)
    closed = execute(p.end, closed, p, False)
    np.testing.assert_allclose(closed, expected, atol=3e-13, rtol=0)


def test_retained_coherent_control_does_not_leak_branch():
    p = plan(((1, 0, 1), ), controlled=True, b=1)
    original = initial(p, True)
    # Force exact |+> control while retaining entanglement in all other inputs.
    original[1::2] = original[::2]
    original /= np.linalg.norm(original)
    expected = xor(original, (1, 0, 1), 2, 1, True)
    # Exhaustive outcome histories for this smallest controlled epoch.
    stages = (p.begin, *p.steps, p.end)
    m = sum(s.resources.num_measurements for s in stages)
    rho01 = np.zeros((len(original) // 2, ) * 2, complex)
    for history in itertools.product((0, 1), repeat=m):
        state, offset = original, 0
        for s in stages:
            count = s.resources.num_measurements
            state = run_tape(s, state, 2, 1, True,
                             history[offset:offset + count])
            offset += count
        np.testing.assert_allclose(state * 2**(m / 2),
                                   expected,
                                   atol=3e-13,
                                   rtol=0)
        rho01 += state[::2] @ state[1::2].conj().T
    target = expected[::2] @ expected[1::2].conj().T
    assert np.linalg.norm(target) > .1
    np.testing.assert_allclose(rho01, target, atol=3e-13, rtol=0)


@pytest.mark.parametrize('kwargs', [
    dict(tables=()),
    dict(tables=((), )),
    dict(block=0),
    dict(block=1),
    dict(block=3),
    dict(block=4),
    dict(block=True),
    dict(controlled=1),
    dict(dirty=1),
    dict(tables=((4, ), ), b=2)
])
def test_retained_rejects_invalid_static_configuration(kwargs):
    with pytest.raises(ValueError):
        plan(**kwargs)


def test_retained_static_layout_and_endpoint_accounting():
    p = plan(((1, ), (3, 2, 1)), controlled=True)
    assert p.num_cache == p.num_output == 2
    assert len(p.steps) == 2
    assert p.resources.num_dirty == 2
    assert p.resources.num_clean == 3  # shared walk line + full-word cache
    stages = (p.begin, *p.steps, p.end)
    assert all(s.resources.num_clean == p.resources.num_clean for s in stages)
    assert all(s.resources.num_dirty == p.resources.num_dirty for s in stages)
    assert p.resources.num_measurements == sum(s.resources.num_measurements
                                               for s in stages)
    assert p.resources.toffoli_count == sum(s.resources.toffoli_count
                                            for s in stages)
    assert all(x < p.resources.num_clean for s in stages
               for op, x, _, _ in s.ops if op in (22, 23, 29))


@pytest.mark.parametrize('block,a,b', [(2, 2, 1), (2, 3, 2), (4, 3, 1)])
@pytest.mark.parametrize('controlled,dirty',
                         tuple(itertools.product((False, True), repeat=2)))
def test_retained_intermediate_bank_cache_invariant(block, a, b, controlled,
                                                    dirty):
    tables = (tuple(
        (3 * x + 1) % (1 << b) for x in range(5 if a == 3 else 3)), (1, ))
    p = plan(tables, a=a, b=b, block=block, controlled=controlled, dirty=dirty)
    original = initial(p, controlled)
    state = execute(p.begin, original, p, controlled, 1)
    cw = int(controlled)
    logical = cw + a + b + p.resources.num_dirty
    cache = max(block.bit_length() - 1, a - (block.bit_length() - 1))
    bank = cache + b
    bank_bit = cw + a + b if dirty else logical + bank

    def invariant(logical_state, table=None):
        result = np.zeros_like(logical_state)
        for source in range(1 << logical):
            c = source & 1 if controlled else 1
            x = source >> cw & ((1 << a) - 1)
            q, r = divmod(x, block)
            d = (source >>
                 (bank_bit +
                  (r - 1) * b)) & ((1 << b) - 1) if r and dirty else 0
            target = source ^ ((c * d) << (logical + cache))
            if table is not None:
                base = table[q * block] if q * block < len(table) else 0
                for i in range(1, block):
                    value = table[q * block +
                                  i] if q * block + i < len(table) else 0
                    target ^= (c * (value ^ base)) << (bank_bit + (i - 1) * b)
            result[target] = logical_state[source]
        return result

    np.testing.assert_allclose(state, invariant(original), atol=3e-13, rtol=0)
    expected = original
    for stage, table in zip(p.steps, tables):
        state = execute(stage, state, p, controlled, 1)
        expected = xor(expected, table, a, b, controlled)
        np.testing.assert_allclose(state,
                                   invariant(expected, table),
                                   atol=3e-13,
                                   rtol=0)
    state = execute(p.end, state, p, controlled, 1)
    np.testing.assert_allclose(state, expected, atol=3e-13, rtol=0)


def test_retained_all_logical_columns_every_measurement_history():
    p = plan(((1, 0, 1), ), controlled=True, b=1)
    logical = 1 + 2 + 1 + p.resources.num_dirty
    original = np.zeros((1 << (logical + p.resources.num_clean), 1 << logical),
                        complex)
    original[:1 << logical] = np.eye(1 << logical)
    expected = xor(original, (1, 0, 1), 2, 1, True)
    stages = (p.begin, *p.steps, p.end)
    m = sum(s.resources.num_measurements for s in stages)
    completeness = np.zeros((1 << logical, ) * 2, complex)
    for history in itertools.product((0, 1), repeat=m):
        state, offset = original, 0
        for s in stages:
            count = s.resources.num_measurements
            state = run_tape(s, state, 2, 1, True,
                             history[offset:offset + count])
            offset += count
        np.testing.assert_allclose(state,
                                   expected * 2**(-m / 2),
                                   atol=3e-13,
                                   rtol=0)
        completeness += state.conj().T @ state
    np.testing.assert_allclose(completeness,
                               np.eye(1 << logical),
                               atol=3e-13,
                               rtol=0)
