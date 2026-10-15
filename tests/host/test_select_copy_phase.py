"""Independent full-isometry checks of parity SELECT-COPY and image erasure."""
import itertools
import numpy as np
import pytest
from qrom_host_loader import load_qrom_host_module
from qrom_channel_reference import apply_gate, measure_reset


def phase_plan(rows, a=2, controlled=False, block=2, **offsets):
    return load_qrom_host_module('_select_copy_plan')._plan_select_copy_phase(
        rows,
        a,
        controlled=controlled,
        block_size=block,
        bank_offset=offsets.get(
            'bank_offset',
            max(a - (block.bit_length() - 1),
                block.bit_length() - 1)),
        tree_offset=offsets.get('tree_offset', 0),
        address_offset=offsets.get('address_offset', 0))


def run(plan, state, a, b, controlled, outcomes, history):
    """Independent gate translation; outcomes are invocation-local, never in plan."""
    cw = int(controlled)
    logical = cw + a + b + plan.resources.num_dirty
    address = lambda j: cw + j
    output = lambda j: cw + a + j
    ladder = lambda j: (logical + j if j < plan.resources.num_clean else cw + a
                        + b + j - plan.resources.num_clean)
    results = list(outcomes)
    history = iter(history)
    for op, x, y, z in plan.ops:
        if op in (22, 23, 29, 24, 25):
            wire = output(x) if op == 25 else ladder(x)
            s = next(history)
            state = measure_reset(apply_gate(state, 'h', wire), wire, s)
            if op in (24, 25):
                results[y] = s
            elif s:
                pair = ((ladder(y), address(z)) if op == 22 else
                        (0, address(z)) if op == 23 else
                        (ladder(y), ladder(z)))
                state = apply_gate(state, 'cz', *pair)
        elif op in (26, 27, 30):
            mask = plan.parity_masks[y if op == 26 else z]
            if sum(results[j] for j in mask) % 2:
                state = apply_gate(
                    state, 'z' if op == 26 else 'cx' if op == 30 else 'cz',
                    *((ladder(x), ) if op == 26 else (ladder(x), ladder(y))))
        else:
            gate, bits = {
                0: ('x', (address(x), )),
                1: ('x', (ladder(x), )),
                2: ('cx', (address(x), ladder(y))),
                3: ('cx', (ladder(x), ladder(y))),
                4: ('ccx', (ladder(x), address(y), ladder(z))),
                8: ('cx', (0, ladder(y))),
                9: ('ccx', (0, address(y), ladder(z))),
                31: ('cz', (ladder(x), ladder(y))),
            }[op]
            state = apply_gate(state, gate, *bits)
    with pytest.raises(StopIteration):
        next(history)
    return state


def embedding(plan, a, controlled, b=0, words=None):
    cw = int(controlled)
    logical = cw + a + plan.resources.num_dirty
    dim = 1 << logical
    state = np.zeros((dim << (b + plan.resources.num_clean), dim), complex)
    erased = np.zeros_like(state)
    for j in range(dim):
        c = j & 1 if controlled else 1
        x = (j >> cw) & ((1 << a) - 1)
        low = j & ((1 << (cw + a)) - 1)
        dirty = j >> (cw + a)
        dest = low | (dirty << (cw + a + b))
        erased[dest, j] = 1
        value = 0 if words is None or x >= len(words) else c * words[x]
        state[dest | (value << (cw + a)), j] = 1
    return state, erased


@pytest.mark.parametrize('controlled', [False, True])
def test_select_copy_phase_all_outcomes_and_dirty_states(controlled):
    words = (1, 3, 2)
    rows = tuple(tuple(t for t in range(2) if w >> t & 1) for w in words)
    plan = phase_plan(rows, controlled=controlled)
    state, _ = embedding(plan, 2, controlled)
    m = plan.resources.num_measurements
    for outcomes in itertools.product((0, 1), repeat=2):
        expected = state.copy()
        for j in range(state.shape[1]):
            c = j & 1 if controlled else 1
            x = (j >> int(controlled)) & 3
            w = words[x] if x < len(words) else 0
            expected[:, j] *= (-1)**(c * sum(
                ((w >> t) & 1) * outcomes[t] for t in range(2)))
        for history in itertools.product((0, 1), repeat=m):
            actual = run(plan, state, 2, 0, controlled, outcomes, history)
            np.testing.assert_allclose(actual,
                                       expected * 2**(-m / 2),
                                       atol=2e-14,
                                       rtol=0)
            # Full isometry also preserves arbitrary complex reference correlations.
            rng = np.random.default_rng(81)
            ref = rng.normal(size=(
                state.shape[1], 2)) + 1j * rng.normal(size=(state.shape[1], 2))
            ref /= np.linalg.norm(ref)
            np.testing.assert_allclose(actual @ ref,
                                       expected @ ref * 2**(-m / 2),
                                       atol=2e-14,
                                       rtol=0)


def test_select_copy_phase_empty_repeated_and_cancelled_masks():
    rows = ((0, 1, 0), (1, ), (2, 2))
    plan = phase_plan(rows)
    assert plan == phase_plan(rows)
    assert len(plan.parity_masks) == len(set(plan.parity_masks))
    assert all(len(mask) == len(set(mask)) for mask in plan.parity_masks)
    state, _ = embedding(plan, 2, False)
    for outcomes in itertools.product((0, 1), repeat=plan.num_results):
        expected = state.copy()
        for j in range(state.shape[1]):
            x = j & 3
            expected[:,
                     j] *= (-1)**sum(outcomes[t] for t in rows[x]
                                     if x < len(rows)) if x < len(rows) else 1
        actual = run(plan, state, 2, 0, False, outcomes,
                     (0, ) * plan.resources.num_measurements)
        np.testing.assert_allclose(actual,
                                   expected *
                                   2**(-plan.resources.num_measurements / 2),
                                   atol=2e-14,
                                   rtol=0)
    assert not any(op in (26, 30) for op, *_ in phase_plan(((), (), ())).ops)


def erase_plan(words, controlled):
    cache = load_qrom_host_module('_qrom_cache_erasure')
    records = load_qrom_host_module('_qrom_plan')
    prefix = cache._plan_cache_measurement(2,
                                           cache_offset=0,
                                           cache_register='output')
    rows = tuple(tuple(t for t in range(2) if w >> t & 1) for w in words)
    phase = phase_plan(rows, controlled=controlled)
    resources = records.QROMResources(
        prefix.resources.stages + phase.resources.stages,
        phase.resources.num_clean, phase.resources.num_dirty, 2)
    return records.QROMPlan(prefix.ops + phase.ops, phase.parity_masks, 2,
                            resources)


@pytest.mark.parametrize('controlled', [False, True])
def test_erasure_image_channel_select_copy_phase(controlled):
    words = (1, 3, 2)
    plan = erase_plan(words, controlled)
    state, expected = embedding(plan, 2, controlled, 2, words)
    m = plan.resources.num_measurements
    probabilities = np.zeros(state.shape[1])
    for history in itertools.product((0, 1), repeat=m):
        actual = run(plan, state, 2, 2, controlled, (0, 0), history)
        np.testing.assert_allclose(actual,
                                   expected * 2**(-m / 2),
                                   atol=2e-14,
                                   rtol=0)
        probabilities += np.sum(abs(actual)**2, axis=0)
    np.testing.assert_allclose(probabilities, 1, atol=2e-14)


def test_arbitrary_output_is_not_an_erasure_inverse():
    plan = erase_plan((1, 3, 2), True)
    state, expected = embedding(plan, 2, True,
                                2)  # wrong active-branch outputs
    m = plan.resources.num_measurements
    first = run(plan, state, 2, 2, True, (0, 0), (0, ) * m)
    second = run(plan, state, 2, 2, True, (0, 0), (1, ) + (0, ) * (m - 1))
    np.testing.assert_allclose(first, expected * 2**(-m / 2), atol=2e-14)
    assert not np.allclose(second, first)
    # No single branch scalar can remove the relative phase across columns.
    assert np.vdot(first, second) != pytest.approx(np.vdot(first, first))


def test_output_erasure_reuses_shared_planner():
    plan = erase_plan((1, 3, 2), True)
    assert plan.ops[:2] == ((25, 0, 0, 0), (25, 1, 1, 0))
    assert (plan.resources.num_clean, plan.resources.num_dirty) == (1, 1)
    assert plan.resources.classical_bits == 2
    ir = load_qrom_host_module('_unary_iteration_ir')
    ir.validate_feedback_ops(plan.ops,
                             parity_masks=plan.parity_masks,
                             num_results=2,
                             num_address=2,
                             num_ladder=2,
                             num_output=2,
                             controlled=True)


@pytest.mark.parametrize('code,operands', [(21, (0, 1, 0)), (28, (0, 1, 2)),
                                           (29, (2, 0, 1)), (30, (0, 1, 0)),
                                           (31, (0, 1, 0))])
def test_shared_schema_select_copy_codes(code, operands):
    ir = load_qrom_host_module('_unary_iteration_ir')
    kwargs = dict(parity_masks=((0, ), ),
                  num_results=1,
                  num_address=1,
                  num_ladder=3,
                  num_output=1,
                  controlled=False,
                  initialized_results=(0, ))
    ir.validate_feedback_ops(((code, *operands), ), **kwargs)
    with pytest.raises(ValueError):
        ir.validate_feedback_ops(((code, 3, operands[1], operands[2]), ),
                                 **kwargs)
    if code in (28, 29, 30, 31):
        with pytest.raises(ValueError):
            ir.validate_feedback_ops(((code, 0, 0, operands[2]), ), **kwargs)


@pytest.mark.parametrize('a,block,controlled', [(3, 4, False), (3, 2, True)])
def test_deeper_phase_walks_all_branches(a, block, controlled):
    words = (1, 0, 1, 1, 1)
    plan = phase_plan(tuple((0, ) if w else () for w in words), a, controlled,
                      block)
    state, _ = embedding(plan, a, controlled)
    m = plan.resources.num_measurements
    for outcome in (0, 1):
        expected = state.copy()
        for j in range(state.shape[1]):
            c = j & 1 if controlled else 1
            x = (j >> int(controlled)) & ((1 << a) - 1)
            expected[:, j] *= (-1)**(c * outcome *
                                     (words[x] if x < len(words) else 0))
        for history in itertools.product((0, 1), repeat=m):
            actual = run(plan, state, a, 0, controlled, (outcome, ), history)
            np.testing.assert_allclose(actual,
                                       expected * 2**(-m / 2),
                                       atol=2e-14,
                                       rtol=0)


def test_phase_offsets_and_result_slots_above_word_size():
    plan = phase_plan(((65, ), (0, 65), (0, )),
                      controlled=True,
                      address_offset=1,
                      tree_offset=1,
                      bank_offset=2)
    state, _ = embedding(plan, 3, True)
    m = plan.resources.num_measurements
    for s0, s65 in itertools.product((0, 1), repeat=2):
        outcomes = [0] * 66
        outcomes[0], outcomes[65] = s0, s65
        expected = state.copy()
        for j in range(state.shape[1]):
            x = (j >> 2) & 3
            phase = (s65, s0 ^ s65, s0, 0)[x]
            expected[:, j] *= (-1)**((j & 1) * phase)
        for history in itertools.product((0, 1), repeat=m):
            actual = run(plan, state, 3, 0, True, outcomes, history)
            np.testing.assert_allclose(actual,
                                       expected * 2**(-m / 2),
                                       atol=2e-14,
                                       rtol=0)


@pytest.mark.parametrize('kwargs', [
    dict(block_size=1),
    dict(block_size=3),
    dict(block_size=4),
    dict(block_size=True),
    dict(address_bits=True),
    dict(bank_offset=0),
    dict(tree_offset=-1),
    dict(address_offset=True),
    dict(controlled=1)
])
def test_phase_rejects_invalid_geometry(kwargs):
    options = dict(address_bits=2,
                   controlled=False,
                   block_size=2,
                   bank_offset=1,
                   tree_offset=0)
    options.update(kwargs)
    with pytest.raises(ValueError):
        load_qrom_host_module('_select_copy_plan')._plan_select_copy_phase(
            ((0, ), ), **options)


@pytest.mark.parametrize('rows', [(), ((), ) * 5, ((-1, ), ), ((True, ), ),
                                  ((1.5, ), )])
def test_phase_rejects_invalid_masks(rows):
    with pytest.raises(ValueError):
        phase_plan(rows)


def test_parity_cx_schema_rejects_unbound_results_and_bad_mask():
    ir = load_qrom_host_module('_unary_iteration_ir')
    kwargs = dict(parity_masks=((0, ), ),
                  num_results=1,
                  num_address=1,
                  num_ladder=3,
                  num_output=0,
                  controlled=False)
    with pytest.raises(ValueError, match='initialized'):
        ir.validate_feedback_ops(((30, 0, 1, 0), ), **kwargs)
    with pytest.raises(ValueError, match='mask'):
        ir.validate_feedback_ops(((30, 0, 1, 1), ),
                                 initialized_results=(0, ),
                                 **kwargs)
    with pytest.raises(ValueError, match='reserved'):
        ir.validate_feedback_ops(((31, 0, 1, 1), ), **kwargs)
