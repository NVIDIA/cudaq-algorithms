"""Independent channel and ownership checks of measured SELECT-COPY tapes."""
import itertools
import numpy as np
import pytest
from qrom_host_loader import load_qrom_host_module
from qrom_channel_reference import apply_gate, measure_reset


def make_plan(a=2,
              data=(1, 3, 2),
              b=2,
              block=2,
              alpha=2,
              controlled=False,
              dirty=True):
    module = load_qrom_host_module('_select_copy_plan')
    return module.plan_select_copy(data,
                                   a,
                                   b,
                                   block_size=block,
                                   packets=module.packet_layout(b,
                                                                alpha=alpha),
                                   controlled=controlled,
                                   dirty_banks=dirty)


def run_tape(plan, state, a, b, controlled, history):
    """Translate operand classes only; use independent dense gate primitives."""
    cw = int(controlled)
    logical = cw + a + b + plan.resources.num_dirty
    address = lambda x: cw + x
    output = lambda x: cw + a + x

    def ladder(x):
        if x < plan.resources.num_clean:
            return logical + x
        return cw + a + b + x - plan.resources.num_clean

    outcomes = iter(history)
    for op, x, y, z in plan.ops:
        if op in (22, 23, 29):
            bit = ladder(x)
            state = apply_gate(state, 'h', bit)
            outcome = next(outcomes)
            state = measure_reset(state, bit, outcome)
            if outcome:
                controls = ((ladder(y), address(z)) if op == 22 else
                            (0, address(z)) if op == 23 else
                            (ladder(y), ladder(z)))
                state = apply_gate(state, 'cz', *controls)
            continue
        gate, bits = {
            0: ('x', (address(x), )),
            1: ('x', (ladder(x), )),
            2: ('cx', (address(x), ladder(y))),
            3: ('cx', (ladder(x), ladder(y))),
            4: ('ccx', (ladder(x), address(y), ladder(z))),
            5: ('cx', (ladder(x), output(y))),
            8: ('cx', (0, ladder(y))),
            9: ('ccx', (0, address(y), ladder(z))),
            21: ('ccx', (ladder(x), ladder(y), output(z))),
            28: ('ccx', (ladder(x), ladder(y), ladder(z))),
        }[op]
        state = apply_gate(state, gate, *bits)
    return state


@pytest.mark.parametrize('a,data,b,block,alpha',
                         [(2, (1, 3, 2), 2, 2, 2),
                          (3, (3, 1, 2, 0, 1), 2, 4, 1)])
@pytest.mark.parametrize('dirty,controlled',
                         list(itertools.product((False, True), repeat=2)))
def test_select_copy_plan_every_history_matches_xor(a, data, b, block, alpha,
                                                    dirty, controlled):
    plan = make_plan(a, data, b, block, alpha, controlled, dirty)
    logical = int(controlled) + a + b + plan.resources.num_dirty
    dim = 1 << logical
    rng = np.random.default_rng(419)
    # Two columns form a joint complex state with an explicit reference qubit.
    psi = rng.normal(size=(dim, 2)) + 1j * rng.normal(size=(dim, 2))
    psi /= np.linalg.norm(psi)
    if logical <= 6:
        psi = np.eye(
            dim, dtype=complex)  # all input columns, one common branch scalar
    state = np.zeros((dim << plan.resources.num_clean, psi.shape[1]), complex)
    state[:dim] = psi
    expected = np.zeros_like(state)
    for source in range(dim):
        c = source & 1 if controlled else 1
        x = (source >> int(controlled)) & ((1 << a) - 1)
        value = data[x] if x < len(data) else 0
        expected[source ^ ((c * value) << (int(controlled) + a))] = psi[source]
    m = plan.resources.num_measurements
    histories = list(itertools.product((0, 1), repeat=m)) if m <= 8 else [
        (0, ) * m,
        (1, ) * m, *[tuple(rng.integers(0, 2, m)) for _ in range(4)]
    ]
    for history in histories:
        actual = run_tape(plan, state, a, b, controlled, history)
        np.testing.assert_allclose(actual * 2**(m / 2),
                                   expected,
                                   atol=2e-13,
                                   rtol=0)


def test_restore_never_measures_borrowed_lines():
    plan = make_plan(a=3,
                     data=(1, 3, 2, 0, 1),
                     block=4,
                     alpha=1,
                     controlled=True)
    assert all(x < plan.resources.num_clean for op, x, _, _ in plan.ops
               if op in (22, 23, 29))
    assert plan.resources.num_measurements == sum(op in (22, 23, 29)
                                                  for op, *_ in plan.ops)


def test_restore_fanout_uses_one_and_per_bank_position():
    module = load_qrom_host_module('_select_copy_plan')
    layout = module.packet_layout(5, packet_width=2)
    plan = module.plan_select_copy((3, 17, 29, 4, 22),
                                   3,
                                   5,
                                   block_size=4,
                                   packets=layout,
                                   controlled=True,
                                   dirty_banks=True)
    restore = next(s for s in plan.resources.stages if s.name == 'restore')
    assert sum(op == 28 for op, *_ in plan.ops) == 3 * 2
    assert restore.temporary_and_count - restore.tree_and_count == 3 * 2
    assert sum(op == 29 for op, *_ in plan.ops) == 3 * 2


def test_restore_scratch_peak_is_in_budget():
    plan = make_plan(a=3,
                     data=(1, 2, 3, 0, 1),
                     block=4,
                     alpha=1,
                     controlled=True)
    scratch = {z for op, _, _, z in plan.ops if op == 28}
    assert len(scratch) == 1
    assert scratch == {plan.resources.num_clean - 1}
    assert plan.resources.num_clean == 3  # max(high, low) + one Restore scratch
    assert plan.resources.num_dirty == 6


def test_invalid_plan_geometry_and_packet_metadata():
    module = load_qrom_host_module('_select_copy_plan')
    for block in (True, 0, 1, 3, 4):
        with pytest.raises(ValueError):
            make_plan(block=block)
    malformed = module.PacketLayout((1, 1), (0, 1), 2, ((0, 1), ))
    with pytest.raises(ValueError):
        module.plan_select_copy((1, 2),
                                2,
                                2,
                                block_size=2,
                                packets=malformed,
                                controlled=False,
                                dirty_banks=True)


def test_nonuniform_packet_plan_joint_state():
    test_select_copy_plan_every_history_matches_xor(3, (3, 17, 29, 4, 22), 5,
                                                    2, 3, True, True)
