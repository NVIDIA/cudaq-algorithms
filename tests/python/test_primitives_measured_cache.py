# SPDX-License-Identifier: Apache-2.0
"""Native persistent-result and measured-cache regressions."""
import cudaq
import numpy as np
import pytest
from cudaq.qis import h, mz, x, z
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import MeasuredQROM
from cudaq_algorithms.primitives._unary_iteration import _mint_interpreter
from cudaq_algorithms.primitives._qrom_measured_plan import _plan_measured_qrom


def test_public_plan_binding():
    kwargs = dict(variant='select_swap',
                  block_size=2,
                  cache_uncompute='measure',
                  phase_block_size=2)
    data = [1, 0, 1, 1, 0]
    selected = _plan_measured_qrom(data, 4, 1, **kwargs)
    qrom = MeasuredQROM(iter(data), 4, 1, **kwargs)
    assert qrom.data == tuple(data)
    assert qrom.resources == selected.plan.resources
    assert qrom.cache_uncompute == 'measure'
    assert qrom.phase_block_size == 2
    assert 'cache_uncompute=' in repr(qrom)
    assert 'result[' in qrom.describe()


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('has_work', [False, True])
def test_high_result_slot_all_signatures(controlled, has_work):
    # Outcome in slot 70 must survive reset; low slot 0 has opposite value.
    kernel = _mint_interpreter([(24, 0, 0, 0), (24, 1, 70, 0), (26, 2, 0, 0)],
                               controlled,
                               has_work,
                               num_results=71,
                               parity_masks=((0, 70), ))
    from measured_harness import feedback_state_harness
    run = feedback_state_harness(kernel, controlled, has_work)
    for bit in [0, 1, 0, 1]:
        initial = np.zeros(128, complex)
        for i in range(8):
            initial[i << 2] = (-1)**(bit * ((i >> 1) & 1)) / np.sqrt(8)
        expected = np.zeros(128, complex)
        expected[0] = 2**-.5
        expected[16] = (-1)**bit * 2**-.5
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(initial))),
                                   expected,
                                   atol=1e-12)


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('phase_size', [1, 2])
def test_cache_complex_padding_reference_repeated(controlled, phase_size):
    data = [1, 0, 1, 1, 1]
    qrom = MeasuredQROM(data,
                        4,
                        1,
                        variant='select_swap',
                        block_size=2,
                        controlled=controlled,
                        cache_uncompute='measure',
                        phase_block_size=phase_size)
    lookup = qrom.kernel
    n = qrom.num_ladder
    c = int(controlled)
    end = c + 4
    width = end + 2  # output and untouched entangled reference
    if controlled:

        @cudaq.kernel
        def run(initial: cudaq.State, repeats: int):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(n)
            for _ in range(repeats):
                lookup(q[0:1], q[1:end], ladder, q[end:end + 1])
    else:

        @cudaq.kernel
        def run(initial: cudaq.State, repeats: int):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(n)
            for _ in range(repeats):
                lookup(q[0:end], ladder, q[end:end + 1])

    for seed in [31, 79]:
        rng = np.random.default_rng(seed)
        logical = rng.normal(
            size=1 << width) + 1j * rng.normal(size=1 << width)
        logical /= np.linalg.norm(logical)
        for repeats in [1, 2, 1]:
            expected = np.zeros(1 << (width + n), complex)
            for i, amplitude in enumerate(logical):
                address = (i >> c) & 15
                word = data[address] if address < len(data) and (not c or i
                                                                 & 1) else 0
                expected[i ^ ((word if repeats % 2 else 0) << end)] = amplitude
            cudaq.set_random_seed(seed)
            np.testing.assert_allclose(np.asarray(
                cudaq.get_state(run, state_from(logical), repeats)),
                                       expected,
                                       atol=1e-12,
                                       rtol=0)


def test_compile_cache_above_64_bits():
    qrom = MeasuredQROM([1 << 34, 1],
                        3,
                        35,
                        variant='select_swap',
                        block_size=2,
                        cache_uncompute='measure',
                        phase_block_size=1)
    assert qrom.resources.classical_bits == 70
    qrom.kernel.compile()


@pytest.mark.parametrize('kwargs', [
    {
        'cache_uncompute': 'unknown'
    },
    {
        'cache_uncompute': 'measure'
    },
    {
        'variant': 'select',
        'cache_uncompute': 'measure'
    },
    {
        'variant': 'select_swap',
        'dirty_banks': True,
        'cache_uncompute': 'measure'
    },
    {
        'variant': 'select_swap',
        'phase_block_size': 1
    },
    {
        'variant': 'select_swap',
        'cache_uncompute': 'measure',
        'phase_block_size': 3
    },
])
def test_public_cache_parameter_errors(kwargs):
    with pytest.raises(ValueError):
        MeasuredQROM([1, 0], 3, 1, **kwargs)


@pytest.mark.parametrize('ops,results,masks', [
    ([(24, 0, 1, 0)], 1, ()),
    ([(26, 0, 0, 0)], 1, ((0, ), )),
    ([(24, 0, 0, 0), (26, 1, 0, 0)], 1, ((0, 0), )),
    ([(24, 0, 0, 1)], 1, ()),
])
def test_feedback_tape_validation(ops, results, masks):
    with pytest.raises(ValueError):
        _mint_interpreter(ops,
                          False,
                          False,
                          num_results=results,
                          parity_masks=masks)


@pytest.mark.parametrize('vector_measurement', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_documented_persistent_vector_probe(vector_measurement, nested):
    # The selected production form is the local bool vector. The alternative
    # direct vector measurement is retained as the bounded SDK capability probe.
    if vector_measurement:

        @cudaq.kernel
        def probe(bit: int):
            q = cudaq.qvector(5)
            if bit == 1:
                x(q[2])
            h(q[3])
            h(q[4])
            outcomes = mz(q[0:3])
            if outcomes[0]:
                x(q[0])
            if outcomes[2]:
                x(q[2])
            parity = outcomes[0] != outcomes[2]
            if parity:
                z(q[3])
                z.ctrl(q[4], q[3])
    else:

        @cudaq.kernel
        def probe(bit: int):
            q = cudaq.qvector(5)
            if bit == 1:
                x(q[2])
            h(q[3])
            h(q[4])
            outcomes = [False for _ in range(3)]
            outcomes[0] = mz(q[0])
            if outcomes[0]:
                x(q[0])
            outcomes[2] = mz(q[2])
            if outcomes[2]:
                x(q[2])
            parity = outcomes[0] != outcomes[2]
            if parity:
                z(q[3])
                z.ctrl(q[4], q[3])

    @cudaq.kernel
    def wrapper(bit: int):
        probe(bit)

    run = wrapper if nested else probe
    for seed, bit in [(7, 0), (13, 1), (23, 0), (31, 1)]:
        cudaq.set_random_seed(seed)
        expected = np.zeros(32, complex)
        expected[[0, 8, 16, 24]] = [.5, (-1)**bit * .5, .5, .5]
        np.testing.assert_allclose(np.asarray(cudaq.get_state(run, bit)),
                                   expected,
                                   atol=1e-12)


@pytest.mark.parametrize('controlled,has_work', [(False, False), (False, True),
                                                 (True, False), (True, True)])
def test_output_erasure_and_parity_cz_all_signatures(controlled, has_work):
    from measured_harness import feedback_state_harness
    kernel = _mint_interpreter([(24, 1, 0, 0), (25, 0, 70, 0), (27, 0, 2, 0)],
                               controlled,
                               has_work,
                               num_results=71,
                               parity_masks=((0, 70), ))
    run = feedback_state_harness(kernel, controlled, has_work)
    # Both phase wires in |+>, measured wires in X eigenstates; reference-like
    # control/work wires in arbitrary complex states remain untouched.
    rng = np.random.default_rng(82)
    spectator = rng.normal(size=8) + 1j * rng.normal(size=8)
    spectator /= np.linalg.norm(spectator)
    for bit in [1, 0, 1]:
        initial = np.zeros(128, complex)
        expected = np.zeros(128, complex)
        for i in range(128):
            control, address, work = i & 1, (i >> 1) & 1, (i >> 6) & 1
            amp = spectator[control + 2 * address + 4 * work]
            initial[i] = amp * (-1)**(bit * ((i >> 5) & 1)) / 4
            if not (i & ((1 << 3) | (1 << 5))):
                expected[i] = amp * (-1)**(bit * ((i >> 2) & 1) *
                                           ((i >> 4) & 1)) / 2
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(initial))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)
