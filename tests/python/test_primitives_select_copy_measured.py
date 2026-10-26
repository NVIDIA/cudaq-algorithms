# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SELECT-COPY channel and independently allocated image erasure."""
import numpy as np
import pytest
import cudaq
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import MeasuredQROM
from test_primitives_qrom_retained import _initial, _xor


@pytest.mark.parametrize('controlled,dirty_banks,packets', [
    (False, False, {
        'alpha': 1
    }),
    (True, False, {
        'alpha': 2
    }),
    (False, True, {
        'packet_width': 2
    }),
    (True, True, {
        'alpha': 2
    }),
])
def test_measured_select_copy_matches_host_channel(controlled, dirty_banks,
                                                   packets):
    data = (7, 0, 5)
    qrom = MeasuredQROM(data,
                        2,
                        3,
                        variant='select_copy',
                        block_size=2,
                        controlled=controlled,
                        dirty_banks=dirty_banks,
                        **packets)
    initial, out, ctrl = _initial(qrom)
    stop = out + 3
    lookup = qrom.kernel
    if controlled:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            lookup(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
    else:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            lookup(q[0:2], q[2:out], q[out:stop])

    expected = _xor(initial, data, 2, out, ctrl if controlled else None)
    # Each branch is the same complex isometry, proved independently on host.
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(initial))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


@pytest.mark.parametrize('variant,dirty_banks', [('select', False),
                                                 ('select_swap', False),
                                                 ('select_copy', False),
                                                 ('select_copy', True)])
@pytest.mark.parametrize('controlled', [False, True])
def test_public_erase_kernel_matches_image_channel(variant, dirty_banks,
                                                   controlled):
    data = (3, 0, 1)
    options = {} if variant == 'select' else {'block_size': 2}
    qrom = MeasuredQROM(data,
                        2,
                        2,
                        variant=variant,
                        controlled=controlled,
                        dirty_banks=dirty_banks,
                        **options)
    # Erase has its own clean-prefix/borrowed-suffix allocation, independent
    # of load. Embed the image directly, avoiding a shared-layout assumption.
    a, clean, dirty = 2, qrom.num_erase_clean, qrom.num_erase_dirty
    out = a + qrom.num_erase_ladder
    stop = out + 2
    ctrl = stop
    width = stop + int(controlled) + 1
    initial = np.zeros(1 << width, complex)
    expected = np.zeros_like(initial)
    rng = np.random.default_rng(671)
    for index in range(len(initial)):
        if (index >> a) & ((1 << clean) - 1) or (index >> out) & 3:
            continue
        address = index & 3
        word = data[address] if address < len(data) else 0
        if controlled and not (index >> ctrl) & 1:
            word = 0
        amplitude = rng.normal() + 1j * rng.normal()
        initial[index | (word << out)] = amplitude
        expected[index] = amplitude
    norm = np.linalg.norm(initial)
    initial /= norm
    expected /= norm
    erase = qrom.erase_kernel
    if controlled:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            erase(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
    else:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            erase(q[0:2], q[2:out], q[out:stop])

    for _ in range(3):
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(initial))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)
    assert qrom.erase_resources.num_clean == clean
    assert qrom.erase_resources.num_dirty == dirty
    assert qrom.num_erase_ladder == clean + dirty
    assert qrom.erase_resources.classical_bits == 2


def test_native_restore_disposal_probe():
    """Fresh scratch disposal preserves complex controls, output and reference."""
    from cudaq_algorithms.primitives._unary_iteration import _mint_interpreter
    restore = _mint_interpreter(((28, 0, 1, 2), (5, 2, 0, 0), (29, 2, 0, 1)),
                                False, False)
    # address q0 is unused; ladder q1,q2 controls, q3 scratch; output q4;
    # q5 is a reference. Every logical column has an independent amplitude.
    rng = np.random.default_rng(191)
    initial = np.zeros(64, complex)
    for index in range(64):
        if not index & 8:
            initial[index] = rng.normal() + 1j * rng.normal()
    initial /= np.linalg.norm(initial)
    expected = np.zeros_like(initial)
    for index, amplitude in enumerate(initial):
        target = index ^ (16 if index & 2 and index & 4 else 0)
        expected[target] = amplitude

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        restore(q[0:1], q[1:4], q[4:5])

    for _ in range(4):
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(initial))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


def test_native_parity_feedback_repeated_invocation_probe():
    """The same compiled kernel reads fresh result slots on every invocation."""
    from cudaq_algorithms.primitives._unary_iteration import _mint_interpreter
    parity = _mint_interpreter(((25, 0, 0, 0), (30, 0, 1, 0)),
                               False,
                               False,
                               num_results=1,
                               parity_masks=((0, ), ))

    @cudaq.kernel
    def run(outcome: int):
        address = cudaq.qvector(1)
        ladder = cudaq.qvector(2)
        output = cudaq.qvector(1)
        x(ladder[0])
        if outcome == 1:
            x(output[0])
        h(output[0])
        parity(address, ladder, output)

    for outcome in (0, 1, 0, 1, 1, 0):
        expected = np.zeros(16, complex)
        expected[2 | (outcome << 2)] = 1
        np.testing.assert_allclose(np.asarray(cudaq.get_state(run, outcome)),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


@pytest.mark.parametrize('controlled,dirty,alpha,entries', [
    (False, False, 1, 8),
    (True, False, 2, 5),
    (False, True, 2, 5),
    (True, True, 3, 8),
])
def test_adaptive_compiled_resources_match_stage_ledger(
        controlled, dirty, alpha, entries):
    from measured_harness import resources_of
    qrom = MeasuredQROM(tuple((3 * j + 1) % 8 for j in range(entries)),
                        3,
                        3,
                        variant='select_copy',
                        block_size=2,
                        alpha=alpha,
                        controlled=controlled,
                        dirty_banks=dirty)
    widths = ([1] if controlled else []) + [3, qrom.num_ladder, 3]
    compiled = resources_of(qrom.kernel, *widths)
    ledger = qrom.resources
    assert compiled.count_controls('x', 2) == ledger.toffoli_count
    assert compiled.count('mz') == ledger.num_measurements
    possible_fixups = sum(op[0] in (22, 23, 29) for op in qrom.plan.ops)
    assert possible_fixups == ledger.num_measurements
    # estimate_resources traces measurement outcomes. Its conditional gates
    # count the observed branch, not every correction that might fire.
    observed_fixups = compiled.count_controls('z', 1)
    assert 0 <= observed_fixups <= possible_fixups
    assert ledger.toffoli_count == sum(stage.toffoli_count
                                       for stage in ledger.stages)
    assert ledger.temporary_and_count >= ledger.tree_and_count
    assert qrom.describe().count('# Toffoli') == ledger.toffoli_count
    # Compiled Toffolis above are actual logical instructions. No T synthesis
    # is requested or inferred from the temporary-AND metadata.


def test_retained_compiled_resources_include_endpoints():
    from measured_harness import resources_of
    from cudaq_algorithms.primitives import RetainedQROMChain
    chain = RetainedQROMChain(((1, 0, 1), (0, 1, 1, 0)),
                              2,
                              1,
                              controlled=True,
                              block_size=2)
    kernels = (chain.begin_kernel, *chain.kernels(), chain.end_kernel)
    plans = (chain.plan.begin, *chain.plan.steps, chain.plan.end)
    totals = [0, 0]
    for kernel, plan in zip(kernels, plans):
        compiled = resources_of(kernel, 1, 2, chain.num_ladder, 1)
        assert compiled.count_controls('x', 2) == plan.resources.toffoli_count
        assert compiled.count('mz') == plan.resources.num_measurements
        totals[0] += compiled.count_controls('x', 2)
        totals[1] += compiled.count('mz')
    assert totals == [
        chain.resources.toffoli_count, chain.resources.num_measurements
    ]


@pytest.mark.parametrize('controlled,has_work', [(False, False), (False, True),
                                                 (True, False), (True, True)])
def test_select_copy_new_arms_all_interpreter_signatures(controlled, has_work):
    from cudaq_algorithms.primitives._unary_iteration import _mint_interpreter
    from measured_harness import feedback_state_harness
    # Input parity source is |->, so the parity CX fires deterministically.
    # Ladder 0 is set, ladder 1 starts zero, scratch starts zero.
    # This exercises every new arm in all four shared signatures.
    tape = ((25, 0, 0, 0), (30, 0, 1, 0), (28, 0, 1, 2), (21, 0, 1, 0),
            (29, 2, 0, 1), (31, 0, 1, 0))
    kernel = _mint_interpreter(tape,
                               controlled,
                               has_work,
                               num_results=1,
                               parity_masks=((0, ), ))
    run = feedback_state_harness(kernel, controlled, has_work)
    initial = np.zeros(128, complex)
    initial[4] = 1 / np.sqrt(2)
    initial[4 | 32] = -1 / np.sqrt(2)
    expected = np.zeros_like(initial)
    expected[4 | 8 | 32] = -1
    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(run, state_from(initial))),
                               expected,
                               atol=1e-12,
                               rtol=0)


def test_native_disposal_deterministic_outcomes():
    """Both disposal branches apply the required phase and reset scratch."""
    from cudaq_algorithms.primitives._unary_iteration import _mint_interpreter
    dispose = _mint_interpreter(((29, 2, 0, 1), ), False, False)

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        dispose(q[0:1], q[1:4], q[4:5])

    for outcome in (0, 1, 0, 1):
        initial = np.zeros(32, complex)
        initial[6] = 1 / np.sqrt(2)
        initial[14] = (-1)**outcome / np.sqrt(2)
        expected = np.zeros_like(initial)
        expected[6] = (-1)**outcome
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(initial))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)
