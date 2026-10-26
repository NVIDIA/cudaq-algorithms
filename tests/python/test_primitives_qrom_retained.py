# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native retained epochs against independent full complex vector oracles."""
import numpy as np
import pytest
import cudaq
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import RetainedQROMChain


def _initial(chain, seed=512, inactive=False):
    a, clean, dirty, b = (chain.num_address, chain.num_clean, chain.num_dirty,
                          chain.num_output)
    output = a + chain.num_ladder
    control = output + b
    reference = control + int(chain.controlled)
    width = reference + 1
    rng = np.random.default_rng(seed)
    state = np.zeros(1 << width, complex)
    for index in range(len(state)):
        if (index >> a) & ((1 << clean) - 1):
            continue
        if inactive and (index >> control) & 1:
            continue
        state[index] = rng.normal() + 1j * rng.normal()
    return state / np.linalg.norm(state), output, control


def _xor(state, table, a, output, control=None):
    result = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        address = index & ((1 << a) - 1)
        word = table[address] if address < len(table) else 0
        if control is not None and not (index >> control) & 1:
            word = 0
        result[index ^ (word << output)] = amplitude
    return result


def _ry(state, bit, theta):
    result = state.copy()
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    for index in range(len(state)):
        if not (index >> bit) & 1:
            other = index | (1 << bit)
            result[index] = c * state[index] - s * state[other]
            result[other] = s * state[index] + c * state[other]
    return result


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('dirty_banks', [False, True])
def test_retained_chain_rotations_match_dense_reference(
        controlled, dirty_banks):
    tables = ((1, 0, 1), (0, 1, 1, 0))
    chain = RetainedQROMChain(tables,
                              2,
                              1,
                              block_size=2,
                              controlled=controlled,
                              dirty_banks=dirty_banks)
    initial, out, ctrl = _initial(chain)
    begin, end = chain.begin_kernel, chain.end_kernel
    first, middle, last = chain.kernels()
    stop = out + 1
    if controlled:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            begin(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
            first(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
            ry(0.47, q[out])
            middle(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
            rz(-0.31, q[out])
            last(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
            end(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
    else:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            begin(q[0:2], q[2:out], q[out:stop])
            first(q[0:2], q[2:out], q[out:stop])
            ry(0.47, q[out])
            middle(q[0:2], q[2:out], q[out:stop])
            rz(-0.31, q[out])
            last(q[0:2], q[2:out], q[out:stop])
            end(q[0:2], q[2:out], q[out:stop])

    control = ctrl if controlled else None
    expected = _xor(initial, tables[0], 2, out, control)
    expected = _ry(expected, out, 0.47)
    expected = _xor(expected, (1, 1, 0, 0), 2, out, control)
    indices = np.arange(len(expected))
    expected *= np.exp(-0.5j * -0.31 * (1 - 2 * ((indices >> out) & 1)))
    expected = _xor(expected, tables[1], 2, out, control)
    # Host branch-isometry proofs justify a vector oracle for each trajectory.
    for _ in range(3):
        actual = np.asarray(cudaq.get_state(run, state_from(initial)))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert len(chain.kernels()) == chain.num_tables + 1
    assert chain.num_cache == 1
    assert chain.resources.num_ladder == chain.num_clean + chain.num_dirty


def test_retained_controlled_inactive_branch():
    chain = RetainedQROMChain(((1, 0, 1), ),
                              2,
                              1,
                              block_size=2,
                              controlled=True,
                              dirty_banks=True)
    initial, out, ctrl = _initial(chain, inactive=True)
    begin, end = chain.begin_kernel, chain.end_kernel
    first, last = chain.kernels()
    stop = out + 1

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        begin(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
        first(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
        last(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])
        end(q[ctrl:ctrl + 1], q[0:2], q[2:out], q[out:stop])

    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(run, state_from(initial))),
                               initial,
                               atol=1e-12,
                               rtol=0)
