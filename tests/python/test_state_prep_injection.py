# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""State-preparation injection into the primitive kernel factories.

Every factory accepts an optional ``state_prep`` kernel with signature
``(qubits: cudaq.qview)``; when provided, the factory returns a
zero-argument, fully hardware-shaped circuit (no ``cudaq.State`` data
anywhere). These tests pin each prep-mode circuit against its
``State``-taking twin fed the identical prepared state as data.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT, Walk
from cudaq_algorithms import sim_utils as sim

HAMILTONIAN = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}
THETA0, THETA1 = 0.37, -0.52


@cudaq.kernel
def product_prep(qubits: cudaq.qview):
    rx(0.37, qubits[0])
    ry(-0.52, qubits[1])


def _prepared_ket():
    """The dense statevector product_prep produces (little-endian)."""
    q0 = np.array([np.cos(0.5 * THETA0), -1.0j * np.sin(0.5 * THETA0)])
    q1 = np.array([np.cos(0.5 * THETA1), np.sin(0.5 * THETA1)])
    return np.kron(q1, q0).astype(np.complex128)


def _state(kernel, *args):
    return np.asarray(cudaq.get_state(kernel, *args), dtype=np.complex128)


def test_encode_kernel_prep_injection():
    enc = PauliLCU(HAMILTONIAN)
    via_prep = _state(enc.encode_kernel(state_prep=product_prep))
    via_state = _state(enc.encode_kernel(), sim.state_from(_prepared_ket()))
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)


def test_walk_kernel_prep_injection():
    enc = PauliLCU(HAMILTONIAN)
    via_prep = _state(enc.walk_kernel(power=2, state_prep=product_prep))
    via_state = _state(enc.walk_kernel(power=2),
                       sim.state_from(_prepared_ket()))
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)


def test_walk_factories_prep_injection():
    walk = Walk(PauliLCU(HAMILTONIAN))
    ket = sim.state_from(_prepared_ket())

    for factory, kwargs in (
        (walk.kernel, dict(power=2)),
        (walk.kernel, dict(power=1, uncompute=False)),
        (walk.adjoint_kernel, dict(power=2)),
        (walk.roundtrip_kernel, dict(power=2)),
        (walk.controlled_kernel, dict(power=2, control_state=1)),
        (walk.controlled_kernel, dict(power=2, control_state=0)),
        (walk.controlled_roundtrip_kernel, dict(power=1)),
    ):
        via_prep = _state(factory(state_prep=product_prep, **kwargs))
        via_state = _state(factory(**kwargs), ket)
        np.testing.assert_allclose(via_prep,
                                   via_state,
                                   atol=1e-12,
                                   err_msg=f"{factory.__name__} {kwargs}")


def test_qsvt_kernels_prep_injection():
    transformer = QSVT(PauliLCU(HAMILTONIAN))
    sequence = PhaseSequence([0.4, -0.2, 0.7],
                             walk_directions=["forward", "adjoint"])
    ket = sim.state_from(_prepared_ket())

    via_prep = _state(transformer.kernel(sequence, state_prep=product_prep))
    via_state = _state(transformer.kernel(sequence), ket)
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)

    via_prep = _state(
        transformer.controlled_kernel(sequence,
                                      control_state=1,
                                      state_prep=product_prep))
    via_state = _state(
        transformer.controlled_kernel(sequence, control_state=1), ket)
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)


def test_moment_prep_injection_matches_ket_path():
    walk = Walk(PauliLCU(HAMILTONIAN))
    ket = _prepared_ket()
    for order in range(4):
        via_ket = walk.moment(ket, order)
        via_prep = walk.moment(None, order, state_prep=product_prep)
        assert via_prep == pytest.approx(via_ket, abs=1e-10)

    np.testing.assert_allclose(walk.moments(None, 4, state_prep=product_prep),
                               walk.moments(ket, 4),
                               atol=1e-10)


def test_moment_requires_exactly_one_input_mode():
    walk = Walk(PauliLCU(HAMILTONIAN))
    ket = _prepared_ket()
    with pytest.raises(ValueError, match="exactly one"):
        walk.moment(ket, 1, state_prep=product_prep)
    with pytest.raises(ValueError, match="exactly one"):
        walk.moment(None, 1)
    with pytest.raises(ValueError, match="exactly one"):
        walk.moments(ket, 3, state_prep=product_prep)


def test_prep_mode_returns_zero_argument_kernels():
    # The injected form must be directly sampleable: no arguments at all.
    enc = PauliLCU(HAMILTONIAN)
    kernel = Walk(enc).kernel(power=1, state_prep=product_prep)
    counts = cudaq.sample(kernel, shots_count=100)
    assert sum(counts.values()) == 100
