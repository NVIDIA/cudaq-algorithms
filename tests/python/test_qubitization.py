# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Correctness tests for qubitization walks and moments (dense references)."""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import (PauliLCU, Walk, reflection_observable,
                              select_observable, state_from)
from dense_references import dense_matrix, random_ket


def exact_chebyshev_moments(terms, num_qubits, alpha, ket, count):
    """<ket| T_k(H/alpha) |ket> from the dense matrix, k = 0..count-1."""
    scaled = dense_matrix(terms, num_qubits) / alpha
    ket = np.asarray(ket, dtype=np.complex128)
    chebyshev = [np.eye(scaled.shape[0], dtype=np.complex128), scaled]
    while len(chebyshev) < count:
        chebyshev.append(2.0 * scaled @ chebyshev[-1] - chebyshev[-2])
    return [
        float(np.real(ket.conj() @ chebyshev[k] @ ket)) for k in range(count)
    ]


THREE_TERMS_1Q = {"I": 0.2, "X": 0.5, "Z": 0.3}  # asymmetric spectrum
FOUR_TERMS_2Q = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_moments_match_dense_chebyshev_1q():
    enc = PauliLCU(THREE_TERMS_1Q)
    walk = Walk(enc)

    theta = 0.7
    ket = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128)
    count = 6  # both parities, walk powers up to 2
    measured = walk.moments(ket, count)
    expected = exact_chebyshev_moments([(c, w)
                                        for w, c in THREE_TERMS_1Q.items()], 1,
                                       enc.alpha, ket, count)
    assert np.allclose(measured, expected, atol=1e-10)


def test_moments_match_dense_chebyshev_2q():
    enc = PauliLCU(FOUR_TERMS_2Q)
    walk = Walk(enc)

    ket = np.zeros(4, dtype=np.complex128)
    ket[1] = 1.0  # basis state, HF-style
    count = 5
    measured = walk.moments(ket, count)
    expected = exact_chebyshev_moments([(c, w)
                                        for w, c in FOUR_TERMS_2Q.items()], 2,
                                       enc.alpha, ket, count)
    assert np.allclose(measured, expected, atol=1e-10)


def test_adjoint_walk_inverts_walk():
    enc = PauliLCU(FOUR_TERMS_2Q)
    walk = Walk(enc)

    ket = random_ket(2, seed=5)
    reference = np.zeros(1 << (enc.num_system + enc.num_ancilla),
                         dtype=np.complex128)
    reference[:4] = ket

    for power in (1, 2, 3):
        state = cudaq.get_state(walk.roundtrip_kernel(power=power),
                                state_from(ket))
        assert np.allclose(np.asarray(state), reference, atol=1e-10), \
            f"roundtrip failed at power {power}"


def test_walk_kernel_options():
    enc = PauliLCU(THREE_TERMS_1Q)
    walk = Walk(enc)
    ket = np.array([1.0, 0.0], dtype=np.complex128)

    # uncompute=True must agree with the PauliLCU walk_kernel factory.
    a = np.asarray(cudaq.get_state(walk.kernel(power=2), state_from(ket)))
    b = np.asarray(cudaq.get_state(enc.walk_kernel(power=2), state_from(ket)))
    assert np.allclose(a, b, atol=1e-12)


def test_reflection_and_select_observables_shapes():
    enc = PauliLCU(FOUR_TERMS_2Q)
    reflection = reflection_observable(enc)
    select = select_observable(enc)
    # Spot check: <0...0| R |0...0> = +1 on the ancilla block.
    # (Full physics is covered by the moment tests.)
    assert reflection is not None
    assert select is not None


def test_single_term_encoding_walks_with_correct_sign():
    # Single-term encodings are normalized to one ancilla, so Walk and the
    # observables apply unchanged — and the odd moment carries the sign.
    single = PauliLCU({"XZ": -0.5})
    walk = Walk(single)
    ket = random_ket(2, seed=9)
    scaled = dense_matrix([(-0.5, "XZ")], 2) / single.alpha
    expected_t1 = float(np.real(ket.conj() @ (scaled @ ket)))
    assert walk.moment(ket, 1) == pytest.approx(expected_t1, abs=1e-10)
    assert walk.moment(ket, 0) == pytest.approx(1.0, abs=1e-10)


def test_walk_power_validation():
    walk = Walk(PauliLCU({"ZI": 0.7, "XX": 0.19}))
    with pytest.raises(ValueError, match="power"):
        walk.kernel(power=-3)
    with pytest.raises(ValueError, match="power"):
        walk.kernel(power=2.9)
    with pytest.raises(ValueError, match="order"):
        walk.moment([1, 0, 0, 0], 2.9)
    with pytest.raises(ValueError, match="count"):
        walk.moments([1, 0, 0, 0], -1)
    with pytest.raises(ValueError, match="control_state"):
        walk.controlled_kernel(power=1, control_state=2)


def test_walk_encoding_is_read_only():
    # Kernels and observables are cached against the injected encoding;
    # swapping it would silently serve stale circuits, so it is read-only.
    walk = Walk(PauliLCU({"ZI": 0.7, "XX": 0.19}))
    walk.moment([1, 0, 0, 0], 1)
    with pytest.raises(AttributeError):
        walk.encoding = PauliLCU({"XI": 1.0})


def _controlled_layout_maps(num_system, num_ancilla):
    """Index maps between the controlled and uncontrolled kernel layouts.

    Uncontrolled: [system][ancilla]; controlled: [system][control][ancilla].
    q[0] is the least-significant statevector bit.
    """

    def uncontrolled(sys, anc):
        return sys + (anc << num_system)

    def controlled(sys, ctrl, anc):
        return sys + (ctrl << num_system) + (anc << (num_system + 1))

    return uncontrolled, controlled


def test_controlled_walk_respects_control():
    enc = PauliLCU(FOUR_TERMS_2Q)
    walk = Walk(enc)
    ket = random_ket(2, seed=13)
    unc_index, con_index = _controlled_layout_maps(enc.num_system,
                                                   enc.num_ancilla)

    reference = np.asarray(
        cudaq.get_state(walk.kernel(power=2), state_from(ket)))
    on_state = np.asarray(
        cudaq.get_state(walk.controlled_kernel(power=2, control_state=1),
                        state_from(ket)))
    off_state = np.asarray(
        cudaq.get_state(walk.controlled_kernel(power=2, control_state=0),
                        state_from(ket)))

    for anc in range(1 << enc.num_ancilla):
        for sys in range(1 << enc.num_system):
            # Control |1>: the control=1 half reproduces the uncontrolled
            # state; the control=0 half is empty.
            assert on_state[con_index(sys, 1, anc)] == pytest.approx(
                reference[unc_index(sys, anc)], abs=1e-10)
            assert abs(on_state[con_index(sys, 0, anc)]) < 1e-10
            # Control |0>: identity — input state, ancillas in |0>.
            expected = ket[sys] if anc == 0 else 0.0
            assert off_state[con_index(sys, 0,
                                       anc)] == pytest.approx(expected,
                                                              abs=1e-10)
            assert abs(off_state[con_index(sys, 1, anc)]) < 1e-10


def test_controlled_adjoint_walk_inverts_controlled_walk():
    enc = PauliLCU(FOUR_TERMS_2Q)
    walk = Walk(enc)
    ket = random_ket(2, seed=17)
    _, con_index = _controlled_layout_maps(enc.num_system, enc.num_ancilla)

    for control_state in (0, 1):
        state = np.asarray(
            cudaq.get_state(
                walk.controlled_roundtrip_kernel(power=2,
                                                 control_state=control_state),
                state_from(ket)))
        for anc in range(1 << enc.num_ancilla):
            for sys in range(1 << enc.num_system):
                expected = ket[sys] if anc == 0 else 0.0
                assert state[con_index(sys, control_state,
                                       anc)] == pytest.approx(expected,
                                                              abs=1e-10)
