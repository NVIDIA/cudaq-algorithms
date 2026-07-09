# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Correctness tests for QSVT (dense and 2x2 signal-model references)."""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import (ADJOINT, FORWARD, PauliLCU, PhaseSequence, QSVT,
                              recover_real_time_evolution, state_from)
from dense_references import dense_matrix, random_ket


def reference_response(sequence: PhaseSequence, x: float) -> complex:
    """Reference 2x2 signal model of the circuit built by ``QSVT.kernel``.

    Returns the upper-left element of the sequence's 2x2 signal matrix at
    scaled eigenvalue ``x``: the good-subspace block of the device circuit
    acting on an eigenstate with eigenvalue ``lambda`` equals
    ``reference_response(sequence, lambda / alpha)`` times that eigenstate.
    The walk's -H/alpha sign is folded into the step, so no caller-side
    negation is needed.

    For qsp-convention sequences the device circuit (which runs doubled
    projector phases) differs from this model by the global phase
    ``exp(i * sum(phases))``; ``recover_real_time_evolution`` accounts
    for it.
    """
    x = float(x)
    assert abs(x) <= 1.0
    s = math.sqrt(max(0.0, 1.0 - x * x))

    # One forward step of the circuit on the 2D invariant subspace:
    # reflect_about_zero * block_encoding = diag(-1, 1) @ [[x, s], [s, -x]].
    step_forward = np.array([[-x, -s], [s, -x]], dtype=np.complex128)
    step_adjoint = step_forward.T.copy()

    def phase_matrix(phi):
        if sequence.convention == "qsp":
            return np.diag(
                [np.exp(1.0j * phi), np.exp(-1.0j * phi)]).astype(complex)
        return np.diag([np.exp(1.0j * phi), 1.0]).astype(complex)

    matrix = phase_matrix(sequence.phases[0])
    for i in range(1, len(sequence.phases)):
        step = (step_adjoint
                if sequence.walk_directions[i - 1] == ADJOINT
                else step_forward)
        matrix = phase_matrix(sequence.phases[i]) @ step @ matrix
    return complex(matrix[0, 0])


TWO_TERMS_1Q = {"X": 0.5, "Z": 0.3}
FOUR_TERMS_2Q = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_phase_sequence_validation_and_conversion():
    seq = PhaseSequence([0.1, -0.2, 0.3])
    assert seq.degree == 2
    assert seq.convention == "qsvt"
    assert seq.walk_directions == (FORWARD, FORWARD)
    assert seq.projector_phases == pytest.approx([0.1, -0.2, 0.3])

    qsp = PhaseSequence([0.1, -0.2], convention="qsp")
    assert qsp.projector_phases == pytest.approx([0.2, -0.4])
    assert qsp.phases == pytest.approx((0.1, -0.2))  # raw stays raw

    mixed = PhaseSequence([0.1, -0.2, 0.3],
                          walk_directions=["forward", "adjoint"])
    assert mixed.walk_directions == (FORWARD, ADJOINT)

    with pytest.raises(ValueError):
        PhaseSequence([])
    with pytest.raises(ValueError):
        PhaseSequence([0.1, 0.2], walk_directions=[0, 1])
    with pytest.raises(ValueError):
        PhaseSequence([0.1, 0.2], walk_directions=["sideways"])
    with pytest.raises(ValueError):
        PhaseSequence([0.1], convention="phaseish")


def test_response_matches_circuit_on_eigenstate():
    # H = 0.5 X + 0.3 Z: eigenvalues +/- lambda, ry(theta)|0> the +lambda vec.
    enc = PauliLCU(TWO_TERMS_1Q)
    transformer = QSVT(enc)
    lam = math.sqrt(0.34)
    theta = math.atan2(0.5, 0.3)
    eigenvector = np.array([math.cos(theta / 2), math.sin(theta / 2)],
                           dtype=np.complex128)

    for phases in ([0.3, -0.4], [0.2, -0.5, 0.1, 0.4]):
        seq = PhaseSequence(phases)
        good = sim.transform(transformer, eigenvector, seq)
        # x is the plain scaled eigenvalue — no caller-side negation.
        response = reference_response(seq, lam / enc.alpha)
        assert np.allclose(good, response * eigenvector, atol=1e-10)

        # Guard: the negated-eigenvalue prediction must NOT match for odd
        # degree (the sign is folded into the model's walk step).
        flipped = reference_response(seq, -lam / enc.alpha)
        assert abs(response - flipped) > 1e-6


def test_full_block_matches_host_model_with_mixed_directions():
    enc = PauliLCU(FOUR_TERMS_2Q)
    transformer = QSVT(enc)
    seq = PhaseSequence([0.3, -0.4, 0.25],
                        walk_directions=["forward", "adjoint"])

    matrix = dense_matrix([(c, w) for w, c in FOUR_TERMS_2Q.items()], 2)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix / enc.alpha)

    columns = []
    for basis in range(4):
        ket = np.zeros(4, dtype=np.complex128)
        ket[basis] = 1.0
        columns.append(sim.transform(transformer, ket, seq))
    device_block = np.column_stack(columns)

    response = np.array(
        [reference_response(seq, float(m)) for m in eigenvalues])
    host_block = eigenvectors @ np.diag(response) @ eigenvectors.conj().T
    assert np.allclose(device_block, host_block, atol=1e-9)


def test_qsp_sequence_executes_as_doubled_projector_phases():
    enc = PauliLCU(TWO_TERMS_1Q)
    transformer = QSVT(enc)
    ket = random_ket(1, seed=9)

    qsp_seq = PhaseSequence([0.15, -0.3, 0.45], convention="qsp")
    doubled = PhaseSequence([0.3, -0.6, 0.9])
    a = sim.transform(transformer, ket, qsp_seq)
    b = sim.transform(transformer, ket, doubled)
    assert np.allclose(a, b, atol=1e-12)

    # And the two conventions genuinely differ in the signal model.
    r_qsp = reference_response(qsp_seq, 0.5)
    r_qsvt = reference_response(PhaseSequence([0.15, -0.3, 0.45]), 0.5)
    assert abs(r_qsp - r_qsvt) > 1e-6


def test_degree_zero_sequence_is_a_signal_phase():
    enc = PauliLCU(TWO_TERMS_1Q)
    transformer = QSVT(enc)
    ket = random_ket(1, seed=4)
    good = sim.transform(transformer, ket, [0.7])
    assert np.allclose(good, np.exp(0.7j) * ket, atol=1e-12)


def test_qsvt_single_term_encoding_block():
    # Single-term encodings are normalized to one ancilla; a degree-1
    # all-zero phase sequence implements the walk block -H/alpha.
    enc = PauliLCU({"XZ": -0.5})
    transformer = QSVT(enc)
    ket = random_ket(2, seed=5)
    block = sim.transform(transformer, ket, [0.0, 0.0])
    expected = -(dense_matrix([(-0.5, "XZ")], 2) / enc.alpha) @ ket
    assert np.allclose(block, expected, atol=1e-10)


def test_phase_sequence_conflicting_retag_rejected():
    seq = PhaseSequence([0.1, 0.2, 0.3], convention="qsp")
    transformer = QSVT(PauliLCU({"ZI": 0.7, "XX": 0.19}))
    with pytest.raises(ValueError, match="reinterpret"):
        transformer.kernel(seq, convention="qsvt")
    # Matching or absent convention tags pass through unchanged.
    assert transformer.kernel(seq, convention="qsp") is not None
    assert transformer.kernel(seq) is not None


def _qsppack_hamiltonian_simulation_phases(tau, degree):
    qsppack = pytest.importorskip("qsppack")
    special = pytest.importorskip("scipy.special")

    cos_coefficients = np.array(
        [0.5 * special.jv(0, tau)] +
        [((-1)**k) * special.jv(2 * k, tau)
         for k in range(1, degree // 2 + 1)],
        dtype=np.float64)
    sin_coefficients = np.array(
        [((-1)**k) * special.jv(2 * k + 1, tau) for k in range(degree // 2)],
        dtype=np.float64)
    options = {
        "criteria": 1e-12,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
    }
    cos_phases, _ = qsppack.solve(cos_coefficients, 0, {
        **options, "targetPre": True
    })
    sin_phases, _ = qsppack.solve(sin_coefficients, 1, {
        **options, "targetPre": False
    })
    return [float(p) for p in cos_phases], [float(p) for p in sin_phases]


def test_hamiltonian_simulation_with_qsppack_phases():
    terms = {"ZI": 0.6, "XX": 0.8}
    enc = PauliLCU(terms)
    transformer = QSVT(enc)

    time = 0.5
    tau = enc.alpha * time
    cos_phases, sin_phases = _qsppack_hamiltonian_simulation_phases(tau, 12)

    rng = np.random.default_rng(21)
    ket = rng.normal(size=4).astype(np.complex128)
    ket /= np.linalg.norm(ket)

    cos_state = sim.transform(
        transformer, ket, PhaseSequence(cos_phases, convention="qsp"))
    sin_state = sim.transform(
        transformer, ket, PhaseSequence(sin_phases, convention="qsp"))
    evolved = recover_real_time_evolution(cos_state, sin_state,
                                          cos_phases, sin_phases)

    matrix = dense_matrix([(c, w) for w, c in terms.items()], 2)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact = eigenvectors @ (np.exp(-1.0j * time * eigenvalues) *
                            (eigenvectors.conj().T @ ket))
    assert np.linalg.norm(evolved - exact) < 1e-8


def test_controlled_sequence_respects_control():
    enc = PauliLCU(FOUR_TERMS_2Q)
    transformer = QSVT(enc)
    seq = PhaseSequence([0.3, -0.4, 0.25],
                        walk_directions=["forward", "adjoint"])
    ket = random_ket(2, seed=23)
    ns, na = enc.num_system, enc.num_ancilla

    def con_index(sys, ctrl, anc):
        return sys + (ctrl << ns) + (anc << (ns + 1))

    reference = np.asarray(
        cudaq.get_state(transformer.kernel(seq), state_from(ket)))
    on_state = np.asarray(
        cudaq.get_state(transformer.controlled_kernel(seq, control_state=1),
                        state_from(ket)))
    off_state = np.asarray(
        cudaq.get_state(transformer.controlled_kernel(seq, control_state=0),
                        state_from(ket)))

    for anc in range(1 << na):
        for sys in range(1 << ns):
            unc = reference[sys + (anc << ns)]
            assert on_state[con_index(sys, 1, anc)] == pytest.approx(
                unc, abs=1e-10)
            assert abs(on_state[con_index(sys, 0, anc)]) < 1e-10
            expected = ket[sys] if anc == 0 else 0.0
            assert off_state[con_index(sys, 0, anc)] == pytest.approx(
                expected, abs=1e-10)
            assert abs(off_state[con_index(sys, 1, anc)]) < 1e-10
