# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Correctness tests for the PauliLCU block encoding (dense references)."""

import math

import numpy as np
import pytest

import cudaq
from cudaq import spin

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import (PauliLCU, PhaseSequence, QSVT,
                              apply_controlled_phase_sequence,
                              apply_phase_sequence, prepare,
                              reflect_about_zero, signal_phase, state_from,
                              unprepare)

from dense_references import dense_matrix, random_ket

FOUR_TERMS = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_action_matches_dense_hamiltonian():
    enc = PauliLCU(FOUR_TERMS)
    assert enc.num_system == 2
    assert enc.num_ancilla == 2
    assert enc.alpha == pytest.approx(1.43)

    ket = random_ket(2, seed=7)
    expected = dense_matrix(list(
        (c, w) for w, c in FOUR_TERMS.items()), 2) @ ket / enc.alpha
    assert np.allclose(sim.action(enc, ket), expected, atol=1e-10)


def test_spin_operator_and_pairs_inputs_agree():
    h = 0.7 * spin.z(0) - 0.43 * spin.z(1) + 0.19 * spin.x(0) * spin.x(1)
    from_op = PauliLCU(h, num_qubits=2)
    from_pairs = PauliLCU([(0.7, "ZI"), (-0.43, "IZ"), (0.19, "XX")])

    assert from_op.alpha == pytest.approx(from_pairs.alpha)
    ket = random_ket(2, seed=11)
    assert np.allclose(sim.action(from_op, ket),
                       sim.action(from_pairs, ket),
                       atol=1e-10)


def test_single_term_negative_coefficient_keeps_sign():
    # Single-term regression: -c * P must encode -c * P, not +c * P.
    # Single-term encodings are normalized to one (idle) ancilla so the
    # full walk/QSVT machinery applies to them unchanged.
    enc = PauliLCU({"XZ": -0.5})
    assert enc.num_ancilla == 1

    ket = random_ket(2, seed=3)
    expected = dense_matrix([(-0.5, "XZ")], 2) @ ket / enc.alpha
    assert np.allclose(sim.action(enc, ket), expected, atol=1e-10)


def test_single_term_walk_keeps_minus_h_over_alpha_sign():
    # walk_kernel's block must be T_power(-H/alpha) for single-term
    # encodings too (regression: the old 0-ancilla path dropped the sign).
    enc = PauliLCU({"X": 1.0})
    ket = np.array([0.6, 0.8], dtype=np.complex128)
    state = np.asarray(
        cudaq.get_state(enc.walk_kernel(power=1), sim.state_from(ket)))
    block = sim.good_subspace(enc, state)
    expected = -dense_matrix([(1.0, "X")], 1) @ ket  # T_1(-H/alpha)|ket>
    assert np.allclose(block, expected, atol=1e-10)


def test_identity_term_handling():
    enc = PauliLCU({"II": 0.2, "XI": 0.5, "ZI": 0.3})
    assert enc.constant_term == pytest.approx(0.2)
    assert enc.num_terms == 3
    assert enc.alpha == pytest.approx(1.0)

    excluded = PauliLCU({
        "II": 0.2,
        "XI": 0.5,
        "ZI": 0.3
    },
                        include_identity=False)
    assert excluded.constant_term == pytest.approx(0.2)
    assert excluded.num_terms == 2
    assert excluded.alpha == pytest.approx(0.8)


def test_walk_moments_match_chebyshev():
    # Asymmetric spectrum 0.2 +/- sqrt(0.34): the reflection expectation after
    # k walks must reproduce <T_2k(H/alpha)> with both eigenvalue weights.
    enc = PauliLCU({"I": 0.2, "X": 0.5, "Z": 0.3})
    assert enc.num_ancilla == 2

    lam = math.sqrt(0.34)
    theta = math.atan2(0.5, 0.3)
    prep_angle = 0.7
    delta = prep_angle - theta
    weights = (math.cos(delta / 2)**2, math.sin(delta / 2)**2)
    eigenvalues = (0.2 + lam, 0.2 - lam)

    ket = np.array([math.cos(prep_angle / 2),
                    math.sin(prep_angle / 2)],
                   dtype=np.complex128)
    for k in (1, 2, 3):
        state = cudaq.get_state(enc.walk_kernel(power=k), state_from(ket))
        zero_probability = float(
            np.sum(np.abs(sim.good_subspace(enc, state))**2))
        moment = 2.0 * zero_probability - 1.0

        expected = sum(w * math.cos(2 * k * math.acos(e / enc.alpha))
                       for w, e in zip(weights, eigenvalues))
        assert moment == pytest.approx(expected, abs=1e-10)


def test_validation_errors():
    with pytest.raises(ValueError):
        PauliLCU({})
    with pytest.raises(ValueError):
        PauliLCU({"XI": 0.5, "XII": 0.3})
    with pytest.raises(ValueError):
        PauliLCU({"XQ": 0.5})
    with pytest.raises(ValueError):
        PauliLCU({"XI": 0.5}, num_qubits=3)
    with pytest.raises(TypeError):
        PauliLCU(42)


def test_repr_reads_well():
    text = repr(PauliLCU(FOUR_TERMS))
    assert "terms=4" in text
    assert "ancilla_qubits=2" in text


def test_module_level_phase_sequence_kernels_compose():
    # Regression: apply_phase_sequence once referenced a stale kernel name
    # and could not compile; both escape hatches must stay composable and
    # agree with the QSVT factories.
    enc = PauliLCU({"ZI": 0.7, "IZ": -0.43, "XX": 0.19})
    seq = PhaseSequence([0.4, -0.2, 0.7])
    phases = seq.projector_phases
    directions = list(seq.walk_directions)
    angles, controls, ops, lengths, signs = enc.kernel_args
    n_anc = enc.num_ancilla
    ket = random_ket(2, seed=13)

    @cudaq.kernel
    def composed(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(n_anc)
        apply_phase_sequence(signal, system, phases, directions, angles,
                             controls, ops, lengths, signs)

    reference = np.asarray(
        cudaq.get_state(QSVT(enc).kernel(seq), sim.state_from(ket)))
    state = np.asarray(cudaq.get_state(composed, sim.state_from(ket)))
    assert np.allclose(state, reference, atol=1e-12)

    @cudaq.kernel
    def composed_controlled(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_signal = cudaq.qvector(1 + n_anc)
        x(control_and_signal[0])
        apply_controlled_phase_sequence(control_and_signal, system, phases,
                                        directions, angles, controls, ops,
                                        lengths, signs)

    reference = np.asarray(
        cudaq.get_state(
            QSVT(enc).controlled_kernel(seq, control_state=1),
            sim.state_from(ket)))
    state = np.asarray(
        cudaq.get_state(composed_controlled, sim.state_from(ket)))
    assert np.allclose(state, reference, atol=1e-12)


def test_kernels_tolerate_empty_register_views():
    # Regression: early `return` in kernels is silently ignored
    # (cuda-quantum#4845); the guards must be positive blocks, or an
    # empty-view call executes the body and crashes the process.
    angles = [0.3]

    @cudaq.kernel
    def probe():
        q = cudaq.qvector(2)
        prepare(q.front(0), angles)
        unprepare(q.front(0), angles)
        reflect_about_zero(q.front(0))
        signal_phase(q.front(0), 0.7)

    state = np.asarray(cudaq.get_state(probe))
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1.0
    assert np.allclose(state, expected, atol=1e-12)


def test_identity_only_hamiltonian_encodes_signed_identity():
    # Regression: identity-only Hamiltonians produce an empty term_ops
    # list, which cannot cross the kernel boundary (cuda-quantum#4847);
    # the extraction pads it with a never-dereferenced entry.
    ket = random_ket(2, seed=21)
    positive = PauliLCU({"II": 1.5})
    assert positive.num_ancilla == 1
    assert np.allclose(sim.action(positive, ket), ket, atol=1e-12)

    negative = PauliLCU({"II": -0.5})
    assert np.allclose(sim.action(negative, ket), -ket, atol=1e-12)

    # The walk machinery applies unchanged: T_1(-H/alpha) = -sign * I.
    walk_state = np.asarray(
        cudaq.get_state(negative.walk_kernel(power=1), sim.state_from(ket)))
    assert np.allclose(sim.good_subspace(negative, walk_state),
                       ket,
                       atol=1e-12)
