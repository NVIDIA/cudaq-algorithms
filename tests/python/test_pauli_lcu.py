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
from cudaq_algorithms import PauliLCU, state_from


def dense_matrix(terms, num_qubits):
    """Dense Pauli-sum matrix in CUDA-Q's little-endian qubit order."""
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for coeff, word in terms:
        for column in range(dimension):
            row = column
            phase = complex(coeff)
            for qubit, label in enumerate(word):
                bit = (column >> qubit) & 1
                if label == "X":
                    row ^= 1 << qubit
                elif label == "Y":
                    row ^= 1 << qubit
                    phase *= 1.0j if bit == 0 else -1.0j
                elif label == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
            matrix[row, column] += phase
    return matrix


def random_ket(num_qubits, seed):
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=1 << num_qubits) + 1.0j * rng.normal(
        size=1 << num_qubits)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


FOUR_TERMS = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_action_matches_dense_hamiltonian():
    enc = PauliLCU(FOUR_TERMS)
    assert enc.num_system == 2
    assert enc.num_ancilla == 2
    assert enc.alpha == pytest.approx(1.43)

    ket = random_ket(2, seed=7)
    expected = dense_matrix(list((c, w) for w, c in FOUR_TERMS.items()),
                            2) @ ket / enc.alpha
    assert np.allclose(sim.action(enc, ket), expected, atol=1e-10)


def test_spin_operator_and_pairs_inputs_agree():
    h = 0.7 * spin.z(0) - 0.43 * spin.z(1) + 0.19 * spin.x(0) * spin.x(1)
    from_op = PauliLCU(h, num_qubits=2)
    from_pairs = PauliLCU([(0.7, "ZI"), (-0.43, "IZ"), (0.19, "XX")])

    assert from_op.alpha == pytest.approx(from_pairs.alpha)
    ket = random_ket(2, seed=11)
    assert np.allclose(sim.action(from_op, ket), sim.action(from_pairs, ket),
                       atol=1e-10)


def test_single_term_negative_coefficient_keeps_sign():
    # Zero-ancilla regression: -c * P must encode -c * P, not +c * P.
    enc = PauliLCU({"XZ": -0.5})
    assert enc.num_ancilla == 0

    ket = random_ket(2, seed=3)
    expected = dense_matrix([(-0.5, "XZ")], 2) @ ket / enc.alpha
    assert np.allclose(sim.action(enc, ket), expected, atol=1e-10)


def test_identity_term_handling():
    enc = PauliLCU({"II": 0.2, "XI": 0.5, "ZI": 0.3})
    assert enc.constant_term == pytest.approx(0.2)
    assert enc.num_terms == 3
    assert enc.alpha == pytest.approx(1.0)

    excluded = PauliLCU({"II": 0.2, "XI": 0.5, "ZI": 0.3},
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

    ket = np.array([math.cos(prep_angle / 2), math.sin(prep_angle / 2)],
                   dtype=np.complex128)
    for k in (1, 2, 3):
        state = cudaq.get_state(enc.walk_kernel(power=k), state_from(ket))
        zero_probability = float(
            np.sum(np.abs(sim.good_subspace(enc, state))**2))
        moment = 2.0 * zero_probability - 1.0

        expected = sum(
            w * math.cos(2 * k * math.acos(e / enc.alpha))
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
