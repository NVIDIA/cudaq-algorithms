# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent calibration of the E05 reference, not a worker evaluation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "checks/ci_oracle.py"
SPEC = importlib.util.spec_from_file_location("strategy_ci_oracle", PATH)
oracle = None
if PATH.is_file():
    oracle = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(oracle)

OCCUPATIONS = ((0, 1), (0, 2), (1, 3), (2, 3))
WEIGHTS = np.array([1, 2j, -3, -4j]) / np.sqrt(30)


def creation_reference(basis, occupations, weights):
    """Apply fermionic creation operators; deliberately no determinant calls."""
    n = len(basis)
    answer = np.zeros(1 << n, dtype=complex)
    for occupied, weight in zip(occupations, weights):
        state = np.zeros(1 << n, dtype=complex)
        state[0] = 1
        for orbital in reversed(occupied):
            after = np.zeros_like(state)
            for old, amplitude in enumerate(state):
                for site in range(n):
                    if not old & (1 << site):
                        parity = (old & ((1 << site) - 1)).bit_count()
                        after[old | (1 << site)] += (
                            -1)**parity * basis[site, orbital] * amplitude
            state = after
        answer += weight * state
    return answer / np.linalg.norm(answer)


def test_identity_basis_has_hand_checked_little_endian_complex_amplitudes():
    assert oracle is not None, "CI oracle is not implemented"
    expected = np.zeros(16, dtype=complex)
    expected[[3, 5, 10, 12]] = WEIGHTS
    np.testing.assert_allclose(oracle.expected_state(np.eye(4), OCCUPATIONS,
                                                     WEIGHTS),
                               expected,
                               atol=1e-12,
                               rtol=0)


@pytest.mark.parametrize("complex_basis", [False, True])
@pytest.mark.parametrize("complex_weights", [False, True])
def test_nontrivial_basis_reference_matches_independent_creation_operators(
        complex_basis, complex_weights):
    assert oracle is not None, "CI oracle is not implemented"
    raw = np.random.default_rng(1729).normal(size=(4, 4))
    if complex_basis:
        raw = raw + 1j * np.random.default_rng(1730).normal(size=(4, 4))
    basis, _ = np.linalg.qr(raw)
    weights = WEIGHTS if complex_weights else np.array([1, -2, 3, -4
                                                        ]) / np.sqrt(30)
    expected = creation_reference(basis, OCCUPATIONS, weights)
    actual = oracle.expected_state(basis, OCCUPATIONS, weights)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert np.isclose(np.linalg.norm(actual), 1, atol=1e-12)
    assert sum(abs(actual[i])**2 for i in range(16)
               if i.bit_count() != 2) < 1e-24


def test_reference_normalizes_the_complete_nonzero_weighted_sum():
    assert oracle is not None, "CI oracle is not implemented"
    expected = np.zeros(16, dtype=complex)
    expected[[3, 5, 10, 12]] = WEIGHTS
    np.testing.assert_allclose(oracle.expected_state(np.eye(4), OCCUPATIONS,
                                                     7 * WEIGHTS),
                               expected,
                               atol=1e-12,
                               rtol=0)


@pytest.mark.parametrize("scale", [1e-310, 7e307])
def test_finite_extreme_weight_scale_does_not_overflow_reference(scale):
    weights = np.array([1 + 1j, 2 - 1j, -1 + 2j, -2 - 2j])
    expected = creation_reference(np.eye(4), OCCUPATIONS, weights)
    np.testing.assert_allclose(oracle.expected_state(np.eye(4), OCCUPATIONS,
                                                     scale * weights),
                               expected,
                               atol=1e-12,
                               rtol=0)


@pytest.mark.parametrize("invalid", [
    "nonsquare", "nonorthogonal", "nonfinite_basis", "wrong_count",
    "empty_occupation", "duplicate_index", "outside", "boolean", "unordered",
    "repeated_determinant", "different_electrons", "bad_weights",
    "zero_weights", "nonfinite_weights", "too_large"
])
def test_invalid_reference_inputs_are_rejected(invalid):
    assert oracle is not None, "CI oracle is not implemented"
    basis, occupations, weights = np.eye(4), list(OCCUPATIONS), WEIGHTS.copy()
    if invalid == "nonsquare": basis = np.ones((4, 2))
    if invalid == "nonorthogonal": basis[0, 0] = 2
    if invalid == "nonfinite_basis": basis[0, 0] = np.nan
    if invalid == "wrong_count": occupations = occupations[:3]
    if invalid == "empty_occupation": occupations = [()] * 4
    if invalid == "duplicate_index": occupations[0] = (0, 0)
    if invalid == "outside": occupations[0] = (0, 4)
    if invalid == "boolean": occupations[0] = (False, 1)
    if invalid == "unordered": occupations[0] = (1, 0)
    if invalid == "repeated_determinant": occupations[1] = occupations[0]
    if invalid == "different_electrons": occupations[0] = (0, )
    if invalid == "bad_weights": weights = weights[:3]
    if invalid == "zero_weights": weights[:] = 0
    if invalid == "nonfinite_weights": weights[0] = np.inf
    if invalid == "too_large": basis = np.eye(9)
    with pytest.raises(ValueError):
        oracle.expected_state(basis, occupations, weights)


def test_case_inputs_preserve_weights_and_offer_both_exact_representations():
    assert oracle is not None, "CI oracle is not implemented"
    basis = np.eye(4)
    weights = 3 * WEIGHTS
    data = oracle.case_inputs(basis, OCCUPATIONS, weights)
    np.testing.assert_array_equal(data["orbital_basis"], basis)
    np.testing.assert_array_equal(data["coefficients"], weights)
    assert data["occupations"] == [list(d) for d in OCCUPATIONS]
    for determinant, matrix in zip(OCCUPATIONS, data["orbital_matrices"]):
        np.testing.assert_array_equal(matrix, basis[:, determinant])
    data["orbital_basis"][0, 0] = 5
    data["coefficients"][0] = 0
    assert basis[0, 0] == 1 and weights[0] != 0


def test_one_overall_phase_is_accepted_without_normalizing_actual_state():
    assert oracle is not None, "CI oracle is not implemented"
    expected = creation_reference(np.eye(4), OCCUPATIONS, WEIGHTS)
    metrics = oracle.assert_prepared_state(np.exp(.71j) * expected, expected)
    assert metrics["fidelity"] == pytest.approx(1)
    assert metrics["max_amplitude_error"] < 1e-12
    assert metrics["norm_error"] < 1e-12


@pytest.mark.parametrize("defect", [
    "branch_phase", "drop_imaginary", "bit_reverse", "scaled", "all_zero",
    "wrong_shape", "garbage_ancilla", "nan", "wrong_particle_number"
])
def test_whole_state_assertion_rejects_plausible_incorrect_preparations(
        defect):
    assert oracle is not None, "CI oracle is not implemented"
    expected = creation_reference(np.eye(4), OCCUPATIONS, WEIGHTS)
    actual = expected.copy()
    if defect == "branch_phase": actual[5] *= -1
    if defect == "drop_imaginary":
        actual = actual.real.astype(complex)
        actual /= np.linalg.norm(actual)
    if defect == "bit_reverse":
        actual = actual[[int(f"{i:04b}"[::-1], 2) for i in range(16)]]
    if defect == "scaled": actual *= .5
    if defect == "all_zero": actual[:] = 0
    if defect == "wrong_shape": actual = actual.reshape(4, 4)
    if defect == "garbage_ancilla":
        actual = np.concatenate((actual, actual)) / np.sqrt(2)
    if defect == "nan": actual[0] = np.nan
    if defect == "wrong_particle_number": actual = np.roll(actual, 1)
    with pytest.raises(AssertionError):
        oracle.assert_prepared_state(actual, expected)


def test_walk_roundtrip_checks_all_ancillas_not_a_renormalized_projection():
    assert oracle is not None, "CI oracle is not implemented"
    expected = creation_reference(np.eye(4), OCCUPATIONS, WEIGHTS)
    actual = np.zeros(64, complex)
    actual[:16] = expected
    assert oracle.assert_clean_ancillas(
        actual, expected, num_ancilla=2)["fidelity"] == pytest.approx(1)
    leaked = actual.copy() / np.sqrt(2)
    leaked[16:32] = expected / np.sqrt(2)
    with pytest.raises(AssertionError):
        oracle.assert_clean_ancillas(leaked, expected, num_ancilla=2)
