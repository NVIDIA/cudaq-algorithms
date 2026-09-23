# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent calibration tests for the bounded E08 evolution oracle."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.linalg import expm

PATH = Path(__file__).resolve().parents[1] / "checks/evolution_oracle.py"
oracle = None
if PATH.is_file():
    spec = importlib.util.spec_from_file_location("strategy_evolution_oracle",
                                                  PATH)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def pauli_rotation(pauli, coefficient, identity, state, time):
    """Hand-derived exp(-it(c I + a P)) for P squared equal to I."""
    return (np.exp(-1j * identity * time) *
            (np.cos(coefficient * time) * state -
             1j * np.sin(coefficient * time) * pauli @ state))


def little_endian_operator(operation, qubit, width):
    factors = [
        operation if index == qubit else I for index in reversed(range(width))
    ]
    answer = factors[0]
    for factor in factors[1:]:
        answer = np.kron(answer, factor)
    return answer


@pytest.mark.parametrize(
    "pauli,coefficient,identity,state,time",
    [
        (X, 0.37, 0.21, np.array([1, 0], complex), 0.83),
        (Z, -0.48, 0.13, np.array([1, 2j], complex) / np.sqrt(5), -0.61),
        (Y, 0.72, -0.31, np.array([1 + 1j, 2 - 1j]) / np.sqrt(7), 0.44),
        (X, 0.37, 0.21, np.array([1, 0], complex), 0.0),
    ],
)
def test_expected_state_matches_analytic_pauli_rotations(
        pauli, coefficient, identity, state, time):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    hamiltonian = identity * I + coefficient * pauli
    expected = pauli_rotation(pauli, coefficient, identity, state, time)
    np.testing.assert_allclose(oracle.expected_state(hamiltonian, state, time),
                               expected,
                               atol=2e-15,
                               rtol=0)


def test_noncommuting_real_xz_rotation_keeps_combined_generator_and_time_scale(
):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    ax, az, identity, time = 0.29, -0.41, 0.17, 0.73
    state = np.array([1, 2], complex) / np.sqrt(5)
    frequency = np.hypot(ax, az)
    direction = (ax * X + az * Z) / frequency
    expected = pauli_rotation(direction, frequency, identity, state, time)
    np.testing.assert_allclose(oracle.expected_state(
        identity * I + ax * X + az * Z, state, time),
                               expected,
                               atol=2e-15,
                               rtol=0)


def test_two_qubit_complex_noncommuting_reference_matches_scipy_exponential():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    # q0 is the right Kronecker factor. Mixed terms expose an endian swap.
    hamiltonian = (0.23 * little_endian_operator(X, 0, 2) -
                   0.31 * little_endian_operator(Z, 1, 2) +
                   0.19 * np.kron(Y, X) + 0.11 * np.kron(I, I))
    state = np.array([1, 2j, -1 + 1j, 3], complex) / 4
    time = -0.57
    expected = expm(-1j * hamiltonian * time) @ state
    np.testing.assert_allclose(oracle.expected_state(hamiltonian, state, time),
                               expected,
                               atol=3e-15,
                               rtol=0)


@pytest.mark.parametrize(
    "defect",
    ["wrong_sign", "missing_time", "missing_scale", "drop_identity_phase"],
)
def test_exact_comparison_catches_plausible_evolution_mutations(defect):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    state = np.array([1, 2j], complex) / np.sqrt(5)
    coefficient, identity, time = 0.43, 0.27, 0.69
    expected = pauli_rotation(Y, coefficient, identity, state, time)
    if defect == "wrong_sign":
        actual = pauli_rotation(Y, coefficient, identity, state, -time)
    elif defect == "missing_time":
        actual = pauli_rotation(Y, coefficient, identity, state, 1)
    elif defect == "missing_scale":
        actual = pauli_rotation(Y, 1, identity, state, time)
    else:
        actual = pauli_rotation(Y, coefficient, 0, state, time)
    with pytest.raises(AssertionError):
        oracle.assert_evolved_state(actual,
                                    expected,
                                    atol=1e-10,
                                    phase_policy="exact")


@pytest.mark.parametrize(
    "invalid",
    [
        "scalar_h",
        "nonsquare_h",
        "nonpower_h",
        "zero_h",
        "oversized_h",
        "nonhermitian_h",
        "nonfinite_h",
        "bad_state_shape",
        "wrong_state_length",
        "nonunit_state",
        "nonfinite_state",
        "bad_h_type",
        "bad_state_type",
    ],
)
def test_expected_state_rejects_invalid_evaluator_inputs(invalid):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    hamiltonian = X.copy()
    state = np.array([1, 0], complex)
    if invalid == "scalar_h": hamiltonian = 1
    if invalid == "nonsquare_h": hamiltonian = np.ones((2, 3))
    if invalid == "nonpower_h": hamiltonian = np.eye(3)
    if invalid == "zero_h": hamiltonian = np.empty((0, 0))
    if invalid == "oversized_h": hamiltonian = np.eye(32)
    if invalid == "nonhermitian_h": hamiltonian[0, 1] = 2
    if invalid == "nonfinite_h": hamiltonian[0, 0] = np.nan
    if invalid == "bad_state_shape": state = state.reshape(2, 1)
    if invalid == "wrong_state_length": state = np.array([1, 0, 0, 0])
    if invalid == "nonunit_state": state = np.array([0.5, 0])
    if invalid == "nonfinite_state": state[0] = np.inf
    if invalid == "bad_h_type": hamiltonian = object()
    if invalid == "bad_state_type": state = object()
    with pytest.raises(ValueError):
        oracle.expected_state(hamiltonian, state, 0.2)


@pytest.mark.parametrize(
    "invalid_time",
    [
        True, False, 0j, 1 + 0j, "0.2", [0.2],
        np.array(0.2), np.nan, np.inf, -np.inf
    ],
)
def test_expected_state_rejects_non_real_scalar_or_nonfinite_time(
        invalid_time):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    with pytest.raises(ValueError):
        oracle.expected_state(X, np.array([1, 0], complex), invalid_time)


def test_expected_state_rejects_nonfinite_intermediate_phase_from_finite_inputs(
):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    hamiltonian = np.diag([1e308, -1e308])
    with pytest.raises(ValueError):
        oracle.expected_state(hamiltonian, np.array([1, 0], complex), 1e308)


def test_huge_real_integer_time_is_an_evaluator_value_error():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    with pytest.raises(ValueError):
        oracle.expected_state(X, np.array([1, 0], complex), 10**10000)


def test_expected_state_does_not_mutate_or_normalize_inputs():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    hamiltonian = np.array([[0.2, 0.3j], [-0.3j, -0.1]], complex)
    state = np.array([1, 2j], complex) / np.sqrt(5)
    original_hamiltonian, original_state = hamiltonian.copy(), state.copy()
    oracle.expected_state(hamiltonian, state, -0.3)
    np.testing.assert_array_equal(hamiltonian, original_hamiltonian)
    np.testing.assert_array_equal(state, original_state)


def test_exact_policy_reports_both_errors_and_json_compatible_metrics():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 2j], complex) / np.sqrt(5)
    metrics = oracle.assert_evolved_state(expected.copy(),
                                          expected,
                                          atol=1e-12,
                                          phase_policy="exact")
    assert metrics["phase_policy"] == "exact"
    assert {
        key: metrics[key]
        for key in (
            "raw_l2_error",
            "aligned_l2_error",
            "norm_error",
            "fidelity",
        )
    } == pytest.approx({
        "raw_l2_error": 0.0,
        "aligned_l2_error": 0.0,
        "norm_error": 0.0,
        "fidelity": 1.0,
    })
    assert all(type(value) in (float, str) for value in metrics.values())


def test_global_phase_is_rejected_exactly_but_allowed_as_a_ray():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 2j], complex) / np.sqrt(5)
    actual = np.exp(0.71j) * expected
    with pytest.raises(AssertionError):
        oracle.assert_evolved_state(actual,
                                    expected,
                                    atol=1e-12,
                                    phase_policy="exact")
    metrics = oracle.assert_evolved_state(actual,
                                          expected,
                                          atol=1e-12,
                                          phase_policy="ray")
    assert metrics["raw_l2_error"] > 0.5
    assert metrics["aligned_l2_error"] < 1e-15
    assert metrics["fidelity"] == pytest.approx(1)


@pytest.mark.parametrize(
    "defect",
    [
        "relative_phase", "drop_imaginary", "bit_permutation", "zero",
        "half_norm"
    ],
)
def test_ray_policy_does_not_repair_incorrect_or_unnormalized_output(defect):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 1j, -1, 1 - 1j], complex) / np.sqrt(5)
    actual = expected.copy()
    if defect == "relative_phase": actual[1] *= -1
    if defect == "drop_imaginary":
        actual = actual.real.astype(complex)
        actual /= np.linalg.norm(actual)
    if defect == "bit_permutation": actual = actual[[0, 2, 1, 3]]
    if defect == "zero": actual[:] = 0
    if defect == "half_norm": actual *= 0.5
    with pytest.raises(AssertionError):
        oracle.assert_evolved_state(actual,
                                    expected,
                                    atol=1e-10,
                                    phase_policy="ray")


@pytest.mark.parametrize("defect", ["wrong_shape", "nan", "inf", "bad_type"])
def test_malformed_actual_output_is_an_assertion_failure(defect):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    actual = expected.copy()
    if defect == "wrong_shape": actual = actual.reshape(2, 1)
    if defect == "nan": actual[0] = np.nan
    if defect == "inf": actual[0] = np.inf
    if defect == "bad_type": actual = object()
    with pytest.raises(AssertionError):
        oracle.assert_evolved_state(actual,
                                    expected,
                                    atol=1e-10,
                                    phase_policy="exact")


@pytest.mark.parametrize(
    "invalid",
    [
        "wrong_shape", "empty", "not_power", "oversized", "nonfinite",
        "nonunit", "bad_type"
    ],
)
def test_invalid_expected_state_is_an_evaluator_value_error(invalid):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    if invalid == "wrong_shape": expected = expected.reshape(2, 1)
    if invalid == "empty": expected = np.array([], complex)
    if invalid == "not_power": expected = np.array([1, 0, 0], complex)
    if invalid == "oversized": expected = np.eye(1, 32, dtype=complex).ravel()
    if invalid == "nonfinite": expected[0] = np.nan
    if invalid == "nonunit": expected *= 0.5
    if invalid == "bad_type": expected = object()
    with pytest.raises(ValueError):
        oracle.assert_evolved_state(expected,
                                    expected,
                                    atol=1e-10,
                                    phase_policy="exact")


@pytest.mark.parametrize(
    "atol",
    [
        True, False, 0, -1e-3, 0.0100001, np.nan, np.inf, 1e-3 + 0j, "0.001",
        [1e-3],
        np.array(1e-3)
    ],
)
def test_invalid_tolerance_is_rejected_as_configuration_error(atol):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    with pytest.raises(ValueError):
        oracle.assert_evolved_state(expected,
                                    expected,
                                    atol=atol,
                                    phase_policy="exact")


def test_huge_real_integer_tolerance_is_a_configuration_value_error():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    with pytest.raises(ValueError):
        oracle.assert_evolved_state(expected,
                                    expected,
                                    atol=10**10000,
                                    phase_policy="exact")


@pytest.mark.parametrize("phase_policy",
                         [None, "", "global", "EXACT", True, 1])
def test_invalid_phase_policy_is_rejected_as_configuration_error(phase_policy):
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    with pytest.raises(ValueError):
        oracle.assert_evolved_state(expected,
                                    expected,
                                    atol=1e-10,
                                    phase_policy=phase_policy)


def test_phase_policy_is_required_explicitly():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    with pytest.raises(TypeError):
        oracle.assert_evolved_state(expected, expected, atol=1e-10)


def test_l2_error_and_norm_error_both_gate_at_the_requested_boundary():
    assert oracle is not None, "E08 evolution oracle is not implemented"
    expected = np.array([1, 0], complex)
    actual = np.array([np.sqrt(1 - 4e-6), 2e-3], complex)
    with pytest.raises(AssertionError):
        oracle.assert_evolved_state(actual,
                                    expected,
                                    atol=1e-3,
                                    phase_policy="ray")
    metrics = oracle.assert_evolved_state(actual,
                                          expected,
                                          atol=3e-3,
                                          phase_policy="ray")
    assert metrics["norm_error"] < 1e-12
    assert metrics["aligned_l2_error"] == pytest.approx(2e-3, rel=2e-6)


def test_actual_output_gates_survive_optimized_python():
    script = textwrap.dedent("""
        import importlib.util
        import sys

        import numpy as np

        assert sys.flags.optimize > 0
        spec = importlib.util.spec_from_file_location("optimized_oracle", sys.argv[1])
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        expected = np.array([1, 0], complex)
        invalid = {
            "wrong shape": expected.reshape(2, 1),
            "nonfinite": np.array([np.nan, 0], complex),
            "wrong norm": 0.5 * expected,
            "wrong amplitude": np.array([0, 1], complex),
        }
        for label, actual in invalid.items():
            try:
                oracle.assert_evolved_state(
                    actual, expected, atol=1e-10, phase_policy="exact")
            except AssertionError:
                continue
            raise RuntimeError(f"optimized oracle accepted {label}")
        metrics = oracle.assert_evolved_state(
            expected.copy(), expected, atol=1e-10, phase_policy="exact")
        if metrics["fidelity"] < 1 - 1e-12:
            raise RuntimeError("optimized oracle rejected the positive control")
    """)
    result = subprocess.run(
        [sys.executable, "-O", "-B", "-c", script,
         str(PATH)],
        capture_output=True,
        text=True,
        check=False)
    assert result.returncode == 0, result.stdout + result.stderr
