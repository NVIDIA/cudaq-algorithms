# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure behavioral controls for the E08 isolated runtime probe."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

CHECKS = Path(__file__).resolve().parents[1] / "checks"
sys.path.insert(0, str(CHECKS))
PATH = CHECKS / "evolution_runtime.py"
runtime = None
if PATH.is_file():
    spec = importlib.util.spec_from_file_location("strategy_evolution_runtime",
                                                  PATH)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)


class Bound:

    def __init__(self,
                 recover,
                 validate=lambda data: True,
                 *,
                 phase_policy="exact",
                 validation_mode="boolean"):
        self.recover = recover
        self.validate = validate
        self.phase_policy = phase_policy
        self.validation_mode = validation_mode


EXPECTED = np.array([3 / 5, 4j / 5], dtype=complex)
COS = np.array([3 / 5, 0], dtype=complex)
SIN = np.array([0, 4j / 5], dtype=complex)
PHASES = ([0.1, -0.2], [0.3, 0.4])


def assess(recover, *, expected=EXPECTED, tolerance=1e-9):
    bound = Bound(recover)
    return runtime._assess_recovery(bound, {"bundle": object()},
                                    COS,
                                    SIN,
                                    *PHASES,
                                    expected,
                                    tolerance=tolerance)


def linear(data):
    return data["cos_state"] + data["sin_state"]


def test_real_linear_recovery_passes_all_bounded_dependency_controls():
    assert runtime is not None, "E08 runtime probe is not implemented"
    state, evidence = assess(linear)
    np.testing.assert_array_equal(state, EXPECTED)
    assert evidence["zero_l2_error"] == pytest.approx(0)
    assert evidence["half_l2_error"] == pytest.approx(0)
    assert evidence["negative_l2_error"] == pytest.approx(0)
    assert evidence["additivity_l2_error"] == pytest.approx(0)
    assert evidence["fidelity"] == pytest.approx(1)


def test_recovery_ignoring_components_cannot_pass_zero_control():
    assert runtime is not None, "E08 runtime probe is not implemented"
    with pytest.raises(AssertionError, match="zero"):
        assess(lambda data: EXPECTED.copy())


@pytest.mark.parametrize("mutation", ["absolute", "norm_weighted"])
def test_absolute_or_norm_weighted_reconstruction_cannot_pass_signed_homogeneity(
        mutation):
    assert runtime is not None, "E08 runtime probe is not implemented"

    def recover(data):
        answer = data["cos_state"] + data["sin_state"]
        if mutation == "absolute":
            return np.abs(answer.real) + 1j * np.abs(answer.imag)
        norm = np.linalg.norm(answer)
        return answer if norm == 0 else answer / norm

    with pytest.raises(AssertionError, match="half|negative"):
        assess(recover)


def test_nonadditive_component_coupling_cannot_pass_separation_control():
    assert runtime is not None, "E08 runtime probe is not implemented"

    def coupled(data):
        cosine, sine = data["cos_state"], data["sin_state"]
        answer = cosine + sine
        if np.any(cosine) and np.any(sine):
            return answer
        if np.any(cosine):
            return answer + np.array([0, 0.1j])
        return answer

    with pytest.raises(AssertionError, match="additive"):
        assess(coupled)


@pytest.mark.parametrize("mutation", ["shape", "nan", "inf", "norm"])
def test_recovery_shape_nonfinite_and_norm_failures_are_not_repaired(mutation):
    assert runtime is not None, "E08 runtime probe is not implemented"

    def recover(data):
        answer = data["cos_state"] + data["sin_state"]
        if mutation == "shape":
            return answer.reshape(2, 1)
        if mutation == "nan":
            answer[0] = np.nan
        if mutation == "inf":
            answer[0] = np.inf
        if mutation == "norm":
            answer *= 0.5
        return answer

    with pytest.raises(AssertionError):
        assess(recover)


def test_recovery_uses_the_oracle_to_reject_nonfinite_or_nonreal_tolerance():
    assert runtime is not None, "E08 runtime probe is not implemented"
    for tolerance in (np.inf, "1e-3"):
        with pytest.raises(ValueError):
            assess(linear, tolerance=tolerance)


def validate(bound, actual=EXPECTED, *, tolerance=1e-9):
    return runtime._assess_validation(bound, {"bundle": object()},
                                      actual,
                                      tolerance=tolerance)


def test_boolean_validator_always_accepting_cannot_pass_bad_state_control():
    assert runtime is not None, "E08 runtime probe is not implemented"
    bound = Bound(linear, validate=lambda data: True)
    with pytest.raises(AssertionError, match="bad state"):
        validate(bound)


def test_boolean_validator_always_rejecting_cannot_pass_valid_state_control():
    assert runtime is not None, "E08 runtime probe is not implemented"
    bound = Bound(linear, validate=lambda data: False)
    with pytest.raises(AssertionError, match="valid state"):
        validate(bound)


def test_boolean_validator_accepts_numpy_booleans_but_not_truthy_integers():
    assert runtime is not None, "E08 runtime probe is not implemented"
    bound = Bound(linear,
                  validate=lambda data: np.bool_(
                      np.allclose(data["actual_state"], EXPECTED)))
    assert validate(bound)["bad_rejected"] is True
    with pytest.raises(runtime.AdapterUnavailable):
        validate(Bound(linear, validate=lambda data: 1))


def test_ray_validation_mutation_changes_a_relative_phase_not_only_global_phase(
):
    assert runtime is not None, "E08 runtime probe is not implemented"

    def ray_accepts_expected(data):
        actual = np.asarray(data["actual_state"])
        overlap = np.vdot(EXPECTED, actual)
        phase = overlap / abs(overlap) if overlap else 1
        return np.linalg.norm(actual - phase * EXPECTED) < 1e-10

    bound = Bound(linear, validate=ray_accepts_expected, phase_policy="ray")
    evidence = validate(bound)
    assert evidence["wrong_selected_error"] > 0.5
    assert evidence["bad_rejected"] is True


def test_raises_validator_nonzero_system_exit_rejects_bad_state():
    assert runtime is not None, "E08 runtime probe is not implemented"

    def validator(data):
        if not np.array_equal(data["actual_state"], EXPECTED):
            raise SystemExit(2)

    bound = Bound(linear, validate=validator, validation_mode="raises")
    assert validate(bound)["bad_rejected"] is True


def test_raises_validator_zero_system_exit_accepts_and_cannot_reject_bad_state(
):
    assert runtime is not None, "E08 runtime probe is not implemented"
    bound = Bound(linear,
                  validate=lambda data: (_ for _ in ()).throw(SystemExit(0)),
                  validation_mode="raises")
    with pytest.raises(AssertionError, match="bad state"):
        validate(bound)


def test_raises_validator_zero_system_exit_accepts_valid_then_value_error_rejects_bad(
):
    assert runtime is not None, "E08 runtime probe is not implemented"
    calls = 0

    def validator(data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SystemExit(None)
        raise ValueError("wrong state")

    bound = Bound(linear, validate=validator, validation_mode="raises")
    assert validate(bound)["bad_rejected"] is True


@pytest.mark.parametrize("phase_policy", ["exact", "ray"])
def test_wrong_state_control_reports_unknown_for_a_single_amplitude_ray(
        phase_policy):
    assert runtime is not None, "E08 runtime probe is not implemented"
    with pytest.raises(runtime.AdapterUnavailable, match="ray-distinguishing"):
        validate(Bound(linear, phase_policy=phase_policy),
                 np.array([1, 0], complex))


@pytest.mark.parametrize("role", ["hamiltonian", "initial_state"])
def test_real_domain_uses_values_not_complex_dtype(role):
    assert runtime is not None, "E08 runtime probe is not implemented"
    domain = {"hamiltonian": "real", "initial_state": "real"}
    hamiltonian = np.array([[1, 0], [0, -1]], dtype=complex)
    state = np.array([1, 0], dtype=complex)
    assert runtime._domain_violations(domain, hamiltonian, state) == []
    if role == "hamiltonian":
        hamiltonian[0, 0] += 2e-12j
    else:
        state[0] += 2e-12j
    assert runtime._domain_violations(domain, hamiltonian, state) == [role]


def test_real_domain_allows_imaginary_roundoff_at_declared_threshold():
    assert runtime is not None, "E08 runtime probe is not implemented"
    domain = {"hamiltonian": "real", "initial_state": "real"}
    hamiltonian = np.array([[1 + 1e-12j, 0], [0, -1]], complex)
    state = np.array([1 + 1e-12j, 0], complex)
    assert runtime._domain_violations(domain, hamiltonian, state) == []


@pytest.mark.parametrize("stage", ["build", "recover", "validate"])
@pytest.mark.parametrize("error", [ValueError, TypeError, AssertionError])
def test_only_allowed_stage_local_artifact_errors_count_as_domain_rejection(
        stage, error):
    assert runtime is not None, "E08 runtime probe is not implemented"

    def reject():
        raise error("unsupported domain")

    evidence, result = runtime._call_domain_stage(stage, reject)
    assert result is None
    assert evidence == {"stage": stage, "type": error.__name__}


def test_nonzero_but_not_zero_system_exit_counts_as_domain_rejection():
    assert runtime is not None, "E08 runtime probe is not implemented"
    evidence, _ = runtime._call_domain_stage(
        "build", lambda: (_ for _ in ()).throw(SystemExit(3)))
    assert evidence == {"stage": "build", "type": "SystemExit"}
    with pytest.raises(SystemExit):
        runtime._call_domain_stage(
            "build", lambda: (_ for _ in ()).throw(SystemExit(0)))


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_runtime_failures_never_count_as_domain_rejection(error):
    assert runtime is not None, "E08 runtime probe is not implemented"
    with pytest.raises(error):
        runtime._call_domain_stage(
            "recover", lambda: (_ for _ in ()).throw(error("runtime")))


def test_binding_and_evaluator_guard_failures_never_count_as_domain_rejection(
):
    assert runtime is not None, "E08 runtime probe is not implemented"
    for error in (runtime.AdapterUnavailable("binding"),
                  runtime._EvaluatorInstrumentationFailure("extraction")):
        with pytest.raises(type(error)):
            runtime._call_domain_stage("build",
                                       lambda error=error:
                                       (_ for _ in ()).throw(error))


def test_absent_trace_and_unmatched_selected_kernels_are_unknown():
    assert runtime is not None, "E08 runtime probe is not implemented"
    selected = (object(), object())
    with pytest.raises(runtime.AdapterUnavailable):
        runtime._selected_calls(selected, [], {"state_prep": object()}, 1, 1.0)
    calls = [{"_kernel": object()}]
    with pytest.raises(runtime.AdapterUnavailable):
        runtime._selected_calls(selected, calls, {"state_prep": object()}, 1,
                                1.0)


def test_selected_qsvt_call_without_declared_state_preparation_is_a_failure():
    assert runtime is not None, "E08 runtime probe is not implemented"
    first, second = object(), object()
    calls = []
    for kernel in (first, second):
        calls.append({
            "_kernel": kernel,
            "kind": "kernel",
            "_state_prep": None,
            "num_system": 1,
            "num_ancilla": 1,
            "alpha": 1.0,
            "phases": [0.1],
            "convention": "qsvt",
            "walk_directions": [],
        })
    with pytest.raises(AssertionError, match="state preparation"):
        runtime._selected_calls((first, second), calls,
                                {"state_prep": object()}, 1, 1.0)


def test_domain_path_allows_wrong_reconstruction_to_reach_artifact_validation(
):
    assert runtime is not None, "E08 runtime probe is not implemented"
    wrong = np.array([0, 1], dtype=complex)
    bound = Bound(lambda data: wrong.copy(), validate=lambda data: False)
    actual = runtime._recover_once(bound, {"bundle": object()}, COS, SIN,
                                   *PHASES)
    np.testing.assert_array_equal(actual, wrong)
    evidence = runtime._domain_validate(bound, {"bundle": object()}, actual)
    assert evidence == {"stage": "validate", "type": "boolean_false"}


def test_domain_boolean_true_and_unsupported_type_are_not_rejections():
    assert runtime is not None, "E08 runtime probe is not implemented"
    with pytest.raises(AssertionError, match="did not reject"):
        runtime._domain_validate(Bound(linear, validate=lambda data: True), {},
                                 EXPECTED)
    with pytest.raises(runtime.AdapterUnavailable):
        runtime._domain_validate(Bound(linear, validate=lambda data: "false"),
                                 {}, EXPECTED)
