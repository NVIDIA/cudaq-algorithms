# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded E08 public-QSVT runtime checks for use inside isolation only.

This module binds no artifact API and supplies no missing algorithm.  It traces
the canonical QSVT factories, extracts simulator states only in evaluator code,
and records bounded numerical and self-validation evidence.  Frozen source,
documentation, entrypoint use of validation, and full regression remain
controller responsibilities.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import inspect

import numpy as np

from evolution_oracle import assert_evolved_state, expected_state
from scale_adapter import AdapterUnavailable, verify_origin

DOMAIN_IMAG_TOLERANCE = 1e-12
MAX_TOTAL_QUBITS = 8
SHOTS = 32


class _EvaluatorInstrumentationFailure(AssertionError):
    """A failure deliberately injected by the evaluator's SDK guard."""


def _detached(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)


def _detached_case(case_data):
    if not isinstance(case_data, Mapping):
        raise ValueError("case_data must be a mapping")
    required = {
        "terms", "time", "state_prep", "initial_state", "degree", "tolerance"
    }
    if not required.issubset(case_data):
        raise ValueError("case_data is missing a required evolution role")
    result = {
        key: _detached(value)
        for key, value in case_data.items() if key != "state_prep"
    }
    # A CUDA-Q kernel is an opaque capability.  Copying it could change which
    # preparation the public QSVT call actually consumed.
    result["state_prep"] = case_data["state_prep"]
    return result


def _fresh_context(base):
    result = {}
    for key, value in base.items():
        if key in ("bundle", "state_prep"):
            result[key] = value
        else:
            result[key] = _detached(value)
    return result


def _domain_violations(domain, hamiltonian, initial_state):
    """Return declared-real roles whose *values* have material imaginary parts."""
    violations = []
    for role, value in (("hamiltonian", hamiltonian), ("initial_state",
                                                       initial_state)):
        if domain[role] == "real":
            try:
                imaginary = np.asarray(value, dtype=np.complex128).imag
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"{role} cannot be checked against its domain") from error
            if not np.isfinite(imaginary).all():
                raise ValueError(f"{role} has nonfinite imaginary values")
            if np.any(np.abs(imaginary) > DOMAIN_IMAG_TOLERANCE):
                violations.append(role)
    return violations


def _call_domain_stage(stage, call):
    """Call exactly one artifact stage and classify only explicit rejections."""
    try:
        return None, call()
    except (AdapterUnavailable, _EvaluatorInstrumentationFailure):
        raise
    except SystemExit as error:
        if error.code not in (None, 0):
            return {"stage": stage, "type": "SystemExit"}, None
        raise
    except (ValueError, TypeError, AssertionError) as error:
        return {"stage": stage, "type": type(error).__name__}, None


@contextmanager
def _without_extraction():
    """Detect ordinary SDK extraction use during artifact construction/run."""
    import cudaq
    from cudaq.runtime import state

    def forbidden(*args, **kwargs):
        raise _EvaluatorInstrumentationFailure(
            "artifact construction and sampling must not extract state")

    saved = [(module, name, getattr(module, name)) for module in (cudaq, state)
             for name in ("get_state", "get_state_async")]
    try:
        for module, name, _ in saved:
            setattr(module, name, forbidden)
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def _sequence_snapshot(sequence):
    return {
        "phases": [float(value) for value in sequence.phases],
        "convention": sequence.convention,
        "walk_directions": [int(value) for value in sequence.walk_directions],
    }


@contextmanager
def _trace_qsvt(artifact_root):
    """Trace the exact canonical public classes and restore them unconditionally."""
    import cudaq_algorithms.qsvt as qsvt_module

    relative = "python/cudaq_algorithms/qsvt.py"
    verify_origin(qsvt_module, artifact_root, relative)
    verify_origin(qsvt_module.PhaseSequence, artifact_root, relative)
    verify_origin(qsvt_module.QSVT, artifact_root, relative)

    phase_type = qsvt_module.PhaseSequence
    qsvt_type = qsvt_module.QSVT
    original_phase_init = phase_type.__init__
    original_kernel = qsvt_type.kernel
    original_controlled = qsvt_type.controlled_kernel
    phase_records = []
    calls = []

    def phase_init(self, phases, walk_directions=None, convention="qsvt"):
        # Materialize each caller iterable once, then give that same snapshot to
        # the real constructor.  This preserves one-shot generators exactly.
        phase_values = tuple(phases)
        direction_values = (None if walk_directions is None else
                            tuple(walk_directions))
        original_phase_init(self, phase_values, direction_values, convention)
        phase_records.append((self, _sequence_snapshot(self)))

    def traced(factory, kind, self, *args, **kwargs):
        signature = inspect.signature(factory)
        arguments = signature.bind(self, *args, **kwargs)
        arguments.apply_defaults()
        before = len(phase_records)
        result = factory(self, *args, **kwargs)
        supplied = arguments.arguments["sequence"]
        if isinstance(supplied, phase_type):
            snapshot = _sequence_snapshot(supplied)
        elif len(phase_records) == before + 1:
            snapshot = dict(phase_records[-1][1])
        else:
            raise _EvaluatorInstrumentationFailure(
                "could not identify the PhaseSequence consumed by QSVT")
        encoding = self.encoding
        calls.append({
            "_kernel": result,
            "_encoding": encoding,
            "_state_prep": arguments.arguments["state_prep"],
            "kind": kind,
            "num_system": getattr(encoding, "num_system", None),
            "num_ancilla": getattr(encoding, "num_ancilla", None),
            "alpha": getattr(encoding, "alpha", None),
            **snapshot,
        })
        return result

    def kernel(self, *args, **kwargs):
        return traced(original_kernel, "kernel", self, *args, **kwargs)

    def controlled(self, *args, **kwargs):
        return traced(original_controlled, "controlled_kernel", self, *args,
                      **kwargs)

    phase_type.__init__ = phase_init
    qsvt_type.kernel = kernel
    qsvt_type.controlled_kernel = controlled
    try:
        yield calls
    finally:
        phase_type.__init__ = original_phase_init
        qsvt_type.kernel = original_kernel
        qsvt_type.controlled_kernel = original_controlled


def _independent_alpha(terms, system_qubits):
    if isinstance(terms, Mapping):
        pairs = [(coefficient, word) for word, coefficient in terms.items()]
    elif isinstance(terms, (list, tuple)):
        pairs = list(terms)
    else:
        raise ValueError("terms must be a mapping or sequence")
    if not pairs:
        raise ValueError("terms must be nonempty")
    total = 0.0
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "each term must pair a coefficient and Pauli word")
        first, second = pair
        if isinstance(first, str):
            word, coefficient = first, second
        else:
            coefficient, word = first, second
        if not isinstance(word, str) or len(word) != system_qubits:
            raise ValueError("Pauli word width does not match the Hamiltonian")
        try:
            coefficient = complex(coefficient)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("term coefficient must be finite") from error
        if not np.isfinite(coefficient):
            raise ValueError("term coefficient must be finite")
        total += abs(coefficient)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("term one-norm must be finite and positive")
    return float(total)


def _selected_calls(selected, calls, case, system_qubits, alpha):
    if not calls:
        raise AdapterUnavailable("no public QSVT factory call was observed")
    matched = []
    for kernel in selected:
        candidates = [call for call in calls if call["_kernel"] is kernel]
        if len(candidates) != 1:
            raise AdapterUnavailable(
                "selected kernel does not match one recorded public QSVT call")
        matched.append(candidates[0])
    if matched[0] is matched[1]:
        raise AdapterUnavailable("two outputs must come from two QSVT calls")

    evidence = []
    for call in matched:
        if call["kind"] != "kernel":
            raise AdapterUnavailable("controlled QSVT layouts are unsupported")
        if call["_state_prep"] is None or call["_state_prep"] is not case[
                "state_prep"]:
            raise AssertionError(
                "QSVT output must consume the declared data-free state preparation"
            )
        if (type(call["num_system"]) is not int
                or type(call["num_ancilla"]) is not int):
            raise AdapterUnavailable("QSVT encoding geometry is unavailable")
        if call["num_system"] != system_qubits or call["num_ancilla"] < 1:
            raise AssertionError(
                "QSVT encoding geometry does not match the input")
        total = call["num_system"] + call["num_ancilla"]
        if total > MAX_TOTAL_QUBITS:
            raise AdapterUnavailable(
                "QSVT output exceeds the bounded total width")
        try:
            recorded_alpha = float(call["alpha"])
        except (TypeError, ValueError, OverflowError) as error:
            raise AdapterUnavailable(
                "QSVT encoding alpha is unavailable") from error
        if not np.isfinite(recorded_alpha) or abs(recorded_alpha -
                                                  alpha) > 1e-12:
            raise AssertionError(
                "QSVT encoding alpha does not match the input")
        evidence.append({
            "kind": call["kind"],
            "num_system": call["num_system"],
            "num_ancilla": call["num_ancilla"],
            "total_qubits": total,
            "alpha": recorded_alpha,
            "phases": list(call["phases"]),
            "convention": call["convention"],
            "walk_directions": list(call["walk_directions"]),
        })
    return matched, evidence


def _inspect_and_sample(selected, matched):
    import cudaq
    from ci_runtime import _inspect_preparation

    counts = []
    with _without_extraction():
        for kernel, call in zip(selected, matched):
            _inspect_preparation(call["_state_prep"])
            count = sum(cudaq.sample(kernel, shots_count=SHOTS).values())
            if count != SHOTS:
                raise AssertionError("incomplete QSVT sampling")
            counts.append(int(count))
    return counts


def _extract_components(selected, matched):
    import cudaq

    components, evidence = [], []
    for kernel, call in zip(selected, matched):
        try:
            full = np.asarray(cudaq.get_state(kernel), dtype=np.complex128)
        except (TypeError, ValueError, OverflowError) as error:
            raise AssertionError(
                "QSVT state extraction did not return a vector") from error
        size = 1 << (call["num_system"] + call["num_ancilla"])
        if full.shape != (size, ):
            raise AssertionError("QSVT state has the wrong register size")
        if not np.isfinite(full).all():
            raise AssertionError("QSVT state contains nonfinite values")
        full_norm = float(np.linalg.norm(full))
        if not np.isfinite(full_norm) or abs(full_norm - 1.0) > 1e-12:
            raise AssertionError("QSVT full state is not normalized")
        block = full[:1 << call["num_system"]].copy()
        components.append(block)
        evidence.append({
            "full_norm": full_norm,
            "block_norm": float(np.linalg.norm(block))
        })
    return components, evidence


def _recover_data(base, cosine, sine, cosine_phases, sine_phases):
    data = _fresh_context(base)
    data.update({
        "cos_state": np.array(cosine, dtype=np.complex128, copy=True),
        "sin_state": np.array(sine, dtype=np.complex128, copy=True),
        "cos_phases": list(cosine_phases),
        "sin_phases": list(sine_phases),
    })
    return data


def _recover_once(bound, base, cosine, sine, cosine_phases, sine_phases):
    return bound.recover(
        _recover_data(base, cosine, sine, cosine_phases, sine_phases))


def _recovered_array(value, shape):
    try:
        result = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssertionError(
            "recovery must return a complex vector") from error
    if result.shape != shape:
        raise AssertionError("recovery returned the wrong shape")
    if not np.isfinite(result).all():
        raise AssertionError("recovery returned nonfinite values")
    return result


def _l2_gate(name, actual, expected, tolerance):
    error = float(np.linalg.norm(actual - expected))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f"recovery failed the {name} control")
    return error


def _validated_tolerance(expected, tolerance, phase_policy):
    # Reuse the independent oracle's bounded scalar and policy validation.
    assert_evolved_state(expected,
                         expected,
                         atol=tolerance,
                         phase_policy=phase_policy)
    return float(tolerance)


def _assess_recovery(bound, base, cosine, sine, cosine_phases, sine_phases,
                     expected, *, tolerance):
    """Check bounded real homogeneity/additivity plus the independent oracle."""
    expected = np.asarray(expected, dtype=np.complex128)
    tolerance = _validated_tolerance(expected, tolerance, bound.phase_policy)
    shape = expected.shape
    good = _recovered_array(
        _recover_once(bound, base, cosine, sine, cosine_phases, sine_phases),
        shape)
    zero = _recovered_array(
        _recover_once(bound, base, np.zeros_like(cosine), np.zeros_like(sine),
                      cosine_phases, sine_phases), shape)
    half = _recovered_array(
        _recover_once(bound, base, .5 * cosine, .5 * sine, cosine_phases,
                      sine_phases), shape)
    negative = _recovered_array(
        _recover_once(bound, base, -cosine, -sine, cosine_phases, sine_phases),
        shape)
    cosine_only = _recovered_array(
        _recover_once(bound, base, cosine, np.zeros_like(sine), cosine_phases,
                      sine_phases), shape)
    sine_only = _recovered_array(
        _recover_once(bound, base, np.zeros_like(cosine), sine, cosine_phases,
                      sine_phases), shape)

    evidence = {
        "zero_l2_error":
        _l2_gate("zero", zero, np.zeros(shape, complex), tolerance),
        "half_l2_error":
        _l2_gate("half", half, .5 * good, tolerance),
        "negative_l2_error":
        _l2_gate("negative", negative, -good, tolerance),
        "additivity_l2_error":
        _l2_gate("additive component separation", cosine_only + sine_only,
                 good, tolerance),
    }
    evidence.update(
        assert_evolved_state(good,
                             expected,
                             atol=tolerance,
                             phase_policy=bound.phase_policy))
    return good, evidence


def _wrong_state(actual, *, tolerance, phase_policy):
    actual = np.asarray(actual, dtype=np.complex128)
    appreciable = np.flatnonzero(np.abs(actual) > DOMAIN_IMAG_TOLERANCE)
    for index in appreciable:
        wrong = actual.copy()
        wrong[index] *= -1
        overlap = np.vdot(actual, wrong)
        phase = overlap / abs(overlap) if overlap != 0 else 1.0 + 0.0j
        raw = float(np.linalg.norm(wrong - actual))
        aligned = float(np.linalg.norm(wrong - phase * actual))
        # The validator negative must be a different physical ray even when
        # the selected positive comparison policy is exact-vector.
        if not np.isfinite(aligned) or aligned <= tolerance:
            continue
        try:
            assert_evolved_state(wrong,
                                 actual,
                                 atol=tolerance,
                                 phase_policy=phase_policy)
        except AssertionError:
            return wrong, {
                "wrong_raw_l2_error":
                raw,
                "wrong_aligned_l2_error":
                aligned,
                "wrong_selected_error":
                raw if phase_policy == "exact" else aligned,
            }
    raise AdapterUnavailable(
        "no norm-preserving ray-distinguishing sign mutation is available")


def _validation_data(base, actual):
    data = _fresh_context(base)
    data["actual_state"] = np.array(actual, dtype=np.complex128, copy=True)
    return data


def _raises_validation(bound, data):
    try:
        bound.validate(data)
    except SystemExit as error:
        if error.code in (None, 0):
            return True
        return False
    except (AssertionError, ValueError):
        return False
    return True


def _boolean_validation(bound, data):
    result = bound.validate(data)
    if not isinstance(result, (bool, np.bool_)):
        raise AdapterUnavailable(
            "boolean validator returned an unsupported type")
    return bool(result)


def _assess_validation(bound, base, actual, *, tolerance):
    """Require artifact acceptance of good output and rejection of a wrong ray."""
    wrong, evidence = _wrong_state(actual,
                                   tolerance=tolerance,
                                   phase_policy=bound.phase_policy)
    invoke = (_boolean_validation
              if bound.validation_mode == "boolean" else _raises_validation)
    if not invoke(bound, _validation_data(base, actual)):
        raise AssertionError("artifact validator rejected the valid state")
    if invoke(bound, _validation_data(base, wrong)):
        raise AssertionError("artifact validator accepted the bad state")
    evidence.update({
        "mode": bound.validation_mode,
        "valid_accepted": True,
        "bad_rejected": True
    })
    return evidence


def _domain_validate(bound, base, actual):
    """Require explicit artifact validation rejection without using the oracle."""
    try:
        rejection, result = _call_domain_stage(
            "validate", lambda: bound.validate(_validation_data(base, actual)))
    except SystemExit as error:
        if error.code in (None, 0):
            raise AssertionError(
                "artifact validator did not reject domain input")
        raise
    if rejection is not None:
        return rejection
    if bound.validation_mode == "boolean":
        if not isinstance(result, (bool, np.bool_)):
            raise AdapterUnavailable(
                "boolean validator returned an unsupported type")
        if not bool(result):
            return {"stage": "validate", "type": "boolean_false"}
    raise AssertionError("artifact validator did not reject domain input")


def _preflight(bound, stage, data):
    # Preflight is binding inspection, not an artifact stage invocation.
    bound.preflight(stage, data)


def probe_example(bound,
                  case_data,
                  hamiltonian,
                  *,
                  artifact_root,
                  expect_domain_rejection=False):
    """Execute one explicitly bound two-kernel example inside isolation.

    The result is JSON-compatible evidence, not a certification verdict.  This
    first slice deliberately excludes controlled/wrapped layouts, nonlinear
    recovery, and four-run complex extensions.
    """
    case = _detached_case(case_data)
    hamiltonian_copy = np.array(hamiltonian, dtype=np.complex128, copy=True)
    initial_copy = np.array(case["initial_state"],
                            dtype=np.complex128,
                            copy=True)
    expected = expected_state(hamiltonian_copy, initial_copy, case["time"])
    _validated_tolerance(expected, case["tolerance"], bound.phase_policy)
    system_qubits = expected.size.bit_length() - 1
    if system_qubits > 4:
        raise ValueError("runtime probe supports at most four system qubits")
    alpha = _independent_alpha(case["terms"], system_qubits)
    violations = _domain_violations(bound.domain, hamiltonian_copy,
                                    initial_copy)
    if expect_domain_rejection and not violations:
        raise ValueError("domain rejection requires an out-of-domain input")
    if not expect_domain_rejection and violations:
        raise ValueError("input violates the artifact's declared domain")

    build_data = _fresh_context(case)
    with _trace_qsvt(artifact_root) as calls:
        with _without_extraction():
            _preflight(bound, "build", build_data)
            if expect_domain_rejection:
                rejection, bundle = _call_domain_stage(
                    "build", lambda: bound.build(_fresh_context(build_data)))
                if rejection is not None:
                    return {
                        "domain_rejected": True,
                        "domain_violations": violations,
                        "rejection": rejection
                    }
            else:
                bundle = bound.build(_fresh_context(build_data))

    selected = bound.kernels(bundle)
    if not isinstance(selected, tuple) or len(selected) != 2:
        raise AdapterUnavailable(
            "binding must select exactly two QSVT kernels")
    matched, call_evidence = _selected_calls(selected, calls, case,
                                             system_qubits, alpha)
    sample_counts = _inspect_and_sample(selected, matched)
    components, component_evidence = _extract_components(selected, matched)
    cosine, sine = components
    cosine_phases = tuple(matched[0]["phases"])
    sine_phases = tuple(matched[1]["phases"])
    base = _fresh_context(case)
    base["bundle"] = bundle

    _preflight(bound, "recover",
               _recover_data(base, cosine, sine, cosine_phases, sine_phases))
    if expect_domain_rejection:
        rejection, actual = _call_domain_stage(
            "recover", lambda: _recover_once(bound, base, cosine, sine,
                                             cosine_phases, sine_phases))
        if rejection is not None:
            return {
                "domain_rejected": True,
                "domain_violations": violations,
                "rejection": rejection
            }
        # Deliberately do not compare this output with the evaluator oracle.
        # A numerically wrong reconstruction is evidence only if the mapped
        # artifact validator itself rejects it explicitly.
        _preflight(bound, "validate", _validation_data(base, actual))
        rejection = _domain_validate(bound, base, actual)
        return {
            "domain_rejected": True,
            "domain_violations": violations,
            "rejection": rejection
        }

    actual, recovery = _assess_recovery(bound,
                                        base,
                                        cosine,
                                        sine,
                                        cosine_phases,
                                        sine_phases,
                                        expected,
                                        tolerance=case["tolerance"])
    _preflight(bound, "validate", _validation_data(base, actual))
    validation = _assess_validation(bound,
                                    base,
                                    actual,
                                    tolerance=case["tolerance"])
    return {
        "domain_rejected": False,
        "domain_violations": [],
        "system_qubits": system_qubits,
        "qsvt_calls": call_evidence,
        "sample_counts": sample_counts,
        "components": component_evidence,
        "recovery": recovery,
        "validation": validation,
    }
