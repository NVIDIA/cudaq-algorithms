# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E05 worker-targeted preparation checks for the isolated native runner.

``E05_CI_ADAPTER`` is an evaluator-owned explicit JSON binding and
``E05_ARTIFACT_ROOT`` is the captured worker artifact.  This suite never
discovers or repairs an artifact API.  Adapter and geometry limitations are
UNKNOWN coverage (pytest skips); valid-input artifact faults remain failures.
"""
from copy import deepcopy
import json
import os
from importlib.metadata import version
from pathlib import Path

import cudaq
import numpy as np
import pytest

import ci_runtime as runtime
from ci_adapter import bind
from ci_oracle import case_inputs, expected_state
from scale_adapter import AdapterUnavailable, verify_origin


@pytest.fixture(scope="module", autouse=True)
def pinned_runtime():
    assert version("cudaq") == "0.15.1", "requires frozen CUDA-Q 0.15.1"
    cudaq.set_target("qpp-cpu")
    assert "fp64" in str(cudaq.get_target())
    yield
    cudaq.reset_target()


@pytest.fixture(scope="module")
def binding():
    path = Path(os.environ["E05_CI_ADAPTER"])
    artifact_root = os.environ["E05_ARTIFACT_ROOT"]
    raw = path.read_bytes()
    assert len(raw) <= 8192, "binding is oversized"
    declaration = json.loads(raw)

    # These imports, provenance checks, and the bound artifact imports below
    # occur only in the isolated runtime that selects this suite.
    from cudaq_algorithms import PauliLCU, Walk
    verify_origin(PauliLCU, artifact_root,
                  "python/cudaq_algorithms/pauli_lcu.py")
    verify_origin(Walk, artifact_root,
                  "python/cudaq_algorithms/qubitization.py")
    return {
        "invoke": bind(declaration, artifact_root=artifact_root),
        "representation": declaration["representation"],
        "policy": declaration["coefficient_policy"],
    }


def _positive_data(basis_kind, complex_weights, *, scale=1.0):
    if basis_kind == "dense":
        basis = np.array(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
            dtype=float) / 2
        occupations = [(0, 1), (0, 2), (1, 3), (2, 3)]
    elif basis_kind == "localized":
        angle = .37
        basis = np.diag([-1., 1., 1., 1.])
        basis[2:, 2:] = [[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]]
        occupations = [(0, 2), (0, 3), (1, 2), (1, 3)]
    else:  # Evaluator-owned fixture programming error, never artifact behavior.
        raise ValueError("unknown positive basis fixture")
    weights = np.array([1, 2j, -3, -4j] if complex_weights else [1, -2, 3, -4],
                       dtype=complex) / np.sqrt(30)
    weights *= scale
    return (case_inputs(basis, occupations,
                        weights), expected_state(basis, occupations, weights))


def _preflight_or_unknown(invoke, data):
    try:
        invoke.preflight(data)
    except AdapterUnavailable as exc:
        pytest.skip(f"declared E05 binding is unavailable: {exc}")


def _probe_or_unknown(invoke, data, expected):
    _preflight_or_unknown(invoke, data)
    try:
        return runtime.probe_preparation(invoke, data, expected)
    except AdapterUnavailable as exc:
        pytest.skip(f"declared E05 geometry is unsupported: {exc}")


def _assert_direct_invalid(invoke, data):
    """Require an ordinary artifact validation error without pre-validating data."""
    _preflight_or_unknown(invoke, data)
    try:
        invoke(deepcopy(data))
    except AdapterUnavailable as exc:
        pytest.skip(f"declared E05 interface is unavailable: {exc}")
    except (ValueError, TypeError):
        return
    pytest.fail("artifact accepted a genuinely invalid E05 input")


@pytest.mark.parametrize("basis_kind", ["dense", "localized"])
@pytest.mark.parametrize("complex_weights", [False, True],
                         ids=["real-coefficients", "complex-coefficients"])
def test_preparation_and_walk_match_independent_reference(
        binding, basis_kind, complex_weights):
    data, expected = _positive_data(basis_kind, complex_weights)
    metrics = _probe_or_unknown(binding["invoke"], data, expected)
    assert metrics["preparation"]["fidelity"] > 1 - 1e-12
    assert metrics["zero_step"]["fidelity"] > 1 - 1e-12
    assert metrics["roundtrip"]["fidelity"] > 1 - 1e-12
    assert metrics["walk_block_max_error"] < 1e-10
    assert metrics["sample_counts"] == {
        "preparation": 32,
        "zero_step": 32,
        "roundtrip": 32,
        "one_step": 32,
    }


def test_scaled_nonzero_coefficients_follow_declared_policy(binding):
    data, expected = _positive_data("dense", True, scale=3.0)
    if binding["policy"] == "normalize":
        metrics = _probe_or_unknown(binding["invoke"], data, expected)
        assert metrics["preparation"]["fidelity"] > 1 - 1e-12
    else:
        assert binding["policy"] == "reject"
        _assert_direct_invalid(binding["invoke"], data)


@pytest.mark.parametrize("invalid", ["count", "shape", "zero", "nonfinite"])
def test_invalid_coefficients_reach_artifact_and_are_rejected(
        binding, invalid):
    data, _ = _positive_data("dense", True)
    if invalid == "count":
        data["coefficients"] = data["coefficients"][:3]
    elif invalid == "shape":
        data["coefficients"] = data["coefficients"].reshape(2, 2)
    elif invalid == "zero":
        data["coefficients"] = np.zeros(4, dtype=complex)
    else:
        for value in (np.nan, np.inf, -np.inf):
            malformed = deepcopy(data)
            malformed["coefficients"][0] = value
            _assert_direct_invalid(binding["invoke"], malformed)
        return
    _assert_direct_invalid(binding["invoke"], data)


@pytest.mark.parametrize("invalid",
                         range(4),
                         ids=[
                             "negative-index-or-rank",
                             "outside-index-or-row-shape",
                             "duplicate-index-or-column-shape",
                             "noninteger-index-or-nonfinite",
                         ])
def test_representation_specific_invalid_data_reach_artifact_and_are_rejected(
        binding, invalid):
    data, _ = _positive_data("localized", False)
    if binding["representation"] == "basis_occupations":
        replacements = [(-1, 2), (0, 4), (0, 0), (0, 1.5)]
        data["occupations"][0] = replacements[invalid]
    else:
        assert binding["representation"] == "occupied_matrices"
        matrix = data["orbital_matrices"][0]
        replacements = [
            matrix.ravel(), matrix[:3, :], matrix[:, :0],
            matrix.copy()
        ]
        if invalid == 3:
            replacements[invalid][0, 0] = np.nan
        data["orbital_matrices"][0] = replacements[invalid]
    _assert_direct_invalid(binding["invoke"], data)
