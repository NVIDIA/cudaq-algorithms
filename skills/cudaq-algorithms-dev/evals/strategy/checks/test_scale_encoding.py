# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E02 targeted checks, deliberately NOT selected by the existing runner.

E02_SCALE_ADAPTER names a trusted JSON binding mounted by the evaluator.
No bindings are guessed. Missing binding/runtime and documented zero rejection
are setup errors (unknown coverage), never skips or scientific passes.
"""
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from scale_adapter import bind, verify_origin
from scale_oracle import (CASES, FACTORS, assert_controlled, assert_inverse,
                          assert_scaled_block, full_matrix, make_input,
                          geometry, ProbeLimit)


@pytest.fixture(scope="module", autouse=True)
def runtime():
    import cudaq
    assert version("cudaq") == "0.15.1", "requires frozen CUDA-Q 0.15.1"
    cudaq.set_target("qpp-cpu")
    assert "fp64" in str(cudaq.get_target())
    yield
    cudaq.reset_target()


@pytest.fixture(scope="module")
def binding():
    path = Path(os.environ["E02_SCALE_ADAPTER"])
    raw = path.read_bytes()
    assert len(raw) <= 8192, "binding is oversized"
    declaration = json.loads(raw)
    artifact_root = os.environ.get("E02_ARTIFACT_ROOT")
    if artifact_root is not None:
        from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT, Walk
        verify_origin(PauliLCU, artifact_root,
                      "python/cudaq_algorithms/pauli_lcu.py")
        verify_origin(Walk, artifact_root,
                      "python/cudaq_algorithms/qubitization.py")
        verify_origin(QSVT, artifact_root, "python/cudaq_algorithms/qsvt.py")
        verify_origin(PhaseSequence, artifact_root,
                      "python/cudaq_algorithms/qsvt.py")
    return (bind(declaration,
                 artifact_root=artifact_root), declaration["zero_policy"])


@pytest.fixture(params=range(2), ids=["one-qubit", "two-qubit"])
def case(request):
    return CASES[request.param]


@pytest.fixture(params=[False, True], ids=["builtin", "foreign"])
def original(case, request, binding):
    encoding = make_input(case, foreign=request.param)
    binding[0].preflight(
        encoding
    )  # Missing/mismapped API is a setup error, not a numerical failure.
    return encoding


def bounded_attempt(invoke, original, factor):
    try:
        output = invoke(original, factor)
        geometry(output, controlled=True)
        return output
    except ProbeLimit:
        raise  # A fixture setup error: bounded coverage is unknown.
    except Exception as exc:
        return exc  # Artifact faults are asserted inside tests, not hidden as setup errors.


@pytest.fixture
def scaled(request, binding, original):
    return request.param, bounded_attempt(binding[0], original, request.param)


@pytest.mark.parametrize("scaled", FACTORS, indirect=True)
def test_full_block_and_complex_linearity(scaled, original, case):
    factor, scaled = scaled
    assert not isinstance(scaled, Exception), repr(scaled)
    assert scaled.num_system == original.num_system, "system width changed"
    dim = len(case[1])
    block = full_matrix(scaled, "apply_kernel")[:dim, :dim]
    assert_scaled_block(block, case[1], factor, scaled.alpha)
    # Probe a genuine coherent input as well as matrix columns.
    ket = np.arange(1, dim + 1) + 1j * np.arange(dim, 0, -1)
    ket = ket / np.linalg.norm(ket)
    from scale_oracle import apply_state
    state = np.zeros(1 << (scaled.num_system + scaled.num_ancilla), complex)
    state[:dim] = ket
    actual = apply_state(scaled, "apply_kernel", state)[:dim]
    np.testing.assert_allclose(actual,
                               factor * case[1] @ ket / scaled.alpha,
                               atol=1e-11,
                               rtol=1e-11)


@pytest.mark.parametrize("scaled", [-.7, 2.], indirect=True)
def test_controls_and_adjoint(scaled):
    _, scaled = scaled
    assert not isinstance(scaled, Exception), repr(scaled)
    apply = full_matrix(scaled, "apply_kernel")
    assert_controlled(full_matrix(scaled, "controlled_apply_kernel"), apply,
                      scaled.num_system)
    forward = full_matrix(scaled, "walk_step_kernel")
    adjoint = full_matrix(scaled, "adjoint_walk_step_kernel")
    assert_inverse(forward, adjoint)
    controlled = full_matrix(scaled, "controlled_walk_step_kernel")
    controlled_adjoint = full_matrix(scaled,
                                     "controlled_adjoint_walk_step_kernel")
    assert_controlled(controlled, forward, scaled.num_system)
    assert_controlled(controlled_adjoint, adjoint, scaled.num_system)
    assert_inverse(controlled, controlled_adjoint)


@pytest.mark.parametrize("scaled", [-.7, 2.], indirect=True)
def test_prepared_walk_has_correct_coherent_sign(scaled, case):
    from scale_oracle import assert_walk_block, prepared_walk_block
    factor, scaled = scaled
    assert not isinstance(scaled, Exception), repr(scaled)
    assert_walk_block(prepared_walk_block(scaled), factor * case[1],
                      scaled.alpha)


@pytest.mark.parametrize("scaled", [-.7, 2.], indirect=True)
def test_walk_and_qsvt_consumers(scaled, case):
    from scale_oracle import assert_consumers
    factor, scaled = scaled
    assert not isinstance(scaled, Exception), repr(scaled)
    assert_consumers(scaled, factor * case[1])


@pytest.fixture
def zero_attempt(binding, original):
    from scale_oracle import zero_attempt as attempt
    output = attempt(binding[0], original, binding[1])
    if not isinstance(output, Exception):
        return bounded_attempt(lambda enc, factor: output, original, 0.)
    return output


def test_zero_without_division_or_downstream_failure(zero_attempt, original,
                                                     case):
    from scale_oracle import assert_consumers
    assert not isinstance(zero_attempt, Exception), repr(zero_attempt)
    assert zero_attempt.num_system == original.num_system
    dim = len(case[1])
    block = full_matrix(zero_attempt, "apply_kernel")[:dim, :dim]
    assert_scaled_block(block, case[1], 0., zero_attempt.alpha)
    assert_consumers(zero_attempt, np.zeros_like(case[1]))


@pytest.mark.parametrize(
    "factor", [float("nan"), float("inf"), -float("inf"), 1 + 1j])
def test_nonfinite_and_nonreal_factors_rejected(binding, original, factor):
    from scale_oracle import assert_invalid_rejected
    assert_invalid_rejected(binding[0], original, factor)
