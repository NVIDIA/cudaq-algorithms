# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E08 worker-targeted checks for an explicitly bound evolution example.

The evaluator owns the small dense references and the state preparations.
``E08_EVOLUTION_ADAPTER`` only maps captured artifact APIs; unavailable
mappings/geometries are UNKNOWN coverage, while artifact and numerical faults
remain failures.  Source/documentation review and entrypoint-validation use
are separate obligations outside this numerical suite.
"""
import json
import os
from importlib.metadata import version
from pathlib import Path

import cudaq
import numpy as np
import pytest

import evolution_runtime as runtime
from evolution_adapter import bind
from scale_adapter import AdapterUnavailable, verify_origin

DEGREE = 16
TOLERANCE = 1e-8
BASE_TERMS = {"II": .17, "ZI": .6, "XX": .8}
SIGNED_TERMS = {"II": .23, "IZ": -.7, "XX": .41, "YY": .19}
COMPLEX_H_TERMS = {**BASE_TERMS, "YI": .25}
PREPARATIONS = {
    "A": (.72, -.38),
    "B": (-.46, 1.02),
}

_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}


@cudaq.kernel
def preparation_a(qubits: cudaq.qview):
    ry(.72, qubits[0])
    ry(-.38, qubits[1])
    rz(0., qubits[0])


@cudaq.kernel
def preparation_b(qubits: cudaq.qview):
    ry(-.46, qubits[0])
    ry(1.02, qubits[1])
    rz(0., qubits[0])


@cudaq.kernel
def preparation_a_complex(qubits: cudaq.qview):
    ry(.72, qubits[0])
    ry(-.38, qubits[1])
    rz(.44, qubits[0])


_PREPARATION_KERNELS = {
    "A": preparation_a,
    "B": preparation_b,
}


@pytest.fixture(scope="module", autouse=True)
def pinned_runtime():
    assert version("cudaq") == "0.15.1", "requires frozen CUDA-Q 0.15.1"
    cudaq.set_target("qpp-cpu")
    assert "fp64" in str(cudaq.get_target())
    yield
    cudaq.reset_target()


@pytest.fixture(scope="module")
def binding():
    path = Path(os.environ["E08_EVOLUTION_ADAPTER"])
    artifact_root = os.environ["E08_ARTIFACT_ROOT"]
    raw = path.read_bytes()
    assert len(raw) <= 8192, "binding is oversized"
    declaration = json.loads(raw)

    # The runtime separately verifies QSVT and PhaseSequence while tracing.
    # PauliLCU is the remaining public class consumed by this bounded path.
    from cudaq_algorithms import PauliLCU
    try:
        verify_origin(PauliLCU, artifact_root,
                      "python/cudaq_algorithms/pauli_lcu.py")
        bound = bind(declaration, artifact_root=artifact_root)
    except AdapterUnavailable as exc:
        pytest.skip(f"declared E08 binding is unavailable: {exc}")
    return {"bound": bound, "artifact_root": artifact_root}


def _dense_hamiltonian(terms):
    """Evaluator-only Pauli sum with word position zero acting on q0/LSB."""
    width = len(next(iter(terms)))
    result = np.zeros((1 << width, 1 << width), dtype=np.complex128)
    for word, coefficient in terms.items():
        factor = np.array([[1]], dtype=np.complex128)
        for label in reversed(word):
            factor = np.kron(factor, _PAULI[label])
        result += coefficient * factor
    return result


def _product_state(angle0, angle1, phase):
    """Independent little-endian state for RY0, RY1, then RZ0."""
    q0 = np.array([
        np.cos(angle0 / 2) * np.exp(-.5j * phase),
        np.sin(angle0 / 2) * np.exp(.5j * phase),
    ],
                  dtype=np.complex128)
    q1 = np.array([np.cos(angle1 / 2), np.sin(angle1 / 2)],
                  dtype=np.complex128)
    return np.kron(q1, q0)


def _case(terms, preparation, time, *, phase=0.):
    angles = PREPARATIONS[preparation]
    kernel = (preparation_a_complex
              if phase else _PREPARATION_KERNELS[preparation])
    return {
        "terms": dict(terms),
        "time": time,
        "state_prep": kernel,
        "initial_state": _product_state(*angles, phase),
        "degree": DEGREE,
        "tolerance": TOLERANCE,
    }


def _probe_or_unknown(binding, case, hamiltonian, *, reject_domain=False):
    try:
        return runtime.probe_example(binding["bound"],
                                     case,
                                     hamiltonian,
                                     artifact_root=binding["artifact_root"],
                                     expect_domain_rejection=reject_domain)
    except AdapterUnavailable as exc:
        pytest.skip(f"declared E08 example geometry is unsupported: {exc}")


def _assert_positive(metrics):
    assert metrics["domain_rejected"] is False
    assert metrics["domain_violations"] == []
    assert metrics["system_qubits"] == 2
    assert metrics["sample_counts"] == [32, 32]
    recovery = metrics["recovery"]
    if recovery["phase_policy"] == "exact":
        selected_error = recovery["raw_l2_error"]
    else:
        assert recovery["phase_policy"] == "ray"
        selected_error = recovery["aligned_l2_error"]
    assert selected_error <= TOLERANCE
    assert recovery["norm_error"] <= TOLERANCE
    assert metrics["validation"]["valid_accepted"] is True
    assert metrics["validation"]["bad_rejected"] is True


@pytest.mark.parametrize("time", [.5, -.5], ids=["forward", "backward"])
@pytest.mark.parametrize("preparation", ["A", "B"], ids=["prep-a", "prep-b"])
@pytest.mark.parametrize("terms", [BASE_TERMS, SIGNED_TERMS],
                         ids=["base-h", "signed-h"])
def test_real_evolution_matches_independent_reference(binding, terms,
                                                      preparation, time):
    case = _case(terms, preparation, time)
    metrics = _probe_or_unknown(binding, case, _dense_hamiltonian(terms))
    _assert_positive(metrics)


@pytest.mark.parametrize("time", [.5, -.5], ids=["forward", "backward"])
@pytest.mark.parametrize("domain_case", ["initial_state", "hamiltonian"],
                         ids=["complex-initial", "imaginary-hamiltonian"])
def test_declared_domain_is_enforced_or_complex_case_is_solved(
        binding, domain_case, time):
    if domain_case == "initial_state":
        terms = BASE_TERMS
        case = _case(terms, "A", time, phase=.44)
    else:
        terms = COMPLEX_H_TERMS
        case = _case(terms, "A", time)
    hamiltonian = _dense_hamiltonian(terms)
    rejects_domain = binding["bound"].domain[domain_case] == "real"
    metrics = _probe_or_unknown(binding,
                                case,
                                hamiltonian,
                                reject_domain=rejects_domain)

    if not rejects_domain:
        assert binding["bound"].domain[domain_case] == "complex"
        _assert_positive(metrics)
        return

    assert metrics["domain_rejected"] is True
    assert metrics["domain_violations"] == [domain_case]
    assert metrics["rejection"]["stage"] in {"build", "recover", "validate"}
    assert metrics["rejection"]["type"] in {
        "ValueError",
        "TypeError",
        "AssertionError",
        "SystemExit",
        "boolean_false",
    }
