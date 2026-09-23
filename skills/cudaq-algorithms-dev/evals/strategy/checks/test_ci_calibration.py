# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E05 native oracle calibration, run explicitly in the frozen isolated runtime.

Passing here certifies neither worker artifacts nor skill quality. The positive
control uses dense unitary synthesis, not the requested product Slater reuse.
"""
from importlib.metadata import version

import cudaq
import numpy as np
import pytest

import ci_calibration as calibration
import ci_runtime as runtime
from ci_adapter import bind
from ci_oracle import case_inputs, expected_state
from scale_adapter import AdapterUnavailable


@pytest.fixture(scope="module", autouse=True)
def pinned_runtime():
    assert version("cudaq") == "0.15.1"
    cudaq.set_target("qpp-cpu")
    assert "fp64" in str(cudaq.get_target())
    yield
    cudaq.reset_target()


def data(localized=True, complex_weights=True, scale=1):
    if localized:
        basis = np.diag([-1., 1., 1., 1.])
        basis[2:, 2:] = [[np.cos(.37), np.sin(.37)],
                         [-np.sin(.37), np.cos(.37)]]
        occupied = [(0, 2), (0, 3), (1, 2), (1, 3)]
    else:
        basis = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1],
                          [1, -1, -1, 1]]) / 2
        occupied = [(0, 1), (0, 2), (1, 3), (2, 3)]
    weights = np.array([1, 2j, -3, -4j] if complex_weights else [1, -2, 3, -4])
    weights = weights * scale / np.sqrt(30)
    return case_inputs(basis, occupied,
                       weights), expected_state(basis, occupied, weights)


def factory(name, policy="reject"):
    return bind({
        "schema_version": 1,
        "module": "ci_calibration",
        "attribute": name,
        "representation": "basis_occupations",
        "coefficient_policy": policy,
        "args": ["orbital_basis", "occupations", "coefficients"],
        "kwargs": {}
    })


def probe(invoke, inputs, expected):
    assert hasattr(
        runtime, "probe_preparation"), "native CI preparation probe is missing"
    return runtime.probe_preparation(invoke, inputs, expected)


@pytest.mark.parametrize("localized", [False, True])
@pytest.mark.parametrize("complex_weights", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_independent_positive_preparation_and_walk(localized, complex_weights,
                                                   normalize):
    inputs, expected = data(localized,
                            complex_weights,
                            scale=3 if normalize else 1)
    invoke = factory("positive_normalize" if normalize else "positive",
                     "normalize" if normalize else "reject")
    metrics = probe(invoke, inputs, expected)
    assert metrics["preparation"]["fidelity"] > 1 - 1e-12
    assert metrics["zero_step"]["fidelity"] > 1 - 1e-12
    assert metrics["roundtrip"]["fidelity"] > 1 - 1e-12
    assert metrics["walk_block_max_error"] < 1e-10
    assert metrics["sample_counts"] == {
        "preparation": 32,
        "zero_step": 32,
        "roundtrip": 32,
        "one_step": 32
    }
    print({
        "localized": localized,
        "complex_weights": complex_weights,
        "normalize": normalize,
        "metrics": metrics
    })


@pytest.mark.parametrize("name,reason", [
    ("wrong_relative_phase", "coherent amplitudes"),
    ("dropped_imaginary", "coherent amplitudes"),
])
def test_actual_defective_kernels_are_rejected(name, reason):
    inputs, expected = data()
    with pytest.raises(AssertionError, match=reason):
        probe(factory(name), inputs, expected)


def test_internal_allocation_needs_explicit_layout_not_an_invented_failure():
    # E05 permits a defined ancilla contract. This probe does not yet bind one.
    inputs, expected = data()
    with pytest.raises(AdapterUnavailable, match="allocation"):
        probe(factory("extra_ancilla"), inputs, expected)


def test_factory_cannot_require_state_extraction_and_guard_restores_api():
    inputs, expected = data()
    original = cudaq.get_state

    def requires_extraction(case):

        @cudaq.kernel
        def entry():
            system = cudaq.qvector(4)

        cudaq.get_state(entry)
        return calibration.positive(case["orbital_basis"], case["occupations"],
                                    case["coefficients"])

    with pytest.raises(AssertionError, match="state extraction"):
        probe(requires_extraction, inputs, expected)
    assert cudaq.get_state is original


@pytest.mark.parametrize("operation", ["measurement", "reset"])
def test_nonunitary_preparation_is_rejected_before_launch(operation):
    inputs, expected = data()

    def invalid(case):

        @cudaq.kernel
        def measured(qubits: cudaq.qview):
            mz(qubits[0])

        @cudaq.kernel
        def resetter(qubits: cudaq.qview):
            reset(qubits[0])

        return measured if operation == "measurement" else resetter

    with pytest.raises(AssertionError, match="unitary"):
        probe(invalid, inputs, expected)


def test_captured_nonunitary_kernel_is_not_hidden_from_inspection():
    inputs, expected = data()

    def invalid(case):

        @cudaq.kernel
        def helper(qubits: cudaq.qview):
            reset(qubits[0])

        @cudaq.kernel
        def outer(qubits: cudaq.qview):
            helper(qubits)

        return outer

    with pytest.raises(AssertionError, match="unitary"):
        probe(invalid, inputs, expected)


def test_captured_valid_kernel_is_not_rejected_for_using_a_helper():
    inputs, expected = data()

    def valid(case):
        helper = calibration.positive(case["orbital_basis"],
                                      case["occupations"],
                                      case["coefficients"])

        @cudaq.kernel
        def outer(qubits: cudaq.qview):
            helper(qubits)

        return outer

    metrics = probe(valid, inputs, expected)
    assert metrics["preparation"]["fidelity"] > 1 - 1e-12


def test_noise_instruction_cannot_pass_the_unitary_boundary():
    inputs, expected = data()

    def invalid(case):

        @cudaq.kernel
        def noisy(qubits: cudaq.qview):
            cudaq.apply_noise(cudaq.BitFlipChannel, 0.0, qubits[0])

        return noisy

    with pytest.raises(AssertionError, match="unitary"):
        probe(invalid, inputs, expected)
