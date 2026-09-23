# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime calibration, run explicitly before freezing E02 acceptance counts.

These tests calibrate the oracle; their passing does NOT certify worker code.
"""
from importlib.metadata import version

import numpy as np
import pytest

from scale_adapter import bind
from scale_calibration import mutant
from scale_oracle import (CASES, FACTORS, assert_consumers, assert_controlled,
                          assert_inverse, assert_scaled_block, full_matrix,
                          make_input, assert_walk_block, prepared_walk_block)


@pytest.fixture(scope="module", autouse=True)
def runtime():
    import cudaq
    assert version("cudaq") == "0.15.1", "requires frozen CUDA-Q 0.15.1"
    cudaq.set_target("qpp-cpu")
    assert "fp64" in str(cudaq.get_target())
    yield
    cudaq.reset_target()


def positive(policy):
    declaration = {
        "schema_version": 1,
        "kind": "callable",
        "module": "scale_calibration",
        "attribute": policy,
        "zero_policy": "encode"
    }
    declaration.update(
        args=["encoding", "factor"] if policy == "compact" else [],
        kwargs={} if policy == "compact" else {
            "base": "encoding",
            "multiplier": "factor"
        })
    return bind(declaration)


@pytest.mark.parametrize("policy", ["compact", "padded"])
@pytest.mark.parametrize("factor", (*FACTORS, 0.))
@pytest.mark.parametrize("case", CASES, ids=["one-qubit", "two-qubit"])
def test_positive_policies_against_literal_reference(policy, factor, case):
    original = make_input(case, foreign=True)
    scaled = positive(policy)(original, factor)
    dim = len(case[1])
    unitary = full_matrix(scaled, "apply_kernel")
    assert_scaled_block(unitary[:dim, :dim], case[1], factor, scaled.alpha)
    assert_controlled(full_matrix(scaled, "controlled_apply_kernel"), unitary,
                      scaled.num_system)
    forward = full_matrix(scaled, "walk_step_kernel")
    adjoint = full_matrix(scaled, "adjoint_walk_step_kernel")
    assert_inverse(forward, adjoint)
    assert_controlled(full_matrix(scaled, "controlled_walk_step_kernel"),
                      forward, scaled.num_system)
    assert_controlled(
        full_matrix(scaled, "controlled_adjoint_walk_step_kernel"), adjoint,
        scaled.num_system)
    assert_walk_block(prepared_walk_block(scaled), factor * case[1],
                      scaled.alpha)
    assert_consumers(scaled, factor * case[1])


@pytest.mark.parametrize(
    "name", ["ignore_factor", "abs_sign", "alpha_only", "ordering"])
def test_operator_mutants_are_rejected(name):
    case, factor = CASES[1], -.7
    scaled = mutant(make_input(case, foreign=True), factor, name)
    with pytest.raises(AssertionError):
        assert_scaled_block(
            full_matrix(scaled, "apply_kernel")[:4, :4], case[1], factor,
            scaled.alpha)


def test_relative_control_phase_mutant_is_rejected():
    scaled = mutant(make_input(CASES[0]), -.7, "control_phase")
    with pytest.raises(AssertionError):
        assert_controlled(full_matrix(scaled, "controlled_apply_kernel"),
                          full_matrix(scaled, "apply_kernel"),
                          scaled.num_system)


def test_adjoint_mutant_is_rejected():
    scaled = mutant(make_input(CASES[0]), -.7, "adjoint")
    with pytest.raises(AssertionError):
        assert_inverse(full_matrix(scaled, "walk_step_kernel"),
                       full_matrix(scaled, "adjoint_walk_step_kernel"))


def test_nonfinite_acceptance_mutant_violates_rejection_predicate():
    from scale_oracle import assert_invalid_rejected
    original = make_input(CASES[0])
    for factor in (np.nan, np.inf, -np.inf, 1 + 1j):
        with pytest.raises(AssertionError):
            assert_invalid_rejected(
                lambda enc, value: mutant(enc, value, "nonfinite_acceptance"),
                original, factor)


def test_zero_division_and_builtin_only_controls_are_not_positive():
    original = make_input(CASES[0], foreign=True)
    with pytest.raises(ZeroDivisionError):
        mutant(original, 0., "zero_division")
    with pytest.raises(TypeError):
        mutant(original, 2., "builtin_only")


def test_coherent_walk_sign_mutant_is_rejected_despite_consistent_inverses():
    case, factor = CASES[0], -.7
    scaled = mutant(make_input(case), factor, "walk_sign")
    forward = full_matrix(scaled, "walk_step_kernel")
    adjoint = full_matrix(scaled, "adjoint_walk_step_kernel")
    assert_inverse(forward, adjoint)
    assert_controlled(full_matrix(scaled, "controlled_walk_step_kernel"),
                      forward, scaled.num_system)
    assert_controlled(
        full_matrix(scaled, "controlled_adjoint_walk_step_kernel"), adjoint,
        scaled.num_system)
    with pytest.raises(AssertionError):
        assert_walk_block(prepared_walk_block(scaled), factor * case[1],
                          scaled.alpha)


def test_bad_subspace_phase_mutant_is_rejected_at_degree_two():
    import cudaq
    from cudaq_algorithms import QSVT, PhaseSequence
    from scale_oracle import assert_qsvt_output
    case, factor = CASES[0], -.7
    scaled = mutant(make_input(case), factor, "bad_subspace_phase")
    unitary = full_matrix(scaled, "apply_kernel")
    assert_scaled_block(unitary[:2, :2], case[1], factor, scaled.alpha)
    assert_controlled(full_matrix(scaled, "controlled_apply_kernel"), unitary,
                      scaled.num_system)
    ket = np.array([1., 2. + 1j], complex) / np.sqrt(6.)
    transformer = QSVT(scaled)
    output = np.asarray(
        cudaq.get_state(transformer.kernel(PhaseSequence([0., 0.])),
                        cudaq.State.from_data(ket)), complex)[:2]
    assert_qsvt_output(output, factor * case[1], scaled.alpha, ket, degree=1)
    output = np.asarray(
        cudaq.get_state(transformer.kernel(PhaseSequence([0., 0., 0.])),
                        cudaq.State.from_data(ket)), complex)[:2]
    with pytest.raises(AssertionError):
        assert_qsvt_output(output,
                           factor * case[1],
                           scaled.alpha,
                           ket,
                           degree=2)
