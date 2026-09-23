# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent oracle calibration without CUDA-Q; runtime controls are separate."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_oracle():
    path = Path(__file__).resolve().parents[1] / "checks/scale_oracle.py"
    assert path.is_file(), "E02 matrix oracle is not implemented"
    spec = importlib.util.spec_from_file_location("scale_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = np.array([[0.3, 0.2 - 0.4j], [0.2 + 0.4j, -0.1]])


@pytest.mark.parametrize("factor,alpha", [(2., 2.), (-.7, .7), (.25, 2.),
                                          (-.7, 2.), (0., 1.)])
def test_valid_normalization_policies_and_zero(factor, alpha):
    oracle = load_oracle()
    oracle.assert_scaled_block(factor * H / alpha, H, factor, alpha)


@pytest.mark.parametrize(
    "mutant",
    ["unchanged", "abs_sign", "transpose", "huge_alpha", "nan", "shape"])
def test_wrong_operator_cannot_pass_by_normalization_or_phase(mutant):
    oracle = load_oracle()
    factor, alpha = -.7, 1.
    block = factor * H / alpha
    if mutant == "unchanged":
        block = H
    elif mutant == "abs_sign":
        block = abs(factor) * H
    elif mutant == "transpose":
        block = block.T
    elif mutant == "huge_alpha":
        alpha, block = 1e30, np.zeros_like(H)
    elif mutant == "nan":
        block[0, 0] = np.nan
    else:
        block = block.ravel()
    with pytest.raises(AssertionError):
        oracle.assert_scaled_block(block, H, factor, alpha)


@pytest.mark.parametrize("alpha", [0., -1., np.nan, np.inf])
def test_unusable_normalization_is_failure(alpha):
    oracle = load_oracle()
    with pytest.raises(AssertionError):
        oracle.assert_scaled_block(H, H, 1., alpha)


def test_combined_control_register_order_and_relative_phase():
    oracle = load_oracle()
    # system bit 0, control bit 1, ancilla bit 2; U flips system.
    unitary = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex)
    control = np.eye(8, dtype=complex)
    enabled = [2, 3, 6, 7]
    control[np.ix_(enabled, enabled)] = unitary
    oracle.assert_controlled(control, unitary, 1)
    wrong = control.copy()
    wrong[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])] *= -1
    with pytest.raises(AssertionError):
        oracle.assert_controlled(wrong, unitary, 1)


def test_adjoint_is_inverse_not_second_forward_step():
    oracle = load_oracle()
    forward = np.array([[1, 0], [0, 1j]])
    oracle.assert_inverse(forward, forward.conj().T)
    with pytest.raises(AssertionError):
        oracle.assert_inverse(forward, forward)


def test_probe_geometry_bounds_do_not_assume_builtin_ancillas():
    oracle = load_oracle()

    class Encoding:
        num_system, num_ancilla = 2, 4

    assert oracle.geometry(Encoding(), controlled=True) == (2, 4, 128)
    Encoding.num_ancilla = 20
    with pytest.raises(oracle.ProbeLimit):
        oracle.geometry(Encoding())


def test_calibration_policies_have_independent_literal_pauli_coefficients():
    path = Path(__file__).resolve().parents[1] / "checks/scale_calibration.py"
    assert path.is_file(), "scale calibration controls are not implemented"
    spec = importlib.util.spec_from_file_location("scale_calibration", path)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    compact = calibration.pauli_terms(H, -2., padding=0.)
    padded = calibration.pauli_terms(H, -2., padding=.5)
    assert dict((word, coeff) for coeff, word in compact) == pytest.approx({
        "I":
        -.2,
        "X":
        -.4,
        "Y":
        -.8,
        "Z":
        -.4
    })
    assert sum(abs(coeff) for coeff, _ in compact) == pytest.approx(1.8)
    assert sum(abs(coeff) for coeff, _ in padded) == pytest.approx(2.8)
    aggregate = {}
    for coeff, word in padded:
        aggregate[word] = aggregate.get(word, 0.) + coeff
    assert aggregate == pytest.approx({"I": -.2, "X": -.4, "Y": -.8, "Z": -.4})


def test_invalid_acceptance_is_a_failure_and_zero_rejection_is_unknown():
    oracle = load_oracle()
    with pytest.raises(AssertionError):
        oracle.assert_invalid_rejected(lambda enc, factor: enc, object(),
                                       np.nan)

    def reject(enc, factor):
        raise ValueError("unsupported")

    oracle.assert_invalid_rejected(reject, object(), np.inf)
    with pytest.raises(oracle.ZeroUnsupported):
        oracle.zero_attempt(reject, object(), "reject")
    assert isinstance(oracle.zero_attempt(reject, object(), "encode"),
                      ValueError)
    assert isinstance(
        oracle.zero_attempt(lambda enc, factor: 1 / factor, object(),
                            "encode"), ZeroDivisionError)


def test_prepared_walk_block_has_signed_operator_not_global_phase_freedom():
    oracle = load_oracle()
    oracle.assert_walk_block(-H, H, 1.)
    with pytest.raises(AssertionError):
        oracle.assert_walk_block(H, H, 1.)


def test_degree_two_qsvt_rejects_bad_subspace_phase_invisible_at_degree_one():
    oracle = load_oracle()
    ket = np.array([1., 0.], complex)
    oracle.assert_qsvt_output(np.array([-.3, -.2 - .4j]), H, 1., ket, degree=1)
    # Literal T2(H)|0>; the UD mutant instead yields |0> at degree two.
    oracle.assert_qsvt_output(np.array([-.42, .08 + .16j]),
                              H,
                              1.,
                              ket,
                              degree=2)
    with pytest.raises(AssertionError):
        oracle.assert_qsvt_output(ket, H, 1., ket, degree=2)
