# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal output controls for the E05 native preparation/Walk probe."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

CHECKS = Path(__file__).resolve().parents[1] / "checks"
sys.path.insert(0, str(CHECKS))
PATH = CHECKS / "ci_runtime.py"
SPEC = importlib.util.spec_from_file_location("strategy_ci_runtime", PATH)
runtime = None
if PATH.is_file():
    runtime = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(runtime)


def dense_reference():
    i = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])

    def kron(a, b, c, d):
        return np.kron(np.kron(np.kron(a, b), c), d)

    return (.5 * kron(i, i, i, z) - .3 * kron(i, i, x, i) +
            .2 * kron(i, y, i, i) + .4 * kron(z, i, i, i))


def bundle():
    ket = np.zeros(16, complex)
    ket[[3, 5, 10, 12]] = np.array([1, 2j, -3, -4j]) / np.sqrt(30)
    phase = np.exp(.47j)
    prep = phase * ket
    clean = np.zeros(64, complex)
    clean[:16] = prep
    walk = np.zeros(64, complex)
    walk[:16] = -dense_reference() @ prep / 1.4
    walk[16] = np.sqrt(1 - np.vdot(walk, walk).real)
    return ket, {
        "preparation": prep,
        "zero_step": clean,
        "roundtrip": clean.copy(),
        "one_step": walk
    }


def test_dense_walk_reference_uses_independent_little_endian_pauli_order():
    assert runtime is not None, "CI runtime probe is not implemented"
    np.testing.assert_allclose(runtime.dense_hamiltonian(),
                               dense_reference(),
                               atol=1e-14,
                               rtol=0)


def test_complete_bundle_passes_without_adjusting_one_step_phase_or_scale():
    assert runtime is not None, "CI runtime probe is not implemented"
    expected, outputs = bundle()
    metrics = runtime.assess_outputs(outputs, expected)
    assert metrics["preparation"]["fidelity"] == pytest.approx(1)
    assert metrics["zero_step"]["fidelity"] == pytest.approx(1)
    assert metrics["roundtrip"]["fidelity"] == pytest.approx(1)
    assert metrics["walk_block_max_error"] < 1e-12


@pytest.mark.parametrize("defect", [
    "preparation_phase", "norm", "zero_step_leak", "roundtrip_leak",
    "one_step_sign", "one_step_scale", "one_step_nan", "one_step_shape",
    "one_step_norm"
])
def test_bad_preparation_or_walk_cannot_pass_bundle(defect):
    assert runtime is not None, "CI runtime probe is not implemented"
    expected, outputs = bundle()
    if defect == "preparation_phase": outputs["preparation"][5] *= -1
    if defect == "norm": outputs["preparation"] *= .5
    if defect in ("zero_step_leak", "roundtrip_leak"):
        name = "zero_step" if defect == "zero_step_leak" else "roundtrip"
        outputs[name][:16] /= np.sqrt(2)
        outputs[name][16:32] = outputs[name][:16]
    if defect == "one_step_sign": outputs["one_step"][:16] *= -1
    if defect == "one_step_scale": outputs["one_step"][:16] *= .5
    if defect == "one_step_nan": outputs["one_step"][0] = np.nan
    if defect == "one_step_shape":
        outputs["one_step"] = outputs["one_step"][:16]
    if defect == "one_step_norm": outputs["one_step"][32] = .1
    with pytest.raises(AssertionError):
        runtime.assess_outputs(outputs, expected)
