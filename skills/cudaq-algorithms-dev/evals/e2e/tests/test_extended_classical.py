# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent contracts for the remaining host/simulation/Trotter surfaces."""
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    return importlib.import_module("extended_classical")


def test_trotter_oracle_pins_phase_pruning_and_raw_resource_semantics():
    p = {
        "terms": [[0.25, "II"], [0.5, "XI"], [1e-10, "ZZ"]],
        "tolerance": 1e-8,
        "ordering": "preserve_input",
        "time": 0.6,
        "steps": 1,
        "order": 1,
        "angles": [0., 0.],
        "phase": 0.
    }
    out = module().expected("trotter_device_surfaces", p)
    np.testing.assert_allclose(out["raw_coefficients"], [0.5])
    np.testing.assert_array_equal(out["raw_words"], [[1, 0]])
    np.testing.assert_allclose(out["identity_coefficient"], 0.25)
    for field in ("injected_state", "state_input_state", "raw_state"):
        np.testing.assert_allclose(out[field],
                                   [np.cos(.3), -1j * np.sin(.3), 0, 0],
                                   atol=1e-14)
    np.testing.assert_allclose(out["invalid_raw_state"], [1, 0, 0, 0])
    np.testing.assert_allclose(out["resource_fields"], [1, 1, 1, 1, 0, .25])
    # The raw estimator counts the appended zero/identity word too.
    np.testing.assert_allclose(out["zero_probe_resources"],
                               [2, 1, 1, 2, 0, .25])


def test_df_norm_spin_and_residual_have_literal_independent_reference():
    target = np.zeros((2, 2, 2, 2))
    target[0, 0, 0, 0] = 2.4
    p = {
        "rotations": [np.eye(2).tolist()],
        "cores": [[[2., 0.], [0., 0.]]],
        "one_body": [[1.5, 0.], [0., .3]],
        "target_eri": target.tolist()
    }
    out = module().expected("df_spin_diagnostics", p)
    assert out["residual"] == pytest.approx(.4)
    assert out["lcu_norm"] == pytest.approx(1.3)
    assert out["burg_norm"] == pytest.approx(1.3)
    assert out["two_electron_energy"] == pytest.approx(.6)
    np.testing.assert_allclose(out["one_body_so"], np.diag([1.5, 1.5, .3, .3]))
    assert out["two_body_so"][0, 1, 1, 0] == 1.
    assert out["two_body_so"][0, 1, 0, 1] == 0.
    assert np.count_nonzero(out["two_body_so"]) == 4


def test_good_subspace_keeps_complex_phase_and_nonunit_block_weight():
    p = {
        "terms": [[.5, "I"], [-1.5, "Y"]],
        "ket_real": [1., 0.],
        "ket_imag": [0., 0.],
        "scale_real": 2.,
        "scale_imag": 0.
    }
    out = module().expected("good_subspace_layout", p)
    np.testing.assert_allclose(out["block"], [.25, -.75j])
    np.testing.assert_allclose(out["scaled_block"], [.5, -1.5j])
    assert out["good_probability"] == pytest.approx(.625)
    assert out["bad_probability"] == pytest.approx(.375)
    assert out["scaled_block_weight"] == pytest.approx(2.5)


def test_new_cases_have_distinct_inputs_and_wrong_outcomes_fail(tmp_path):
    from grading import compare_output, required_calls_seen
    for spec in module().case_specs():
        assert spec["required_public_apis"] and spec["required_symbols"]
        if spec["id"] != "df_spin_diagnostics":
            assert spec["required_kernels"]
            assert not required_calls_seen(spec["required_kernels"],
                                           [])["passed"]
        results = []
        for variant in (0, 1):
            params = module().parameters(spec["id"], variant)
            json.dumps(params, allow_nan=False)
            expected = module().expected(spec["id"], params)
            assert set(expected) == set(spec["outputs"])
            assert all(np.isfinite(v).all() for v in expected.values())
            results.append(expected)
            path = tmp_path / "result.npz"
            np.savez(path, **expected)
            assert compare_output(path, expected, spec)["passed"]
            for key in expected:
                wrong = dict(expected)
                wrong[key] = np.asarray(expected[key]) + .2
                np.savez(path, **wrong)
                assert not compare_output(path, expected, spec)["passed"], key
        assert any(not np.allclose(results[0][k], results[1][k])
                   for k in results[0])
    with pytest.raises(ValueError):
        module().parameters("df_spin_diagnostics", 2)


@pytest.mark.skipif(not os.environ.get("CUDAQ_E2E_PYTHON"),
                    reason="supported runtime required")
@pytest.mark.parametrize(
    "case_id",
    ["trotter_device_surfaces", "df_spin_diagnostics", "good_subspace_layout"])
def test_real_extended_gold_on_both_inputs(tmp_path, monkeypatch, case_id):
    import run
    cases = {s["id"]: (module(), s) for s in module().case_specs()}
    monkeypatch.setattr(run, "CASES", cases)
    result = run.execute_app(run.REPO, cases[case_id][1],
                             module().reference_source(case_id),
                             os.environ["CUDAQ_E2E_PYTHON"],
                             tmp_path / "grading")
    assert result["passed"], result
