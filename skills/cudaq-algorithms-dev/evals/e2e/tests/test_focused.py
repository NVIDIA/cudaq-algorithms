# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused discovery keeps public hints separate from private numerical checks."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_focused_prompts_hide_callable_names_in_every_arm():
    import focused
    import run
    specs = focused.case_specs()
    assert len(specs) == 3
    for spec in specs:
        assert spec[
            "required_public_apis"]  # Private grading requirements stay intact.
        for arm in ("baseline", "previous", "skill"):
            prompt = run.prompt_for(spec, "/python", arm)
            assert "Required package workflow entry points" not in prompt
            for public_api in spec["required_public_apis"]:
                assert public_api not in prompt
                # Generic output names such as "action" are scientific fields,
                # not a disclosure of the callable sim_utils.action.
                assert public_api.removeprefix(
                    "cudaq_algorithms.") not in prompt
        assert run.prompt_for(spec, "/python", "previous") == run.prompt_for(
            spec, "/python", "skill")
    for _, historical in run.CASES.values():
        prompt = run.prompt_for(historical, "/python", "baseline")
        assert "Required package workflow entry points" in prompt
        assert json.dumps(historical["required_public_apis"]) in prompt


def test_focused_inputs_are_new_and_each_held_out_changes_science():
    import focused
    import quantum_cases
    for spec in focused.case_specs():
        cid, parent = spec["id"], spec["oracle_case"]
        variants = []
        for variant in (0, 1):
            parameters = focused.parameters(cid, variant)
            assert parameters != quantum_cases.parameters(parent, 0)
            assert parameters != quantum_cases.parameters(parent, 1)
            json.dumps(parameters, allow_nan=False)
            result = focused.expected(cid, parameters)
            reference = quantum_cases.expected(parent, parameters)
            assert result.keys() == reference.keys() == spec["outputs"].keys()
            for key in result:
                np.testing.assert_allclose(result[key],
                                           reference[key],
                                           atol=1e-14)
                assert np.isfinite(result[key]).all()
            variants.append((parameters, result))
        assert variants[0][0] != variants[1][0]
        assert any(not np.allclose(variants[0][1][key], variants[1][1][key])
                   for key in variants[0][1])


def test_focused_oracle_uses_supplied_parameters_not_cached_answers():
    import focused
    result = focused.expected(
        "focused_signed_action", {
            "terms": [[.5, "I"], [-1.5, "Y"]],
            "ket_real": [1., 0.],
            "ket_imag": [0., 0.]
        })
    np.testing.assert_allclose(result["action"], [.5, -1.5j], atol=1e-14)
    np.testing.assert_allclose(result["block"], [.25, -.75j], atol=1e-14)
    assert result["alpha"] == 2.
    assert result["success_probability"] == .625


@pytest.mark.parametrize("case_id",
                         ["focused_signed_action", "focused_phase_filter"])
def test_host_construction_without_circuit_execution_cannot_satisfy_focused_contract(
        case_id):
    import focused
    from grading import required_calls_seen
    spec = next(s for s in focused.case_specs() if s["id"] == case_id)
    # Even correct dense outputs plus every required host factory call must
    # fail without the trusted get_state-bound compiled-kernel trace.
    assert required_calls_seen(spec["required_symbols"],
                               spec["required_symbols"])["passed"]
    assert not required_calls_seen(spec.get("required_kernels", []),
                                   [])["passed"]
    shared_trace = [
        "cudaq_algorithms.pauli_lcu.apply",
        "cudaq_algorithms.common_kernels.signal_phase",
        "cudaq_algorithms.common_kernels.reflect_about_zero"
    ]
    assert required_calls_seen(spec["required_kernels"],
                               shared_trace)["passed"]


def test_focused_cli_does_not_mutate_historical_registry_and_marks_campaign(
        tmp_path, monkeypatch):
    import focused
    import run
    from test_runner import fixture_source
    source = fixture_source(tmp_path / "source", "skill")
    monkeypatch.setattr(run, "REPO", source)
    monkeypatch.setattr(run.subprocess, "check_output",
                        lambda *a, **k: "revision\n")
    monkeypatch.setattr(run, "runtime_metadata", lambda p: {"executable": p})
    original = dict(run.CASES)
    output = tmp_path / "campaign"
    focused.main([
        "prepare", "--output",
        str(output), "--repetitions", "1", "--python", sys.executable
    ])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["suite"] == "focused_natural_language"
    assert len(manifest["runs"]) == 6
    assert {r["case"]
            for r in manifest["runs"]} == {
                "focused_determinant_energy", "focused_signed_action",
                "focused_phase_filter"
            }
    assert run.CASES == original
    with pytest.raises(RuntimeError, match="suite"):
        run.main(["report", "--output", str(output)])
    focused.main(["report", "--output", str(output)])
    report = (output / "REPORT.md").read_text()
    assert "focused_natural_language" in report
    assert "unaided" in report
    assert "Prompts disclose required public workflow entry points" not in report


@pytest.mark.skipif(not os.environ.get("CUDAQ_E2E_PYTHON"),
                    reason="supported runtime required for real gold checks")
@pytest.mark.parametrize("case_id", [
    "focused_determinant_energy", "focused_signed_action",
    "focused_phase_filter"
])
@pytest.mark.parametrize("route", ["helper", "direct"])
def test_focused_gold_and_direct_circuit_routes_pass_full_grading(
        tmp_path, monkeypatch, case_id, route):
    import focused
    import run
    # Keep the established gold IO/observable code, substituting only a public
    # core circuit route for its convenience factory/helper. No answer values
    # enter the application; execute_app supplies each real input independently.
    source = focused.reference_source(case_id)
    substitutions = {
        "focused_signed_action":
        ("block = sim.action(enc, ket)",
         "state = cudaq.get_state(enc.encode_kernel(), sim.state_from(ket))\n"
         "block = sim.good_subspace(enc, state)",
         "cudaq_algorithms.sim_utils.action"),
        "focused_phase_filter":
        ("block = sim.transform(QSVT(enc), ket, seq)",
         "state = cudaq.get_state(QSVT(enc).kernel(seq), sim.state_from(ket))\n"
         "block = sim.good_subspace(enc, state)",
         "cudaq_algorithms.sim_utils.transform"),
        "focused_determinant_energy":
        ("prep = stateprep.slater_determinant_kernel(schedule)",
         "indices = stateprep.get_givens_rotation_indices(schedule)\n"
         "angles = stateprep.get_givens_rotation_angles(schedule)\n"
         "phases = stateprep.get_givens_rotation_phases(schedule)\n"
         "final_phases = [float(x) for x in schedule.final_phases]\n"
         "electrons = int(schedule.num_electrons)\n"
         "@cudaq.kernel\n"
         "def prep(q: cudaq.qview):\n"
         "    stateprep.complex_slater_determinant(q, indices, angles, phases, final_phases, electrons)",
         "cudaq_algorithms.stateprep._givens.slater_determinant_kernel"),
    }
    original, replacement, omitted = substitutions[case_id]
    if route == "direct":
        assert original in source
        source = source.replace(original, replacement)
    specs = {spec["id"]: (focused, spec) for spec in focused.case_specs()}
    monkeypatch.setattr(run, "CASES", specs)
    result = run.execute_app(run.REPO, specs[case_id][1], source,
                             os.environ["CUDAQ_E2E_PYTHON"],
                             tmp_path / "grading")
    for variant, check in enumerate(result["checks"]):
        assert check["numeric"]["passed"], check
        trace = json.loads(
            (tmp_path / "grading" / str(variant) / "trace.json").read_text())
        if route == "direct":
            assert omitted not in trace["calls"]
        assert check["passed"], check
    assert result["passed"]
