# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller invariants: no answer leakage and numerical grading, not prose."""
import json
from pathlib import Path
import sys
import time
import os
import subprocess
from argparse import Namespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading import compare_output, required_calls_seen
from runtime import inventory, stage_workspace, scope_changes


def spec(**overrides):
    return {
        "outputs": {
            "state": "amplitudes"
        },
        "atol": 1e-9,
        "rtol": 1e-9,
        **overrides
    }


def test_wrong_numeric_output_cannot_pass_with_exit_zero(tmp_path):
    output = tmp_path / "result.npz"
    np.savez(output, state=[1., 0.])
    assert not compare_output(output, {"state": np.array([0., 1.])},
                              spec())["passed"]


def test_missing_nonfinite_and_wrong_shape_fail(tmp_path):
    output = tmp_path / "result.npz"
    assert not compare_output(output, {"state": [1., 0.]}, spec())["passed"]
    for value in ([float("nan"), 0.], [[1., 0.]], [1., float("inf")]):
        np.savez(output, state=value)
        assert not compare_output(output, {"state": [1., 0.]},
                                  spec())["passed"]


def test_phase_alignment_only_for_explicitly_allowed_state(tmp_path):
    output = tmp_path / "result.npz"
    np.savez(output, state=[1j, 0])
    expected = {"state": [1., 0.]}
    assert not compare_output(output, expected, spec())["passed"]
    assert compare_output(output, expected,
                          spec(phase_invariant_fields=["state"]))["passed"]


def test_object_arrays_cannot_be_loaded(tmp_path):
    output = tmp_path / "result.npz"
    np.savez(output, state=np.array([{}], dtype=object))
    assert not compare_output(output, {"state": [0.]}, spec())["passed"]


def test_corrupt_zip_is_failed_output_not_controller_exception(tmp_path):
    output = tmp_path / "result.npz"
    output.write_bytes(b"PK\x03\x04junk")
    assert not compare_output(output, {"state": [0.]}, spec())["passed"]


def test_fifo_and_symlink_output_are_rejected_before_parent_reads(tmp_path):
    output = tmp_path / "result.npz"
    os.mkfifo(output)
    assert not compare_output(output, {"state": [0.]}, spec())["passed"]
    output.unlink()
    actual = tmp_path / "elsewhere.npz"
    np.savez(actual, state=[0.])
    output.symlink_to(actual)
    assert not compare_output(output, {"state": [0.]}, spec())["passed"]


def test_timeout_still_applies_after_child_closes_stdout(tmp_path):
    from run import run_native
    from telemetry import TelemetryCollector
    command = [
        sys.executable, "-c", "import os,time; os.close(1); time.sleep(3)"
    ]
    start = time.monotonic()
    result = run_native(command, tmp_path, "", tmp_path, .1,
                        TelemetryCollector())
    assert result["timed_out"]
    assert time.monotonic() - start < 2


def test_timeout_remains_model_failure_without_final_usage():
    from run import classify_attempt
    status, reasons = classify_attempt({
        "timed_out": True,
        "returncode": -15
    }, {
        "input_tokens": None,
        "output_tokens": None
    }, False)
    assert status == "model_failure"
    assert reasons == []


def test_invalid_reference_is_infrastructure_failure(tmp_path):
    output = tmp_path / "result.npz"
    np.savez(output, state=[1.])
    with pytest.raises(ValueError, match="reference"):
        compare_output(output, {"state": [float("nan")]}, spec())


def test_real_api_evidence_uses_exact_symbols_not_substrings():
    assert required_calls_seen(["pkg.f"], ["pkg.f"])["passed"]
    assert not required_calls_seen(["pkg.f"], ["pkg.fake"])["passed"]


def test_staging_only_difference_is_operational_skill(tmp_path):
    source = tmp_path / "source"
    for name in [
            "python/cudaq_algorithms/a.py",
            "docs/sphinx/examples/python/ex.py", "tests/python/test_a.py",
            "pyproject.toml", "README.md", "skills/cudaq-algorithms/SKILL.md",
            "skills/cudaq-algorithms/references/runtime.md",
            "skills/cudaq-algorithms/assets/template.md",
            "skills/cudaq-algorithms/evals/secret.json",
            "docs/superpowers/plans/evaluator-design.md", ".git/history",
            "AGENTS.md"
    ]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    public = {"seed": 1}
    baseline, treatment = tmp_path / "baseline", tmp_path / "treatment"
    stage_workspace(source, baseline, "baseline", public)
    stage_workspace(source, treatment, "skill", public)
    b, s = inventory(baseline), inventory(treatment)
    assert {k: v for k, v in s.items() if not k.startswith("skills/")} == b
    assert not any("evals" in k or ".git" in k or "AGENTS.md" in k for k in s)
    assert not any("superpowers" in k for k in s)
    assert json.loads((baseline / "input.json").read_text()) == public


def test_readonly_source_mutation_and_symlinks_are_detected(tmp_path):
    (tmp_path / "source.py").write_text("original")
    before = inventory(tmp_path)
    (tmp_path / "app.py").write_text("allowed")
    assert scope_changes(before, inventory(tmp_path)) == []
    (tmp_path / "source.py").write_text("changed")
    assert scope_changes(before, inventory(tmp_path)) == ["source.py"]
    (tmp_path / "app.py").unlink()
    (tmp_path / "app.py").symlink_to(tmp_path / "source.py")
    assert inventory(tmp_path)["app.py"].startswith("symlink:")


def fixture_source(root, text):
    for name, content in {
            "python/cudaq_algorithms/__init__.py": "# package",
            "skills/cudaq-algorithms/SKILL.md": text,
            "skills/cudaq-algorithms/references/api.md": text,
            "skills/cudaq-algorithms/authoring/secret.md": "excluded",
            "skills/cudaq-algorithms/coverage/history.md": "excluded",
            "skills/cudaq-algorithms/assets/template.md": "excluded",
            "skills/cudaq-algorithms/evals/results/answer.json": "excluded"
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def prepare_fixture(tmp_path, monkeypatch, previous=True):
    import run
    source = fixture_source(tmp_path / "repo", "new skill")
    old = fixture_source(tmp_path / "old",
                         "old skill") / "skills/cudaq-algorithms"
    monkeypatch.setattr(run, "REPO", source)
    monkeypatch.setattr(run.subprocess, "check_output",
                        lambda *a, **k: "revision\n")
    monkeypatch.setattr(run, "runtime_metadata", lambda p: {"executable": p})
    args = Namespace(output=tmp_path / "campaign",
                     previous_skill=old if previous else None,
                     repetitions=5,
                     parallel=2,
                     timeout=480,
                     seed=3,
                     cases="stateprep,uccsd",
                     python=sys.executable,
                     psi4_python=None,
                     rg=None)
    # Select existing case IDs without assuming scientific case internals.
    args.cases = ",".join(list(run.CASES)[:2])
    run.prepare(args)
    return args.output, json.loads(
        (args.output / "manifest.json").read_text()), old


def test_three_arm_snapshots_are_isolated_and_prompts_equal(
        tmp_path, monkeypatch):
    from run import prompt_for, verify_snapshots
    output, manifest, old = prepare_fixture(tmp_path, monkeypatch)
    assert set(manifest["arms"]) == {"baseline", "previous", "skill"}
    snapshots = {
        arm: Path(data["snapshot"])
        for arm, data in manifest["arms"].items()
    }
    assert not (snapshots["baseline"] / "skills").exists()
    assert (snapshots["previous"] /
            "skills/cudaq-algorithms/SKILL.md").read_text() == "old skill"
    assert (snapshots["skill"] /
            "skills/cudaq-algorithms/SKILL.md").read_text() == "new skill"
    for path in snapshots.values():
        assert not any(part in inventory(path) for part in [
            "skills/cudaq-algorithms/authoring/secret.md",
            "skills/cudaq-algorithms/coverage/history.md",
            "skills/cudaq-algorithms/evals/results/answer.json"
        ])
        if (path / "skills").exists():
            assert (path / "skills/cudaq-algorithms/assets/template.md"
                    ).read_text() == "excluded"
    (old / "SKILL.md").write_text("later author edit")
    verify_snapshots(manifest)
    task = {"task": "Compute", "outputs": {}}
    assert prompt_for(task, "/python",
                      "previous") == prompt_for(task, "/python", "skill")
    (snapshots["previous"] /
     "skills/cudaq-algorithms/SKILL.md").write_text("tampered")
    with pytest.raises(RuntimeError, match="snapshot"):
        verify_snapshots(manifest)


@pytest.mark.parametrize("previous,width", [(True, 3), (False, 2)])
def test_ordering_balances_blocks_and_each_position(tmp_path, monkeypatch,
                                                    previous, width):
    from collections import Counter
    _, manifest, _ = prepare_fixture(tmp_path, monkeypatch, previous)
    arms = {"baseline", "previous", "skill"
            } if previous else {"baseline", "skill"}
    assert len(manifest["runs"]) == 10 * width
    positions = [Counter() for _ in range(width)]
    for offset in range(0, len(manifest["runs"]), width):
        block = manifest["runs"][offset:offset + width]
        assert len({(r["case"], r["repetition"]) for r in block}) == 1
        assert {r["arm"] for r in block} == arms
        for index, record in enumerate(block):
            positions[index][record["arm"]] += 1
    assert all(max(c.values()) - min(c.values()) <= 1 for c in positions)


def test_explicit_py_compile_writes_only_allowed_task_cache(tmp_path):
    from runtime import environment
    workspace = tmp_path / "workspace"
    stage_workspace(fixture_source(tmp_path / "repo", "skill"), workspace,
                    "baseline", {})
    (workspace / "app.py").write_text("print('ok')\n")
    initial = inventory(workspace)
    subprocess.run([
        sys.executable, "-B", "-m", "py_compile", "app.py",
        "python/cudaq_algorithms/__init__.py"
    ],
                   cwd=workspace,
                   env=environment(workspace),
                   check=True)
    assert scope_changes(initial, inventory(workspace)) == []
    assert list((workspace / ".tmp").rglob("*.pyc"))


def test_real_rg_is_staged_and_accessible_inside_worker_sandbox(tmp_path):
    import shutil
    from runtime import CODEX, environment, sandbox_command
    rg = os.environ.get("CUDAQ_E2E_RG") or shutil.which("rg")
    python = os.environ.get("CUDAQ_E2E_PYTHON")
    if not rg or not python or not CODEX.exists():
        pytest.skip(
            "set CUDAQ_E2E_RG and CUDAQ_E2E_PYTHON for real sandbox check")
    for arm in ("baseline", "previous", "skill"):
        source = fixture_source(tmp_path / arm / "repo", "skill")
        workspace = tmp_path / arm / "workspace"
        stage_workspace(source, workspace, arm, {}, rg_path=rg)
        initial = inventory(workspace)
        command = sandbox_command(workspace, python, [
            "-c",
            "import subprocess; subprocess.run(['rg', '--version'], check=True); "
            "subprocess.run(['rg', 'package', 'python'], check=True); "
            "import py_compile; py_compile.compile('python/cudaq_algorithms/__init__.py', doraise=True)"
        ])
        result = subprocess.run(command,
                                cwd=workspace,
                                env=environment(workspace),
                                capture_output=True,
                                text=True,
                                timeout=30)
        assert result.returncode == 0, result.stderr
        assert "ripgrep" in result.stdout and "# package" in result.stdout
        assert scope_changes(initial, inventory(workspace)) == []


def test_common_sources_cannot_differ_even_with_rehashed_arm(
        tmp_path, monkeypatch):
    from run import verify_snapshots
    _, manifest, _ = prepare_fixture(tmp_path, monkeypatch)
    snapshot = Path(manifest["arms"]["previous"]["snapshot"])
    (snapshot /
     "python/cudaq_algorithms/__init__.py").write_text("changed package")
    manifest["arms"]["previous"]["inventory"] = inventory(snapshot)
    with pytest.raises(RuntimeError, match="common source"):
        verify_snapshots(manifest)


def test_all_arm_sources_must_match_canonical_gold_source(
        tmp_path, monkeypatch):
    from run import verify_snapshots
    _, manifest, _ = prepare_fixture(tmp_path, monkeypatch)
    for arm in manifest["arms"].values():
        snapshot = Path(arm["snapshot"])
        (snapshot / "python/cudaq_algorithms/__init__.py"
         ).write_text("shared wrong package")
        arm["inventory"] = inventory(snapshot)
    with pytest.raises(RuntimeError, match="common source"):
        verify_snapshots(manifest)


@pytest.mark.parametrize("action", ["preflight", "run_all"])
def test_snapshot_tampering_blocks_preflight_and_resume(
        tmp_path, monkeypatch, action):
    import run
    output, manifest, _ = prepare_fixture(tmp_path, monkeypatch)
    snapshot = Path(manifest["arms"]["skill"]["snapshot"])
    (snapshot /
     "skills/cudaq-algorithms/SKILL.md").write_text("changed candidate")
    with pytest.raises(RuntimeError, match="snapshot"):
        getattr(run, action)(output)


def test_resume_keeps_finalized_and_partial_attempt_evidence(tmp_path):
    from run import run_one
    record = {
        "id": "case--1--skill",
        "case": "case",
        "arm": "skill",
        "repetition": 1
    }
    destination = tmp_path / "runs" / record["id"]
    destination.mkdir(parents=True)
    (destination / "events.jsonl").write_text("retained raw evidence")
    partial = run_one(record, {}, tmp_path)
    assert partial["attempted"] and partial[
        "status"] == "infrastructure_failure"
    assert (destination /
            "events.jsonl").read_text() == "retained raw evidence"
    assert run_one(record, {}, tmp_path) == partial
