# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Posthoc evidence recovery stays bounded and never executes worker code here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import capture
import numerical


@pytest.fixture
def checker():
    path = HERE / "artifact_checks.py"
    assert path.is_file(
    ), "posthoc artifact/check orchestration is not implemented"
    spec = importlib.util.spec_from_file_location("strategy_artifact_checks",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def freeze_manifest(root, manifest):
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "manifest.sha256.json", {
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        })


def fixture_run(tmp_path,
                *,
                case="E01",
                terminal=True,
                original=True,
                changed_consumer=False):
    root = tmp_path / "run"
    attempt = {
        "attempt_id": "opaque123",
        "job_name": "opaque123",
        "case_id": case,
        "arm": "baseline",
        "dataset_dir": "datasets/opaque123"
    }
    source = root / attempt["dataset_dir"] / "opaque123/environment/repo"
    (source / "tests/python").mkdir(parents=True)
    (source / "example.py").write_text("value = 1\n")
    (source /
     "tests/python/test_original.py").write_text("def test_original(): pass\n")
    consumers = (numerical.E02_CONSUMERS
                 if case == "E02" else numerical.E05_CONSUMERS if case == "E05"
                 else numerical.E08_CONSUMERS if case == "E08" else ())
    for name in consumers:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen public consumer\n")
    source_hashes = {
        p.relative_to(source).as_posix():
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*")) if p.is_file()
    }
    manifest = {
        "schema_version": 1,
        "image_id": numerical.IMAGE_ID,
        "attempts": [attempt],
        "input_hashes": {
            "source_inventory":
            hashlib.sha256(json.dumps(source_hashes,
                                      sort_keys=True).encode()).hexdigest(),
            "capture":
            hashlib.sha256((HERE / "capture.py").read_bytes()).hexdigest()
        }
    }
    freeze_manifest(root, manifest)
    trial = root / "jobs/opaque123/opaque123__trial"
    worktree = trial / "agent/worktree"
    shutil.copytree(source, worktree)
    (worktree / "example.py").write_text("value = 2\n")
    if changed_consumer:
        (worktree /
         consumers[0]).write_text("# worker changed frozen consumer\n")
    if terminal:
        write_json(trial / "result.json", {
            "finished_at": "2026-09-15T12:00:00Z",
            "exception_info": None
        })
    if original:
        capture.capture(source, worktree, trial / "verifier/artifacts/capture")
    return root, source, worktree, trial


def docker_boundary(monkeypatch,
                    checker,
                    *,
                    targeted=b"32 passed in 1s\n",
                    regression=b"307 passed, 4 skipped in 1s\n",
                    timeout=False,
                    on_check=None):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv[
            0] == "docker", "worker commands must only run inside Docker"
        if argv[1] == "kill":
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        assert argv[1] == "run"
        mounts = {}
        for index, token in enumerate(argv):
            if token == "--mount":
                parts = dict(
                    part.split("=", 1) for part in argv[index + 1].split(",")
                    if "=" in part)
                mounts[parts["target"]] = Path(parts["source"])
        if "/capture.py" in mounts:
            if timeout:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            capture.capture(mounts["/source"], mounts["/worktree"],
                            mounts["/output"] / "capture")
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if on_check is not None:
            on_check(argv)
        output = targeted if "--confcutdir=/checks " in argv[-1] else regression
        code = 1 if b"failed" in output else 0
        return subprocess.CompletedProcess(argv, code, output, b"")

    monkeypatch.setattr(checker.subprocess, "run", run)
    return calls


def test_complete_capture_is_reconstructed_without_running_worker_code(
        checker, tmp_path, monkeypatch):
    root, source, worktree, trial = fixture_run(tmp_path)
    calls = docker_boundary(monkeypatch, checker)
    original = trial / "verifier/artifacts/capture"
    before = (original / "capture.json").read_bytes()
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == "complete"
    assert result["capture_status"] == "complete"
    assert result["capture_selected_path"] == str(original)
    assert result["targeted"] == result["regression"] == "not_applicable"
    assert (
        root /
        "posthoc/opaque123/repository/example.py").read_text() == "value = 2\n"
    assert (original / "capture.json").read_bytes() == before
    assert not calls
    assert result["hashes"]["capture_helper"] == hashlib.sha256(
        (HERE / "capture.py").read_bytes()).hexdigest()
    assert result == json.loads(
        (root / "posthoc/opaque123/artifact_checks.json").read_text())


@pytest.mark.parametrize("state", ["missing", "incomplete", "tampered"])
def test_bad_original_capture_is_retained_and_recovered_with_only_scoped_mounts(
        checker, tmp_path, monkeypatch, state):
    root, source, worktree, trial = fixture_run(tmp_path,
                                                original=state != "missing")
    original = trial / "verifier/artifacts/capture"
    if state == "incomplete":
        data = json.loads((original / "capture.json").read_text())
        data["status"] = "incomplete"
        write_json(original / "capture.json", data)
    elif state == "tampered":
        (original / "changes.diff").write_text("corrupted\n")
    before = {
        str(p): p.read_bytes()
        for p in original.rglob("*") if p.is_file()
    }
    calls = docker_boundary(monkeypatch, checker)
    monkeypatch.setenv("NVIDIA_API_KEY", "must-not-forward")
    result = checker.check_attempt(root, "opaque123")
    assert result["capture_status"] == "complete"
    assert result["capture_selected_path"] == str(
        root / "posthoc/opaque123/recovered/capture")
    assert {
        str(p): p.read_bytes()
        for p in original.rglob("*") if p.is_file()
    } == before
    assert len(calls) == 1
    argv, kwargs = calls[0]
    for option, value in (("--network", "none"), ("--memory",
                                                  "512m"), ("--cpus", "2"),
                          ("--cap-add", "DAC_OVERRIDE"), ("--user", "0:0")):
        assert argv[argv.index(option) + 1] == value
    assert "--read-only" in argv and "no-new-privileges" in argv
    assert kwargs["timeout"] == 120
    assert "NVIDIA_API_KEY" not in kwargs["env"]
    mounts = [
        argv[i + 1] for i, token in enumerate(argv) if token == "--mount"
    ]
    assert len(mounts) == 4
    assert sum(m.endswith(",readonly") for m in mounts) == 3
    assert any("source=" + str(source) + "," in m for m in mounts)
    assert any("source=" + str(worktree) + "," in m for m in mounts)
    assert not any("source=" + str(root / "jobs") + "," in m for m in mounts)


@pytest.mark.parametrize("result_kind",
                         ["aggregate", "unfinished", "ambiguous"])
def test_only_one_finished_trial_allows_posthoc_work(checker, tmp_path,
                                                     monkeypatch, result_kind):
    root, _, _, trial = fixture_run(tmp_path, terminal=False)
    if result_kind == "aggregate":
        write_json(trial.parent / "result.json", {"finished_at": "yes"})
    elif result_kind == "unfinished":
        write_json(trial / "result.json", {"finished_at": None})
    else:
        write_json(trial / "result.json", {"finished_at": "yes"})
        write_json(trial.parent / "opaque123__another/result.json",
                   {"finished_at": "yes"})
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == "unknown" and result[
        "capture_status"] == "unknown"
    assert not (root / "posthoc").exists()
    assert not calls


def test_terminal_timeout_can_still_recover_evidence(checker, tmp_path,
                                                     monkeypatch):
    root, _, _, trial = fixture_run(tmp_path, original=False)
    write_json(
        trial / "result.json", {
            "finished_at": "yes",
            "exception_info": {
                "exception_type": "AgentTimeoutError",
                "exception_message": "worker timed out"
            }
        })
    docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["capture_status"] == "complete"
    assert result["worker_execution_status"] == "failed"


@pytest.mark.parametrize("targeted, expected",
                         [(b"32 passed in 1s\n", "pass"),
                          (b"31 passed in 1s\n", "unknown"),
                          (b"31 passed, 1 skipped in 1s\n", "unknown"),
                          (b"1 failed, 31 passed in 1s\n", "fail")])
def test_e03_runs_both_frozen_check_suites_and_enforces_coverage(
        checker, tmp_path, monkeypatch, targeted, expected):
    root, source, _, _ = fixture_run(tmp_path, case="E03")
    calls = docker_boundary(monkeypatch, checker, targeted=targeted)
    result = checker.check_attempt(root, "opaque123")
    assert result["targeted"] == expected
    assert result["regression"] == "pass"
    assert len(calls) == 2
    for kind in ("targeted", "regression"):
        assert (root / "posthoc/opaque123" / kind / "result.json").is_file()
        assert (root / "posthoc/opaque123" / kind / "pytest.log").is_file()
    assert "/frozen-source/tests/python" in calls[1][0][-1]
    assert result["hashes"]["checks_inventory"]
    assert result["hashes"]["numerical_helper"]
    for index, kind in enumerate(("targeted", "regression")):
        output = root / "posthoc/opaque123" / kind
        projected = output / "frozen-source"
        assert "type=bind,source=" + str(
            projected) + ",target=/frozen-source,readonly" in calls[index][0]
        assert "type=bind,source=" + str(
            source) + ",target=/frozen-source,readonly" not in calls[index][0]
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert stat.S_IMODE(projected.stat().st_mode) == 0o755
        check_result = json.loads((output / "result.json").read_text())
        projection = check_result["frozen_source_projection"]
        assert projection["source_inventory_sha256"] == result["hashes"][
            "source_inventory"]
        assert projection["projection_inventory_sha256"] == projection[
            "source_inventory_sha256"]
        assert projection["source_unchanged"] and projection[
            "projection_unchanged"]
        assert projection["manifest_sha256"] == hashlib.sha256(
            (output /
             "frozen-source-projection.json").read_bytes()).hexdigest()
        for name, digest in projection["source_files"].items():
            assert hashlib.sha256(
                (source / name).read_bytes()).hexdigest() == digest
            assert hashlib.sha256(
                (projected / name).read_bytes()).hexdigest() == digest
            assert stat.S_IMODE((projected / name).stat().st_mode) == 0o644
        assert result["hashes"][kind + "_result"] == hashlib.sha256(
            (output / "result.json").read_bytes()).hexdigest()
        assert result["hashes"][kind + "_log"] == hashlib.sha256(
            (output / "pytest.log").read_bytes()).hexdigest()


def test_e02_requires_explicit_binding_but_still_runs_regression(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E02")
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == result["targeted"] == "unknown"
    assert result["regression"] == "pass"
    assert "e02_binding_unavailable" in result["failures"]
    assert not (root / "posthoc/opaque123/targeted").exists()
    assert (root / "posthoc/opaque123/regression/result.json").is_file()
    assert len(calls) == 1
    assert "/frozen-source/tests/python" in calls[0][0][-1]


def test_unreadable_e02_binding_does_not_discard_regression_evidence(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E02")
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root,
                                   "opaque123",
                                   e02_binding=tmp_path / "missing.json")
    assert result["targeted"] == "unknown"
    assert result["regression"] == "pass"
    assert "e02_binding_unavailable" in result["failures"]
    assert len(calls) == 1


def test_e02_runs_registered_60_test_suite_with_frozen_explicit_binding(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E02")
    binding = tmp_path / "evaluator-binding.json"
    binding.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "callable",
            "module": "cudaq_algorithms.composition",
            "attribute": "scale",
            "args": ["encoding", "factor"],
            "kwargs": {},
            "zero_policy": "encode"
        }) + "\n")
    calls = docker_boundary(monkeypatch,
                            checker,
                            targeted=b"60 passed in 1s\n")
    result = checker.check_attempt(root, "opaque123", e02_binding=binding)
    assert result["status"] == "complete"
    assert result["targeted"] == result["regression"] == "pass"
    assert result["hashes"]["e02_binding"] == hashlib.sha256(
        binding.read_bytes()).hexdigest()
    targeted_result = json.loads(
        (root / "posthoc/opaque123/targeted/result.json").read_text())
    assert targeted_result["case_id"] == "E02"
    assert targeted_result["binding_sha256"] == result["hashes"]["e02_binding"]
    assert targeted_result["e02_inputs"]["unchanged"] is True
    assert result["hashes"]["e02_inputs_manifest"] == hashlib.sha256(
        (root / "posthoc/opaque123/targeted/e02-inputs.json"
         ).read_bytes()).hexdigest()
    assert targeted_result["e02_inputs"]["manifest_sha256"] == result[
        "hashes"]["e02_inputs_manifest"]
    assert len(calls) == 2
    assert "/checks/test_scale_encoding.py" in calls[0][0][-1]
    assert any("target=/e02-input,readonly" in token for token in calls[0][0])
    assert "E02_SCALE_ADAPTER=/e02-input/binding.json" in calls[0][0]


def test_cli_rejects_e02_binding_for_all_terminal_or_non_e02(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E03")
    binding = tmp_path / "binding.json"
    binding.write_text("{}\n")
    calls = docker_boundary(monkeypatch, checker)
    with pytest.raises(SystemExit):
        checker.main([
            "--run-dir",
            str(root), "--all-terminal", "--e02-binding",
            str(binding)
        ])
    with pytest.raises(SystemExit):
        checker.main([
            "--run-dir",
            str(root), "--attempt-id", "opaque123", "--e02-binding",
            str(binding)
        ])
    assert not (root / "posthoc").exists()
    assert not calls


def e05_binding(path):
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "artifact",
            "artifact": "captured",
            "selection": ["qubitization", "pauli_lcu"]
        }) + "\n")
    return path


def e08_binding(path):
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "build": {
                "kind": "callable",
                "module": "example",
                "attribute": "build",
                "args": ["terms", "time", "state_prep"],
                "kwargs": {}
            },
            "recover": {
                "kind": "callable",
                "module": "example",
                "attribute": "recover",
                "args": ["cos_state", "sin_state"],
                "kwargs": {}
            },
            "validate": {
                "kind": "callable",
                "module": "example",
                "attribute": "validate",
                "args": ["actual_state", "initial_state"],
                "kwargs": {}
            },
            "kernels": {
                "kind": "mapping",
                "cosine": "cosine",
                "sine": "sine"
            },
            "domain": {
                "hamiltonian": "complex",
                "initial_state": "complex"
            },
            "phase_policy": "exact",
            "recovery_semantics": "real_linear",
            "validation_mode": "raises",
        }) + "\n")
    return path


def test_e05_requires_explicit_binding_but_still_runs_original_regression(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E05")
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == result["targeted"] == "unknown"
    assert result["regression"] == "pass"
    assert "e05_binding_unavailable" in result["failures"]
    assert not (root / "posthoc/opaque123/targeted").exists()
    assert (root / "posthoc/opaque123/regression/result.json").is_file()
    assert len(calls) == 1


def test_e05_valid_binding_joins_capture_checks_consumers_and_exact_bytes(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E05")
    binding = e05_binding(tmp_path / "e05-binding.json")
    count = numerical.CHECK_SUITES["E05"]["targeted"][0]
    calls = docker_boundary(monkeypatch,
                            checker,
                            targeted=f"{count} passed in 1s\n".encode())
    result = checker.check_attempt(root, "opaque123", e05_binding=binding)
    targeted = root / "posthoc/opaque123/targeted"
    report = json.loads((targeted / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["targeted"] == result["regression"] == "pass"
    assert result["hashes"]["e05_binding"] == hashlib.sha256(
        binding.read_bytes()).hexdigest()
    assert report["binding_sha256"] == result["hashes"]["e05_binding"]
    assert report["e05_inputs"]["unchanged"] is True
    assert result["hashes"]["e05_inputs_manifest"] == hashlib.sha256(
        (targeted / "e05-inputs.json").read_bytes()).hexdigest()
    assert report["e05_inputs"]["manifest_sha256"] == result["hashes"][
        "e05_inputs_manifest"]
    assert result["hashes"]["e05_captured_inventory"] == report["e05_inputs"][
        "captured_inventory_sha256"]
    assert len(calls) == 2


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_e05_artifact_freeze_rejects_missing_or_changed_staged_binding(
        checker, tmp_path, monkeypatch, fault):
    root, _, _, _ = fixture_run(tmp_path, case="E05")
    binding = e05_binding(tmp_path / "e05-binding.json")
    count = numerical.CHECK_SUITES["E05"]["targeted"][0]
    docker_boundary(monkeypatch,
                    checker,
                    targeted=f"{count} passed in 1s\n".encode())
    record = checker.check_attempt(root, "opaque123", e05_binding=binding)
    output = root / "posthoc/opaque123/targeted"
    staged = output / "e05-input/binding.json"
    if fault == "missing":
        staged.unlink()
    else:
        staged.write_text("{}\n")
    check_result = json.loads((output / "result.json").read_text())
    with pytest.raises(ValueError, match="incomplete or unbound"):
        checker._bind_inputs(check_result,
                             output,
                             root / "posthoc/opaque123/repository",
                             record,
                             case_id="E05")


def test_e05_cli_runs_one_bound_attempt(checker, tmp_path, monkeypatch,
                                        capsys):
    root, _, _, _ = fixture_run(tmp_path, case="E05")
    binding = e05_binding(tmp_path / "e05-binding.json")
    count = numerical.CHECK_SUITES["E05"]["targeted"][0]
    calls = docker_boundary(monkeypatch,
                            checker,
                            targeted=f"{count} passed in 1s\n".encode())
    assert checker.main([
        "--run-dir",
        str(root), "--attempt-id", "opaque123", "--e05-binding",
        str(binding)
    ]) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 1 and records[0]["attempt_id"] == "opaque123"
    assert records[0]["targeted"] == records[0]["regression"] == "pass"
    assert len(calls) == 2


def test_e05_incomplete_targeted_coverage_remains_unknown(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E05")
    binding = e05_binding(tmp_path / "e05-binding.json")
    count = numerical.CHECK_SUITES["E05"]["targeted"][0]
    docker_boundary(monkeypatch,
                    checker,
                    targeted=f"{count - 1} passed in 1s\n".encode())
    result = checker.check_attempt(root, "opaque123", e05_binding=binding)
    assert result["targeted"] == result["status"] == "unknown"
    assert result["regression"] == "pass"
    assert "targeted_coverage_or_execution_unknown" in result["failures"]


@pytest.mark.parametrize(
    "fault", ["binding_changed", "freeze_changed", "consumer_changed"])
def test_e05_changed_provenance_is_unknown_while_regression_survives(
        checker, tmp_path, monkeypatch, fault):
    root, _, _, _ = fixture_run(tmp_path,
                                case="E05",
                                changed_consumer=fault == "consumer_changed")
    binding = e05_binding(tmp_path / "e05-binding.json")

    def tamper(_argv):
        if fault == "binding_changed":
            binding.write_text("{}\n")
        elif fault == "freeze_changed":
            freezes = list(
                (root / "posthoc/opaque123").glob("*/e05-inputs.json"))
            if freezes:
                freezes[0].write_text("{}\n")

    count = numerical.CHECK_SUITES["E05"]["targeted"][0]
    calls = docker_boundary(monkeypatch,
                            checker,
                            targeted=f"{count} passed in 1s\n".encode(),
                            on_check=tamper)
    result = checker.check_attempt(root, "opaque123", e05_binding=binding)
    assert result["targeted"] == "unknown"
    assert result["regression"] == "pass"
    assert result["status"] == "unknown"
    assert calls


def test_e05_cli_binding_guards_refuse_wrong_case_both_bindings_and_all_terminal(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E03")
    binding = e05_binding(tmp_path / "e05-binding.json")
    calls = docker_boundary(monkeypatch, checker)
    invocations = [
        [
            "--run-dir",
            str(root), "--all-terminal", "--e05-binding",
            str(binding)
        ],
        [
            "--run-dir",
            str(root), "--attempt-id", "opaque123", "--e05-binding",
            str(binding)
        ],
        [
            "--run-dir",
            str(root), "--attempt-id", "opaque123", "--e02-binding",
            str(binding), "--e05-binding",
            str(binding)
        ],
    ]
    for argv in invocations:
        with pytest.raises(SystemExit):
            checker.main(argv)
    assert not (root / "posthoc").exists()
    assert not calls


def test_e08_requires_explicit_binding_but_still_runs_original_regression(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == result["targeted"] == "unknown"
    assert result["regression"] == "pass"
    assert "e08_binding_unavailable" in result["failures"]
    assert not (root / "posthoc/opaque123/targeted").exists()
    assert (root / "posthoc/opaque123/regression/result.json").is_file()
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["malformed", "non_object", "unsafe"])
def test_e08_invalid_binding_never_launches_targeted_but_keeps_regression(
        checker, tmp_path, monkeypatch, state):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    binding = tmp_path / "e08-binding.json"
    if state == "malformed":
        binding.write_text("{\n")
    elif state == "non_object":
        binding.write_text("[]\n")
    else:
        target = e08_binding(tmp_path / "actual-binding.json")
        binding.symlink_to(target)
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123", e08_binding=binding)
    assert result["targeted"] == result["status"] == "unknown"
    assert result["regression"] == "pass"
    assert "e08_binding_unavailable" in result["failures"]
    assert not (root / "posthoc/opaque123/targeted").exists()
    assert len(calls) == 1


def test_e08_valid_binding_joins_source_capture_checker_consumers_and_exact_bytes(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    binding = e08_binding(tmp_path / "e08-binding.json")
    count = numerical.CHECK_SUITES["E08"]["targeted"][0]
    calls = docker_boundary(monkeypatch,
                            checker,
                            targeted=f"{count} passed in 1s\n".encode())
    result = checker.check_attempt(root, "opaque123", e08_binding=binding)
    targeted = root / "posthoc/opaque123/targeted"
    report = json.loads((targeted / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["targeted"] == result["regression"] == "pass"
    assert result["hashes"]["e08_binding"] == hashlib.sha256(
        binding.read_bytes()).hexdigest()
    assert report["case_id"] == "E08"
    assert report["binding_sha256"] == result["hashes"]["e08_binding"]
    assert report["e08_inputs"]["unchanged"] is True
    assert result["hashes"]["e08_inputs_manifest"] == hashlib.sha256(
        (targeted / "e08-inputs.json").read_bytes()).hexdigest()
    assert report["e08_inputs"]["manifest_sha256"] == result["hashes"][
        "e08_inputs_manifest"]
    assert result["hashes"]["e08_captured_inventory"] == report["e08_inputs"][
        "captured_inventory_sha256"]
    assert len(calls) == 2
    assert "/checks/test_evolution_example.py" in calls[0][0][-1]
    assert any("target=/e08-input,readonly" in token for token in calls[0][0])
    assert "E08_EVOLUTION_ADAPTER=/e08-input/binding.json" in calls[0][0]


@pytest.mark.parametrize(
    "targeted", [b"11 passed in 1s\n", b"11 passed, 1 skipped in 1s\n"])
def test_e08_incomplete_or_skipped_targeted_coverage_remains_unknown(
        checker, tmp_path, monkeypatch, targeted):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    binding = e08_binding(tmp_path / "e08-binding.json")
    docker_boundary(monkeypatch, checker, targeted=targeted)
    result = checker.check_attempt(root, "opaque123", e08_binding=binding)
    assert result["targeted"] == result["status"] == "unknown"
    assert result["regression"] == "pass"
    assert "targeted_coverage_or_execution_unknown" in result["failures"]


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_e08_artifact_freeze_rejects_missing_or_changed_staged_binding(
        checker, tmp_path, monkeypatch, fault):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    binding = e08_binding(tmp_path / "e08-binding.json")
    count = numerical.CHECK_SUITES["E08"]["targeted"][0]
    docker_boundary(monkeypatch,
                    checker,
                    targeted=f"{count} passed in 1s\n".encode())
    record = checker.check_attempt(root, "opaque123", e08_binding=binding)
    output = root / "posthoc/opaque123/targeted"
    staged = output / "e08-input/binding.json"
    if fault == "missing":
        staged.unlink()
    else:
        staged.write_text("{}\n")
    check_result = json.loads((output / "result.json").read_text())
    with pytest.raises(ValueError, match="incomplete or unbound"):
        checker._bind_inputs(check_result,
                             output,
                             root / "posthoc/opaque123/repository",
                             record,
                             case_id="E08")


@pytest.mark.parametrize("fault",
                         ["wrong_consumer", "wrong_checker", "wrong_capture"])
def test_e08_artifact_freeze_rejects_wrong_join_inventory(
        checker, tmp_path, monkeypatch, fault):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    binding = e08_binding(tmp_path / "e08-binding.json")
    count = numerical.CHECK_SUITES["E08"]["targeted"][0]
    docker_boundary(monkeypatch,
                    checker,
                    targeted=f"{count} passed in 1s\n".encode())
    record = checker.check_attempt(root, "opaque123", e08_binding=binding)
    output = root / "posthoc/opaque123/targeted"
    check_result = json.loads((output / "result.json").read_text())
    if fault == "wrong_consumer":
        record["hashes"]["source_files"][numerical.E08_CONSUMERS[0]] = "0" * 64
    elif fault == "wrong_checker":
        record["hashes"]["checks_inventory"] = "0" * 64
    else:
        (root /
         "posthoc/opaque123/repository/example.py").write_text("changed\n")
    with pytest.raises(ValueError, match="incomplete or unbound"):
        checker._bind_inputs(check_result,
                             output,
                             root / "posthoc/opaque123/repository",
                             record,
                             case_id="E08")


def test_e08_cli_runs_one_bound_attempt(checker, tmp_path, monkeypatch,
                                        capsys):
    root, _, _, _ = fixture_run(tmp_path, case="E08")
    binding = e08_binding(tmp_path / "e08-binding.json")
    count = numerical.CHECK_SUITES["E08"]["targeted"][0]
    calls = docker_boundary(monkeypatch,
                            checker,
                            targeted=f"{count} passed in 1s\n".encode())
    assert checker.main([
        "--run-dir",
        str(root), "--attempt-id", "opaque123", "--e08-binding",
        str(binding)
    ]) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 1 and records[0]["attempt_id"] == "opaque123"
    assert records[0]["targeted"] == records[0]["regression"] == "pass"
    assert len(calls) == 2


def test_e08_cli_binding_guards_refuse_wrong_case_multiple_bindings_and_all_terminal(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E03")
    binding = e08_binding(tmp_path / "e08-binding.json")
    calls = docker_boundary(monkeypatch, checker)
    invocations = [
        [
            "--run-dir",
            str(root), "--all-terminal", "--e08-binding",
            str(binding)
        ],
        [
            "--run-dir",
            str(root), "--attempt-id", "opaque123", "--e08-binding",
            str(binding)
        ],
        [
            "--run-dir",
            str(root), "--attempt-id", "opaque123", "--e05-binding",
            str(binding), "--e08-binding",
            str(binding)
        ],
    ]
    for argv in invocations:
        with pytest.raises(SystemExit):
            checker.main(argv)
    assert not (root / "posthoc").exists()
    assert not calls


def test_existing_outputs_are_not_overwritten_or_reused_as_fresh_evidence(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path)
    output = root / "posthoc/opaque123"
    output.mkdir(parents=True)
    (output / "sentinel").write_bytes(b"retain")
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == "unknown"
    assert result["failures"] == ["posthoc_output_exists"]
    assert sorted(p.name for p in output.iterdir()) == ["sentinel"]
    assert not calls


@pytest.mark.parametrize("unsafe", [
    "source_changed", "capture_helper_changed", "worktree_symlink",
    "trial_symlink"
])
def test_untrusted_inputs_fail_closed_without_container_access(
        checker, tmp_path, monkeypatch, unsafe):
    root, source, worktree, trial = fixture_run(tmp_path, original=False)
    if unsafe == "source_changed":
        (source / "example.py").write_text("tampered source\n")
    elif unsafe == "capture_helper_changed":
        data = json.loads((root / "manifest.json").read_text())
        data["input_hashes"]["capture"] = "0" * 64
        freeze_manifest(root, data)
    elif unsafe == "worktree_symlink":
        moved = trial / "agent/moved"
        worktree.rename(moved)
        worktree.symlink_to(moved, target_is_directory=True)
    else:
        moved = root / "moved-trial"
        trial.rename(moved)
        trial.symlink_to(moved, target_is_directory=True)
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == "unknown"
    assert result["capture_status"] == "unknown"
    assert result["failures"]
    assert not calls


def test_recovery_timeout_stops_its_own_container_and_retains_unknown_record(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, original=False)
    calls = docker_boundary(monkeypatch, checker, timeout=True)
    result = checker.check_attempt(root, "opaque123")
    assert result["capture_status"] == "unknown"
    assert "recovery_timeout" in result["failures"]
    assert len(calls) == 2
    assert calls[1][0] == [
        "docker", "kill", calls[0][0][calls[0][0].index("--name") + 1]
    ]
    assert (root / "posthoc/opaque123/artifact_checks.json").is_file()


def test_all_terminal_cli_leaves_pending_attempts_untouched(
        checker, tmp_path, monkeypatch, capsys):
    root, _, _, _ = fixture_run(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["attempts"].append({
        "attempt_id": "pending",
        "job_name": "pending"
    })
    freeze_manifest(root, manifest)
    calls = docker_boundary(monkeypatch, checker)
    assert checker.main(["--run-dir", str(root), "--all-terminal"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [r["attempt_id"] for r in result] == ["opaque123"]
    assert not (root / "posthoc/pending").exists()
    assert not calls


def test_regression_coverage_cannot_pass_with_five_skips(
        checker, tmp_path, monkeypatch):
    root, _, _, _ = fixture_run(tmp_path, case="E03")
    docker_boundary(monkeypatch,
                    checker,
                    regression=b"306 passed, 5 skipped in 1s\n")
    result = checker.check_attempt(root, "opaque123")
    assert result["targeted"] == "pass"
    assert result["regression"] == result["status"] == "unknown"


@pytest.mark.parametrize("field", ["task_name", "trial_name"])
def test_misplaced_terminal_result_cannot_authorize_a_different_trial(
        checker, tmp_path, monkeypatch, field):
    root, _, _, trial = fixture_run(tmp_path)
    write_json(trial / "result.json", {
        "finished_at": "yes",
        field: "another-trial"
    })
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == result["capture_status"] == "unknown"
    assert not (root / "posthoc").exists()
    assert not calls


@pytest.mark.parametrize("manifest", [[], {
    "schema_version": 1,
    "attempts": [None]
}])
def test_malformed_manifest_is_explicitly_unknown(checker, tmp_path,
                                                  monkeypatch, manifest):
    root, _, _, _ = fixture_run(tmp_path)
    freeze_manifest(root, manifest)
    calls = docker_boundary(monkeypatch, checker)
    result = checker.check_attempt(root, "opaque123")
    assert result["status"] == "unknown"
    assert result["failures"] == ["terminal_trial_or_manifest_unavailable"]
    assert not calls


def test_native_harbor_identity_uses_namespaced_task_and_independent_trial_uuid(
        checker, tmp_path, monkeypatch):
    root, source, _, trial = fixture_run(tmp_path)
    write_json(
        trial / "result.json", {
            "finished_at": "yes",
            "id": "unrelated-trial-uuid",
            "task_name": "nvidia/skillevaluator-opaque123",
            "trial_name": trial.name,
            "task_id": {
                "path": str(source.parent.parent)
            }
        })
    docker_boundary(monkeypatch, checker)
    assert checker.check_attempt(root,
                                 "opaque123")["capture_status"] == "complete"
