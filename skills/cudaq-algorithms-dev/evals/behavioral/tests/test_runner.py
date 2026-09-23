# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import behavioral_run
from behavioral_cases import stage_case
from behavioral_run import (
    DEFAULT_PYTHON,
    applicable_rubric,
    behavioral_sandbox_command,
    native_status,
    overall_outcome,
    pending_runs,
    prepare_campaign,
    report_campaign,
    verify_campaign,
)
from runtime import environment


def test_prepare_freezes_cases_fixtures_evaluator_and_balanced_plan(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        behavioral_run, "_runtime_provenance", lambda _python: {
            "python": {
                "sha256": "p",
                "version": "Python 3.12.3"
            },
            "codex": {
                "sha256": "c",
                "version": "codex-cli 0.144.4"
            },
        })
    monkeypatch.setattr(
        behavioral_run, "_worker_runtime_metadata", lambda _python: {
            "prefix": "/clean",
            "base_prefix": "/usr",
            "discoverable": {
                "cudaq": False,
                "cudaq_algorithms": False
            },
        })
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.txt").write_text("fixed")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("VALUE = 1\n")
    cases = [{
        "id": "one",
        "files": ["fixture.txt"]
    }, {
        "id": "two",
        "files": []
    }]
    output = tmp_path / "campaign"
    manifest = prepare_campaign(output,
                                cases,
                                source, [evaluator],
                                repetitions=3,
                                seed=4,
                                python="/clean/python")
    assert manifest["stop_on_pass"] is False
    assert manifest["grading_masking"].startswith("arm identifier")
    assert manifest["runtime_provenance"]["python"]["sha256"]
    assert manifest["runtime_provenance"]["codex"]["version"].startswith(
        "codex-cli ")
    assert len(manifest["runs"]) == 12
    verify_campaign(output, cases, source, [evaluator], "/clean/python")

    (source / "fixture.txt").write_text("drift")
    with pytest.raises(RuntimeError, match="fixture drift"):
        verify_campaign(output, cases, source, [evaluator], "/clean/python")


def test_cli_canonicalizes_relative_output_before_prepare_and_run(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = []
    monkeypatch.setattr(behavioral_run, "load_all_cases", lambda _include: [])
    monkeypatch.setattr(
        behavioral_run, "prepare_campaign",
        lambda output, *_args, **_kwargs: captured.append(("prepare", output)))
    behavioral_run.main([
        "prepare", "--output", "relative-campaign", "--python", "/clean/python"
    ])
    assert captured[-1] == ("prepare",
                            (tmp_path / "relative-campaign").resolve())

    (tmp_path / "relative-campaign").mkdir()
    monkeypatch.setattr(behavioral_run, "verify_campaign",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        behavioral_run, "run_workers",
        lambda output, *_args, **_kwargs: captured.append(("run", output)))
    behavioral_run.main(
        ["run", "--output", "relative-campaign", "--python", "/clean/python"])
    assert captured[-1] == ("run", (tmp_path / "relative-campaign").resolve())


def test_preflight_uses_absolute_manifest_forbidden_path_from_relative_api_root(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = Path("relative-campaign")
    output.mkdir()
    snapshot = tmp_path / "staging/candidate"
    (output / "manifest.json").write_text(
        json.dumps({"candidate_snapshot": str(snapshot)}))
    scripts = []

    def fake_stage(_case, workspace, _fixtures, **_kwargs):
        workspace.mkdir(parents=True)
        (workspace / "input").mkdir()
        (workspace / "input/evidence.txt").write_text("fixed")
        (workspace / ".tmp").mkdir()

    def fake_sandbox(_workspace, _python, args):
        scripts.append(args[-1])
        return ["sandbox-probe"]

    class FakeListener:

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def bind(self, _address):
            return None

        def listen(self, _count):
            return None

        def getsockname(self):
            return ("127.0.0.1", 12345)

    monkeypatch.setattr(behavioral_run, "stage_case", fake_stage)
    monkeypatch.setattr(behavioral_run, "behavioral_sandbox_command",
                        fake_sandbox)
    monkeypatch.setattr(behavioral_run.socket, "socket", FakeListener)
    monkeypatch.setattr(
        behavioral_run.subprocess, "run", lambda *_args, **_kwargs:
        SimpleNamespace(returncode=0, stdout="{}\n", stderr=""))
    behavioral_run.preflight_campaign(output, [{
        "files": ["files/evidence.txt"]
    }], "/clean/python")
    expected = str((tmp_path / "relative-campaign/manifest.json").resolve())
    assert scripts and all(expected in script for script in scripts)


def test_worker_command_uses_absolute_final_path_from_relative_api_root(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = Path("relative-campaign")
    output.mkdir()
    captured = {}

    class FakeCollector:
        config_args = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def summary(self):
            return {
                "native_turn_completed": True,
                "totals": {
                    "total_tokens": 1
                }
            }

    def fake_stage(_case, workspace, _fixtures, **_kwargs):
        workspace.mkdir(parents=True)
        (workspace / ".tmp").mkdir()

    def fake_native(command, _workspace, _python):
        captured["command"] = command
        return command

    def fake_run(command, _workspace, _prompt, destination, _timeout,
                 _collector):
        (destination / "events.jsonl").write_text("")
        (destination / "final.txt").write_text("done\n")
        return {"returncode": 0, "timed_out": False, "elapsed_s": 1}

    monkeypatch.setattr(behavioral_run, "TelemetryCollector", FakeCollector)
    monkeypatch.setattr(behavioral_run, "stage_case", fake_stage)
    monkeypatch.setattr(behavioral_run, "behavioral_inventory",
                        lambda _path: {})
    monkeypatch.setattr(behavioral_run, "build_prompt",
                        lambda *_args: "prompt")
    monkeypatch.setattr(behavioral_run, "behavioral_configuration",
                        lambda *_args: [])
    monkeypatch.setattr(behavioral_run, "native_command", fake_native)
    monkeypatch.setattr(behavioral_run, "run_native", fake_run)
    monkeypatch.setattr(behavioral_run, "artifact_checks",
                        lambda *_args: {"passed": True})
    monkeypatch.setattr(behavioral_run, "diagnose_read_assertions",
                        lambda *_args: [])
    run = {
        "id": "x--1--baseline",
        "case": "x",
        "repetition": 1,
        "arm": "baseline"
    }
    case = {
        "id": "x",
        "files": [],
        "required_writes": [],
        "optional_writes": []
    }
    manifest = {
        "candidate_snapshot": str(tmp_path / "staging/candidate"),
        "model": "gpt-5.5",
        "worker_timeout_s": 480
    }
    behavioral_run._worker_result(run, case, manifest, output, "/clean/python")
    command = captured["command"]
    final_path = Path(command[command.index("-o") + 1])
    assert final_path == (
        tmp_path /
        "relative-campaign/runs/x--1--baseline/final.txt").resolve()


def test_partial_resume_never_retries_started_attempt(tmp_path):
    runs = [{"id": "new"}, {"id": "done"}, {"id": "partial"}]
    root = tmp_path / "runs"
    (root / "done").mkdir(parents=True)
    (root / "done/result.json").write_text("{}")
    (root / "partial").mkdir()
    assert pending_runs(runs, root) == ([runs[0]], [runs[2]])


def test_interrupted_worker_preserves_attempt_and_available_telemetry(
        tmp_path):
    output = tmp_path / "campaign"
    partial = output / "runs/x--1--baseline"
    partial.mkdir(parents=True)
    (partial / "command.json").write_text("[]\n")
    telemetry = {
        "native_turn_completed": False,
        "totals": {
            "total_tokens": 17
        }
    }
    (partial / "worker-telemetry.json").write_text(json.dumps(telemetry))
    run = {
        "id": "x--1--baseline",
        "case": "x",
        "repetition": 1,
        "arm": "baseline"
    }
    (output / "manifest.json").write_text(
        json.dumps({
            "runs": [run],
            "parallel_pairs": 1
        }))
    (output / "preflight.json").write_text(json.dumps([{"passed": True}]))
    behavioral_run.run_workers(output, [{"id": "x"}], DEFAULT_PYTHON)
    result = json.loads((partial / "result.json").read_text())
    assert result["worker_attempted"]
    assert result["worker_telemetry"] == telemetry


def test_worker_exception_finalizes_failure_and_continues_pair(
        tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    output.mkdir()
    runs = [
        {
            "id": "x--1--baseline",
            "case": "x",
            "repetition": 1,
            "arm": "baseline"
        },
        {
            "id": "x--1--candidate",
            "case": "x",
            "repetition": 1,
            "arm": "candidate"
        },
    ]
    (output / "manifest.json").write_text(
        json.dumps({
            "runs": runs,
            "parallel_pairs": 1
        }))
    (output / "preflight.json").write_text(json.dumps([{"passed": True}]))

    def fake(run, _case, _manifest, root, _python):
        destination = root / "runs" / run["id"]
        destination.mkdir(parents=True)
        if run["arm"] == "baseline":
            raise RuntimeError("boom")
        (destination / "result.json").write_text(
            json.dumps({
                **run, "worker_attempted": True
            }))

    monkeypatch.setattr(behavioral_run, "_worker_result", fake)
    behavioral_run.run_workers(output, [{"id": "x"}], DEFAULT_PYTHON)
    failed = json.loads(
        (output / "runs/x--1--baseline/result.json").read_text())
    continued = json.loads(
        (output / "runs/x--1--candidate/result.json").read_text())
    assert failed["status"] == "infrastructure_failure"
    assert continued["worker_attempted"]


def test_interrupted_grader_is_finalized_without_retry(tmp_path):
    output = tmp_path / "campaign"
    destination = output / "runs/x--1--baseline"
    (destination / "grader").mkdir(parents=True)
    run = {
        "id": "x--1--baseline",
        "case": "x",
        "repetition": 1,
        "arm": "baseline"
    }
    (output / "manifest.json").write_text(
        json.dumps({
            "runs": [run],
            "parallel_pairs": 1,
            "seed": 1,
            "candidate_snapshot": str(tmp_path / "skill"),
        }))
    (destination / "result.json").write_text(
        json.dumps({
            **run,
            "worker_attempted": True,
            "worker_completed": True,
            "grader_attempted": False,
        }))
    behavioral_run.grade_attempts(output, [{
        "id": "x",
        "prompt": "x",
        "shared_assertions": []
    }], DEFAULT_PYTHON)
    result = json.loads((destination / "result.json").read_text())
    assert result["grader_attempted"]
    assert not result["grader_completed"]
    assert result["status"] == "infrastructure_failure"


def test_native_status_requires_zero_exit_no_timeout_and_final_turn():
    complete = {"returncode": 0, "timed_out": False}
    assert native_status(complete,
                         {"native_turn_completed": True}) == "completed"
    assert native_status(
        complete, {"native_turn_completed": False}) == "infrastructure_failure"
    assert native_status({
        "returncode": 1,
        "timed_out": False
    }, {"native_turn_completed": True}) == "infrastructure_failure"
    assert native_status({
        "returncode": -15,
        "timed_out": True
    }, {"native_turn_completed": False}) == "model_failure"
    assert overall_outcome(worker_completed=True,
                           grader_status="completed",
                           semantic_passed=True,
                           deterministic_passed=True)
    assert not overall_outcome(worker_completed=False,
                               grader_status="completed",
                               semantic_passed=True,
                               deterministic_passed=True)
    assert not overall_outcome(worker_completed=True,
                               grader_status="infrastructure_failure",
                               semantic_passed=True,
                               deterministic_passed=True)


def test_grading_rubric_does_not_expose_superseded_original_text():
    case = {
        "shared_assertions": [{
            "id":
            "a1",
            "text":
            "identify both ambiguities",
            "original_text":
            "activate skill and identify both ambiguities"
        }]
    }
    assert applicable_rubric(case) == [{
        "id": "a1",
        "text": "identify both ambiguities"
    }]


def test_behavioral_sandbox_denies_fixture_overwrite_and_chmod(tmp_path):
    worker_python = os.environ.get("BEHAVIORAL_WORKER_PYTHON")
    if not worker_python:
        pytest.skip(
            "set BEHAVIORAL_WORKER_PYTHON to the clean stdlib-only venv")
    fixture = tmp_path / "fixtures/files"
    fixture.mkdir(parents=True)
    (fixture / "evidence.txt").write_text("fixed\n")
    workspace = tmp_path / "workspace"
    stage_case(
        {
            "id": "probe",
            "files": ["files/evidence.txt"],
            "authorized_targets": {}
        },
        workspace,
        fixture.parent,
        arm="baseline")
    script = (
        "import json,pathlib,stat\n"
        "p=pathlib.Path('input/evidence.txt'); before=p.read_bytes(); out={}\n"
        "try: p.write_bytes(b'changed'); out['write']=False\n"
        "except OSError: out['write']=p.read_bytes()==before\n"
        "try: p.chmod(p.stat().st_mode|stat.S_IWUSR); out['chmod']=False\n"
        "except OSError: out['chmod']=not bool(p.stat().st_mode&stat.S_IWUSR)\n"
        "print(json.dumps(out)); assert all(out.values()),out\n")
    completed = subprocess.run(
        behavioral_sandbox_command(workspace, worker_python, ["-c", script]),
        cwd=workspace,
        env=environment(workspace),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stderr


def test_authoring_set_is_separate_and_contains_two_real_edits_and_one_advisory(
):
    path = Path(__file__).resolve().parents[1] / "authoring_cases.json"
    cases = json.loads(path.read_text())["cases"]
    assert [case["mode"] for case in cases].count("implementation") == 2
    assert [case["mode"] for case in cases].count("advisory") == 1
    assert all(case["suite"] == "authoring" for case in cases)
    assert all(case["files"] for case in cases)


def test_report_separates_workflows_reads_and_grader_resources(tmp_path):
    output = tmp_path / "campaign"
    runs = output / "runs"
    runs.mkdir(parents=True)
    manifest = {
        "suite":
        "authored_behavioral_with_separately_scored_authoring",
        "runs": [
            {
                "id": "original--1--candidate",
                "case": "original",
                "repetition": 1,
                "arm": "candidate"
            },
            {
                "id": "author--1--baseline",
                "case": "author",
                "repetition": 1,
                "arm": "baseline"
            },
        ],
        "cases": [
            {
                "id": "original",
                "read_assertions": [{
                    "id": "a1",
                    "kind": "skill_read"
                }]
            },
            {
                "id": "author",
                "suite": "authoring",
                "read_assertions": []
            },
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    rows = [
        {
            "id":
            "original--1--candidate",
            "case":
            "original",
            "repetition":
            1,
            "arm":
            "candidate",
            "worker_attempted":
            True,
            "grader_attempted":
            True,
            "worker_elapsed_s":
            3,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 10
                },
                "observed_peak_request_input_tokens": 7,
                "completeness": {
                    "observed_peak_request_input_tokens": True
                }
            },
            "grader_elapsed_s":
            20,
            "grader_telemetry": {
                "totals": {
                    "total_tokens": 100
                }
            },
            "read_diagnostics": [{
                "id": "a1",
                "kind": "skill_read",
                "skill": {
                    "status": "observed"
                },
                "fixtures": {}
            }],
            "semantic": {
                "passed": True,
                "assertions": [{
                    "id": "a2",
                    "verdict": "PASS"
                }]
            },
            "deterministic": {
                "passed": True
            },
            "overall_pass":
            True
        },
        {
            "id": "author--1--baseline",
            "case": "author",
            "repetition": 1,
            "arm": "baseline",
            "worker_attempted": True,
            "grader_attempted": True,
            "worker_elapsed_s": 2,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": None
                },
                "observed_peak_request_input_tokens": None,
                "completeness": {
                    "observed_peak_request_input_tokens": False
                }
            },
            "grader_elapsed_s": 10,
            "grader_telemetry": {
                "totals": {
                    "total_tokens": 50
                }
            },
            "read_diagnostics": [],
            "semantic": {
                "passed": False,
                "assertions": []
            },
            "deterministic": {
                "passed": True
            },
            "overall_pass": False
        },
    ]
    for row in rows:
        path = runs / row["id"]
        path.mkdir()
        (path / "result.json").write_text(json.dumps(row))
    summary = report_campaign(output)
    assert summary["workflows"]["original"]["overall_passes"] == 1
    assert summary["workflows"]["authoring"]["overall_passes"] == 0
    assert summary["read_diagnostics"]["planned_assertion_checks"] == 1
    assert summary["read_diagnostics"]["skill_status_counts"] == {
        "observed": 1
    }
    assert summary["resource_completeness"]["worker_total_tokens"] == "1/2"
    assert len(summary["grader_resources"]) == 2
