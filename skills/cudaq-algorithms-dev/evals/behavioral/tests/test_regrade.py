# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

import behavioral_regrade as subject


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _source_campaign(root, *, finalized=True):
    cases = [
        {
            "id": "original-case",
            "prompt": "Assess the original task.",
            "shared_assertions": [{
                "id": "a1",
                "text": "The answer is supported."
            }]
        },
        {
            "id": "authoring-case",
            "prompt": "Assess the authoring task.",
            "suite": "authoring",
            "shared_assertions": [{
                "id": "a1",
                "text": "The artifact is valid."
            }]
        },
    ]
    runs = []
    for index in range(270):
        case = cases[index % 2]["id"]
        arm = ("baseline", "candidate")[index % 2]
        run = {
            "id": f"attempt-{index:03d}",
            "case": case,
            "repetition": index // 2 + 1,
            "arm": arm
        }
        runs.append(run)
        destination = root / "runs" / run["id"]
        destination.mkdir(parents=True)
        (destination / "final.txt").write_text(f"answer {index}\n")
        (destination / "events.jsonl").write_text("")
        if index == 0:
            (destination / "initial-artifacts").mkdir()
            (destination / "initial-artifacts" /
             "answer.txt").write_text("before\n")
            (destination / "artifacts").mkdir()
            (destination / "artifacts" / "answer.txt").write_text("after\n")
        result = {
            **run,
            "worker_attempted": True,
            "worker_completed": True,
            "worker_elapsed_s": index / 10,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": index
                }
            },
            "deterministic": {
                "passed": index % 3 != 0
            },
            "grader_attempted": finalized or index != 269,
            "semantic": {
                "valid": True,
                "passed": True
            },
        }
        _save(destination / "result.json", result)
    _save(
        root / "manifest.json", {
            "kind": "source behavioral campaign",
            "model": "gpt-5.5",
            "seed": 123,
            "cases": cases,
            "runs": runs,
        })
    return runs, cases


@pytest.fixture
def frozen_provenance(monkeypatch):
    monkeypatch.setattr(subject, "_runtime_provenance", lambda python: {
        "python": python,
        "clean": True
    })
    monkeypatch.setattr(subject, "_evaluator_inventory",
                        lambda: {"behavioral_regrade.py": "evaluator-hash"})


def test_stable_evidence_ids_preserve_projected_text(tmp_path):
    run = tmp_path / "run"
    (run / "artifacts").mkdir(parents=True)
    (run / "initial-artifacts").mkdir()
    (run / "final.txt").write_text("first\nsecond\n")
    (run / "events.jsonl").write_text(
        json.dumps({
            "event": {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python check.py",
                    "exit_code": 0,
                    "aggregated_output": "ok"
                }
            }
        }) + "\n")
    (run / "initial-artifacts" / "answer.txt").write_text("old\n")
    (run / "artifacts" / "answer.txt").write_text("new\n")

    evidence = subject.stable_evidence(run)

    assert list(evidence) == [
        "F0001", "F0002", "C0001", "A0001", "A0002", "A0003", "A0004", "A0005",
        "A0006", "A0007", "A0008"
    ]
    assert evidence["F0001"] == "first"
    assert evidence["F0002"] == "second"
    assert evidence["C0001"] == "command=python check.py exit=0 output=ok"
    assert evidence["A0001"] == "DIFF answer.txt"
    assert evidence["A0008"] == "new"


def test_schema_uses_closed_id_enums_and_postparse_rejects_bad_rows():
    rubric = [{"id": "a1", "text": "one"}, {"id": "a2", "text": "two"}]
    evidence = {"F0001": "answer", "F0002": "  ", "C0001": "command"}
    schema = subject.grader_schema(rubric, evidence)
    row = schema["properties"]["assertions"]["items"]["properties"]
    assert row["id"] == {"type": "string", "enum": ["a1", "a2"]}
    assert row["evidence"]["items"] == {
        "type": "string",
        "enum": ["F0001", "C0001"]
    }

    valid = {
        "assertions": [
            {
                "id": "a1",
                "verdict": "PASS",
                "reason": "direct",
                "evidence": ["F0001"]
            },
            {
                "id": "a2",
                "verdict": "FAIL",
                "reason": "contradicted",
                "evidence": ["C0001"]
            },
        ]
    }
    checked = subject.validate_grade(valid, rubric, evidence)
    assert checked["valid"] and not checked["passed"]
    assert subject.validate_grade(
        {"assertions": list(reversed(valid["assertions"]))}, rubric,
        evidence)["valid"]
    for broken in (
        {
            "assertions": [valid["assertions"][0], valid["assertions"][0]]
        },
        {
            "assertions": [valid["assertions"][0]]
        },
        {
            "assertions": [
                valid["assertions"][0], {
                    **valid["assertions"][1], "reason": " "
                }
            ]
        },
        {
            "assertions": [
                valid["assertions"][0], {
                    **valid["assertions"][1], "evidence": []
                }
            ]
        },
        {
            "assertions": [
                valid["assertions"][0], {
                    **valid["assertions"][1], "evidence": ["F0002"]
                }
            ]
        },
        {
            "assertions": [
                valid["assertions"][0], {
                    **valid["assertions"][1], "evidence": ["F9999"]
                }
            ]
        },
    ):
        assert not subject.validate_grade(broken, rubric, evidence)["valid"]


def test_projection_is_deterministic_and_masks_arm_paths_and_run_identity():
    source = {
        "id": "hidden--candidate",
        "arm": "candidate",
        "case": "original-case",
        "repetition": 2,
        "workspace": "/tmp/private/candidate",
        "deterministic": {
            "passed": False,
            "detail": "/tmp/private/candidate/hidden--candidate"
        },
    }
    case = {"id": "original-case", "prompt": "Public task"}
    evidence = {
        "F0001":
        "failure at /tmp/cudaq-behavioral-campaign-AAA/runs/x--candidate"
    }
    first = subject.project_attempt(source, case, evidence, salt="fixed")
    second = subject.project_attempt(source, case, evidence, salt="fixed")
    assert first == second
    rendered = json.dumps(first)
    assert "hidden--candidate" not in rendered
    assert "/tmp/private" not in rendered
    assert "--candidate" not in rendered
    assert first["deterministic"]["passed"] is False
    assert first["evidence"]["F0001"] == "failure at <campaign>/runs/x--<arm>"


def test_prepare_freezes_every_source_input_and_detects_drift(
        tmp_path, frozen_provenance):
    source, output = tmp_path / "source", tmp_path / "corrected"
    _source_campaign(source)
    manifest = subject.prepare_campaign(source, output, "/clean/python")

    assert manifest["source_manifest"]["sha256"]
    assert len(manifest["runs"]) == 270
    inventory = manifest["source_inventory"]
    assert sum(name.endswith("/result.json") for name in inventory) == 270
    assert sum(name.endswith("/final.txt") for name in inventory) == 270
    assert sum(name.endswith("/events.jsonl") for name in inventory) == 270
    assert "runs/attempt-000/artifacts/answer.txt" in inventory
    assert "runs/attempt-000/initial-artifacts/answer.txt" in inventory
    assert manifest["rubrics"]["original-case"] == [{
        "id":
        "a1",
        "text":
        "The answer is supported."
    }]

    (source / "runs" / "attempt-000" / "final.txt").write_text("drift\n")
    with pytest.raises(RuntimeError, match="source evidence drift"):
        subject.verify_campaign(output, "/clean/python")


def test_prepare_refuses_unfinished_or_wrong_sized_source(
        tmp_path, frozen_provenance):
    unfinished, output = tmp_path / "unfinished", tmp_path / "output"
    _source_campaign(unfinished, finalized=False)
    with pytest.raises(RuntimeError,
                       match="source grading session is not finalized"):
        subject.prepare_campaign(unfinished, output, "/clean/python")

    wrong, output2 = tmp_path / "wrong", tmp_path / "output2"
    runs, _ = _source_campaign(wrong)
    manifest = json.loads((wrong / "manifest.json").read_text())
    manifest["runs"] = runs[:-1]
    _save(wrong / "manifest.json", manifest)
    with pytest.raises(ValueError, match="exactly 270"):
        subject.prepare_campaign(wrong, output2, "/clean/python")

    extra, output3 = tmp_path / "extra", tmp_path / "output3"
    _source_campaign(extra)
    (extra / "runs" / "not-in-manifest").mkdir()
    with pytest.raises(RuntimeError, match="worker IDs do not exactly match"):
        subject.prepare_campaign(extra, output3, "/clean/python")


def test_timeout_nonzero_and_invalid_output_are_measurement_failures():
    rubric = [{"id": "a1", "text": "one"}]
    evidence = {"F0001": "answer"}
    grade = {
        "assertions": [{
            "id": "a1",
            "verdict": "FAIL",
            "reason": "wrong",
            "evidence": ["F0001"]
        }]
    }
    cases = [
        ({
            "timed_out": True,
            "returncode": None
        }, {
            "native_turn_completed": False
        }, grade),
        ({
            "timed_out": False,
            "returncode": 7
        }, {
            "native_turn_completed": True
        }, grade),
        ({
            "timed_out": False,
            "returncode": 0
        }, {
            "native_turn_completed": True
        }, {}),
    ]
    for execution, telemetry, raw in cases:
        result = subject.classify_grade(execution, telemetry, raw, rubric,
                                        evidence)
        assert result["status"] == "measurement_failure"
        assert not result["valid_grade"]
        assert "model_failure" not in json.dumps(result)
    assert cases[2][1]["native_turn_completed"]
    assert subject.classify_grade(*cases[2], rubric,
                                  evidence)["native_completed"]

    completed = subject.classify_grade({
        "timed_out": False,
        "returncode": 0
    }, {"native_turn_completed": True}, grade, rubric, evidence)
    assert completed["status"] == "completed"
    assert completed["valid_grade"]
    assert not completed["semantic"]["passed"]


def test_strict_pass_is_the_full_worker_deterministic_and_grade_conjunction():
    source = {
        "worker_completed": True,
        "worker_execution": {
            "returncode": 0,
            "timed_out": False
        },
        "deterministic": {
            "passed": True
        }
    }
    corrected = {
        "status": "completed",
        "valid_grade": True,
        "semantic": {
            "valid": True,
            "passed": True
        }
    }
    assert subject._strict_pass(source, corrected)
    for broken_source in (
        {
            **source, "worker_completed": False
        },
        {
            **source, "worker_execution": {
                "returncode": 1,
                "timed_out": False
            }
        },
        {
            **source, "deterministic": {
                "passed": False
            }
        },
    ):
        assert not subject._strict_pass(broken_source, corrected)
    assert not subject._strict_pass(
        source, {
            **corrected, "status": "measurement_failure"
        })
    assert not subject._strict_pass(source, {
        **corrected, "semantic": {
            "valid": True,
            "passed": False
        }
    })


def test_interrupted_attempt_is_finalized_once_and_never_retried(
        tmp_path, monkeypatch):
    output = tmp_path / "corrected"
    source = tmp_path / "source"
    run = {"id": "one", "case": "case", "repetition": 1, "arm": "baseline"}
    _save(source / "runs" / "one" / "result.json", {
        **run, "worker_attempted": True
    })
    manifest = {
        "runs": [run],
        "source_root": str(source),
        "parallel": 8,
        "timeout_s": 480,
        "rubrics": {
            "case": [{
                "id": "a1",
                "text": "x"
            }]
        },
        "cases": {
            "case": {
                "id": "case",
                "prompt": "task"
            }
        },
        "seed": 1,
        "model": "gpt-5.5",
        "staging_root": str(tmp_path / "stage")
    }
    _save(output / "manifest.json", manifest)
    (output / "runs" / "one" / "grader").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(subject, "verify_campaign",
                        lambda output, python: manifest)
    monkeypatch.setattr(subject, "_execute_grade",
                        lambda *args: calls.append(args))

    subject.run_campaign(output, "/clean/python")
    first = json.loads((output / "runs" / "one" / "result.json").read_text())
    subject.run_campaign(output, "/clean/python")

    assert calls == []
    assert first["status"] == "measurement_failure"
    assert first["attempt_reserved"]
    assert not first["grader_attempted"] and not first["valid_grade"]
    assert "not retried" in first["reason"]

    prelaunch = output / "runs" / "prelaunch"
    (prelaunch / "grader").mkdir(parents=True)
    _save(prelaunch / "grader" / "command.json", ["codex", "exec"])
    not_launched = subject._measurement_result(
        {
            "id": "prelaunch",
            "case": "case",
            "repetition": 2,
            "arm": "baseline"
        }, prelaunch, "Popen failed; not retried")
    assert not_launched["grader_started"]
    assert not not_launched["grader_attempted"]

    retained = output / "runs" / "retained"
    (retained / "grader").mkdir(parents=True)
    _save(retained / "grader" / "command.json", ["codex", "exec"])
    execution = {"returncode": 0, "timed_out": False, "elapsed_s": 9.25}
    telemetry = {"native_turn_completed": True, "totals": {"total_tokens": 17}}
    _save(retained / "grader" / "execution.json", execution)
    _save(retained / "grader" / "telemetry.json", telemetry)
    recovered = subject._measurement_result(
        {
            "id": "retained",
            "case": "case",
            "repetition": 2,
            "arm": "candidate"
        }, retained, "interrupted; not retried")
    assert recovered["grader_attempted"] and recovered["grader_completed"]
    assert recovered["grader_elapsed_s"] == 9.25
    assert recovered["grader_execution"] == execution
    assert recovered["grader_telemetry"] == telemetry

    events_only = output / "runs" / "events-only"
    (events_only / "grader").mkdir(parents=True)
    _save(events_only / "grader" / "command.json", ["codex", "exec"])
    wrapped = [
        {
            "elapsed_s": 0.2,
            "event": {
                "type": "turn.started"
            }
        },
        {
            "elapsed_s": 6.5,
            "event": {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 7,
                    "reasoning_output_tokens": 3,
                    "total_tokens": 107
                }
            }
        },
    ]
    (events_only / "grader" / "events.jsonl").write_text("".join(
        json.dumps(item) + "\n" for item in wrapped))
    reconstructed = subject._measurement_result(
        {
            "id": "events-only",
            "case": "case",
            "repetition": 3,
            "arm": "baseline"
        }, events_only, "interrupted; not retried")
    assert reconstructed["grader_attempted"]
    assert reconstructed["native_turn_completed"]
    assert not reconstructed[
        "grader_completed"]  # native process exit status was not retained
    assert reconstructed["grader_telemetry"]["totals"]["total_tokens"] == 107
    assert reconstructed["grader_elapsed_s"] == 6.5
    assert reconstructed["grader_elapsed_is_lower_bound"]
    assert "exit status" in reconstructed["grader_execution"]["missing_reason"]


def test_report_separates_imported_worker_and_corrected_grader_kpis(tmp_path):
    output, source = tmp_path / "corrected", tmp_path / "source"
    runs = [
        {
            "id": "o-b",
            "case": "o",
            "repetition": 1,
            "arm": "baseline"
        },
        {
            "id": "a-c",
            "case": "a",
            "repetition": 1,
            "arm": "candidate"
        },
    ]
    manifest = {
        "runs": runs,
        "source_root": str(source),
        "cases": {
            "o": {
                "id": "o",
                "prompt": "o"
            },
            "a": {
                "id": "a",
                "prompt": "a",
                "suite": "authoring"
            }
        },
        "diagnostic_role": "sensitivity analysis; not a replacement of v2"
    }
    _save(output / "manifest.json", manifest)
    workers = {
        "o-b": {
            **runs[0], "status": "model_failure",
            "worker_attempted": True,
            "worker_completed": True,
            "worker_elapsed_s": 2.5,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 11
                }
            },
            "deterministic": {
                "passed": True
            }
        },
        "a-c": {
            **runs[1], "worker_attempted": True,
            "worker_completed": True,
            "worker_elapsed_s": 3.5,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 13
                }
            },
            "deterministic": {
                "passed": False
            }
        },
    }
    for run in runs:
        _save(source / "runs" / run["id"] / "result.json", workers[run["id"]])
    _save(
        output / "runs" / "o-b" / "result.json", {
            **runs[0],
            "status": "completed",
            "valid_grade": True,
            "grader_attempted": True,
            "grader_completed": True,
            "semantic": {
                "valid": True,
                "passed": True,
                "assertions": [{
                    "id": "a1",
                    "verdict": "PASS"
                }]
            },
            "strict_pass": True,
            "grader_elapsed_s": 4,
            "grader_telemetry": {
                "totals": {
                    "total_tokens": 20
                }
            },
        })
    _save(
        output / "runs" / "a-c" / "result.json", {
            **runs[1],
            "status": "measurement_failure",
            "valid_grade": False,
            "grader_attempted": True,
            "grader_completed": False,
            "semantic": {
                "valid": False,
                "passed": False,
                "assertions": []
            },
            "strict_pass": False,
            "grader_elapsed_s": 5,
            "grader_telemetry": {
                "totals": {
                    "total_tokens": 21
                }
            },
        })

    report = subject.report_campaign(output)

    assert report["workflows"]["original"]["valid_grades"] == 1
    assert report["workflows"]["authoring"]["measurement_failures"] == 1
    assert report["arms"]["baseline"]["strict_passes"] == 1
    assert report["arms"]["candidate"]["strict_passes"] == 0
    assert report["native_grader_attempts"] == 2
    assert report["native_grader_completions"] == 1
    assert report["worker_resources"][0]["worker_telemetry"] == workers["a-c"][
        "worker_telemetry"] or report["worker_resources"][0][
            "worker_telemetry"] == workers["o-b"]["worker_telemetry"]
    assert all("grader_telemetry" not in item
               for item in report["worker_resources"])
    assert all("status" not in item for item in report["worker_resources"])
    assert all("worker_telemetry" not in item
               for item in report["corrected_grader_resources"])
    assert "not a replacement" in report["interpretation"]
