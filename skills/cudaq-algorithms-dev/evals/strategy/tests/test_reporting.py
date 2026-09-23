# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real collection/grade/report integration; only external model transport is fake.

Synthetic grades test binding and gates, never scientific task correctness.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

STRATEGY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY))
import judging
import posthoc
import suite
from test_posthoc import run_fixture, terminal, captured, check, write_json

SPEC = importlib.util.spec_from_file_location("strategy_reporting",
                                              STRATEGY / "reporting.py")
reporting = None
if Path(SPEC.origin).is_file():
    reporting = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(reporting)


@pytest.fixture
def evidence(tmp_path):
    root, attempts = run_fixture(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["purpose"] = "readiness-pilot"
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "manifest.sha256.json", {
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        })
    for attempt in attempts:
        trial = terminal(
            root,
            attempt,
            final="Synthetic fixture output, not an evaluated solution.")
        captured(root, attempt, trial)
        if attempt["case_id"] == "E03":
            check(root, attempt, "targeted", {"passed": 32})
            check(root, attempt, "regression", {"passed": 307, "skipped": 4})
    collection = tmp_path / "collection"
    assert posthoc.main(
        ["--run-dir", str(root), "--output-dir",
         str(collection)]) == 0
    return root, collection


def grade_fixture(evidence,
                  output,
                  *,
                  case_id="E01",
                  arm="baseline",
                  structured=False,
                  numbered=False,
                  bad_checks=False,
                  bad_status=False,
                  transport_failure=False,
                  token_counter=None,
                  expected_status=None):
    root, collection = evidence
    source = posthoc.collect(root, mapping_path=collection / "blinding.json")
    row = next(r for r in source["rows"]
               if r["case_id"] == case_id and r["arm"] == arm)
    payload = next(p for p in source["payloads"]
                   if p["attempt_key"] == row["attempt_key"])
    citation = [{"id": "final", "start": 1, "end": 1}]
    raw = {
        "attempt_key":
        row["attempt_key"],
        "total":
        10,
        "dimensions": [{
            "dimension": d["dimension"],
            "score": 2,
            "reason": "Synthetic validator fixture only.",
            "evidence": citation
        } for d in payload["score_rubric"]],
        "critical_checks": [{
            "id": c["id"],
            "verdict": "pass",
            "reason": "Synthetic validator fixture only.",
            "evidence": citation
        } for c in payload["critical_verdicts"]]
    }

    def request(**kwargs):
        if transport_failure:
            raise RuntimeError("synthetic non-retryable transport failure")
        return {
            "model":
            judging.MODEL,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120
            },
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(raw)
                }
            }]
        }

    checks = dict(row["independent_checks"])
    if bad_checks:
        checks["capture"] = "incomplete"
    result = judging.grade(
        payload,
        output_dir=output,
        api_key="synthetic-fixture-key",
        request=request,
        checks=checks,
        execution_status="failed" if bad_status else row["execution_status"],
        structured_output=structured,
        numbered_evidence=numbered,
        token_counter=token_counter)
    assert result["status"] == (
        expected_status
        or ("transport_error" if transport_failure else "graded"))
    return row


@pytest.mark.parametrize("structured,numbered", [(False, False), (True, False),
                                                 (False, True), (True, True)])
def test_saved_grade_is_bound_to_fresh_evidence_without_dropping_ungraded_rows(
        evidence, tmp_path, structured, numbered):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    grade_dir = tmp_path / "grade"
    original = grade_fixture(evidence,
                             grade_dir,
                             structured=structured,
                             numbered=numbered)
    result = reporting.report(root,
                              mapping_path=collection / "blinding.json",
                              grade_dirs=[grade_dir])
    assert len(result["rows"]) == 8
    row = next(r for r in result["rows"]
               if r["attempt_key"] == original["attempt_key"])
    assert row["grade"]["successful"] is True
    assert row["grade"]["raw_score"] == 10
    assert sum(r["grade"] is not None for r in result["rows"]) == 1
    assert all(r["usage"]["total_tokens"] == 12 for r in result["rows"])
    assert result["summary"]["judge"]["observed_total_tokens"] == 120
    assert result["summary"]["judge"]["missing_attempts"] == 3
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    assert result["provenance"]["manifest_freeze"]["manifest_sha256"] == digest
    confidence = result["summary"]["efficiency"]["confidence"]
    seed = int.from_bytes(
        hashlib.sha256(
            f'{confidence["method"]}:{digest}'.encode()).digest()[:8], "big")
    assert confidence["seed"] == seed
    assert confidence["statistical_pass"] is None
    assert result["summary"]["efficiency"]["statistical_pass"] is None
    for name in ("request.json", "response.json", "result.json"):
        assert result["provenance"]["input_hashes"][str(
            grade_dir / name)] == hashlib.sha256(
                (grade_dir / name).read_bytes()).hexdigest()


def test_failed_and_missing_grades_stay_missing_not_successful(
        evidence, tmp_path):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    grade_dir = tmp_path / "grade"
    original = grade_fixture(evidence, grade_dir, transport_failure=True)
    result = reporting.report(root,
                              mapping_path=collection / "blinding.json",
                              grade_dirs=[grade_dir])
    assert len(result["rows"]) == 8
    assert all(r["grade"] is None for r in result["rows"])
    row = next(r for r in result["rows"]
               if r["attempt_key"] == original["attempt_key"])
    assert row["judge"]["status"] == "transport_error"
    assert result["summary"]["judge"]["observed_total_tokens"] is None
    assert result["summary"]["judge"]["missing_attempts"] == 3
    assert result["summary"]["execution"]["baseline"]["mean_raw_score"] is None
    assert result["summary"]["efficiency"]["statistical_pass"] is None


@pytest.mark.parametrize("control", [
    "foreign_key", "trigger_key", "duplicate", "payload", "request_hash",
    "wire_payload", "wire_model", "raw_response", "response_model",
    "response_truncated", "stored_grade", "wrong_checks", "wrong_status",
    "missing_response", "symlink_response", "failed_with_grade"
])
def test_inconsistent_or_unsafe_saved_grade_is_rejected(
        evidence, tmp_path, control):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    grade_dir = tmp_path / "grade"
    grade_fixture(evidence,
                  grade_dir,
                  case_id="E03" if control == "wrong_checks" else "E01",
                  bad_checks=control == "wrong_checks",
                  bad_status=control == "wrong_status")
    request = json.loads((grade_dir / "request.json").read_text())
    response = json.loads((grade_dir / "response.json").read_text())
    result = json.loads((grade_dir / "result.json").read_text())
    if control == "foreign_key": result["attempt_key"] = "f" * 32
    if control == "trigger_key":
        result["attempt_key"] = json.loads(
            (collection /
             "blinding.json").read_text())["attempt_keys"]["opaque0"]
    if control == "payload":
        request["payload"]["evidence"][0]["lines"] = ["altered"]
    if control == "request_hash": result["request_sha256"] = "0" * 64
    if control == "wire_payload":
        request["request"]["messages"][1]["content"] = "{}"
    if control == "wire_model": request["request"]["model"] = "different-model"
    if control in ("wire_payload", "wire_model"):
        result["request_sha256"] = hashlib.sha256(
            json.dumps(request["request"], sort_keys=True,
                       ensure_ascii=False).encode()).hexdigest()
    if control == "raw_response":
        raw = json.loads(response["choices"][0]["message"]["content"])
        raw["dimensions"][0]["score"] = 0
        raw["total"] = 8
        response["choices"][0]["message"]["content"] = json.dumps(raw)
    if control == "response_model": response["model"] = "different-model"
    if control == "response_truncated":
        response["choices"][0]["finish_reason"] = "length"
    if control == "stored_grade": result["grade"]["raw_score"] = 0
    if control == "failed_with_grade": result["status"] = "transport_error"
    write_json(grade_dir / "request.json", request)
    write_json(grade_dir / "response.json", response)
    write_json(grade_dir / "result.json", result)
    if control == "missing_response": (grade_dir / "response.json").unlink()
    if control == "symlink_response":
        destination = tmp_path / "outside-response.json"
        (grade_dir / "response.json").rename(destination)
        (grade_dir / "response.json").symlink_to(destination)
    with pytest.raises((ValueError, OSError)):
        reporting.report(root,
                         mapping_path=collection / "blinding.json",
                         grade_dirs=[grade_dir, grade_dir]
                         if control == "duplicate" else [grade_dir])


def test_changed_independent_check_cannot_reuse_old_passing_grade(
        evidence, tmp_path):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    grade_dir = tmp_path / "grade"
    row = grade_fixture(evidence, grade_dir, case_id="E03")
    attempt = next(
        a for a in json.loads((root / "manifest.json").read_text())["attempts"]
        if a["attempt_id"] == row["attempt_id"])
    check(root,
          attempt,
          "targeted", {
              "passed": 31,
              "failed": 1
          },
          status="failed")
    with pytest.raises(ValueError):
        reporting.report(root,
                         mapping_path=collection / "blinding.json",
                         grade_dirs=[grade_dir])


@pytest.mark.parametrize("missing", ["freeze", "mapping"])
def test_reporting_requires_verified_freeze_and_saved_blinding(
        evidence, missing):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    if missing == "freeze": (root / "manifest.sha256.json").unlink()
    with pytest.raises((ValueError, OSError)):
        reporting.report(
            root,
            mapping_path=None if missing == "mapping" else collection /
            "blinding.json")


def test_cli_publishes_private_hashed_report_and_refuses_overwrite(
        evidence, tmp_path):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    grade_dir = tmp_path / "grade"
    grade_fixture(evidence, grade_dir)
    output = tmp_path / "report"
    args = [
        "--run-dir",
        str(root), "--mapping",
        str(collection / "blinding.json"), "--grade-dir",
        str(grade_dir), "--output-dir",
        str(output)
    ]
    assert reporting.main(args) == 0
    assert output.stat().st_mode & 0o777 == 0o700
    hashes = json.loads((output / "hashes.json").read_text())
    for name in ("rows.json", "summary.json", "provenance.json"):
        assert (output / name).stat().st_mode & 0o777 == 0o600
        assert hashes["outputs"][name] == hashlib.sha256(
            (output / name).read_bytes()).hexdigest()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert reporting.main(args) == 1
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_invalid_cli_input_does_not_publish_output(evidence, tmp_path):
    assert reporting is not None, "verified reporting is not implemented"
    root, collection = evidence
    (root / "manifest.sha256.json").unlink()
    output = tmp_path / "report"
    assert reporting.main([
        "--run-dir",
        str(root), "--mapping",
        str(collection / "blinding.json"), "--output-dir",
        str(output)
    ]) == 1
    assert not output.exists()


def test_all_worker_metrics_use_same_successful_complete_cohort(
        evidence, tmp_path):
    root, collection = evidence
    directories = []
    for case in ("E01", "E03"):
        for arm in ("baseline", "candidate"):
            directory = tmp_path / f"grade-{case}-{arm}"
            grade_fixture(evidence, directory, case_id=case, arm=arm)
            directories.append(directory)
    full = reporting.report(root,
                            mapping_path=collection / "blinding.json",
                            grade_dirs=directories)
    assert full["summary"]["judge"]["total_tokens"] == 480
    assert sum(r["usage"]["total_tokens"] for r in full["rows"]) == 96
    assert len(full["summary"]["efficiency"]["cohort"]["pair_ids"]) == 4
    row = next(r for r in full["rows"]
               if r["case_id"] == "E03" and r["arm"] == "candidate")
    session = Path(row["session_paths"][0])
    events = [json.loads(line) for line in session.read_text().splitlines()]
    del events[1]["payload"]["info"]["last_token_usage"]["input_tokens"]
    session.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    reduced = reporting.report(root,
                               mapping_path=collection / "blinding.json",
                               grade_dirs=directories)
    efficiency = reduced["summary"]["efficiency"]
    assert len(efficiency["cohort"]["pair_ids"]) == 3
    for metric, expected in (("task_time_s", 8), ("total_tokens", 12),
                             ("peak_request_input_tokens_proxy", 8)):
        assert efficiency[metric]["measured_pairs"] == 3
        assert efficiency[metric]["baseline_median"] == expected
        assert efficiency[metric]["candidate_median"] == expected
    assert efficiency["statistical_pass"] is None
    assert efficiency["confidence"]["statistical_pass"] is None


def test_changed_observed_input_during_reduction_cannot_be_published(
        evidence, tmp_path, monkeypatch):
    root, collection = evidence
    # Model calls are not involved; induce a concurrent file change at reduction.
    summarize = reporting.judging.summarize

    def changing_summary(*args, **kwargs):
        (root / "manifest.sha256.json").write_text("{}\n")
        return summarize(*args, **kwargs)

    monkeypatch.setattr(reporting.judging, "summarize", changing_summary)
    output = tmp_path / "report"
    assert reporting.main([
        "--run-dir",
        str(root), "--mapping",
        str(collection / "blinding.json"), "--output-dir",
        str(output)
    ]) == 1
    assert not output.exists()


@pytest.mark.parametrize("change", [
    "unknown_purpose", "partial_main", "duplicate_case", "boolean_repetition"
])
def test_purpose_cannot_relabel_an_incomplete_or_corrupt_schedule(
        evidence, tmp_path, change):
    root, _ = evidence
    manifest = json.loads((root / "manifest.json").read_text())
    if change == "unknown_purpose": manifest["purpose"] = "unrecognized"
    if change == "partial_main": manifest["purpose"] = "disclosed-study"
    if change == "duplicate_case": manifest["attempts"][0]["case_id"] = "N07"
    if change == "boolean_repetition":
        manifest["attempts"][0]["repetition"] = True
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "manifest.sha256.json", {
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        })
    collection = tmp_path / "new-collection"
    assert posthoc.main(
        ["--run-dir", str(root), "--output-dir",
         str(collection)]) == 0
    with pytest.raises(ValueError):
        reporting.report(root, mapping_path=collection / "blinding.json")


def test_complete_main_schedule_retains_all_one_hundred_missing_outcomes(
        tmp_path):
    root, _ = run_fixture(tmp_path, ())
    attempts = []
    for index, attempt in enumerate(suite.build_schedule(suite.load_cases())):
        identity = f"attempt{index:03d}"
        attempts.append({
            "attempt_id": identity,
            "case_id": attempt.case_id,
            "arm": attempt.arm,
            "repetition": attempt.repetition,
            "job_name": identity,
            "dataset_dir": f"datasets/{identity}"
        })
    write_json(root / "manifest.json", {
        "schema_version": 1,
        "purpose": "disclosed-study",
        "attempts": attempts
    })
    write_json(
        root / "manifest.sha256.json", {
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        })
    collection = tmp_path / "collection"
    assert posthoc.main(
        ["--run-dir", str(root), "--output-dir",
         str(collection)]) == 0
    result = reporting.report(root, mapping_path=collection / "blinding.json")
    assert len(result["rows"]) == 100
    assert result["summary"]["pilot"] is False
    assert result["summary"]["outcomes"] == {"unknown": 100}
    assert result["summary"]["judge"]["missing_attempts"] == 56
    assert result["summary"]["efficiency"]["cohort"]["pair_ids"] == []
    assert result["summary"]["efficiency"]["statistical_pass"] is None
    assert result["summary"]["efficiency"]["confidence"][
        "statistical_pass"] is None
    assert result["provenance"]["input_hashes"][str(
        suite.BRIEF)] == hashlib.sha256(suite.BRIEF.read_bytes()).hexdigest()


@pytest.mark.parametrize("bound", [None, -1, True, 59905])
def test_graded_record_cannot_claim_impossible_token_admission(
        evidence, tmp_path, bound):
    root, collection = evidence
    directory = tmp_path / "grade"
    grade_fixture(evidence, directory, token_counter=lambda request: 10000)
    for name in ("request.json", "result.json"):
        value = json.loads((directory / name).read_text())
        value["input_token_bound"] = bound
        write_json(directory / name, value)
    with pytest.raises(ValueError):
        reporting.report(root,
                         mapping_path=collection / "blinding.json",
                         grade_dirs=[directory])


@pytest.mark.parametrize("byte_mode", [False, True])
def test_large_payload_can_only_admit_grade_with_valid_token_bound(
        evidence, tmp_path, byte_mode):
    root, collection = evidence
    attempt = next(
        a for a in json.loads((root / "manifest.json").read_text())["attempts"]
        if a["case_id"] == "E01" and a["arm"] == "baseline")
    trajectory = root / "jobs" / attempt["job_name"] / (
        attempt["attempt_id"] + "__trial") / "agent/trajectory.json"
    value = json.loads(trajectory.read_text())
    value["steps"][0]["message"] += "\n" + "x" * 60000
    write_json(trajectory, value)
    directory = tmp_path / "grade"
    row = grade_fixture(evidence,
                        directory,
                        token_counter=lambda request: 59904)
    if byte_mode:
        for name in ("request.json", "result.json"):
            value = json.loads((directory / name).read_text())
            value.update(budget_mode="utf8_bytes", input_token_bound=None)
            write_json(directory / name, value)
        with pytest.raises(ValueError):
            reporting.report(root,
                             mapping_path=collection / "blinding.json",
                             grade_dirs=[directory])
    else:
        result = reporting.report(root,
                                  mapping_path=collection / "blinding.json",
                                  grade_dirs=[directory])
        matched = next(r for r in result["rows"]
                       if r["attempt_key"] == row["attempt_key"])
        assert matched["payload_status"] == "evidence_oversized"
        assert matched["grade"]["successful"] is True


@pytest.mark.parametrize("bound,status", [(None, "invalid_token_bound"),
                                          (59905, "evidence_oversized")])
def test_genuine_budget_failure_is_retained_without_grade(
        evidence, tmp_path, bound, status):
    root, collection = evidence
    directory = tmp_path / "grade"
    row = grade_fixture(evidence,
                        directory,
                        token_counter=lambda request: bound,
                        expected_status=status)
    result = reporting.report(root,
                              mapping_path=collection / "blinding.json",
                              grade_dirs=[directory])
    matched = next(r for r in result["rows"]
                   if r["attempt_key"] == row["attempt_key"])
    assert matched["grade"] is None
    assert matched["judge"]["status"] == status


@pytest.mark.parametrize("control", ["choice", "message", "cost_container"])
def test_malformed_saved_container_fails_privately_without_publication(
        evidence, tmp_path, capsys, control):
    root, collection = evidence
    directory = tmp_path / "grade"
    grade_fixture(evidence, directory)
    filename = "result.json" if control == "cost_container" else "response.json"
    value = json.loads((directory / filename).read_text())
    if control == "cost_container": value["judge_usage"] = None
    if control == "choice": value["choices"] = [None]
    if control == "message": value["choices"][0]["message"] = None
    write_json(directory / filename, value)
    output = tmp_path / "report"
    capsys.readouterr()
    assert reporting.main([
        "--run-dir",
        str(root), "--mapping",
        str(collection / "blinding.json"), "--grade-dir",
        str(directory), "--output-dir",
        str(output)
    ]) == 1
    assert not output.exists()
    assert "report_failed" in capsys.readouterr().out
