# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

STRATEGY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY))
import capture

SPEC = importlib.util.spec_from_file_location("strategy_posthoc",
                                              STRATEGY / "posthoc.py")
posthoc = None
if (STRATEGY / "posthoc.py").is_file():
    posthoc = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(posthoc)

E05_BINDING = b'{"schema_version":1,"kind":"artifact"}\n'
E05_BINDING_SHA256 = hashlib.sha256(E05_BINDING).hexdigest()
E08_BINDING = b'{"schema_version":1,"build":{"kind":"callable"}}\n'
E08_BINDING_SHA256 = hashlib.sha256(E08_BINDING).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def run_fixture(tmp_path, cases=("T02", "N07", "E01", "E03")):
    root = tmp_path / "private-run"
    attempts = []
    for case in cases:
        for arm in ("baseline", "candidate"):
            identity = f"opaque{len(attempts)}"
            attempt = {
                "attempt_id": identity,
                "case_id": case,
                "arm": arm,
                "repetition": 1,
                "job_name": identity,
                "dataset_dir": f"datasets/{identity}"
            }
            attempts.append(attempt)
            source = root / attempt[
                "dataset_dir"] / identity / "environment/repo"
            source.mkdir(parents=True)
            (source / "example.py").write_text("original = True\n")
            consumers = (posthoc.numerical.E02_CONSUMERS if case == "E02" else
                         posthoc.numerical.E05_CONSUMERS if case == "E05" else
                         posthoc.numerical.E08_CONSUMERS if case == "E08" else
                         ())
            for name in consumers:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# frozen public consumer\n")
    write_json(root / "manifest.json", {
        "schema_version": 1,
        "attempts": attempts
    })
    return root, attempts


def terminal(root, attempt, *, exception=None, final="Completed."):
    trial = root / "jobs" / attempt["job_name"] / (attempt["attempt_id"] +
                                                   "__trial")
    write_json(
        trial / "result.json", {
            "finished_at": "2026-09-15T10:00:10Z",
            "exception_info": exception,
            "agent_execution": {
                "started_at": "2026-09-15T10:00:01Z",
                "finished_at": "2026-09-15T10:00:09Z"
            }
        })
    write_json(
        trial / "agent/trajectory.json",
        {"steps": [{
            "source": "agent",
            "message": final,
            "tool_calls": []
        }]})
    session = trial / "agent/sessions/2026/09/15/session.jsonl"
    events = [{
        "type": "session_meta",
        "payload": {
            "id": attempt["attempt_id"]
        }
    }, {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12
                },
                "last_token_usage": {
                    "input_tokens": 8,
                    "output_tokens": 2,
                    "total_tokens": 10
                }
            }
        }
    }]
    session.parent.mkdir(parents=True)
    session.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return trial


def captured(root, attempt, trial):
    source = root / attempt["dataset_dir"] / attempt[
        "attempt_id"] / "environment/repo"
    worktree = trial / "agent/worktree"
    worktree.mkdir(parents=True)
    (worktree / "example.py").write_text("original = False\n")
    consumers = (
        posthoc.numerical.E02_CONSUMERS if attempt["case_id"] == "E02" else
        posthoc.numerical.E05_CONSUMERS if attempt["case_id"] == "E05" else
        posthoc.numerical.E08_CONSUMERS if attempt["case_id"] == "E08" else ())
    for name in consumers:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    directory = trial / "verifier/artifacts/capture"
    manifest = capture.capture(source, worktree, directory)
    digest = hashlib.sha256(
        (directory / "capture.json").read_bytes()).hexdigest()
    write_json(
        root / "posthoc" / attempt["attempt_id"] / "artifact_checks.json", {
            "attempt_id": attempt["attempt_id"],
            "case_id": attempt["case_id"],
            "capture_status": "complete",
            "capture_selected_path": str(directory),
            "reconstruction_status": "complete",
            "reconstruction": {
                "status": "complete",
                "capture_json_sha256": digest,
                "worktree_inventory_sha256":
                manifest["worktree_inventory_sha256"]
            },
            "hashes": {
                "capture_manifest":
                digest,
                "worktree_inventory":
                manifest["worktree_inventory_sha256"],
                "source_files": {
                    name: item["sha256"]
                    for name, item in manifest["source_inventory"].items()
                },
                "manifest":
                hashlib.sha256(
                    (root / "manifest.json").read_bytes()).hexdigest(),
                "trial_result":
                hashlib.sha256(
                    (trial / "result.json").read_bytes()).hexdigest()
            },
            "failures": []
        })
    return directory


def check(root,
          attempt,
          kind,
          counts,
          *,
          status="passed",
          binding_sha256=None):
    output = root / "posthoc" / attempt["attempt_id"] / kind
    raw = ("independent scientific result\n" + str(counts) + "\n").encode()
    bound_inputs = None
    bound_case = attempt["case_id"] in {"E02", "E05", "E08"
                                        } and kind == "targeted"
    if bound_case and binding_sha256 is not None:
        record = json.loads(
            (output.parent / "artifact_checks.json").read_text())
        prefix = attempt["case_id"].lower()
        configured_consumers = (
            posthoc.numerical.E02_CONSUMERS if attempt["case_id"] == "E02" else
            posthoc.numerical.E05_CONSUMERS if attempt["case_id"] == "E05" else
            posthoc.numerical.E08_CONSUMERS)
        consumers = {
            name: record["hashes"]["source_files"][name]
            for name in configured_consumers
        }
        frozen = {
            "binding_sha256": binding_sha256,
            "captured_inventory_sha256": "c" * 64,
            "checks_inventory_sha256": "d" * 64,
            "checks_files": {},
            "consumer_files": consumers
        }
        freeze_path = output / f"{prefix}-inputs.json"
        write_json(freeze_path, frozen)
        bound_inputs = {
            **frozen, "manifest_sha256":
            hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "unchanged":
            True
        }
        if attempt["case_id"] == "E05":
            assert binding_sha256 == E05_BINDING_SHA256
            staged_binding = output / "e05-input/binding.json"
            staged_binding.parent.mkdir(parents=True)
            staged_binding.write_bytes(E05_BINDING)
        elif attempt["case_id"] == "E08":
            assert binding_sha256 == E08_BINDING_SHA256
            staged_binding = output / "e08-input/binding.json"
            staged_binding.parent.mkdir(parents=True)
            staged_binding.write_bytes(E08_BINDING)
    result = {
        "status": status,
        "kind": kind,
        "case_id": attempt["case_id"],
        "binding_sha256": binding_sha256,
        "pytest_counts": counts,
        "returncode": 0 if status == "passed" else 1,
        "output_bytes": len(raw),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "worker_metric": False
    }
    if bound_inputs is not None:
        result[attempt["case_id"].lower() + "_inputs"] = bound_inputs
    write_json(output / "result.json", result)
    (output / "pytest.log").write_bytes(raw)
    record_path = output.parent / "artifact_checks.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        record[kind] = "pass" if status == "passed" else "fail"
        record["hashes"][kind + "_result"] = hashlib.sha256(
            (output / "result.json").read_bytes()).hexdigest()
        record["hashes"][kind + "_log"] = hashlib.sha256(raw).hexdigest()
        if binding_sha256 is not None:
            prefix = attempt["case_id"].lower()
            record["hashes"][prefix + "_binding"] = binding_sha256
            record["hashes"][
                prefix + "_inputs_manifest"] = bound_inputs["manifest_sha256"]
            record["hashes"][prefix + "_captured_inventory"] = bound_inputs[
                "captured_inventory_sha256"]
            record["hashes"]["checks_inventory"] = bound_inputs[
                "checks_inventory_sha256"]
            record["hashes"]["checks_files"] = bound_inputs["checks_files"]
        write_json(record_path, record)


def test_all_attempts_survive_missing_results_and_job_summary_is_not_a_trial(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path)
    write_json(root / "jobs" / attempts[0]["job_name"] / "result.json", {
        "finished_at": "2026-09-15T10:00:10Z",
        "exception_info": None
    })
    result = posthoc.collect(root)
    assert len(result["rows"]) == 8
    assert all(row["execution_status"] == "unknown" for row in result["rows"])
    assert all(row["usage"]["total_tokens"] is None for row in result["rows"])
    assert all(row["usage"]["activation"] is None for row in result["rows"])
    assert all(row["independent_checks"]["capture"] == "unknown"
               for row in result["rows"])
    assert all(row["grade"] is None for row in result["rows"])
    assert len(result["payloads"]) == 4
    assert not (root / "posthoc").exists()


@pytest.mark.parametrize("record", [{}, {
    "wrong_field": "value"
}, {
    "sha256": None
}, {
    "sha256": True
}, {
    "sha256": "0" * 64
}, [], None])
def test_present_manifest_freeze_must_contain_the_matching_digest(
        tmp_path, record):
    root, _ = run_fixture(tmp_path, ("E01", ))
    write_json(root / "manifest.sha256.json", record)
    with pytest.raises(ValueError, match="manifest hash"):
        posthoc.collect(root)


@pytest.mark.parametrize("frozen", [False, True])
def test_collector_exports_observed_manifest_and_explicit_freeze_verification(
        tmp_path, frozen):
    root, _ = run_fixture(tmp_path, ("E01", ))
    observed = hashlib.sha256(
        (root / "manifest.json").read_bytes()).hexdigest()
    freeze_path = root / "manifest.sha256.json"
    if frozen:
        write_json(freeze_path, {"sha256": observed})
    result = posthoc.collect(root)
    assert result["manifest_freeze"] == {
        "status":
        "verified" if frozen else "missing",
        "manifest_sha256":
        observed,
        "record_sha256":
        hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        if frozen else None
    }
    assert len(result["rows"]) == 2
    assert all(row["grade"] is None for row in result["rows"])


@pytest.mark.parametrize("frozen", [False, True])
def test_collection_cli_preserves_manifest_freeze_status_in_hashed_provenance(
        tmp_path, frozen):
    root, _ = run_fixture(tmp_path, ("E01", ))
    observed = hashlib.sha256(
        (root / "manifest.json").read_bytes()).hexdigest()
    if frozen:
        write_json(root / "manifest.sha256.json", {"sha256": observed})
    output = tmp_path / "collection"
    assert posthoc.main(["--run-dir",
                         str(root), "--output-dir",
                         str(output)]) == 0
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["manifest_freeze"]["status"] == ("verified" if frozen
                                                       else "missing")
    assert provenance["manifest_freeze"]["manifest_sha256"] == observed
    hashes = json.loads((output / "hashes.json").read_text())
    assert hashes["outputs"]["provenance.json"] == hashlib.sha256(
        provenance_path.read_bytes()).hexdigest()
    assert hashes["inputs"][str(root / "manifest.json")] == observed


def test_invalid_empty_freeze_cannot_create_a_collection_output(tmp_path):
    root, _ = run_fixture(tmp_path, ("E01", ))
    write_json(root / "manifest.sha256.json", {})
    output = tmp_path / "collection"
    assert posthoc.main(["--run-dir",
                         str(root), "--output-dir",
                         str(output)]) == 1
    assert not output.exists()
    assert json.loads((root / "manifest.sha256.json").read_text()) == {}


def test_complete_attempt_uses_native_usage_full_patch_and_verified_check_coverage(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E03", ))
    trial = terminal(root,
                     attempts[0],
                     final=f"Changed {root}/source/example.py\narm: baseline")
    captured(root, attempts[0], trial)
    check(root, attempts[0], "targeted", {"passed": 32})
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    result = posthoc.collect(root)
    row = result["rows"][0]
    assert row["execution_status"] == "complete"
    assert row["usage"]["total_tokens"] == 12
    assert row["usage"]["task_time_s"] == 8
    assert row["usage"]["peak_request_input_tokens_proxy"] == 8
    assert row["independent_checks"] == {
        "capture": "complete",
        "targeted": "pass",
        "regression": "pass"
    }
    payload = next(p for p in result["payloads"]
                   if p["attempt_key"] == row["attempt_key"])
    encoded = json.dumps(payload)
    assert str(root) not in encoded
    assert '"arm"' not in encoded
    assert "arm: baseline" not in encoded
    assert "-original = True" in encoded and "+original = False" in encoded
    assert "independent scientific result" in encoded
    assert result["checks_mapping"][
        row["attempt_key"]] == row["independent_checks"]


def test_e02_uses_60_test_registration_and_explicit_binding_hash(tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E02", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    binding_sha256 = "b" * 64
    check(root,
          attempts[0],
          "targeted", {"passed": 60},
          binding_sha256=binding_sha256)
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"] == {
        "capture": "complete",
        "targeted": "pass",
        "regression": "pass"
    }
    assert row["check_records"]["targeted"]["binding"] == "verified"


def test_e02_targeted_without_frozen_binding_is_unknown_but_regression_survives(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E02", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    check(root, attempts[0], "targeted", {"passed": 60})
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"
    assert row["independent_checks"]["regression"] == "pass"
    assert row["check_records"]["targeted"]["binding"] == "unknown"


def test_e05_accepts_only_the_complete_bound_provenance_join(tmp_path):
    root, attempts = run_fixture(tmp_path, ("E05", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    binding_sha256 = E05_BINDING_SHA256
    count = posthoc.numerical.CHECK_SUITES["E05"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=binding_sha256)
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"] == {
        "capture": "complete",
        "targeted": "pass",
        "regression": "pass"
    }
    assert row["check_records"]["targeted"]["binding"] == "verified"


def test_e05_missing_binding_is_unknown_but_original_regression_survives(
        tmp_path):
    root, attempts = run_fixture(tmp_path, ("E05", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E05"]["targeted"][0]
    check(root, attempts[0], "targeted", {"passed": count})
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"
    assert row["independent_checks"]["regression"] == "pass"


@pytest.mark.parametrize("fault", [
    "missing_freeze", "malformed_freeze", "wrong_binding", "unchanged_false",
    "missing_consumer", "wrong_capture_inventory", "wrong_checks_inventory"
])
def test_e05_tampered_or_malformed_provenance_remains_unknown(tmp_path, fault):
    root, attempts = run_fixture(tmp_path, ("E05", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    digest = E05_BINDING_SHA256
    count = posthoc.numerical.CHECK_SUITES["E05"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=digest)
    output = root / "posthoc" / attempts[0]["attempt_id"] / "targeted"
    freeze_path = output / "e05-inputs.json"
    result_path = output / "result.json"
    record_path = output.parent / "artifact_checks.json"
    frozen = json.loads(freeze_path.read_text())
    result = json.loads(result_path.read_text())
    record = json.loads(record_path.read_text())
    if fault == "missing_freeze":
        freeze_path.unlink()
    elif fault == "malformed_freeze":
        freeze_path.write_text("[]\n")
    elif fault == "wrong_binding":
        result["binding_sha256"] = "6" * 64
    elif fault == "unchanged_false":
        result["e05_inputs"]["unchanged"] = False
    elif fault == "missing_consumer":
        frozen["consumer_files"].pop(posthoc.numerical.E05_CONSUMERS[0])
    elif fault == "wrong_capture_inventory":
        frozen["captured_inventory_sha256"] = "a" * 64
    else:
        frozen["checks_inventory_sha256"] = "a" * 64
    if fault in {
            "missing_consumer", "wrong_capture_inventory",
            "wrong_checks_inventory"
    }:
        write_json(freeze_path, frozen)
        result["e05_inputs"] = {
            **frozen, "manifest_sha256":
            hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "unchanged":
            True
        }
        record["hashes"]["e05_inputs_manifest"] = result["e05_inputs"][
            "manifest_sha256"]
    if fault not in {"missing_freeze", "malformed_freeze"}:
        write_json(result_path, result)
        record["hashes"]["targeted_result"] = hashlib.sha256(
            result_path.read_bytes()).hexdigest()
    write_json(record_path, record)
    assert posthoc.collect(
        root)["rows"][0]["independent_checks"]["targeted"] == "unknown"


def test_e05_bound_numerical_failure_remains_fail(tmp_path):
    root, attempts = run_fixture(tmp_path, ("E05", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E05"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {
              "passed": count - 1,
              "failed": 1
          },
          status="failed",
          binding_sha256=E05_BINDING_SHA256)
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "fail"


def test_e05_complete_targeted_evidence_cannot_waive_original_regression(
        tmp_path):
    root, attempts = run_fixture(tmp_path, ("E05", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E05"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=E05_BINDING_SHA256)
    checks = posthoc.collect(root)["rows"][0]["independent_checks"]
    assert checks["targeted"] == "pass"
    assert checks["regression"] == "unknown"


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_e05_posthoc_rejects_missing_or_changed_staged_binding(
        tmp_path, fault):
    root, attempts = run_fixture(tmp_path, ("E05", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E05"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=E05_BINDING_SHA256)
    staged = root / "posthoc" / attempts[0][
        "attempt_id"] / "targeted/e05-input/binding.json"
    if fault == "missing":
        staged.unlink()
    else:
        staged.write_text("{}\n")
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"
    assert row["check_records"]["targeted"]["binding"] == "unknown"


def test_e08_accepts_only_complete_source_bound_provenance_join(tmp_path):
    root, attempts = run_fixture(tmp_path, ("E08", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E08"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=E08_BINDING_SHA256)
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    result = posthoc.collect(root)
    row = result["rows"][0]
    assert row["case_id"] == "E08"
    assert row["independent_checks"] == {
        "capture": "complete",
        "targeted": "pass",
        "regression": "pass"
    }
    assert row["check_records"]["targeted"]["binding"] == "verified"
    assert any("E08" in limitation for limitation in result["limitations"])


def test_e08_unbound_result_is_unknown_but_original_regression_survives(
        tmp_path):
    root, attempts = run_fixture(tmp_path, ("E08", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E08"]["targeted"][0]
    check(root, attempts[0], "targeted", {"passed": count})
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"
    assert row["independent_checks"]["regression"] == "pass"


@pytest.mark.parametrize("counts", [{
    "passed": 11
}, {
    "passed": 11,
    "skipped": 1
}])
def test_e08_incomplete_or_skipped_coverage_cannot_pass(tmp_path, counts):
    root, attempts = run_fixture(tmp_path, ("E08", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    check(root,
          attempts[0],
          "targeted",
          counts,
          binding_sha256=E08_BINDING_SHA256)
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"


@pytest.mark.parametrize(
    "fault",
    ["wrong_case", "wrong_consumer", "wrong_checker", "wrong_capture"])
def test_e08_wrong_result_or_inventory_join_is_unknown(tmp_path, fault):
    root, attempts = run_fixture(tmp_path, ("E08", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E08"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=E08_BINDING_SHA256)
    output = root / "posthoc" / attempts[0]["attempt_id"] / "targeted"
    result_path = output / "result.json"
    freeze_path = output / "e08-inputs.json"
    record_path = output.parent / "artifact_checks.json"
    result = json.loads(result_path.read_text())
    frozen = json.loads(freeze_path.read_text())
    record = json.loads(record_path.read_text())
    if fault == "wrong_case":
        result["case_id"] = "E05"
    elif fault == "wrong_consumer":
        frozen["consumer_files"].pop(posthoc.numerical.E08_CONSUMERS[0])
    elif fault == "wrong_checker":
        frozen["checks_inventory_sha256"] = "a" * 64
    else:
        frozen["captured_inventory_sha256"] = "a" * 64
    if fault != "wrong_case":
        write_json(freeze_path, frozen)
        result["e08_inputs"] = {
            **frozen, "manifest_sha256":
            hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "unchanged":
            True
        }
        record["hashes"]["e08_inputs_manifest"] = result["e08_inputs"][
            "manifest_sha256"]
    write_json(result_path, result)
    record["hashes"]["targeted_result"] = hashlib.sha256(
        result_path.read_bytes()).hexdigest()
    write_json(record_path, record)
    assert posthoc.collect(
        root)["rows"][0]["independent_checks"]["targeted"] == "unknown"


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_e08_posthoc_rejects_missing_or_changed_staged_binding(
        tmp_path, fault):
    root, attempts = run_fixture(tmp_path, ("E08", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    count = posthoc.numerical.CHECK_SUITES["E08"]["targeted"][0]
    check(root,
          attempts[0],
          "targeted", {"passed": count},
          binding_sha256=E08_BINDING_SHA256)
    staged = root / "posthoc" / attempts[0][
        "attempt_id"] / "targeted/e08-input/binding.json"
    if fault == "missing":
        staged.unlink()
    else:
        staged.write_text("{}\n")
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"
    assert row["check_records"]["targeted"]["binding"] == "unknown"


@pytest.mark.parametrize("fault", [
    "missing_freeze", "changed_freeze", "unchanged_false",
    "missing_consumer_origin"
])
def test_e02_targeted_requires_the_bound_unchanged_input_freeze(
        tmp_path, fault):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E02", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    check(root,
          attempts[0],
          "targeted", {"passed": 60},
          binding_sha256="b" * 64)
    output = root / "posthoc" / attempts[0]["attempt_id"] / "targeted"
    if fault == "missing_freeze":
        (output / "e02-inputs.json").unlink()
    elif fault == "changed_freeze":
        (output / "e02-inputs.json").write_text("{}\n")
    elif fault == "unchanged_false":
        result_path = output / "result.json"
        result = json.loads(result_path.read_text())
        result["e02_inputs"]["unchanged"] = False
        write_json(result_path, result)
        record_path = output.parent / "artifact_checks.json"
        record = json.loads(record_path.read_text())
        record["hashes"]["targeted_result"] = hashlib.sha256(
            result_path.read_bytes()).hexdigest()
        write_json(record_path, record)
    else:
        freeze_path = output / "e02-inputs.json"
        frozen = json.loads(freeze_path.read_text())
        frozen["consumer_files"].pop(posthoc.numerical.E02_CONSUMERS[0])
        write_json(freeze_path, frozen)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text())
        result["e02_inputs"] = {
            **frozen, "manifest_sha256":
            hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "unchanged":
            True
        }
        write_json(result_path, result)
        record_path = output.parent / "artifact_checks.json"
        record = json.loads(record_path.read_text())
        record["hashes"]["e02_inputs_manifest"] = result["e02_inputs"][
            "manifest_sha256"]
        record["hashes"]["targeted_result"] = hashlib.sha256(
            result_path.read_bytes()).hexdigest()
        write_json(record_path, record)
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"


def test_unregistered_mutation_case_cannot_adopt_a_plausible_check_result(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E04", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    check(root, attempts[0], "targeted", {"passed": 60})
    check(root, attempts[0], "regression", {"passed": 307, "skipped": 4})
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"] == {
        "capture": "complete",
        "targeted": "unknown",
        "regression": "unknown"
    }


def conditional_capture(tmp_path, *, case_id="E07", change=None):
    """Real frozen capture/reconstruction; never execute a worker or checker."""
    import artifact_checks

    root, attempts = run_fixture(tmp_path, (case_id, ))
    attempt = attempts[0]
    trial = terminal(
        root,
        attempt,
        final="Use an application/example; no core change claimed.")
    source = root / attempt["dataset_dir"] / attempt[
        "attempt_id"] / "environment/repo"
    worktree = trial / "agent/worktree"
    shutil.copytree(source, worktree)
    if change == "modify":
        (worktree / "example.py").write_text("def vqe(): return None\n")
    elif change == "add":
        (worktree / "new.py").write_text("def vqe(): return None\n")
    elif change == "delete":
        (worktree / "example.py").unlink()
    elif change == "mode":
        (worktree / "example.py").chmod(0o755)
    source_hashes = {
        p.relative_to(source).as_posix():
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*") if p.is_file()
    }
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(
        image_id=posthoc.numerical.IMAGE_ID,
        input_hashes={
            "source_inventory":
            hashlib.sha256(json.dumps(source_hashes,
                                      sort_keys=True).encode()).hexdigest(),
            "capture":
            hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest()
        })
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "manifest.sha256.json", {
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        })
    directory = trial / "verifier/artifacts/capture"
    capture.capture(source, worktree, directory)
    record = artifact_checks.check_attempt(root, attempt["attempt_id"])
    assert record["capture_status"] == record[
        "reconstruction_status"] == "complete"
    return root, attempt, directory


def test_unchanged_e07_capture_flows_through_collection_grading_and_summary(
        tmp_path):
    root, attempt, _ = conditional_capture(tmp_path)
    result = posthoc.collect(root)
    row = result["rows"][0]
    checks = row["independent_checks"]
    assert checks == {
        "capture": "complete",
        "unchanged_capture": "verified",
        "targeted": "not_applicable",
        "regression": "not_applicable"
    }
    assert result["checks_mapping"][row["attempt_key"]] == checks
    payload = next(p for p in result["payloads"]
                   if p["attempt_key"] == row["attempt_key"])
    cite = [{"id": "final", "start": 1, "end": 1}]
    grade = {
        "attempt_key":
        row["attempt_key"],
        "total":
        10,
        "dimensions": [{
            "dimension": d["dimension"],
            "score": 2,
            "reason": "Synthetic gate input, not an agent grade.",
            "evidence": cite
        } for d in payload["score_rubric"]],
        "critical_checks": [{
            "id": c["id"],
            "verdict": "pass",
            "reason": "Synthetic input.",
            "evidence": cite
        } for c in payload["critical_verdicts"]]
    }
    row["grade"] = posthoc.judging.validate_grade(grade,
                                                  payload,
                                                  checks=checks)
    assert row["grade"]["certification"] == "supported"
    assert row["grade"]["successful"] is True
    summary = posthoc.judging.summarize([row])
    assert summary["execution"][attempt["arm"]]["successful_attempts"] == 1
    assert summary["successful_matched_pairs"] == 0
    assert summary["efficiency"]["statistical_pass"] is None


@pytest.mark.parametrize("change", ["modify", "add", "delete", "mode"])
def test_changed_e07_needs_checks_even_when_worker_claims_no_change(
        tmp_path, change):
    root, _, _ = conditional_capture(tmp_path, change=change)
    checks = posthoc.collect(root)["rows"][0]["independent_checks"]
    assert checks == {
        "capture": "complete",
        "unchanged_capture": "changed",
        "targeted": "unknown",
        "regression": "unknown"
    }


@pytest.mark.parametrize("change,expected_result_binding",
                         [(None, "not_applicable"), ("modify", "unknown")])
def test_e07_judge_evidence_distinguishes_capture_from_numerical_result_binding(
        tmp_path, change, expected_result_binding):
    root, _, _ = conditional_capture(tmp_path, change=change)
    result = posthoc.collect(root)
    row = result["rows"][0]
    payload = next(item for item in result["payloads"]
                   if item["attempt_key"] == row["attempt_key"])
    lines = next(item["lines"] for item in payload["evidence"]
                 if item["id"] == "tests")
    for kind in ("targeted", "regression"):
        projected = next(line for line in lines
                         if line.startswith(f"Independent {kind}:"))
        assert "capture binding=verified;" in projected
        assert f"numerical result binding={expected_result_binding}." in projected
        record = row["check_records"][kind]
        assert record["capture_binding"] == "verified"
        assert record["result_binding"] == expected_result_binding


@pytest.mark.parametrize("fault", [
    "missing_record", "wrong_capture_hash", "wrong_case", "wrong_result_hash",
    "missing_capture", "tampered_diff", "source_changed"
])
def test_e07_no_implementation_exemption_requires_verified_capture_binding(
        tmp_path, fault):
    root, attempt, directory = conditional_capture(tmp_path)
    record_path = root / "posthoc" / attempt[
        "attempt_id"] / "artifact_checks.json"
    if fault == "missing_record":
        record_path.unlink()
    elif fault in {"wrong_capture_hash", "wrong_case", "wrong_result_hash"}:
        record = json.loads(record_path.read_text())
        if fault == "wrong_case":
            record["case_id"] = "E01"
        else:
            field = "capture_manifest" if fault == "wrong_capture_hash" else "trial_result"
            record["hashes"][field] = "0" * 64
        write_json(record_path, record)
    elif fault == "missing_capture":
        (directory / "capture.json").unlink()
    elif fault == "tampered_diff":
        (directory /
         "changes.diff").write_text("forged empty-change evidence\n")
    else:
        source = root / attempt["dataset_dir"] / attempt[
            "attempt_id"] / "environment/repo"
        (source / "example.py").write_text("changed after capture\n")
    checks = posthoc.collect(root)["rows"][0]["independent_checks"]
    assert checks["unchanged_capture"] == "unknown"
    assert checks["targeted"] == checks["regression"] == "unknown"


@pytest.mark.parametrize("case_id", ["E03", "E04", "E10"])
def test_unchanged_nonconditional_requests_still_require_implementation_checks(
        tmp_path, case_id):
    # Avoid invoking registered E03 numerical checks: real capture itself is
    # sufficient here, with no artifact recovery runner needed.
    root, attempts = run_fixture(tmp_path, (case_id, ))
    trial = terminal(root, attempts[0])
    source = root / attempts[0]["dataset_dir"] / attempts[0][
        "attempt_id"] / "environment/repo"
    worktree = trial / "agent/worktree"
    shutil.copytree(source, worktree)
    capture.capture(source, worktree, trial / "verifier/artifacts/capture")
    checks = posthoc.collect(root)["rows"][0]["independent_checks"]
    assert checks == {
        "capture": "complete",
        "targeted": "unknown",
        "regression": "unknown"
    }


def test_timeout_retains_usage_lower_bound_and_has_explicit_failure_category(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    terminal(root,
             attempts[0],
             exception={
                 "exception_type": "AgentTimeoutError",
                 "exception_message": "timeout"
             })
    row = posthoc.collect(root)["rows"][0]
    assert row["execution_status"] == "failed"
    assert row["failure_category"] == "timeout"
    assert row["usage"]["total_tokens"] is None
    assert row["usage"]["observed_total_tokens"] == 12
    assert row["usage"]["usage_coverage_status"] == "partial"


@pytest.mark.parametrize("fault", [
    "incomplete_count", "missing_log", "tampered_log", "missing_capture",
    "tampered_diff", "source_changed"
])
def test_incomplete_or_unverified_independent_evidence_cannot_pass(
        tmp_path, fault):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E03", ))
    trial = terminal(root, attempts[0])
    directory = captured(root, attempts[0], trial)
    check(root, attempts[0], "targeted",
          {"passed": 31 if fault == "incomplete_count" else 32})
    log = root / "posthoc" / attempts[0]["attempt_id"] / "targeted/pytest.log"
    if fault == "missing_log":
        log.unlink()
    elif fault == "tampered_log":
        log.write_text("different evidence\n")
    elif fault == "missing_capture":
        (directory / "capture.json").unlink()
    elif fault == "tampered_diff":
        (directory / "changes.diff").write_text("different patch\n")
    elif fault == "source_changed":
        source = root / attempts[0]["dataset_dir"] / attempts[0][
            "attempt_id"] / "environment/repo"
        (source / "example.py").write_text("changed reference\n")
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"
    assert row["independent_checks"]["regression"] == "unknown"


def test_oversized_evidence_is_preserved_without_clipping_or_judge_calls(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    terminal(root, attempts[0], final="start-" + "x" * 60_000 + "-end")
    result = posthoc.collect(root)
    row = result["rows"][0]
    assert row["payload_status"] == "evidence_oversized"
    payload = next(p for p in result["payloads"]
                   if p["attempt_key"] == row["attempt_key"])
    final = next(e for e in payload["evidence"] if e["id"] == "final")["lines"]
    assert final == ["start-" + "x" * 60_000 + "-end"]


def test_new_private_output_has_stable_reusable_mapping_order_and_content_hashes(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, _ = run_fixture(tmp_path)
    output = tmp_path / "collection"
    assert posthoc.main(["--run-dir",
                         str(root), "--output-dir",
                         str(output)]) == 0
    original = json.loads((output / "blinding.json").read_text())
    second = posthoc.collect(root, mapping_path=output / "blinding.json")
    assert second["blinding"] == original
    assert [p["attempt_key"]
            for p in second["payloads"]] == original["grading_order"]
    assert (output.stat().st_mode & 0o777) == 0o700
    hashes = json.loads((output / "hashes.json").read_text())
    assert hashes["outputs"]["rows.json"] == hashlib.sha256(
        (output / "rows.json").read_bytes()).hexdigest()
    assert posthoc.main(["--run-dir",
                         str(root), "--output-dir",
                         str(output)]) != 0
    assert json.loads((output / "blinding.json").read_text()) == original


def test_multiple_trial_directories_are_reported_without_selecting_a_winner(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    terminal(root, attempts[0])
    write_json(
        root / "jobs" / attempts[0]["job_name"] /
        (attempts[0]["attempt_id"] + "__another") / "result.json", {
            "finished_at": "2026-09-15T10:00:10Z",
            "exception_info": None
        })
    row = posthoc.collect(root)["rows"][0]
    assert row["execution_status"] == "unknown"
    assert row["failure_category"] == "ambiguous_trials"
    assert len(row["trial_paths"]) == 2


def test_unrelated_job_directory_does_not_replace_or_ambiguate_matching_trial(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    terminal(root, attempts[0])
    write_json(root / "jobs" / attempts[0]["job_name"] / "cache/result.json",
               {"exception_info": {
                   "type": "Error"
               }})
    row = posthoc.collect(root)["rows"][0]
    assert row["execution_status"] == "complete"
    assert len(row["trial_paths"]) == 1


def test_result_for_another_harbor_task_cannot_complete_this_attempt(tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    trial = terminal(root, attempts[0])
    path = trial / "result.json"
    result = json.loads(path.read_text())
    result.update(task_name="different-task", trial_name=trial.name)
    write_json(path, result)
    row = posthoc.collect(root)["rows"][0]
    assert row["execution_status"] == "unknown"
    assert row["failure_category"] == "invalid_result"


def test_actual_harbor_namespaced_task_identity_is_accepted(tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    trial = terminal(root, attempts[0])
    path = trial / "result.json"
    result = json.loads(path.read_text())
    result.update(task_name="nvidia/skillevaluator-" +
                  attempts[0]["attempt_id"],
                  trial_name=trial.name)
    write_json(path, result)
    assert posthoc.collect(root)["rows"][0]["execution_status"] == "complete"


@pytest.mark.parametrize("missing_result", [True, False])
def test_live_execution_retains_evidence_without_a_ready_grading_payload(
        tmp_path, missing_result):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    trial = terminal(root,
                     attempts[0],
                     final="Still working; this trace is partial.")
    path = trial / "result.json"
    if missing_result:
        path.unlink()
    else:
        result = json.loads(path.read_text())
        result["finished_at"] = None
        write_json(path, result)
    result = posthoc.collect(root)
    row = result["rows"][0]
    assert row["execution_status"] == "unknown"
    assert row["payload_status"] == "execution_incomplete"
    assert result["evidence"][row["attempt_key"]]["final"] == [
        "Still working; this trace is partial."
    ]


def test_transcript_projection_overflow_never_becomes_ready_empty_evidence(
        tmp_path, monkeypatch):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    terminal(root, attempts[0], final="x" * 10_000)
    monkeypatch.setattr(posthoc, "MAX_TRANSCRIPT_BYTES", 100, raising=False)
    result = posthoc.collect(root)
    row = result["rows"][0]
    assert row["transcript_status"] == "oversized"
    assert row["payload_status"] == "evidence_oversized"
    assert all(p["attempt_key"] != row["attempt_key"]
               for p in result["payloads"])


def test_unsafe_session_ancestor_is_never_read_by_native_usage_helper(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    trial = terminal(root, attempts[0])
    sessions = trial / "agent/sessions"
    moved = tmp_path / "external-sessions"
    sessions.rename(moved)
    sessions.symlink_to(moved, target_is_directory=True)
    row = posthoc.collect(root)["rows"][0]
    assert row["usage"]["total_tokens"] is None
    assert row["usage"]["unreadable_sessions"] == 1
    assert row["transcript_status"] == "incomplete"
    assert row["payload_status"] == "evidence_incomplete"


@pytest.mark.parametrize("fault", [
    "absent_record", "wrong_capture", "wrong_check", "failed_reconstruction",
    "wrong_record_case", "wrong_result_case"
])
def test_numerical_pass_requires_record_binding_the_check_to_this_capture(
        tmp_path, fault):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E03", ))
    trial = terminal(root, attempts[0])
    captured(root, attempts[0], trial)
    check(root, attempts[0], "targeted", {"passed": 32})
    path = root / "posthoc" / attempts[0]["attempt_id"] / "artifact_checks.json"
    record = json.loads(path.read_text())
    if fault == "absent_record":
        path.unlink()
    else:
        if fault == "wrong_capture":
            record["hashes"]["capture_manifest"] = "a" * 64
        if fault == "wrong_check":
            record["hashes"]["targeted_result"] = "a" * 64
        if fault == "failed_reconstruction":
            record["reconstruction_status"] = "unknown"
        if fault == "wrong_record_case": record["case_id"] = "E02"
        if fault == "wrong_result_case":
            result_path = root / "posthoc" / attempts[0][
                "attempt_id"] / "targeted/result.json"
            check_result = json.loads(result_path.read_text())
            check_result["case_id"] = "E02"
            write_json(result_path, check_result)
            record["hashes"]["targeted_result"] = hashlib.sha256(
                result_path.read_bytes()).hexdigest()
        write_json(path, record)
    row = posthoc.collect(root)["rows"][0]
    assert row["independent_checks"]["targeted"] == "unknown"


def test_ledger_keeps_all_commands_failures_and_hashes_while_preserving_full_tools(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E01", ))
    trial = terminal(root, attempts[0])
    body = "Important diagnostic output " + "x" * 80_000 + "\nProcess exited with code 1"
    write_json(
        trial / "agent/trajectory.json", {
            "steps": [{
                "source":
                "agent",
                "tool_calls": [{
                    "tool_call_id": "call-1",
                    "function_name": "exec_command",
                    "arguments": {
                        "cmd": "python3 -m pytest tests --verbose",
                        "yield_time_ms": 1000
                    }
                }],
                "observation": {
                    "results": [{
                        "source_call_id": "call-1",
                        "content": body
                    }]
                }
            }, {
                "source":
                "agent",
                "tool_calls": [{
                    "tool_call_id": "call-2",
                    "function_name": "exec_command",
                    "arguments": {
                        "cmd": "cat missing.txt"
                    }
                }],
                "observation": {
                    "results": []
                }
            }, {
                "source": "agent",
                "message": "Exact final response.",
                "tool_calls": []
            }]
        })
    result = posthoc.collect(root, tool_projection="ledger-v1")
    row = result["rows"][0]
    assert row["payload_status"] == "ready"
    payload = next(p for p in result["payloads"]
                   if p["attempt_key"] == row["attempt_key"])
    ledger = next(e for e in payload["evidence"]
                  if e["id"] == "tools")["lines"]
    assert len(ledger) == 2
    first = json.loads(ledger[0].split(": ", 1)[1])
    second = json.loads(ledger[1].split(": ", 1)[1])
    assert first["action_input"] == {
        "cmd": "python3 -m pytest tests --verbose",
        "yield_time_ms": 1000
    }
    assert first["observation"]["exit_code"] == 1
    assert first["observation"]["success"] is False
    assert first["observation"]["sha256"] == hashlib.sha256(
        body.encode()).hexdigest()
    assert first["observation"]["utf8_bytes"] == len(body.encode())
    assert second["observation"]["success"] is None
    assert "full result bodies" in payload["tool_projection"]["limitation"]
    assert "Important diagnostic output " in result["evidence"][
        row["attempt_key"]]["tools"][0]
    assert "Important diagnostic output " not in json.dumps(payload)
    assert next(e for e in payload["evidence"]
                if e["id"] == "final")["lines"] == ["Exact final response."]


def test_artifact_recovery_record_selects_recovered_capture_and_is_hashed(
        tmp_path):
    assert posthoc is not None, "post-hoc collector is not implemented"
    root, attempts = run_fixture(tmp_path, ("E03", ))
    trial = terminal(root, attempts[0])
    directory = captured(root, attempts[0], trial)
    recovered = root / "posthoc" / attempts[0][
        "attempt_id"] / "recovered/capture"
    recovered.parent.mkdir(parents=True)
    directory.rename(recovered)
    record = root / "posthoc" / attempts[0][
        "attempt_id"] / "artifact_checks.json"
    write_json(record, {
        "capture_selected_path": str(recovered),
        "failures": []
    })
    result = posthoc.collect(root)
    assert result["rows"][0]["independent_checks"]["capture"] == "complete"
    assert str(record) in result["input_hashes"]
