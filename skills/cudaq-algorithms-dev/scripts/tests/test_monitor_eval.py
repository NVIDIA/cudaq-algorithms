"""Operational monitoring must tolerate live files without changing evidence."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "monitor_eval.py"
SPEC = importlib.util.spec_from_file_location("monitor_eval", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def campaign(tmp_path, cases=("case-a", ), seeds=(0, )):
    root = tmp_path / "campaigns" / "model-a"
    write(
        root / "campaign.json", {
            "models": [{
                "alias": "model-a",
                "api_key": "SECRET"
            }],
            "selected_cases": list(cases),
            "protocol": {
                "seeds": list(seeds)
            },
        })
    return root


def attempt(root, case="case-a", arm="baseline", seed=0):
    return root / "attempts" / "model-a" / case / str(seed) / arm


def result(path, outcome="answered", **extra):
    write(
        path / "result.json", {
            "case_id": path.parents[1].name,
            "seed": int(path.parent.name),
            "arm": path.name,
            "outcome": outcome,
            "final_answer": "PRIVATE ANSWER",
            **extra,
        })


def test_live_partial_result_is_not_completed_or_lost(tmp_path):
    root = campaign(tmp_path)
    path = attempt(root)
    write(path / "started.json", {"pid": os.getpid()})
    (path / "result.json").write_text('{"outcome":')
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert (row["scheduled"], row["completed"], row["active"],
            row["pending"]) == (2, 0, 1, 1)
    assert row["partial_or_unreadable_files"] == 1
    result(path)
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert (row["completed"], row["active"], row["pending"]) == (1, 0, 1)


def test_dead_and_mismatched_reservations_are_not_active(tmp_path):
    root = campaign(tmp_path)
    write(attempt(root) / "started.json", {"pid": 999999999})
    write(attempt(root, arm="skill") / "started.json", {"pid": os.getpid()})
    write(tmp_path / "launches.json",
          {"models": {
              "model-a": {
                  "controller_pid": 999999998
              }
          }})
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert row["unfinished"] == 2 and row["active"] == 0
    assert row["alerts"]["unfinished_without_live_pid"] == 1
    assert row["alerts"]["reservation_pid_mismatch"] == 1


def test_terminal_outcomes_verification_and_grading_are_separate(tmp_path):
    root = campaign(tmp_path, cases=("case-a", "case-b"))
    first = attempt(root)
    result(first)
    write(first / "verification.json", {"status": "passed"})
    write(first / "assessment.json", {"case_id": "case-a", "assertions": []})
    write(first / "assessment.provenance.json", {"result_sha256": "a"})
    result(attempt(root, arm="skill"),
           outcome="backend_error",
           final_answer="")
    result(attempt(root, case="case-b"),
           outcome="budget_timeout",
           final_answer=None)
    result(attempt(root, case="case-b", arm="skill"),
           outcome="no_answer",
           final_answer=None)
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert row["completed"] == row["scheduled"] == 4
    assert row["pending"] == row["active"] == row["unfinished"] == 0
    assert row["outcomes"] == {
        "answered": 1,
        "backend_error": 1,
        "budget_timeout": 1,
        "no_answer": 1
    }
    assert row["numerical_checks"]["passed"] == 1 and row["graded"] == 1
    assert row["alerts"]["backend_error"] == 1
    assert row["alerts"]["no_final_answer"] == 3


def test_every_canonical_terminal_outcome_counts_as_completed(tmp_path):
    schema = json.loads(
        (SCRIPT.parents[1] / "evals" / "results.schema.json").read_text())
    canonical = schema["$defs"]["run"]["properties"]["outcome"]["enum"]
    root = campaign(tmp_path,
                    cases=tuple("case-" + outcome for outcome in canonical))
    for outcome in canonical:
        result(attempt(root, case="case-" + outcome),
               outcome=outcome,
               final_answer="answer" if outcome == "answered" else None)
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert row["completed"] == len(canonical)
    assert row["unfinished"] == row["active"] == 0
    assert row["pending"] == len(canonical)
    assert row["outcomes"] == {outcome: 1 for outcome in canonical}
    assert "invalid_outcome" not in row["alerts"]


def test_missing_checks_follow_explicit_frozen_contract_only(tmp_path):
    root = campaign(tmp_path,
                    cases=("discussion", "implementation", "unknown"))
    spec = json.loads((root / "campaign.json").read_text())
    spec["case_contracts"] = {
        "discussion": {
            "executable_check": False
        },
        "implementation": {
            "executable_check": True
        },
    }
    write(root / "campaign.json", spec)
    for case in ("discussion", "implementation", "unknown"):
        result(attempt(root, case=case))
    partial = attempt(root, case="discussion", arm="skill")
    result(partial)
    (partial / "verification.json").write_text('{"status":')
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert row["completed"] == 4
    assert row["numerical_checks"] == {
        "not_applicable": 1,
        "missing_or_unreadable": 3
    }
    assert row["partial_or_unreadable_files"] == 1


def test_supervisor_pause_is_explicit_and_separate_from_pid_liveness(tmp_path):
    campaign(tmp_path)
    for state in ("paused", "cooldown"):
        write(tmp_path / "launches.json",
              {"models": {
                  "model-a": {
                      "pid": os.getpid(),
                      "status": state
                  }
              }})
        row = monitor.snapshot(tmp_path)["models"]["model-a"]
        assert row["controller"]["state"] == "running"
        assert row["controller"]["supervisor_state"] == state
        assert row["alerts"]["supervisor_" + state] == 1
    write(
        tmp_path / "launches.json", {
            "models": {
                "model-a": {
                    "pid": os.getpid(),
                    "status": "SECRET UNKNOWN STATUS"
                }
            }
        })
    data = monitor.snapshot(tmp_path)
    assert data["models"]["model-a"]["controller"]["supervisor_state"] is None
    assert "SECRET" not in json.dumps(data)


def test_wrong_attempt_result_and_unknown_outcome_do_not_count(tmp_path):
    root = campaign(tmp_path)
    result(attempt(root), case_id="another-case")
    result(attempt(root, arm="skill"), outcome="SECRET ERROR CONTENT")
    data = monitor.snapshot(tmp_path)
    row = data["models"]["model-a"]
    assert row["completed"] == 0 and row["unfinished"] == 2
    assert row["alerts"]["result_identity_mismatch"] == 1
    assert row["alerts"]["invalid_outcome"] == 1
    assert "SECRET" not in json.dumps(data)


def test_partial_reservation_and_campaign_do_not_crash(tmp_path):
    root = campaign(tmp_path)
    path = attempt(root)
    path.mkdir(parents=True)
    (path / "started.json").write_text('{"pid":')
    row = monitor.snapshot(tmp_path)["models"]["model-a"]
    assert row["unfinished"] == 1 and row["pending"] == 1
    (root / "campaign.json").write_text('{"models":')
    assert monitor.snapshot(tmp_path)["alerts"]["unreadable_campaign"] == 1


def test_malformed_status_values_do_not_crash_or_leak(tmp_path):
    root = campaign(tmp_path)
    result(attempt(root), outcome={"SECRET": True})
    result(attempt(root, arm="skill"))
    write(
        attempt(root, arm="skill") / "verification.json",
        {"status": ["SECRET"]})
    data = monitor.snapshot(tmp_path)
    row = data["models"]["model-a"]
    assert row["completed"] == 1 and row["unfinished"] == 1
    assert row["numerical_checks"]["missing_or_unreadable"] == 1
    assert "SECRET" not in json.dumps(data)


def test_cli_is_read_only_and_never_prints_content(tmp_path):
    root = campaign(tmp_path)
    path = attempt(root)
    result(path, outcome="backend_error", error={"message": "SECRET TOKEN"})
    (path / "transcript.jsonl").write_text('PROMPT SECRET\n')
    write(tmp_path / "launches.json",
          {"models": {
              "model-a": {
                  "pid": os.getpid()
              }
          }})
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "status.json"
    proc = subprocess.run([
        sys.executable,
        str(SCRIPT),
        str(tmp_path), "--output",
        str(output), "--once"
    ],
                          text=True,
                          capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert before == {
        p: p.read_bytes()
        for p in root.rglob("*") if p.is_file()
    }
    combined = proc.stdout + proc.stderr
    for name in ("status.json", "status.txt", "status.events.jsonl"):
        combined += (tmp_path / name).read_text()
    assert all(secret not in combined
               for secret in ("SECRET", "PRIVATE ANSWER", "PROMPT"))
    assert json.loads(output.read_text(
    ))["models"]["model-a"]["controller"]["state"] == "running"
    assert "not delivery metrics" in combined


def test_output_cannot_overwrite_campaign_evidence(tmp_path):
    root = campaign(tmp_path)
    target = root / "campaign.json"
    before = target.read_bytes()
    proc = subprocess.run([
        sys.executable,
        str(SCRIPT),
        str(tmp_path), "--output",
        str(target), "--once"
    ],
                          capture_output=True)
    assert proc.returncode != 0 and target.read_bytes() == before


def test_event_symlink_cannot_append_to_campaign_evidence(tmp_path):
    root = campaign(tmp_path)
    target = root / "campaign.json"
    before = target.read_bytes()
    (tmp_path / "status.events.jsonl").symlink_to(target)
    proc = subprocess.run([
        sys.executable,
        str(SCRIPT),
        str(tmp_path), "--output",
        str(tmp_path / "status.json"), "--once"
    ],
                          capture_output=True)
    assert proc.returncode != 0 and target.read_bytes() == before


def test_exit_file_takes_precedence_and_transcript_mtime_is_activity(tmp_path):
    root = campaign(tmp_path)
    path = attempt(root)
    write(path / "started.json", {"pid": os.getpid()})
    transcript = path / "transcript.jsonl"
    transcript.write_text("SECRET")
    os.utime(transcript, (2000000000, 2000000000))
    (tmp_path / "model-a.exit").write_text("2\n")
    write(tmp_path / "launches.json",
          {"model-a": {
              "pid": os.getpid(),
              "exit_code_file": "model-a.exit"
          }})
    row = monitor.snapshot(tmp_path, now=2000000010)["models"]["model-a"]
    assert row["controller"]["state"] == "exited" and row["controller"][
        "exit_code"] == 2
    assert row["last_activity_age_seconds"] == 10


def test_json_exit_record_matches_current_launcher_pid(tmp_path):
    launch = {
        "pid": os.getpid(),
        "controller_pid": 999999999,
        "exit_code_file": "model-a.exit"
    }
    write(tmp_path / "model-a.exit", {
        "pid": os.getpid(),
        "exit_code": 2,
        "finished_at": 2000000000
    })
    status = monitor.controller_status(tmp_path, launch)
    assert status["state"] == "exited" and status["exit_code"] == 2


def test_stale_partial_or_malformed_json_exit_cannot_end_current_launch(
        tmp_path):
    launch = {"pid": os.getpid(), "exit_code_file": "model-a.exit"}
    path = tmp_path / "model-a.exit"
    for record in ({
            "pid": os.getpid() + 1,
            "exit_code": 0
    }, {
            "pid": str(os.getpid()),
            "exit_code": 0
    }, {
            "pid": os.getpid(),
            "exit_code": True
    }, {
            "pid": os.getpid(),
            "exit_code": "SECRET"
    }, {
            "exit_code": 0
    }):
        write(path, record)
        status = monitor.controller_status(tmp_path, launch)
        assert status["state"] == "running" and status["exit_code"] is None
        assert "SECRET" not in json.dumps(status)
    path.write_text('{"pid":')
    assert monitor.controller_status(tmp_path, launch)["state"] == "running"


def test_json_exit_record_uses_controller_pid_when_launcher_pid_absent(
        tmp_path):
    launch = {"controller_pid": os.getpid(), "exit_code_file": "model-a.exit"}
    write(tmp_path / "model-a.exit", {"pid": os.getpid(), "exit_code": -15})
    status = monitor.controller_status(tmp_path, launch)
    assert status["state"] == "exited" and status["exit_code"] == -15
