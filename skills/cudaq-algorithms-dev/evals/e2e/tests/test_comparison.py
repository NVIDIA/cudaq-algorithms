# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal paired fixtures retain failures, missing telemetry and blocked slots."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def record(arm,
           tokens,
           seconds,
           *,
           rep=1,
           passed=True,
           attempted=True,
           scientific=True):
    return {
        "id":
        f"case--{rep}--{arm}",
        "case":
        "case",
        "repetition":
        rep,
        "arm":
        arm,
        "status":
        "pass" if passed else "model_failure",
        "passed":
        passed,
        "attempted":
        attempted,
        "elapsed_s":
        seconds,
        "time_to_verified_solution_s":
        seconds + 2 if passed and seconds is not None else None,
        "telemetry": {
            "totals": {
                "total_tokens": tokens
            }
        },
        "grading": {
            "checks": [{
                "returncode": 0,
                "numeric": {
                    "passed": scientific
                },
                "host_api": {
                    "passed": True
                },
                "device_api": {
                    "passed": True
                }
            } for _ in range(2)]
        }
    }


def test_paired_differences_use_all_attempts_and_only_complete_pairs():
    from comparison import paired_comparisons
    results = [
        record("baseline", 100, 10),
        record("previous", 120, 12),
        record("skill", 90, 9),
        record("baseline", 200, 20, rep=2),
        record("previous", None, 22, rep=2),
        record("skill", 250, 25, rep=2, passed=False),
        record("baseline", None, None, rep=3, passed=False, attempted=False),
        record("previous", None, None, rep=3, passed=False, attempted=False),
        record("skill", None, None, rep=3, passed=False, attempted=False)
    ]
    comparisons = paired_comparisons(results,
                                     ["baseline", "previous", "skill"])
    previous = comparisons["candidate_minus_previous"]
    baseline = comparisons["candidate_minus_baseline"]
    assert previous["total_tokens"]["sum"] == -30
    assert previous["total_tokens"]["n"] == 1
    assert previous["elapsed_s"]["sum"] == 0
    assert baseline["total_tokens"][
        "sum"] == 40  # -10 and +50, failed run retained
    assert baseline["elapsed_s"]["sum"] == 4
    assert baseline["paired_attempts"] == 2
    assert baseline["blocked_pairs"] == 1
    assert baseline["time_to_verified_solution_s"]["n"] == 1
    assert baseline["time_to_verified_solution_s"]["sum"] == -1
    assert baseline["strict_pass_delta"]["sum"] == -1
    assert baseline["scientific_api_pass_delta"]["sum"] == 0


def test_scientific_outcome_does_not_hide_contract_failures_or_missing_checks(
):
    from comparison import scientific_api_passed
    assert scientific_api_passed(record("skill", 100, 10, passed=False))
    assert not scientific_api_passed(record("skill", 100, 10,
                                            scientific=False))
    assert not scientific_api_passed({"passed": True})
    incomplete = record("skill", 100, 10)
    incomplete["grading"]["checks"].pop()
    assert not scientific_api_passed(incomplete)


def test_paired_request_peaks_retain_missing_values_and_failures():
    from comparison import paired_comparisons
    rows = [
        record("baseline", 100, 10),
        record("previous", 120, 12),
        record("skill", 90, 9, passed=False),
        record("baseline", 130, 10, rep=2),
        record("previous", 140, 12, rep=2),
        record("skill", 150, 9, rep=2)
    ]
    for row, peak in zip(rows, [60, 80, 50, 70, None, 90]):
        row["telemetry"]["observed_peak_request_input_tokens"] = peak
    comparisons = paired_comparisons(rows, ["baseline", "previous", "skill"])
    assert comparisons["candidate_minus_previous"][
        "observed_peak_request_input_tokens"]["sum"] == -30
    assert comparisons["candidate_minus_previous"][
        "observed_peak_request_input_tokens"]["n"] == 1
    assert comparisons["candidate_minus_baseline"][
        "observed_peak_request_input_tokens"]["sum"] == 10
    assert comparisons["candidate_minus_baseline"][
        "observed_peak_request_input_tokens"]["n"] == 2


def test_entirely_pending_blocks_count_as_missing_pairs():
    from comparison import paired_comparisons
    planned = [
        record(arm, None, None, rep=rep, passed=False, attempted=False)
        for rep in (1, 2) for arm in ("baseline", "previous", "skill")
    ]
    actual = [
        record("baseline", 100, 10),
        record("previous", 120, 12),
        record("skill", 90, 9)
    ]
    for rows, expected_missing in [(actual, 1), ([], 2)]:
        comparisons = paired_comparisons(rows,
                                         ["baseline", "previous", "skill"],
                                         planned_runs=planned)
        assert comparisons["candidate_minus_previous"][
            "missing_pairs"] == expected_missing
        assert comparisons["candidate_minus_baseline"][
            "missing_pairs"] == expected_missing


def test_three_arm_report_keeps_all_outcomes_and_native_completeness(tmp_path):
    import json
    from run import report
    from telemetry import TelemetryCollector
    rows = [
        record("baseline", 100, 10),
        record("previous", 120, 12),
        record("skill", 90, 9, passed=False)
    ]
    blocked = record("skill",
                     None,
                     None,
                     rep=2,
                     passed=False,
                     attempted=False,
                     scientific=False)
    blocked["status"] = "infrastructure_failure"
    rows.append(blocked)
    for row in rows:
        totals = row["telemetry"]["totals"]
        row["telemetry"] = TelemetryCollector().summary()
        row["telemetry"]["totals"].update(totals)
        path = tmp_path / "runs" / row["id"] / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row))
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "arms": {
                a: {}
                for a in ("baseline", "previous", "skill")
            },
            "runs": rows,
            "cases": [{
                "id": "case",
                "families": ["example"]
            }]
        }))
    report(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["executed_attempts"] == 3
    assert summary["preflight_blocked_slots"] == 1
    assert summary["arms"]["skill"]["strict_passes"] == 0
    assert summary["arms"]["skill"]["scientific_api_passes"] == 1
    assert summary["arms"]["skill"]["tokens_all_attempts"]["sum"] == 90
    assert summary["arms"]["skill"]["attempt_seconds"]["sum"] == 9
    assert summary["arms"]["skill"]["successful_solution_seconds"]["n"] == 0
    assert summary["arms"]["skill"]["metric_completeness_counts"][
        "totals"] == 0
    assert summary["paired_comparisons"]["candidate_minus_previous"][
        "total_tokens"]["sum"] == -30
    assert summary["paired_comparisons"]["candidate_minus_baseline"][
        "total_tokens"]["sum"] == -10
    assert summary["families"]["example"]["skill"][
        "scientific_api_passes"] == 1
    assert "previous strict/scientific/executed" in (tmp_path /
                                                     "REPORT.md").read_text()
