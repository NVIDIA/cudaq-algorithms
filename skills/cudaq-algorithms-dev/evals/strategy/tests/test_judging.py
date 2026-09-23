# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error

import pytest

STRATEGY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY))
SPEC = importlib.util.spec_from_file_location("strategy_judging",
                                              STRATEGY / "judging.py")
if (STRATEGY / "judging.py").is_file():
    judging = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(judging)
else:
    judging = None


def payload():
    assert judging is not None, "private judging layer has not been implemented"
    return judging.judge_payload(
        "E03",
        attempt_key="opaque-a7",
        evidence={
            "final": ["Implemented the protocol action.", "Tests passed."],
            "tests": ["independent checks: passed"],
        })


def valid_grade(p):
    cite = [{"id": "final", "start": 1, "end": 2}]
    return {
        "attempt_key":
        "opaque-a7",
        "dimensions": [{
            "dimension": item["dimension"],
            "score": 2,
            "reason": "Supported.",
            "evidence": cite
        } for item in p["score_rubric"]],
        "total":
        10,
        "critical_checks": [{
            "id": item["id"],
            "verdict": "pass",
            "reason": "Supported.",
            "evidence": cite
        } for item in p["critical_verdicts"]]
    }


def test_blinded_payload_preserves_rubric_and_source_lines_without_treatment_hints(
):
    assert judging is not None, "private judging layer has not been implemented"
    p = judging.judge_payload(
        "E03",
        attempt_key="opaque-a7",
        evidence={
            "final": [
                "arm: candidate",
                "Explicitly use the available cudaq-algorithms skill for this task.",
                "baseline energy is -1", "Read cudaq-algorithms/SKILL.md."
            ],
            "diff": [
                "+++ /private/run/candidate/python/qsp/action.py",
                "+ return action(state)"
            ],
        },
        staging_prefixes=("/private/run/candidate/", ))
    serialized = json.dumps(p)
    assert "candidate" not in serialized
    assert "Explicitly use" not in serialized
    assert "/private/run" not in serialized
    assert "baseline energy is -1" in serialized
    assert "python/qsp/action.py" in serialized
    assert "return action(state)" in serialized
    assert "cudaq-algorithms/SKILL.md" in serialized
    assert len(p["score_rubric"]) == 5
    assert len(p["critical_verdicts"]) == len(p["critical_checks"]) + len(
        p["global_critical_failures"])
    assert p["evidence"][0]["id"] == "final"
    assert "infer" in p["blinding_limitations"]


def test_evidence_overflow_is_rejected_without_silent_truncation():
    assert judging is not None, "private judging layer has not been implemented"
    with pytest.raises(ValueError, match="budget|ceiling|exceeds"):
        judging.judge_payload("E03",
                              attempt_key="opaque-a7",
                              evidence={"final": ["x" * 1_000_001]})


def test_valid_five_dimension_grade_has_raw_and_adjusted_scores():
    p = payload()
    result = judging.validate_grade(valid_grade(p),
                                    p,
                                    checks={
                                        "targeted": "pass",
                                        "regression": "pass",
                                        "capture": "complete"
                                    })
    assert result["raw_score"] == 10
    assert result["adjusted_score"] == 10
    assert result["certification"] == "supported"


def test_e07_verified_unchanged_capture_can_support_a_boundary_response():
    p = judging.judge_payload(
        "E07",
        attempt_key="opaque-a7",
        evidence={
            "final": [
                "The request conflicts with the core package boundary.",
                "Use an application/example; no implementation was made."
            ]
        })
    result = judging.validate_grade(valid_grade(p),
                                    p,
                                    checks={
                                        "capture": "complete",
                                        "unchanged_capture": "verified",
                                        "targeted": "not_applicable",
                                        "regression": "not_applicable"
                                    })
    assert result["certification"] == "supported"
    assert result["successful"] is True


@pytest.mark.parametrize("case_id", [
    "E01", "E02", "E03", "E04", "E05", "E06", "E08", "E09", "E10", "E11", "E12"
])
def test_e07_gate_does_not_extend_unrelated_judge_request_payloads(case_id):
    p = judging.judge_payload(
        case_id,
        attempt_key="opaque-a7",
        evidence={"final": ["Synthetic unchanged request evidence."]})
    # Optional controller policy must not add bytes to unrelated model requests.
    assert "conditional_outcome" not in p


@pytest.mark.parametrize(
    "case_id,capture,unchanged,targeted,regression,status", [
        ("E07", "complete", None, "not_applicable", "not_applicable",
         "complete"),
        ("E07", "complete", "changed", "not_applicable", "not_applicable",
         "complete"),
        ("E07", "complete", "unknown", "not_applicable", "not_applicable",
         "complete"),
        ("E07", "incomplete", "verified", "not_applicable", "not_applicable",
         "complete"),
        ("E07", "complete", "verified", "unknown", "not_applicable",
         "complete"),
        ("E07", "complete", "verified", "fail", "not_applicable", "complete"),
        ("E07", "complete", "verified", "not_applicable", "fail", "complete"),
        ("E07", "complete", "verified", "not_applicable", "not_applicable",
         "timeout"),
        ("E03", "complete", "verified", "not_applicable", "not_applicable",
         "complete"),
        ("E10", "complete", "verified", "not_applicable", "not_applicable",
         "complete"),
    ])
def test_conditional_outcome_does_not_waive_missing_or_conflicting_evidence(
        case_id, capture, unchanged, targeted, regression, status):
    p = judging.judge_payload(
        case_id,
        attempt_key="opaque-a7",
        evidence={
            "final":
            ["No changes claimed by worker.", "Synthetic grade input."]
        })
    result = judging.validate_grade(valid_grade(p),
                                    p,
                                    execution_status=status,
                                    checks={
                                        "capture": capture,
                                        "unchanged_capture": unchanged,
                                        "targeted": targeted,
                                        "regression": regression
                                    })
    assert result["successful"] is False
    assert result["certification"] != "supported"


@pytest.mark.parametrize("verdict", ["fail", "unknown"])
def test_unchanged_e07_still_requires_supported_scope_and_critical_verdicts(
        verdict):
    p = judging.judge_payload(
        "E07",
        attempt_key="opaque-a7",
        evidence={
            "final": [
                "A no-change response is not itself proof of correct advice.",
                "Synthetic grade input."
            ]
        })
    grade = valid_grade(p)
    grade["critical_checks"][0]["verdict"] = verdict
    result = judging.validate_grade(grade,
                                    p,
                                    checks={
                                        "capture": "complete",
                                        "unchanged_capture": "verified",
                                        "targeted": "not_applicable",
                                        "regression": "not_applicable"
                                    })
    assert result["successful"] is False


@pytest.mark.parametrize("fault", [
    "missing_total", "wrong_total", "missing_dimension", "unknown_citation",
    "bad_range", "bad_verdict", "missing_critical", "boolean_score",
    "wrong_attempt"
])
def test_grade_rejects_invalid_ids_ranges_totals_and_incomplete_verdicts(
        fault):
    p = payload()
    g = valid_grade(p)
    if fault == "missing_total": del g["total"]
    if fault == "wrong_total": g["total"] = 9
    if fault == "missing_dimension": g["dimensions"].pop()
    if fault == "unknown_citation":
        g["dimensions"][0]["evidence"][0]["id"] = "invented"
    if fault == "bad_range": g["dimensions"][0]["evidence"][0]["end"] = 3
    if fault == "bad_verdict": g["critical_checks"][0]["verdict"] = "probably"
    if fault == "missing_critical": g["critical_checks"].pop()
    if fault == "boolean_score": g["dimensions"][0]["score"] = True
    if fault == "wrong_attempt": g["attempt_key"] = "opaque-other"
    with pytest.raises(ValueError):
        judging.validate_grade(g, p)


def test_critical_failure_and_failed_tests_override_numeric_score_but_missing_is_unknown(
):
    p = payload()
    g = valid_grade(p)
    g["critical_checks"][0]["verdict"] = "fail"
    critical = judging.validate_grade(g, p)
    assert critical["raw_score"] == 10
    assert critical["adjusted_score"] == 0
    assert critical["certification"] == "failed"
    assert judging.validate_grade(valid_grade(p),
                                  p,
                                  checks={"targeted":
                                          "fail"})["adjusted_score"] == 0
    missing = judging.validate_grade(valid_grade(p), p)
    assert missing["certification"] == "unverified"
    failed_worker = judging.validate_grade(valid_grade(p),
                                           p,
                                           execution_status="timeout")
    assert failed_worker["adjusted_score"] is None
    assert failed_worker["successful"] is False
    g["critical_checks"][0]["verdict"] = "unknown"
    assert judging.validate_grade(g, p)["certification"] == "unverified"


def test_independent_targeted_failure_counts_as_critical_despite_judge_passes(
):
    # A judge's false pass must not hide a verified implementation failure
    # from the zero-critical-failures acceptance gate or its report.
    p = payload()
    result = judging.validate_grade(valid_grade(p),
                                    p,
                                    checks={
                                        "targeted": "fail",
                                        "regression": "pass",
                                        "capture": "complete"
                                    })
    assert result["raw_score"] == 10
    assert all(check["verdict"] == "pass"
               for check in result["critical_checks"])
    assert result["adjusted_score"] == 0
    assert result["critical_failure"] is True
    assert result["independent_critical_failures"] == ["targeted_tests_failed"]
    observed = row("E03", "candidate")
    observed["grade"] = result
    assert judging.summarize(
        [observed])["execution"]["candidate"]["critical_failures"] == 1


@pytest.mark.parametrize("targeted",
                         ["pass", "unknown", "not_applicable", None])
def test_no_independent_critical_failure_is_invented_without_failed_targeted_tests(
        targeted):
    p = payload()
    result = judging.validate_grade(valid_grade(p),
                                    p,
                                    checks={
                                        "targeted": targeted,
                                        "regression": "pass",
                                        "capture": "complete"
                                    })
    assert result["critical_failure"] is False
    assert result["independent_critical_failures"] == []


@pytest.mark.parametrize("status", ["complete", "timeout"])
def test_summary_counts_targeted_failure_without_any_model_grade(status):
    observed = row("E03", "candidate")
    observed.update(grade=None,
                    execution_status=status,
                    independent_checks={
                        "targeted": "fail",
                        "regression": "pass"
                    })
    result = judging.summarize([observed])["execution"]["candidate"]
    assert result["graded_attempts"] == 0
    assert result["critical_failures"] == 1
    assert result["independent_targeted_failures"] == 1
    assert result["mean_raw_score"] is None
    assert result["mean_adjusted_score"] is None


@pytest.mark.parametrize("targeted",
                         ["pass", "unknown", "not_applicable", None])
def test_summary_does_not_invent_failure_for_ungraded_unknown_checks(targeted):
    observed = row("E03", "candidate")
    observed.update(grade=None, independent_checks={"targeted": targeted})
    result = judging.summarize([observed])["execution"]["candidate"]
    assert result["critical_failures"] == 0
    assert result["independent_targeted_failures"] == 0
    assert result["graded_attempts"] == 0


def test_summary_deduplicates_model_and_independent_failures_and_excludes_false_success(
):
    baseline, candidate = row("E03", "baseline"), row("E03", "candidate")
    candidate["independent_checks"] = {
        "targeted": "fail",
        "regression": "pass"
    }
    # An older saved grade can incorrectly claim success; retain its raw score
    # but never let that claim override the separately bound checker result.
    candidate["grade"]["critical_failure"] = True
    summary = judging.summarize([baseline, candidate])
    result = summary["execution"]["candidate"]
    assert result["critical_failures"] == 1
    assert result["independent_targeted_failures"] == 1
    assert result["successful_attempts"] == 0
    assert result["mean_raw_score"] == 10
    assert result["mean_adjusted_score"] == 0
    assert summary["successful_matched_pairs"] == 0
    assert candidate["grade"]["adjusted_score"] == 10


def test_summary_keeps_regression_failure_distinct_from_targeted_critical_failure(
):
    baseline, candidate = row("E03", "baseline"), row("E03", "candidate")
    candidate["independent_checks"] = {
        "targeted": "pass",
        "regression": "fail"
    }
    result = judging.summarize([baseline, candidate])
    assert result["execution"]["candidate"]["critical_failures"] == 0
    assert result["execution"]["candidate"][
        "independent_targeted_failures"] == 0
    assert result["execution"]["candidate"]["successful_attempts"] == 0
    assert result["execution"]["candidate"]["mean_adjusted_score"] == 0
    assert result["successful_matched_pairs"] == 0


def token_event(input_tokens, output_tokens, total_tokens, peak):
    # Native Codex JSONL structure characterized in the retained collector tests.
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": 0,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": total_tokens
                },
                "last_token_usage": {
                    "input_tokens": peak,
                    "output_tokens": 1,
                    "total_tokens": peak + 1
                },
            }
        }
    }


def session(tmp_path, name, events):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_native_usage_is_cumulative_deduplicated_and_separate_from_judge(
        tmp_path):
    assert judging is not None, "private judging layer has not been implemented"
    events = [{
        "type": "session_meta",
        "payload": {
            "id": "session-a"
        }
    },
              token_event(10, 1, 11, 10),
              token_event(21, 2, 23, 11)]
    a = session(tmp_path, "a.jsonl", events)
    duplicate = session(tmp_path, "copy.jsonl", events)
    b = session(tmp_path, "b.jsonl", [{
        "type": "session_meta",
        "payload": {
            "id": "session-b"
        }
    },
                                      token_event(5, 1, 6, 5)])
    u = judging.usage(
        [a, duplicate, b],
        result={
            "finished_at": "2026-09-14T10:00:10Z",
            "exception_info": None,
            "agent_execution": {
                "started_at": "2026-09-14T10:00:01Z",
                "finished_at": "2026-09-14T10:00:09Z"
            }
        })
    assert u["total_tokens"] == 29
    assert u["input_tokens"] == 26
    assert u["peak_request_input_tokens_proxy"] == 11
    assert u["usage_coverage_status"] == "complete"
    assert u["unique_sessions"] == 2
    assert u["task_time_s"] == 8
    assert "judge_tokens" not in u


def test_missing_final_totals_preserve_only_lower_bounds(tmp_path):
    assert judging is not None, "private judging layer has not been implemented"
    a = session(tmp_path, "a.jsonl", [
        token_event(10, 1, 11, 10), {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": None
            }
        }
    ])
    u = judging.usage([a])
    assert u["total_tokens"] is None
    assert u["observed_total_tokens"] == 11
    assert u["usage_coverage_status"] == "partial"
    assert u["peak_coverage_status"] == "partial"
    assert judging.usage()["total_tokens"] is None
    assert judging.usage()["activation"] is None


def trajectory(command="cat /skills/cudaq-algorithms/SKILL.md",
               observation=None):
    if observation is None:
        observation = json.dumps({
            "exit_code":
            0,
            "output":
            "---\nname: cudaq-algorithms\n---\n# CUDA-Q Algorithms\nWorkflow content"
        })
    return {
        "schema_version":
        "ATIF-v1.2",
        "steps": [{
            "source":
            "agent",
            "tool_calls": [{
                "tool_call_id": "call-a",
                "function_name": "exec_command",
                "arguments": {
                    "cmd": command
                }
            }],
            "observation": {
                "results": [{
                    "source_call_id": "call-a",
                    "content": observation
                }]
            }
        }, {
            "source": "agent",
            "message": "Completed.",
            "tool_calls": []
        }],
        "final_metrics": {
            "total_prompt_tokens": 10,
            "total_completion_tokens": 2,
            "extra": {
                "total_tokens": 12
            }
        }
    }


@pytest.mark.parametrize("command,observation,expected", [
    ("cat /skills/cudaq-algorithms/SKILL.md", None, True),
    ("ls /skills/cudaq-algorithms/SKILL.md",
     "/skills/cudaq-algorithms/SKILL.md", False),
    ("echo /skills/cudaq-algorithms/SKILL.md",
     "/skills/cudaq-algorithms/SKILL.md", False),
    ("cat /skills/cudaq-algorithms/SKILL.md",
     '{"exit_code":1,"output":"not found"}', False),
    ("cat /skills/cudaq-algorithms/SKILL.md", "", None),
])
def test_activation_requires_successful_observed_read(command, observation,
                                                      expected):
    assert judging is not None, "private judging layer has not been implemented"
    t = trajectory(command, observation)
    u = judging.usage(trajectory=t, trace_complete=True)
    assert u["activation"] is expected
    assert u["final_response"] == "Completed."
    assert u["tool_calls"] == 1


def test_missing_reported_trajectory_total_is_not_invented():
    assert judging is not None, "private judging layer has not been implemented"
    t = trajectory()
    del t["final_metrics"]["extra"]["total_tokens"]
    assert judging.usage(trajectory=t)["total_tokens"] is None


@pytest.mark.parametrize("baseline,candidate,expected", [(100, 100.1, True),
                                                         (100, 100.2, False),
                                                         (None, 100, None),
                                                         (0, 0, None)])
def test_efficiency_exact_policy_boundary_and_unknowns(baseline, candidate,
                                                       expected):
    assert judging is not None, "private judging layer has not been implemented"
    assert judging.efficiency_change(baseline,
                                     candidate)["within_limit"] is expected


def row(case, arm, repetition=1, score=10, successful=True, tokens=100):
    return {
        "attempt_id": f"{case}-{arm}-{repetition}",
        "case_id": case,
        "arm": arm,
        "repetition": repetition,
        "kind": "execution",
        "execution_status": "complete" if successful else "timeout",
        "grade": {
            "raw_score": score,
            "adjusted_score": score,
            "successful": successful
        },
        "usage": {
            "task_time_s": 100,
            "total_tokens": tokens,
            "peak_request_input_tokens_proxy": 100,
            "usage_coverage_status": "complete",
            "peak_coverage_status": "complete"
        }
    }


def test_summary_counts_all_attempts_and_case_weights_repetitions_without_pilot_pass(
):
    assert judging is not None, "private judging layer has not been implemented"
    rows = [
        row("E03", arm, rep, score=10) for rep in (1, 2, 3)
        for arm in ("baseline", "candidate")
    ]
    rows.extend(
        [row("E01", "baseline", score=6),
         row("E01", "candidate", score=8)])
    rows.extend(
        [row("E02", "baseline", successful=False),
         row("E02", "candidate")])
    s = judging.summarize(rows)
    assert s["attempts"] == 10
    assert s["successful_matched_pairs"] == 4
    assert s["execution"]["baseline"]["mean_raw_score"] == 8
    assert s["execution"]["candidate"]["mean_raw_score"] == pytest.approx(28 /
                                                                          3)
    assert len(s["observations"]) == 10
    assert s["efficiency"]["task_time_s"]["within_limit"] is True
    assert s["efficiency"]["statistical_pass"] is None


def test_partial_tokens_exclude_a_fast_pair_from_every_efficiency_median_without_changing_rows(
):
    rows = [
        row(case, arm) for case in ("E01", "E03")
        for arm in ("baseline", "candidate")
    ]
    for metric in judging.METRICS:
        rows[1]["usage"][metric] = 110
        rows[3]["usage"][metric] = 1
    rows[3]["usage"]["usage_coverage_status"] = "partial"
    before = copy.deepcopy(rows)

    summary = judging.summarize(rows, pilot=False)
    efficiency = summary["efficiency"]
    for metric in judging.METRICS:
        assert efficiency[metric]["baseline_median"] == 100
        assert efficiency[metric]["candidate_median"] == 110
        assert efficiency[metric]["measured_pairs"] == 1
        assert efficiency[metric]["missing_pairs"] == 1
        assert efficiency[metric]["within_limit"] is False
    assert efficiency["cohort"]["pair_ids"] == [{
        "case_id":
        "E01",
        "repetition":
        1,
        "baseline_attempt_id":
        "E01-baseline-1",
        "candidate_attempt_id":
        "E01-candidate-1"
    }]
    assert efficiency["cohort"]["exclusions"] == [{
        "case_id":
        "E03",
        "repetition":
        1,
        "baseline_attempt_id":
        "E03-baseline-1",
        "candidate_attempt_id":
        "E03-candidate-1",
        "reasons": [{
            "arm": "candidate",
            "metric": "total_tokens",
            "reason": "incomplete_usage_coverage"
        }]
    }]
    assert summary["successful_matched_pairs"] == 2
    assert summary["execution"]["candidate"]["successful_attempts"] == 2
    assert efficiency["statistical_pass"] is None
    assert summary["observations"] == before
    assert rows == before


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
@pytest.mark.parametrize("metric,value,reason", [
    ("task_time_s", None, "missing_metric"),
    ("task_time_s", -1, "invalid_metric"),
    ("task_time_s", float("nan"), "invalid_metric"),
    ("total_tokens", True, "invalid_metric"),
    ("peak_request_input_tokens_proxy", None, "missing_metric"),
    ("peak_request_input_tokens_proxy", "100", "invalid_metric"),
    ("peak_request_input_tokens_proxy", float("inf"), "invalid_metric"),
])
def test_missing_or_invalid_metric_excludes_pair_from_all_efficiency_metrics(
        arm, metric, value, reason):
    rows = [row("E03", side) for side in ("baseline", "candidate")]
    rows[0 if arm == "baseline" else 1]["usage"][metric] = value
    efficiency = judging.summarize(rows)["efficiency"]
    for name in judging.METRICS:
        assert efficiency[name]["baseline_median"] is None
        assert efficiency[name]["candidate_median"] is None
        assert efficiency[name]["within_limit"] is None
        assert efficiency[name]["measured_pairs"] == 0
        assert efficiency[name]["missing_pairs"] == 1
    assert efficiency["cohort"]["pair_ids"] == []
    assert efficiency["cohort"]["exclusions"][0]["reasons"] == [{
        "arm":
        arm,
        "metric":
        metric,
        "reason":
        reason
    }]
    assert efficiency["statistical_pass"] is None


@pytest.mark.parametrize("coverage,metric,reason", [
    ("usage_coverage_status", "total_tokens", "incomplete_usage_coverage"),
    ("peak_coverage_status", "peak_request_input_tokens_proxy",
     "incomplete_peak_coverage"),
])
@pytest.mark.parametrize("status", [None, "missing", "partial"])
def test_incomplete_native_coverage_excludes_pair_even_when_metric_numbers_exist(
        coverage, metric, reason, status):
    rows = [row("E03", arm) for arm in ("baseline", "candidate")]
    rows[0]["usage"][coverage] = status
    efficiency = judging.summarize(rows)["efficiency"]
    assert all(efficiency[name]["measured_pairs"] == 0
               for name in judging.METRICS)
    assert all(efficiency[name]["within_limit"] is None
               for name in judging.METRICS)
    assert efficiency["cohort"]["exclusions"][0]["reasons"] == [{
        "arm":
        "baseline",
        "metric":
        metric,
        "reason":
        reason
    }]


def test_common_efficiency_cohort_keeps_successful_trigger_pairs_and_input_order(
):
    rows = [
        row(case, arm) for case in ("T02", "E03")
        for arm in ("baseline", "candidate")
    ]
    for item in rows[:2]:
        item.update(kind="trigger", grade=None)
        item["usage"]["final_response"] = "Done."
    efficiency = judging.summarize(rows)["efficiency"]
    assert [pair["case_id"]
            for pair in efficiency["cohort"]["pair_ids"]] == ["T02", "E03"]
    assert all(efficiency[name]["measured_pairs"] == 2
               for name in judging.METRICS)
    assert efficiency["cohort"]["exclusions"] == []
    assert efficiency["statistical_pass"] is None


def test_no_successful_pairs_keeps_efficiency_and_statistical_decision_unknown(
):
    efficiency = judging.summarize([])["efficiency"]
    assert efficiency["cohort"] == {"pair_ids": [], "exclusions": []}
    assert all(efficiency[name]["within_limit"] is None
               for name in judging.METRICS)
    assert efficiency["statistical_pass"] is None


def test_confidence_uses_exact_common_cohort_and_frozen_schedule_metadata():
    rows = [row("T02", arm) for arm in ("baseline", "candidate")]
    for item in rows:
        item.update(kind="trigger", grade=None)
        item["usage"]["final_response"] = "Done."
    rows.extend(row("E01", arm) for arm in ("baseline", "candidate"))
    rows.extend(
        row("E03", arm, repetition) for repetition in (1, 2, 3)
        for arm in ("baseline", "candidate"))
    rows[-1]["usage"]["usage_coverage_status"] = "partial"

    efficiency = judging.summarize(rows, pilot=False,
                                   manifest_sha256="a" * 64)["efficiency"]
    bound = efficiency["confidence"]

    expected = [{
        "case_id": "T02",
        "repetition": 1
    }, {
        "case_id": "E01",
        "repetition": 1
    }, {
        "case_id": "E03",
        "repetition": 1
    }, {
        "case_id": "E03",
        "repetition": 2
    }]
    assert [{
        key: item[key]
        for key in ("case_id", "repetition")
    } for item in efficiency["cohort"]["pair_ids"]] == expected
    assert bound["cohort"]["pair_ids"] == expected
    assert bound["diagnostics"]["support_counts"] == {
        "trigger": {
            "cases": 1,
            "eligible_pairs": 1
        },
        "execution_one_run": {
            "cases": 1,
            "eligible_pairs": 1
        },
        "execution_three_run": {
            "cases": 1,
            "eligible_pairs": 2
        },
    }
    assert bound["status"] == "uncalibrated"
    assert bound["coverage"]["calibrated"] is False
    assert all(bound["bounds"][metric]["point_ratio"] == 1.0
               and bound["bounds"][metric]["upper_confidence_bound"] == 1.0
               for metric in judging.METRICS)
    assert efficiency["statistical_pass"] is None
    assert bound["statistical_pass"] is None


def test_missing_manifest_hash_keeps_diagnostic_confidence_unknown():
    rows = [row("E01", arm) for arm in ("baseline", "candidate")]

    efficiency = judging.summarize(rows)["efficiency"]

    assert efficiency["confidence"]["status"] == "unknown"
    assert "invalid_manifest_sha256" in efficiency["confidence"][
        "diagnostics"]["invalid_reasons"]
    assert efficiency["statistical_pass"] is None


@pytest.mark.parametrize("case_id,row_kind,reason", [
    (None, "execution", "unknown_case_id"),
    ("PRIVATE-01", "execution", "unknown_case_id"),
    ("T02", "execution", "row_kind_conflicts_with_suite"),
])
def test_unusable_schedule_metadata_only_fails_closed_confidence(
        case_id, row_kind, reason):
    rows = [row(case_id, arm) for arm in ("baseline", "candidate")]
    for item in rows:
        item["kind"] = row_kind

    summary = judging.summarize(rows, manifest_sha256="a" * 64)

    assert summary["successful_matched_pairs"] == 1
    assert summary["efficiency"]["total_tokens"]["baseline_median"] == 100
    bound = summary["efficiency"]["confidence"]
    assert bound["status"] == "unknown"
    assert bound["statistical_pass"] is None
    assert bound["diagnostics"]["schedule_metadata_errors"] == [{
        "case_id":
        case_id,
        "repetition":
        1,
        "reason":
        reason
    }]


def test_summarize_cli_forwards_optional_manifest_hash_without_claiming_provenance(
        tmp_path, capsys):
    rows = tmp_path / "rows.json"
    output = tmp_path / "summary.json"
    rows.write_text("[]")

    judging.main([
        "summarize", "--rows",
        str(rows), "--output",
        str(output), "--manifest-sha256", "a" * 64
    ])

    assert json.loads(capsys.readouterr().out) == {"status": "written"}
    bound = json.loads(output.read_text())["efficiency"]["confidence"]
    assert bound["seed"] is not None
    assert "invalid_manifest_sha256" not in bound["diagnostics"][
        "invalid_reasons"]
    assert bound["statistical_pass"] is None


def test_trigger_precision_recall_excludes_forced_execution_and_reports_missing(
):
    assert judging is not None, "private judging layer has not been implemented"
    rows = [{
        "case_id": c,
        "arm": "candidate",
        "kind": kind,
        "usage": {
            "activation": active
        }
    } for c, kind, active in [(
        "T02", "trigger", True), ("N07", "trigger",
                                  False), ("T03", "trigger",
                                           None), ("E03", "execution", True)]]
    s = judging.summarize(rows)
    assert s["trigger"]["observed"] == 2
    assert s["trigger"]["missing"] == 1
    assert s["trigger"]["precision"] == 1
    assert s["trigger"]["recall"] == 1
    assert s["trigger"]["certification"] == "unverified"


def test_native_wrapped_observed_skill_read_and_safe_complete_projection(
        tmp_path):
    assert judging is not None
    a = session(tmp_path, "native.jsonl", [
        {
            "type": "response_item",
            "payload": {
                "type":
                "custom_tool_call",
                "name":
                "exec",
                "call_id":
                "outer",
                "input":
                'const r = await tools.exec_command({"cmd":"cat /skills/cudaq-algorithms/SKILL.md"}); text(r.output);'
            }
        },
        {
            "type": "response_item",
            "payload": {
                "type":
                "custom_tool_call_output",
                "call_id":
                "outer",
                "output":
                "---\nname: cudaq-algorithms\ndescription: Repository algorithms skill\n---\n# CUDA-Q Algorithms\nRead source before proposing changes."
            }
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{
                    "type": "output_text",
                    "text": "Final response."
                }]
            }
        },
    ])
    assert judging.usage([a], trace_complete=True)["activation"] is True
    projected = judging.transcript_evidence([a])
    assert projected["status"] == "complete"
    assert projected["evidence"]["final"] == ["Final response."]
    assert "Read source before proposing changes" in "\n".join(
        projected["evidence"]["tools"])


def test_projection_overflow_is_explicit_and_does_not_return_a_truncated_trace(
):
    assert judging is not None
    t = trajectory(observation="x" * 1000)
    result = judging.transcript_evidence(trajectory=t, max_bytes=100)
    assert result["status"] == "oversized"
    assert result["evidence"] is None
    assert result["observed_bytes"] > 100


def completion(g):
    return {
        "id":
        "completion-opaque",
        "object":
        "chat.completion",
        "created":
        123,
        "model":
        "nvidia/nemotron-3-super-120b-a12b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps(g)
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def test_private_grade_retries_transport_only_and_preserves_request_before_call(
        tmp_path):
    p = payload()
    out = tmp_path / "private-grade"
    calls = []

    def request(**kwargs):
        assert json.loads(
            (out / "request.json"
             ).read_text())["payload"]["attempt_key"] == "opaque-a7"
        calls.append(kwargs)
        if len(calls) == 1:
            raise urllib.error.HTTPError("https://example.invalid", 503,
                                         "secret-do-not-persist", {}, None)
        return completion(valid_grade(p))

    result = judging.grade(p,
                           output_dir=out,
                           request=request,
                           retry_delays=(0, 0),
                           checks={
                               "targeted": "pass",
                               "regression": "pass",
                               "capture": "complete"
                           })
    assert result["status"] == "graded"
    assert result["grade"]["adjusted_score"] == 10
    assert result["judge_usage"]["total_tokens"] is None
    assert result["judge_usage"]["observed_total_tokens"] == 150
    assert result["judge_usage"]["coverage"] == "partial"
    assert result["judge_attempts"] == 2
    assert result["judge_retries"] == 1
    assert result["transport"][0]["failure"] == "http_503"
    assert calls[0]["timeout"] == 300
    assert calls[0]["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert out.stat().st_mode & 0o077 == 0
    for file in out.iterdir():
        assert file.stat().st_mode & 0o077 == 0
        assert "secret-do-not-persist" not in file.read_text()
    with pytest.raises(FileExistsError):
        judging.grade(p, output_dir=out, request=request)


@pytest.mark.parametrize("mode,status,attempts",
                         [("429", "transport_error", 3),
                          ("400", "transport_error", 1),
                          ("invalid", "invalid_grade", 1),
                          ("missing_usage", "graded", 1)])
def test_transport_cap_semantic_no_retry_and_missing_judge_usage(
        tmp_path, mode, status, attempts):
    p = payload()
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if mode.isdigit():
            raise urllib.error.HTTPError("https://example.invalid", int(mode),
                                         "secret", {}, None)
        response = completion(valid_grade(p))
        if mode == "invalid":
            response["choices"][0]["message"]["content"] = '{"total":10}'
        if mode == "missing_usage": del response["usage"]
        return response

    result = judging.grade(p,
                           output_dir=tmp_path / mode,
                           request=request,
                           retry_delays=(0, 0))
    assert result["status"] == status
    assert result["judge_attempts"] == attempts
    assert len(calls) == attempts
    assert result["judge_time_s"] >= 0
    if mode == "missing_usage":
        assert result["judge_usage"]["total_tokens"] is None


def test_context_ceiling_prevents_any_request_without_truncating_payload(
        tmp_path):
    assert judging is not None
    p = judging.judge_payload("E03",
                              attempt_key="opaque-a7",
                              evidence={"final": ["x" * 65_000]})

    def request(**kwargs):
        pytest.fail("oversized prompt was sent")

    result = judging.grade(p,
                           output_dir=tmp_path / "overflow",
                           request=request)
    assert result["status"] == "evidence_oversized"
    assert result["judge_attempts"] == 0
    saved = json.loads((tmp_path / "overflow" / "request.json").read_text())
    assert len(saved["payload"]["evidence"][0]["lines"][0]) == 65_000


def test_grade_rejects_duplicate_json_keys_and_redacts_runtime_key(tmp_path):
    p = payload()
    key = "nvapi-fakeUnitTestCredential1234567890"

    def request(**kwargs):
        response = completion(valid_grade(p))
        response["choices"][0]["message"][
            "content"] = '{"total": 10, "total": 1, "reason": "' + key + '"}'
        return response

    result = judging.grade(p,
                           output_dir=tmp_path / "redaction",
                           api_key=key,
                           request=request)
    assert result["status"] == "invalid_grade"
    assert all(key not in file.read_text()
               for file in (tmp_path / "redaction").iterdir())


def test_valid_grade_also_cannot_serialize_runtime_key(tmp_path):
    p = payload()
    key = "nvapi-fakeUnitTestCredential1234567890"
    g = valid_grade(p)
    g["dimensions"][0]["reason"] = "Evidence contains " + key
    result = judging.grade(p,
                           output_dir=tmp_path / "valid-redaction",
                           api_key=key,
                           request=lambda **kwargs: completion(g))
    assert result["status"] == "graded"
    assert key not in json.dumps(result)
    assert all(key not in f.read_text()
               for f in (tmp_path / "valid-redaction").iterdir())


def test_timed_out_worker_latest_usage_is_a_lower_bound(tmp_path):
    a = session(tmp_path, "timeout.jsonl", [token_event(10, 1, 11, 10)])
    u = judging.usage(
        [a],
        result={"exception_info": {
            "exception_type": "AgentTimeoutError"
        }})
    assert u["total_tokens"] is None
    assert u["observed_total_tokens"] == 11
    assert u["usage_coverage_status"] == "partial"


def test_raw_sdk_adapter_pins_nvidia_and_disables_implicit_retries(
        monkeypatch):
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    sdk, request = judging._request_client("offline-adapter-characterization")
    try:
        assert str(sdk.base_url) == "https://integrate.api.nvidia.com/v1/"
        assert sdk.max_retries == 0
        assert sdk.timeout == 300
        assert callable(request)
    finally:
        sdk.close()


def test_cli_usage_writes_private_result_and_stdout_contains_only_status(
        tmp_path, capsys):
    a = session(tmp_path, "a.jsonl", [token_event(10, 1, 11, 10)])
    output = tmp_path / "usage.json"
    judging.main(["usage", "--session", str(a), "--output", str(output)])
    assert json.loads(output.read_text())["total_tokens"] == 11
    assert output.stat().st_mode & 0o077 == 0
    assert "input_tokens" not in capsys.readouterr().out


def test_shuffle_preserves_all_opaque_payloads_and_rejects_duplicate_ids():
    p = payload()
    other = copy.deepcopy(p)
    other["attempt_key"] = "opaque-b9"
    order = judging.grading_order([p, other], seed=1)
    assert [item["attempt_key"]
            for item in order] == ["opaque-b9", "opaque-a7"]
    with pytest.raises(ValueError):
        judging.grading_order([p, p])


def test_json_arm_labels_are_blinded_without_removing_numerical_baselines():
    p = judging.judge_payload(
        "E03",
        attempt_key="opaque-a7",
        evidence={
            "tools": [
                '{"arm":"candidate","energy_baseline":-1,"condition":"with_skill"}'
            ]
        })
    value = "\n".join(p["evidence"][0]["lines"])
    assert "candidate" not in value
    assert "with_skill" not in value
    assert "energy_baseline" in value


def test_malformed_native_usage_record_blocks_total_without_crashing(tmp_path):
    a = session(tmp_path, "invalid.jsonl", [
        token_event(10, 1, 11, 10), {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": "malformed"
            }
        }
    ])
    u = judging.usage([a])
    assert u["total_tokens"] is None
    assert u["observed_total_tokens"] == 11


def test_summary_exposes_quality_coverage_uplift_and_separate_judge_costs():
    rows = [row("E01", "baseline", score=6), row("E01", "candidate", score=8)]
    rows[0]["judge"] = {
        "judge_time_s": 2,
        "judge_attempts": 1,
        "judge_retries": 0,
        "judge_usage": {
            "total_tokens": 30,
            "observed_total_tokens": 30,
            "coverage": "complete"
        }
    }
    s = judging.summarize(rows)
    assert s["execution"]["uplift_raw_points"] == 2
    assert s["execution"]["baseline"]["graded_attempts"] == 1
    assert s["judge"]["observed_total_tokens"] == 30
    assert s["judge"]["missing_attempts"] == 1
    assert s["judge"]["total_tokens"] is None
    assert s["efficiency"]["total_tokens"]["baseline_median"] == 100


@pytest.mark.parametrize("missing_execution_judge", [False, True])
def test_judge_cost_coverage_counts_execution_attempts_only(
        missing_execution_judge):
    rows = [row("E01", arm) for arm in ("baseline", "candidate")]
    for item in rows:
        item["judge"] = {
            "judge_time_s": 2,
            "judge_attempts": 1,
            "judge_retries": 0,
            "judge_usage": {
                "total_tokens": 30,
                "observed_total_tokens": 30,
                "coverage": "complete"
            }
        }
    rows.extend({
        "case_id": case,
        "kind": "trigger",
        "arm": arm,
        "usage": {
            "activation": case == "T02"
        }
    } for case in ("T02", "N07") for arm in ("baseline", "candidate"))
    if missing_execution_judge:
        del rows[1]["judge"]
    costs = judging.summarize(rows)["judge"]
    assert costs["missing_attempts"] == (1 if missing_execution_judge else 0)
    assert costs["measured_attempts"] == (1 if missing_execution_judge else 2)
    assert costs["observed_total_tokens"] == (30 if missing_execution_judge
                                              else 60)
    assert costs["total_tokens"] == (None if missing_execution_judge else 60)
    assert costs["observed_time_s"] == (2 if missing_execution_judge else 4)


@pytest.mark.parametrize("later", [
    {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "A later answer."
            }]
        }
    },
    {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": "later",
            "name": "exec_command",
            "arguments": '{"cmd":"pwd"}'
        }
    },
    {
        "type": "response_item",
        "payload": {
            "type":
            "custom_tool_call",
            "call_id":
            "later",
            "name":
            "exec",
            "input":
            'const r = await tools.exec_command({"cmd":"pwd"}); text(r.output);'
        }
    },
    {
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "summary": [{
                "type": "summary_text",
                "text": "Later reasoning."
            }]
        }
    },
    {
        "type": "event_msg",
        "payload": {
            "type": "agent_message",
            "message": "A later answer."
        }
    },
])
def test_model_output_after_last_native_usage_leaves_totals_and_peak_partial(
        tmp_path, later):
    a = session(tmp_path, "later-model-output.jsonl", [
        token_event(10, 1, 11, 10), later, {
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "last_agent_message": "A later answer."
            }
        }
    ])
    u = judging.usage([a])
    assert u["total_tokens"] is None
    assert u["observed_total_tokens"] == 11
    assert u["usage_coverage_status"] == "partial"
    assert u["peak_request_input_tokens_proxy"] == 10
    assert u["peak_coverage_status"] == "partial"


def test_native_bookkeeping_after_usage_is_complete_and_later_usage_restores_coverage(
        tmp_path):
    later_answer = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "Done."
            }]
        }
    }
    done = {
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "last_agent_message": "Done."
        }
    }
    a = session(tmp_path, "bookkeeping.jsonl",
                [later_answer, token_event(10, 1, 11, 10), done])
    b = session(tmp_path, "restored.jsonl", [
        token_event(10, 1, 11, 10), later_answer,
        token_event(21, 2, 23, 11), done
    ])
    first, restored = judging.usage([a]), judging.usage([b])
    assert first["total_tokens"] == 11
    assert first["usage_coverage_status"] == "complete"
    assert first["peak_coverage_status"] == "complete"
    assert restored["total_tokens"] == 23
    assert restored["usage_coverage_status"] == "complete"
    assert restored["peak_coverage_status"] == "complete"


@pytest.mark.parametrize("structured_output", [False, True])
def test_structured_output_is_opt_in_on_actual_sdk_wire_request_and_private_record(
        tmp_path, structured_output):
    import httpx
    import jsonschema
    from openai import OpenAI
    p = payload()
    out = tmp_path / "wire"
    captured = []

    def transport(request):
        wire = json.loads(request.content)
        captured.append(wire)
        saved = json.loads((out / "request.json").read_text())
        assert saved["structured_output"] is structured_output
        if structured_output:
            format = wire["response_format"]
            assert format["type"] == "json_schema"
            assert format["json_schema"]["strict"] is True
            assert format["json_schema"]["name"] == "strategy_grade"
            jsonschema.Draft202012Validator.check_schema(
                format["json_schema"]["schema"])
            jsonschema.validate(valid_grade(p),
                                format["json_schema"]["schema"])
            assert saved["request"]["response_format"] == format
        else:
            assert "response_format" not in wire
            assert "response_format" not in saved["request"]
        return httpx.Response(200, json=completion(valid_grade(p)))

    with OpenAI(api_key="offline-wire-test",
                base_url="https://integrate.api.nvidia.com/v1",
                max_retries=0,
                http_client=httpx.Client(
                    transport=httpx.MockTransport(transport))) as sdk:
        result = judging.grade(p,
                               output_dir=out,
                               request=sdk.chat.completions.create,
                               structured_output=structured_output)
    assert result["status"] == "graded"
    assert len(captured) == 1


@pytest.mark.parametrize("fault", [
    "missing_dimension", "bad_dimension", "bad_score", "missing_critical",
    "bad_critical_id", "bad_verdict", "empty_reason", "wrong_attempt",
    "bad_total", "extra_property", "empty_channel_citation", "zero_start",
    "zero_end"
])
def test_response_schema_rejects_invalid_shapes_before_semantic_validation(
        fault):
    import jsonschema
    p = payload()
    p["evidence"].append({"id": "diff", "channel": "diff", "lines": []})
    g = valid_grade(p)
    if fault == "missing_dimension": g["dimensions"].pop()
    if fault == "bad_dimension": g["dimensions"][0]["dimension"] = "Invented"
    if fault == "bad_score": g["dimensions"][0]["score"] = 3
    if fault == "missing_critical": g["critical_checks"].pop()
    if fault == "bad_critical_id": g["critical_checks"][0]["id"] = "invented"
    if fault == "bad_verdict": g["critical_checks"][0]["verdict"] = "maybe"
    if fault == "empty_reason": g["dimensions"][0]["reason"] = ""
    if fault == "wrong_attempt": g["attempt_key"] = "wrong"
    if fault == "bad_total": g["total"] = 11
    if fault == "extra_property": g["dimensions"][0]["extra"] = "forbidden"
    if fault == "empty_channel_citation":
        g["dimensions"][0]["evidence"][0]["id"] = "diff"
    if fault == "zero_start": g["dimensions"][0]["evidence"][0]["start"] = 0
    if fault == "zero_end": g["dimensions"][0]["evidence"][0]["end"] = 0
    schema = judging.grade_response_format(p)["json_schema"]["schema"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(g, schema)


def test_response_schema_keeps_cross_field_checks_in_existing_validator():
    import jsonschema
    p = payload()
    g = valid_grade(p)
    g["total"] = 9
    schema = judging.grade_response_format(p)["json_schema"]["schema"]
    jsonschema.validate(g, schema)
    with pytest.raises(ValueError, match="total"):
        judging.validate_grade(g, p)


def test_structured_output_keeps_invalid_json_failure_without_retry(tmp_path):
    p = payload()
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        response = completion(valid_grade(p))
        response["choices"][0]["message"]["content"] = '{"total":10,"total":1}'
        return response

    result = judging.grade(p,
                           output_dir=tmp_path / "invalid-structured",
                           request=request,
                           structured_output=True)
    assert result["status"] == "invalid_grade"
    assert result["judge_attempts"] == 1
    assert len(calls) == 1


@pytest.mark.parametrize("input_bound,expected_status",
                         [(0, "graded"), (59904, "graded"),
                          (59905, "evidence_oversized")])
def test_token_bound_admits_large_requests_only_within_fixed_context(
        tmp_path, input_bound, expected_status):
    p = payload()
    p["evidence"][0]["lines"][0] = "x" * 65_000
    calls = []
    seen = []

    def counter(kwargs):
        seen.append(kwargs)
        assert len(kwargs["messages"][1]["content"]) > 58_000
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        assert kwargs["max_tokens"] == 4096
        return input_bound

    def request(**kwargs):
        calls.append(kwargs)
        assert kwargs == seen[0]
        return completion(valid_grade(p))

    out = tmp_path / "token-bound"
    result = judging.grade(p,
                           output_dir=out,
                           request=request,
                           structured_output=True,
                           token_counter=counter)
    assert result["status"] == expected_status
    assert len(calls) == (1 if expected_status == "graded" else 0)
    for artifact in (result, json.loads((out / "request.json").read_text())):
        assert artifact["budget_mode"] == "input_token_bound"
        assert artifact["input_token_bound"] == input_bound
    assert len(seen) == 1


@pytest.mark.parametrize("bad_bound", [
    None, True, -1, 59_904.0, "59904",
    RuntimeError("DO_NOT_SERIALIZE_COUNTER_ERROR")
])
def test_invalid_token_bound_fails_closed_without_request_or_error_text(
        tmp_path, bad_bound):
    p = payload()

    def counter(kwargs):
        if isinstance(bad_bound, Exception): raise bad_bound
        return bad_bound

    def request(**kwargs):
        pytest.fail("invalid counter allowed a model call")

    out = tmp_path / "invalid-counter"
    result = judging.grade(p,
                           output_dir=out,
                           request=request,
                           token_counter=counter)
    assert result["status"] == "invalid_token_bound"
    assert result["judge_attempts"] == 0
    assert result["input_token_bound"] is None
    assert all("DO_NOT_SERIALIZE_COUNTER_ERROR" not in path.read_text()
               for path in out.iterdir())


def test_counter_receives_redacted_request_and_cannot_modify_saved_or_sent_request(
        tmp_path):
    p = payload()
    key = "nvapi-counterRedaction1234567890"
    p["evidence"][0]["lines"][0] = key

    def counter(kwargs):
        assert key not in json.dumps(kwargs)
        kwargs["model"] = "forbidden-change"
        return 100

    def request(**kwargs):
        assert kwargs["model"] == "nvidia/nemotron-3-super-120b-a12b"
        return completion(valid_grade(p))

    out = tmp_path / "redacted-counter"
    result = judging.grade(p,
                           output_dir=out,
                           api_key=key,
                           request=request,
                           token_counter=counter)
    assert result["status"] == "graded"
    saved = json.loads((out / "request.json").read_text())
    assert saved["request"]["model"] == "nvidia/nemotron-3-super-120b-a12b"


def test_default_byte_guard_retains_explicit_budget_mode(tmp_path):
    p = payload()
    p["evidence"][0]["lines"][0] = "x" * 65_000

    def request(**kwargs):
        pytest.fail("default byte cap bypassed")

    out = tmp_path / "default-byte-guard"
    result = judging.grade(p, output_dir=out, request=request)
    assert result["status"] == "evidence_oversized"
    assert result["budget_mode"] == "utf8_bytes"
    assert result["input_token_bound"] is None
    assert json.loads(
        (out / "request.json").read_text())["budget_mode"] == "utf8_bytes"


@pytest.mark.parametrize("numbered", [False, True])
def test_numbered_evidence_only_changes_wire_copy_and_counter_sees_exact_presentation(
        tmp_path, numbered):
    p = payload()
    p["evidence"][0]["lines"] = [
        "First line with α.", "Existing 7: label.", ""
    ]
    original = copy.deepcopy(p)
    counted = []
    out = tmp_path / "numbered-wire"

    def counter(kwargs):
        counted.append(kwargs)
        return 1000

    def request(**kwargs):
        assert kwargs == counted[0]
        wire = json.loads(kwargs["messages"][1]["content"])
        for actual, source in zip(wire["evidence"], original["evidence"]):
            assert actual["id"] == source["id"]
            assert actual["channel"] == source["channel"]
            if numbered:
                assert actual["line_count"] == len(source["lines"])
                assert "lines" not in actual
                indexed = actual["numbered_text"].split(
                    "\n") if actual["line_count"] else []
                restored = []
                for start in range(0, len(source["lines"]), 8):
                    end = min(start + 8, len(source["lines"]))
                    assert indexed.pop(0) == f"[lines {start + 1}-{end}]"
                    restored.extend(indexed[:end - start])
                    del indexed[:end - start]
                assert not indexed
                assert restored == source["lines"]
            else:
                assert actual == source
        return completion(valid_grade(p))

    result = judging.grade(p,
                           output_dir=out,
                           request=request,
                           structured_output=True,
                           token_counter=counter,
                           numbered_evidence=numbered)
    assert result["status"] == "graded"
    assert p == original
    saved = json.loads((out / "request.json").read_text())
    assert saved["payload"] == original
    assert saved["evidence_presentation"] == (
        "numbered_blocks_8_v1" if numbered else "original_lines_v1")
    assert result["evidence_presentation"] == saved["evidence_presentation"]
    assert len(counted) == 1


@pytest.mark.parametrize("lines,expected", [
    ([], ""),
    (["1", "2", "3", "4", "5", "6", "7", "8"
      ], "[lines 1-8]\n1\n2\n3\n4\n5\n6\n7\n8"),
    (["1", "2", "3", "4", "5", "6", "7", "8", ""
      ], "[lines 1-8]\n1\n2\n3\n4\n5\n6\n7\n8\n[lines 9-9]\n"),
    ([
        "α", "", "[lines 9-16]", "4", "5", "6", "7", "8", "9", "10", "11",
        "12", "13", "14", "15", "16", "end"
    ],
     "[lines 1-8]\nα\n\n[lines 9-16]\n4\n5\n6\n7\n8\n[lines 9-16]\n9\n10\n11\n12\n13\n14\n15\n16\n[lines 17-17]\nend"
     ),
])
def test_numbered_blocks_preserve_empty_channels_and_original_citation_boundaries(
        tmp_path, lines, expected):
    import jsonschema
    p = payload()
    p["evidence"].append({"id": "tools", "channel": "tools", "lines": lines})
    out = tmp_path / "blocks"
    result = judging.grade(p,
                           output_dir=out,
                           request=lambda **kwargs: completion(valid_grade(p)),
                           structured_output=True,
                           numbered_evidence=True)
    assert result["status"] == "graded"
    saved = json.loads((out / "request.json").read_text())
    actual = json.loads(
        saved["request"]["messages"][1]["content"])["evidence"][-1]
    assert actual == {
        "id": "tools",
        "channel": "tools",
        "line_count": len(lines),
        "numbered_text": expected
    }
    remaining = actual["numbered_text"].split("\n") if lines else []
    restored = []
    while remaining:
        header = remaining.pop(0)
        first, last = map(
            int,
            header.removeprefix("[lines ").removesuffix("]").split("-"))
        assert first == len(restored) + 1
        restored.extend(remaining[:last - first + 1])
        del remaining[:last - first + 1]
    assert restored == lines
    schema = saved["request"]["response_format"]["json_schema"]["schema"]
    g = valid_grade(p)
    if lines:
        g["dimensions"][0]["evidence"] = [{
            "id": "tools",
            "start": len(lines),
            "end": len(lines)
        }]
        jsonschema.validate(g, schema)
        judging.validate_grade(g, p)
    g["dimensions"][0]["evidence"] = [{
        "id": "tools",
        "start": 1,
        "end": len(lines) + 1
    }]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(g, schema)


@pytest.mark.parametrize("channel,start,end",
                         [("final", 1, 3), ("tests", 1, 2), ("final", 3, 3),
                          ("tests", 2, 2)])
def test_response_schema_rejects_out_of_range_citations_per_channel(
        channel, start, end):
    import jsonschema
    p = payload()
    g = valid_grade(p)
    g["dimensions"][0]["evidence"] = [{
        "id": channel,
        "start": start,
        "end": end
    }]
    schema = judging.grade_response_format(p)["json_schema"]["schema"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(g, schema)


@pytest.mark.parametrize("verdict,has_evidence,accepted", [
    ("pass", False, False),
    ("fail", False, False),
    ("unknown", False, True),
    ("pass", True, True),
    ("fail", True, True),
    ("unknown", True, True),
])
def test_critical_schema_requires_evidence_for_known_verdicts(
        verdict, has_evidence, accepted):
    import jsonschema
    p = payload()
    g = valid_grade(p)
    g["critical_checks"][0]["verdict"] = verdict
    if not has_evidence:
        g["critical_checks"][0]["evidence"] = []
    schema = judging.grade_response_format(p)["json_schema"]["schema"]
    if accepted:
        jsonschema.validate(g, schema)
        judging.validate_grade(g, p)
    else:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(g, schema)
        with pytest.raises(ValueError, match="nonempty"):
            judging.validate_grade(g, p)


def test_channel_schema_accepts_last_line_but_validator_still_rejects_reversed_range(
):
    import jsonschema
    p = payload()
    g = valid_grade(p)
    schema = judging.grade_response_format(p)["json_schema"]["schema"]
    g["dimensions"][0]["evidence"] = [{
        "id": "final",
        "start": 2,
        "end": 2
    }, {
        "id": "tests",
        "start": 1,
        "end": 1
    }]
    jsonschema.validate(g, schema)
    judging.validate_grade(g, p)
    g["dimensions"][0]["evidence"] = [{"id": "final", "start": 2, "end": 1}]
    jsonschema.validate(g, schema)
    with pytest.raises(ValueError, match="range"):
        judging.validate_grade(g, p)


@pytest.mark.parametrize("line",
                         ["embedded\n2: misleading label", "embedded\rreturn"])
def test_numbering_rejects_multiline_entries_before_model_call(tmp_path, line):
    p = payload()
    p["evidence"][0]["lines"][0] = line

    def request(**kwargs):
        pytest.fail("ambiguous evidence sent to model")

    with pytest.raises(ValueError, match="single-line"):
        judging.grade(p,
                      output_dir=tmp_path / "multiline",
                      request=request,
                      numbered_evidence=True)
