# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

STRATEGY = Path(__file__).resolve().parents[1]
SUITE_PATH = STRATEGY / "suite.py"
if SUITE_PATH.is_file():
    SPEC = importlib.util.spec_from_file_location("strategy_suite", SUITE_PATH)
    suite = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    sys.modules[SPEC.name] = suite
    SPEC.loader.exec_module(suite)
else:
    suite = None


def test_load_cases_parses_the_disclosed_brief_without_inventing_holdouts():
    assert suite is not None, "suite parser has not been implemented"
    cases = suite.load_cases()

    assert len(cases) == 34
    assert [case.case_id for case in cases[:2]] == ["T01", "T02"]
    assert [case.case_id for case in cases[-2:]] == ["E11", "E12"]
    assert sum(case.kind == "trigger" for case in cases) == 22
    assert sum(case.kind == "execution" for case in cases) == 12
    assert sum(case.expected_activation is True for case in cases) == 12
    assert sum(case.expected_activation is False for case in cases) == 10
    assert all(not case.holdout for case in cases)
    assert suite.case_by_id("T01", cases).prompt == (
        "Add a `sum` combinator for two CUDA-Q Algorithms block encodings.")
    assert suite.case_by_id("E02", cases).prompt == (
        "Implement a `scale` operation for block encodings. Support positive, "
        "negative, zero, and invalid scaling factors. Preserve downstream "
        "compatibility with Walk and QSVT.")
    assert "No division-by-zero path for zero scaling." in suite.case_by_id(
        "E02", cases).critical_checks


def test_final_execution_case_stops_before_development_diagnostics():
    expected = suite.case_by_id("E12").expected_behavior

    assert len(expected) == 7
    assert expected[
        -1] == "Avoids chemistry-specific data structures in the generic interface."


def test_schedule_runs_baseline_first_and_repeats_every_mutation_request_three_times(
):
    schedule = suite.build_schedule(suite.load_cases())

    assert len(schedule) == 100
    assert all(
        schedule[index].arm == "baseline" and schedule[index + 1].arm ==
        "candidate" and schedule[index].case_id == schedule[index + 1].case_id
        and schedule[index].repetition == schedule[index + 1].repetition
        for index in range(0, len(schedule), 2))
    expected_mutations = {
        "E02", "E03", "E04", "E05", "E07", "E08", "E10", "E12"
    }
    execution = [
        case for case in suite.load_cases() if case.kind == "execution"
    ]
    assert {case.case_id
            for case in execution
            if case.mutation_request} == expected_mutations
    for case in execution:
        repetitions = {
            attempt.repetition
            for attempt in schedule if attempt.case_id == case.case_id
        }
        assert repetitions == (set(range(1, 4))
                               if case.mutation_request else {1})
    assert suite.case_by_id("E07").conditional_outcome is True


def test_worker_payload_exposes_only_the_prompt_and_a_separate_execution_treatment(
):
    baseline = suite.worker_payload("E02", "baseline")
    candidate = suite.worker_payload("E02", "candidate")
    trigger = suite.worker_payload("T01", "candidate")

    assert baseline == {
        "prompt":
        "Implement a `scale` operation for block encodings. Support positive, "
        "negative, zero, and invalid scaling factors. Preserve downstream "
        "compatibility with Walk and QSVT."
    }
    assert candidate["prompt"] == baseline["prompt"]
    assert candidate["treatment_directive"] == (
        "Explicitly use the available cudaq-algorithms skill for this task.")
    assert trigger == {
        "prompt":
        "Add a `sum` combinator for two CUDA-Q Algorithms block encodings."
    }
    for payload in (baseline, candidate, trigger):
        assert not ({"case_id", "id", "arm", "expected", "rubric", "critical"}
                    & payload.keys())


def test_private_grader_receives_failure_and_critical_criteria_hidden_from_worker(
):
    payload = suite.grader_payload(
        "E10",
        attempt_key="opaque-attempt-017",
        evidence={
            "final": ["Rewrote the unsupported claim."],
            "tests": ["2 passed"]
        },
    )

    serialized = json.dumps(payload)
    assert payload["attempt_key"] == "opaque-attempt-017"
    assert payload["task"].startswith("Update the documentation")
    assert len(payload["score_rubric"]) == 5
    assert sum(row["max_points"] for row in payload["score_rubric"]) == 10
    assert payload["expected_behavior"]
    assert payload["critical_checks"] == [
        "Publishing the requested claim unchanged."
    ]
    assert "Inventing an API or current capability" in payload[
        "global_critical_failures"][0]
    assert payload["failure_signals"] == []
    assert "E10" not in serialized
    assert "baseline" not in serialized
    assert "candidate" not in serialized
    assert "Completing an implementation case with failing targeted tests" in serialized

    worker = json.dumps(suite.worker_payload("E10", "candidate"))
    assert "Publishing the requested claim unchanged" not in worker
    assert "Completing an implementation case with failing targeted tests" not in worker

    failure_payload = suite.grader_payload("E03",
                                           attempt_key="opaque-attempt-018",
                                           evidence={"tools": []})
    assert failure_payload["failure_signals"][0] == (
        "Hard-codes a second custom encoding type.")


def test_grader_payload_accepts_only_bounded_named_evidence_channels():
    with pytest.raises(ValueError, match="evidence channel"):
        suite.grader_payload("E02",
                             attempt_key="opaque-1",
                             evidence={"arm": ["baseline"]})
    with pytest.raises(ValueError, match="evidence channel"):
        suite.grader_payload("E02",
                             attempt_key="opaque-1",
                             evidence={"/tmp/trace": ["content"]})
    with pytest.raises(ValueError, match="evidence exceeds"):
        suite.grader_payload("E02",
                             attempt_key="opaque-1",
                             evidence={"final": ["x" * 1_000_001]})


@pytest.mark.parametrize(
    "attempt_key", ["baseline-1", "candidate-1", "/tmp/run-1", "runs\\run-1"])
def test_grader_payload_rejects_arm_labels_and_paths_as_attempt_keys(
        attempt_key):
    with pytest.raises(ValueError, match="opaque"):
        suite.grader_payload("E02", attempt_key=attempt_key, evidence={})


def test_summary_reports_disclosed_development_cases_and_no_holdout_claim():
    summary = suite.summary(suite.load_cases())

    assert summary == {
        "developer_cases": 34,
        "holdout_cases": 0,
        "trigger_cases": {
            "positive": 12,
            "negative": 10,
            "total": 22
        },
        "execution_cases": {
            "mutation_requests": 8,
            "other": 4,
            "total": 12
        },
        "planned_attempts": 100,
        "ordering": "baseline-first within every case/repetition pair",
    }


def test_rejects_duplicate_and_malformed_brief_contracts():
    brief = (STRATEGY / "BRIEF.md").read_text()
    duplicate = brief.replace("| T02 |", "| T01 |", 1)
    missing_case = brief.replace("### E12: Beyond-chemistry extension",
                                 "### X12: Missing case")

    with pytest.raises(suite.BriefFormatError, match="duplicate case ID"):
        suite.parse_brief(duplicate)
    with pytest.raises(suite.BriefFormatError, match="execution IDs"):
        suite.parse_brief(missing_case)


def test_rejects_unknown_ids_invalid_arms_and_nonexecution_grading():
    with pytest.raises(KeyError, match="unknown case ID"):
        suite.worker_payload("E99", "baseline")
    with pytest.raises(ValueError, match="arm"):
        suite.worker_payload("E01", "mystery")
    with pytest.raises(ValueError, match="execution"):
        suite.grader_payload("T01", attempt_key="opaque", evidence={})
