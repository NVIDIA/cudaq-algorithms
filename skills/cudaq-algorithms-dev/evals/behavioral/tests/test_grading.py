# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from behavioral_grading import (
    blind_attempt,
    paired_worker_deltas,
    validate_grade,
)


def evidence():
    return {
        "final": ["first", "second"],
        "commands": ["read SKILL.md"],
        "artifacts": ["ok"]
    }


def test_grade_requires_exact_unique_assertions_and_valid_line_citations():
    rubric = [{"id": "a1", "text": "one"}, {"id": "a2", "text": "two"}]
    valid = {
        "assertions": [
            {
                "id": "a1",
                "verdict": "PASS",
                "reason": "shown",
                "evidence": [{
                    "source": "final",
                    "line": 1
                }]
            },
            {
                "id": "a2",
                "verdict": "FAIL",
                "reason": "missing",
                "evidence": [{
                    "source": "final",
                    "line": 2
                }]
            },
        ]
    }
    checked = validate_grade(valid, rubric, evidence())
    assert checked["valid"]
    assert not checked["passed"]

    duplicate = {
        "assertions": [valid["assertions"][0], valid["assertions"][0]]
    }
    assert not validate_grade(duplicate, rubric, evidence())["valid"]
    missing = {"assertions": [valid["assertions"][0]]}
    assert not validate_grade(missing, rubric, evidence())["valid"]
    bad_line = {
        "assertions": [
            valid["assertions"][0], {
                **valid["assertions"][1], "evidence": [{
                    "source": "final",
                    "line": 3
                }]
            }
        ]
    }
    assert not validate_grade(bad_line, rubric, evidence())["valid"]
    unclear = {
        "assertions": [
            valid["assertions"][0], {
                **valid["assertions"][1], "verdict": "UNCLEAR"
            }
        ]
    }
    assert validate_grade(unclear, rubric, evidence())["valid"]
    assert not validate_grade(unclear, rubric, evidence())["passed"]


def test_blinded_attempt_removes_arm_and_staging_paths():
    payload = blind_attempt(
        {
            "id": "case--1--candidate",
            "arm": "candidate",
            "workspace": "/tmp/private/candidate",
            "case": "case",
            "repetition": 1,
            "final": {
                "detail": "error at /tmp/private/candidate/case--1--candidate"
            }
        },
        salt="fixed")
    rendered = str(payload)
    assert "candidate" not in rendered
    assert "/tmp/private" not in rendered
    assert payload["case"] == "case"
    assert payload["attempt_key"]


def test_grader_usage_is_not_added_to_worker_pairing():
    rows = [
        {
            "case": "x",
            "repetition": 1,
            "arm": "baseline",
            "worker_elapsed_s": 4,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 10
                }
            },
            "grader_elapsed_s": 100,
            "grader_telemetry": {
                "totals": {
                    "total_tokens": 1000
                }
            }
        },
        {
            "case": "x",
            "repetition": 1,
            "arm": "candidate",
            "worker_elapsed_s": 7,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 15
                }
            },
            "grader_elapsed_s": 200,
            "grader_telemetry": {
                "totals": {
                    "total_tokens": 2000
                }
            }
        },
    ]
    delta = paired_worker_deltas(rows)
    assert delta == [{
        "case": "x",
        "repetition": 1,
        "elapsed_s": 3,
        "total_tokens": 5
    }]


def test_paired_worker_failures_remain_with_missing_resource_values():
    rows = [
        {
            "case": "x",
            "repetition": 1,
            "arm": "baseline",
            "worker_elapsed_s": None,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": None
                }
            }
        },
        {
            "case": "x",
            "repetition": 1,
            "arm": "candidate",
            "worker_elapsed_s": 7,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 15
                }
            }
        },
    ]
    assert paired_worker_deltas(rows) == [{
        "case": "x",
        "repetition": 1,
        "elapsed_s": None,
        "total_tokens": None
    }]
    rows[0]["worker_telemetry"] = None
    assert paired_worker_deltas(rows)[0]["total_tokens"] is None
