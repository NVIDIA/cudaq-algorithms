# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blind grading contracts; no CUDA-Q or skill knowledge is embedded here."""
from __future__ import annotations

import hashlib
import json

VERDICTS = {"PASS", "FAIL", "UNCLEAR"}


def validate_grade(value: dict, rubric: list[dict],
                   evidence: dict[str, list[str]]) -> dict:
    problems = []
    rows = value.get("assertions") if isinstance(value, dict) else None
    rows = rows if isinstance(rows, list) else []
    expected = [item["id"] for item in rubric]
    actual = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(actual) != len(set(actual)):
        problems.append("duplicate assertion verdict")
    if set(actual) != set(expected) or len(actual) != len(expected):
        problems.append("missing or unknown assertion verdict")
    for row in rows:
        if not isinstance(row, dict):
            problems.append("verdict row is not an object")
            continue
        if row.get("verdict") not in VERDICTS:
            problems.append(f"unsupported verdict for {row.get('id')}")
        if not isinstance(row.get("reason"), str) or not row.get("reason",
                                                                 "").strip():
            problems.append(f"missing reason for {row.get('id')}")
        citations = row.get("evidence")
        if not isinstance(citations, list) or not citations:
            problems.append(f"missing evidence for {row.get('id')}")
            continue
        for citation in citations:
            source = citation.get("source") if isinstance(citation,
                                                          dict) else None
            line = citation.get("line") if isinstance(citation, dict) else None
            if source not in evidence or not isinstance(
                    line, int) or line < 1 or line > len(
                        evidence.get(source, [])):
                problems.append(
                    f"invalid evidence citation for {row.get('id')}")
    passed = not problems and all(row.get("verdict") == "PASS" for row in rows)
    return {
        "valid": not problems,
        "passed": passed,
        "problems": problems,
        "assertions": rows
    }


def blind_attempt(attempt: dict, *, salt: str) -> dict:
    identity = f"{salt}:{attempt.get('case')}:{attempt.get('repetition')}:{attempt.get('id')}"
    replacements = {
        str(attempt[key]): replacement
        for key, replacement in (("id", "<attempt>"), ("workspace",
                                                       "<workspace>"),
                                 ("skill_root",
                                  "<skill-root>"), ("snapshot", "<snapshot>"))
        if attempt.get(key)
    }

    def scrub(value):
        if isinstance(value, str):
            for original, replacement in replacements.items():
                value = value.replace(original, replacement)
            return value
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    allowed = {
        key: value
        for key, value in attempt.items()
        if key not in {"id", "arm", "workspace", "skill_root", "snapshot"}
    }
    allowed.pop("worker_telemetry", None)
    allowed.pop("grader_telemetry", None)
    allowed["attempt_key"] = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return scrub(allowed)


def paired_worker_deltas(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        groups.setdefault((row["case"], row["repetition"]),
                          {})[row["arm"]] = row
    result = []
    for (case, repetition), arms in sorted(groups.items()):
        if set(arms) != {"baseline", "candidate"}:
            continue
        baseline, candidate = arms["baseline"], arms["candidate"]
        left = (baseline.get("worker_telemetry")
                or {}).get("totals", {}).get("total_tokens")
        right = (candidate.get("worker_telemetry")
                 or {}).get("totals", {}).get("total_tokens")
        left_time = baseline.get("worker_elapsed_s")
        right_time = candidate.get("worker_elapsed_s")
        result.append({
            "case":
            case,
            "repetition":
            repetition,
            "elapsed_s":
            right_time - left_time
            if left_time is not None and right_time is not None else None,
            "total_tokens":
            right - left if left is not None and right is not None else None
        })
    return result


def grader_schema(rubric: list[dict]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assertions"],
        "properties": {
            "assertions": {
                "type": "array",
                "minItems": len(rubric),
                "maxItems": len(rubric),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "verdict", "reason", "evidence"],
                    "properties": {
                        "id": {
                            "type": "string"
                        },
                        "verdict": {
                            "enum": sorted(VERDICTS)
                        },
                        "reason": {
                            "type": "string"
                        },
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["source", "line"],
                                "properties": {
                                    "source": {
                                        "type": "string"
                                    },
                                    "line": {
                                        "type": "integer",
                                        "minimum": 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


__all__ = [
    "blind_attempt", "grader_schema", "paired_worker_deltas", "validate_grade"
]
