#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse the disclosed strategy brief into private evaluation contracts.

This module prepares controller metadata only.  It does not run agents, author
holdouts, or expose expected behavior and grading criteria to workers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
BRIEF = HERE / "BRIEF.md"

TRIGGER_IDS = tuple([f"T{index:02d}" for index in range(1, 13)] +
                    [f"N{index:02d}" for index in range(1, 11)])
EXECUTION_IDS = tuple(f"E{index:02d}" for index in range(1, 13))
MUTATION_REQUEST_IDS = frozenset(
    {"E02", "E03", "E04", "E05", "E07", "E08", "E10", "E12"})
CONDITIONAL_OUTCOME_IDS = frozenset({"E07"})
EXECUTION_TREATMENT = (
    "Explicitly use the available cudaq-algorithms skill for this task.")
EVIDENCE_CHANNELS = frozenset({"final", "tests", "tools", "diff"})
MAX_EVIDENCE_UTF8_BYTES = 1_000_000


class BriefFormatError(ValueError):
    """The checked-in strategy brief does not match its frozen case schema."""


@dataclass(frozen=True)
class ScoreDimension:
    name: str
    max_points: int
    full_credit: str


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    prompt: str
    expected_activation: bool | None
    expected_routing: str | None
    expected_behavior: tuple[str, ...]
    failure_signals: tuple[str, ...]
    critical_checks: tuple[str, ...]
    global_critical_failures: tuple[str, ...]
    score_rubric: tuple[ScoreDimension, ...]
    mutation_request: bool
    conditional_outcome: bool
    holdout: bool = False


@dataclass(frozen=True)
class Attempt:
    case_id: str
    kind: str
    repetition: int
    arm: str


def _between(text: str, start: str, end: str | None) -> str:
    try:
        tail = text.split(start, 1)[1]
    except IndexError as exc:
        raise BriefFormatError(f"missing section: {start}") from exc
    if end is None:
        return tail
    if end not in tail:
        raise BriefFormatError(f"missing section boundary: {end}")
    return tail.split(end, 1)[0]


def _table_rows(block: str, prefix: str,
                columns: int) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for line in block.splitlines():
        if not re.match(rf"^\|\s*{re.escape(prefix)}\d{{2}}\s*\|", line):
            continue
        values = tuple(value.strip()
                       for value in line.strip().strip("|").split("|"))
        if len(values) != columns:
            raise BriefFormatError(f"malformed {prefix} trigger row: {line}")
        rows.append(values)
    return rows


def _list_under(section: str, heading: str) -> tuple[str, ...]:
    marker = f"**{heading}**"
    if marker not in section:
        return ()
    tail = section.split(marker, 1)[1]
    tail = re.split(r"^\*\*[^*]+\*\*\s*$|^#{2,}\s+",
                    tail,
                    maxsplit=1,
                    flags=re.MULTILINE)[0]
    values = []
    for raw in tail.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^-\s+", "", line)
        values.append(line)
    return tuple(values)


def _prompt_under(section: str) -> str:
    marker = "**Prompt**"
    if marker not in section:
        raise BriefFormatError("execution case has no Prompt block")
    tail = section.split(marker, 1)[1]
    lines: list[str] = []
    started = False
    for raw in tail.splitlines():
        if raw.startswith(">"):
            started = True
            lines.append(raw[1:].lstrip())
        elif started:
            break
    prompt = "\n".join(lines).strip()
    if not prompt:
        raise BriefFormatError("execution Prompt block is empty")
    return prompt


def _score_rubric(text: str) -> tuple[ScoreDimension, ...]:
    block = _between(text, "## Execution rubric", "### Critical failures")
    rows: list[ScoreDimension] = []
    for line in block.splitlines():
        if not line.startswith("|") or line.startswith(
                "|---") or "Dimension" in line:
            continue
        values = [
            value.strip() for value in line.strip().strip("|").split("|")
        ]
        if len(values) != 3:
            continue
        match = re.fullmatch(r"0 to (\d+)", values[1])
        if match:
            rows.append(
                ScoreDimension(values[0], int(match.group(1)), values[2]))
    if len(rows) != 5 or sum(row.max_points for row in rows) != 10:
        raise BriefFormatError(
            "execution score rubric must contain five dimensions totaling 10")
    return tuple(rows)


def _global_critical_failures(text: str) -> tuple[str, ...]:
    block = _between(text, "### Critical failures", "## Execution cases")
    failures = tuple(line.strip()[2:].strip() for line in block.splitlines()
                     if line.strip().startswith("- "))
    if not failures:
        raise BriefFormatError("global critical failure list is empty")
    return failures


def _ensure_ids(actual: Iterable[str], expected: tuple[str, ...],
                label: str) -> None:
    values = list(actual)
    if len(values) != len(set(values)):
        duplicate = next(value for index, value in enumerate(values)
                         if value in values[:index])
        raise BriefFormatError(f"duplicate case ID: {duplicate}")
    if tuple(values) != expected:
        raise BriefFormatError(
            f"{label} IDs must be {', '.join(expected)}; got {', '.join(values)}"
        )


def parse_brief(text: str) -> list[Case]:
    """Parse the exact disclosed cases and private grading material."""
    positive_block = _between(
        text,
        "### Prompts that should trigger the skill",
        "### Prompts that should not trigger the skill",
    )
    negative_block = _between(
        text,
        "### Prompts that should not trigger the skill",
        "## Execution rubric",
    )
    positive = _table_rows(positive_block, "T", 2)
    negative = _table_rows(negative_block, "N", 3)
    trigger_rows = [*positive, *negative]
    _ensure_ids((row[0] for row in trigger_rows), TRIGGER_IDS, "trigger")

    cases = [
        Case(
            case_id=row[0],
            kind="trigger",
            prompt=row[1],
            expected_activation=row[0].startswith("T"),
            expected_routing=row[2] if len(row) == 3 else None,
            expected_behavior=(),
            failure_signals=(),
            critical_checks=(),
            global_critical_failures=(),
            score_rubric=(),
            mutation_request=False,
            conditional_outcome=False,
        ) for row in trigger_rows
    ]

    rubric = _score_rubric(text)
    global_critical = _global_critical_failures(text)
    matches = list(re.finditer(r"^### (E\d{2}): .+$", text, re.MULTILINE))
    execution_ids = [match.group(1) for match in matches]
    _ensure_ids(execution_ids, EXECUTION_IDS, "execution")
    for index, match in enumerate(matches):
        end = matches[index +
                      1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end():end]
        case_id = match.group(1)
        critical = (*_list_under(section, "Critical checks"),
                    *_list_under(section, "Critical failure"))
        cases.append(
            Case(
                case_id=case_id,
                kind="execution",
                prompt=_prompt_under(section),
                expected_activation=None,
                expected_routing=None,
                expected_behavior=_list_under(section, "Expected behavior"),
                failure_signals=_list_under(section, "Failure signals"),
                critical_checks=tuple(critical),
                global_critical_failures=global_critical,
                score_rubric=rubric,
                mutation_request=case_id in MUTATION_REQUEST_IDS,
                conditional_outcome=case_id in CONDITIONAL_OUTCOME_IDS,
            ))

    for case in cases:
        if not case.prompt:
            raise BriefFormatError(f"empty prompt: {case.case_id}")
        if case.kind == "execution" and not case.expected_behavior:
            raise BriefFormatError(
                f"missing expected behavior: {case.case_id}")
    return cases


def load_cases(path: Path = BRIEF) -> list[Case]:
    return parse_brief(Path(path).read_text(encoding="utf-8"))


def case_by_id(case_id: str, cases: Iterable[Case] | None = None) -> Case:
    for case in load_cases() if cases is None else cases:
        if case.case_id == case_id:
            return case
    raise KeyError(f"unknown case ID: {case_id}")


def build_schedule(cases: Iterable[Case]) -> list[Attempt]:
    """Build the brief-mandated baseline-first, no-stop attempt schedule."""
    schedule: list[Attempt] = []
    for case in cases:
        repetitions = 3 if case.mutation_request else 1
        for repetition in range(1, repetitions + 1):
            for arm in ("baseline", "candidate"):
                schedule.append(
                    Attempt(case.case_id, case.kind, repetition, arm))
    return schedule


def worker_payload(case_id: str,
                   arm: str,
                   cases: Iterable[Case] | None = None) -> dict[str, str]:
    """Return only worker-visible task text, never private evaluation metadata."""
    if arm not in {"baseline", "candidate"}:
        raise ValueError("arm must be baseline or candidate")
    case = case_by_id(case_id, cases)
    payload = {"prompt": case.prompt}
    if arm == "candidate" and case.kind == "execution":
        payload["treatment_directive"] = EXECUTION_TREATMENT
    return payload


def _validate_evidence(
        evidence: Mapping[str, list[str]]) -> dict[str, list[str]]:
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    result: dict[str, list[str]] = {}
    total_bytes = 0
    for source, lines in evidence.items():
        if not isinstance(source, str) or not isinstance(
                lines, list) or not all(
                    isinstance(line, str) for line in lines):
            raise TypeError(
                "evidence must map string sources to lists of strings")
        if source not in EVIDENCE_CHANNELS:
            raise ValueError(f"unsupported evidence channel: {source!r}")
        total_bytes += len(source.encode("utf-8"))
        total_bytes += sum(len(line.encode("utf-8")) + 1 for line in lines)
        if total_bytes > MAX_EVIDENCE_UTF8_BYTES:
            raise ValueError(
                f"evidence exceeds {MAX_EVIDENCE_UTF8_BYTES}-byte UTF-8 budget"
            )
        result[source] = list(lines)
    return result


def grader_payload(
    case_id: str,
    *,
    attempt_key: str,
    evidence: Mapping[str, list[str]],
    cases: Iterable[Case] | None = None,
) -> dict[str, Any]:
    """Omit controller mappings; traces may still make treatment inferable."""
    case = case_by_id(case_id, cases)
    if case.kind != "execution":
        raise ValueError("grader payload is defined only for execution cases")
    if (not isinstance(attempt_key, str) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_key)
            or "baseline" in attempt_key.casefold()
            or "candidate" in attempt_key.casefold()):
        raise ValueError(
            "attempt_key must be opaque and contain no arm label or path")
    return {
        "attempt_key":
        attempt_key,
        "task":
        case.prompt,
        "score_rubric": [{
            "dimension": row.name,
            "max_points": row.max_points,
            "full_credit": row.full_credit,
        } for row in case.score_rubric],
        "expected_behavior":
        list(case.expected_behavior),
        "failure_signals":
        list(case.failure_signals),
        "critical_checks":
        list(case.critical_checks),
        "global_critical_failures":
        list(case.global_critical_failures),
        "evidence":
        _validate_evidence(evidence),
    }


def summary(cases: Iterable[Case]) -> dict[str, Any]:
    selected = list(cases)
    triggers = [case for case in selected if case.kind == "trigger"]
    execution = [case for case in selected if case.kind == "execution"]
    return {
        "developer_cases":
        len(selected) - sum(case.holdout for case in selected),
        "holdout_cases": sum(case.holdout for case in selected),
        "trigger_cases": {
            "positive":
            sum(case.expected_activation is True for case in triggers),
            "negative":
            sum(case.expected_activation is False for case in triggers),
            "total": len(triggers),
        },
        "execution_cases": {
            "mutation_requests":
            sum(case.mutation_request for case in execution),
            "other": sum(not case.mutation_request for case in execution),
            "total": len(execution),
        },
        "planned_attempts": len(build_schedule(selected)),
        "ordering": "baseline-first within every case/repetition pair",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--summary", action="store_true")
    actions.add_argument("--worker-payload",
                         nargs=2,
                         metavar=("CASE_ID", "ARM"))
    parser.add_argument("--brief", type=Path, default=BRIEF)
    args = parser.parse_args(argv)
    cases = load_cases(args.brief)
    value = summary(cases) if args.summary else worker_payload(
        *args.worker_payload, cases=cases)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
