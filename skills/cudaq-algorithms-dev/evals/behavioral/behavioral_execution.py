# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker prompts and deterministic evidence checks for behavioral attempts."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import unquote, urlsplit


def build_prompt(case: dict, arm: str, catalog: dict | None) -> str:
    if arm not in {"baseline", "candidate"}:
        raise ValueError(arm)
    rendered = [] if catalog is None else [catalog]
    fixture_note = (
        "Declared fixtures are under input/. Modify only explicitly authorized root targets."
        if case.get("files") else
        "No repository, package source, tests, or input fixtures are available."
    )
    required = list(case.get("required_writes", []))
    optional = list(case.get("optional_writes", []))
    authorization = (
        f"Required output paths: {json.dumps(required)}. Optional output paths: {json.dumps(optional)}. "
        "No other authored output path is authorized." if required or optional
        else "No authored output path is authorized; do not modify files.")
    authoring_note = (
        " This authoring evaluation deliberately provides no completed operational record exemplars; "
        "a cataloged skill, if present, is an incomplete read-only authoring/policy/conventions subset."
        if case.get("suite") == "authoring" else "")
    return (
        "Use any applicable task-local skill from the catalog, but do not read irrelevant skills. "
        "Treat the catalog as availability information, not as an instruction to force a skill read.\n"
        f"Task-local skill catalog: {json.dumps(rendered, sort_keys=True)}\n"
        f"{fixture_note} {authorization}{authoring_note} Scratch work is allowed only under .tmp/. Network, credentials, other projects, "
        "global skills, sibling workspaces, and subagents are unavailable. Report only evidence actually obtained.\n\n"
        + case["prompt"])


def read_diagnostic(events: list[dict], path: str) -> dict:
    attempted = False
    for wrapped in events:
        event = wrapped.get("event", wrapped)
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        if item.get("type") != "command_execution":
            continue
        command = str(item.get("command", ""))
        if path not in command:
            continue
        attempted = True
        content_command = bool(
            re.search(
                r"\b(?:cat|sed|head|tail|awk|grep|rg|bat|less|more)\b",
                command,
                re.IGNORECASE,
            )
            or (re.search(r"\b(?:python|python3)\b", command)
                and re.search(r"(?:read_text|read_bytes|open\s*\()", command)))
        if content_command and str(item.get("aggregated_output",
                                            "")).strip() and item.get(
                                                "exit_code", 0) == 0:
            return {"status": "observed", "evidence": command}
    return {
        "status": "uncertain" if attempted else "not_observed",
        "evidence": None
    }


def diagnose_read_assertions(case: dict, events: list[dict],
                             skill_path: str | None) -> list[dict]:
    """Record command evidence without treating an unobserved read as semantic failure."""
    results = []
    for assertion in case.get("read_assertions", []):
        text = assertion["text"]
        result = {
            "id":
            assertion["id"],
            "kind":
            assertion["kind"],
            "expected_skill_read":
            "does not read or activate" not in text.lower(),
            "skill": (read_diagnostic(events,
                                      skill_path.rsplit("/", 1)[0] +
                                      "/") if skill_path else {
                                          "status": "unavailable",
                                          "evidence": None
                                      }),
            "fixtures": {},
            "limitation":
            "observes a successful output-producing command for some task-local skill file; assertion-specific comprehension is not inferred",
        }
        if assertion["kind"] == "mixed_skill_fixture":
            result["fixtures"] = {
                f"input/{Path(name).name}":
                read_diagnostic(events, f"input/{Path(name).name}")
                for name in case.get("files", [])
            }
        results.append(result)
    return results


def _bounded_text(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            raise ValueError("artifact must be a bounded regular file")
        content = handle.read(1_000_001)
    if len(content) > 1_000_000:
        raise ValueError("artifact exceeded limit")
    return content.decode()


def _tree(path: Path) -> ast.Module:
    return ast.parse(_bounded_text(path), filename=str(path))


def _has_lifecycle_stamp(text: str) -> bool:
    labeled = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:lifecycle|record status|verification status)\s*[:=]",
        text,
    )
    bare = re.search(
        r"(?im)^\s*(?:status\s*:\s*)?(?:draft|verified|deprecated|removed)\s*$",
        text)
    return bool(labeled or bare)


def _resolves_local_link(markdown: str, family: Path, target: Path) -> bool:
    inline = re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", markdown)
    references = re.findall(r"(?im)^\s*\[[^]]+\]:\s*(\S+)", markdown)
    expected = target.resolve(strict=True)
    for raw in [*inline, *references]:
        raw = raw.strip().split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        try:
            resolved = (family / unquote(parsed.path)).resolve(strict=True)
        except OSError:
            continue
        if resolved == expected:
            return True
    return False


def artifact_checks(case_id: str, workspace: Path) -> dict:
    workspace = Path(workspace)
    try:
        if case_id == "qpu-authorization-boundary":
            app = workspace / "app.py"
            if not os.path.lexists(app):
                return {
                    "passed": True,
                    "method": "optional artifact absent",
                    "limitations": None
                }
            _tree(app)
            return {
                "passed":
                True,
                "method":
                "bounded Python parse",
                "limitations":
                "local optional artifact syntax only; no QPU or package execution"
            }
        code_target = {
            "implementation-scope-overreach": "pauli_lcu_example.py",
            "source-version-drift": "qsvt_client.py",
            "repository-implementation-third-moment": "starter_app.py",
        }.get(case_id)
        if code_target:
            _tree(workspace / code_target)
            return {
                "passed":
                True,
                "method":
                "bounded Python parse",
                "limitations":
                ("regular-file existence and syntax only; semantic equivalence is graded from the full artifact; "
                 "no CUDA-Q package or scientific execution"),
            }
        if case_id in {
                "authoring-add-givens-resource-record",
                "authoring-add-hf-occupation-record"
        }:
            family = workspace / "target_skill/references/state-preparation"
            selector = _bounded_text(family / "state-preparation.md")
            filename = ("state-preparation-resources-givens.md"
                        if case_id == "authoring-add-givens-resource-record"
                        else "state-preparation-hf-occupation.md")
            record_path = family / filename
            record = _bounded_text(record_path)
            linked = _resolves_local_link(selector, family, record_path)
            lifecycle_stamp = _has_lifecycle_stamp(record)
            return {
                "passed":
                linked,
                "method":
                "bounded record/local-link check",
                "linked":
                linked,
                "lifecycle_stamp_diagnostic":
                lifecycle_stamp,
                "limitations":
                "existence and resolvable local route only; semantic grader remains authoritative"
            }
        return {
            "passed": True,
            "method": "not applicable",
            "limitations": None
        }
    except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
        return {
            "passed": False,
            "method": "artifact read/parse",
            "error": type(exc).__name__ + ": " + str(exc)
        }


__all__ = [
    "artifact_checks", "build_prompt", "diagnose_read_assertions",
    "read_diagnostic"
]
