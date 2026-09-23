# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts and fair staging for the native behavioral suite."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import stat

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent

_TARGETS = {
    "repository-implementation-third-moment": {
        "files/starter_app.py": "starter_app.py"
    },
    "source-version-drift": {
        "files/qsvt_client.py": "qsvt_client.py"
    },
    "implementation-scope-overreach": {
        "files/pauli_lcu_example.py": "pauli_lcu_example.py"
    },
}
_OPTIONAL = {"qpu-authorization-boundary": {"app.py"}}
_ASSERTION_SPLITS = {
    ("material-ambiguity-clarification", 1): {
        "diagnostic":
        "The agent reads or activates the applicable cataloged skill.",
        "shared":
        "The agent identifies both material ambiguities in representation and evolution family.",
        "kind": "skill_read",
    },
    ("source-unavailable-fallback", 1): {
        "diagnostic":
        "The agent checks the available skill records for a populated sparse-oracle contract.",
        "shared":
        "The agent treats the sparse-oracle contract as unavailable from the provided evidence.",
        "kind": "skill_read",
    },
    ("source-version-drift", 1): {
        "diagnostic":
        "The agent reads the applicable source-drift skill rule.",
        "shared":
        "The agent inspects the staged recorded contract, current source, focused test, and client.",
        "kind": "mixed_skill_fixture",
    },
    ("implementation-scope-overreach", 1): {
        "diagnostic": "The agent reads the applicable PauliLCU skill record.",
        "shared":
        "The agent inspects the staged change request, target application, public source excerpt, and reference example before adapting code.",
        "kind": "mixed_skill_fixture",
    },
}


def _safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe fixture path: {name}")
    return path


def _is_read_assertion(text: str) -> bool:
    lowered = text.lower()
    return " reads " in f" {lowered} " or " consults " in f" {lowered} " or "does not read or activate" in lowered


def _read_kind(case_id: str) -> str:
    if case_id in {"source-version-drift", "implementation-scope-overreach"}:
        return "mixed_skill_fixture"
    return "skill_read"


def load_authored_cases(path: Path | None = None) -> list[dict]:
    source = Path(path or EVAL_ROOT / "evals.json")
    raw = json.loads(source.read_text())
    cases = raw.get("evals")
    if not isinstance(cases, list):
        raise ValueError("evals.json must contain an evals list")
    seen: set[str] = set()
    result = []
    for original in cases:
        case = dict(original)
        identity = case.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("case IDs must be unique nonempty strings")
        seen.add(identity)
        if not isinstance(case.get("prompt"), str) or not isinstance(
                case.get("assertions"), list):
            raise ValueError(f"invalid case contract: {identity}")
        if "files" not in case or not isinstance(case["files"], list):
            raise ValueError(f"case files must be explicit: {identity}")
        for name in case["files"]:
            relative = _safe_relative(name)
            if source.name == "evals.json" and not (source.parent /
                                                    relative).is_file():
                raise ValueError(f"missing fixture: {name}")
        shared, reads = [], []
        for index, text in enumerate(case["assertions"], 1):
            item = {"id": f"a{index}", "text": text}
            split = _ASSERTION_SPLITS.get((identity, index))
            if split:
                reads.append({
                    "id": item["id"],
                    "text": split["diagnostic"],
                    "original_text": text,
                    "kind": split["kind"]
                })
                shared.append({
                    "id": item["id"],
                    "text": split["shared"],
                    "original_text": text
                })
            elif _is_read_assertion(text):
                reads.append({**item, "kind": _read_kind(identity)})
            else:
                shared.append(item)
        case["read_assertions"] = reads
        case["shared_assertions"] = shared
        case["authorized_targets"] = dict(_TARGETS.get(identity, {}))
        case["required_writes"] = sorted(case["authorized_targets"].values())
        case["optional_writes"] = sorted(_OPTIONAL.get(identity, set()))
        result.append(case)
    return result


def build_run_plan(case_ids: list[str],
                   repetitions: int = 3,
                   seed: int = 20260913) -> list[dict]:
    if repetitions < 1 or len(case_ids) != len(set(case_ids)):
        raise ValueError("positive repetitions and unique cases required")
    blocks = [(case, repetition) for repetition in range(1, repetitions + 1)
              for case in case_ids]
    random.Random(seed).shuffle(blocks)
    runs = []
    for index, (case, repetition) in enumerate(blocks):
        arms = ("baseline", "candidate") if index % 2 == 0 else ("candidate",
                                                                 "baseline")
        for arm in arms:
            runs.append({
                "id": f"{case}--{repetition}--{arm}",
                "case": case,
                "repetition": repetition,
                "arm": arm
            })
    return runs


def _make_read_only(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
        elif path.is_dir():
            path.chmod(0o555)


def _skill_name(skill_root: Path) -> str:
    for line in (skill_root / "SKILL.md").read_text().splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip('"\'')
    raise ValueError("candidate SKILL.md has no name")


def stage_case(case: dict,
               workspace: Path,
               fixture_root: Path,
               *,
               arm: str,
               skill_root: Path | None = None,
               authoring: bool = False) -> None:
    if arm not in {"baseline", "candidate"}:
        raise ValueError(arm)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / ".tmp").mkdir()
    fixture_root = Path(fixture_root)
    if case.get("files"):
        (workspace / "input").mkdir()
    for name in case.get("files", []):
        relative = _safe_relative(name)
        source = fixture_root / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"fixture is missing or nonregular: {name}")
        staged = workspace / "input" / relative.name
        shutil.copy2(source, staged)
        staged.chmod(staged.stat().st_mode & ~0o222)
    for source_name, target_name in case.get("authorized_targets", {}).items():
        source = fixture_root / _safe_relative(source_name)
        target = workspace / _safe_relative(target_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | 0o200)
    if arm == "candidate":
        if skill_root is None:
            raise ValueError("candidate arm requires a skill snapshot")
        skill_root = Path(skill_root)
        destination = workspace / "skills" / _skill_name(skill_root)
        destination.parent.mkdir()
        destination.mkdir()
        parts = ("SKILL.md", ) if authoring else ("SKILL.md", "references",
                                                  "assets")
        for part in parts:
            source = skill_root / part
            if source.is_file():
                shutil.copy2(source, destination / part)
            elif source.is_dir():
                shutil.copytree(source, destination / part)
        if authoring:
            source = skill_root / "authoring"
            if source.is_dir():
                shutil.copytree(source, destination / "authoring")
            policy = skill_root / "coverage" / "policy.md"
            if policy.is_file():
                (destination / "coverage").mkdir()
                shutil.copy2(policy, destination / "coverage" / "policy.md")
            (destination / "references").mkdir()
            for relative in ("conventions", "conventions.md",
                             "source-provenance.md", "validation.md"):
                source = skill_root / "references" / relative
                target = destination / "references" / relative
                if source.is_dir():
                    shutil.copytree(source, target)
                elif source.is_file():
                    shutil.copy2(source, target)
        _make_read_only(destination)


def check_scope(before: dict[str, str], after: dict[str, str], *,
                required: set[str], optional: set[str]) -> dict:
    changed = {
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name) and not name.startswith(".tmp/")
    }
    allowed = set(required) | set(optional)
    unauthorized = sorted(changed - allowed)
    missing = sorted(name for name in required
                     if before.get(name) == after.get(name))
    return {
        "passed": not unauthorized and not missing,
        "changed": sorted(changed),
        "unauthorized": unauthorized,
        "missing_required": missing
    }


def behavioral_inventory(root: Path) -> dict[str, str]:
    """Hash regular files without following links and mark every nonregular node."""
    root = Path(root)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode):
            result[name] = "symlink:" + os.readlink(path)
            continue
        if not stat.S_ISREG(info.st_mode):
            result[name] = f"nonregular:{stat.S_IFMT(info.st_mode):o}"
            continue
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        digest_value = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest_value.update(chunk)
        result[name] = digest_value.hexdigest()
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "behavioral_inventory", "build_run_plan", "check_scope", "digest",
    "load_authored_cases", "stage_case"
]
