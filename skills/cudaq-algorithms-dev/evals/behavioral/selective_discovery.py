# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a small, frozen current-vs-keywords skill-discovery experiment.

This deliberately reuses the native behavioral worker and its sandbox.  It is
diagnostic infrastructure, not a correctness benchmark or an LLM grader.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import shlex
import shutil

from behavioral_run import (
    DEFAULT_PYTHON,
    MODEL,
    _catalog,
    _runtime_provenance,
    _worker_result,
    _worker_runtime_metadata,
    preflight_campaign,
)

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
E2E = HERE.parent / "e2e"
REPETITIONS = 3
SEED = 20260914
PARALLEL = 4
WORKER_TIMEOUT = 180
VARIANTS = ("current", "keywords")
SKILL_CATALOG_ROOT = "skills/cudaq-algorithms"
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case_hash(cases: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(cases, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def _save(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe_relative(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} entries must be relative paths")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {field} path: {value}")
    return path.as_posix()


def _tree_inventory(root: Path) -> dict[str, str]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(
            f"skill snapshot root is missing or nonregular: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(
                f"skill snapshots may not contain symlinks: {relative}")
        if path.is_file():
            result[relative] = _hash(path)
    return result


def _supplied_inventory(root: Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for name in ("SKILL.md", "references", "assets"):
        path = root / name
        if path.is_symlink():
            raise ValueError(
                f"skill snapshots may not contain symlinks: {name}")
        if path.is_file():
            result[name] = _hash(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                relative = child.relative_to(root).as_posix()
                if child.is_symlink():
                    raise ValueError(
                        f"skill snapshots may not contain symlinks: {relative}"
                    )
                if child.is_file():
                    result[relative] = _hash(child)
    if "SKILL.md" not in result:
        raise ValueError(f"skill snapshot has no SKILL.md: {root}")
    return result


def _allowed_difference(path: str, allowed: list[str]) -> bool:
    return any(path == item or path.startswith(item.rstrip("/") + "/")
               for item in allowed)


def _validate_cases(cases: object,
                    inventories: dict[str, dict[str, str]]) -> list[dict]:
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    frozen = copy.deepcopy(cases)
    seen = set()
    for case in frozen:
        identity = case.get("id") if isinstance(case, dict) else None
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("case IDs must be unique nonempty strings")
        seen.add(identity)
        if not isinstance(case.get("prompt"), str):
            raise ValueError(f"invalid prompt: {identity}")
        if not isinstance(case.get("files"), list):
            raise ValueError(f"files must be explicit: {identity}")
        for name in case["files"]:
            _safe_relative(name, "files")
        if not isinstance(case.get("expected_activation"), bool):
            raise ValueError(
                f"expected_activation must be boolean: {identity}")
        records = case.get("expected_records")
        if not isinstance(records, list) or len(records) != len(set(records)):
            raise ValueError(
                f"expected_records must be a unique list: {identity}")
        case["expected_records"] = [
            _safe_relative(name, "expected_records") for name in records
        ]
        for name in case["expected_records"]:
            missing = [
                variant for variant in VARIANTS
                if name not in inventories[variant]
            ]
            if missing:
                raise ValueError(
                    f"expected record missing from {','.join(missing)}: {name}"
                )
        for field in ("routing_required", "activation_required"):
            if field in case and not isinstance(case[field], bool):
                raise ValueError(f"{field} must be boolean: {identity}")
    return frozen


def _fixture_inventory(cases: list[dict]) -> dict[str, str]:
    names = sorted({name for case in cases for name in case["files"]})
    result = {}
    for name in names:
        path = EVAL_ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or nonregular fixture: {name}")
        result[name] = _hash(path)
    return result


def _evaluator_files() -> list[Path]:
    return [
        HERE / "selective_discovery.py",
        HERE / "behavioral_run.py",
        HERE / "behavioral_cases.py",
        HERE / "behavioral_execution.py",
        HERE / "behavioral_runtime.py",
        E2E / "runtime.py",
        E2E / "telemetry.py",
        E2E / "run.py",
    ]


def _copy_snapshot(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in ("SKILL.md", "references", "assets"):
        origin = source / name
        target = destination / name
        if origin.is_file():
            shutil.copy2(origin, target)
        elif origin.is_dir():
            shutil.copytree(origin, target)


def _build_schedule(case_ids: list[str]) -> list[dict]:
    units = [(case, repetition) for repetition in range(1, REPETITIONS + 1)
             for case in case_ids]
    rng = random.Random(SEED)
    rng.shuffle(units)
    schedule = []
    attempt = 0
    for pair, (case, repetition) in enumerate(units, 1):
        order = list(VARIANTS)
        rng.shuffle(order)
        for variant in order:
            attempt += 1
            worker_run = {
                "id": f"attempt-{attempt:03d}",
                "case": case,
                "repetition": repetition,
                "arm": "candidate",
            }
            schedule.append({
                "pair": pair,
                "variant": variant,
                "case": case,
                "repetition": repetition,
                "worker_run": worker_run,
            })
    return schedule


def prepare(output: Path, spec: dict, *, python: str = DEFAULT_PYTHON) -> dict:
    """Validate and freeze an explicit two-variant experiment contract."""
    if not isinstance(spec, dict):
        raise ValueError("experiment input must be an object")
    fixed_controls = {
        "repetitions": REPETITIONS,
        "seed": SEED,
        "worker_timeout_s": WORKER_TIMEOUT,
    }
    for name, expected in fixed_controls.items():
        if spec.get(name, expected) != expected:
            raise ValueError(f"{name} must be fixed at {expected}")
    parallel = spec.get("parallel", PARALLEL)
    if not isinstance(parallel, int) or isinstance(
            parallel, bool) or not 1 <= parallel <= PARALLEL:
        raise ValueError(
            f"parallel must be an integer from 1 through {PARALLEL}")
    snapshots = spec.get("snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(VARIANTS):
        raise ValueError("snapshots must contain exactly current and keywords")
    sources = {
        variant: Path(snapshots[variant]).resolve()
        for variant in VARIANTS
    }
    source_inventories = {
        variant: _supplied_inventory(path)
        for variant, path in sources.items()
    }
    allowed = spec.get("allowed_snapshot_differences", ["SKILL.md"])
    if not isinstance(allowed, list):
        raise ValueError("allowed_snapshot_differences must be a list")
    allowed = [
        _safe_relative(item, "allowed_snapshot_differences")
        for item in allowed
    ]
    all_names = set(source_inventories["current"]) | set(
        source_inventories["keywords"])
    differences = sorted(name for name in all_names
                         if source_inventories["current"].get(name) !=
                         source_inventories["keywords"].get(name))
    unapproved = [
        name for name in differences if not _allowed_difference(name, allowed)
    ]
    if unapproved:
        raise ValueError("unapproved snapshot differences: " +
                         ", ".join(unapproved))
    for variant in VARIANTS:
        catalog = _catalog(sources[variant])
        if catalog["name"] != "cudaq-algorithms" or catalog[
                "path"] != f"{SKILL_CATALOG_ROOT}/SKILL.md":
            raise ValueError(
                "both snapshots must expose the cudaq-algorithms catalog path")
    cases = _validate_cases(spec.get("cases"), source_inventories)
    fixtures = _fixture_inventory(cases)
    evaluator_inventory = {
        str(path.resolve()): _hash(path)
        for path in _evaluator_files()
    }
    runtime = _runtime_provenance(python)
    worker_runtime = _worker_runtime_metadata(python)

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_paths = {}
    frozen_inventory = {}
    for index, variant in enumerate(VARIANTS, 1):
        destination = output / "frozen" / f"snapshot-{index:02d}" / "cudaq-algorithms"
        _copy_snapshot(sources[variant], destination)
        frozen_paths[variant] = str(destination)
        frozen_inventory[variant] = _tree_inventory(destination)
    manifest = {
        "kind":
        "bounded native selective discovery diagnostic; no semantic grader",
        "model": MODEL,
        "reasoning_effort": "low",
        "repetitions": REPETITIONS,
        "seed": SEED,
        "parallel": parallel,
        "worker_timeout_s": WORKER_TIMEOUT,
        "stop_on_pass": False,
        "labels_visible_to_worker": False,
        "catalog_path": f"{SKILL_CATALOG_ROOT}/SKILL.md",
        "allowed_snapshot_differences": allowed,
        "observed_snapshot_differences": differences,
        "cases": cases,
        "case_contract_sha256": _case_hash(cases),
        "fixture_inventory": fixtures,
        "evaluator_inventory": evaluator_inventory,
        "runtime_provenance": runtime,
        "worker_runtime": worker_runtime,
        "snapshot_paths": frozen_paths,
        "snapshot_inventory": frozen_inventory,
        "schedule": _build_schedule([case["id"] for case in cases]),
        "limitations": {
            "activation":
            "heuristic command/output observation, not proof of comprehension",
            "per_file_tokens": None,
        },
    }
    _save(output / "manifest.json", manifest)
    return manifest


def _load_manifest(output: Path) -> dict:
    try:
        return json.loads(
            (Path(output).resolve() / "manifest.json").read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"invalid selective discovery manifest: {exc}") from exc


def _verify_fixtures(manifest: dict) -> None:
    cases = manifest["cases"]
    if _fixture_inventory(cases) != manifest["fixture_inventory"]:
        raise RuntimeError("fixture drift")


def _verify_snapshots_and_evaluators(manifest: dict) -> None:
    for variant in VARIANTS:
        if _tree_inventory(
                Path(manifest["snapshot_paths"]
                     [variant])) != manifest["snapshot_inventory"][variant]:
            raise RuntimeError(f"{variant} snapshot drift")
    current_evaluators = {
        str(path.resolve()): _hash(path)
        for path in _evaluator_files()
    }
    if current_evaluators != manifest["evaluator_inventory"]:
        raise RuntimeError("evaluator drift")


def _verify_frozen(manifest: dict,
                   python: str,
                   *,
                   runtime: bool = True) -> None:
    if _case_hash(manifest.get("cases",
                               [])) != manifest.get("case_contract_sha256"):
        raise RuntimeError("case contract drift")
    fixed = {
        "model": MODEL,
        "reasoning_effort": "low",
        "repetitions": REPETITIONS,
        "seed": SEED,
        "worker_timeout_s": WORKER_TIMEOUT,
        "stop_on_pass": False,
    }
    if any(manifest.get(name) != expected for name, expected in fixed.items()):
        raise RuntimeError("fixed experiment control drift")
    parallel = manifest.get("parallel")
    if not isinstance(parallel, int) or isinstance(
            parallel, bool) or not 1 <= parallel <= PARALLEL:
        raise RuntimeError("parallel control drift")
    expected_schedule = _build_schedule(
        [case["id"] for case in manifest["cases"]])
    if manifest.get("schedule") != expected_schedule:
        raise RuntimeError("schedule drift")
    _verify_fixtures(manifest)
    _verify_snapshots_and_evaluators(manifest)
    if runtime:
        if _runtime_provenance(python) != manifest["runtime_provenance"]:
            raise RuntimeError("interpreter or native CLI drift")
        if _worker_runtime_metadata(python) != manifest["worker_runtime"]:
            raise RuntimeError("worker runtime drift")


def preflight(output: Path, *, python: str = DEFAULT_PYTHON) -> dict:
    """Run the existing isolation preflight against both frozen variants."""
    output = Path(output).resolve()
    manifest = _load_manifest(output)
    if (output / "runs").exists() and any((output / "runs").iterdir()):
        raise RuntimeError("preflight must precede all attempts")
    _verify_frozen(manifest, python)
    checks = []
    for index, variant in enumerate(VARIANTS, 1):
        wrapper = output / "preflight" / f"snapshot-{index:02d}"
        wrapper.mkdir(parents=True, exist_ok=False)
        _save(wrapper / "manifest.json",
              {"candidate_snapshot": manifest["snapshot_paths"][variant]})
        _verify_fixtures(manifest)
        result = preflight_campaign(wrapper, manifest["cases"], python)
        _verify_fixtures(manifest)
        checks.append({"variant": variant, "result": result})
    payload = {
        "passed": all(item["result"].get("passed") for item in checks),
        "checks": checks
    }
    _save(output / "preflight.json", payload)
    if not payload["passed"]:
        raise RuntimeError("selective discovery isolation preflight failed")
    return payload


def _mentions_path(command: str, path: str) -> bool:
    escaped = re.escape(path)
    return bool(
        re.search(rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_.\-/])",
                  command))


def _command_segments(command: str) -> list[list[str]]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return []
    segments, current = [], []
    for token in tokens:
        if token and all(character in ";&|" for character in token):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _segment_mode(tokens: list[str]) -> str:
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return "unrecognized"
    executable = Path(tokens[0]).name
    command_options = [
        index for index, token in enumerate(tokens[1:], 1)
        if token == "--command"
        or bool(re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", token))
    ]
    if executable in {"bash", "sh", "zsh"} and command_options:
        position = command_options[0]
        nested = _command_segments(
            tokens[position + 1]) if position + 1 < len(tokens) else []
        modes = [_segment_mode(segment) for segment in nested]
        return "content" if "content" in modes else "unrecognized" if "unrecognized" in modes else "noncontent"
    if executable in {"python", "python3"}:
        joined = " ".join(tokens)
        return "content" if re.search(r"(?:read_text|read_bytes|open\s*\()",
                                      joined) else "unrecognized"
    if executable in {
            "cat", "sed", "head", "tail", "awk", "bat", "less", "more"
    }:
        return "content"
    if executable in {"grep", "rg"}:
        options = [token for token in tokens[1:] if token.startswith("-")]
        name_only = any(
            option in {
                "--files", "--files-with-matches", "--files-without-match",
                "--count", "-c"
            } or (option.startswith("-") and not option.startswith("--")
                  and "l" in option[1:]) for option in options)
        return "noncontent" if name_only else "content"
    if executable in {
            "echo", "printf", "ls", "find", "pwd", "stat", "test", "realpath",
            "dirname", "basename", "which", "type"
    }:
        return "noncontent"
    return "unrecognized"


def _path_use(command: str, target: str) -> str | None:

    def classify(segment: list[str]) -> str | None:
        executable = Path(segment[0]).name if segment else ""
        command_options = [
            index for index, token in enumerate(segment[1:], 1)
            if token == "--command"
            or bool(re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", token))
        ]
        if executable in {"bash", "sh", "zsh"} and command_options:
            position = command_options[0]
            if position + 1 >= len(segment):
                return "unrecognized" if _mentions_path(
                    " ".join(segment), target) else None
            nested_modes = [
                classify(nested)
                for nested in _command_segments(segment[position + 1])
            ]
            return combine(nested_modes)
        if not _mentions_path(" ".join(segment), target):
            return None
        return _segment_mode(segment)

    def combine(modes: list[str | None]) -> str | None:
        if "content" in modes:
            return "content"
        if "unrecognized" in modes:
            return "unrecognized"
        if "noncontent" in modes:
            return "noncontent"
        return None

    modes = [classify(segment) for segment in _command_segments(command)]
    return combine(modes)


def _observation(events: list[dict], targets: list[str], *,
                 available: bool) -> dict:
    attempts = []
    for index, wrapped in enumerate(events, 1):
        event = wrapped.get("event", wrapped)
        if event.get("type") != "item.completed":
            continue
        item = event.get("item", {})
        if item.get("type") != "command_execution":
            continue
        command = str(item.get("command", ""))
        uses = {target: _path_use(command, target) for target in targets}
        matched = [
            target for target, mode in uses.items()
            if mode in {"content", "unrecognized"}
        ]
        if not matched:
            continue
        output = item.get("aggregated_output")
        exit_code = item.get("exit_code")
        complete_native_result = type(exit_code) is int and isinstance(
            output, str)
        if not complete_native_result or all(uses[target] == "unrecognized"
                                             for target in matched):
            status = "unrecognized"
        else:
            status = "observed" if exit_code == 0 and output.strip(
            ) else "failed" if exit_code != 0 else "empty"
        attempts.append({
            "status": status,
            "event_index": index,
            "order": index,
            "elapsed_s": wrapped.get("elapsed_s"),
            "command": command,
            "matched_paths": matched,
        })
    observed = next(
        (item for item in attempts if item["status"] == "observed"), None)
    if observed:
        return observed
    if attempts:
        return attempts[0]
    return {
        "status": "not_observed" if available else "unavailable",
        "event_index": None,
        "order": None,
        "elapsed_s": None,
        "command": None,
        "matched_paths": []
    }


def observe_reads(events: list[dict],
                  expected_records: list[str],
                  *,
                  worker_completed: bool,
                  events_available: bool = True,
                  known_skill_paths: list[str] | None = None) -> dict:
    """Extract conservative read timing from raw native command events.

    This records only successful, output-producing content commands.  It makes
    no claim about exact per-file tokens, context inclusion, or comprehension.
    """
    relative_records = [
        _safe_relative(path, "expected_records") for path in expected_records
    ]
    records = [f"{SKILL_CATALOG_ROOT}/{path}" for path in relative_records]
    known = known_skill_paths or ["SKILL.md", *expected_records]
    skill_targets = [
        f"{SKILL_CATALOG_ROOT}/{_safe_relative(path, 'skill inventory')}"
        for path in known
    ]
    available = bool(events_available and worker_completed)
    skill = _observation(events, skill_targets, available=available)
    record_results = {
        relative: _observation(events, [full], available=available)
        for relative, full in zip(relative_records, records)
    }
    observed_records = [
        item for item in record_results.values()
        if item["status"] == "observed"
    ]
    first_record = min(
        observed_records,
        key=lambda item: item["order"]) if observed_records else None
    first_skill = skill if skill["status"] == "observed" else None
    if skill["status"] == "observed":
        activation: bool | str = True
    elif not worker_completed or not events_available or skill["status"] in {
            "failed", "empty", "unrecognized"
    }:
        activation = "unknown"
    else:
        activation = False
    command_count = sum(
        (wrapped.get("event", wrapped).get("type") == "item.completed"
         and wrapped.get("event", wrapped).get("item", {}).get(
             "type") == "command_execution") for wrapped in events)
    return {
        "activation":
        activation,
        "skill":
        skill,
        "records":
        record_results,
        "first_skill_content_read":
        first_skill,
        "first_expected_record_read":
        first_record,
        "command_count":
        command_count,
        "limitation":
        ("heuristic successful command/output evidence only; no exact per-file billed tokens, "
         "context inclusion, or comprehension is inferred"),
    }


def _events(destination: Path) -> tuple[list[dict], bool, str | None]:
    path = destination / "events.jsonl"
    if not path.is_file():
        return [], False, "events.jsonl unavailable"
    try:
        return [
            json.loads(line) for line in path.read_text().splitlines() if line
        ], True, None
    except (OSError, ValueError, TypeError) as exc:
        return [], False, f"events.jsonl unreadable: {type(exc).__name__}: {exc}"


def _failure_result(destination: Path, row: dict, exc: Exception,
                    known_skill_paths: list[str]) -> dict:
    result_path = destination / "result.json"
    try:
        retained = json.loads(result_path.read_text())
    except (OSError, ValueError, TypeError):
        retained = dict(row["worker_run"])
    retained.update({
        "variant": row["variant"],
        "worker_completed": False,
        "status": "infrastructure_failure",
        "reason": f"{type(exc).__name__}: {exc}",
    })
    retained.setdefault("worker_attempted",
                        (destination / "command.json").is_file())
    retained.setdefault("worker_elapsed_s", None)
    retained.setdefault("worker_telemetry", None)
    events, available, event_error = _events(destination)
    retained["discovery"] = observe_reads(
        events,
        row.get("expected_records", []),
        worker_completed=False,
        events_available=available,
        known_skill_paths=known_skill_paths,
    )
    if event_error:
        retained["discovery"]["event_error"] = event_error
    _save(result_path, retained)
    return retained


def run(output: Path, *, python: str = DEFAULT_PYTHON) -> None:
    """Execute every paired attempt once, retaining every failure and timeout."""
    output = Path(output).resolve()
    manifest = _load_manifest(output)
    try:
        preflight_result = json.loads((output / "preflight.json").read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "successful isolation preflight required before live runs"
        ) from exc
    if not preflight_result.get("passed"):
        raise RuntimeError(
            "successful isolation preflight required before live runs")
    if (output / "runs").exists() and any((output / "runs").iterdir()):
        raise RuntimeError(
            "selective discovery attempts are one-shot and cannot be retried")
    _verify_frozen(manifest, python)
    (output / "runs").mkdir(exist_ok=True)
    cases = {case["id"]: case for case in manifest["cases"]}
    pairs = [
        manifest["schedule"][index:index + 2]
        for index in range(0, len(manifest["schedule"]), 2)
    ]

    def execute_pair(pair: list[dict]) -> None:
        for row in pair:
            worker_run = row["worker_run"]
            destination = output / "runs" / worker_run["id"]
            case = cases[row["case"]]
            augmented = {**row, "expected_records": case["expected_records"]}
            try:
                _verify_fixtures(manifest)
                worker_manifest = {
                    **manifest,
                    "candidate_snapshot":
                    manifest["snapshot_paths"][row["variant"]],
                }
                result = _worker_result(worker_run, case, worker_manifest,
                                        output, python)
                _verify_fixtures(manifest)
                events, available, event_error = _events(destination)
                known_paths = list(
                    manifest["snapshot_inventory"][row["variant"]])
                discovery = observe_reads(
                    events,
                    case["expected_records"],
                    worker_completed=result.get("worker_completed", False),
                    events_available=available,
                    known_skill_paths=known_paths,
                )
                if event_error:
                    discovery["event_error"] = event_error
                result.update({
                    "variant":
                    row["variant"],
                    "expected_activation":
                    case["expected_activation"],
                    "routing_required":
                    case.get("routing_required", True),
                    "activation_required":
                    case.get("activation_required", True),
                    "discovery":
                    discovery,
                })
                _save(destination / "result.json", result)
            except Exception as exc:  # each native attempt remains one-shot
                destination.mkdir(parents=True, exist_ok=True)
                _failure_result(
                    destination,
                    augmented,
                    exc,
                    list(manifest["snapshot_inventory"][row["variant"]]),
                )

    with ThreadPoolExecutor(
            max_workers=min(PARALLEL, manifest["parallel"])) as executor:
        list(executor.map(execute_pair, pairs))


def _resource(values: list[object]) -> dict:
    available = [
        value for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "sum": sum(available) if available else None,
        "available": len(available),
        "attempts": len(values)
    }


def _activation_group(case: dict) -> str:
    if not case["expected_activation"]:
        return "negative"
    if not case.get("activation_required", True) or not case.get(
            "routing_required", True):
        return "optional"
    return "positive"


def _load_results(output: Path,
                  manifest: dict) -> tuple[list[dict], list[dict]]:
    results = []
    missing_attempts = []
    for row in manifest["schedule"]:
        destination = output / "runs" / row["worker_run"]["id"]
        path = destination / "result.json"
        if path.is_file():
            results.append(json.loads(path.read_text()))
        else:
            partial = sorted(
                str(item.relative_to(destination))
                for item in destination.rglob("*")
                if item.is_file()) if destination.is_dir() else []
            missing_attempts.append({
                "id": row["worker_run"]["id"],
                "case": row["case"],
                "variant": row["variant"],
                "partial_evidence": partial,
            })
    return results, missing_attempts


def _write_report(destination: Path, manifest: dict, results: list[dict],
                  missing_attempts: list[dict], *, title: str) -> dict:
    destination = Path(destination).resolve()
    cases = {case["id"]: case for case in manifest["cases"]}
    activation = {
        kind: {
            variant: {
                "true": 0,
                "false": 0,
                "unknown": 0
            }
            for variant in VARIANTS
        }
        for kind in ("positive", "optional", "negative")
    }
    for row in results:
        kind = _activation_group(cases[row["case"]])
        observed = row.get("discovery", {}).get("activation", "unknown")
        key = "true" if observed is True else "false" if observed is False else "unknown"
        activation[kind][row["variant"]][key] += 1

    per_case = {}
    for case_id in cases:
        per_case[case_id] = {}
        for variant in VARIANTS:
            subset = [
                row for row in results
                if row["case"] == case_id and row["variant"] == variant
            ]
            totals = {
                field:
                _resource([(row.get("worker_telemetry")
                            or {}).get("totals", {}).get(field)
                           for row in subset])
                for field in TOKEN_FIELDS
            }
            totals["uncached_input_tokens"] = _resource([
                (telemetry["input_tokens"] - telemetry["cached_input_tokens"]
                 if isinstance(telemetry.get("input_tokens"), (int, float))
                 and isinstance(telemetry.get("cached_input_tokens"),
                                (int, float)) else None) for row in subset
                for telemetry in [(
                    row.get("worker_telemetry") or {}).get("totals", {})]
            ])
            per_case[case_id][variant] = {
                "attempts":
                len(subset),
                "completed":
                sum(row.get("worker_completed", False) for row in subset),
                "activation": {
                    "true":
                    sum(
                        row.get("discovery", {}).get("activation") is True
                        for row in subset),
                    "false":
                    sum(
                        row.get("discovery", {}).get("activation") is False
                        for row in subset),
                    "unknown":
                    sum(
                        row.get("discovery", {}).get("activation") not in (
                            True, False) for row in subset),
                },
                "elapsed_s":
                _resource([row.get("worker_elapsed_s") for row in subset]),
                "native_totals":
                totals,
                "command_count":
                _resource([
                    row.get("discovery", {}).get("command_count")
                    for row in subset
                ]),
                "first_skill_read_s":
                _resource([
                    (row.get("discovery", {}).get("first_skill_content_read")
                     or {}).get("elapsed_s") for row in subset
                ]),
                "first_expected_record_read_s":
                _resource([
                    (row.get("discovery", {}).get("first_expected_record_read")
                     or {}).get("elapsed_s") for row in subset
                ]),
                "diagnostics": [{
                    "id":
                    row["id"],
                    "status":
                    row.get("status"),
                    "activation":
                    row.get("discovery", {}).get("activation", "unknown"),
                    "skill_status":
                    row.get("discovery",
                            {}).get("skill", {}).get("status", "unavailable"),
                    "record_statuses": {
                        name: value.get("status", "unavailable")
                        for name, value in row.get("discovery", {}).get(
                            "records", {}).items()
                    },
                    "first_skill_content_read":
                    row.get("discovery", {}).get("first_skill_content_read"),
                    "first_expected_record_read":
                    row.get("discovery", {}).get("first_expected_record_read"),
                } for row in subset],
            }
    selective_outcomes = {
        "required_positive_misses": {
            variant:
            sum(cases[row["case"]]["expected_activation"]
                and cases[row["case"]].get("activation_required", True)
                and row.get("discovery", {}).get("activation") is False
                for row in results if row["variant"] == variant)
            for variant in VARIANTS
        },
        "negative_unwanted_loads": {
            variant:
            sum(not cases[row["case"]]["expected_activation"]
                and row.get("discovery", {}).get("activation") is True
                for row in results if row["variant"] == variant)
            for variant in VARIANTS
        },
        "unknown_observations": {
            variant:
            sum(
                row.get("discovery", {}).get("activation") not in (True, False)
                for row in results if row["variant"] == variant)
            for variant in VARIANTS
        },
    }
    summary = {
        "kind":
        manifest.get("kind"),
        "planned_attempts":
        len(manifest["schedule"]),
        "finalized_attempts":
        len(results),
        "missing_attempts":
        missing_attempts,
        "activation":
        activation,
        "selective_outcomes":
        selective_outcomes,
        "per_case":
        per_case,
        "diagnostic_only_cases":
        sorted(case_id for case_id, case in cases.items()
               if not case.get("routing_required", True)
               or not case.get("activation_required", True)),
        "failures": [{
            "id": row["id"],
            "case": row["case"],
            "variant": row["variant"],
            "status": row.get("status"),
            "reason": row.get("reason")
        } for row in results if not row.get("worker_completed", False)],
        "limitations": {
            "interpretation":
            "routing/read diagnostics only; semantic correctness requires manual final-answer review",
            "read_observation":
            "successful output-producing native command events are a heuristic, not proof of comprehension",
            "read_timing":
            "elapsed values are controller receive times from raw events, conditional on observed reads, not exact tool durations",
            "per_file_tokens":
            None,
            "token_accounting":
            "native whole-attempt totals are retained; cached input and reasoning output are subsets",
        },
    }
    _save(destination / "summary.json", summary)
    lines = [
        f"# {title}",
        "",
        f"Planned attempts: {summary['planned_attempts']}; finalized: {summary['finalized_attempts']}.",
        f"Missing scheduled attempts: {len(missing_attempts)}.",
        "",
        "## Activation observations",
        "",
        "| expected | variant | true | false | unknown |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("positive", "optional", "negative"):
        for variant in VARIANTS:
            counts = activation[kind][variant]
            lines.append(
                f"| {kind} | {variant} | {counts['true']} | {counts['false']} | {counts['unknown']} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a routing/read diagnostic, not a scientific correctness score. Failed and unavailable observations remain unknown. "
        "Timing/order comes from controller-receive timestamps and ordering in raw events only; no per-file billed-token or exact context claim is made.",
        "",
        "See `summary.json` and each run directory for native totals, command counts, first-read evidence, failures, and raw events.",
        "",
    ])
    (destination / "summary.md").write_text("\n".join(lines))
    return summary


def report(output: Path) -> dict:
    """Write diagnostic JSON and Markdown summaries without semantic scoring."""
    output = Path(output).resolve()
    manifest = _load_manifest(output)
    results, missing = _load_results(output, manifest)
    return _write_report(output,
                         manifest,
                         results,
                         missing,
                         title="Selective discovery diagnostic")


def reanalyze(output: Path, destination: Path) -> dict:
    """Recompute read observations from immutable raw events into a new report."""
    output = Path(output).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    manifest_path = output / "manifest.json"
    manifest = _load_manifest(output)
    consumed = [manifest_path]
    for row in manifest["schedule"]:
        run_root = output / "runs" / row["worker_run"]["id"]
        for name in ("result.json", "events.jsonl"):
            path = run_root / name
            if path.is_file():
                consumed.append(path)
    before = {str(path.relative_to(output)): _hash(path) for path in consumed}
    # Re-read after the initial inventory so the hashes bracket all consumed data.
    manifest = json.loads(manifest_path.read_text())
    results, missing = _load_results(output, manifest)
    schedule = {row["worker_run"]["id"]: row for row in manifest["schedule"]}
    corrected_results = []
    observations = []
    for original in results:
        row = schedule[original["id"]]
        run_root = output / "runs" / original["id"]
        events, available, event_error = _events(run_root)
        corrected_discovery = observe_reads(
            events,
            next(case for case in manifest["cases"]
                 if case["id"] == row["case"])["expected_records"],
            worker_completed=original.get("worker_completed", False),
            events_available=available,
            known_skill_paths=list(
                manifest["snapshot_inventory"][row["variant"]]),
        )
        if event_error:
            corrected_discovery["event_error"] = event_error
        corrected = copy.deepcopy(original)
        corrected["discovery"] = corrected_discovery
        corrected_results.append(corrected)
        observations.append({
            "id":
            original["id"],
            "case":
            row["case"],
            "variant":
            row["variant"],
            "worker_completed":
            original.get("worker_completed", False),
            "original_discovery":
            original.get("discovery"),
            "corrected_discovery":
            corrected_discovery,
            "worker_elapsed_s":
            original.get("worker_elapsed_s"),
            "worker_telemetry":
            original.get("worker_telemetry"),
        })
    after = {str(path.relative_to(output)): _hash(path) for path in consumed}
    if after != before:
        raise RuntimeError("source campaign changed during parser reanalysis")
    parser_path = str(Path(__file__).resolve())
    provenance = {
        "kind":
        "post-run read-observation parser reanalysis; no worker reruns",
        "parser_version":
        2,
        "original_parser_sha256":
        manifest.get("evaluator_inventory", {}).get(parser_path),
        "corrected_parser_sha256":
        _hash(Path(__file__)),
        "consumed_files":
        before,
        "source_campaign":
        str(output),
        "source_results_modified":
        False,
        "native_metrics":
        "copied in memory from original result.json without alteration",
        "limitation":
        "controller-receive timestamps and command/output heuristics; not exact tool duration or proof of comprehension",
    }
    destination.mkdir(parents=True, exist_ok=False)
    summary = _write_report(
        destination,
        manifest,
        corrected_results,
        missing,
        title="Selective discovery parser reanalysis v2",
    )
    summary["reanalysis_provenance"] = "provenance.json"
    _save(destination / "summary.json", summary)
    _save(destination / "observations.json", observations)
    _save(destination / "provenance.json", provenance)
    return {"summary": summary, "provenance": provenance}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=("prepare", "preflight", "run", "report",
                                 "reanalyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input",
                        type=Path,
                        help="frozen experiment JSON (prepare only)")
    parser.add_argument("--destination",
                        type=Path,
                        help="new output directory (reanalyze only)")
    parser.add_argument("--python",
                        default=DEFAULT_PYTHON,
                        help="clean stdlib-only behavioral worker interpreter")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if args.input is None:
            parser.error("prepare requires --input")
        prepare(args.output,
                json.loads(args.input.read_text()),
                python=args.python)
    elif args.action == "preflight":
        print(json.dumps(preflight(args.output, python=args.python)))
    elif args.action == "run":
        lock_path = args.output.resolve() / "campaign.lock"
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.output, python=args.python)
    elif args.action == "report":
        report(args.output)
    else:
        if args.destination is None:
            parser.error("reanalyze requires --destination")
        reanalyze(args.output, args.destination)


if __name__ == "__main__":
    main()
