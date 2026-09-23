# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one frozen current-vs-pruned E03 pair on each local Qwen model.

Preparation and preflight make no model calls.  The live ``run`` action is
separate, one-shot, and must be explicitly reviewed before use.  Scientific
checks are a separate post-capture action and are never worker metrics.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


def _dependency(name: str, default: str) -> Path:
    """Preserved experiment dependencies live outside this repository.

    Point at them with environment variables; the defaults are placeholders
    that preflight reports as missing.
    """
    return Path(os.environ.get(name, default))


OLD_EVALS = _dependency("CUDAQ_PRUNING_OLD_EVALS", "dependencies/old-evals")
OLD_BEHAVIORAL = OLD_EVALS / "behavioral"
ROUTING_PATH = OLD_BEHAVIORAL / "routing_qwen.py"
DUAL_MODEL_PATH = OLD_BEHAVIORAL / "dual_model_smoke.py"
REPAIRED_BRIDGE_PATH = OLD_EVALS / "strategy/local_bridge.py"
PRESERVED_E03_RUNNER = _dependency("CUDAQ_PRUNING_E03_RUNNER",
                                   "dependencies/paired_app_e03.py")
CHECKER_PATH = _dependency("CUDAQ_PRUNING_E03_CHECKER",
                           "dependencies/native_e03_checker.py")
E2E_PYTHON = _dependency("CUDAQ_E2E_PYTHON", sys.executable)

EXPECTED_DEPENDENCY_SHA256 = {
    str(ROUTING_PATH):
    "2a63e07b34bde0087fc55be6669056deb973a72a0c863b9d205aed62fd2e8fa4",
    str(DUAL_MODEL_PATH):
    "6813725481f057b75a764341c21dd77d40fcd36993f32c8e49fd0c1e6981c9ca",
    str(REPAIRED_BRIDGE_PATH):
    "27eda9757b13dc024f44a4ea671f392beeb5d28de8f06a3b82b3ed235b351411",
    str(PRESERVED_E03_RUNNER):
    "2b2f2f0b2cffd9342cfde1fd3990c8dfadff1d7d9a126a99acc016a1df5aec4a",
    str(CHECKER_PATH):
    "de4c2c1f66a752cb51db38ee038791ac929416703270d7d79097de9765f9c53d",
}
CHECKER_HELPER_SHA256 = {
    "adapter": EXPECTED_DEPENDENCY_SHA256[str(CHECKER_PATH)],
    "capture":
    "26875cfd0964c8aeaeccd54c2747279bbfe08b13fd8d1fc2209400dcfb305a21",
    "numerical":
    "64ec74a0ae7fd45a79ed2272873ae5a1e87d6d886ffa04fbd6f3938ec925e27f",
    "runtime":
    "e069ffbe964fcb739e5c156a0cde4a41b87de0bb4af737d3ffd710bffc179d29",
    "oracle":
    "be7d6fb1377635798cdce12c70582bb89ff89c7021bd35b3af884185f1a506f2",
}
ORACLE_PATH = Path(
    __file__).resolve().parents[1] / "strategy/checks/test_protocol_action.py"
MANIFEST_NAME = "manifest.json"
PREFLIGHT_NAME = "preflight.json"
RUN_RESULT_NAME = "run-result.json"
CHECK_RESULT_NAME = "scientific-result.json"
RESERVATION_NAME = "reserved"
GENERATION_CONFIG: dict[str, float | int] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
}
BUDGETS = {
    "arm_timeout_seconds": 300,
    "max_backend_requests_per_arm": 20,
    "max_output_tokens_per_request": 2048,
    "backend_timeout_seconds": 120,
    "context_window": 32768,
    "compact_limit": 24576,
    "worker_retries": 0,
}


@dataclass(frozen=True)
class Attempt:
    model: str
    condition: str

    def as_dict(self) -> dict[str, str]:
        return {"model": self.model, "condition": self.condition}


SCHEDULE = (
    Attempt("qwen3-coder-30b", "current"),
    Attempt("qwen3-coder-30b", "pruned"),
    Attempt("qwen3-8b", "pruned"),
    Attempt("qwen3-8b", "current"),
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _save(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def operational_inventory(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"skill root is missing or unsafe: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] not in {"SKILL.md", "references"}:
            continue
        if path.is_symlink():
            raise ValueError(f"skill inventory contains a symlink: {relative}")
        if path.is_file() and path.suffix != ".pyc":
            result[relative.as_posix()] = sha256_file(path)
    if "SKILL.md" not in result:
        raise ValueError("operational skill has no SKILL.md")
    return result


def _inventory_sha256(inventory: Mapping[str, str]) -> str:
    raw = (json.dumps(
        dict(inventory), indent=2, sort_keys=True, ensure_ascii=True) +
           "\n").encode()
    return sha256_bytes(raw)


def _model_contracts() -> list[dict[str, Any]]:
    return [
        {
            "key": "qwen3-coder-30b",
            "slug": "local-qwen3-coder-30b",
            "provider": "local_qwen3_coder_30b",
            "endpoint": "http://127.0.0.1:18080/v1/chat/completions",
            "display_name": "Local Qwen3 Coder 30B",
            "context_window": 32768,
        },
        {
            "key": "qwen3-8b",
            "slug": "local-qwen3-8b",
            "provider": "local_qwen3_8b",
            "endpoint": "http://127.0.0.1:18081/v1/chat/completions",
            "display_name": "Local Qwen3 8B",
            "context_window": 32768,
        },
    ]


def manifest_contract(
    *,
    output: Path,
    current_skill: Path,
    pruned_skill: Path,
    source_inventory: Mapping[str, str],
    prompt: str,
    dependency_hashes: Mapping[str, str],
) -> dict[str, Any]:
    skills = {
        "current": operational_inventory(current_skill),
        "pruned": operational_inventory(pruned_skill),
    }
    if skills["current"].get("SKILL.md") == skills["pruned"].get("SKILL.md"):
        raise ValueError("current and pruned SKILL.md must differ")
    return {
        "schema_version": 1,
        "kind": "paired local-Qwen skill-pruning pilot; diagnostic only",
        "case_id": "E03",
        "output": str(Path(output).resolve()),
        "schedule": [row.as_dict() for row in SCHEDULE],
        "one_attempt_per_condition_per_model": True,
        "worker_retries": 0,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "source_inventory": dict(source_inventory),
        "source_inventory_sha256": _inventory_sha256(source_inventory),
        "skill_sources": {
            "current": str(Path(current_skill).resolve()),
            "pruned": str(Path(pruned_skill).resolve()),
        },
        "skill_inventories": skills,
        "models": _model_contracts(),
        "generation_config": dict(GENERATION_CONFIG),
        "budgets": dict(BUDGETS),
        "repaired_bridge": {
            "path": str(REPAIRED_BRIDGE_PATH),
            "sha256": EXPECTED_DEPENDENCY_SHA256[str(REPAIRED_BRIDGE_PATH)],
        },
        "dependency_sha256": dict(dependency_hashes),
        "scientific_checker": {
            "path": str(CHECKER_PATH),
            "sha256": EXPECTED_DEPENDENCY_SHA256[str(CHECKER_PATH)],
            "case_id": "E03",
            "targeted": {
                "expected_collected": 32,
                "allowed_skips": 0
            },
            "regression": {
                "expected_collected": 311,
                "allowed_skips": 4
            },
            "worker_metric": False,
        },
        "prepared_without_model_or_app_server": True,
        "quality_score": None,
        "acceptance_assessed": False,
    }


def is_budget_censored(result: Mapping[str, Any]) -> bool:
    if result.get("completed") is True or result.get(
            "failure_stage") != "turn_completion":
        return False
    audit = result.get("model_input_audit")
    requests = audit.get("request_count") if isinstance(audit,
                                                        Mapping) else None
    elapsed = result.get("elapsed_seconds")
    return (isinstance(requests, int) and not isinstance(requests, bool)
            and requests >= BUDGETS["max_backend_requests_per_arm"]) or (
                isinstance(elapsed,
                           (int, float)) and not isinstance(elapsed, bool)
                and elapsed >= BUDGETS["arm_timeout_seconds"])


def validated_attempt_captures(
        manifest: Mapping[str, Any],
        run_result: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    schedule = manifest.get("schedule")
    expected_schedule = [row.as_dict() for row in SCHEDULE]
    if schedule != expected_schedule:
        raise RuntimeError("manifest schedule changed")
    attempts = run_result.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(SCHEDULE):
        raise RuntimeError("exactly four recorded attempts are required")
    pair_roots = manifest.get("pair_roots")
    source_hash = manifest.get("source_inventory_sha256")
    if not isinstance(pair_roots, Mapping) or not isinstance(source_hash, str):
        raise RuntimeError("manifest capture binding is incomplete")
    result: dict[str, dict[str, str]] = {}
    for index, (scheduled, record) in enumerate(zip(SCHEDULE, attempts), 1):
        identity = scheduled.as_dict()
        if not isinstance(record,
                          Mapping) or record.get("index") != index or any(
                              record.get(name) != value
                              for name, value in identity.items()):
            raise RuntimeError("recorded attempt identities or order changed")
        key = f"{scheduled.model}/{scheduled.condition}"
        if key in result:
            raise RuntimeError("recorded attempt identity is duplicated")
        root_value = pair_roots.get(key)
        if not isinstance(root_value, str):
            raise RuntimeError("recorded attempt has no pair root")
        capture_path = Path(root_value) / "captures/candidate"
        capture = record.get("result", {}).get("capture") if isinstance(
            record.get("result"), Mapping) else None
        if not isinstance(capture,
                          Mapping) or capture.get("status") != "complete":
            raise RuntimeError("recorded attempt capture is incomplete")
        recorded_path = capture.get("capture_dir")
        if recorded_path is not None and Path(
                recorded_path).resolve() != capture_path.resolve():
            raise RuntimeError("recorded capture path changed")
        recorded_hash = capture.get("capture_json_sha256")
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise RuntimeError("recorded capture hash is invalid")
        if sha256_file(capture_path / "capture.json") != recorded_hash:
            raise RuntimeError("capture.json changed after the worker run")
        if capture.get("source_inventory_sha256") != source_hash:
            raise RuntimeError(
                "capture source binding differs from the frozen source")
        result[key] = {
            "path": str(capture_path),
            "capture_json_sha256": recorded_hash,
            "source_inventory_sha256": source_hash,
        }
    if list(result) != [f"{row.model}/{row.condition}" for row in SCHEDULE]:
        raise RuntimeError("recorded attempt set changed")
    return result


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dependencies() -> tuple[Any, Any, Any, Any]:
    actual = {
        path: sha256_file(Path(path))
        for path in EXPECTED_DEPENDENCY_SHA256
    }
    if actual != EXPECTED_DEPENDENCY_SHA256:
        changed = sorted(path for path in actual
                         if actual[path] != EXPECTED_DEPENDENCY_SHA256[path])
        raise RuntimeError(f"frozen dependency hash mismatch: {changed}")
    old_path = str(OLD_BEHAVIORAL)
    if old_path not in sys.path:
        sys.path.insert(0, old_path)
    routing = _load("pruning_pilot_routing", ROUTING_PATH)
    dual = _load("pruning_pilot_dual", DUAL_MODEL_PATH)
    bridge = _load("pruning_pilot_repaired_bridge", REPAIRED_BRIDGE_PATH)
    helper = routing._load_helper()
    return routing, dual, bridge, helper


def _spec(dual: Any, key: str) -> Any:
    return next(item for item in dual.MODELS if item.key == key)


def _pair_root(output: Path, model: str, condition: str) -> Path:
    return Path(output) / "pairs" / model / condition


@contextmanager
def _bound_model(helper: Any, dual: Any, bridge: Any, spec: Any,
                 root: Path) -> Iterator[None]:
    original = {
        (helper, "MODEL"): helper.MODEL,
        (helper, "PROVIDER"): helper.PROVIDER,
        (helper, "ARM_TIMEOUT_SECONDS"): helper.ARM_TIMEOUT_SECONDS,
        (helper, "MAX_REQUESTS"): helper.MAX_REQUESTS,
        (helper, "MAX_OUTPUT_TOKENS"): helper.MAX_OUTPUT_TOKENS,
        (helper, "BACKEND_TIMEOUT_SECONDS"): helper.BACKEND_TIMEOUT_SECONDS,
        (helper, "MODEL_CONTEXT_WINDOW"): helper.MODEL_CONTEXT_WINDOW,
        (helper, "COMPACT_LIMIT"): helper.COMPACT_LIMIT,
        (helper, "LocalBridgeConfig"): helper.LocalBridgeConfig,
        (helper, "_production_backend"): helper._production_backend,
        (helper, "ModelInputAudit"): helper.ModelInputAudit,
        (helper, "_command"): helper._command,
        (helper.v4, "MODEL"): helper.v4.MODEL,
        (helper.v4, "PROVIDER"): helper.v4.PROVIDER,
        (helper.v4, "_start"): helper.v4._start,
    }
    original_audit = helper.ModelInputAudit
    original_command = helper._command

    def bridge_config(**kwargs: Any) -> Any:
        return dual._ModelBridgeConfig(model=spec, **kwargs)

    class SamplingAudit:

        def __init__(self, skill: Path, backend: Any) -> None:
            self.delegate = original_audit(skill, backend)
            self.summary = self.delegate.summary
            self.path = Path(root) / "runs/candidate/backend-request.json"

        def __call__(self, payload: dict[str, Any], timeout: float) -> object:
            actual = {**payload, **GENERATION_CONFIG}
            if not self.path.exists():
                _save(self.path,
                      dual.setup_smoke._safe_payload(actual),
                      exclusive=True)
            return self.delegate(actual, timeout)

    def command(arm: str,
                base: str,
                collector: Any,
                current_root: Path = root) -> list[str]:
        argv = original_command(arm, base, collector, current_root)
        marker = "model_catalog_json=" + json.dumps(
            str(Path(root) / "model-catalog.json"))
        return [*argv, "-c", marker]

    replacements = {
        (helper, "MODEL"): spec.slug,
        (helper, "PROVIDER"): spec.provider,
        (helper, "ARM_TIMEOUT_SECONDS"): BUDGETS["arm_timeout_seconds"],
        (helper, "MAX_REQUESTS"): BUDGETS["max_backend_requests_per_arm"],
        (helper, "MAX_OUTPUT_TOKENS"):
        BUDGETS["max_output_tokens_per_request"],
        (helper, "BACKEND_TIMEOUT_SECONDS"):
        BUDGETS["backend_timeout_seconds"],
        (helper, "MODEL_CONTEXT_WINDOW"): BUDGETS["context_window"],
        (helper, "COMPACT_LIMIT"): BUDGETS["compact_limit"],
        (helper, "LocalBridgeConfig"): bridge_config,
        (helper, "_production_backend"): bridge._production_backend,
        (helper, "ModelInputAudit"): SamplingAudit,
        (helper, "_command"): command,
        (helper.v4, "MODEL"): spec.slug,
        (helper.v4, "PROVIDER"): spec.provider,
        (helper.v4, "_start"): bridge._start,
    }
    try:
        for (owner, name), value in replacements.items():
            setattr(owner, name, value)
        yield
    finally:
        for (owner, name), value in reversed(tuple(original.items())):
            setattr(owner, name, value)


def prepare(output: Path, current_skill: Path,
            pruned_skill: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    routing, dual, bridge, helper = _dependencies()
    source, _, excluded = helper._source_inputs()
    payload = helper.suite.worker_payload("E03", "candidate")
    prompt = payload["prompt"] + "\n\n" + payload["treatment_directive"]
    dependency_hashes = dict(EXPECTED_DEPENDENCY_SHA256)
    dependency_hashes[str(Path(__file__).resolve())] = sha256_file(
        Path(__file__))
    manifest = manifest_contract(
        output=output,
        current_skill=current_skill,
        pruned_skill=pruned_skill,
        source_inventory=source,
        prompt=prompt,
        dependency_hashes=dependency_hashes,
    )
    roots: dict[str, str] = {}
    freezes: dict[str, str] = {}
    for attempt in SCHEDULE:
        spec = _spec(dual, attempt.model)
        root = _pair_root(output, attempt.model, attempt.condition)
        inventory = manifest["skill_inventories"][attempt.condition]
        skill_root = Path(manifest["skill_sources"][attempt.condition])
        with _bound_model(helper, dual, bridge, spec, root):
            with routing._patched_prepare_inputs(helper, skill_root, source,
                                                 inventory, excluded):
                helper.prepare(root)
        dual._write_catalog(root / "model-catalog.json", spec)
        freeze = _read(root / "freeze.json")
        if freeze["rendered_prompts"]["candidate"] != prompt:
            raise RuntimeError("E03 prompt drifted during preparation")
        if freeze["candidate_skill_inventory"] != inventory:
            raise RuntimeError("staged skill differs from supplied skill")
        key = f"{attempt.model}/{attempt.condition}"
        roots[key] = str(root)
        freezes[key] = sha256_file(root / "freeze.json")
    manifest["pair_roots"] = roots
    manifest["pair_freeze_sha256"] = freezes
    manifest["model_catalog_sha256"] = {
        key: sha256_file(Path(root) / "model-catalog.json")
        for key, root in roots.items()
    }
    _save(output / MANIFEST_NAME, manifest, exclusive=True)
    return manifest


def _verify(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = _read(output / MANIFEST_NAME)
    if manifest.get("schedule") != [row.as_dict() for row in SCHEDULE]:
        raise RuntimeError("schedule changed")
    if manifest.get("dependency_sha256",
                    {}).get(str(Path(__file__).resolve())) != sha256_file(
                        Path(__file__)):
        raise RuntimeError("adapter changed after preparation")
    for path, expected in EXPECTED_DEPENDENCY_SHA256.items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"dependency changed after preparation: {path}")
    for key, raw_root in manifest["pair_roots"].items():
        root = Path(raw_root).resolve()
        if root != output / "pairs" / key:
            raise RuntimeError("pair root escaped output")
        if sha256_file(root /
                       "freeze.json") != manifest["pair_freeze_sha256"][key]:
            raise RuntimeError("pair freeze changed")
        if sha256_file(
                root /
                "model-catalog.json") != manifest["model_catalog_sha256"][key]:
            raise RuntimeError("model catalog changed")
    return manifest


def preflight(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = _verify(output)
    _, dual, bridge, helper = _dependencies()
    reports: dict[str, Any] = {}
    for attempt in SCHEDULE:
        key = f"{attempt.model}/{attempt.condition}"
        root = Path(manifest["pair_roots"][key])
        with _bound_model(helper, dual, bridge, _spec(dual, attempt.model),
                          root):
            helper.preflight(root)
        report = _read(root / "preflight.json")
        reports[key] = report
    result = {
        "passed": all(item.get("passed") is True for item in reports.values()),
        "pairs": reports,
        "schedule": manifest["schedule"],
        "model_contacted": False,
        "app_server_started": False,
        "bridge_started": False,
        "reservation_created": False,
    }
    _save(output / PREFLIGHT_NAME, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError("preflight failed")
    return result


def run(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = _verify(output)
    preflight_result = _read(output / PREFLIGHT_NAME)
    if preflight_result.get("passed") is not True:
        raise RuntimeError("passing preflight is required")
    descriptor = os.open(output / RESERVATION_NAME,
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    _, dual, bridge, helper = _dependencies()
    attempts: list[dict[str, Any]] = []
    for index, attempt in enumerate(SCHEDULE, 1):
        key = f"{attempt.model}/{attempt.condition}"
        root = Path(manifest["pair_roots"][key])
        freeze = helper.verify_ready(root)
        raised: BaseException | None = None
        try:
            with _bound_model(helper, dual, bridge, _spec(dual, attempt.model),
                              root):
                result = helper._run_arm("candidate", freeze, root)
        except BaseException as exc:
            raised = exc
            result = _read(root / "runs/candidate/result.json")
        record = {
            "index": index,
            **attempt.as_dict(),
            "result": result,
            "budget_censored": is_budget_censored(result),
            "raised_exception_type": type(raised).__name__ if raised else None,
        }
        attempts.append(record)
        _save(output / "attempts" /
              f"{index:02d}-{attempt.model}-{attempt.condition}.json",
              record,
              exclusive=True)
        _save(
            output / RUN_RESULT_NAME,
            {
                "schedule": manifest["schedule"],
                "attempts": attempts,
                "schedule_complete": len(attempts) == len(SCHEDULE),
                "scientific_checks_run": False,
                "acceptance_assessed": False,
                "quality_score": None,
            },
        )
        capture = result.get("capture")
        if not isinstance(capture,
                          Mapping) or capture.get("status") != "complete":
            raise RuntimeError(f"unsafe incomplete capture: {key}") from raised
    return _read(output / RUN_RESULT_NAME)


def check(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = _verify(output)
    run_result = _read(output / RUN_RESULT_NAME)
    if run_result.get("schedule_complete") is not True:
        raise RuntimeError(
            "all four captures are required before scientific checks")
    bindings = validated_attempt_captures(manifest, run_result)
    results: dict[str, Any] = {}
    for model in ("qwen3-coder-30b", "qwen3-8b"):
        current_root = Path(manifest["pair_roots"][f"{model}/current"])
        current_binding = bindings[f"{model}/current"]
        pruned_binding = bindings[f"{model}/pruned"]
        current_capture = Path(current_binding["path"])
        pruned_capture = Path(pruned_binding["path"])
        destination = output / "checks" / model
        command = [
            str(E2E_PYTHON),
            "-B",
            str(CHECKER_PATH),
            "--frozen-source",
            str(current_root / "frozen_public"),
            "--baseline-capture",
            str(current_capture),
            "--candidate-capture",
            str(pruned_capture),
            "--baseline-capture-sha256",
            current_binding["capture_json_sha256"],
            "--candidate-capture-sha256",
            pruned_binding["capture_json_sha256"],
            "--adapter-sha256",
            CHECKER_HELPER_SHA256["adapter"],
            "--capture-helper-sha256",
            CHECKER_HELPER_SHA256["capture"],
            "--numerical-helper-sha256",
            CHECKER_HELPER_SHA256["numerical"],
            "--runtime-helper-sha256",
            CHECKER_HELPER_SHA256["runtime"],
            "--oracle-sha256",
            CHECKER_HELPER_SHA256["oracle"],
            "--oracle",
            str(ORACLE_PATH),
            "--python",
            str(E2E_PYTHON),
            "--output",
            str(destination),
        ]
        completed = subprocess.run(command,
                                   text=True,
                                   capture_output=True,
                                   check=False)
        results[model] = {
            "returncode":
            completed.returncode,
            "command":
            command,
            "result":
            _read(destination / "result.json") if
            (destination / "result.json").is_file() else None,
            "stderr":
            completed.stderr,
        }
    payload = {
        "case_id": "E03",
        "results": results,
        "worker_metric": False,
        "acceptance_assessed": False,
    }
    _save(output / CHECK_RESULT_NAME, payload, exclusive=True)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--current-skill", required=True, type=Path)
    prepare_parser.add_argument("--pruned-skill", required=True, type=Path)
    for action in ("preflight", "run", "check"):
        command = actions.add_parser(action)
        command.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.action == "prepare":
        value = prepare(args.output, args.current_skill, args.pruned_skill)
    elif args.action == "preflight":
        value = preflight(args.output)
    elif args.action == "run":
        value = run(args.output)
    else:
        value = check(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
