#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover terminal pilot artifacts and run registered independent checks.

Only the frozen capture helper reads a stopped worker tree, inside a bounded
container. Host reconstruction verifies bytes without importing worker code.
All outputs are fresh, private, and separate from worker/verifier artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid

import capture
import numerical

HERE = Path(__file__).absolute().parent
LIMITS = {
    "max_file_bytes": capture.MAX_FILE_BYTES,
    "max_tree_bytes": capture.MAX_TREE_BYTES,
    "max_entries": capture.MAX_ENTRIES,
    "max_judge_bytes": capture.MAX_JUDGE_BYTES
}


def _read(path: Path, limit: int = 16 * 1024**2) -> bytes:
    directory = capture._directory(path.absolute().parent)
    try:
        return capture._read(directory, path.name, limit)
    finally:
        os.close(directory)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value,
                                     sort_keys=True).encode()).hexdigest()


def _manifest(root: Path) -> dict:
    raw = _read(root / "manifest.json")
    expected = json.loads(_read(root / "manifest.sha256.json"))["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("manifest hash mismatch")
    result = json.loads(raw)
    if (not isinstance(result, dict) or result.get("schema_version") != 1
            or not isinstance(result.get("attempts"), list)
            or any(not isinstance(attempt, dict)
                   for attempt in result["attempts"])):
        raise ValueError("invalid manifest")
    return result


def _terminal_trial(root: Path, attempt: dict) -> tuple[Path, dict]:
    identity = attempt["attempt_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}",
                        identity) or attempt.get("job_name") != identity:
        raise ValueError("invalid attempt identity")
    job = root / "jobs" / identity
    directory = capture._directory(job)
    try:
        with os.scandir(directory) as entries:
            names = []
            for index, entry in enumerate(entries):
                if index >= 100:
                    raise ValueError("too many trial entries")
                if entry.name.startswith(identity + "__"):
                    if not re.fullmatch(
                            re.escape(identity) + r"__[A-Za-z0-9_-]{1,128}",
                            entry.name):
                        raise ValueError("invalid trial name")
                    if not entry.is_dir(follow_symlinks=False):
                        raise ValueError("unsafe trial directory")
                    names.append(entry.name)
        if len(names) != 1:
            raise ValueError("exactly one trial required")
    finally:
        os.close(directory)
    trial = job / names[0]
    result = json.loads(_read(trial / "result.json", 2 * 1024**2))
    if not isinstance(result, dict) or not result.get("finished_at"):
        raise ValueError("trial is not terminal")
    if (result.get("task_name")
            not in (None, identity, "nvidia/skillevaluator-" + identity)
            or result.get("trial_name") not in (None, trial.name)):
        raise ValueError("terminal result identity mismatch")
    if "task_id" in result:
        task_id = result["task_id"]
        if (not isinstance(task_id, dict) or task_id.get("path") != str(
                root / attempt["dataset_dir"] / identity)):
            raise ValueError("terminal task path mismatch")
    return trial, result


def _inventory(root: Path) -> dict[str, str]:
    snapshot = capture._snapshot(root, "frozen", LIMITS)
    if snapshot["rejections"]:
        raise ValueError("incomplete frozen inventory")
    return {
        name: item["sha256"]
        for name, item in snapshot["inventory"].items()
    }


def _fresh_directory(path: Path) -> None:
    parent = capture._directory(path.parent)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def _mount(path: Path,
           target: str,
           *,
           writable: bool = False,
           file: bool = False) -> str:
    path = path.absolute()
    if any(c in str(path) for c in ",\r\n") or ".." in path.parts:
        raise ValueError("unsupported mount path")
    directory = capture._directory(path.parent if file else path)
    try:
        if file:
            info = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("unsafe helper mount")
    finally:
        os.close(directory)
    return f"type=bind,source={path},target={target}" + ("" if writable else
                                                         ",readonly")


def recovery_command(source: Path, worktree: Path, helper: Path, output: Path,
                     *, image_id: str, name: str) -> list[str]:
    """Mount only four exact inputs/outputs; DAC_OVERRIDE reads root-owned 0600 files."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("immutable image ID required")
    if not re.fullmatch(r"cudaq-capture-[a-z0-9-]{1,64}", name):
        raise ValueError("invalid recovery container name")
    return [
        "docker", "run", "--rm", "--name", name, "--pull", "never",
        "--network", "none", "--read-only", "--cap-drop", "ALL", "--cap-add",
        "DAC_OVERRIDE", "--security-opt", "no-new-privileges", "--user", "0:0",
        "--pids-limit", "64", "--memory", "512m", "--cpus", "2", "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m", "--env",
        "PYTHONDONTWRITEBYTECODE=1", "--mount",
        _mount(source, "/source"), "--mount",
        _mount(worktree, "/worktree"), "--mount",
        _mount(helper, "/capture.py", file=True), "--mount",
        _mount(output, "/output",
               writable=True), "--workdir", "/tmp", image_id, "timeout", "-s",
        "KILL", "120s", "python3", "-I", "/capture.py", "--source", "/source",
        "--worktree", "/worktree", "--output", "/output/capture"
    ]


def _recover(source: Path, worktree: Path, output: Path, image_id: str,
             record: dict) -> Path | None:
    # Validate the worktree before creating an output or starting Docker.
    _mount(worktree, "/worktree")
    _fresh_directory(output)
    name = "cudaq-capture-" + uuid.uuid4().hex
    argv = recovery_command(source,
                            worktree,
                            Path(capture.__file__),
                            output,
                            image_id=image_id,
                            name=name)
    env = {
        key: value
        for key, value in os.environ.items() if key in
        {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"}
    }
    started = time.monotonic()
    code = None
    try:
        result = subprocess.run(argv,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=120,
                                env=env)
        code = result.returncode
    except subprocess.TimeoutExpired:
        record["failures"].append("recovery_timeout")
        try:
            subprocess.run(["docker", "kill", name],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           timeout=20,
                           env=env)
        except (OSError, subprocess.TimeoutExpired):
            record["failures"].append("recovery_container_stop_unknown")
    finally:
        record["recovery"] = {
            "returncode": code,
            "elapsed_s": time.monotonic() - started,
            "timeout_s": 120,
            "network": "none",
            "worker_metric": False,
            "api_credentials_present": False,
            "image_id": image_id
        }
    if code != 0:
        record["failures"].append("recovery_incomplete_or_failed")
        return None
    return output / "capture"


def _save(output: Path, record: dict) -> dict:
    directory = capture._directory(output)
    try:
        capture._write(directory, "artifact_checks.json",
                       (json.dumps(record, indent=2, sort_keys=True) +
                        "\n").encode())
    finally:
        os.close(directory)
    return record


def _bind_inputs(check: dict, output: Path, repository: Path, record: dict, *,
                 case_id: str) -> None:
    """Bind a numerical freeze to the capture, checker, consumers and binding."""
    prefix = case_id.lower()
    raw = _read(output / f"{prefix}-inputs.json", 2 * 1024**2)
    digest = hashlib.sha256(raw).hexdigest()
    frozen = json.loads(raw)
    if not isinstance(frozen, dict):
        raise ValueError(f"invalid {case_id} input freeze")
    reported = check.get(f"{prefix}_inputs")
    hashes = record["hashes"]
    consumers = frozen.get("consumer_files")
    configured_consumers = numerical.BOUND_CHECKS[case_id][2]
    expected_consumers = {
        name: hashes["source_files"].get(name)
        for name in configured_consumers
    }
    staged_binding_digest = hashes.get(f"{prefix}_binding")
    # E02 predates staged-byte retention. Preserve that legacy wire contract;
    # newer bound checks must re-read the exact staged bytes at acceptance.
    if case_id != "E02":
        try:
            staged_binding_digest = hashlib.sha256(
                _read(output / f"{prefix}-input/binding.json",
                      8192)).hexdigest()
        except (OSError, ValueError):
            raise ValueError(
                f"{case_id} input freeze is incomplete or unbound") from None
    if (not isinstance(reported, dict) or reported.get("unchanged") is not True
            or reported.get("manifest_sha256") != digest or any(
                reported.get(key) != value for key, value in frozen.items())
            or frozen.get("binding_sha256") != hashes.get(f"{prefix}_binding")
            or staged_binding_digest != hashes.get(f"{prefix}_binding")
            or frozen.get("captured_inventory_sha256") != _digest(
                _inventory(repository))
            or frozen.get("checks_inventory_sha256")
            != hashes.get("checks_inventory")
            or frozen.get("checks_files") != hashes.get("checks_files")
            or consumers != expected_consumers
            or any(value is None for value in consumers.values())):
        raise ValueError(f"{case_id} input freeze is incomplete or unbound")
    hashes[f"{prefix}_inputs_manifest"] = digest
    hashes[f"{prefix}_captured_inventory"] = frozen[
        "captured_inventory_sha256"]


def _bind_e02_inputs(check: dict, output: Path, repository: Path,
                     record: dict) -> None:
    """Backward-compatible internal entry point for the E02 freeze."""
    _bind_inputs(check, output, repository, record, case_id="E02")


def check_attempt(run_dir: str | Path,
                  attempt_id: str,
                  *,
                  e02_binding: Path | None = None,
                  e05_binding: Path | None = None,
                  e08_binding: Path | None = None) -> dict:
    """Check one terminal attempt once; missing/unsafe evidence remains unknown.

    A preexisting posthoc attempt directory is retained and never reused as a
    fresh result. Its returned refusal record is not written over existing data.
    """
    root = Path(run_dir).absolute()
    record = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "status": "unknown",
        "capture_status": "unknown",
        "capture_selected_path": None,
        "reconstruction_status": "unknown",
        "worker_execution_status": "unknown",
        "targeted": "unknown",
        "regression": "unknown",
        "failures": [],
        "hashes": {},
        "worker_metric": False
    }
    try:
        if not isinstance(attempt_id, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,128}", attempt_id):
            raise ValueError("invalid attempt ID")
        manifest = _manifest(root)
        matches = [
            a for a in manifest["attempts"]
            if a.get("attempt_id") == attempt_id
        ]
        if len(matches) != 1:
            raise ValueError("unknown or duplicate attempt")
        attempt = matches[0]
        trial, result = _terminal_trial(root, attempt)
    except (OSError, ValueError, KeyError, TypeError):
        record["failures"].append("terminal_trial_or_manifest_unavailable")
        return record
    record["case_id"] = attempt.get("case_id")
    record["worker_execution_status"] = "failed" if result.get(
        "exception_info") else "complete"
    registered = record["case_id"] in numerical.CHECK_SUITES
    if not registered:
        record["targeted"] = record["regression"] = "not_applicable"
    supplied_bindings = {
        "E02": e02_binding,
        "E05": e05_binding,
        "E08": e08_binding
    }
    if sum(value is not None for value in supplied_bindings.values()) > 1:
        record["failures"].append("multiple_bindings_not_applicable")
        return record
    for binding_case, supplied in supplied_bindings.items():
        if supplied is not None and record["case_id"] != binding_case:
            record["failures"].append(binding_case.lower() +
                                      "_binding_not_applicable")
            return record
    output = root / "posthoc" / attempt_id
    try:
        try:
            _fresh_directory(root / "posthoc")
        except FileExistsError:
            pass
        _fresh_directory(output)
    except FileExistsError:
        record["failures"].append("posthoc_output_exists")
        return record
    except (OSError, ValueError):
        record["failures"].append("posthoc_output_unsafe_or_unavailable")
        return record
    try:
        relative = attempt["dataset_dir"]
        capture._relative_parts(relative)
        source = root / relative / attempt_id / "environment/repo"
        source_files = _inventory(source)
        source_digest = _digest(source_files)
        helper_digest = hashlib.sha256(_read(Path(
            capture.__file__))).hexdigest()
        record["hashes"].update(
            source_inventory=source_digest,
            source_files=source_files,
            capture_helper=helper_digest,
            numerical_helper=hashlib.sha256(_read(Path(
                numerical.__file__))).hexdigest(),
            orchestration_helper=hashlib.sha256(_read(
                Path(__file__))).hexdigest(),
            manifest=hashlib.sha256(_read(root / "manifest.json")).hexdigest(),
            trial_result=hashlib.sha256(
                _read(trial / "result.json", 2 * 1024**2)).hexdigest())
        if (source_digest != manifest["input_hashes"]["source_inventory"]
                or helper_digest != manifest["input_hashes"]["capture"]):
            raise ValueError("frozen source or capture helper mismatch")
        if manifest["image_id"] != numerical.IMAGE_ID:
            raise ValueError("frozen checker image mismatch")
        record["source_path"] = str(source)
        binding = None
        if record["case_id"] in numerical.BOUND_CHECKS:
            prefix = record["case_id"].lower()
            supplied_binding = supplied_bindings[record["case_id"]]
            if supplied_binding is None:
                record["failures"].append(prefix + "_binding_unavailable")
            else:
                try:
                    candidate = Path(supplied_binding).absolute()
                    binding_raw = _read(candidate, 8192)
                    declaration = json.loads(binding_raw)
                    if not isinstance(declaration, dict):
                        raise ValueError(
                            f"invalid {record['case_id']} binding")
                    record["hashes"][prefix + "_binding"] = hashlib.sha256(
                        binding_raw).hexdigest()
                    binding = candidate
                except (OSError, ValueError, TypeError, UnicodeError):
                    record["failures"].append(prefix + "_binding_unavailable")
    except (OSError, ValueError, KeyError, TypeError):
        record["failures"].append("frozen_inputs_unverified")
        return _save(output, record)
    original = trial / "verifier/artifacts/capture"
    repository = output / "repository"
    record["original_capture_path"] = str(original)
    selected = original
    try:
        reconstruction = capture.reconstruct(source, original, repository)
    except (OSError, ValueError, KeyError, TypeError):
        record["failures"].append(
            "original_capture_missing_incomplete_or_unverified")
        # Failed input verification creates no repository. Preserve any partial
        # output from a later filesystem failure instead of attempting overwrite.
        if repository.exists() or repository.is_symlink():
            record["failures"].append("partial_reconstruction_retained")
            return _save(output, record)
        try:
            selected = _recover(source, trial / "agent/worktree",
                                output / "recovered", manifest["image_id"],
                                record)
            if selected is None:
                return _save(output, record)
            reconstruction = capture.reconstruct(source, selected, repository)
        except (OSError, ValueError, KeyError, TypeError,
                subprocess.SubprocessError):
            record["failures"].append(
                "recovered_capture_unavailable_or_unverified")
            return _save(output, record)
    record.update(capture_status="complete",
                  capture_selected_path=str(selected),
                  reconstruction_status="complete",
                  reconstruction=reconstruction)
    record["hashes"]["capture_manifest"] = reconstruction[
        "capture_json_sha256"]
    record["hashes"]["worktree_inventory"] = reconstruction[
        "worktree_inventory_sha256"]
    if registered:
        try:
            check_files = _inventory(HERE / "checks")
            record["hashes"].update(checks_inventory=_digest(check_files),
                                    checks_files=check_files,
                                    frozen_tests_inventory=_digest(
                                        _inventory(source / "tests/python")))
        except (OSError, ValueError):
            record["failures"].append("frozen_checks_unavailable")
            return _save(output, record)
        for kind in ("targeted", "regression"):
            count, skips = numerical.CHECK_SUITES[record["case_id"]][kind]
            bound_targeted = record[
                "case_id"] in numerical.BOUND_CHECKS and kind == "targeted"
            if bound_targeted and binding is None:
                continue
            try:
                check = numerical.run_checks(
                    repository,
                    source / "tests/python",
                    HERE / "checks",
                    output / kind,
                    kind=kind,
                    case_id=record["case_id"],
                    binding_path=binding if kind == "targeted" else None)
                record["hashes"][kind + "_result"] = hashlib.sha256(
                    _read(output / kind / "result.json")).hexdigest()
                record["hashes"][kind + "_log"] = hashlib.sha256(
                    _read(output / kind / "pytest.log")).hexdigest()
                if check.get("case_id") != record["case_id"]:
                    record[kind] = "unknown"
                    record["failures"].append(kind + "_case_binding_unknown")
                    continue
                if bound_targeted:
                    prefix = record["case_id"].lower()
                    current_binding_hash = hashlib.sha256(_read(
                        binding, 8192)).hexdigest()
                    if (current_binding_hash
                            != record["hashes"][prefix + "_binding"]
                            or check.get("binding_sha256")
                            != current_binding_hash):
                        record[kind] = "unknown"
                        record["failures"].append(
                            prefix + "_binding_changed_or_unbound")
                        continue
                    _bind_inputs(check,
                                 output / kind,
                                 repository,
                                 record,
                                 case_id=record["case_id"])
                record[kind] = numerical.judge_check(check,
                                                     expected_collected=count,
                                                     allowed_skips=skips)
                if record[kind] == "unknown":
                    record["failures"].append(kind +
                                              "_coverage_or_execution_unknown")
            except (OSError, ValueError, subprocess.SubprocessError):
                record["failures"].append(kind + "_execution_unknown")
    if all(record[kind] in {"pass", "fail", "not_applicable"}
           for kind in ("targeted", "regression")):
        record["status"] = "complete"
    return _save(output, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--attempt-id")
    selection.add_argument("--all-terminal", action="store_true")
    bindings = parser.add_mutually_exclusive_group()
    bindings.add_argument(
        "--e02-binding",
        type=Path,
        help="Trusted evaluator-authored adapter JSON for one E02 attempt")
    bindings.add_argument(
        "--e05-binding",
        type=Path,
        help="Trusted evaluator-authored adapter JSON for one E05 attempt")
    bindings.add_argument(
        "--e08-binding",
        type=Path,
        help="Trusted evaluator-authored adapter JSON for one E08 attempt")
    args = parser.parse_args(argv)
    binding_options = (("E02", args.e02_binding), ("E05", args.e05_binding),
                       ("E08", args.e08_binding))
    selected_case, selected_binding = next(
        ((case_id, value)
         for case_id, value in binding_options if value is not None),
        (None, None))
    if selected_binding is not None:
        if args.all_terminal:
            parser.error(
                f"--{selected_case.lower()}-binding requires --attempt-id")
        try:
            matches = [
                attempt
                for attempt in _manifest(args.run_dir.absolute())["attempts"]
                if attempt.get("attempt_id") == args.attempt_id
            ]
        except (OSError, ValueError, KeyError, TypeError):
            parser.error(f"cannot verify {selected_case} attempt identity")
        if len(matches) != 1 or matches[0].get("case_id") != selected_case:
            parser.error(
                f"--{selected_case.lower()}-binding is valid only for an {selected_case} attempt"
            )
    identities = [args.attempt_id]
    if args.all_terminal:
        identities = []
        for attempt in _manifest(args.run_dir.absolute())["attempts"]:
            try:
                _terminal_trial(args.run_dir.absolute(), attempt)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            identities.append(attempt["attempt_id"])
    records = [
        check_attempt(args.run_dir,
                      identity,
                      e02_binding=args.e02_binding,
                      e05_binding=args.e05_binding,
                      e08_binding=args.e08_binding) for identity in identities
    ]
    print(json.dumps(records, sort_keys=True))
    return 0 if all(record["status"] == "complete"
                    for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
