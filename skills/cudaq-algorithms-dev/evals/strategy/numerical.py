#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run captured implementations against private and frozen tests, without credentials.

Inputs must be a verified reconstruction and trusted evaluator-owned tests.
Never mount live worker logs or an API key. This is post-hoc checking time,
not worker time. A zero exit without a pytest result is not a pass.
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

IMAGE_ID = "sha256:0390e31ee0dbb81f0ab54919cf88c5440c2ff88fe5e276f7001419aff6fa48cd"
MAX_OUTPUT = 128 * 1024
PROJECTION_LIMITS = {
    "max_file_bytes": capture.MAX_FILE_BYTES,
    "max_tree_bytes": capture.MAX_TREE_BYTES,
    "max_entries": capture.MAX_ENTRIES
}
PRIVATE_SOURCE_PARTS = {
    "evals", "skills", "SKILL.md", ".runtime-skill", ".agents", ".codex",
    ".claude", ".ssh", "secrets"
}
CHECK_SUITES = {
    "E02": {
        "targeted": (60, 0),
        "regression": (311, 4)
    },
    "E03": {
        "targeted": (32, 0),
        "regression": (311, 4)
    },
    "E05": {
        "targeted": (13, 0),
        "regression": (311, 4)
    },
    "E08": {
        "targeted": (12, 0),
        "regression": (311, 4)
    },
}
# E02 uses these public consumers/helpers as part of its independent oracle.
# Changes require review; refusing automatic certification is not a code failure.
E02_CONSUMERS = tuple(f"python/cudaq_algorithms/{name}.py"
                      for name in ("qubitization", "qsvt", "common_kernels",
                                   "sim_utils", "pauli_lcu", "block_encoding"))
E05_CONSUMERS = tuple(f"python/cudaq_algorithms/{name}.py"
                      for name in ("qubitization", "pauli_lcu",
                                   "common_kernels", "block_encoding"))
E08_CONSUMERS = tuple(f"python/cudaq_algorithms/{name}.py"
                      for name in ("qubitization", "qsvt", "common_kernels",
                                   "sim_utils", "pauli_lcu", "block_encoding"))
BOUND_CHECKS = {
    "E02": ("test_scale_encoding.py", "E02_SCALE_ADAPTER", E02_CONSUMERS),
    "E05": ("test_ci_preparation.py", "E05_CI_ADAPTER", E05_CONSUMERS),
    "E08":
    ("test_evolution_example.py", "E08_EVOLUTION_ADAPTER", E08_CONSUMERS),
}


def _mount(path: Path, target: str) -> str:
    path = Path(path).absolute()
    if any(part == ".." for part in path.parts) or any(c in str(path)
                                                       for c in ",\r\n"):
        raise ValueError("unsupported mount path")
    for parent in [*reversed(path.parents), path]:
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError("unsafe mount directory")
    return f"type=bind,source={path},target={target},readonly"


def command(repository: Path,
            frozen_tests: Path,
            checks: Path,
            *,
            kind: str,
            name: str,
            image_id: str = IMAGE_ID,
            timeout_s: int = 900,
            case_id: str = "E03",
            binding_dir: Path | None = None) -> list[str]:
    if case_id not in CHECK_SUITES or kind not in CHECK_SUITES[case_id]:
        raise ValueError("unsupported check case or kind")
    bound = case_id in BOUND_CHECKS and kind == "targeted"
    if bound != (binding_dir is not None):
        raise ValueError(
            "bound targeted checks require an explicit binding directory; other checks forbid it"
        )
    if not re.fullmatch(r"cudaq-check-[a-z0-9-]{1,64}", name):
        raise ValueError("invalid private container name")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("an immutable local image ID is required")
    if type(timeout_s) is not int or not 1 <= timeout_s <= 1200:
        raise ValueError("invalid check timeout")
    frozen_tests = Path(frozen_tests).absolute()
    if frozen_tests.name != "python" or frozen_tests.parent.name != "tests":
        raise ValueError(
            "frozen tests must retain the source tests/python layout")
    targeted_file = BOUND_CHECKS[case_id][
        0] if bound else "test_protocol_action.py"
    selection = (
        f"--confcutdir=/checks /checks/{targeted_file}"
        if kind == "targeted" else
        "--confcutdir=/frozen-source/tests/python /frozen-source/tests/python")
    prefix = case_id.lower()
    binding_options = ([
        "--mount",
        _mount(binding_dir, f"/{prefix}-input"), "--env",
        f"{BOUND_CHECKS[case_id][1]}=/{prefix}-input/binding.json", "--env",
        f"{case_id}_ARTIFACT_ROOT=/workspace/project"
    ] if bound else [])
    # Bound output inside the container before the Docker client receives it.
    # The executable tmpfs is needed by the CUDA-Q JIT; no network or secrets.
    wrapper = (f"ulimit -f 2048; timeout -s KILL {timeout_s}s "
               f"python3 -m pytest -q -p no:cacheprovider {selection} "
               ">/tmp/checks.log 2>&1; code=$?; "
               f"head -c {MAX_OUTPUT + 1} /tmp/checks.log; exit \"$code\"")
    return [
        "docker", "run", "--rm", "--name", name, "--pull", "never",
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--pids-limit", "256", "--memory", "4g", "--cpus", "2", "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=1g,mode=1777", "--env",
        "CUDAQ_DEFAULT_SIMULATOR=qpp-cpu", "--env",
        "PYTHONPATH=/workspace/project/python", "--env",
        "PYTHONDONTWRITEBYTECODE=1", "--env",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "--env", "OMP_NUM_THREADS=2",
        "--env", "OPENBLAS_NUM_THREADS=2", "--env",
        "XDG_CACHE_HOME=/tmp/cache", "--mount",
        _mount(repository, "/workspace/project"), "--mount",
        _mount(frozen_tests.parent.parent, "/frozen-source"), "--mount",
        _mount(checks, "/checks"), *binding_options, "--workdir",
        "/workspace/project", image_id, "sh", "-c", wrapper
    ]


def classify(returncode: int | None, output: bytes) -> dict:
    counts = {
        label: int(count)
        for count, label in re.findall(
            r"\b(\d+) (passed|failed|skipped|error|errors)\b",
            output.decode("utf-8", errors="replace"))
    }
    if len(output) > MAX_OUTPUT:
        status = "output_over_limit"
    elif returncode in {None, 124, 137}:
        status = "timeout"
    elif returncode == 0 and counts.get("passed", 0) and not any(
            counts.get(key, 0) for key in ("failed", "error", "errors")):
        status = "passed"
    elif returncode == 1 and counts.get("failed", 0):
        status = "failed"
    else:
        status = "invalid_output" if returncode == 0 else "execution_error"
    return {
        "status": status,
        "returncode": returncode,
        "pytest_counts": counts,
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest()
    }


def judge_check(result: dict,
                *,
                expected_collected: int,
                allowed_skips: int = 0) -> str:
    """Map checker outcomes without accepting skipped or incomplete coverage.

    Freeze these counts from the independent test inventory before attempts:
    E03 targeted=32/no skips; existing suite=311/up to four documented skips.
    A pytest failure is a failure; missing coverage is unknown, never a pass.
    """
    if result.get("status") == "failed":
        return "fail"
    counts = result.get("pytest_counts", {})
    if (result.get("status") != "passed"
            or sum(counts.values()) != expected_collected
            or counts.get("skipped", 0) > allowed_skips
            or counts.get("passed", 0) < expected_collected - allowed_skips):
        return "unknown"
    return "pass"


def _public_source_snapshot(source: Path) -> dict:
    snapshot = capture._snapshot(source, "frozen_public_source",
                                 PROJECTION_LIMITS)
    if snapshot["rejections"]:
        raise ValueError(
            "frozen public source contains unsafe or incomplete entries")
    if any(
            any(part in PRIVATE_SOURCE_PARTS or part.startswith(".env")
                for part in Path(name).parts)
            for name in snapshot["inventory"]):
        raise ValueError(
            "frozen public source contains private evaluation or skill files")
    return snapshot


def _content_inventory(snapshot: dict) -> tuple[dict, str]:
    files = {
        name: item["sha256"]
        for name, item in snapshot["inventory"].items()
    }
    digest = hashlib.sha256(json.dumps(files,
                                       sort_keys=True).encode()).hexdigest()
    return files, digest


def _readable_projection(source: Path,
                         output: Path) -> tuple[dict, dict, dict]:
    """Copy bounded public bytes, never chmod or import the frozen source.

    Keep the full source layout: tests locate documentation examples relative
    to tests/python. The reference python/ directory is not added to imports;
    command() continues to select only the captured repository's PYTHONPATH.
    """
    original = _public_source_snapshot(source)
    directory = capture._directory(output)
    try:
        os.mkdir("frozen-source", 0o755, dir_fd=directory)
        projected_fd = os.open("frozen-source",
                               capture.DIRECTORY_FLAGS,
                               dir_fd=directory)
        try:
            # mkdir/open modes are filtered by the controller's umask. Set
            # modes explicitly only on newly created evaluator-owned copies.
            os.fchmod(projected_fd, 0o755)
            for name, content in original["contents"].items():
                capture._write(projected_fd, name, content)
                parent = os.dup(projected_fd)
                try:
                    parts = Path(name).parts
                    for component in parts[:-1]:
                        child = os.open(component,
                                        capture.DIRECTORY_FLAGS,
                                        dir_fd=parent)
                        os.close(parent)
                        parent = child
                        os.fchmod(parent, 0o755)
                    descriptor = os.open(parts[-1],
                                         os.O_RDONLY | os.O_NOFOLLOW,
                                         dir_fd=parent)
                    try:
                        os.fchmod(descriptor, 0o644)
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(parent)
        finally:
            os.close(projected_fd)
        projected = _public_source_snapshot(output / "frozen-source")
        files, source_digest = _content_inventory(original)
        projected_files, projected_digest = _content_inventory(projected)
        if files != projected_files:
            raise ValueError(
                "readable projection differs from frozen source bytes")
        if _public_source_snapshot(
                source)["inventory"] != original["inventory"]:
            raise ValueError(
                "frozen source changed while building readable projection")
        record = {
            "policy":
            "readable-public-source-v1",
            "source_root":
            str(source),
            "projection_root":
            str(output / "frozen-source"),
            "source_files":
            files,
            "file_count":
            len(files),
            "source_inventory_sha256":
            source_digest,
            "projection_inventory_sha256":
            projected_digest,
            "source_metadata_sha256":
            hashlib.sha256(capture._json(original["inventory"])).hexdigest(),
            "projection_metadata_sha256":
            hashlib.sha256(capture._json(projected["inventory"])).hexdigest(),
            "capture_helper_sha256":
            hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest(),
            "directory_mode":
            "0755",
            "file_mode":
            "0644",
            "outer_output_mode":
            "0700",
            "exclusions":
            original["exclusions"],
            "import_policy":
            "PYTHONPATH=/workspace/project/python; reference python/ is not added"
        }
        metadata = capture._json(record)
        capture._write(directory, "frozen-source-projection.json", metadata)
        record["manifest_sha256"] = hashlib.sha256(metadata).hexdigest()
        return original, projected, record
    finally:
        os.close(directory)


def _binding_bytes(path: Path) -> bytes:
    path = Path(path).absolute()
    directory = capture._directory(path.parent)
    try:
        return capture._read(directory, path.name, 8192)
    finally:
        os.close(directory)


def _bound_inputs(repository: Path, source: Path, checks: Path,
                  binding_path: Path, *, case_id: str) -> tuple[dict, bytes]:
    """Freeze bytes only; never import or invoke the captured API on the host."""
    raw = _binding_bytes(binding_path)
    captured = _public_source_snapshot(repository)
    original = _public_source_snapshot(source)
    oracle = _public_source_snapshot(checks)
    worker_files, worker_digest = _content_inventory(captured)
    source_files, _ = _content_inventory(original)
    check_files, check_digest = _content_inventory(oracle)
    consumers = BOUND_CHECKS[case_id][2]
    if any(name not in source_files
           or worker_files.get(name) != source_files[name]
           for name in consumers):
        raise ValueError(
            f"{case_id} consumer source is missing or changed; independent review required"
        )
    return {
        "binding_sha256": hashlib.sha256(raw).hexdigest(),
        "captured_inventory_sha256": worker_digest,
        "checks_inventory_sha256": check_digest,
        "checks_files": check_files,
        "consumer_files": {
            name: source_files[name]
            for name in consumers
        }
    }, raw


def _stage_binding_inputs(output: Path, raw: bytes, frozen: dict, *,
                          case_id: str) -> str:
    prefix = case_id.lower()
    directory = capture._directory(output)
    try:
        os.mkdir(f"{prefix}-input", 0o755, dir_fd=directory)
        binding = os.open(f"{prefix}-input",
                          capture.DIRECTORY_FLAGS,
                          dir_fd=directory)
        try:
            os.fchmod(binding, 0o755)
            capture._write(binding, "binding.json", raw)
            file = os.open("binding.json",
                           os.O_RDONLY | os.O_NOFOLLOW,
                           dir_fd=binding)
            try:
                os.fchmod(file, 0o644)
            finally:
                os.close(file)
        finally:
            os.close(binding)
        if _binding_bytes(output / f"{prefix}-input/binding.json") != raw:
            raise ValueError(
                f"{case_id} staged binding changed while freezing inputs")
        encoded = capture._json(frozen)
        capture._write(directory, f"{prefix}-inputs.json", encoded)
        return hashlib.sha256(encoded).hexdigest()
    finally:
        os.close(directory)


def run_checks(repository: Path,
               frozen_tests: Path,
               checks: Path,
               output: Path,
               *,
               kind: str,
               timeout_s: int = 900,
               case_id: str = "E03",
               binding_path: Path | None = None) -> dict:
    name = "cudaq-check-" + uuid.uuid4().hex
    bound = case_id in BOUND_CHECKS and kind == "targeted"
    prefix = case_id.lower()
    if bound != (binding_path is not None):
        raise ValueError(
            "bound targeted checks require an explicit binding file; other checks forbid it"
        )
    # Validate the unchanged command API before creating any output.
    command(
        repository,
        frozen_tests,
        checks,
        kind=kind,
        name=name,
        timeout_s=timeout_s,
        case_id=case_id,
        binding_dir=Path(binding_path).absolute().parent if bound else None)
    output = Path(output).absolute()
    source = Path(frozen_tests).absolute().parent.parent
    for original in (source, Path(repository).absolute(),
                     Path(checks).absolute()):
        if output == original or output in original.parents or original in output.parents:
            raise ValueError("checker output must not overlap input trees")
    frozen, binding_raw = (_bound_inputs(
        repository, source, checks, binding_path, case_id=case_id)
                           if bound else (None, None))
    parent = capture._directory(output.parent)
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent)  # Fresh output required.
    finally:
        os.close(parent)
    original_snapshot, projected_snapshot, projection = _readable_projection(
        source, output)
    freeze_hash = _stage_binding_inputs(
        output, binding_raw, frozen, case_id=case_id) if bound else None
    argv = command(repository,
                   output / "frozen-source/tests/python",
                   checks,
                   kind=kind,
                   name=name,
                   timeout_s=timeout_s,
                   case_id=case_id,
                   binding_dir=output / f"{prefix}-input" if bound else None)
    started = time.monotonic()
    env = {
        key: value
        for key, value in os.environ.items() if key in
        {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"}
    }
    returncode = None
    raw = b""
    try:
        result = subprocess.run(argv,
                                capture_output=True,
                                timeout=timeout_s + 45,
                                env=env)
        returncode, raw = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or b"") + (exc.stderr or b"")
        subprocess.run(["docker", "kill", name],
                       capture_output=True,
                       timeout=20,
                       env=env)
    report = {
        **classify(returncode, raw), "kind": kind,
        "case_id": case_id,
        "image_id": IMAGE_ID,
        "checker_elapsed_s": time.monotonic() - started,
        "worker_metric": False,
        "network": "none",
        "api_credentials_present": False,
        "timeout_s": timeout_s,
        "command": argv
    }
    for key, tree, before in (("source_unchanged", source, original_snapshot),
                              ("projection_unchanged",
                               output / "frozen-source", projected_snapshot)):
        try:
            after = _public_source_snapshot(tree)
            projection[key] = after["inventory"] == before["inventory"]
        except (OSError, ValueError):
            projection[key] = False
    if not projection["source_unchanged"] or not projection[
            "projection_unchanged"]:
        report["status"] = "input_changed"
    if bound:
        try:
            current, current_raw = _bound_inputs(repository,
                                                 source,
                                                 checks,
                                                 binding_path,
                                                 case_id=case_id)
            unchanged = (
                current == frozen and current_raw == binding_raw
                and _binding_bytes(output / f"{prefix}-input/binding.json")
                == binding_raw and hashlib.sha256(
                    _binding_bytes(
                        output / f"{prefix}-inputs.json")).hexdigest()
                == freeze_hash)
        except (OSError, ValueError):
            unchanged = False
        report["binding_sha256"] = frozen["binding_sha256"]
        report[f"{prefix}_inputs"] = {
            **frozen, "manifest_sha256": freeze_hash,
            "unchanged": unchanged
        }
        if not unchanged:
            report["status"] = "input_changed"
    report["frozen_source_projection"] = projection
    (output / "pytest.log").write_bytes(raw)
    (output / "result.json"
     ).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--frozen-tests", required=True, type=Path)
    parser.add_argument("--checks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind",
                        required=True,
                        choices=("targeted", "regression"))
    parser.add_argument("--case-id",
                        choices=tuple(CHECK_SUITES),
                        default="E03")
    parser.add_argument("--binding-path", type=Path)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    print(
        json.dumps(run_checks(args.repository,
                              args.frozen_tests,
                              args.checks,
                              args.output,
                              kind=args.kind,
                              timeout_s=args.timeout_s,
                              case_id=args.case_id,
                              binding_path=args.binding_path),
                   sort_keys=True))


if __name__ == "__main__":
    main()
