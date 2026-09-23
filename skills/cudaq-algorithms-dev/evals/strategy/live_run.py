#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, serial NVIDIA pilot/study controller; prepare never calls a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import uuid

import suite

MODEL = "nvidia/nemotron-3-super-120b-a12b"
IMAGE_TAG = "cudaq-strategy-runtime:20260915-1z7jNt-qsp"
IMAGE_ID = "sha256:0390e31ee0dbb81f0ab54919cf88c5440c2ff88fe5e276f7001419aff6fa48cd"
SOURCE_SHA256 = "d014746b7ae312f2af952ffb99170cee5d9cea5b5c7aed2603ce6d10f309a1db"
SKILL_SHA256 = "e884e44be881d460591442e78ea83711a530b4999507d42b663f26988926d54a"
SOURCE_COMMIT = "a3fdbb926e47bdfd8c86f9df11f0ed6f9e7b3eff"
AGENT_IMPORT = "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildCodex"
RESOURCES = {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in frozen input: {relative}")
        if path.is_file():
            files[relative.as_posix()] = sha256(path)
    return files


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value,
                                     sort_keys=True).encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def pilot_attempts() -> list[dict]:
    return [
        dict(case_id=case_id,
             arm=arm,
             repetition=1,
             worker_timeout_s=1200 if case_id == "E03" else 600)
        for case_id in ("T02", "N07", "E01", "E03")
        for arm in ("baseline", "candidate")
    ]


def _preparation_attempts(
        mode: str, worker_timeouts_s: dict[str, int] | None) -> list[dict]:
    if mode == "pilot":
        if worker_timeouts_s is not None:
            raise ValueError(
                "pilot mode has fixed worker budgets; overrides are not supported"
            )
        return pilot_attempts()
    if mode != "disclosed-study":
        raise ValueError("preparation mode must be pilot or disclosed-study")
    cases = suite.load_cases()
    if (not isinstance(worker_timeouts_s, dict)
            or set(worker_timeouts_s) != {case.case_id
                                          for case in cases}):
        raise ValueError(
            "study worker timeout budgets must cover exactly every disclosed case"
        )
    # Harbor stores timeout seconds as floats; keep every declared integer exact.
    if any(
            type(value) is not int or not 0 < value < 2**53
            for value in worker_timeouts_s.values()):
        raise ValueError(
            "worker timeout budgets must be positive safe integer seconds")
    return [
        dict(case_id=a.case_id,
             arm=a.arm,
             repetition=a.repetition,
             worker_timeout_s=worker_timeouts_s[a.case_id])
        for a in suite.build_schedule(cases)
    ]


def worker_instruction(case_id: str, arm: str) -> str:
    payload = suite.worker_payload(case_id, arm)
    return payload["prompt"] + ("\n\n" + payload["treatment_directive"]
                                if "treatment_directive" in payload else "")


def worktree_setup_command(source=Path("/workspace/repo"),
                           logs=Path("/logs/agent"),
                           project=Path("/workspace/project"),
                           *,
                           python_site=None) -> str:
    """Initialize once, preserving the actual editable tree in Harbor's log mount."""
    script = '''import hashlib, json, pathlib, shutil, site, subprocess
source, logs, project = map(pathlib.Path, PATHS)
sentinel = logs / "worktree-initial.json"
worktree, seed = logs / "worktree", logs / "source-seed"
if sentinel.exists():
    if not (worktree / ".git").is_file() or not project.is_symlink() or project.resolve() != worktree:
        raise RuntimeError("Initialized worktree is missing or replaced")
else:
    if worktree.exists() or seed.exists() or project.is_symlink():
        raise RuntimeError("Refusing incomplete or existing worktree initialization")
    discovery_backup = None
    if project.exists():
        discovery_roots = (".agents/skills", ".claude/commands", ".claude/skills",
                           ".cline/skills", ".codex/skills", ".config/goose/skills",
                           ".config/opencode/skills", ".cursor/skills", ".gemini/skills",
                           ".gemini/extensions", ".opencode/skills", ".qwen/skills")
        allowed = {path for relative in discovery_roots
                   for path in (pathlib.Path(relative), *pathlib.Path(relative).parents)
                   if str(path) != "."}
        for path in project.rglob("*"):
            if path.is_symlink() or not path.is_dir() or path.relative_to(project) not in allowed:
                raise RuntimeError("Refusing unexpected data in initial project directory")
        discovery_backup = project.with_name(project.name + "-runtime-discovery")
        if discovery_backup.exists() or discovery_backup.is_symlink():
            raise RuntimeError("Refusing existing runtime discovery backup")
        shutil.move(str(project), str(discovery_backup))
    logs.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, seed)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(seed), *args], stderr=subprocess.STDOUT).decode().strip()
    git("init", "-q")
    git("config", "user.email", "evaluation@localhost")
    git("config", "user.name", "Evaluation source snapshot")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", "/dev/null")
    git("add", "--force", ".")
    git("commit", "-q", "-m", "Sanitized evaluation source")
    git("worktree", "add", "--detach", str(worktree), "HEAD")
    clean = not subprocess.check_output(["git", "-C", str(worktree), "status", "--porcelain"])
    if not clean:
        raise RuntimeError("Initial worktree is not clean")
    project.symlink_to(worktree, target_is_directory=True)
    if (worktree / "python").is_dir():
        python_site = PYTHON_SITE
        package_root = pathlib.Path(python_site or site.getsitepackages()[0])
        package_root.mkdir(parents=True, exist_ok=True)
        with (package_root / "cudaq_strategy_worktree.pth").open("x") as handle:
            handle.write(str(project / "python") + "\\n")
    files = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(source.rglob("*")) if p.is_file()}
    evidence = {"clean": clean, "detached": True, "seed_commit": git("rev-parse", "HEAD"),
                "source_files": files, "worktree": str(worktree),
                "runtime_discovery_backup": str(discovery_backup) if discovery_backup else None}
    with sentinel.open("x") as handle:
        json.dump(evidence, handle, sort_keys=True)
'''.replace("PATHS",
            repr([str(source), str(logs), str(project)
                  ])).replace("PYTHON_SITE",
                              repr(str(python_site) if python_site else None))
    return "python3 -c " + shlex.quote(script)


def _extract(archive: Path, destination: Path, *, skill: bool) -> None:
    destination.mkdir()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"unsafe archive member: {member.name}")
            if any(part in
                   {".git", "evals", "__pycache__", ".agents", ".codex"}
                   for part in path.parts):
                raise ValueError(
                    f"private or cached archive member: {member.name}")
            if not skill and ("skills" in path.parts
                              or path.name == "SKILL.md"):
                raise ValueError(
                    f"source archive contains skill: {member.name}")
        handle.extractall(destination, filter="data")


def runtime_identity() -> dict:
    """Hash actual loaded modules, including a child import under the own launcher interpreter."""
    names = [
        "skillevaluator.tier3.harbor." + name
        for name in ("adapter", "runner", "local_agents",
                     "nvidia_build_bridge", "sensitive_stdin",
                     "artifact_retention", "secure_docker_environment")
    ]
    names += ["harbor.agents.installed.codex", "harbor.cli.main"]
    modules = {
        name: sha256(Path(importlib.import_module(name).__file__))
        for name in names
    }
    runner = importlib.import_module("skillevaluator.tier3.harbor.runner")
    launcher = Path(runner._harbor_bin()).absolute()
    if launcher != Path(sys.executable).absolute().parent / "harbor":
        raise ValueError(
            "Harbor launcher does not belong to the active interpreter")
    shebang = launcher.read_text().splitlines()[0]
    if shebang != "#!" + sys.executable:
        raise ValueError("Harbor launcher interpreter mismatch")
    child_code = (
        "import importlib,importlib.metadata,hashlib,json; "
        f"names={names!r}; "
        "print(json.dumps({n:hashlib.sha256(open(importlib.import_module(n).__file__,'rb').read()).hexdigest() for n in names},sort_keys=True))"
    )
    child = subprocess.run([sys.executable, "-c", child_code],
                           check=True,
                           capture_output=True,
                           text=True,
                           timeout=60)
    if json.loads(child.stdout) != modules:
        raise ValueError("Harbor child module hashes differ from controller")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("harbor", "skillevaluator")
    }
    if versions != {"harbor": "0.13.2", "skillevaluator": "0.2.1"}:
        raise ValueError(f"unsupported evaluator/Harbor versions: {versions}")
    return {
        "versions": versions,
        "modules": modules,
        "launcher_sha256": sha256(launcher),
        "python": sys.version.split()[0]
    }


def prepare(run_dir: Path,
            *,
            source_archive: Path,
            skill_archive: Path,
            capture_path: Path = HERE / "capture.py",
            codex_version: str = "0.154.0",
            expected_source_sha256: str = SOURCE_SHA256,
            expected_skill_sha256: str = SKILL_SHA256,
            mode: str = "pilot",
            worker_timeouts_s: dict[str, int] | None = None) -> dict:
    attempts = _preparation_attempts(mode, worker_timeouts_s)
    schedule_hash = _digest(attempts)
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    root = Path(run_dir).absolute()
    if root.exists():
        raise FileExistsError(root)
    for label, path, expected in (("source", source_archive,
                                   expected_source_sha256),
                                  ("skill", skill_archive,
                                   expected_skill_sha256)):
        if sha256(path) != expected:
            raise ValueError(f"{label} archive hash does not match freeze")
    if not re.fullmatch(r"\d+\.\d+\.\d+", codex_version):
        raise ValueError("Codex version must be an exact numeric release")
    identity = runtime_identity()
    root.mkdir(mode=0o700)
    (root / "jobs").mkdir()
    (root / "reservations").mkdir()
    (root / "outcomes").mkdir()
    seed = root / "source"
    _extract(Path(source_archive), seed, skill=False)
    source_files = inventory(seed)
    for args in (("init", "-q"), ("config", "user.email",
                                  "evaluation@localhost"),
                 ("config", "user.name", "Evaluation source snapshot"),
                 ("config", "commit.gpgsign",
                  "false"), ("config", "core.hooksPath", "/dev/null"),
                 ("add", "--force", "."), ("commit", "-q", "-m",
                                           "Sanitized evaluation source")):
        subprocess.run(["git", "-C", str(seed), *args],
                       check=True,
                       capture_output=True)
    anchor = seed / ".runtime-skill"
    _extract(Path(skill_archive), anchor, skill=True)
    skill_path = anchor / "cudaq-algorithms"
    skill_files = inventory(skill_path)
    if not (skill_path / "SKILL.md").is_file():
        raise ValueError(
            "candidate archive must contain cudaq-algorithms/SKILL.md")
    frozen_hashes = {
        "source_archive": expected_source_sha256,
        "skill_archive": expected_skill_sha256,
        "source_inventory": _digest(source_files),
        "skill_inventory": _digest(skill_files),
        "capture": sha256(capture_path),
        "suite": sha256(HERE / "suite.py"),
        "brief": sha256(suite.BRIEF),
        "controller": sha256(Path(__file__))
    }
    if mode == "disclosed-study":
        frozen_hashes["schedule"] = schedule_hash
    for attempt in attempts:
        opaque = uuid.uuid4().hex
        dataset_relative = f"datasets/{opaque}"
        evaluator = root / "private-evaluator" / opaque
        evals = evaluator / "evals"
        evals.mkdir(parents=True)
        instruction = worker_instruction(attempt["case_id"], attempt["arm"])
        _write_json(evals / "evals.json", [{
            "id": opaque,
            "question": instruction
        }])
        shutil.copyfile(capture_path, evals / "grader.py")
        tasks = generate_harbor_tasks(
            skill_path,
            root / dataset_relative,
            with_skill=attempt["arm"] == "candidate",
            evaluator_skill_path=evaluator,
            grading_mode="custom_only",
            copy_repo=True,
            repo_context_exclude_paths=(anchor, ),
            base_image=IMAGE_TAG,
            agent_workdir="/workspace/project",
            pre_agent_setup=[worktree_setup_command()],
            task_resources=RESOURCES,
            runtime_env={"CUDAQ_DEFAULT_SIMULATOR": "qpp-cpu"},
            verifier_env={},
        )
        if len(tasks) != 1 or tasks[0].name != opaque:
            raise ValueError(
                "adapter did not generate exactly one opaque task")
        task = tasks[0]
        config_path = task / "task.toml"
        config = config_path.read_text()
        for table, timeout in (("agent", attempt["worker_timeout_s"]),
                               ("verifier", 120)):
            config, count = re.subn(rf"(\[{table}\]\ntimeout_sec = )[^\n]+",
                                    rf"\g<1>{timeout}.0", config)
            if count != 1:
                raise ValueError(
                    f"unexpected adapter {table} timeout contract")
        tomllib.loads(config)
        config_path.write_text(config)
        if inventory(task / "environment/repo") != source_files:
            raise ValueError(
                "staged repository projection differs from frozen source")
        attempt.update(attempt_id=opaque,
                       dataset_dir=dataset_relative,
                       job_name=opaque,
                       input_hashes={
                           **frozen_hashes, "instruction":
                           hashlib.sha256(instruction.encode()).hexdigest(),
                           "task":
                           _digest(inventory(root / dataset_relative))
                       })
    manifest = {
        "schema_version": 1,
        "purpose": "readiness-pilot" if mode == "pilot" else "disclosed-study",
        "model": MODEL,
        "source_commit": SOURCE_COMMIT,
        "image_tag": IMAGE_TAG,
        "image_id": IMAGE_ID,
        "codex_version": codex_version,
        "resources": RESOURCES,
        "capture_source": str(Path(capture_path).resolve()),
        "verifier_timeout_s": 120,
        "runtime": identity,
        "input_hashes": frozen_hashes,
        "attempts": attempts
    }
    if mode == "disclosed-study":
        manifest["worker_timeouts_s"] = {
            a["case_id"]: a["worker_timeout_s"]
            for a in attempts
        }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "manifest.sha256.json",
                {"sha256": sha256(root / "manifest.json")})
    return manifest


def _manifest(root: Path) -> dict:
    expected = json.loads(
        (root / "manifest.sha256.json").read_text())["sha256"]
    if sha256(root / "manifest.json") != expected:
        raise ValueError("manifest hash changed")
    return json.loads((root / "manifest.json").read_text())


def preflight(run_dir: Path, *, check_runtime: bool = True) -> dict:
    root = Path(run_dir).absolute()
    manifest = _manifest(root)
    helpers = {
        "controller": Path(__file__),
        "suite": HERE / "suite.py",
        "brief": suite.BRIEF,
        "capture": Path(manifest["capture_source"])
    }
    for label, path in helpers.items():
        if sha256(path) != manifest["input_hashes"][label]:
            raise ValueError(f"frozen helper {label} hash changed")
    purpose = manifest.get("purpose")
    if purpose == "disclosed-study":
        expected = _preparation_attempts("disclosed-study",
                                         manifest.get("worker_timeouts_s"))
    elif purpose == "readiness-pilot":
        if "worker_timeouts_s" in manifest or "schedule" in manifest[
                "input_hashes"]:
            raise ValueError(
                "pilot schedule cannot contain full-study budget metadata")
        expected = pilot_attempts()
    else:
        raise ValueError("unknown frozen schedule purpose")
    actual = [{
        key: a.get(key)
        for key in ("case_id", "arm", "repetition", "worker_timeout_s")
    } for a in manifest["attempts"]]
    # JSON hashing also distinguishes True/1.0 from the integer 1.
    actual_hash = _digest(actual)
    if (actual_hash != _digest(expected)
            or (purpose == "disclosed-study"
                and manifest["input_hashes"].get("schedule") != actual_hash)):
        raise ValueError(
            "frozen schedule or worker budget differs from the contract")
    for attempt in manifest["attempts"]:
        dataset = root / attempt["dataset_dir"]
        if _digest(inventory(dataset)) != attempt["input_hashes"]["task"]:
            raise ValueError(
                f"frozen task hash changed: {attempt['attempt_id']}")
        task = dataset / attempt["attempt_id"]
        instruction = worker_instruction(attempt["case_id"], attempt["arm"])
        entry = json.loads((task / "tests/entry.json").read_text())
        if (entry.get("id") != attempt["attempt_id"]
                or entry.get("question") != instruction
                or (task / "instruction.md").read_text() != instruction + "\n"
                or attempt["input_hashes"].get("instruction")
                != hashlib.sha256(instruction.encode()).hexdigest()):
            raise ValueError(
                "staged entry/instruction does not match its declared case and arm"
            )
        config = tomllib.loads((task / "task.toml").read_text())
        if (config["agent"]["timeout_sec"] != attempt["worker_timeout_s"]
                or config["verifier"]["timeout_sec"] != 120):
            raise ValueError("task timeout differs from the protocol")
        source = inventory(task / "environment/repo")
        if _digest(source) != manifest["input_hashes"]["source_inventory"]:
            raise ValueError("repository projection hash mismatch")
        skills = inventory(task / "environment/skills")
        if attempt["arm"] == "baseline" and skills:
            raise ValueError("baseline contains a skill")
        if attempt["arm"] == "candidate":
            candidate = inventory(task / "environment/skills/cudaq-algorithms")
            if _digest(
                    candidate) != manifest["input_hashes"]["skill_inventory"]:
                raise ValueError("candidate skill projection hash mismatch")
    if check_runtime:
        if runtime_identity() != manifest["runtime"]:
            raise ValueError("runtime version/module hash changed")
        image = subprocess.run([
            "docker", "image", "inspect", manifest["image_tag"], "--format",
            "{{.Id}}"
        ],
                               check=True,
                               capture_output=True,
                               text=True,
                               timeout=30)
        if image.stdout.strip() != manifest["image_id"]:
            raise ValueError("runtime image ID differs from freeze")
    return {
        "status": "ready",
        "attempts": len(manifest["attempts"]),
        "model_calls": 0,
        "runtime_checked": check_runtime
    }


def reserve_attempt(run_dir: Path, attempt_id: str) -> Path:
    root = Path(run_dir).absolute()
    attempts = _manifest(root)["attempts"]
    selected = next(
        (i for i, a in enumerate(attempts) if a["attempt_id"] == attempt_id),
        None)
    if selected is None:
        raise ValueError("unknown attempt ID")
    path = root / "reservations" / f"{attempt_id}.json"
    if path.exists():
        raise FileExistsError(f"attempt already reserved: {attempt_id}")
    for previous in attempts[:selected]:
        if not (root / "outcomes" / f"{previous['attempt_id']}.json").exists():
            raise RuntimeError(
                "attempt order requires every previous outcome first")
    _write_json(
        path, {
            "attempt_id": attempt_id,
            "reserved_at": datetime.now(timezone.utc).isoformat()
        })
    return path


def run(run_dir: Path,
        *,
        api_key: str | None = None,
        attempt_id: str | None = None) -> list[dict]:
    """Run every selected attempt once; retain failures and stop on interruption.

    Supplying an attempt ID lets the controller capture/check each result before
    launching the next one. The complete schedule always remains baseline first.
    """
    from skillevaluator.tier3.harbor import runner

    root = Path(run_dir).absolute()
    preflight(root)
    manifest = _manifest(root)
    api_key = api_key if api_key is not None else os.environ.get(
        "NVIDIA_API_KEY", "")
    if not api_key or "\n" in api_key:
        raise ValueError(
            "a nonempty NVIDIA_API_KEY is required at execution only")
    attempts = [
        a for a in manifest["attempts"]
        if attempt_id is None or a["attempt_id"] == attempt_id
    ]
    if not attempts:
        raise ValueError("unknown attempt ID")
    allowed_env = {
        "PATH", "HOME", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV", "LANG",
        "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"
    }
    run_env = {
        key: value
        for key, value in os.environ.items() if key in allowed_env
    }
    run_env.update(NVIDIA_API_KEY=api_key, SKILL_EVAL_LLM_PROVIDER="nv_build")
    original_builder = runner.build_harbor_run_command

    def pinned_command(**kwargs):
        return original_builder(**kwargs) + [
            "--agent-kwarg", f"version={manifest['codex_version']}",
            "--max-retries", "0"
        ]

    results = []
    try:
        runner.build_harbor_run_command = pinned_command
        for attempt in attempts:
            reserve_attempt(root, attempt["attempt_id"])
            started = time.monotonic()
            outcome = {
                "attempt_id": attempt["attempt_id"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "interrupted",
                "detail": "Worker interrupted before a terminal result"
            }
            try:
                ok, detail = runner._run_harbor(
                    dataset=root / attempt["dataset_dir"],
                    agent="codex",
                    job_name=attempt["job_name"],
                    env_mode="docker",
                    model=MODEL,
                    jobs_dir=root / "jobs",
                    run_env=run_env,
                    n_attempts=1,
                    n_concurrent=1,
                    timeout_multiplier=1.0,
                    override_cpus=None,
                    override_memory_mb=None,
                    override_storage_mb=None,
                    agent_import_path=AGENT_IMPORT,
                    verifier_env={},
                    expected_trials=1,
                    expected_total_trials=1,
                )
                outcome.update(status="completed" if ok else "failed",
                               detail=detail.replace(api_key, "[REDACTED]"))
            except Exception as exc:
                # Preserve the failure without copying arbitrary exception text,
                # which can contain process input or credentials.
                outcome.update(status="failed", detail=type(exc).__name__)
            finally:
                outcome["controller_elapsed_s"] = time.monotonic() - started
                outcome["finished_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(
                    root / "outcomes" / f"{attempt['attempt_id']}.json",
                    outcome)
            results.append(outcome)
    finally:
        runner.build_harbor_run_command = original_builder
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser(
        "prepare", help="Freeze private tasks without model calls")
    prepare_parser.add_argument("run_dir", type=Path)
    prepare_parser.add_argument("--source-archive", type=Path, required=True)
    prepare_parser.add_argument("--skill-archive", type=Path, required=True)
    prepare_parser.add_argument("--capture-path",
                                type=Path,
                                default=HERE / "capture.py")
    prepare_parser.add_argument("--codex-version", default="0.154.0")
    prepare_parser.add_argument("--mode",
                                choices=("pilot", "disclosed-study"),
                                default="pilot")
    prepare_parser.add_argument(
        "--worker-timeouts",
        type=Path,
        help=
        "JSON object of all disclosed case IDs to integer seconds; required for study mode"
    )
    preflight_parser = actions.add_parser(
        "preflight", help="Check hashes, isolation and local runtime")
    preflight_parser.add_argument("run_dir", type=Path)
    preflight_parser.add_argument("--staging-only", action="store_true")
    run_parser = actions.add_parser(
        "run", help="Run selected attempts once using NVIDIA_API_KEY")
    run_parser.add_argument("run_dir", type=Path)
    run_parser.add_argument("--attempt-id")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        budgets = json.loads(
            args.worker_timeouts.read_text()) if args.worker_timeouts else None
        value = prepare(args.run_dir,
                        source_archive=args.source_archive,
                        skill_archive=args.skill_archive,
                        capture_path=args.capture_path,
                        codex_version=args.codex_version,
                        mode=args.mode,
                        worker_timeouts_s=budgets)
    elif args.action == "preflight":
        value = preflight(args.run_dir, check_runtime=not args.staging_only)
    else:
        value = run(args.run_dir, attempt_id=args.attempt_id)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
