# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tomllib

import pytest

STRATEGY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRATEGY))
SPEC = importlib.util.spec_from_file_location("strategy_live_run",
                                              STRATEGY / "live_run.py")
live = None
if SPEC and SPEC.loader and Path(SPEC.origin).exists():
    live = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = live
    SPEC.loader.exec_module(live)


def test_pilot_is_baseline_first_and_only_forces_execution_activation():
    assert live is not None, "live pilot controller has not been implemented"
    attempts = live.pilot_attempts()
    assert [(a["case_id"], a["arm"]) for a in attempts] == [
        ("T02", "baseline"),
        ("T02", "candidate"),
        ("N07", "baseline"),
        ("N07", "candidate"),
        ("E01", "baseline"),
        ("E01", "candidate"),
        ("E03", "baseline"),
        ("E03", "candidate"),
    ]
    assert all(a["repetition"] == 1 for a in attempts)
    assert "Explicitly use" not in live.worker_instruction("T02", "candidate")
    assert live.worker_instruction("E03", "candidate").endswith(
        "Explicitly use the available cudaq-algorithms skill for this task.")
    assert [a["worker_timeout_s"] for a in attempts] == [600] * 6 + [1200] * 2


DISCLOSED_IDS = ([f"T{i:02d}" for i in range(1, 13)] +
                 [f"N{i:02d}"
                  for i in range(1, 11)] + [f"E{i:02d}" for i in range(1, 13)])
REPEATED_IDS = {"E02", "E03", "E04", "E05", "E07", "E08", "E10", "E12"}


def study_budgets():
    # Synthetic, distinct limits expose accidentally inherited pilot defaults.
    return {
        case_id: 601 + index
        for index, case_id in enumerate(DISCLOSED_IDS)
    }


def test_disclosed_study_stages_full_ordered_schedule_without_private_rubric(
        tmp_path, inputs, monkeypatch):
    original_cases = live.suite.load_cases()
    private_marker = "SYNTHETIC_EVALUATOR_ONLY_RUBRIC"
    cases = [
        replace(case,
                expected_behavior=(private_marker, ),
                critical_checks=(private_marker, )) for case in original_cases
    ]
    monkeypatch.setattr(live.suite, "load_cases", lambda: cases)
    budgets = study_budgets()
    root = tmp_path / "full-run"
    manifest = live.prepare(root,
                            **inputs,
                            mode="disclosed-study",
                            worker_timeouts_s=budgets)
    expected = [
        (case_id, arm, repetition, budgets[case_id])
        for case_id in DISCLOSED_IDS
        for repetition in range(1, (3 if case_id in REPEATED_IDS else 1) + 1)
        for arm in ("baseline", "candidate")
    ]
    attempts = manifest["attempts"]
    assert [(a["case_id"], a["arm"], a["repetition"], a["worker_timeout_s"])
            for a in attempts] == expected
    assert len(attempts) == 100
    assert sum(a["case_id"].startswith("E") for a in attempts) == 56
    assert len({a["attempt_id"] for a in attempts}) == 100
    assert manifest["purpose"] == "disclosed-study"
    assert manifest["worker_timeouts_s"] == budgets
    frozen_schedule = [{
        key: a[key]
        for key in ("case_id", "arm", "repetition", "worker_timeout_s")
    } for a in attempts]
    assert manifest["input_hashes"]["schedule"] == hashlib.sha256(
        json.dumps(frozen_schedule, sort_keys=True).encode()).hexdigest()
    for attempt in attempts:
        task = root / attempt["dataset_dir"] / attempt["attempt_id"]
        entry = json.loads((task / "tests/entry.json").read_text())
        assert entry["id"] == attempt["attempt_id"]
        assert entry["question"] == live.worker_instruction(
            attempt["case_id"], attempt["arm"])
        assert private_marker not in json.dumps(entry)
        assert not {"score_rubric", "expected_behavior", "critical_checks"
                    } & entry.keys()
        assert not list((task / "environment/repo").rglob("SKILL.md"))
        assert not (task / "environment/repo/.git").exists()
        assert (task /
                "environment/repo/python/example.py").read_text() == "x = 1\n"
        package = task / "environment/skills/cudaq-algorithms"
        assert package.exists() is (attempt["arm"] == "candidate")
        if package.exists():
            assert (package / "authoring/guide.md"
                    ).read_text() == "Public authoring support\n"
        config = tomllib.loads((task / "task.toml").read_text())
        assert config["agent"]["timeout_sec"] == budgets[attempt["case_id"]]
        assert config["verifier"]["timeout_sec"] == 120
        assert config["environment"]["cpus"] == 2
        assert config["environment"]["memory_mb"] == 4096
        assert config["environment"]["storage_mb"] == 10240
    assert live.preflight(root, check_runtime=False) == {
        "status": "ready",
        "attempts": 100,
        "model_calls": 0,
        "runtime_checked": False
    }
    # Exercise the full manifest through the real collector/reducer without
    # fabricating any worker outcomes or model measurements.
    import posthoc
    collected = posthoc.collect(root)
    assert len(collected["rows"]) == 100
    assert {row["case_id"] for row in collected["rows"]} == set(DISCLOSED_IDS)
    assert all(row["failure_category"] == "missing_result"
               for row in collected["rows"])
    summary = posthoc.judging.summarize(collected["rows"],
                                        pilot=False,
                                        manifest_sha256=live.sha256(
                                            root / "manifest.json"))
    assert summary["successful_matched_pairs"] == 0
    assert summary["efficiency"]["statistical_pass"] is None
    with pytest.raises(RuntimeError, match="order"):
        live.reserve_attempt(root, attempts[-1]["attempt_id"])


@pytest.mark.parametrize("fault", [
    "missing", "omitted", "extra", "zero", "negative", "bool", "float",
    "string", "list", "unsafe_integer", "pilot_override", "unknown_mode"
])
def test_invalid_study_budgets_fail_before_staging_or_runtime(
        tmp_path, inputs, monkeypatch, fault):
    budgets, mode = study_budgets(), "disclosed-study"
    if fault == "missing":
        budgets = None
    elif fault == "omitted":
        del budgets["E12"]
    elif fault == "extra":
        budgets["UNDECLARED_CASE"] = 600
    elif fault == "list":
        budgets = list(budgets)
    elif fault == "pilot_override":
        mode = "pilot"
    elif fault == "unknown_mode":
        mode = "made-up"
    else:
        budgets["E02"] = {
            "zero": 0,
            "negative": -1,
            "bool": True,
            "float": 1.5,
            "string": "600",
            "unsafe_integer": 2**53
        }[fault]

    def no_runtime():
        raise AssertionError("invalid preparation must not inspect runtime")

    monkeypatch.setattr(live, "runtime_identity", no_runtime)
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="budget|timeout|mode"):
        live.prepare(root, **inputs, mode=mode, worker_timeouts_s=budgets)
    assert not root.exists()


def test_full_study_preflight_rejects_internally_inconsistent_schedule(
        tmp_path, inputs):
    root = tmp_path / "full-run"
    manifest = live.prepare(root,
                            **inputs,
                            mode="disclosed-study",
                            worker_timeouts_s=study_budgets())
    for fault in ("reorder", "missing_attempt", "repetition", "arm", "budget",
                  "schedule_hash", "bool_repetition", "float_repetition",
                  "float_timeout"):
        changed = json.loads(json.dumps(manifest))
        if fault == "reorder":
            changed["attempts"][0], changed["attempts"][1] = changed[
                "attempts"][1], changed["attempts"][0]
        elif fault == "missing_attempt":
            changed["attempts"].pop()
        elif fault in ("repetition", "arm"):
            changed["attempts"][0][
                fault] = 4 if fault == "repetition" else "candidate"
        elif fault == "budget":
            changed["worker_timeouts_s"]["E02"] += 1
        elif fault == "schedule_hash":
            changed["input_hashes"]["schedule"] = "0" * 64
        elif fault == "float_timeout":
            changed["attempts"][0]["worker_timeout_s"] = float(
                changed["attempts"][0]["worker_timeout_s"])
        else:
            changed["attempts"][0][
                "repetition"] = True if fault == "bool_repetition" else 1.0
        (root / "manifest.json").write_text(json.dumps(changed))
        (root / "manifest.sha256.json").write_text(
            json.dumps({
                "sha256":
                hashlib.sha256(
                    (root / "manifest.json").read_bytes()).hexdigest()
            }))
        with pytest.raises(ValueError, match="schedule|budget"):
            live.preflight(root, check_runtime=False)


def test_full_study_cannot_bypass_schedule_checks_by_relabeling_purpose(
        tmp_path, inputs):
    root = tmp_path / "full-run"
    manifest = live.prepare(root,
                            **inputs,
                            mode="disclosed-study",
                            worker_timeouts_s=study_budgets())
    for purpose in ("readiness-pilot", "unknown-study"):
        changed = json.loads(json.dumps(manifest))
        changed["purpose"] = purpose
        (root / "manifest.json").write_text(json.dumps(changed))
        (root / "manifest.sha256.json").write_text(
            json.dumps({
                "sha256":
                hashlib.sha256(
                    (root / "manifest.json").read_bytes()).hexdigest()
            }))
        with pytest.raises(ValueError, match="schedule|purpose|budget"):
            live.preflight(root, check_runtime=False)


@pytest.mark.parametrize("restore_declared_instruction_hash", [False, True])
def test_study_preflight_binds_case_label_to_actual_staged_prompt(
        tmp_path, inputs, restore_declared_instruction_hash):
    root = tmp_path / "full-run"
    manifest = live.prepare(
        root,
        **inputs,
        mode="disclosed-study",
        worker_timeouts_s={case_id: 600
                           for case_id in DISCLOSED_IDS})
    first, second = manifest["attempts"][0], manifest["attempts"][2]
    # Same arm, same budget, valid task hashes; only case/prompt association breaks.
    for key in ("attempt_id", "dataset_dir", "job_name", "input_hashes"):
        first[key], second[key] = second[key], first[key]
    if restore_declared_instruction_hash:
        for attempt in (first, second):
            expected = live.worker_instruction(attempt["case_id"],
                                               attempt["arm"])
            attempt["input_hashes"]["instruction"] = hashlib.sha256(
                expected.encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "manifest.sha256.json").write_text(
        json.dumps({
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        }))
    with pytest.raises(ValueError, match="instruction|prompt|entry"):
        live.preflight(root, check_runtime=False)


@pytest.mark.parametrize("mode", ["pilot", "disclosed-study"])
def test_prepare_cli_passes_explicit_mode_and_budgets_without_execution(
        tmp_path, monkeypatch, mode):
    calls = []

    def prepare_only(root, **kwargs):
        calls.append((root, kwargs))
        return {"prepared": True}

    monkeypatch.setattr(live, "prepare", prepare_only)

    def no_execution(*args, **kwargs):
        raise AssertionError("prepare CLI must not execute a model")

    monkeypatch.setattr(live, "run", no_execution)
    args = [
        "prepare",
        str(tmp_path / "run"), "--source-archive", "source.tar",
        "--skill-archive", "skill.tar"
    ]
    if mode == "disclosed-study":
        budget_path = tmp_path / "budgets.json"
        budget_path.write_text(json.dumps(study_budgets()))
        args += ["--mode", mode, "--worker-timeouts", str(budget_path)]
    live.main(args)
    assert len(calls) == 1
    assert calls[0][1]["mode"] == mode
    assert calls[0][1]["worker_timeouts_s"] == (study_budgets() if mode
                                                == "disclosed-study" else None)


def archive(path, entries):
    with tarfile.open(path, "w") as handle:
        for name, text in entries.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source.tar"
    source_hash = archive(source, {
        "python/example.py": "x = 1\n",
        "README.md": "Public source\n"
    })
    skill = tmp_path / "skill.tar"
    skill_hash = archive(
        skill, {
            "cudaq-algorithms/SKILL.md":
            "---\nname: cudaq-algorithms\ndescription: CUDA-Q algorithms.\n---\nRead [authoring](authoring/guide.md).\n",
            "cudaq-algorithms/authoring/guide.md":
            "Public authoring support\n",
        })
    capture = tmp_path / "capture.py"
    capture.write_text(
        "# Test capture interface fixture; no model or scientific runtime.\n")
    return dict(source_archive=source,
                skill_archive=skill,
                capture_path=capture,
                expected_source_sha256=source_hash,
                expected_skill_sha256=skill_hash,
                codex_version="0.154.0")


@pytest.fixture
def prepared(tmp_path, inputs):
    assert live is not None, "live pilot controller has not been implemented"
    run_root = tmp_path / "run"
    manifest = live.prepare(run_root, **inputs)
    return run_root, manifest


def test_real_adapter_stages_identical_source_and_private_capture_only(
        prepared):
    root, manifest = prepared
    assert len(manifest["attempts"]) == 8
    for index, attempt in enumerate(manifest["attempts"]):
        assert set(("attempt_id", "case_id", "arm", "repetition",
                    "dataset_dir", "job_name", "worker_timeout_s",
                    "input_hashes")) <= attempt.keys()
        assert attempt["case_id"] not in attempt["attempt_id"]
        task = root / attempt["dataset_dir"] / attempt["attempt_id"]
        assert (task /
                "environment/repo/python/example.py").read_text() == "x = 1\n"
        assert not list((task / "environment/repo").rglob("SKILL.md"))
        assert not (task / "environment/repo/.git").exists()
        entry = json.loads((task / "tests/entry.json").read_text())
        assert entry["question"] == live.worker_instruction(
            attempt["case_id"], attempt["arm"])
        assert entry["grading_mode"] == "custom_only"
        assert not {"score_rubric", "expected_behavior", "critical_checks"
                    } & entry.keys()
        package = task / "environment/skills/cudaq-algorithms"
        assert package.exists() is (index % 2 == 1)
        if package.exists():
            assert (package / "authoring/guide.md"
                    ).read_text() == "Public authoring support\n"
        config = tomllib.loads((task / "task.toml").read_text())
        assert config["agent"]["timeout_sec"] == (1200 if index >= 6 else 600)
        assert config["verifier"]["timeout_sec"] == 120
        assert config["environment"]["storage_mb"] == 10240
    result = live.preflight(root, check_runtime=False)
    assert result["status"] == "ready"
    assert result["model_calls"] == 0


def test_prepare_refuses_wrong_archive_hash_and_an_existing_run(
        tmp_path, inputs, prepared):
    with pytest.raises(ValueError, match="source.*hash"):
        live.prepare(tmp_path / "bad", **{
            **inputs, "expected_source_sha256": "0" * 64
        })
    with pytest.raises(FileExistsError):
        live.prepare(prepared[0], **inputs)


def test_recurring_setup_preserves_real_worktree_edits(tmp_path):
    assert live is not None, "live pilot controller has not been implemented"
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.py").write_text("original\n")
    logs = tmp_path / "logs/agent"
    worktree = logs / "worktree"
    project = tmp_path / "project"
    script = live.worktree_setup_command(source, logs, project)
    subprocess.run(["bash", "-c", script],
                   check=True,
                   capture_output=True,
                   text=True)
    assert project.resolve() == worktree
    assert (worktree / ".git").is_file()
    assert subprocess.check_output(
        ["git", "-C", str(worktree), "status", "--porcelain"]) == b""
    assert subprocess.run(
        ["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"],
        capture_output=True).returncode == 1
    initial = json.loads((logs / "worktree-initial.json").read_text())
    assert initial["clean"] is True
    (worktree / "file.py").write_text("worker edit\n")
    subprocess.run(["bash", "-c", script],
                   check=True,
                   capture_output=True,
                   text=True)
    assert (worktree / "file.py").read_text() == "worker edit\n"
    assert json.loads((logs / "worktree-initial.json").read_text()) == initial


def test_reservation_is_atomic_one_shot_and_cannot_skip_baseline(prepared):
    root, manifest = prepared
    first, second = manifest["attempts"][:2]
    with pytest.raises(RuntimeError, match="order"):
        live.reserve_attempt(root, second["attempt_id"])
    reservation = live.reserve_attempt(root, first["attempt_id"])
    assert json.loads(
        reservation.read_text())["attempt_id"] == first["attempt_id"]
    with pytest.raises(FileExistsError):
        live.reserve_attempt(root, first["attempt_id"])


def test_preflight_rejects_changes_to_frozen_tasks(prepared):
    root, manifest = prepared
    attempt = manifest["attempts"][0]
    task = root / attempt["dataset_dir"] / attempt["attempt_id"]
    (task / "instruction.md").write_text("Changed after freezing\n")
    with pytest.raises(ValueError, match="hash"):
        live.preflight(root, check_runtime=False)


@pytest.mark.parametrize("fault",
                         ["entry_id", "instruction_file", "instruction_hash"])
def test_preflight_rejects_refrozen_inconsistent_prompt_identity(
        prepared, fault):
    root, manifest = prepared
    attempt = manifest["attempts"][0]
    task = root / attempt["dataset_dir"] / attempt["attempt_id"]
    if fault == "entry_id":
        path = task / "tests/entry.json"
        entry = json.loads(path.read_text())
        entry["id"] = "wrong-opaque-id"
        path.write_text(json.dumps(entry))
    elif fault == "instruction_file":
        (task / "instruction.md").write_text("Not the declared worker task.\n")
    else:
        attempt["input_hashes"]["instruction"] = "0" * 64
    # Recompute ordinary byte hashes to exercise semantic task/case consistency.
    attempt["input_hashes"]["task"] = live._digest(
        live.inventory(root / attempt["dataset_dir"]))
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "manifest.sha256.json").write_text(
        json.dumps({
            "sha256":
            hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        }))
    with pytest.raises(ValueError, match="instruction|prompt|entry"):
        live.preflight(root, check_runtime=False)


def test_run_uses_secure_stdin_serial_single_attempts_and_restores_command_builder(
        prepared, monkeypatch):
    root, manifest = prepared
    from skillevaluator.tier3.harbor import runner
    original_run = subprocess.run
    original_builder = runner.build_harbor_run_command
    observed = []

    def external_process(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0,
                                               manifest["image_id"] + "\n", "")
        if command[0] == runner._harbor_bin() and command[1] == "run":
            observed.append((list(command), kwargs))
            return subprocess.CompletedProcess(
                command, 1, "", "deliberate infrastructure failure")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external_process)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-pass-to-child")
    result = live.run(root, api_key="test-credential")
    assert len(result) == 8
    assert all(row["status"] == "failed" for row in result)
    assert runner.build_harbor_run_command is original_builder
    for (command, kwargs), attempt in zip(observed,
                                          manifest["attempts"],
                                          strict=True):
        assert command[command.index("--job-name") + 1] == attempt["job_name"]
        assert command[command.index("--n-attempts") + 1] == "1"
        assert command[command.index("--n-concurrent") + 1] == "1"
        assert command[command.index("--max-retries") + 1] == "0"
        assert command[command.index("--agent-kwarg") + 1] == "version=0.154.0"
        assert command[command.index("--agent-import-path") +
                       1].endswith(":SkillEvaluatorNvidiaBuildCodex")
        assert "--timeout-multiplier" not in command
        assert kwargs["input"] == "test-credential"
        assert "test-credential" not in json.dumps(command)
        assert "test-credential" not in json.dumps(kwargs["env"])
        assert "UNRELATED_SECRET" not in kwargs["env"]
    assert len(list((root / "outcomes").glob("*.json"))) == 8
    with pytest.raises(FileExistsError):
        live.run(root,
                 api_key="test-credential",
                 attempt_id=manifest["attempts"][0]["attempt_id"])
    assert len(observed) == 8


def test_interrupted_worker_stays_reserved_and_restores_builder(
        prepared, monkeypatch):
    root, manifest = prepared
    from skillevaluator.tier3.harbor import runner
    original_run = subprocess.run
    original_builder = runner.build_harbor_run_command

    def interrupt(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0,
                                               manifest["image_id"] + "\n", "")
        if command[0] == runner._harbor_bin() and command[1] == "run":
            raise KeyboardInterrupt()
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", interrupt)
    first = manifest["attempts"][0]["attempt_id"]
    with pytest.raises(KeyboardInterrupt):
        live.run(root, api_key="test-credential", attempt_id=first)
    assert runner.build_harbor_run_command is original_builder
    assert json.loads((root / "outcomes" /
                       f"{first}.json").read_text())["status"] == "interrupted"
    with pytest.raises(FileExistsError):
        live.reserve_attempt(root, first)


def test_worktree_bootstrap_exposes_editable_source_via_plain_pth(tmp_path):
    source = tmp_path / "source"
    package = source / "python/bootstrap_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 17\n")
    project, logs, python_site = tmp_path / "project", tmp_path / "agent", tmp_path / "site"
    setup = live.worktree_setup_command(source,
                                        logs,
                                        project,
                                        python_site=python_site)
    subprocess.run(["bash", "-c", setup], check=True, capture_output=True)
    assert (python_site / "cudaq_strategy_worktree.pth").read_text() == str(
        project / "python") + "\n"
    code = "import site; site.addsitedir(" + repr(
        str(python_site
            )) + "); import bootstrap_fixture; print(bootstrap_fixture.VALUE)"
    assert subprocess.check_output([sys.executable, "-c", code],
                                   text=True).strip() == "17"
    (project /
     "python/bootstrap_fixture/__init__.py").write_text("VALUE = 1700\n")
    subprocess.run(["bash", "-c", setup], check=True, capture_output=True)
    assert subprocess.check_output([sys.executable, "-c", code],
                                   text=True).strip() == "1700"


def test_setup_preserves_adapter_discovery_directories_outside_product_worktree(
        tmp_path):
    from skillevaluator.tier3.harbor.adapter import _PROJECT_SKILL_RELATIVE_DIRS
    source, project, logs = tmp_path / "source", tmp_path / "project", tmp_path / "agent"
    source.mkdir()
    (source / "file.py").write_text("original\n")
    for relative in _PROJECT_SKILL_RELATIVE_DIRS:
        (project / relative).mkdir(parents=True, exist_ok=True)
    setup = live.worktree_setup_command(source, logs, project)
    subprocess.run(["bash", "-c", setup],
                   check=True,
                   capture_output=True,
                   text=True)
    preserved = tmp_path / "project-runtime-discovery"
    assert all((preserved / relative).is_dir()
               for relative in _PROJECT_SKILL_RELATIVE_DIRS)
    assert not any((project / relative).exists()
                   for relative in _PROJECT_SKILL_RELATIVE_DIRS)
    assert not list(project.rglob("SKILL.md"))
    assert subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"]) == b""
    (project / "file.py").write_text("worker edit\n")
    subprocess.run(["bash", "-c", setup],
                   check=True,
                   capture_output=True,
                   text=True)
    assert (project / "file.py").read_text() == "worker edit\n"


@pytest.mark.parametrize(
    "relative", ["user-file.txt", ".agents/skills/cudaq-algorithms/SKILL.md"])
def test_setup_refuses_arbitrary_project_content_without_moving_it(
        tmp_path, relative):
    source, project, logs = tmp_path / "source", tmp_path / "project", tmp_path / "agent"
    source.mkdir()
    (source / "file.py").write_text("original\n")
    existing = project / relative
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("preserve this\n")
    result = subprocess.run(
        ["bash", "-c",
         live.worktree_setup_command(source, logs, project)],
        capture_output=True,
        text=True)
    assert result.returncode != 0
    assert existing.read_text() == "preserve this\n"
    assert not (tmp_path / "project-runtime-discovery").exists()
    assert not (logs / "worktree").exists()


@pytest.mark.parametrize("helper", ["controller", "suite", "brief", "capture"])
def test_preflight_rejects_helper_source_changes_after_freeze(
        prepared, inputs, tmp_path, monkeypatch, helper):
    root, _ = prepared
    changed = tmp_path / "changed-helper"
    changed.write_text("changed after freeze\n")
    if helper == "controller":
        monkeypatch.setattr(live, "__file__", str(changed))
    elif helper == "suite":
        (tmp_path / "suite.py").write_text("changed after freeze\n")
        monkeypatch.setattr(live, "HERE", tmp_path)
    elif helper == "brief":
        monkeypatch.setattr(live.suite, "BRIEF", changed)
    else:
        inputs["capture_path"].write_text("changed after freeze\n")
    with pytest.raises(ValueError, match="helper.*hash"):
        live.preflight(root, check_runtime=False)


def test_setup_preserves_empty_discovery_tree_when_overlayfs_rename_returns_exdev(
        tmp_path):
    from skillevaluator.tier3.harbor.adapter import _PROJECT_SKILL_RELATIVE_DIRS
    source, project, logs = tmp_path / "source", tmp_path / "project", tmp_path / "agent"
    source.mkdir()
    (source / "file.py").write_text("original\n")
    for relative in _PROJECT_SKILL_RELATIVE_DIRS:
        (project / relative).mkdir(parents=True, exist_ok=True)
    setup = live.worktree_setup_command(source, logs, project)
    # OverlayFS may refuse a lower-layer directory rename even within one mount.
    # Inject only that OS failure; let the real filesystem relocation run.
    inject_exdev = (
        "import errno, os\n"
        "original_rename = os.rename\n"
        "def overlay_rename(src, dst, *args, **kwargs):\n"
        f"    if os.fspath(src) == {str(project)!r}:\n"
        "        raise OSError(errno.EXDEV, 'Invalid cross-device link')\n"
        "    return original_rename(src, dst, *args, **kwargs)\n"
        "os.rename = overlay_rename\n")
    subprocess.run(
        [sys.executable, "-c", inject_exdev + shlex.split(setup)[2]],
        check=True,
        capture_output=True,
        text=True)
    preserved = tmp_path / "project-runtime-discovery"
    assert all((preserved / relative).is_dir()
               for relative in _PROJECT_SKILL_RELATIVE_DIRS)
    assert project.is_symlink()
    assert project.resolve() == logs / "worktree"
    assert (project / "file.py").read_text() == "original\n"
    assert subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"]) == b""
