# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Psi4-only strict-runtime permissions and task-local side artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import classical_cases
import run
import runtime


def _runtime(tmp_path: Path, name: str, record: dict | None = None) -> Path:
    prefix = tmp_path / name
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    if record is not None:
        metadata = prefix / "conda-meta"
        metadata.mkdir()
        (metadata / "psi4-1.10-test_0.json").write_text(json.dumps(record))
    return python


def _filesystem_config(workspace: Path, python: Path) -> str:
    args = runtime.profile_args(workspace, python)
    return next(value for value in args
                if value.startswith("permissions.cudaq_e2e.filesystem="))


def test_only_a_detected_psi4_conda_runtime_gets_process_self_read(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    core = _runtime(tmp_path, "core")
    wrong_record = _runtime(tmp_path, "wrong", {
        "name": "not-psi4",
        "version": "1.10"
    })
    psi4 = _runtime(tmp_path, "psi4", {"name": "psi4", "version": "1.10"})

    assert '"/proc/self"="read"' not in _filesystem_config(workspace, core)
    assert '"/proc/self"="read"' not in _filesystem_config(
        workspace, wrong_record)
    psi4_config = _filesystem_config(workspace, psi4)
    assert '"/proc/self"="read"' in psi4_config
    assert '"/proc"=' not in psi4_config
    assert '"/sys"=' not in psi4_config


def test_non_psi4_staging_adds_no_runtime_artifact(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / ".tmp").mkdir(parents=True)

    runtime.stage_runtime_artifacts(workspace, "pyscf_energy")

    assert sorted(path.name for path in workspace.iterdir()) == [".tmp"]


def test_psi4_timer_is_fixed_inside_scratch_and_link_integrity_is_checked(
        tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / ".tmp").mkdir(parents=True)
    runtime.stage_runtime_artifacts(workspace, "psi4_energy")
    timer = workspace / "timer.dat"
    before = runtime.inventory(workspace)

    assert timer.is_symlink()
    assert os.readlink(timer) == ".tmp/timer.dat"
    timer.write_text("provider timer\n")
    assert runtime.scope_changes(before, runtime.inventory(workspace)) == []

    timer.unlink()
    timer.symlink_to(".tmp/repointed.dat")
    assert runtime.scope_changes(
        before, runtime.inventory(workspace)) == ["timer.dat"]


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
def test_psi4_timer_staging_rejects_every_existing_timer_path(tmp_path, kind):
    workspace = tmp_path / "workspace"
    (workspace / ".tmp").mkdir(parents=True)
    timer = workspace / "timer.dat"
    if kind == "file":
        timer.write_text("owned by caller")
    else:
        timer.symlink_to("missing-target")

    with pytest.raises(FileExistsError):
        runtime.stage_runtime_artifacts(workspace, "psi4_energy")


def test_psi4_timer_staging_rejects_scratch_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / ".tmp").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="scratch"):
        runtime.stage_runtime_artifacts(workspace, "psi4_energy")
    assert not (workspace / "timer.dat").exists()


def test_run_one_stages_psi4_timer_before_its_initial_inventory(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = source / "python/cudaq_algorithms/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    manifest = {
        "source_snapshot": str(source),
        "staging_root": str(staging),
        "timeout_s": 1,
        "interpreters": {
            "psi4_energy": sys.executable
        },
        "model": "test-model",
    }
    record = {
        "id": "psi4_energy--1--baseline",
        "case": "psi4_energy",
        "arm": "baseline",
        "repetition": 1
    }

    def fake_native(command, workspace, prompt, destination, timeout,
                    collector):
        (Path(workspace) / "timer.dat").write_text("provider timer\n")
        (Path(workspace) / "app.py").write_text("print('application')\n")
        return {
            "returncode": 0,
            "timed_out": False,
            "elapsed_s": 0.,
            "commands": 0,
            "tool_output_bytes": 0
        }

    class FakeTelemetry:
        config_args = []
        events = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def summary(self):
            return {"totals": {"input_tokens": 0, "output_tokens": 0}}

    monkeypatch.setattr(run, "run_native", fake_native)
    monkeypatch.setattr(run, "TelemetryCollector", FakeTelemetry)
    monkeypatch.setattr(run, "execute_app", lambda *args, **kwargs: {
        "passed": True,
        "checks": []
    })

    result = run.run_one(record, manifest, output)

    initial = json.loads((output / "runs" / record["id"] /
                          "initial-inventory.json").read_text())
    assert initial["timer.dat"] == "symlink:.tmp/timer.dat"
    assert result["scope_changes"] == []


@pytest.mark.skipif(not os.environ.get("E2E_SANDBOX_PSI4_PYTHON"),
                    reason="explicit restrictive Psi4 runtime required")
def test_real_psi4_gold_passes_both_variants_in_strict_sandbox(tmp_path):
    interpreter = os.environ["E2E_SANDBOX_PSI4_PYTHON"]
    spec = run.CASES["psi4_energy"][1]

    result = run.execute_app(runtime.REPO,
                             spec,
                             classical_cases.reference_source("psi4_energy"),
                             interpreter,
                             tmp_path / "checks",
                             limit=180)

    assert result["passed"], result
    assert len(result["checks"]) == 2
    assert all(check["scope_changes"] == [] for check in result["checks"])


@pytest.mark.skipif(not os.environ.get("E2E_SANDBOX_PSI4_PYTHON"),
                    reason="explicit restrictive Psi4 runtime required")
def test_psi4_process_self_read_cannot_escape_isolation(tmp_path):
    interpreter = os.environ["E2E_SANDBOX_PSI4_PYTHON"]
    primary = tmp_path / "primary"
    sibling = tmp_path / "sibling"
    runtime.stage_workspace(runtime.REPO, primary, "baseline", {"probe": 1})
    runtime.stage_workspace(runtime.REPO, sibling, "baseline", {"probe": 2})
    evaluator = ROOT / "classical_cases.py"
    forbidden = [
        evaluator,
        runtime.SKILL / "SKILL.md",
        sibling / "input.json",
        "/proc/1/environ",
        "/proc/2/statm",
        "/proc/self/root" + str(evaluator),
        "/proc/self/root" + str(sibling / "input.json"),
        "/proc/self/cwd/../sibling/input.json",
    ]

    result = runtime.isolation_probe(primary, interpreter, forbidden)

    assert result["passed"], result
    observed = json.loads(result["stdout"])
    assert all(observed[str(path)] for path in forbidden)
    assert observed["network_blocked"]
