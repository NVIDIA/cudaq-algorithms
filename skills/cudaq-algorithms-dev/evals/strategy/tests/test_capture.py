# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture uses real files; no Git status or worker execution is trusted."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[1] / "capture.py"


@pytest.fixture
def capture_module():
    assert MODULE.is_file(), "durable capture has not been implemented"
    spec = importlib.util.spec_from_file_location("strategy_capture", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trees(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "python").mkdir()
    (source / "python/example.py").write_text("value = 1\n")
    (source / "docs").mkdir()
    (source / "docs/removed.md").write_text("original\n")
    worktree = tmp_path / "agent/worktree"
    shutil.copytree(source, worktree)
    return source, worktree, tmp_path / "capture"


def test_changed_deleted_added_files_have_complete_hashed_artifacts(
        capture_module, trees):
    source, worktree, output = trees
    (worktree / "python/example.py").write_text("value = 2\n")
    (worktree / "docs/removed.md").unlink()
    (worktree / "notes.txt").write_text("new file without newline")

    result = capture_module.capture(source, worktree, output)

    assert result["status"] == "complete"
    assert result["changed_paths"] == ["notes.txt", "python/example.py"]
    assert result["added_paths"] == ["notes.txt"]
    assert result["deleted_paths"] == ["docs/removed.md"]
    assert (output / "files/python/example.py").read_bytes() == b"value = 2\n"
    assert (output /
            "files/notes.txt").read_bytes() == b"new file without newline"
    diff = (output / "changes.diff").read_text()
    assert "-value = 1\n+value = 2\n" in diff
    assert "--- a/docs/removed.md\n+++ /dev/null\n" in diff
    assert "\\ No newline at end of file" in diff
    stored = json.loads((output / "capture.json").read_text())
    assert stored["changed_paths"] == result["changed_paths"]
    for artifact in [*result["artifacts"], result["capture_json"]]:
        raw = (output / artifact["path"]).read_bytes()
        assert len(raw) == artifact["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"]
    assert set(
        p.relative_to(output).as_posix() for p in output.rglob("*")
        if p.is_file()) == {
            artifact["path"]
            for artifact in [*result["artifacts"], result["capture_json"]]
        }


def test_unchanged_tree_and_exclusions_are_explicit(capture_module, trees):
    source, worktree, output = trees
    (worktree / ".git").write_text("gitdir: /not/read\n")
    (worktree / "python/__pycache__").mkdir()
    (worktree / "python/__pycache__/example.pyc").write_bytes(b"cache")

    result = capture_module.capture(source, worktree, output)

    assert result["status"] == "complete"
    assert result["changed_paths"] == []
    assert result["deleted_paths"] == []
    assert (output / "changes.diff").read_bytes() == b""
    assert {r["path"]
            for r in result["exclusions"]} == {".git", "python/__pycache__"}


def test_worker_committing_changes_cannot_hide_them(capture_module, trees):
    source, worktree, output = trees
    for command in (["git", "init", "-q"], ["git", "add", "."], [
            "git", "-c", "commit.gpgsign=false", "-c",
            "user.name=Capture Test", "-c",
            "user.email=capture@example.invalid", "commit", "-qm", "initial"
    ]):
        subprocess.run(command, cwd=worktree, check=True, capture_output=True)
    (worktree /
     "python/example.py").write_text("committed_worker_change = True\n")
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run([
        "git", "-c", "commit.gpgsign=false", "-c", "user.name=Capture Test",
        "-c", "user.email=capture@example.invalid", "commit", "-qm",
        "worker commit"
    ],
                   cwd=worktree,
                   check=True)
    assert subprocess.run(["git", "status", "--porcelain"],
                          cwd=worktree,
                          check=True,
                          capture_output=True).stdout == b""

    result = capture_module.capture(source, worktree, output)
    assert result["changed_paths"] == ["python/example.py"]
    assert result["source_inventory"]["python/example.py"][
        "sha256"] == hashlib.sha256(b"value = 1\n").hexdigest()


@pytest.mark.parametrize("kind",
                         ["symlink", "directory_symlink", "hardlink", "fifo"])
def test_unsafe_entries_are_rejected_without_following(capture_module, trees,
                                                       kind):
    source, worktree, output = trees
    secret = source.parent / "outside"
    secret.mkdir()
    (secret / "private").write_text("outside must not enter capture")
    unsafe = worktree / "unsafe"
    if kind == "symlink":
        unsafe.symlink_to(secret / "private")
    elif kind == "directory_symlink":
        unsafe.symlink_to(secret, target_is_directory=True)
    elif kind == "hardlink":
        os.link(secret / "private", unsafe)
    else:
        os.mkfifo(unsafe)

    result = capture_module.capture(source, worktree, output)

    assert result["status"] == "incomplete"
    assert any(r["tree"] == "worktree" and r["path"] == "unsafe"
               for r in result["rejections"])
    assert not (output / "files/unsafe").exists()
    assert "outside must not enter capture" not in (
        output / "changes.diff").read_text()


def test_symlink_root_is_rejected_and_missing_files_are_not_claimed_deleted(
        capture_module, trees):
    source, worktree, output = trees
    alias = worktree.parent / "alias"
    alias.symlink_to(worktree, target_is_directory=True)
    result = capture_module.capture(source, alias, output)
    assert result["status"] == "incomplete"
    assert result["deleted_paths"] == []
    assert result["rejections"]


def test_rejected_directory_does_not_turn_its_children_into_deletions(
        capture_module, trees):
    source, worktree, output = trees
    (worktree / "docs/removed.md").unlink()
    (worktree / "docs").rmdir()
    (worktree / "docs").symlink_to(source / "docs", target_is_directory=True)
    result = capture_module.capture(source, worktree, output)
    assert result["status"] == "incomplete"
    assert result["deleted_paths"] == []
    assert "docs/removed.md" in result["unverified_deletions"]


def test_source_unsafe_entries_also_prevent_complete_capture(
        capture_module, trees):
    source, worktree, output = trees
    (source / "unsafe").symlink_to(source / "python/example.py")
    result = capture_module.capture(source, worktree, output)
    assert result["status"] == "incomplete"
    assert any(r["tree"] == "source" for r in result["rejections"])


def test_file_and_total_byte_limits_fail_closed(capture_module, trees):
    source, worktree, output = trees
    (worktree / "oversized.bin").write_bytes(b"x" * 33)
    result = capture_module.capture(source,
                                    worktree,
                                    output,
                                    max_file_bytes=32,
                                    max_tree_bytes=100)
    assert result["status"] == "incomplete"
    assert any(
        r["path"] == "oversized.bin" and r["reason"] == "file_byte_limit"
        for r in result["rejections"])
    assert not (output / "files/oversized.bin").exists()
    (worktree / "oversized.bin").unlink()
    (worktree / "a.bin").write_bytes(b"a" * 25)
    (worktree / "b.bin").write_bytes(b"b" * 25)
    result = capture_module.capture(source,
                                    worktree,
                                    output.parent / "limited",
                                    max_file_bytes=32,
                                    max_tree_bytes=40)
    assert result["status"] == "incomplete"
    assert any(r["reason"] == "tree_byte_limit" for r in result["rejections"])
    assert sum(r["size_bytes"]
               for r in result["worktree_inventory"].values()) <= 40


def test_entry_limit_bounds_empty_files(capture_module, trees):
    source, worktree, output = trees
    for index in range(8):
        (worktree / f"empty-{index}").touch()
    result = capture_module.capture(source, worktree, output, max_entries=5)
    assert result["status"] == "incomplete"
    assert any(r["reason"] == "entry_limit" for r in result["rejections"])


def test_judge_limit_keeps_full_artifacts_and_marks_projection_unavailable(
        capture_module, trees):
    source, worktree, output = trees
    (worktree / "python/example.py").write_text("x = '" + "a" * 2048 + "'\n")
    result = capture_module.capture(source,
                                    worktree,
                                    output,
                                    max_judge_bytes=1024)
    assert result["status"] == "complete"
    assert (output / "files/python/example.py").stat().st_size > 2048
    assert result["judge_evidence"]["status"] == "over_limit"
    projection = json.loads((output / "judge_evidence.json").read_text())
    assert projection["status"] == "over_limit"
    assert projection["truncated"] is False
    assert (output / "judge_evidence.json").stat().st_size <= 1024


def test_binary_and_executable_mode_changes_survive_capture(
        capture_module, trees):
    source, worktree, output = trees
    (worktree / "data.bin").write_bytes(b"\x00\xff\xfe")
    (worktree / "python/example.py").chmod(0o755)
    result = capture_module.capture(source, worktree, output)
    assert result["changed_paths"] == ["data.bin", "python/example.py"]
    assert (output / "files/data.bin").read_bytes() == b"\x00\xff\xfe"
    assert result["worktree_inventory"]["python/example.py"]["mode"] == 0o755
    assert "Binary files" in (output / "changes.diff").read_text()


def test_duplicate_output_is_refused_without_overwrite(capture_module, trees):
    source, worktree, output = trees
    capture_module.capture(source, worktree, output)
    before = (output / "capture.json").read_bytes()
    with pytest.raises(FileExistsError):
        capture_module.capture(source, worktree, output)
    assert (output / "capture.json").read_bytes() == before


def test_output_symlink_parent_and_overlap_are_refused(capture_module, trees):
    source, worktree, output = trees
    alias = output.parent / "alias"
    alias.symlink_to(output.parent, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        capture_module.capture(source, worktree, alias / "capture")
    with pytest.raises(ValueError):
        capture_module.capture(source, worktree, worktree / "capture")
    assert not output.exists()


def test_capture_never_executes_worker_python(capture_module, trees):
    source, worktree, output = trees
    marker = output.parent / "executed"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    for name in ("sitecustomize.py", "conftest.py", "setup.py"):
        (worktree / name).write_text(payload)
    result = capture_module.capture(source, worktree, output)
    assert result["status"] == "complete"
    assert not marker.exists()


def test_cli_recovers_persistent_worktree_from_job_logs(capture_module, trees):
    source, worktree, output = trees
    (worktree / "python/example.py").write_text("after_timeout = True\n")
    completed = subprocess.run([
        sys.executable,
        str(MODULE), "--source",
        str(source), "--job-logs",
        str(worktree.parent.parent), "--output",
        str(output)
    ],
                               capture_output=True,
                               text=True,
                               timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(
        completed.stdout)["changed_paths"] == ["python/example.py"]
    assert (output / "capture.json").is_file()


def test_standalone_grader_reward_reports_capture_only(capture_module, trees):
    source, worktree, output = trees
    grader = output.parent / "grader.py"
    shutil.copyfile(MODULE, grader)
    env = dict(os.environ,
               HARBOR_CAPTURE_SOURCE=str(source),
               HARBOR_CAPTURE_WORKTREE=str(worktree),
               HARBOR_CAPTURE_OUTPUT=str(output),
               HARBOR_REWARD_JSON=str(output.parent / "reward.json"),
               HARBOR_REWARD_TXT=str(output.parent / "reward.txt"))
    completed = subprocess.run([sys.executable, str(grader)],
                               env=env,
                               capture_output=True,
                               text=True,
                               timeout=20)
    assert completed.returncode == 0, completed.stderr
    reward = json.loads((output.parent / "reward.json").read_text())
    assert reward["overall"] == 1.0
    assert reward["custom_metrics"] == {"capture_complete": 1.0}
    assert reward["details"]["repository_capture"]["quality_grade"] is False
    assert reward["details"]["repository_capture"]["status"] == "complete"
    assert (output.parent / "reward.txt").read_text().strip() == "1.0"


def test_reconstruct_restores_exact_captured_bytes_deletions_and_modes(
        capture_module, trees):
    source, worktree, output = trees
    (worktree / "python/example.py").write_text("captured = True\n")
    (worktree / "python/example.py").chmod(0o755)
    (worktree / "docs/removed.md").unlink()
    (worktree / "binary").write_bytes(b"\x00\xff")
    capture_module.capture(source, worktree, output)
    reconstructed = output.parent / "reconstructed"

    result = capture_module.reconstruct(source, output, reconstructed)

    assert result["status"] == "complete"
    assert (reconstructed /
            "python/example.py").read_bytes() == b"captured = True\n"
    assert (reconstructed /
            "python/example.py").stat().st_mode & 0o777 == 0o755
    assert not (reconstructed / "docs/removed.md").exists()
    assert (reconstructed / "binary").read_bytes() == b"\x00\xff"
    assert result["worktree_inventory_sha256"] == json.loads(
        (output / "capture.json").read_text())["worktree_inventory_sha256"]


@pytest.mark.parametrize("tamper", [
    "source", "blob", "blob_symlink", "manifest_path", "manifest_hash",
    "deletion", "incomplete"
])
def test_reconstruct_rejects_untrusted_or_incomplete_evidence(
        capture_module, trees, tamper):
    source, worktree, output = trees
    (worktree / "python/example.py").write_text("changed = True\n")
    (worktree / "docs/removed.md").unlink()
    capture_module.capture(source, worktree, output)
    manifest = json.loads((output / "capture.json").read_text())
    if tamper == "source":
        (source / "python/example.py").write_text("wrong_source = True\n")
    elif tamper == "blob":
        (output /
         "files/python/example.py").write_text("unrecorded_blob = True\n")
    elif tamper == "blob_symlink":
        (output / "files/python/example.py").unlink()
        (output / "files/python/example.py").symlink_to(worktree /
                                                        "python/example.py")
    elif tamper == "manifest_path":
        manifest["changed_paths"] = ["../outside"]
    elif tamper == "manifest_hash":
        manifest["worktree_inventory_sha256"] = "0" * 64
    elif tamper == "deletion":
        manifest["deleted_paths"] = []
    else:
        manifest["status"] = "incomplete"
    (output / "capture.json").write_text(json.dumps(manifest))
    reconstructed = output.parent / "reconstructed"

    with pytest.raises((OSError, ValueError)):
        capture_module.reconstruct(source, output, reconstructed)
    assert not reconstructed.exists()


def test_reconstruct_refuses_duplicate_and_symlink_output(
        capture_module, trees):
    source, worktree, output = trees
    capture_module.capture(source, worktree, output)
    existing = output.parent / "existing"
    existing.mkdir()
    (existing / "preserved").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        capture_module.reconstruct(source, output, existing)
    assert (existing / "preserved").read_bytes() == b"original"
    alias = output.parent / "alias"
    alias.symlink_to(existing, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        capture_module.reconstruct(source, output, alias / "result")
    assert not (existing / "result").exists()


def test_unified_diff_preserves_non_newline_control_characters(
        capture_module, trees):
    source, worktree, output = trees
    (source /
     "python/example.py").write_bytes(b"before\x0bpart\rstill-the-line\nlast")
    (worktree /
     "python/example.py").write_bytes(b"after\x0bpart\rstill-the-line\nfinal")
    capture_module.capture(source, worktree, output)
    patched = output.parent / "patched"
    shutil.copytree(source, patched)

    applied = subprocess.run(
        ["git", "apply", str(output / "changes.diff")],
        cwd=patched,
        capture_output=True,
        text=True)

    assert applied.returncode == 0, applied.stderr
    assert (patched / "python/example.py").read_bytes() == (
        worktree / "python/example.py").read_bytes()


def test_small_judge_ceiling_still_bounds_the_unavailable_record(
        capture_module, trees):
    source, worktree, output = trees
    (worktree / "python/example.py").write_bytes(b"x" * 1000)
    result = capture_module.capture(source,
                                    worktree,
                                    output,
                                    max_judge_bytes=256)
    assert result["judge_evidence"]["status"] == "over_limit"
    assert (output / "judge_evidence.json").stat().st_size <= 256
