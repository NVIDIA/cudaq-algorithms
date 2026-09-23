#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, non-executing repository capture; also a standalone Harbor grader.

``capture(source, worktree, output)`` compares the two physical trees, never
Git's mutable index. The output directory must be new. ``files/<repo path>``
contains complete changed/added bytes; ``deleted_paths`` names removals.
Reconstruct only a COMPLETE capture over its matching source inventory,
verify every blob hash, and apply modes from ``worktree_inventory``.

The returned ``capture_json`` descriptor hashes the on-disk manifest; it is
deliberately outside that manifest to avoid a self-referential hash. All other
files are listed in ``artifacts``. This module imports and executes no worker
code. Capture completeness is an evidence channel, never a quality grade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

MAX_FILE_BYTES = 2 * 1024**2
MAX_TREE_BYTES = 50 * 1024**2
MAX_JUDGE_BYTES = 1024**2
MAX_ENTRIES = 20_000
MAX_DEPTH = 64
EXCLUDED_NAMES = {".git", "__pycache__", ".pytest_cache"}
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) +
            "\n").encode("utf-8")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _directory(path: Path, *, create: bool = False) -> int:
    """Open every absolute path component with O_NOFOLLOW, including parents."""
    descriptor = os.open("/", DIRECTORY_FLAGS)
    try:
        for component in path.absolute().parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("noncanonical directory component")
            if create:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _snapshot(path: Path, label: str, limits: dict[str,
                                                   int]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    contents: dict[str, bytes] = {}
    exclusions: list[dict[str, str]] = []
    rejections: list[dict[str, str]] = []
    total = 0
    entries = 0

    def reject(name: str, reason: str) -> None:
        rejections.append({"tree": label, "path": name, "reason": reason})

    def visit(descriptor: int, prefix: str, depth: int) -> None:
        nonlocal total, entries
        # Read at most the remaining entry budget plus one before sorting.
        # listdir() would allocate an unbounded list in a worker-created tree.
        names = []
        with os.scandir(descriptor) as directory:
            for entry in directory:
                entries += 1
                if entries > limits["max_entries"]:
                    reject(prefix or ".", "entry_limit")
                    break
                names.append(entry.name)
        for name in sorted(names):
            relative = f"{prefix}/{name}" if prefix else name
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                # No unsafe entry is called complete, even if its name would
                # normally be excluded. Ordinary .git files/dirs are excluded.
                if stat.S_ISLNK(info.st_mode):
                    reject(relative, "symlink")
                    continue
                if not (stat.S_ISDIR(info.st_mode)
                        or stat.S_ISREG(info.st_mode)):
                    reject(relative, "nonregular")
                    continue
                if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                    reject(relative, "hardlink")
                    continue
                if name in EXCLUDED_NAMES or name.endswith((".pyc", ".pyo")):
                    exclusions.append({
                        "tree":
                        label,
                        "path":
                        relative,
                        "reason":
                        "git_database"
                        if name == ".git" else "generated_python_cache"
                    })
                    continue
                if stat.S_ISDIR(info.st_mode):
                    if depth >= MAX_DEPTH:
                        reject(relative, "depth_limit")
                        continue
                    child = os.open(name, DIRECTORY_FLAGS, dir_fd=descriptor)
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev,
                                                              info.st_ino):
                            reject(relative, "directory_changed_during_read")
                        else:
                            visit(child, relative, depth + 1)
                    finally:
                        os.close(child)
                    continue
                if info.st_size > limits["max_file_bytes"]:
                    reject(relative, "file_byte_limit")
                    continue
                if total + info.st_size > limits["max_tree_bytes"]:
                    reject(relative, "tree_byte_limit")
                    continue
                file_fd = os.open(name,
                                  os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                  dir_fd=descriptor)
                with os.fdopen(file_fd, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(opened.st_mode)
                            or opened.st_nlink != 1
                            or (opened.st_dev, opened.st_ino, opened.st_size,
                                opened.st_mtime_ns, opened.st_ctime_ns)
                            != (info.st_dev, info.st_ino, info.st_size,
                                info.st_mtime_ns, info.st_ctime_ns)):
                        reject(relative, "file_changed_during_read")
                        continue
                    content = stream.read(
                        min(limits["max_file_bytes"],
                            limits["max_tree_bytes"] - total) + 1)
                    after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns,
                        after.st_nlink) != (opened.st_size, opened.st_mtime_ns,
                                            opened.st_ctime_ns, 1):
                    reject(relative, "file_changed_during_read")
                    continue
                if len(content) > limits["max_file_bytes"]:
                    reject(relative, "file_byte_limit")
                    continue
                if total + len(content) > limits["max_tree_bytes"]:
                    reject(relative, "tree_byte_limit")
                    continue
                if len(content) != info.st_size:
                    reject(relative, "file_changed_during_read")
                    continue
                total += len(content)
                inventory[relative] = {
                    "sha256": _digest(content),
                    "size_bytes": len(content),
                    "mode": stat.S_IMODE(info.st_mode)
                }
                contents[relative] = content
            except OSError:
                reject(relative, "unavailable_or_changed_entry")

    try:
        root = _directory(path)
    except (OSError, ValueError):
        reject(".", "unavailable_or_unsafe_root")
    else:
        try:
            visit(root, "", 0)
        except OSError:
            reject(".", "directory_read_failed")
        finally:
            os.close(root)
    return {
        "inventory": dict(sorted(inventory.items())),
        "contents": contents,
        "exclusions": exclusions,
        "rejections": rejections
    }


def _write(directory_fd: int, name: str, content: bytes) -> dict[str, Any]:
    """Exclusive output creation relative to an already-open trusted root."""
    parts = Path(name).parts
    if not parts or any(part in {"", ".", ".."}
                        for part in parts) or Path(name).is_absolute():
        raise ValueError("unsafe artifact path")
    parent = os.dup(directory_fd)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, 0o755, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(component, DIRECTORY_FLAGS, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1],
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW,
                             0o644,
                             dir_fd=parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent)
    finally:
        os.close(parent)
    return {
        "path": name,
        "sha256": _digest(content),
        "size_bytes": len(content)
    }


def _diff(path: str, before: bytes | None, after: bytes | None,
          old_mode: int | None, new_mode: int | None) -> str:
    """Linear-time full-file unified hunks; blobs are authoritative for binaries."""

    def quote(name: str) -> str:
        return json.dumps(name, ensure_ascii=True) if any(
            ord(c) < 32 or ord(c) > 126 or c in '\\"' for c in name) else name

    old_name = quote("a/" + path) if before is not None else "/dev/null"
    new_name = quote("b/" + path) if after is not None else "/dev/null"
    header = f"diff --git {quote('a/' + path)} {quote('b/' + path)}\n"
    if old_mode is None:
        header += f"new file mode {stat.S_IFREG | new_mode:06o}\n"
    elif new_mode is None:
        header += f"deleted file mode {stat.S_IFREG | old_mode:06o}\n"
    elif old_mode != new_mode:
        header += f"old mode {stat.S_IFREG | old_mode:06o}\nnew mode {stat.S_IFREG | new_mode:06o}\n"
    if before == after:
        return header
    try:
        old = (before or b"").decode("utf-8")
        new = (after or b"").decode("utf-8")
        if "\x00" in old or "\x00" in new:
            raise UnicodeError
    except UnicodeError:
        return header + f"Binary files {old_name} and {new_name} differ\n"

    def lines(text: str) -> list[str]:
        # Unified diffs count LF separators only; splitlines() would turn a
        # vertical tab or bare CR inside a line into a false patch boundary.
        parts = text.split("\n")
        return [part + "\n"
                for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

    old_lines = lines(old)
    new_lines = lines(new)
    header += f"--- {old_name}\n+++ {new_name}\n"
    if not old_lines and not new_lines:
        return header
    header += f"@@ -{1 if old_lines else 0},{len(old_lines)} +{1 if new_lines else 0},{len(new_lines)} @@\n"
    body = []
    for prefix, lines in (("-", old_lines), ("+", new_lines)):
        for line in lines:
            body.append(prefix + line)
            if not line.endswith("\n"):
                body.append("\n\\ No newline at end of file\n")
    return header + "".join(body)


def capture(source: str | Path,
            worktree: str | Path,
            output: str | Path,
            *,
            max_file_bytes: int = MAX_FILE_BYTES,
            max_tree_bytes: int = MAX_TREE_BYTES,
            max_judge_bytes: int = MAX_JUDGE_BYTES,
            max_entries: int = MAX_ENTRIES) -> dict[str, Any]:
    """Capture bounded regular files. Rejections retain evidence with status incomplete.

    Limits can be lowered for tests; frozen ceilings cannot be raised. Capture
    assumes the worker has stopped and source is a separately frozen inventory.
    Concurrent changes detected during reads fail closed; this is not an atomic
    filesystem snapshot of a running worker.
    """
    limits = {
        "max_file_bytes": max_file_bytes,
        "max_tree_bytes": max_tree_bytes,
        "max_judge_bytes": max_judge_bytes,
        "max_entries": max_entries
    }
    ceilings = {
        "max_file_bytes": MAX_FILE_BYTES,
        "max_tree_bytes": MAX_TREE_BYTES,
        "max_judge_bytes": MAX_JUDGE_BYTES,
        "max_entries": MAX_ENTRIES
    }
    if any(
            type(value) is not int or not 0 < value <= ceilings[key]
            for key, value in limits.items()):
        raise ValueError(
            "capture limits must be positive integers within frozen ceilings")
    if max_judge_bytes < 256:
        raise ValueError(
            "judge evidence ceiling must allow an explicit unavailable record")
    source, worktree, output = (Path(path).absolute()
                                for path in (source, worktree, output))
    if any(source == path or source in path.parents or path in source.parents
           for path in (worktree, output)):
        raise ValueError("source, worktree and output must not overlap")
    if output == worktree or output in worktree.parents or worktree in output.parents:
        raise ValueError("source, worktree and output must not overlap")
    parent = _directory(output.parent, create=True)
    try:
        os.mkdir(output.name, 0o755, dir_fd=parent)
        directory = os.open(output.name, DIRECTORY_FLAGS, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    try:
        original = _snapshot(source, "source", limits)
        edited = _snapshot(worktree, "worktree", limits)
        old, new = original["inventory"], edited["inventory"]
        changed = sorted(path for path in new if old.get(path) != new[path])
        missing = sorted(set(old) - set(new))
        deleted = missing if not edited["rejections"] else []
        artifacts = []
        for path in changed:
            artifacts.append(
                _write(directory, "files/" + path, edited["contents"][path]))
        diff = "".join(
            _diff(path, original["contents"].get(path), edited["contents"].get(
                path),
                  old.get(path, {}).get("mode"),
                  new.get(path, {}).get("mode"))
            for path in sorted(set(changed) | set(deleted)))
        artifacts.append(
            _write(directory, "changes.diff", diff.encode("utf-8")))
        projection: dict[str, Any] = {
            "status": "complete",
            "truncated": False,
            "changed_paths": changed,
            "deleted_paths": deleted,
            "diff": diff
        }
        projection_bytes = _json(projection)
        projected_size = len(projection_bytes)
        if projected_size > max_judge_bytes:
            projection = {
                "status":
                "over_limit",
                "truncated":
                False,
                "required_size_bytes":
                projected_size,
                "limit_bytes":
                max_judge_bytes,
                "reason":
                "Full patch remains in changes.diff and files/; judge projection unavailable."
            }
            projection_bytes = _json(projection)
        artifacts.append(
            _write(directory, "judge_evidence.json", projection_bytes))
        rejections = original["rejections"] + edited["rejections"]
        result: dict[str, Any] = {
            "schema_version":
            1,
            "status":
            "incomplete" if rejections else "complete",
            "comparison":
            "immutable_source_contents_and_modes",
            "quality_grade":
            False,
            "changed_paths":
            changed,
            "added_paths":
            sorted(set(new) - set(old)),
            "deleted_paths":
            deleted,
            "unverified_deletions":
            missing if edited["rejections"] else [],
            "source_inventory":
            old,
            "worktree_inventory":
            new,
            "source_inventory_sha256":
            _digest(_json(old)),
            "worktree_inventory_sha256":
            _digest(_json(new)),
            "exclusions":
            original["exclusions"] + edited["exclusions"],
            "rejections":
            rejections,
            "limits":
            limits,
            "artifacts":
            artifacts,
            "judge_evidence": {
                "status":
                projection["status"],
                "path":
                "judge_evidence.json",
                "required_size_bytes":
                projected_size,
                "limit_bytes":
                max_judge_bytes,
                "scope":
                "repository patch only; final combined judge payload must also enforce this ceiling"
            },
            "limitations": [
                "Capture requires a stopped worker and independently frozen source.",
                "Directory metadata and excluded Git/Python caches are not captured.",
                "Binaries require their hashed blobs; unified diff alone is insufficient.",
                "No worker code, test, or scientific evaluation was executed."
            ],
        }
        result["capture_json"] = _write(directory, "capture.json",
                                        _json(result))
        os.fsync(directory)
        return result
    finally:
        os.close(directory)


def _reward(result: dict[str, Any]) -> None:
    score = float(result["status"] == "complete")
    reward = {
        "overall": score,
        "custom_metrics": {
            "capture_complete": score
        },
        "details": {
            "repository_capture": result
        }
    }
    for path, content in ((Path(
            os.environ.get("HARBOR_REWARD_JSON",
                           "/logs/verifier/reward.json")), _json(reward)),
                          (Path(
                              os.environ.get("HARBOR_REWARD_TXT",
                                             "/logs/verifier/reward.txt")),
                           f"{score}\n".encode())):
        directory = _directory(path.absolute().parent, create=True)
        try:
            _write(directory, path.name, content)
        finally:
            os.close(directory)


def _relative_parts(name: str) -> tuple[str, ...]:
    if (not isinstance(name, str) or not name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or "\x00" in name):
        raise ValueError("unsafe relative path")
    return tuple(name.split("/"))


def _read(directory_fd: int, name: str, limit: int) -> bytes:
    parts = _relative_parts(name)
    parent = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, DIRECTORY_FLAGS, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1],
                             os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(
                    before.st_mode
            ) or before.st_nlink != 1 or before.st_size > limit:
                raise ValueError("unsafe or oversized evidence file")
            content = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if (len(content) > limit or len(content) != before.st_size
                or (before.st_mtime_ns, before.st_ctime_ns, 1)
                != (after.st_mtime_ns, after.st_ctime_ns, after.st_nlink)):
            raise ValueError("evidence changed during read")
        return content
    finally:
        os.close(parent)


def _validated_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or len(value) > MAX_ENTRIES:
        raise ValueError("invalid inventory")
    total = 0
    for name, record in value.items():
        parts = _relative_parts(name)
        if any(part in EXCLUDED_NAMES or part.endswith((".pyc", ".pyo"))
               for part in parts):
            raise ValueError("excluded entry in inventory")
        if any("/".join(parts[:end]) in value for end in range(1, len(parts))):
            raise ValueError("file/directory collision in inventory")
        if (not isinstance(record, dict)
                or set(record) != {"sha256", "size_bytes", "mode"}
                or type(record["size_bytes"]) is not int
                or not 0 <= record["size_bytes"] <= MAX_FILE_BYTES
                or type(record["mode"]) is not int
                or not 0 <= record["mode"] <= 0o7777
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in record["sha256"])):
            raise ValueError("invalid file inventory record")
        total += record["size_bytes"]
    if total > MAX_TREE_BYTES:
        raise ValueError("oversized inventory")
    return value


def reconstruct(source: str | Path, capture_dir: str | Path,
                output: str | Path) -> dict[str, Any]:
    """Verify and reconstruct a complete capture without importing any worker code.

    Read only frozen source and hashed blobs; never apply an untrusted patch
    command. All inputs are validated before a fresh output directory is made.
    The caller separately retains the trusted capture.json hash/provenance.
    """
    source, capture_dir, output = (Path(path).absolute()
                                   for path in (source, capture_dir, output))
    if any(output == path or output in path.parents or path in output.parents
           for path in (source, capture_dir)):
        raise ValueError("reconstruction output overlaps input")
    limits = {
        "max_file_bytes": MAX_FILE_BYTES,
        "max_tree_bytes": MAX_TREE_BYTES,
        "max_entries": MAX_ENTRIES,
        "max_judge_bytes": MAX_JUDGE_BYTES
    }
    original = _snapshot(source, "source", limits)
    if original["rejections"]:
        raise ValueError("source inventory is incomplete")
    directory = _directory(capture_dir)
    try:
        raw_manifest = _read(directory, "capture.json", 256 * 1024**2)
        manifest = json.loads(raw_manifest)
        if (not isinstance(manifest, dict)
                or manifest.get("status") != "complete"
                or manifest.get("schema_version") != 1
                or manifest.get("rejections") != []
                or manifest.get("unverified_deletions") != []):
            raise ValueError("capture must be complete")
        old = _validated_inventory(manifest.get("source_inventory"))
        new = _validated_inventory(manifest.get("worktree_inventory"))
        if (old != original["inventory"] or _digest(_json(old))
                != manifest.get("source_inventory_sha256") or _digest(
                    _json(new)) != manifest.get("worktree_inventory_sha256")):
            raise ValueError("source or captured inventory hash mismatch")
        changed = sorted(path for path in new if old.get(path) != new[path])
        if (manifest.get("changed_paths") != changed
                or manifest.get("added_paths") != sorted(set(new) - set(old))
                or manifest.get("deleted_paths")
                != sorted(set(old) - set(new))):
            raise ValueError("change lists do not match inventories")
        artifacts = manifest.get("artifacts")
        expected = {"files/" + path
                    for path in changed
                    } | {"changes.diff", "judge_evidence.json"}
        if (not isinstance(artifacts, list) or len(artifacts) != len(expected)
                or any(not isinstance(item, dict) for item in artifacts)
                or {item.get("path")
                    for item in artifacts} != expected):
            raise ValueError(
                "artifact manifest does not match captured changes")
        blobs: dict[str, bytes] = {}
        for item in artifacts:
            name = item["path"]
            limit = MAX_FILE_BYTES if name.startswith(
                "files/"
            ) else MAX_JUDGE_BYTES if name == "judge_evidence.json" else 256 * 1024**2
            raw = _read(directory, name, limit)
            if len(raw) != item.get("size_bytes") or _digest(raw) != item.get(
                    "sha256"):
                raise ValueError("captured artifact hash mismatch")
            if name.startswith("files/"):
                path = name.removeprefix("files/")
                if len(raw) != new[path]["size_bytes"] or _digest(
                        raw) != new[path]["sha256"]:
                    raise ValueError("blob differs from captured inventory")
                blobs[path] = raw
    finally:
        os.close(directory)
    parent = _directory(output.parent, create=True)
    try:
        os.mkdir(output.name, 0o755, dir_fd=parent)
        directory = os.open(output.name, DIRECTORY_FLAGS, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    try:
        for path, record in sorted(new.items()):
            content = blobs[path] if path in blobs else original["contents"][
                path]
            _write(directory, path, content)
            # _write creates every component itself under a fresh output. Reopen
            # with the same no-follow directory policy before applying its mode.
            parts = _relative_parts(path)
            file_parent = _directory(output.joinpath(*parts[:-1]))
            try:
                descriptor = os.open(parts[-1],
                                     os.O_RDONLY | os.O_NOFOLLOW,
                                     dir_fd=file_parent)
                try:
                    os.fchmod(descriptor, record["mode"])
                finally:
                    os.close(descriptor)
            finally:
                os.close(file_parent)
        os.fsync(directory)
    finally:
        os.close(directory)
    final = _snapshot(output, "reconstructed", limits)
    if final["rejections"] or final["inventory"] != new:
        raise ValueError("reconstructed inventory verification failed")
    return {
        "status": "complete",
        "worktree_inventory_sha256": _digest(_json(new)),
        "capture_json_sha256": _digest(raw_manifest),
        "file_count": len(new)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",
                        type=Path,
                        default=Path(
                            os.environ.get("HARBOR_CAPTURE_SOURCE",
                                           "/workspace/repo")))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--worktree", type=Path)
    group.add_argument(
        "--job-logs",
        type=Path,
        help="Recovered trial log directory containing agent/worktree")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get("HARBOR_CAPTURE_OUTPUT",
                           "/logs/verifier/artifacts/capture")))
    parser.add_argument("--reward",
                        action="store_true",
                        help="Write a capture-only Harbor reward")
    args = parser.parse_args(argv)
    worktree = args.worktree or (
        args.job_logs / "agent/worktree" if args.job_logs else Path(
            os.environ.get("HARBOR_CAPTURE_WORKTREE", "/logs/agent/worktree")))
    try:
        result = capture(args.source, worktree, args.output)
        if args.reward or Path(sys.argv[0]).name == "grader.py":
            _reward(result)
    except (OSError, ValueError) as exc:
        # Paths/content are intentionally absent from errors and never echoed.
        print(json.dumps({
            "status": "capture_error",
            "error_type": type(exc).__name__
        }),
              file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "changed_paths": result["changed_paths"],
                "deleted_paths": result["deleted_paths"],
                "capture_json": result["capture_json"]
            },
            sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
