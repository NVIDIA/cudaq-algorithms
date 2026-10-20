# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check repository-specific skill packaging without rewriting frozen fixtures.

SkillEvaluator owns the catalog gate. This check preserves this repository's
metadata and runtime/development split, including requirements it only advises.
"""
import argparse
import json
from pathlib import Path
import re

import yaml


def check(root):
    runtime = root / "skills/cudaq-algorithms"
    dev = root / "skills-dev/cudaq-algorithms-dev"
    errors = []

    def fail(name, path, message):
        errors.append({
            "check": name,
            "path": str(path.relative_to(root)),
            "message": message
        })

    manifest = runtime / "SKILL.md"
    text = ""
    try:
        text = manifest.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        if parts[0].strip() or len(parts) != 3:
            raise ValueError("Expected YAML frontmatter")
        header = yaml.safe_load(parts[1])
        if not isinstance(header, dict):
            raise ValueError("Expected a frontmatter mapping")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        fail("frontmatter", manifest, str(exc))
        header = {}
    metadata = header.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if header.get("name") != runtime.name:
        fail("name", manifest, "Name must match the runtime directory")
    description = header.get("description")
    if not isinstance(description, str) or not 50 <= len(description) <= 150:
        fail("description", manifest,
             "Description must contain 50-150 characters")
    version = header.get("version")
    if (not isinstance(version, str) or re.fullmatch(
            r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version) is None
            or version != metadata.get("version")):
        fail(
            "version", manifest,
            "Top-level and metadata versions must be matching release versions"
        )
    if header.get("permissions") != ["file_read", "file_write", "shell"]:
        fail("permissions", manifest, "Expected file_read, file_write, shell")
    if header.get("license") != "Apache-2.0":
        fail("license", manifest, "Expected Apache-2.0")
    tags = header.get("tags")
    if (not isinstance(tags, list) or not tags
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
            or tags != metadata.get("tags")):
        fail("tags", manifest,
             "Nonempty top-level and metadata tags must match")
    # Match SkillEvaluator v0.5.0's documented estimate, not a tokenizer claim.
    estimated_tokens = len(text) // 4
    if estimated_tokens >= 5000:
        fail("size", manifest,
             "SKILL.md must remain below 5000 estimated tokens")

    required = [
        runtime / name
        for name in ("skill-card.md", "references/quick-contract.md",
                     "scripts/route.py", "scripts/check_calls.py",
                     "scripts/check_output.py", "evals/evals.json",
                     "evals/config.yml")
    ]
    required.append(dev / "authoring/architecture.md")
    for path in required:
        if not path.is_file():
            fail("placement", path, "Required package file is missing")
    old_dev = root / "skills/cudaq-algorithms-dev"
    # Ignored interpreter caches are not shipped package content.
    for path in old_dev.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            fail("placement", path,
                 "Development companion belongs in skills-dev/")
    for path in runtime.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(runtime)
        allowed = relative.parts[0] in {
            "SKILL.md", "skill-card.md", "references", "scripts", "evals"
        }
        if relative.parts[0] == "scripts":
            allowed = relative.as_posix() in {
                "scripts/route.py", "scripts/check_calls.py",
                "scripts/check_output.py"
            }
        if relative.parts[0] == "evals":
            allowed = (relative.as_posix() in {
                "evals/evals.json", "evals/config.yml", "evals/EVAL.md"
            } or relative.as_posix().startswith("evals/files/"))
        if not allowed:
            fail(
                "separation", path,
                "Private grading, authoring, and run evidence belong outside the runtime skill"
            )
    for tree in (runtime, dev):
        for path in tree.rglob("*.py"):
            relative = path.relative_to(tree)
            if relative.parts[:2] == ("evals", "files"):
                continue  # Fixture bytes are protected by the catalog manifest tests.
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()[:5]
            if (not any(
                    line.startswith("# SPDX-FileCopyrightText: ")
                    for line in lines)
                    or "# SPDX-License-Identifier: Apache-2.0" not in lines):
                fail("spdx", path,
                     "Repository SPDX copyright/license headers required")
    return {
        "passed": not errors,
        "estimated_tokens": estimated_tokens,
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",
                        type=Path,
                        default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    report = check(args.root.resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
