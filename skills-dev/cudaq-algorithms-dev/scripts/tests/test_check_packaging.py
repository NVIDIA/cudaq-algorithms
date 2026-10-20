# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise packaging regressions against disposable package trees."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_packaging.py"
HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
MANIFEST = """---
name: cudaq-algorithms
description: Use for CUDA-Q Algorithms primitives and molecular workflows; excludes installation and standalone kernels.
license: Apache-2.0
version: "0.2.0"
permissions: [file_read, file_write, shell]
tags: [cudaq-algorithms]
metadata:
  author: CUDA-Q Algorithms Team <cuda-quantum@nvidia.com>
  version: "0.2.0"
  tags: [cudaq-algorithms]
---
# CUDA-Q Algorithms
Runtime instructions.
"""


@pytest.fixture
def package(tmp_path):
    runtime = tmp_path / "skills/cudaq-algorithms"
    dev = tmp_path / "skills-dev/cudaq-algorithms-dev"
    for path, text in {
            runtime / "SKILL.md": MANIFEST,
            runtime / "skill-card.md": "Catalog entry\n",
            runtime / "scripts/route.py": HEADER + "print('route')\n",
            runtime / "scripts/check_calls.py": HEADER + "print('check calls')\n",
            runtime / "scripts/check_output.py": HEADER + "print('check output')\n",
            runtime / "references/quick-contract.md": "Public contract\n",
            runtime / "evals/evals.json": '{"evals": []}',
            runtime / "evals/config.yml": "schema_version: 1\n",
            runtime / "evals/EVAL.md": "Authored evaluation guidance\n",
            runtime / "evals/files/frozen.py":
            "# Deliberately unmodified fixture\n",
            dev / "authoring/architecture.md": "Maintainer instructions\n",
            dev / "scripts/check.py": HEADER,
            dev / "scripts/tests/test_check_output.py": HEADER,
            dev / "evals/files/frozen.py":
            "# Deliberately unmodified fixture\n",
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def run_check(root):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root",
         str(root)],
        capture_output=True,
        text=True)
    return result, json.loads(result.stdout) if result.stdout else {}


def test_valid_split_package_and_frozen_headerless_fixtures_pass(package):
    result, report = run_check(package)
    assert result.returncode == 0, result.stderr
    assert report["passed"] is True


@pytest.mark.parametrize("script",
                         ["route.py", "check_calls.py", "check_output.py"])
def test_missing_runtime_script_is_a_placement_failure(package, script):
    relative = "skills/cudaq-algorithms/scripts/" + script
    (package / relative).unlink()
    result, report = run_check(package)
    assert result.returncode == 1
    assert any(error["check"] == "placement" and error["path"] == relative
               for error in report["errors"])


@pytest.mark.parametrize("before,after,code", [
    ('version: "0.2.0"', 'version: "0.3.0"', "version"),
    ('version: "0.2.0"', 'version: "invalid"', "version"),
    ("permissions: [file_read, file_write, shell]", "permissions: [file_read]",
     "permissions"),
    ("tags: [cudaq-algorithms]", "tags: []", "tags"),
    ("license: Apache-2.0", "license: unknown", "license"),
    ("description: Use for", "description: x\nunused: Use for", "description"),
])
def test_metadata_regression_fails(package, before, after, code):
    manifest = package / "skills/cudaq-algorithms/SKILL.md"
    manifest.write_text(manifest.read_text().replace(before, after, 1))
    result, report = run_check(package)
    assert result.returncode == 1
    assert code in {error["check"] for error in report["errors"]}


def test_oversized_entrypoint_fails(package):
    manifest = package / "skills/cudaq-algorithms/SKILL.md"
    manifest.write_text(manifest.read_text() + "x" * 20000)
    result, report = run_check(package)
    assert result.returncode == 1
    assert "size" in {error["check"] for error in report["errors"]}


@pytest.mark.parametrize("relative,content,code", [
    ("skills/cudaq-algorithms/scripts/route.py", "print('route')", "spdx"),
    ("skills/cudaq-algorithms/scripts/check_calls.py", "print('check')", "spdx"),
    ("skills/cudaq-algorithms/scripts/check_output.py", "print('check')", "spdx"),
    ("skills-dev/cudaq-algorithms-dev/scripts/new.py", "pass", "spdx"),
    ("skills/cudaq-algorithms-dev/authoring.md", "Maintainer material",
     "placement"),
    ("skills/cudaq-algorithms/scripts/report_eval.py", HEADER, "separation"),
    ("skills/cudaq-algorithms/scripts/check_packaging.py", HEADER, "separation"),
    ("skills/cudaq-algorithms/scripts/tests/test_check_calls.py", HEADER,
     "separation"),
    ("skills/cudaq-algorithms/scripts/tests/test_check_output.py", HEADER,
     "separation"),
    ("skills/cudaq-algorithms/evals/runner/cli.py", HEADER, "separation"),
    ("skills/cudaq-algorithms/coverage/features.json", "{}", "separation"),
])
def test_layout_and_license_regressions_fail(package, relative, content, code):
    path = package / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    result, report = run_check(package)
    assert result.returncode == 1
    assert code in {error["check"] for error in report["errors"]}


def test_malformed_frontmatter_is_a_reported_failure(package):
    (package / "skills/cudaq-algorithms/SKILL.md").write_text("---\n[\n---")
    result, report = run_check(package)
    assert result.returncode == 1
    assert "frontmatter" in {error["check"] for error in report["errors"]}


def test_relocated_optional_maintainer_skill_is_allowed(package):
    (package / "skills-dev/cudaq-algorithms-dev/SKILL.md").write_text(
        "Optional maintainer instructions outside the runtime catalog\n")
    result, report = run_check(package)
    assert result.returncode == 0
    assert report["passed"] is True


def test_evals_regular_file_returns_structured_placement_failure(package):
    evals = package / "skills/cudaq-algorithms/evals"
    shutil.rmtree(evals)
    evals.write_text("Misplaced regular file\n")
    result, report = run_check(package)
    assert result.returncode == 1
    assert report["passed"] is False
    assert "placement" in {error["check"] for error in report["errors"]}
    assert "Traceback" not in result.stderr
