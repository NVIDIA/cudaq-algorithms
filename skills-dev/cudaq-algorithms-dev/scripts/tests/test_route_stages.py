# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional routing branches preserve default delivery and complete sections."""
import importlib.util
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / "skills/cudaq-algorithms"
SCRIPT = SKILL / "scripts/route.py"


def run(*args, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=cwd,
                          capture_output=True,
                          text=True)


def test_default_delivery_remains_the_full_bounded_contract():
    result = run("Compare Trotter evolution with dense exact dynamics.")
    assert result.returncode == 0
    assert "## Selected reference text" in result.stdout
    assert "## Compact reference bundle" not in result.stdout


def test_plan_selects_branches_without_emitting_a_contract_bundle():
    result = run("--stage", "plan", "Compare Trotter dynamics.")
    assert result.returncode == 0
    assert "trotter/trotter.md" in result.stdout
    assert "--record PATH --section HEADING" in result.stdout
    assert "## Selected reference text" not in result.stdout
    assert "## Compact reference bundle" not in result.stdout


@pytest.mark.parametrize("args", [
    ["--compact"],
    ["--stage", "contract"],
    ["--stage", "verify"],
])
def test_explicit_bundle_modes_advertise_missing_sections(args):
    result = run(*args,
                 "Compare Trotter dynamics with independent validation.")
    assert result.returncode == 0
    assert "## Compact reference bundle" in result.stdout
    assert "--record" in result.stdout
    assert "<skill-dir>" not in result.stdout


def test_exact_section_fetch_returns_complete_source_from_another_cwd(
        tmp_path):
    text = (SKILL / "references/trotter/trotter.md").read_text()
    expected = re.search(r"(?ms)^## Workflow\n.*?(?=^## |\Z)", text).group(0)
    result = run("--record",
                 "trotter/trotter.md",
                 "--section",
                 "Workflow",
                 cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith(expected.rstrip())


def test_generated_fetch_command_quotes_a_real_script_path(tmp_path):
    skill = tmp_path / "skill with spaces $literal"
    (skill / "scripts").mkdir(parents=True)
    (skill / "references/trotter").mkdir(parents=True)
    script = skill / "scripts/route.py"
    shutil.copyfile(SCRIPT, script)
    shutil.copyfile(SKILL / "references/trotter/trotter.md",
                    skill / "references/trotter/trotter.md")
    spec = importlib.util.spec_from_file_location("spaced_router", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.section_command("trotter/trotter.md", "Workflow")
    assert shlex.split(command)[:2] == [sys.executable, str(script)]
    result = subprocess.run(command,
                            shell=True,
                            cwd="/",
                            capture_output=True,
                            text=True)
    assert result.returncode == 0
    assert "## Workflow" in result.stdout


@pytest.fixture
def isolated_router(tmp_path):
    skill = tmp_path / "skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "references").mkdir()
    script = skill / "scripts/route.py"
    shutil.copyfile(SCRIPT, script)
    spec = importlib.util.spec_from_file_location("isolated_router", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_section_fetch_rejects_symlink_outside_references(
        isolated_router, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("## Contract\n\nOUTSIDE_REFERENCE_MARKER\n")
    (isolated_router.REFS / "linked.md").symlink_to(outside)
    result = subprocess.run([
        sys.executable, isolated_router.__file__, "--record", "linked.md",
        "--section", "Contract"
    ],
                            capture_output=True,
                            text=True)
    assert result.returncode == 2
    assert "OUTSIDE_REFERENCE_MARKER" not in result.stdout
    assert "outside" in result.stderr.lower()


def test_compact_bundle_preserves_prose_immediately_after_heading(
        isolated_router):
    (isolated_router.REFS / "inline.md").write_text(
        "## Contract\nThis complete requirement must remain visible.\n")
    output = isolated_router.compact_context(["inline.md"],
                                             "Use this contract")
    assert "This complete requirement must remain visible." in output


@pytest.mark.parametrize("body, message", [
    ("## Contract\n\nFirst.\n\n## Contract\n\nSecond.\n", "unique heading"),
    ("## Contract\n\n" + "long requirement " * 1100, "exceeds 16000"),
])
def test_section_fetch_rejects_ambiguous_or_oversized_sections(
        isolated_router, body, message):
    (isolated_router.REFS / "contract.md").write_text(body)
    result = subprocess.run([
        sys.executable, isolated_router.__file__, "--record", "contract.md",
        "--section", "Contract"
    ],
                            capture_output=True,
                            text=True)
    assert result.returncode == 2
    assert message in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("args", [
    ["--record", "../../SKILL.md", "--section", "Workflow"],
    ["--record", "/etc/passwd", "--section", "Workflow"],
    ["--record", "trotter/trotter.md", "--section", "missing"],
    ["--record", "trotter/trotter.md"],
    ["--stage", "plan", "--full", "Trotter"],
])
def test_invalid_or_conflicting_fetch_arguments_are_rejected(args):
    assert run(*args).returncode == 2
