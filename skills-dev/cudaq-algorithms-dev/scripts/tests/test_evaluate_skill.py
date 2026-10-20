# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CI exit/report fidelity at the external evaluator process boundary."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_skill.py"


@pytest.fixture
def evaluator(tmp_path):
    # The real v0.5.0 CLI writes an aggregate report and BENCHMARK for Tier 1,
    # and prints a list of path/status/message records for dataset validation.
    # External scans are replaced here; subprocess execution and report handling
    # remain real. Run the official tool separately for scanner coverage.
    executable = tmp_path / "skillevaluator"
    executable.write_text(f"#!{sys.executable}\n" + '''
import json
from pathlib import Path
import sys
Path(__file__).with_suffix(".invoked").touch()
args = sys.argv[1:]
if args[0] == "validate":
    mode = Path(args[1]).name
    assert args[2:9] == ["--tiers", "1", "--no-llm", "--no-dedup", "--continue-on-failure", "--report", "cli,json"]
    out = Path(args[args.index("--output-dir") + 1])
    if mode == "crash":
        sys.exit(2)
    report = {"overall_passed": mode != "failed", "overall_status": "passed",
              "incomplete_scans": [], "results": [{"validator": "Schema", "passed": True}]}
    if mode == "incomplete":
        report["incomplete_scans"] = ["semgrep"]
        report["overall_status"] = "incomplete"
    if mode == "empty":
        report["results"] = []
    if mode != "missing-json":
        (out / "skillevaluator-output-test.json").write_text(json.dumps(report))
    if mode != "missing-benchmark":
        (out / "BENCHMARK.md").write_text("# Official draft benchmark\\nLive tiers: not run\\n")
    sys.exit(1 if mode == "exit-failure" else 0)
assert args[:2] == ["tier3", "validate"] and args[3:] == ["--strict", "--json"]
mode = Path(args[2]).name
if mode == "dataset-malformed":
    print("not JSON")
else:
    print(json.dumps([{"path": "evals/evals.json", "status": "warning" if mode == "dataset-warning" else "ok", "message": "Dataset checked"}]))
''')
    executable.chmod(0o755)
    return executable


def run_evaluation(tmp_path, evaluator, mode):
    skill = tmp_path / mode
    skill.mkdir()
    output = tmp_path / "reports"
    result = subprocess.run([
        sys.executable,
        str(SCRIPT), "--skill",
        str(skill), "--evaluator",
        str(evaluator), "--output-dir",
        str(output)
    ],
                            capture_output=True,
                            text=True)
    return result, output


def test_success_requires_both_real_process_results_and_reports(
        tmp_path, evaluator):
    result, output = run_evaluation(tmp_path, evaluator, "success")
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "run.json").read_text())
    assert report["passed"] is True
    assert report["tier1"]["exit_code"] == 0
    assert report["tier3_dataset"]["exit_code"] == 0
    assert (output / "BENCHMARK.md").read_text().startswith("# Official draft")


@pytest.mark.parametrize("mode", [
    "failed", "incomplete", "empty", "missing-json", "missing-benchmark",
    "exit-failure", "crash", "dataset-warning", "dataset-malformed"
])
def test_failures_keep_artifacts_and_never_skip_dataset_validation(
        tmp_path, evaluator, mode):
    result, output = run_evaluation(tmp_path, evaluator, mode)
    assert result.returncode == 1, result.stderr
    report = json.loads((output / "run.json").read_text())
    assert report["passed"] is False
    assert report["tier3_dataset"]["exit_code"] == 0
    assert (output / "tier3-dataset.json").is_file()
    assert (output / "BENCHMARK.md").is_file()
    if mode == "missing-benchmark":
        assert "INCOMPLETE" in (output / "BENCHMARK.md").read_text()


def test_missing_evaluator_keeps_explicit_incomplete_results(tmp_path):
    result, output = run_evaluation(tmp_path, tmp_path / "not-installed",
                                    "skill")
    assert result.returncode == 1
    report = json.loads((output / "run.json").read_text())
    assert report["passed"] is False
    assert report["tier1"]["exit_code"] is None
    assert "INCOMPLETE" in (output / "BENCHMARK.md").read_text()


@pytest.mark.parametrize("tree", ["runtime", "development", "input"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("installed", [False, True])
def test_protected_output_rejected_before_writes_or_evaluator_calls(
        tmp_path, evaluator, tree, alias, nested, installed):
    # Copy the real script so regressions can never write into the real sources.
    repository = tmp_path / "repository"
    script = repository / "skills-dev/cudaq-algorithms-dev/scripts/evaluate_skill.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    runtime = repository / "skills/cudaq-algorithms"
    runtime.mkdir(parents=True)
    skill = tmp_path / "success"
    skill.mkdir()
    protected = {
        "runtime": runtime,
        "development": script.parent.parent,
        "input": skill
    }[tree]
    (protected / "sentinel.txt").write_text("Preserve source bytes\n")
    before = {
        str(path.relative_to(protected)):
        path.read_bytes() if path.is_file() else None
        for path in protected.rglob("*")
    }
    destination = protected
    if alias:
        destination = tmp_path / "output-alias"
        destination.symlink_to(protected, target_is_directory=True)
    if nested:
        destination = destination / "new-reports"
    command = evaluator if installed else tmp_path / "not-installed"
    result = subprocess.run([
        sys.executable,
        str(script), "--skill",
        str(skill), "--evaluator",
        str(command), "--output-dir",
        str(destination)
    ],
                            capture_output=True,
                            text=True)
    assert result.returncode != 0
    assert "outside" in result.stderr
    assert not evaluator.with_suffix(".invoked").exists()
    after = {
        str(path.relative_to(protected)):
        path.read_bytes() if path.is_file() else None
        for path in protected.rglob("*")
    }
    assert after == before
