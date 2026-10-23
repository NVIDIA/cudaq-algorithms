# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run official SkillEvaluator static/dataset gates and retain failure evidence.

This is only a process/report adapter. It never grades model outputs, changes
upstream findings, or claims that a repository run is official NVSkills CI.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def run_command(command, stdout, stderr):
    result = {"command": command, "exit_code": None, "error": None}
    try:
        with stdout.open("w", encoding="utf-8") as out, stderr.open(
                "w", encoding="utf-8") as err:
            process = subprocess.run(command,
                                     stdout=out,
                                     stderr=err,
                                     timeout=300)
        result["exit_code"] = process.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    for path in (stdout, stderr):
        if path.exists():
            print(path.read_text(encoding="utf-8", errors="replace"), end="")
    return result


def evaluate(skill, output, evaluator):
    skill = skill.resolve()
    output = output.resolve()
    repository = Path(__file__).resolve().parents[3]
    for protected in (skill, repository / "skills/cudaq-algorithms",
                      repository / "skills-dev/cudaq-algorithms-dev"):
        if output.is_relative_to(protected.resolve()):
            raise ValueError(
                "write evaluation reports outside input and shipped skill directories"
            )
    output.mkdir(parents=True, exist_ok=True)
    tier1 = output / "tier1"
    # Refuse to consume an earlier run's reports as evidence for this run.
    tier1.mkdir(exist_ok=False)
    report = {
        "scope":
        "Repository Tier 1 static and Tier 3 dataset checks; not official NVSkills CI or live model evaluation",
        "passed": False,
        "errors": [],
    }
    benchmark = output / "BENCHMARK.md"
    benchmark.write_text(
        "# Draft benchmark: INCOMPLETE\n\n"
        "The official evaluator did not produce a benchmark. See run.json and logs.\n"
        "Live tiers were not run. This is not NVSkills approval.\n",
        encoding="utf-8")
    report["tier1"] = run_command([
        evaluator, "validate",
        str(skill), "--tiers", "1", "--no-llm", "--no-dedup",
        "--continue-on-failure", "--report", "cli,json", "--output-dir",
        str(tier1)
    ], tier1 / "stdout.log", tier1 / "stderr.log")
    upstream_reports = list(tier1.glob("skillevaluator-output-*.json"))
    try:
        if len(upstream_reports) != 1:
            raise ValueError(
                "Expected exactly one official Tier 1 JSON report")
        payload = json.loads(upstream_reports[0].read_text(encoding="utf-8"))
        tier1_passed = (isinstance(payload, dict)
                        and payload.get("overall_passed") is True
                        and payload.get("overall_status") == "passed"
                        and payload.get("incomplete_scans") == []
                        and isinstance(payload.get("results"), list)
                        and bool(payload["results"]))
        if not tier1_passed:
            report["errors"].append(
                "Official Tier 1 report is failed, incomplete, or empty")
    except (OSError, ValueError) as exc:
        tier1_passed = False
        report["errors"].append(str(exc))
    upstream_benchmark = tier1 / "BENCHMARK.md"
    if upstream_benchmark.is_file() and upstream_benchmark.stat().st_size:
        shutil.copyfile(upstream_benchmark, benchmark)
    else:
        tier1_passed = False
        report["errors"].append(
            "Official Tier 1 BENCHMARK.md is missing or empty")

    dataset = output / "tier3-dataset.json"
    report["tier3_dataset"] = run_command(
        [evaluator, "tier3", "validate",
         str(skill), "--strict", "--json"], dataset,
        output / "tier3-stderr.log")
    try:
        checks = json.loads(dataset.read_text(encoding="utf-8"))
        dataset_passed = (isinstance(checks, list) and bool(checks) and all(
            isinstance(row, dict) and row.get("status") == "ok"
            for row in checks))
        if not dataset_passed:
            report["errors"].append(
                "Strict Tier 3 dataset report is not fully passing")
    except (OSError, ValueError) as exc:
        dataset_passed = False
        report["errors"].append(str(exc))
    report["passed"] = (tier1_passed and dataset_passed
                        and report["tier1"]["exit_code"] == 0
                        and report["tier3_dataset"]["exit_code"] == 0)
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill",
                        type=Path,
                        default=Path(__file__).resolve().parents[3] /
                        "skills/cudaq-algorithms")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evaluator", default="skillevaluator")
    args = parser.parse_args()
    try:
        return 0 if evaluate(args.skill, args.output_dir,
                             args.evaluator) else 1
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
