# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired, all-attempt measurements with explicit missing-data denominators."""
from __future__ import annotations

import statistics


def distribution(values):
    values = [v for v in values if v is not None]
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sum": sum(values) if values else None
    }


def scientific_api_passed(result):
    checks = result.get("grading", {}).get("checks", [])
    return len(checks) == 2 and all(
        c.get("returncode") == 0 and all(
            c.get(key, {}).get("passed") is True
            for key in ("numeric", "host_api", "device_api")) for c in checks)


def paired_comparisons(results, arms, planned_runs=None):
    indexed = {(r["case"], r["repetition"], r["arm"]): r for r in results}
    if len(indexed) != len(results):
        raise ValueError("duplicate case/repetition/arm results")
    blocks = sorted({
        (r["case"], r["repetition"])
        for r in (planned_runs if planned_runs is not None else results)
    })
    comparisons = {}
    for control in ("previous", "baseline"):
        if control not in arms or "skill" not in arms:
            continue
        paired = []
        missing = blocked = 0
        for case, repetition in blocks:
            candidate = indexed.get((case, repetition, "skill"))
            reference = indexed.get((case, repetition, control))
            if candidate is None or reference is None:
                missing += 1
                continue
            if not candidate.get("attempted", True) or not reference.get(
                    "attempted", True):
                blocked += 1
                continue
            metrics = {}
            for field in ("input_tokens", "cached_input_tokens",
                          "output_tokens", "reasoning_output_tokens",
                          "total_tokens"):
                left = candidate.get("telemetry", {}).get("totals",
                                                          {}).get(field)
                right = reference.get("telemetry", {}).get("totals",
                                                           {}).get(field)
                metrics[
                    field] = left - right if left is not None and right is not None else None
            for field in ("elapsed_s", "time_to_verified_solution_s"):
                left, right = candidate.get(field), reference.get(field)
                eligible = field != "time_to_verified_solution_s" or (
                    candidate["passed"] and reference["passed"])
                metrics[
                    field] = left - right if eligible and left is not None and right is not None else None
            field = "observed_peak_request_input_tokens"
            left = candidate.get("telemetry", {}).get(field)
            right = reference.get("telemetry", {}).get(field)
            metrics[
                field] = left - right if left is not None and right is not None else None
            metrics["strict_pass_delta"] = int(candidate["passed"]) - int(
                reference["passed"])
            metrics["scientific_api_pass_delta"] = int(
                scientific_api_passed(candidate)) - int(
                    scientific_api_passed(reference))
            paired.append({
                "case": case,
                "repetition": repetition,
                "candidate_id": candidate["id"],
                "control_id": reference["id"],
                "observed_peak_completeness": {
                    "candidate":
                    candidate.get("telemetry", {}).get("completeness",
                                                       {}).get(field),
                    "control":
                    reference.get("telemetry", {}).get("completeness",
                                                       {}).get(field)
                },
                **metrics
            })
        fields = ("input_tokens", "cached_input_tokens", "output_tokens",
                  "reasoning_output_tokens", "total_tokens",
                  "observed_peak_request_input_tokens", "elapsed_s",
                  "time_to_verified_solution_s", "strict_pass_delta",
                  "scientific_api_pass_delta")
        comparisons["candidate_minus_" + control] = {
            "paired_attempts": len(paired),
            "missing_pairs": missing,
            "blocked_pairs": blocked,
            **{
                field: distribution([p[field] for p in paired])
                for field in fields
            }, "pairs": paired
        }
    return comparisons
