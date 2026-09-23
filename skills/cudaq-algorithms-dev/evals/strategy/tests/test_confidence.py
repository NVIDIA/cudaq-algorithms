# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

STRATEGY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strategy_confidence",
                                              STRATEGY / "confidence.py")
if (STRATEGY / "confidence.py").is_file():
    confidence = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(confidence)
else:
    confidence = None

METRICS = ("task_time_s", "total_tokens", "peak_request_input_tokens_proxy")
MANIFEST_A = "a" * 64
MANIFEST_B = "b" * 64


def module():
    assert confidence is not None, "confidence calculation has not been implemented"
    return confidence


def pair(case_id,
         repetition,
         planned_repetitions,
         kind,
         baseline,
         candidate=None):
    candidate = baseline if candidate is None else candidate
    return {
        "case_id": case_id,
        "repetition": repetition,
        "planned_repetitions": planned_repetitions,
        "kind": kind,
        "baseline": {
            name: baseline
            for name in METRICS
        },
        "candidate": {
            name: candidate
            for name in METRICS
        }
    }


def cohort():
    rows = [
        pair("T01", 1, 1, "trigger", 1),
        pair("T02", 1, 1, "trigger", 100),
        pair("E01", 1, 1, "execution", 2),
        pair("E06", 1, 1, "execution", 80)
    ]
    for case_id, value in (("E02", 4), ("E03", 60)):
        rows.extend(
            pair(case_id, repetition, 3, "execution", value)
            for repetition in range(1, 4))
    return rows


def test_paired_case_bootstrap_keeps_identical_heterogeneous_arms_at_one():
    result = module().bootstrap_median_ratio_bounds(cohort(),
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=999,
                                                    batch_size=37)

    assert result["status"] == "uncalibrated"
    assert result["statistical_pass"] is None
    for metric in METRICS:
        assert result["bounds"][metric]["point_ratio"] == 1.0
        assert result["bounds"][metric]["upper_confidence_bound"] == 1.0
    assert result["coverage"]["calibrated"] is False
    assert "approximate" in result["coverage"]["interpretation"]


def test_first_draw_carries_every_repetition_of_a_sampled_case_together():
    result = module().bootstrap_median_ratio_bounds(cohort(),
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=11,
                                                    batch_size=4)

    repeated = result["diagnostics"]["first_draw_pair_ids_by_stratum"][
        "execution_three_run"]
    assert len(repeated) == 6
    for offset in (0, 3):
        block = repeated[offset:offset + 3]
        assert len({item["case_id"] for item in block}) == 1
        assert [item["repetition"] for item in block] == [1, 2, 3]


def test_point_estimator_is_ratio_of_arm_medians_not_median_of_ratios():
    rows = [
        pair("A", 1, 1, "trigger", 1, 1),
        pair("B", 1, 1, "trigger", 100, 1),
        pair("C", 1, 1, "trigger", 100, 100)
    ]

    assert module().ratio_of_arm_medians(rows,
                                         "task_time_s") == pytest.approx(0.01)
    assert sorted([1 / 1, 1 / 100, 100 / 100])[1] == 1


def test_manifest_seed_and_bootstrap_output_are_deterministic_across_batch_sizes(
):
    first = module().bootstrap_median_ratio_bounds(cohort(),
                                                   manifest_sha256=MANIFEST_A,
                                                   draws=257,
                                                   batch_size=7)
    second = module().bootstrap_median_ratio_bounds(cohort(),
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=257,
                                                    batch_size=113)
    different = module().bootstrap_median_ratio_bounds(
        cohort(), manifest_sha256=MANIFEST_B, draws=257, batch_size=7)

    assert first["seed"] == second["seed"]
    assert first["bounds"] == second["bounds"]
    assert first["diagnostics"]["draw_sha256"] == second["diagnostics"][
        "draw_sha256"]
    assert first["seed"] != different["seed"]
    assert first["diagnostics"]["draw_sha256"] != different["diagnostics"][
        "draw_sha256"]


def test_discrete_even_sample_uses_average_of_middle_values():
    rows = [pair("A", 1, 1, "trigger", 1, 2), pair("B", 1, 1, "trigger", 3, 4)]

    assert module().ratio_of_arm_medians(rows, "total_tokens") == 1.5


@pytest.mark.parametrize("rows, reason", [
    ([], "no_eligible_pairs"),
    ([pair("T01", 1, 1, "trigger", 1)], "missing_stratum"),
    (cohort() + [pair("T01", 1, 1, "trigger", 2)], "duplicate_pair_id"),
    ([{
        **row, "baseline": {
            **row["baseline"], "task_time_s": 0
        }
    } if row["case_id"] == "T01" else row
      for row in cohort()], "nonpositive_baseline"),
])
def test_invalid_or_non_resampleable_cohort_is_unknown(rows, reason):
    result = module().bootstrap_median_ratio_bounds(rows,
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=17,
                                                    batch_size=5)

    assert result["status"] == "unknown"
    assert result["statistical_pass"] is None
    assert reason in result["diagnostics"]["invalid_reasons"]
    assert all(value["upper_confidence_bound"] is None
               for value in result["bounds"].values())


def test_frozen_strata_use_kind_and_planned_repetitions_not_observed_count():
    rows = cohort()
    rows = [
        row for row in rows
        if not (row["case_id"] == "E02" and row["repetition"] == 3)
    ]
    result = module().bootstrap_median_ratio_bounds(rows,
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=31,
                                                    batch_size=8)

    assert result["status"] == "uncalibrated"
    support = result["diagnostics"]["support_counts"]
    assert support["execution_three_run"] == {"cases": 2, "eligible_pairs": 5}
    assert result["cohort"]["pair_ids"][-2:] == [{
        "case_id": "E03",
        "repetition": 2
    }, {
        "case_id": "E03",
        "repetition": 3
    }]


def test_invalid_frozen_stratum_identity_is_unknown():
    rows = cohort()
    rows[0] = {**rows[0], "planned_repetitions": 3}

    result = module().bootstrap_median_ratio_bounds(rows,
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=17)

    assert result["status"] == "unknown"
    assert "invalid_stratum_identity" in result["diagnostics"][
        "invalid_reasons"]


@pytest.mark.parametrize("planned_repetitions", [True, 1.0])
def test_frozen_stratum_rejects_non_integer_planned_repetitions(
        planned_repetitions):
    rows = cohort()
    rows[0] = {**rows[0], "planned_repetitions": planned_repetitions}

    result = module().bootstrap_median_ratio_bounds(rows,
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=17)

    assert result["status"] == "unknown"
    assert "invalid_stratum_identity" in result["diagnostics"][
        "invalid_reasons"]


def test_public_point_estimator_rejects_an_invalid_member_hidden_by_the_median(
):
    rows = [
        pair("A", 1, 1, "trigger", -1, 1),
        pair("B", 1, 1, "trigger", 2, 2),
        pair("C", 1, 1, "trigger", 3, 3)
    ]

    with pytest.raises(ValueError, match="baseline members"):
        module().ratio_of_arm_medians(rows, "total_tokens")


@pytest.mark.parametrize(
    "overflow", ["point_median", "bootstrap_ratio", "array_conversion"])
def test_finite_members_that_overflow_aggregation_fail_closed_with_json_finite_output(
        overflow):
    rows = cohort()
    if overflow == "point_median":
        rows = [{
            **row, "candidate": {
                name: 1e308
                for name in METRICS
            }
        } for row in rows]
    elif overflow == "bootstrap_ratio":
        extreme_cases = {"T01", "E01", "E02"}
        rows = [{
            **row, "baseline": {
                name: 1e-308
                for name in METRICS
            },
            "candidate": {
                name: 1e308
                for name in METRICS
            }
        } if row["case_id"] in extreme_cases else row for row in rows]
    else:
        rows = [{
            **row, "baseline": {
                name: 10**1000
                for name in METRICS
            },
            "candidate": {
                name: 10**1000
                for name in METRICS
            }
        } for row in rows]

    result = module().bootstrap_median_ratio_bounds(rows,
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=999,
                                                    batch_size=37)

    assert result["status"] == "unknown"
    assert "nonfinite_ratio" in result["diagnostics"]["invalid_reasons"]
    assert all(value["upper_confidence_bound"] is None
               for value in result["bounds"].values())
    json.dumps(result, allow_nan=False)


def test_nonfinite_control_input_also_returns_json_finite_unknown_output():
    result = module().bootstrap_median_ratio_bounds(cohort(),
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=float("nan"))

    assert result["status"] == "unknown"
    assert "invalid_draw_count" in result["diagnostics"]["invalid_reasons"]
    json.dumps(result, allow_nan=False)


def test_noniterable_pair_input_returns_unknown_instead_of_raising():
    result = module().bootstrap_median_ratio_bounds(None,
                                                    manifest_sha256=MANIFEST_A,
                                                    draws=17)

    assert result["status"] == "unknown"
    assert result["statistical_pass"] is None
    assert result["diagnostics"]["invalid_reasons"] == ["noniterable_pairs"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("controls, reason", [
    ({
        "draws": 10**100
    }, "draw_count_exceeds_limit"),
    ({
        "batch_size": 8193
    }, "batch_size_exceeds_limit"),
])
def test_overlarge_work_controls_fail_before_bootstrap_allocation(
        controls, reason):
    result = module().bootstrap_median_ratio_bounds(cohort(),
                                                    manifest_sha256=MANIFEST_A,
                                                    **controls)

    assert result["status"] == "unknown"
    assert result["draws"] is None if "draws" in controls else result[
        "draws"] == 99999
    assert reason in result["diagnostics"]["invalid_reasons"]
    assert all(value["upper_confidence_bound"] is None
               for value in result["bounds"].values())
