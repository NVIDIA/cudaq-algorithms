#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired case-cluster bootstrap bounds for the strategy efficiency metrics.

This module calculates an approximate, deliberately uncalibrated confidence
bound.  It does not make an acceptance decision; calibration against a frozen
simulation envelope is a separate prerequisite for using the bound as evidence.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import statistics
from typing import Iterable, Mapping

import numpy as np

METRICS = ("task_time_s", "total_tokens", "peak_request_input_tokens_proxy")
STRATA = ("trigger", "execution_one_run", "execution_three_run")
# Production resampling controls are deliberately bounded before allocation.
DEFAULT_DRAWS = 99_999
MAX_DRAWS = 99_999
DEFAULT_BATCH_SIZE = 2_048
MAX_BATCH_SIZE = 8_192
CONFIDENCE_LEVEL = 0.95
_METHOD_VERSION = "paired-case-cluster-percentile-v1"


def ratio_of_arm_medians(pairs: Iterable[Mapping], metric: str) -> float:
    """Return median(candidate) / median(baseline), never median(pair ratios)."""
    rows = list(pairs)
    if not rows:
        raise ValueError("no eligible pairs")
    baseline, candidate = [], []
    for row in rows:
        try:
            baseline_value = row["baseline"][metric]
            candidate_value = row["candidate"][metric]
        except (KeyError, TypeError):
            raise ValueError(
                "every pair must contain both arm metrics") from None
        if not _positive(baseline_value):
            raise ValueError(
                "all baseline members must be finite and positive")
        if not _nonnegative(candidate_value):
            raise ValueError(
                "all candidate members must be finite and nonnegative")
        baseline.append(baseline_value)
        candidate.append(candidate_value)
    denominator = statistics.median(baseline)
    if not _positive(denominator):
        raise ValueError("baseline median must be finite and positive")
    numerator = statistics.median(candidate)
    if not _nonnegative(numerator):
        raise ValueError("candidate median must be finite and nonnegative")
    ratio = float(numerator / denominator)
    if not math.isfinite(ratio):
        raise ValueError("median ratio must be finite")
    return ratio


def _positive(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    return isinstance(value, float) and math.isfinite(value) and value > 0


def _nonnegative(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def _stratum(row: Mapping) -> str | None:
    kind, planned = row.get("kind"), row.get("planned_repetitions")
    if type(planned) is not int:
        return None
    if kind == "trigger" and planned == 1:
        return "trigger"
    if kind == "execution" and planned == 1:
        return "execution_one_run"
    if kind == "execution" and planned == 3:
        return "execution_three_run"
    return None


def _seed(manifest_sha256: str) -> int | None:
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF"
                   for character in manifest_sha256)):
        return None
    digest = hashlib.sha256(
        f"{_METHOD_VERSION}:{manifest_sha256.lower()}".encode(
            "ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _empty_result(*, draws, seed, reasons, support=None, cohort=None):
    safe_draws = draws if type(
        draws) is int and 1 <= draws <= MAX_DRAWS else None
    return {
        "schema_version": 1,
        "method": _METHOD_VERSION,
        "status": "unknown",
        "statistical_pass": None,
        "confidence_level": CONFIDENCE_LEVEL,
        "draws": safe_draws,
        "seed": seed,
        "bounds": {
            metric: {
                "point_ratio": None,
                "upper_confidence_bound": None
            }
            for metric in METRICS
        },
        "coverage": {
            "calibrated":
            False,
            "interpretation":
            "approximate percentile coverage is uncalibrated; no confidence claim is established.",
        },
        "cohort": cohort or {
            "eligible_pairs": 0,
            "cases": 0,
            "pair_ids": []
        },
        "assumptions": _assumptions(),
        "diagnostics": {
            "invalid_reasons": sorted(set(reasons)),
            "support_counts": support or {},
            "warnings": ["inference_not_computed"]
        },
    }


def _assumptions():
    return [
        "Cases are independent and exchangeable within each frozen schedule stratum.",
        "All jointly eligible repetitions of a case are resampled together and may be dependent.",
        "The target is similar future tasks conditional on joint success and complete measurement.",
        "The percentile interval is approximate and uncalibrated until a separate frozen simulation envelope is established.",
    ]


def _prepare(pairs):
    rows = list(pairs)
    reasons = []
    grouped = {name: defaultdict(list) for name in STRATA}
    identities = set()
    case_metadata = {}
    for row in rows:
        if not isinstance(row, Mapping):
            reasons.append("invalid_pair")
            continue
        case_id, repetition = row.get("case_id"), row.get("repetition")
        planned, stratum = row.get("planned_repetitions"), _stratum(row)
        if stratum is None:
            reasons.append("invalid_stratum_identity")
            continue
        if (not isinstance(case_id, str) or not case_id
                or not isinstance(repetition, int)
                or isinstance(repetition, bool) or repetition < 1
                or repetition > planned):
            reasons.append("invalid_pair_id")
            continue
        identity = (case_id, repetition)
        if identity in identities:
            reasons.append("duplicate_pair_id")
            continue
        identities.add(identity)
        metadata = (stratum, planned)
        if case_id in case_metadata and case_metadata[case_id] != metadata:
            reasons.append("inconsistent_case_metadata")
            continue
        case_metadata[case_id] = metadata
        baseline, candidate = row.get("baseline"), row.get("candidate")
        if not isinstance(baseline, Mapping) or not isinstance(
                candidate, Mapping):
            reasons.append("missing_metrics")
            continue
        bad = False
        for metric in METRICS:
            if not _positive(baseline.get(metric)):
                reasons.append("nonpositive_baseline")
                bad = True
            if not _nonnegative(candidate.get(metric)):
                reasons.append("invalid_candidate")
                bad = True
        if not bad:
            grouped[stratum][case_id].append(row)
    if not rows:
        reasons.append("no_eligible_pairs")
    for stratum in STRATA:
        if not grouped[stratum]:
            reasons.append("missing_stratum")
        for case_rows in grouped[stratum].values():
            case_rows.sort(key=lambda row: row["repetition"])
            if len(case_rows) > case_rows[0]["planned_repetitions"]:
                reasons.append("too_many_repetitions")
    ordered = {
        stratum:
        [grouped[stratum][case_id] for case_id in sorted(grouped[stratum])]
        for stratum in STRATA
    }
    support = {
        stratum: {
            "cases": len(ordered[stratum]),
            "eligible_pairs": sum(len(case) for case in ordered[stratum])
        }
        for stratum in STRATA
    }
    flat = [
        row for stratum in STRATA for case in ordered[stratum] for row in case
    ]
    cohort = {
        "eligible_pairs":
        len(flat),
        "cases":
        sum(item["cases"] for item in support.values()),
        "pair_ids": [{
            "case_id": row["case_id"],
            "repetition": row["repetition"]
        } for row in flat]
    }
    return ordered, flat, support, cohort, reasons


def _stratum_array(cases):
    width = max(len(case) for case in cases)
    values = np.full((len(cases), width, 2, len(METRICS)),
                     np.nan,
                     dtype=np.float64)
    for case_index, case in enumerate(cases):
        for repetition_index, row in enumerate(case):
            for metric_index, metric in enumerate(METRICS):
                values[case_index, repetition_index, 0,
                       metric_index] = row["baseline"][metric]
                values[case_index, repetition_index, 1,
                       metric_index] = row["candidate"][metric]
    return values


def bootstrap_median_ratio_bounds(
        pairs: Iterable[Mapping],
        *,
        manifest_sha256: str,
        draws: int = DEFAULT_DRAWS,
        batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Compute coupled one-sided percentile bounds without making a gate decision."""
    seed = _seed(manifest_sha256)
    parameter_reasons = []
    if seed is None:
        parameter_reasons.append("invalid_manifest_sha256")
    if type(draws) is not int or draws < 1:
        parameter_reasons.append("invalid_draw_count")
    elif draws > MAX_DRAWS:
        parameter_reasons.append("draw_count_exceeds_limit")
    if type(batch_size) is not int or batch_size < 1:
        parameter_reasons.append("invalid_batch_size")
    elif batch_size > MAX_BATCH_SIZE:
        parameter_reasons.append("batch_size_exceeds_limit")
    if parameter_reasons:
        return _empty_result(draws=draws, seed=seed, reasons=parameter_reasons)
    try:
        pair_rows = list(pairs)
    except TypeError:
        return _empty_result(draws=draws,
                             seed=seed,
                             reasons=["noniterable_pairs"])
    ordered, flat, support, cohort, reasons = _prepare(pair_rows)
    if reasons:
        return _empty_result(draws=draws,
                             seed=seed,
                             reasons=reasons,
                             support=support,
                             cohort=cohort)

    try:
        point = {
            metric: ratio_of_arm_medians(flat, metric)
            for metric in METRICS
        }
    except (OverflowError, ValueError):
        return _empty_result(draws=draws,
                             seed=seed,
                             reasons=["nonfinite_ratio"],
                             support=support,
                             cohort=cohort)
    try:
        arrays = {
            stratum: _stratum_array(ordered[stratum])
            for stratum in STRATA
        }
    except (OverflowError, TypeError, ValueError):
        return _empty_result(draws=draws,
                             seed=seed,
                             reasons=["nonfinite_ratio"],
                             support=support,
                             cohort=cohort)
    if any(np.isinf(values).any() for values in arrays.values()):
        return _empty_result(draws=draws,
                             seed=seed,
                             reasons=["nonfinite_ratio"],
                             support=support,
                             cohort=cohort)
    rngs = {
        stratum:
        np.random.default_rng(
            np.random.SeedSequence([seed & 0xffffffff, seed >> 32, index]))
        for index, stratum in enumerate(STRATA)
    }
    ratios = np.empty((draws, len(METRICS)), dtype=np.float64)
    stratum_hashes = {stratum: hashlib.sha256() for stratum in STRATA}
    first_draw = {}
    offset = 0
    while offset < draws:
        count = min(batch_size, draws - offset)
        sampled = {}
        pooled = []
        for stratum in STRATA:
            case_count = len(ordered[stratum])
            indices = rngs[stratum].integers(0,
                                             case_count,
                                             size=(count, case_count),
                                             dtype=np.int64)
            sampled[stratum] = indices
            stratum_hashes[stratum].update(indices.tobytes())
            selected = arrays[stratum][indices]
            pooled.append(selected.reshape(count, -1, 2, len(METRICS)))
        if offset == 0:
            for stratum in STRATA:
                expanded = []
                for case_index in sampled[stratum][0]:
                    expanded.extend({
                        "case_id": row["case_id"],
                        "repetition": row["repetition"]
                    } for row in ordered[stratum][int(case_index)])
                first_draw[stratum] = expanded
        with np.errstate(divide="ignore",
                         invalid="ignore",
                         over="ignore",
                         under="ignore"):
            medians = np.nanmedian(np.concatenate(pooled, axis=1), axis=1)
            batch_ratios = medians[:, 1, :] / medians[:, 0, :]
        if (not np.isfinite(medians).all() or np.any(medians[:, 0, :] <= 0)
                or np.any(medians[:, 1, :] < 0)
                or not np.isfinite(batch_ratios).all()):
            return _empty_result(draws=draws,
                                 seed=seed,
                                 reasons=["nonfinite_ratio"],
                                 support=support,
                                 cohort=cohort)
        ratios[offset:offset + count] = batch_ratios
        offset += count

    rank = min(draws, math.ceil(CONFIDENCE_LEVEL * (draws + 1)))
    warnings = ["coverage_uncalibrated"]
    for stratum, counts in support.items():
        if counts["cases"] < 2:
            warnings.append(f"sparse_stratum_support:{stratum}")
    draw_hash = hashlib.sha256()
    for stratum in STRATA:
        draw_hash.update(
            stratum.encode("ascii") + b"\0" + stratum_hashes[stratum].digest())
    bounds = {}
    for metric_index, metric in enumerate(METRICS):
        values = ratios[:, metric_index]
        upper = float(np.partition(values, rank - 1)[rank - 1])
        bounds[metric] = {
            "point_ratio": point[metric],
            "upper_confidence_bound": upper,
            "bootstrap_min": float(np.min(values)),
            "bootstrap_max": float(np.max(values)),
            "distinct_bootstrap_ratios": int(np.unique(values).size),
        }
    return {
        "schema_version": 1,
        "method": _METHOD_VERSION,
        "status": "uncalibrated",
        "statistical_pass": None,
        "confidence_level": CONFIDENCE_LEVEL,
        "draws": draws,
        "seed": seed,
        "bounds": bounds,
        "coverage": {
            "calibrated":
            False,
            "percentile_rank":
            rank,
            "interpretation":
            "approximate one-sided percentile bound; coverage is uncalibrated pending a frozen simulation envelope.",
        },
        "cohort": cohort,
        "assumptions": _assumptions(),
        "diagnostics": {
            "invalid_reasons": [],
            "support_counts": support,
            "warnings": warnings,
            "resampling_unit": "case_id_with_all_jointly_eligible_repetitions",
            "same_draws_for_all_metrics": True,
            "first_draw_pair_ids_by_stratum": first_draw,
            "draw_sha256": draw_hash.hexdigest(),
        },
    }
