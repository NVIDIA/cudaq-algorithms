#!/usr/bin/env python3
"""Validate paired evaluation evidence and render two tables per model.

This is a reporter, not a model runner or a scientific grader. See evals/EVAL.md.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

SUPPORT = Path(__file__).resolve().parents[1]
EVALS = SUPPORT / "evals"
TRIPWIRES = ("controlled_measurement", "sample_feedback", "heredoc_kernel",
             "remint_in_loop", "partial_statevector")
ARMS = ("baseline", "skill")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def check_finite(value):
    if isinstance(value, float):
        require(math.isfinite(value),
                "all numeric measurements must be finite")
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_file(root, reference):
    path = Path(reference)
    require(not path.is_absolute(),
            f"evidence must use relative paths: {reference}")
    path = (root / path).resolve()
    require(path.is_relative_to(root),
            f"evidence escapes results directory: {reference}")
    require(path.is_file() and path.stat().st_size > 0,
            f"missing or empty evidence file: {reference}")
    return path


def validate(data, evidence_root):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ValueError(
            "reporting requires jsonschema; install with python3 -m pip install jsonschema"
        ) from exc
    check_finite(data)
    schema = load_json(EVALS / "results.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(data),
                    key=lambda e: str(list(e.absolute_path)))
    require(
        not errors,
        "schema: " + "; ".join(f"{list(e.absolute_path)}: {e.message}"
                               for e in errors[:5]))
    suite = load_json(EVALS / "evals.json")
    cases = {c["id"]: c for c in suite["evals"]}
    manifest = load_json(EVALS / "manifest.json")
    groups = {
        g["name"]: [c["id"] for c in g["cases"]]
        for g in manifest["groups"]
    }
    require(data["suite_sha256"] == digest(EVALS / "evals.json"),
            "suite SHA256 mismatch")
    for group in manifest["groups"]:
        for item in group["cases"]:
            raw = json.dumps(cases[item["id"]],
                             sort_keys=True,
                             separators=(",", ":"),
                             ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
            require(
                hashlib.sha256(raw).hexdigest() == item["sha256"],
                f"case differs from delivery manifest: {item['id']}")
    require(
        set(data["case_contracts"]) == set(cases),
        "case_contracts must cover every canonical case exactly")
    for case_id, contract in data["case_contracts"].items():
        require(not contract["implementation"] or contract["executable_check"],
                f"implementation case requires executable check: {case_id}")
    model_ids = [m["id"] for m in data["models"]]
    require(len(set(model_ids)) == len(model_ids), "duplicate model ID")
    expected = {(case_id, seed, arm)
                for case_id in cases
                for seed in data["protocol"]["seeds"]
                for arm in ARMS}
    evidence = {}
    evidence_root = evidence_root.resolve()
    for model in data["models"]:
        seen = set()
        for run in model["runs"]:
            key = (run["case_id"], run["seed"], run["arm"])
            label = f"{model['id']} / {key}"
            require(key in expected, f"unknown case/seed/arm: {label}")
            require(key not in seen, f"duplicate case/seed/arm: {label}")
            seen.add(key)
            require(
                len(run["assertions"]) == len(
                    cases[run["case_id"]]["assertions"]),
                f"assertion count mismatch: {label}")
            if run["arm"] == "baseline":
                require(run["skill_opened"] is None,
                        f"baseline skill_opened must be null: {label}")
            if run["outcome"] != "answered":
                require(
                    run["claimed_success"] is not True
                    and run["expected_output"] is not True
                    and True not in run["assertions"],
                    f"unanswered run cannot claim successful final output/assertions: {label}"
                )
            contract = data["case_contracts"][run["case_id"]]
            verification = run["verification"]
            require((verification == "not_applicable") == (
                not contract["executable_check"]
            ), f"verification applicability differs from preregistered contract: {label}"
                    )
            assessed = verification in ("passed", "failed")
            require(
                (run["verification_evidence"] is not None) == assessed,
                f"verification evidence required exactly for passed/failed checks: {label}"
            )
            graded = (assessed or any(v is not None for v in run["assertions"])
                      or run["expected_output"] is not None
                      or run["critical_failures"] is not None
                      or any(v is not None for v in run["tripwires"].values()))
            require(
                not graded or run["grading_evidence"] is not None,
                f"grading evidence required for assessed results: {label}")
            for field in ("transcript", "grading_evidence",
                          "verification_evidence"):
                reference = run[field]
                if reference is not None and reference not in evidence:
                    evidence[reference] = digest(
                        evidence_file(evidence_root, reference))
            resource = run["resources"]
            require(
                math.isclose(resource["wall_seconds"],
                             resource["task_seconds"] +
                             resource["backend_wait_seconds"],
                             rel_tol=1e-8,
                             abs_tol=1e-6),
                f"wall time must equal task time plus backend wait: {label}")
            completion = run["verified_completion"]
            require((completion is not None) == (
                verification == "passed"
            ), f"verified_completion required exactly when checks pass: {label}"
                    )
            if completion is not None:
                require(
                    completion["wall_seconds"] >= completion["task_seconds"],
                    f"completion wall time smaller than task time: {label}")
                require(
                    completion["wall_seconds"] - completion["task_seconds"]
                    <= resource["backend_wait_seconds"] + 1e-6,
                    f"completion backend wait exceeds run total: {label}")
                for field in ("task_seconds", "wall_seconds", "tokens",
                              "tool_calls"):
                    if completion[field] is not None and resource[
                            field] is not None:
                        require(
                            completion[field] <= resource[field],
                            f"verified completion {field} exceeds run total: {label}"
                        )
            require(
                model["wall_seconds"][run["arm"]] >= resource["wall_seconds"],
                f"campaign wall time smaller than individual run: {label}")
        require(
            seen == expected,
            f"model {model['id']}: missing {len(expected - seen)} paired observations; retain unanswered runs"
        )
    return cases, groups, evidence


def rate(values):
    return {
        "passed": sum(v is True for v in values),
        "total": len(values),
        "unknown": sum(v is None for v in values)
    }


def rate_number(metric):
    if not metric["total"] or metric["unknown"]:
        return None
    return metric["passed"] / metric["total"]


def rate_text(metric, incidents=False):
    if not metric["total"]:
        return "N/A (0 eligible)"
    count = f"{metric['passed']}/{metric['total']}"
    if metric["unknown"]:
        text = f"{count} confirmed {'incidents' if incidents else 'passes'}; {metric['unknown']} not assessed"
    else:
        text = f"{count} ({100 * rate_number(metric):.1f}%)"
    failures = metric["total"] - metric["passed"] - metric["unknown"]
    if failures and not incidents:
        text += f"; {failures} failed"
    return text


def rate_status(metric, threshold, higher=True):
    if not metric["total"]:
        return "n/a"
    lower = metric["passed"] / metric["total"]
    upper = (metric["passed"] + metric["unknown"]) / metric["total"]
    # A known failure can disprove a target even with other grades missing.
    if (higher and upper < threshold) or (not higher and lower > threshold):
        return "not met"
    return target_status(rate_number(metric), threshold, higher)


def strict(run):
    if run["outcome"] != "answered" or False in run["assertions"]:
        return False
    return None if None in run["assertions"] else True


def verified(run):
    return {
        "passed": True,
        "failed": False,
        "not_run": None,
        "not_applicable": None
    }[run["verification"]]


def silent(run):
    if run["claimed_success"] is False or run["verification"] == "passed":
        return False
    if run["claimed_success"] is True and run["verification"] == "failed":
        return True
    return None


def unverified_claim(run):
    if run["verification"] != "not_run" or run["claimed_success"] is False:
        return False
    return run["claimed_success"]


def total(values):
    return None if any(v is None for v in values) else sum(values)


def incident_count(values):
    return {
        "observed": sum(v for v in values if v is not None),
        "unknown": sum(v is None for v in values),
        "total": len(values)
    }


def incident_text(count):
    if count["unknown"]:
        return (f"{count['observed']} known incidents; "
                f"{count['unknown']}/{count['total']} assessments unknown")
    return str(count["observed"])


def incident_status(count):
    if count["observed"]:
        return "not met"
    return "not assessed" if count["unknown"] else "met"


def statistic(values, fn=statistics.median):
    return None if not values or any(v is None for v in values) else fn(values)


def number(value, unit=""):
    if value is None:
        return "not assessed"
    formatted = str(value) if isinstance(
        value, int) else f"{value:.3f}".rstrip("0").rstrip(".")
    return formatted + unit


def ratio(a, b):
    return None if a is None or b is None or a == 0 else b / a


def target_status(value, target, higher=True):
    if value is None or target is None:
        return "not assessed"
    return "met" if (
        value >= target if higher else value <= target) else "not met"


def summarize(runs, contracts):
    eligible = [r for r in runs if contracts[r["case_id"]]["executable_check"]]
    implementations = [
        r for r in runs if contracts[r["case_id"]]["implementation"]
    ]
    scores = [
        None if None in r["assertions"] else 10 * sum(r["assertions"]) /
        len(r["assertions"]) for r in runs
    ]
    tripwire_counts = {
        k: total([r["tripwires"][k] for r in runs])
        for k in TRIPWIRES
    }
    return {
        "observations":
        len(runs),
        "judged":
        sum(None not in r["assertions"] and r["expected_output"] is not None
            for r in runs),
        "strict_pass":
        rate([strict(r) for r in runs]),
        "expected_output":
        rate([
            False if r["outcome"] != "answered" else r["expected_output"]
            for r in runs
        ]),
        "verified_pass":
        rate([verified(r) for r in eligible]),
        "implementation_pass":
        rate([verified(r) for r in implementations]),
        "silent_wrongness":
        rate([silent(r) for r in eligible]),
        "unverified_success_claims":
        rate([unverified_claim(r) for r in eligible]),
        "mean_assertion_score":
        statistic(scores, statistics.mean),
        "critical_failures":
        total([r["critical_failures"] for r in runs]),
        "critical_failure_assessments":
        incident_count([r["critical_failures"] for r in runs]),
        "tripwire_counts":
        tripwire_counts,
        "tripwires":
        total(list(tripwire_counts.values())),
        "tripwire_assessments": {
            k: incident_count([r["tripwires"][k] for r in runs])
            for k in TRIPWIRES
        },
        "all_tripwire_assessments":
        incident_count([r["tripwires"][k] for r in runs for k in TRIPWIRES]),
        "no_answer":
        sum(r["outcome"] != "answered" for r in runs),
        "budget_timeouts":
        sum(r["outcome"] == "budget_timeout" for r in runs),
        "no_answer_causes":
        dict(
            sorted(
                Counter(r["outcome"] for r in runs
                        if r["outcome"] != "answered").items())),
        "median_time":
        statistic([r["resources"]["task_seconds"] for r in runs]),
        "median_tokens":
        statistic([r["resources"]["tokens"] for r in runs]),
    }


def make_pairs(model, groups):
    indexed = {}
    for run in model["runs"]:
        key = (run["case_id"], run["seed"])
        indexed.setdefault(key, {})[run["arm"]] = run
    pairs = []
    for (case_id, seed), arms in indexed.items():
        completions = [arms[a]["verified_completion"] for a in ARMS]
        both = all(v is not None for v in completions)
        ratios = {
            field:
            ratio(completions[0][field], completions[1][field])
            if both else None
            for field in ("task_seconds", "tokens", "tool_calls")
        }
        pairs.append({
            "case_id":
            case_id,
            "seed":
            seed,
            "group":
            next(name for name, ids in groups.items() if case_id in ids),
            "both_verified":
            both,
            "completion_ratios":
            ratios,
            "arms": {
                a: {
                    "strict_pass": strict(r),
                    "expected_output": r["expected_output"],
                    "verification": r["verification"],
                    "outcome": r["outcome"],
                    "resources": r["resources"],
                    "verified_completion": r["verified_completion"],
                    "tripwires": r["tripwires"]
                }
                for a, r in arms.items()
            }
        })
    return pairs


def group_report(ids, model, data, cases, pairs):
    runs = {
        a: [r for r in model["runs"] if r["case_id"] in ids and r["arm"] == a]
        for a in ARMS
    }
    stats = {a: summarize(runs[a], data["case_contracts"]) for a in ARMS}
    b, s = stats["baseline"], stats["skill"]
    quality = []

    def add(key, label, baseline, skill, target, status):
        quality.append(
            dict(key=key,
                 metric=label,
                 baseline=baseline,
                 skill=skill,
                 target=target,
                 status=status))

    for key, expected, label in (("negative_controls", False,
                                  "Negative controls: skill stayed closed"),
                                 ("positive_opening", True,
                                  "Positive cases: skill opened")):
        observations = [
            r for r in runs["skill"]
            if bool(cases[r["case_id"]]["expected_skill"]) == expected
        ]
        activation = rate([
            None
            if r["skill_opened"] is None else r["skill_opened"] == expected
            for r in observations
        ])
        add(key, label, "N/A (no skill installed)", rate_text(activation),
            ">=90%", rate_status(activation, .90))
    add("mean_assertion_score", "Mean execution score (assertions)",
        number(b["mean_assertion_score"], " / 10"),
        number(s["mean_assertion_score"], " / 10"), ">=8.5 / 10",
        target_status(s["mean_assertion_score"], 8.5))
    for key, label in (("strict_pass", "Strict pass rate"),
                       ("expected_output", "Expected output reached")):
        add(key, label, rate_text(b[key]), rate_text(s[key]), ">=85%",
            rate_status(s[key], .85))
    lifts = []
    for key in ("strict_pass", "expected_output"):
        before, after = rate_number(b[key]), rate_number(s[key])
        lifts.append(None if before is None or after is None else 100 *
                     (after - before))
    lift_text = "; ".join(f"{number(v)} pp {name}"
                          for name, v in zip(("strict", "output"), lifts))
    add("uplift", "Improvement over no-skill baseline", "—", lift_text,
        ">=15 pp, both",
        target_status(None if None in lifts else min(lifts), 15))
    for key, label in (("implementation_pass",
                        "Implementation cases passing executable tests"),
                       ("verified_pass", "Executable verification passed"),
                       ("silent_wrongness",
                        "Silent wrongness (claimed success, failed check)"),
                       ("unverified_success_claims",
                        "Success claims without executable verification")):
        target = 1 if key == "implementation_pass" else 0
        # Verified-pass is an additional diagnostic, without inventing a guide target.
        status = ("n/a" if not s[key]["total"] or key == "verified_pass" else
                  rate_status(s[key], target, higher=bool(target)))
        incidents = key in ("silent_wrongness", "unverified_success_claims")
        add(
            key, label, rate_text(b[key], incidents),
            rate_text(s[key],
                      incidents), "reported" if key == "verified_pass" else
            ("100%" if target else "0%"), status)
    for key, label in (("critical_failures", "Critical failures"),
                       ("tripwires", "Tripwire violations"),
                       ("no_answer", "No final answer (all causes)"),
                       ("budget_timeouts", "Budget timeouts")):
        values = [number(stats[a][key]) for a in ARMS]
        status = target_status(s[key], 0, higher=False)
        if key in ("critical_failures", "tripwires"):
            count_key = "critical_failure_assessments" if key == "critical_failures" else "all_tripwire_assessments"
            values = [incident_text(stats[a][count_key]) for a in ARMS]
            status = incident_status(s[count_key])
        if key in ("no_answer", "budget_timeouts"):
            values = [f"{stats[a][key]}/{len(runs[a])}" for a in ARMS]
        add(key, label, *values, "0", status)
    for key, label, limit, unit in (
        ("median_time", "Median task time (backend waits excluded)",
         "time_regression_limit", " s"), ("median_tokens",
                                          "Median context used, tokens",
                                          "token_regression_limit", "")):
        r = ratio(b[key], s[key])
        tolerance = data["protocol"][limit]
        suffix = f" ({r:.3f}x)" if r is not None else " (ratio not assessed)"
        add(
            key, label, number(b[key], unit),
            number(s[key], unit) + suffix,
            f"<= {tolerance:g}x baseline" if tolerance is not None else
            "No material regression (tolerance unset)",
            target_status(r, tolerance, higher=False))

    execution = []

    def row(key, label, values):
        execution.append(
            dict(key=key, metric=label, baseline=values[0], skill=values[1]))

    row("judged", "Fully judged observations",
        [f"{stats[a]['judged']}/{len(runs[a])}" for a in ARMS])
    for field, label, unit, functions in (
        ("task_seconds", "task time", " s",
         (("median", statistics.median), ("mean", statistics.mean),
          ("total", sum))), ("tokens", "tokens", "",
                             (("median", statistics.median), ("total", sum))),
        ("tool_calls", "tool calls", "", (("median", statistics.median),
                                          ("total", sum))),
        ("backend_wait_seconds",
         "backend wait (cumulative across runs; may overlap)", " s",
         (("total", sum), )), ("cost_usd", "measured cost (USD)", " USD",
                               (("total", sum), ))):
        for name, fn in functions:
            values = [
                statistic([r["resources"][field] for r in runs[a]], fn)
                for a in ARMS
            ]
            row(f"{name}_{field}", f"{name.title()} {label}", [
                "unavailable (not measured)"
                if field == "cost_usd" and v is None else number(v, unit)
                for v in values
            ])
    row("campaign_wall_seconds",
        "Measured campaign wall time (all tasks, not their sum)",
        [number(model["wall_seconds"][a], " s") for a in ARMS])
    row("no_answer_causes", "No-answer causes", [
        ", ".join(f"{k}: {v}"
                  for k, v in stats[a]["no_answer_causes"].items()) or "none"
        for a in ARMS
    ])
    for key in TRIPWIRES:
        row(key, f"Tripwire: {key}", [
            incident_text(stats[a]["tripwire_assessments"][key]) for a in ARMS
        ])
    completed = {
        a: [
            r["verified_completion"] for r in runs[a]
            if r["verified_completion"] is not None
        ]
        for a in ARMS
    }
    row("verified_completions", "Verified completions (eligible denominator)",
        [rate_text(stats[a]["verified_pass"]) for a in ARMS])
    row("failed_verification", "Failed executable checks (run observations)", [
        str(sum(r["verification"] == "failed" for r in runs[a])) for a in ARMS
    ])
    row("verification_not_run",
        "Executable checks not run (eligible observations)", [
            str(sum(r["verification"] == "not_run" for r in runs[a]))
            for a in ARMS
        ])
    for field, label, unit in (("task_seconds", "task time",
                                " s"), ("wall_seconds", "wall time", " s"),
                               ("tokens", "tokens", ""), ("tool_calls",
                                                          "tool calls", ""),
                               ("failed_attempts", "failed attempts", "")):
        row(
            f"verified_median_{field}",
            f"Median {label} until first verified completion (successful runs)",
            [
                number(statistic([c[field] for c in completed[a]]), unit)
                if completed[a] else "N/A (0 verified completions)"
                for a in ARMS
            ])
    selected_pairs = [p for p in pairs if p["case_id"] in ids]
    paired = [p for p in selected_pairs if p["both_verified"]]
    row("paired_both_verified", "Pairs verified in both arms / all pairs",
        [f"{len(paired)}/{len(selected_pairs)}"] * 2)
    for field, key, label in (("task_seconds", "paired_time_ratio",
                               "task time"), ("tokens", "paired_token_ratio",
                                              "tokens"),
                              ("tool_calls", "paired_toolcall_ratio",
                               "tool calls")):
        values = [
            p["completion_ratios"][field] for p in paired
            if p["completion_ratios"][field] is not None
        ]
        text = (f"{statistics.median(values):.3f}x"
                if values else "not assessed")
        text += f"; usable {len(values)}/{len(paired)} both-verified pairs"
        row(key, f"Median paired skill/baseline {label} to verification",
            ["reference", text])
    return {
        "arms": stats,
        "quality_rows": quality,
        "execution_rows": execution
    }


def cell(value):
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace(
        "\n", " ").replace("\r", " ")


def render_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |", "| " +
        " | ".join("---" for _ in headers) + " |"
    ] + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def render(model, data, report, tiers):
    protocol = data["protocol"]
    lines = [
        f"# {cell(model['id'])}", "",
        f"Tier: {model['tier']}. Skill: {cell(data['skill_label'])}.", "",
        f"62 tasks × {len(protocol['seeds'])} paired repetitions = 310 observations per arm. "
        f"Budget: {protocol['budget_seconds']:g} task seconds per observation. Exposure: {protocol['skill_exposure']}.",
        "",
        "Original strict/output rubrics include skill-use requirements: their uplift alone is not an independent correctness gain. "
        "Executable verification and paired completion costs provide separate evidence. "
        "Verification applies only to preregistered eligible cases; unknowns are not passes.",
        "",
        "Negative-control avoidance is not conventional trigger precision. "
        "Injected-skill opening is not organic trigger recall.", ""
    ]
    if tiers != {"frontier", "mid"}:
        lines += [
            "Campaign coverage: missing a frontier or mid-tier model; the guide's two-tier comparison is incomplete.",
            ""
        ]
    lines += [
        "## Quality and targets", "",
        render_table([
            "Metric", "Baseline (no skill)",
            f"With skill ({data['skill_label']})", "Target", "Status"
        ], [[
            r[k] for k in ("metric", "baseline", "skill", "target", "status")
        ] for r in report["quality_rows"]]), "", "## Cost and execution", "",
        render_table(["Metric", "Baseline (no skill)", "With skill"],
                     [[r[k] for k in ("metric", "baseline", "skill")]
                      for r in report["execution_rows"]]), "",
        "Completion medians describe successful runs. Paired ratios use only pairs verified in both arms; "
        "the usable count excludes unknown costs and zero baselines. Failures and timeouts remain in full-population metrics. "
        "Cumulative backend waits can overlap; measured campaign wall time is recorded separately.",
        "",
        "[Task-level paired evidence and regression/science/control summaries](details.json). "
        "Control-task slowdowns and individual failure cases must be inspected before interpreting overall medians.",
        ""
    ]
    for group, note in model["notes"].items():
        lines += [f"**{group.title()} observations:** {cell(note)}", ""]
    return "\n".join(lines)


def write_reports(data, cases, groups, evidence, source, destination):
    destination = destination.resolve()
    for protected in (SUPPORT.resolve(),
                      (SUPPORT.parent / "cudaq-algorithms").resolve()):
        require(not destination.is_relative_to(protected),
                "write reports outside shipped skill directories")
    require(
        not destination.exists()
        or (destination.is_dir() and not any(destination.iterdir())),
        "output directory must be absent or empty; preserve earlier run reports"
    )
    tiers = {m["tier"] for m in data["models"]}
    provenance = {k: v for k, v in data.items() if k != "models"}
    provenance.update(results_sha256=digest(source),
                      schema_sha256=digest(EVALS / "results.schema.json"),
                      evidence_sha256=evidence,
                      evidence_base=str(source.parent.resolve()))
    outputs = {}
    index = [
        "# Evaluation reports", "",
        "Two tables per model; no cross-model aggregate.", ""
    ]
    for i, model in enumerate(data["models"], 1):
        folder = f"model-{i:02d}"
        pairs = make_pairs(model, groups)
        selections = {
            "all": set(cases),
            **{
                g: set(ids)
                for g, ids in groups.items()
            }, "controls":
            {k
             for k, c in cases.items() if c["expected_skill"] is None}
        }
        summaries = {
            g: group_report(ids, model, data, cases, pairs)
            for g, ids in selections.items()
        }
        detail = {
            "provenance": provenance,
            "model": model,
            "groups": summaries,
            "pairs": pairs
        }
        outputs[f"{folder}/details.json"] = json.dumps(
            detail, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        outputs[f"{folder}/report.md"] = render(model, data, summaries["all"],
                                                tiers)
        index += [
            f"- Model {i}: {cell(model['id'])} — [report]({folder}/report.md)"
        ]
    outputs["index.md"] = "\n".join(index) + "\n"
    # Build every report in memory after validation, before creating any files.
    for relative, content in outputs.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="new/empty output directory outside shipped skills")
    parser.add_argument("--validate-only",
                        action="store_true",
                        help="validate without writing reports")
    args = parser.parse_args(argv)
    try:
        require(args.validate_only or args.output is not None,
                "supply --output or --validate-only")
        source = args.results.resolve()
        data = load_json(source)
        cases, groups, evidence = validate(data, source.parent)
        if not args.validate_only:
            write_reports(data, cases, groups, evidence, source, args.output)
        print(
            f"Validated {len(data['models'])} model(s), {len(cases)} tasks, 5 paired repetitions; "
            + ("no files written." if args.
               validate_only else f"reports: {args.output}"))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"report_eval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
