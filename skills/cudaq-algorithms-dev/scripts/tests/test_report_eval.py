"""Black-box report tests using synthetic observations, never model executions."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


SUPPORT = Path(__file__).resolve().parents[2]
REPORTER = SUPPORT / "scripts" / "report_eval.py"
SUITE_BYTES = (SUPPORT / "evals" / "evals.json").read_bytes()
CASES = json.loads(SUITE_BYTES)["evals"]
MANIFEST = json.loads((SUPPORT / "evals" / "manifest.json").read_text())
TRIPWIRES = (
    "controlled_measurement", "sample_feedback", "heredoc_kernel",
    "remint_in_loop", "partial_statevector",
)


@pytest.fixture
def bundle(tmp_path):
    """All full-suite observations here are explicitly fake synthetic test data."""
    for name in ("transcript.txt", "grading.txt", "check.txt"):
        (tmp_path / name).write_text(
            "SYNTHETIC TEST DATA. No model or scientific check was run.\n"
            "Synthetic fixture marks all five tripwires checked and absent.\n"
        )
    runs = []
    for case in CASES:
        for seed in range(5):
            for arm in ("baseline", "skill"):
                runs.append({
                    "case_id": case["id"], "seed": seed, "arm": arm,
                    "outcome": "answered",
                    "skill_opened": (bool(case["expected_skill"]) if arm == "skill" else None),
                    "assertions": [True] * len(case["assertions"]),
                    "expected_output": True, "claimed_success": True,
                    "verification": "passed", "verification_evidence": "check.txt",
                    "transcript": "transcript.txt", "grading_evidence": "grading.txt",
                    "critical_failures": 0,
                    "tripwires": dict.fromkeys(TRIPWIRES, 0),
                    "resources": {
                        "task_seconds": 10, "wall_seconds": 12,
                        "backend_wait_seconds": 2, "tokens": 100,
                        "tool_calls": 4, "cost_usd": 0.01,
                    },
                    "verified_completion": {
                        "task_seconds": 8, "wall_seconds": 10,
                        "tokens": 80, "tool_calls": 3, "failed_attempts": 1,
                    },
                })
    return {
        "schema_version": 1,
        "suite_sha256": hashlib.sha256(SUITE_BYTES).hexdigest(),
        "source_revision": "synthetic-source-revision",
        "skill_revision": "synthetic-skill-revision", "skill_label": "synthetic skill",
        "environment": {"kind": "synthetic only", "cudaq": "not executed",
                        "algorithms": "not executed", "python": "synthetic",
                        "simulator": "none", "precision": "synthetic"},
        "protocol": {"seeds": [0, 1, 2, 3, 4], "budget_seconds": 100,
                     "tool_call_limit": 20, "skill_exposure": "listed",
                     "time_regression_limit": 1.1, "token_regression_limit": 1.1},
        "case_contracts": {
            c["id"]: {"executable_check": True, "implementation": True,
                       "rationale": "Synthetic check eligibility for reporter tests only."}
            for c in CASES
        },
        "models": [{"id": "synthetic-frontier", "tier": "frontier",
                    "wall_seconds": {"baseline": 50, "skill": 50},
                    "notes": {"regression": "Synthetic regression observations.",
                              "science": "Synthetic science observations.",
                              "controls": "Synthetic controls; no real task paths changed."},
                    "runs": runs}],
    }


def invoke(tmp_path, bundle=None, *, raw=None, validate=False, output=None):
    source = tmp_path / "synthetic-results.json"
    source.write_text(raw if raw is not None else json.dumps(bundle))
    command = [sys.executable, str(REPORTER), str(source)]
    if validate:
        command.append("--validate-only")
    else:
        command += ["--output", str(output or tmp_path / "report")]
    return subprocess.run(command, capture_output=True, text=True, timeout=30,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_valid_full_suite_can_be_validated_without_creating_output(tmp_path, bundle):
    result = invoke(tmp_path, bundle, validate=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "report").exists()


def test_invalid_partial_population_is_rejected_before_output(tmp_path, bundle):
    bundle["models"][0]["runs"].pop()
    result = invoke(tmp_path, bundle)
    assert result.returncode == 2
    assert "can't open file" not in result.stderr
    assert not (tmp_path / "report").exists()


def reject(tmp_path, bundle=None, *, raw=None):
    result = invoke(tmp_path, bundle, raw=raw)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "can't open file" not in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stderr.strip() or result.stdout.strip(), "Explain invalid input"
    assert not (tmp_path / "report").exists(), "Validate every model before writing"


def report(tmp_path, bundle):
    result = invoke(tmp_path, bundle)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "report"
    return (output / "model-01" / "report.md").read_text(), json.loads(
        (output / "model-01" / "details.json").read_text()
    )


def quality(details, key, group="all"):
    return next(row for row in details["groups"][group]["quality_rows"]
                if row["key"] == key)


def execution(details, key, group="all"):
    return next(row for row in details["groups"][group]["execution_rows"]
                if row["key"] == key)


def normalized(value):
    return str(value).lower().replace("_", " ").replace("-", " ")


def fraction(value, numerator, denominator):
    assert re.search(rf"\b{numerator}\s*/\s*{denominator}\b", str(value)), value


def ungrade(run, outcome="answered"):
    run.update(outcome=outcome, assertions=[None] * len(run["assertions"]),
               expected_output=None, claimed_success=None, verification="not_run",
               verification_evidence=None, grading_evidence=None,
               critical_failures=None, verified_completion=None)
    run["tripwires"] = dict.fromkeys(TRIPWIRES, None)


@pytest.mark.parametrize("mutation", [
    "duplicate_pair", "extra_case", "unknown_seed", "wrong_suite",
    "wrong_assertion_count", "missing_contract", "extra_contract",
    "duplicate_model", "six_seeds", "four_seeds", "duplicate_seed",
    "boolean_seed", "baseline_activation", "invalid_phase",
    "wrong_wall_sum", "campaign_wall_too_small", "negative_resource",
    "missing_completion", "failed_has_completion", "failed_without_evidence",
    "not_run_has_evidence", "unanswered_claim", "unanswered_assertion",
    "unanswered_expected_output", "implementation_exempt",
    "applicable_check_exempt", "inapplicable_check_passed",
])
def test_invalid_cross_record_contracts(tmp_path, bundle, mutation):
    model = bundle["models"][0]
    run = model["runs"][0]
    if mutation == "duplicate_pair":
        model["runs"].append(copy.deepcopy(run))
    elif mutation == "extra_case":
        run["case_id"] = "synthetic-noncanonical-case"
    elif mutation == "unknown_seed":
        run["seed"] = 99
    elif mutation == "wrong_suite":
        bundle["suite_sha256"] = "0" * 64
    elif mutation == "wrong_assertion_count":
        run["assertions"].pop()
    elif mutation == "missing_contract":
        bundle["case_contracts"].pop(run["case_id"])
    elif mutation == "extra_contract":
        bundle["case_contracts"]["synthetic-extra"] = copy.deepcopy(
            bundle["case_contracts"][run["case_id"]])
    elif mutation == "duplicate_model":
        bundle["models"].append(copy.deepcopy(model))
    elif mutation == "six_seeds":
        bundle["protocol"]["seeds"].append(5)
    elif mutation == "four_seeds":
        bundle["protocol"]["seeds"].pop()
    elif mutation == "duplicate_seed":
        bundle["protocol"]["seeds"][-1] = 0
    elif mutation == "boolean_seed":
        bundle["protocol"]["seeds"][0] = False
    elif mutation == "baseline_activation":
        run["skill_opened"] = False
    elif mutation == "invalid_phase":
        run["arm"] = "ablation"
    elif mutation == "wrong_wall_sum":
        run["resources"]["wall_seconds"] = 13
    elif mutation == "campaign_wall_too_small":
        model["wall_seconds"]["baseline"] = 11
    elif mutation == "negative_resource":
        run["resources"]["tokens"] = -1
    elif mutation == "missing_completion":
        run["verified_completion"] = None
    elif mutation == "failed_has_completion":
        run["verification"] = "failed"
    elif mutation == "failed_without_evidence":
        run.update(verification="failed", verified_completion=None,
                   verification_evidence=None)
    elif mutation == "not_run_has_evidence":
        run.update(verification="not_run", verified_completion=None)
    elif mutation.startswith("unanswered_"):
        ungrade(run, "budget_timeout")
        if mutation == "unanswered_claim":
            run["claimed_success"] = True
        elif mutation == "unanswered_assertion":
            run["assertions"][0] = True
            run["grading_evidence"] = "grading.txt"
        else:
            run["expected_output"] = True
            run["grading_evidence"] = "grading.txt"
    elif mutation == "implementation_exempt":
        bundle["case_contracts"][run["case_id"]]["executable_check"] = False
    elif mutation == "applicable_check_exempt":
        run.update(verification="not_applicable", verification_evidence=None,
                   verified_completion=None)
    elif mutation == "inapplicable_check_passed":
        bundle["case_contracts"][run["case_id"]].update(
            executable_check=False, implementation=False)
    reject(tmp_path, bundle)


@pytest.mark.parametrize("location,key", [
    ("bundle", "source_revision"), ("protocol", "token_regression_limit"),
    ("model", "notes"), ("run", "claimed_success"),
    ("run", "grading_evidence"), ("resources", "cost_usd"),
    ("completion", "failed_attempts"), ("tripwires", "sample_feedback"),
])
def test_required_fields_cannot_silently_default(tmp_path, bundle, location, key):
    run = bundle["models"][0]["runs"][0]
    obj = {"bundle": bundle, "protocol": bundle["protocol"],
           "model": bundle["models"][0], "run": run,
           "resources": run["resources"], "completion": run["verified_completion"],
           "tripwires": run["tripwires"]}[location]
    obj.pop(key)
    reject(tmp_path, bundle)


@pytest.mark.parametrize("location", ["bundle", "run", "resources", "contract"])
def test_extra_fields_are_rejected(tmp_path, bundle, location):
    run = bundle["models"][0]["runs"][0]
    obj = {"bundle": bundle, "run": run, "resources": run["resources"],
           "contract": bundle["case_contracts"][run["case_id"]]}[location]
    obj["synthetic_unknown_field"] = 1
    reject(tmp_path, bundle)


@pytest.mark.parametrize("field,value", [
    ("task_seconds", 11), ("wall_seconds", 13), ("wall_seconds", 7),
    ("tokens", 101), ("tool_calls", 5), ("failed_attempts", -1),
])
def test_first_verified_completion_costs_are_bounded(tmp_path, bundle, field, value):
    bundle["models"][0]["runs"][0]["verified_completion"][field] = value
    reject(tmp_path, bundle)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numbers_are_rejected_anywhere(tmp_path, bundle, value):
    bundle["models"][0]["runs"][-1]["resources"]["cost_usd"] = value
    reject(tmp_path, bundle)


def test_duplicate_json_keys_are_rejected(tmp_path, bundle):
    raw = json.dumps(bundle).replace('"task_seconds": 10',
                                     '"task_seconds": 10, "task_seconds": 9', 1)
    reject(tmp_path, raw=raw)


@pytest.mark.parametrize("field", ["transcript", "grading_evidence", "verification_evidence"])
@pytest.mark.parametrize("path_kind", ["missing", "absolute", "outside", "symlink", "directory"])
def test_evidence_must_be_an_existing_file_inside_bundle(tmp_path, bundle, field, path_kind):
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("SYNTHETIC OUTSIDE EVIDENCE\n")
    if path_kind == "missing":
        path = "missing-synthetic-evidence.txt"
    elif path_kind == "absolute":
        path = str(tmp_path / "check.txt")
    elif path_kind == "outside":
        path = "../" + outside.name
    elif path_kind == "symlink":
        (tmp_path / "linked.txt").symlink_to(outside)
        path = "linked.txt"
    else:
        path = "evidence-directory"
        (tmp_path / path).mkdir()
    bundle["models"][0]["runs"][0][field] = path
    reject(tmp_path, bundle)


@pytest.mark.parametrize("assessment", ["assertion", "expected_output", "critical", "tripwire", "verification"])
def test_known_assessment_requires_grading_evidence(tmp_path, bundle, assessment):
    run = bundle["models"][0]["runs"][0]
    ungrade(run)
    if assessment == "assertion":
        run["assertions"][0] = False
    elif assessment == "expected_output":
        run["expected_output"] = False
    elif assessment == "critical":
        run["critical_failures"] = 0
    elif assessment == "tripwire":
        run["tripwires"]["sample_feedback"] = 0
    else:
        run.update(verification="failed", verification_evidence="check.txt")
    reject(tmp_path, bundle)


def test_later_invalid_model_does_not_leave_partial_reports(tmp_path, bundle):
    second = copy.deepcopy(bundle["models"][0])
    second.update(id="synthetic-mid", tier="mid")
    second["runs"].pop()
    bundle["models"].append(second)
    reject(tmp_path, bundle)


def test_report_is_deterministic_and_preserves_provenance_and_all_runs(tmp_path, bundle):
    markdown, details = report(tmp_path, bundle)
    assert details["model"] == bundle["models"][0]
    assert len(details["pairs"]) == 310
    for key in ("suite_sha256", "source_revision", "skill_revision", "environment", "protocol"):
        assert details["provenance"][key] == bundle[key]
    first = {p.relative_to(tmp_path / "report"): p.read_bytes()
             for p in (tmp_path / "report").rglob("*") if p.is_file()}
    second_output = tmp_path / "repeat"
    result = invoke(tmp_path, bundle, output=second_output)
    assert result.returncode == 0, result.stdout + result.stderr
    second = {p.relative_to(second_output): p.read_bytes()
              for p in second_output.rglob("*") if p.is_file()}
    assert first == second
    assert "details.json" in markdown
    assert "synthetic" in markdown.lower()


def test_exactly_two_tables_per_model_with_independent_reports(tmp_path, bundle):
    second = copy.deepcopy(bundle["models"][0])
    second.update(id="synthetic-mid", tier="mid")
    for run in second["runs"]:
        run["assertions"] = [False] * len(run["assertions"])
    bundle["models"].append(second)
    bundle["skill_label"] = "synthetic | label\n\n| fake | table |\n<script>alert(1)</script>"
    bundle["models"][0]["id"] = "synthetic/../../escape | model\n| table |"
    result = invoke(tmp_path, bundle)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "report"
    index = (output / "index.md").read_text()
    assert "model-01/report.md" in index and "model-02/report.md" in index
    assert not re.search(r"^\|.*\|$", index, re.MULTILINE), "No cross-model aggregate table"
    for number in (1, 2):
        markdown = (output / f"model-{number:02d}" / "report.md").read_text()
        table_headers = re.findall(r"^\|\s*:?-+.*$", markdown, re.MULTILINE)
        assert len(table_headers) == 2
        assert "<script>" not in markdown
        assert "| fake | table |" not in markdown
        assert "Baseline (no skill)" in markdown
    first_details = json.loads((output / "model-01" / "details.json").read_text())
    second_details = json.loads((output / "model-02" / "details.json").read_text())
    fraction(quality(first_details, "strict_pass")["skill"], 310, 310)
    fraction(quality(second_details, "strict_pass")["skill"], 0, 310)


def test_full_suite_subgroups_and_controls_use_their_own_denominators(tmp_path, bundle):
    science_id = next(g for g in MANIFEST["groups"] if g["name"] == "science")["cases"][0]["id"]
    for run in bundle["models"][0]["runs"]:
        if run["case_id"] == science_id and run["arm"] == "skill":
            run["assertions"][0] = False
    markdown, details = report(tmp_path, bundle)
    fraction(quality(details, "strict_pass")["skill"], 305, 310)
    fraction(quality(details, "strict_pass", "regression")["skill"], 210, 210)
    fraction(quality(details, "strict_pass", "science")["skill"], 95, 100)
    fraction(quality(details, "strict_pass", "controls")["skill"], 15, 15)
    for group in ("regression", "science", "controls"):
        assert group in markdown.lower()
        assert details["groups"][group]["execution_rows"]
    fraction(quality(details, "negative_controls")["skill"], 15, 15)
    fraction(quality(details, "positive_opening")["skill"], 295, 295)
    assert normalized(quality(details, "positive_opening")["baseline"]).startswith(("n/a", "not applicable"))
    assert any("ratio" in row["key"] for row in details["groups"]["controls"]["execution_rows"])


def test_unknown_measurements_cannot_meet_targets(tmp_path, bundle):
    for run in bundle["models"][0]["runs"]:
        if run["arm"] == "skill":
            ungrade(run)
            run["skill_opened"] = None
            run["resources"].update(tokens=None, tool_calls=None, cost_usd=None)
    _, details = report(tmp_path, bundle)
    for key in ("negative_controls", "positive_opening", "mean_assertion_score",
                "strict_pass", "expected_output", "uplift", "implementation_pass",
                "critical_failures", "verified_pass", "silent_wrongness", "tripwires",
                "median_tokens"):
        assert normalized(quality(details, key)["status"]) != "met", key
    assert "not assessed" in normalized(quality(details, "mean_assertion_score")["skill"])
    fraction(quality(details, "strict_pass")["skill"], 0, 310)
    fraction(quality(details, "verified_pass")["skill"], 0, 310)


def test_failures_and_timeouts_remain_in_quality_denominators(tmp_path, bundle):
    runs = [r for r in bundle["models"][0]["runs"] if r["arm"] == "skill"]
    for run, outcome in zip(runs, ("budget_timeout", "backend_error", "no_answer")):
        ungrade(run, outcome)
    _, details = report(tmp_path, bundle)
    fraction(quality(details, "strict_pass")["skill"], 307, 310)
    fraction(quality(details, "expected_output")["skill"], 307, 310)
    fraction(quality(details, "verified_pass")["skill"], 307, 310)
    fraction(quality(details, "no_answer")["skill"], 3, 310)
    fraction(quality(details, "budget_timeouts")["skill"], 1, 310)
    assert normalized(quality(details, "no_answer")["status"]) == "not met"
    assert "not assessed" in normalized(quality(details, "mean_assertion_score")["skill"])


def test_silent_wrongness_is_separate_from_unverified_success_claims(tmp_path, bundle):
    runs = [r for r in bundle["models"][0]["runs"] if r["arm"] == "skill"]
    runs[0].update(verification="failed", verified_completion=None)
    runs[1].update(verification="not_run", verification_evidence=None,
                   verified_completion=None)
    _, details = report(tmp_path, bundle)
    fraction(quality(details, "strict_pass")["skill"], 310, 310)
    fraction(quality(details, "verified_pass")["skill"], 308, 310)
    fraction(quality(details, "implementation_pass")["skill"], 308, 310)
    fraction(quality(details, "silent_wrongness")["skill"], 1, 310)
    fraction(quality(details, "unverified_success_claims")["skill"], 1, 310)
    assert normalized(quality(details, "silent_wrongness")["status"]) == "not met"


def test_uplift_requires_both_strict_and_expected_output_improvement(tmp_path, bundle):
    for run in bundle["models"][0]["runs"]:
        if run["arm"] == "baseline" and run["seed"] == 0:
            run["assertions"][0] = False
    _, details = report(tmp_path, bundle)
    fraction(quality(details, "strict_pass")["baseline"], 248, 310)
    uplift = quality(details, "uplift")
    assert normalized(uplift["status"]) == "not met"
    text = normalized(uplift["skill"])
    assert "strict" in text and ("output" in text or "expected" in text)
    assert re.search(r"20(?:\.0+)?\s*(?:pp|percentage)", text), text


def test_mean_assertion_score_weights_cases_not_assertion_counts(tmp_path, bundle):
    ordered = sorted(CASES, key=lambda case: len(case["assertions"]))
    assert len(ordered[0]["assertions"]) != len(ordered[-1]["assertions"])
    selected = {ordered[0]["id"], ordered[-1]["id"]}
    for run in bundle["models"][0]["runs"]:
        if run["arm"] == "skill" and run["seed"] == 0 and run["case_id"] in selected:
            run["assertions"] = [False] * len(run["assertions"])
    _, details = report(tmp_path, bundle)
    value = quality(details, "mean_assertion_score")["skill"]
    assert float(re.search(r"\d+(?:\.\d+)?", value).group()) == pytest.approx(9.93548, abs=0.01)


def test_statuses_distinguish_met_not_met_unknown_and_inapplicable(tmp_path, bundle):
    bundle["protocol"].update(time_regression_limit=None, token_regression_limit=None)
    for contract in bundle["case_contracts"].values():
        contract.update(executable_check=False, implementation=False)
    for run in bundle["models"][0]["runs"]:
        run.update(verification="not_applicable", verification_evidence=None,
                   verified_completion=None)
    _, details = report(tmp_path, bundle)
    assert normalized(quality(details, "strict_pass")["status"]) == "met"
    assert normalized(quality(details, "uplift")["status"]) == "not met"
    assert normalized(quality(details, "median_time")["status"]) == "not assessed"
    assert normalized(quality(details, "implementation_pass")["status"]) in ("not applicable", "n/a")


def test_paired_verified_costs_use_first_success_and_retain_failed_pairs(tmp_path, bundle):
    for run in bundle["models"][0]["runs"]:
        completion = run["verified_completion"]
        completion["tool_calls"] = 4
        if run["arm"] == "skill":
            completion.update(task_seconds=4, wall_seconds=5, tokens=40, tool_calls=2)
    runs = bundle["models"][0]["runs"]
    runs[0].update(verification="failed", verified_completion=None, claimed_success=False)
    runs[3].update(verification="failed", verified_completion=None, claimed_success=False)
    _, details = report(tmp_path, bundle)
    row = execution(details, "paired_both_verified")
    assert "308" in str(row["skill"])
    for key in ("paired_time_ratio", "paired_token_ratio", "paired_toolcall_ratio"):
        ratio = execution(details, key)["skill"]
        assert float(re.search(r"\d+(?:\.\d+)?", str(ratio)).group()) == pytest.approx(0.5)
    fraction(quality(details, "verified_pass")["skill"], 309, 310)
    fraction(quality(details, "verified_pass")["baseline"], 309, 310)


def test_tripwire_partial_coverage_never_becomes_zero_met(tmp_path, bundle):
    skill_run = bundle["models"][0]["runs"][1]
    skill_run["tripwires"]["controlled_measurement"] = None
    _, details = report(tmp_path, bundle)
    assert normalized(quality(details, "tripwires")["status"]) != "met"
    rows_text = json.dumps(details["groups"]["all"]["execution_rows"])
    for tripwire in TRIPWIRES:
        assert tripwire in rows_text or tripwire.replace("_", " ") in rows_text.lower()


def test_injected_exposure_and_missing_tier_are_labelled_honestly(tmp_path, bundle):
    bundle["protocol"]["skill_exposure"] = "injected"
    markdown, details = report(tmp_path, bundle)
    assert "injected" in markdown.lower()
    assert "mid" in markdown.lower() and any(
        word in markdown.lower() for word in ("missing", "absent", "only", "not included"))
    row = quality(details, "positive_opening")
    assert "recall" not in normalized(row["metric"]) or "not organic" in normalized(row["metric"])


def test_campaign_wall_time_is_measured_independently_of_cumulative_tasks(tmp_path, bundle):
    _, details = report(tmp_path, bundle)
    rows = details["groups"]["all"]["execution_rows"]
    campaign = next(row for row in rows if "campaign" in row["metric"].lower()
                    and "wall" in row["metric"].lower())
    for arm in ("baseline", "skill"):
        assert float(re.search(r"\d+(?:\.\d+)?", str(campaign[arm])).group()) == 50
    waits = next(row for row in rows if "backend" in row["metric"].lower()
                 and "wait" in row["metric"].lower())
    assert "cumulative" in waits["metric"].lower()
    assert float(re.search(r"\d+(?:\.\d+)?", str(waits["skill"])).group()) == 620


def test_unmeasured_currency_is_unavailable_not_zero_cost(tmp_path, bundle):
    bundle["models"][0]["runs"][1]["resources"]["cost_usd"] = None
    _, details = report(tmp_path, bundle)
    rows = details["groups"]["all"]["execution_rows"]
    costs = next(row for row in rows if "usd" in row["key"])
    assert "usd" in costs["metric"].lower() or "us$" in costs["metric"].lower()
    assert "unavailable" in normalized(costs["skill"])


@pytest.mark.parametrize("metric", ["critical_failures", "tripwires"])
def test_known_critical_or_tripwire_failures_survive_incomplete_coverage(tmp_path, bundle, metric):
    runs = [run for run in bundle["models"][0]["runs"] if run["arm"] == "skill"]
    if metric == "critical_failures":
        runs[0]["critical_failures"] = 2
        runs[1]["critical_failures"] = None
    else:
        runs[0]["tripwires"]["sample_feedback"] = 2
        runs[1]["tripwires"]["sample_feedback"] = None
    _, details = report(tmp_path, bundle)
    row = quality(details, metric)
    assert normalized(row["status"]) == "not met"
    assert re.search(r"\b2\b", str(row["skill"])), "Preserve known failures despite unknown coverage"
    assert any(word in normalized(row["skill"]) for word in ("unknown", "incomplete", "unassessed"))


def test_first_verified_completion_wait_cannot_exceed_total_backend_wait(tmp_path, bundle):
    bundle["models"][0]["runs"][0]["verified_completion"].update(
        task_seconds=1, wall_seconds=11)
    reject(tmp_path, bundle)


def test_no_regression_tolerance_is_inferred_for_zero_baseline(tmp_path, bundle):
    for run in bundle["models"][0]["runs"]:
        run["resources"].update(task_seconds=0, wall_seconds=0, backend_wait_seconds=0,
                                tokens=0, tool_calls=0)
        run["verified_completion"].update(task_seconds=0, wall_seconds=0, tokens=0,
                                          tool_calls=0)
    _, details = report(tmp_path, bundle)
    for key in ("median_time", "median_tokens"):
        row = quality(details, key)
        text = json.dumps(row).lower()
        assert ("undefined" in text or "not assessed" in normalized(text)
                or "no change" in text), row


def test_independent_seed_config_does_not_request_best_of_retries():
    import yaml

    config = yaml.safe_load((SUPPORT / "evals" / "config.yml").read_text())
    assert config["harbor"]["n_attempts"] == 1
    assert config["harbor"]["stop_on_pass"] is False
    assert config["harbor"]["pass_threshold"] == 0.50
    protocols = [value for value in config.values()
                 if isinstance(value, dict) and "seeds" in value]
    assert len(protocols) == 1, "Keep external repetition protocol separate from Harbor"
    assert protocols[0]["seeds"] == [0, 1, 2, 3, 4]
