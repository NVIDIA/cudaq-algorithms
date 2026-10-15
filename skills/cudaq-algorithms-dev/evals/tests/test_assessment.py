"""Private judge evidence and canonical exports, using synthetic observations."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

EVALS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assessment", EVALS / "runner" / "assessment.py")
assessment = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(assessment)

CASE = {
    "id": "sample",
    "prompt": "Prepare and check the state.",
    "assertions": ["Prepares the state.", "Checks its phase."],
    "expected_output": "A validated state, including its phase."
}
TRANSCRIPT = '{"role":"assistant","content":"I prepared the state; phase not checked."}\n'
QUOTE = "I prepared the state; phase not checked."


def observation(value, evidence=None):
    return {
        "value": value,
        "evidence": evidence or ([QUOTE] if value is not None else []),
        "reason": "Based on the recorded response."
    }


def raw_grade(case=CASE):
    return {
        "case_id":
        case["id"],
        "assertions": [{
            "index": i,
            **observation(None)
        } for i in range(len(case["assertions"]))],
        "expected_output":
        observation(None),
        "claimed_success":
        observation(None),
        "critical_failures":
        observation(None),
        "tripwires": {
            k: observation(None)
            for k in ("controlled_measurement", "sample_feedback",
                      "heredoc_kernel", "remint_in_loop",
                      "partial_statevector")
        }
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class AssessmentTests(unittest.TestCase):

    def setUp(self):
        self.assertTrue(
            hasattr(assessment, "validate_assessment"),
            "private grading and export implementation is missing")

    def test_private_prompt_preserves_original_indices_and_rubric(self):
        messages = assessment.grade_prompt(CASE, TRANSCRIPT)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["case"]["assertions"],
                         [{
                             "index": 0,
                             "text": "Prepares the state."
                         }, {
                             "index": 1,
                             "text": "Checks its phase."
                         }])
        self.assertEqual(payload["case"]["expected_output"],
                         CASE["expected_output"])
        self.assertEqual(payload["transcript"], TRANSCRIPT)

    def test_partial_evidence_does_not_become_a_pass_or_zero_incidents(self):
        raw = raw_grade()
        raw["assertions"][0] = {"index": 0, **observation(True)}
        raw["assertions"][1] = {"index": 1, **observation(False)}
        validated = assessment.validate_assessment(CASE, raw, TRANSCRIPT)
        self.assertEqual([g["value"] for g in validated["assertions"]],
                         [True, False])
        self.assertIsNone(validated["expected_output"]["value"])
        self.assertIsNone(validated["critical_failures"]["value"])
        self.assertTrue(
            all(g["value"] is None for g in validated["tripwires"].values()))

    def test_rejects_fabricated_quotes_indices_and_coerced_boolean_values(
            self):
        malformed = []
        grade = raw_grade()
        grade["assertions"][0] = {
            "index": 0,
            **observation(True, ["All 32 tests passed."])
        }
        malformed.append(grade)
        grade = raw_grade()
        grade["assertions"].reverse()
        malformed.append(grade)
        grade = raw_grade()
        grade["assertions"][0]["value"] = 1
        malformed.append(grade)
        grade = raw_grade()
        grade["critical_failures"] = observation(True)
        malformed.append(grade)
        grade = raw_grade()
        grade["case_id"] = "other"
        malformed.append(grade)
        for grade in malformed:
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                assessment.validate_assessment(CASE, grade, TRANSCRIPT)

    def test_rejects_duplicate_json_keys_and_nonfinite_judge_values(self):
        for text in ('{"case_id":"sample","case_id":"other"}',
                     '{"value": NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                assessment.validate_assessment(CASE, text, TRANSCRIPT)


class BundleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads((EVALS / "evals.json").read_text())

    def setUp(self):
        self.assertTrue(hasattr(assessment, "build_bundle"),
                        "canonical export implementation is missing")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "suite.json").write_bytes(
            (EVALS / "evals.json").read_bytes())
        contracts = json.loads((EVALS / "case_contracts.json").read_text())
        write_json(self.root / "case_contracts.json", contracts)
        self.meta = {
            "schema_version":
            1,
            "mode":
            "full",
            "suite_sha256":
            hashlib.sha256((EVALS / "evals.json").read_bytes()).hexdigest(),
            "source_revision":
            "synthetic-source",
            "skill_revision":
            "synthetic-skill",
            "skill_label":
            "synthetic test",
            "environment": {
                "python": "synthetic"
            },
            "protocol": {
                "seeds": [0, 1, 2, 3, 4],
                "budget_seconds": 60,
                "tool_call_limit": None,
                "skill_exposure": "listed",
                "time_regression_limit": None,
                "token_regression_limit": None
            },
            "models": [{
                "alias": "test",
                "id": "synthetic-model",
                "tier": "mid",
                "wall_seconds": {
                    "baseline": 310,
                    "skill": 310
                },
                "notes": {
                    g: "Synthetic test observations."
                    for g in ("regression", "science", "controls")
                }
            }]
        }
        write_json(self.root / "campaign.json", self.meta)
        for case in self.suite["evals"]:
            for seed in range(5):
                for arm in ("baseline", "skill"):
                    folder = self.root / "attempts" / "test" / case[
                        "id"] / str(seed) / arm
                    write_json(
                        folder / "result.json", {
                            "case_id": case["id"],
                            "seed": seed,
                            "arm": arm,
                            "outcome": "answered",
                            "skill_opened": None,
                            "resources": {
                                "task_seconds": 1,
                                "wall_seconds": 1,
                                "backend_wait_seconds": 0,
                                "tokens": None,
                                "tool_calls": None,
                                "cost_usd": None
                            }
                        })
                    (folder / "transcript.jsonl").write_text(TRANSCRIPT)
        self.case = self.suite["evals"][0]
        self.folder = self.root / "attempts" / "test" / self.case[
            "id"] / "0" / "baseline"

    def test_full_ungraded_bundle_matches_reporter_and_retains_unknowns(self):
        bundle = assessment.build_bundle(self.root)
        self.assertEqual(len(bundle["models"][0]["runs"]), 620)
        run = bundle["models"][0]["runs"][0]
        self.assertEqual(run["assertions"],
                         [None] * len(self.case["assertions"]))
        self.assertIsNone(run["expected_output"])
        self.assertIsNone(run["critical_failures"])
        self.assertIsNone(run["resources"]["tokens"])
        self.assertIsNone(run["grading_evidence"])
        self.assertTrue(all(n is None for n in run["tripwires"].values()))
        eligible = next(
            r for r in bundle["models"][0]["runs"]
            if r["case_id"] == "repository-implementation-third-moment")
        self.assertEqual(eligible["verification"], "not_run")
        self.assertEqual(run["verification"], "not_applicable")
        spec = importlib.util.spec_from_file_location(
            "report_eval", EVALS.parent / "scripts" / "report_eval.py")
        reporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reporter)
        reporter.validate(bundle, self.root)

    def test_pilot_missing_run_and_missing_timing_cannot_export_as_full(self):
        self.meta["mode"] = "pilot"
        write_json(self.root / "campaign.json", self.meta)
        with self.assertRaisesRegex(ValueError, "pilot|full"):
            assessment.build_bundle(self.root)
        self.meta["mode"] = "full"
        write_json(self.root / "campaign.json", self.meta)
        result_path = self.folder / "result.json"
        result = json.loads(result_path.read_text())
        result_path.unlink()
        with self.assertRaises(ValueError):
            assessment.build_bundle(self.root)
        del result["resources"]["task_seconds"]
        write_json(result_path, result)
        with self.assertRaises(ValueError):
            assessment.build_bundle(self.root)

    def test_grades_bind_to_the_transcript_and_unanswered_never_gains_true(
            self):
        raw = raw_grade(self.case)
        raw["assertions"][0] = {"index": 0, **observation(True)}
        write_json(self.folder / "assessment.json", raw)
        write_json(
            self.folder / "assessment.provenance.json", {
                "result_sha256":
                hashlib.sha256(
                    (self.folder / "result.json").read_bytes()).hexdigest(),
                "transcript_sha256":
                hashlib.sha256((self.folder /
                                "transcript.jsonl").read_bytes()).hexdigest()
            })
        bundle = assessment.build_bundle(self.root)
        run = bundle["models"][0]["runs"][0]
        self.assertTrue(run["assertions"][0])
        self.assertTrue(run["grading_evidence"].endswith("assessment.json"))
        result_path = self.folder / "result.json"
        result = json.loads(result_path.read_text())
        result["outcome"] = "budget_timeout"
        write_json(result_path, result)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            assessment.build_bundle(self.root)
        write_json(
            self.folder / "assessment.provenance.json", {
                "result_sha256":
                hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "transcript_sha256":
                hashlib.sha256((self.folder /
                                "transcript.jsonl").read_bytes()).hexdigest()
            })
        with self.assertRaisesRegex(ValueError, "unanswered"):
            assessment.build_bundle(self.root)

    def test_verification_is_not_accepted_without_binding_and_evidence(self):
        case_id = "repository-implementation-third-moment"
        folder = self.root / "attempts" / "test" / case_id / "0" / "baseline"
        write_json(folder / "verification.json", {"status": "passed"})
        with self.assertRaises(ValueError):
            assessment.build_bundle(self.root)

    def test_passed_check_requires_unchanged_log_and_retains_unknown_rubric(
            self):
        case_id = "repository-implementation-third-moment"
        folder = self.root / "attempts" / "test" / case_id / "0" / "baseline"
        evidence = folder / "check.log"
        evidence.write_text(
            "Independent dense comparison passed: max error 2e-15.\n")
        write_json(
            folder / "verification.json", {
                "status":
                "passed",
                "evidence":
                evidence.relative_to(self.root).as_posix(),
                "evidence_sha256":
                hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "result_sha256":
                hashlib.sha256(
                    (folder / "result.json").read_bytes()).hexdigest(),
                "transcript_sha256":
                hashlib.sha256(
                    (folder / "transcript.jsonl").read_bytes()).hexdigest(),
                "verified_completion": {
                    "task_seconds": 1,
                    "wall_seconds": 1,
                    "tokens": None,
                    "tool_calls": None,
                    "failed_attempts": 0
                }
            })
        bundle = assessment.build_bundle(self.root)
        run = next(r for r in bundle["models"][0]["runs"]
                   if r["case_id"] == case_id)
        self.assertEqual(run["verification"], "passed")
        self.assertTrue(all(value is None for value in run["assertions"]))
        self.assertIsNone(run["claimed_success"])
        self.assertTrue(run["grading_evidence"].endswith("verification.json"))
        evidence.write_text("The log has been replaced after the check.\n")
        with self.assertRaisesRegex(ValueError, "evidence.*binding mismatch"):
            assessment.build_bundle(self.root)

    def test_executable_check_cannot_be_attached_to_an_advice_case(self):
        write_json(
            self.folder / "verification.json", {
                "status":
                "not_run",
                "evidence":
                None,
                "result_sha256":
                hashlib.sha256(
                    (self.folder / "result.json").read_bytes()).hexdigest(),
                "transcript_sha256":
                hashlib.sha256((self.folder /
                                "transcript.jsonl").read_bytes()).hexdigest()
            })
        with self.assertRaisesRegex(ValueError, "applicability"):
            assessment.build_bundle(self.root)

    def test_symlinked_transcript_cannot_escape_evidence_bundle(self):
        external = self.root.parent / (self.root.name + "-outside.jsonl")
        external.write_text(TRANSCRIPT)
        self.addCleanup(external.unlink)
        transcript = self.folder / "transcript.jsonl"
        transcript.unlink()
        transcript.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "escapes"):
            assessment.build_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
