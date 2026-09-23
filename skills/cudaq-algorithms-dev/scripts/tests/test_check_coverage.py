# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise coverage consistency with small real trees and run records."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).resolve().parents[1] / "check_coverage.py"


class CoverageTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = "references/family/operation.md"
        self.write("SKILL.md",
                   "# Skill\n\n[Operation](references/family/operation.md)\n")
        self.write(self.record,
                   "# Operation\n\n## Inputs\n\nScientific contract.\n")
        self.write("evals/evals.json", {"evals": [{"id": "route-operation"}]})
        digest = hashlib.sha256(
            (self.root / self.record).read_bytes()).hexdigest()
        self.manifest = {
            "cases": [{
                "id": "execute-operation",
                "required_public_apis": ["pkg.operation"]
            }],
            "runs": [{
                "id": "execute-operation--1--skill",
                "case": "execute-operation",
                "arm": "skill",
                "repetition": 1
            }],
            "source_inventory": {
                "skills/cudaq-algorithms/" + self.record: digest
            },
        }
        self.run = {
            "id": "execute-operation--1--skill",
            "case": "execute-operation",
            "arm": "skill",
            "repetition": 1,
            "attempted": True,
            "status": "pass",
            "passed": True,
            "grading": {
                "passed":
                True,
                "checks": [{
                    "variant": variant,
                    "returncode": 0,
                    "passed": True,
                    "numeric": {
                        "passed": True
                    },
                    "host_api": {
                        "passed": True
                    },
                    "device_api": {
                        "passed": True
                    }
                } for variant in (0, 1)]
            }
        }
        self.registry = {
            "schema_version":
            1,
            "campaigns": [{
                "id": "example",
                "manifest": "evals/results/example/manifest.json"
            }],
            "features": [{
                "id":
                "operation",
                "family":
                "family",
                "record":
                self.record,
                "symbols": ["pkg.operation"],
                "behavioral_evals": ["route-operation"],
                "executions": [{
                    "campaign": "example",
                    "case": "execute-operation",
                    "arm": "skill",
                    "repetitions": 1,
                    "status": "scientific_pass",
                    "scope": "Two small inputs only."
                }]
            }],
            "support_records": [],
        }

    def write(self, name, content):
        self.write_at(self.root, name, content)

    def write_at(self, root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            content if isinstance(content, str) else json.dumps(content))

    def prepare_split(self):
        runtime = self.root / "skills/cudaq-algorithms"
        support = self.root / "skills/cudaq-algorithms-dev"
        self.write_at(
            runtime, "SKILL.md",
            "# Skill\n\n[Operation](references/family/operation.md)\n")
        self.write_at(runtime, self.record,
                      "# Operation\n\n## Inputs\n\nScientific contract.\n")
        self.write_at(support, "coverage/features.json", self.registry)
        self.write_at(support, "evals/evals.json",
                      {"evals": [{
                          "id": "route-operation"
                      }]})
        self.write_at(support, "evals/results/example/manifest.json",
                      self.manifest)
        self.write_at(
            support,
            "evals/results/example/runs/execute-operation--1--skill/result.json",
            self.run)
        self.write_at(
            support,
            "authoring/architecture.md",
            "# Authoring\n\n[Operation](../../cudaq-algorithms/references/family/operation.md)\n",
        )
        return runtime, support

    def execute_split(self, runtime, support, *args):
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER), "--root",
                str(runtime), "--support-root",
                str(support), *args
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ, "PYTHONDONTWRITEBYTECODE": "1"
            },
            timeout=10,
        )

    def execute(self, *args):
        self.write("coverage/features.json", self.registry)
        self.write("evals/results/example/manifest.json", self.manifest)
        self.write(
            "evals/results/example/runs/execute-operation--1--skill/result.json",
            self.run)
        return subprocess.run(
            [sys.executable,
             str(CHECKER), "--root",
             str(self.root), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ, "PYTHONDONTWRITEBYTECODE": "1"
            },
            timeout=10)

    def assert_problem(self, diagnostic):
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(diagnostic, result.stdout + result.stderr)

    def test_valid_registry_reports_scoped_evidence_and_single_feature(self):
        result = self.execute("--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["features"], 1)
        self.assertEqual(report["features_with_scientific_evidence"], 1)
        self.assertEqual(report["features_with_current_record_evidence"], 1)
        result = self.execute("--feature", "operation")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["id"], "operation")

    def test_split_roots_accept_declared_cross_root_links(self):
        runtime, support = self.prepare_split()
        result = self.execute_split(runtime, support, "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["features"], 1)
        self.assertEqual(report["features_with_current_record_evidence"], 1)

    def test_split_behavioral_eval_suites_use_the_union_of_declared_ids(self):
        self.registry["behavioral_eval_suites"] = [
            "evals/evals.json", "evals/regression-evals.json"
        ]
        runtime, support = self.prepare_split()
        self.write_at(support, "evals/evals.json",
                      {"evals": [{
                          "id": "new-active-case"
                      }]})
        self.write_at(support, "evals/regression-evals.json",
                      {"evals": [{
                          "id": "route-operation"
                      }]})
        result = self.execute_split(runtime, support)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_duplicate_behavioral_eval_ids_across_suites_are_rejected(self):
        self.registry["behavioral_eval_suites"] = [
            "evals/evals.json", "evals/regression-evals.json"
        ]
        runtime, support = self.prepare_split()
        self.write_at(support, "evals/regression-evals.json",
                      {"evals": [{
                          "id": "route-operation"
                      }]})
        result = self.execute_split(runtime, support)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate behavioral eval id",
                      result.stdout + result.stderr)

    def test_behavioral_eval_suite_must_exist_inside_support_root(self):
        self.registry["behavioral_eval_suites"] = ["evals/missing.json"]
        runtime, support = self.prepare_split()
        result = self.execute_split(runtime, support)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing behavioral eval suite",
                      result.stdout + result.stderr)

        self.registry["behavioral_eval_suites"] = ["../other-evals.json"]
        self.write_at(support, "coverage/features.json", self.registry)
        self.write_at(support.parent, "other-evals.json", {"evals": []})
        result = self.execute_split(runtime, support)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside support root", result.stdout + result.stderr)

    def test_split_roots_keep_registry_paths_separately_contained(self):
        runtime, support = self.prepare_split()
        self.registry["features"][0][
            "record"] = "../cudaq-algorithms-dev/references/family/operation.md"
        self.write_at(support, "references/family/operation.md",
                      "# Wrong tree\n")
        self.write_at(support, "coverage/features.json", self.registry)
        result = self.execute_split(runtime, support)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside runtime root", result.stdout + result.stderr)

        self.registry["features"][0]["record"] = self.record
        self.registry["campaigns"][0][
            "manifest"] = "../cudaq-algorithms/manifest.json"
        self.write_at(runtime, "manifest.json", self.manifest)
        self.write_at(support, "coverage/features.json", self.registry)
        result = self.execute_split(runtime, support)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside support root", result.stdout + result.stderr)

    def test_support_document_cannot_link_to_an_undeclared_sibling(self):
        runtime, support = self.prepare_split()
        sibling = self.root / "skills/other-skill"
        self.write_at(sibling, "secret.md", "# Secret\n")
        self.write_at(
            support,
            "authoring/architecture.md",
            "# Authoring\n\n[Secret](../../other-skill/secret.md)\n",
        )
        result = self.execute_split(runtime, support)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside declared roots", result.stdout + result.stderr)

    def test_missing_record_and_duplicate_feature_are_rejected(self):
        (self.root / self.record).unlink()
        self.assert_problem("missing record")
        self.write(self.record, "# Operation\n")
        self.registry["features"].append(dict(self.registry["features"][0]))
        self.assert_problem("duplicate feature id")

    def test_unknown_behavioral_and_execution_case_are_rejected(self):
        self.registry["features"][0]["behavioral_evals"] = ["invented-eval"]
        self.assert_problem("unknown behavioral eval")
        self.registry["features"][0]["behavioral_evals"] = []
        self.registry["features"][0]["executions"][0]["case"] = "invented-case"
        self.assert_problem("unknown execution case")

    def test_unregistered_reference_and_broken_link_are_rejected(self):
        self.write("references/family/new-operation.md", "# New operation\n")
        self.assert_problem("unregistered operational record")
        self.registry["support_records"] = [
            "references/family/new-operation.md"
        ]
        self.write(self.record, "# Operation\n\n[Missing](missing.md)\n")
        self.assert_problem("broken link")

    def test_false_numeric_claim_cannot_pass_from_top_level_status(self):
        self.run["grading"]["checks"][1]["numeric"]["passed"] = False
        self.assert_problem("scientific evidence does not pass")

    def test_blocked_slot_is_not_execution_or_scientific_evidence(self):
        self.run.update(attempted=False,
                        status="infrastructure_failure",
                        passed=False)
        self.run["grading"] = {}
        self.assert_problem("scientific evidence does not pass")
        self.registry["features"][0]["executions"][0]["status"] = "blocked"
        result = self.execute("--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["features_with_scientific_evidence"], 0)

    def test_wrong_symbol_and_missing_repetition_are_rejected(self):
        self.registry["features"][0]["symbols"] = ["pkg.unrelated"]
        self.assert_problem("no required API matches feature")
        self.registry["features"][0]["symbols"] = ["pkg.operation"]
        self.registry["features"][0]["executions"][0]["repetitions"] = 2
        self.assert_problem("repetition count")

    def test_hygiene_failure_does_not_erase_scientific_evidence(self):
        self.run.update(status="model_failure",
                        passed=False,
                        scope_changes=["__pycache__/app.pyc"])
        result = self.execute("--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["features_with_scientific_evidence"], 1)

    def test_changed_record_is_historical_not_current_evidence(self):
        self.write(self.record,
                   "# Operation\n\nReorganized scientific contract.\n")
        result = self.execute("--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["features_with_current_record_evidence"],
            0)

    def test_grading_hygiene_is_separate_from_scientific_evidence(self):
        self.run["grading"]["passed"] = False
        self.run["grading"]["checks"][0].update(passed=False,
                                                scope_changes=["extra.txt"])
        result = self.execute("--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["features_with_scientific_evidence"], 1)

    def test_runtime_reachability_cannot_depend_on_authoring(self):
        self.write("SKILL.md",
                   "# Skill\n\n[Maintain](authoring/architecture.md)\n")
        self.write("authoring/architecture.md",
                   "# Authoring\n\n[Operation](../" + self.record + ")\n")
        self.assert_problem("unreachable operational record")

    def test_family_and_symbol_types_are_validated(self):
        self.registry["features"][0]["family"] = "wrong"
        self.assert_problem("family does not match record")
        self.registry["features"][0]["family"] = "family"
        self.registry["features"][0]["symbols"] = "pkg.operation"
        self.assert_problem("public symbols must be a nonempty list")

    def test_historical_metadata_indices_match_the_record(self):
        self.registry[
            "historical_metadata_inventory"] = "evals/results/migration.json"
        self.write(
            "evals/results/migration.json", {
                "removed_metadata": [{
                    "source_record": "references/family/another.md",
                    "text": "Status: draft."
                }]
            })
        self.registry["features"][0]["historical_metadata_indices"] = [0]
        self.assert_problem("historical metadata source mismatch")
        self.registry["features"][0]["historical_metadata_indices"] = [3]
        self.assert_problem("historical metadata index out of range")

    def test_old_baseline_cannot_claim_skill_record_evidence(self):
        self.registry["features"][0]["executions"][0]["arm"] = "baseline"
        self.manifest["runs"][0]["arm"] = "baseline"
        self.run["arm"] = "baseline"
        self.assert_problem("baseline has no skill record evidence")

    def test_catalog_contract_cannot_be_demoted_to_support(self):
        self.write(
            self.record, "# Operation\n\n"
            "| Operation + object | Public entry point | Kind / layer | Capabilities | Record |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| act / object | `pkg.operation` | host | none | [Operation](operation.md) |\n"
        )
        self.registry["features"] = []
        self.registry["support_records"] = [self.record]
        self.assert_problem("selectable catalog contract is not a feature")

    def test_outside_paths_and_unreachable_records_are_rejected(self):
        self.registry["features"][0]["record"] = "../outside.md"
        self.assert_problem("outside skill root")
        self.registry["features"][0]["record"] = self.record
        self.write("SKILL.md", "# Skill\n")
        self.assert_problem("unreachable operational record")

    def test_fenced_example_links_are_not_live_dependencies(self):
        self.write(
            self.record,
            "# Operation\n\n```markdown\n[Example](not-created.md)\n```\n")
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
