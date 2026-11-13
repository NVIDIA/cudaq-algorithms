# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the independently installable runtime eval package in canonical sync.

The catalog dataset is a reviewed subset of the development suite: every
record it carries must be identical to the development record with the same
id, in development order, and the set of ids is fixed here. The full suite
stays in the development package for the paired campaign; nvskills-ci runs
only this subset.
"""

import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unittest

DEV_EVALS = Path(__file__).resolve().parents[1]
CATALOG_EVALS = DEV_EVALS.parents[2] / "skills" / "cudaq-algorithms" / "evals"
NEGATIVE_IDS = {
    "negative-cudaq-install",
    "negative-generic-quantum-concept",
    "negative-generic-cudaq-kernel",
}
# Reviewed canonical subset shipped to the catalog and run by nvskills-ci.
CATALOG_IDS = {
    # routing and activation
    "explicit-primitive-reasoning",
    "contextual-existing-primitive",
    "unknown-future-primitive",
    "unsupported-library-workflow",
    # state preparation contracts
    "state-preparation-provider-selection",
    "state-preparation-injection-composition",
    # one boundary case per area
    "block-encoding-capability-boundary",
    "qsvt-phase-and-recovery-boundary",
    "trotter-evolution-resource-boundary",
    "chemistry-bridge-dependency-boundary",
    # behaviour and safety
    "qpu-authorization-boundary",
    "fixture-prompt-injection-boundary",
    "unexecuted-code-evidence",
    "source-unavailable-fallback",
    "implementation-scope-overreach",
    "chemistry-end-to-end-composition",
    # one scientific case per area
    "science-block-encoding-truncation-tradeoff",
    "science-qsvt-low-energy-filter-choice",
    "science-trotter-conservation-versus-accuracy",
    "science-chemistry-reference-quality",
    "science-double-factorization-leaf-budget",
} | NEGATIVE_IDS
CATALOG_CASE_COUNT = 24
CATALOG_POSITIVE_COUNT = 21
CATALOG_FIXTURE_COUNT = 5


class CatalogEvalTests(unittest.TestCase):

    def test_runtime_dataset_is_an_identical_ordered_subset(self):
        canonical = json.loads((DEV_EVALS / "evals.json").read_text())
        path = CATALOG_EVALS / "evals.json"
        self.assertTrue(path.is_file(),
                        "runtime skill needs its own eval dataset")
        self.assertFalse(path.is_symlink())
        self.assertTrue(path.read_bytes().isascii())
        catalog = json.loads(path.read_text())
        self.assertEqual(catalog["skill_name"], canonical["skill_name"])
        self.assertEqual(set(catalog.keys()), set(canonical.keys()))
        expected = [
            case for case in canonical["evals"] if case["id"] in CATALOG_IDS
        ]
        self.assertEqual(len(expected), len(CATALOG_IDS),
                         "every catalog id must exist in the canonical suite")
        self.assertEqual(catalog["evals"], expected)

    def test_runtime_negatives_are_explicit_and_unchanged(self):
        path = CATALOG_EVALS / "evals.json"
        self.assertTrue(path.is_file(),
                        "runtime evals must include negative cases")
        cases = json.loads(path.read_text())["evals"]
        self.assertEqual(len(cases), CATALOG_CASE_COUNT)
        self.assertEqual(len({case["id"]
                              for case in cases}), CATALOG_CASE_COUNT)
        self.assertEqual({case["id"] for case in cases}, CATALOG_IDS)
        self.assertTrue(all("expected_skill" in case for case in cases))
        self.assertEqual(
            {case["id"]
             for case in cases if case["expected_skill"] is None},
            NEGATIVE_IDS)
        self.assertEqual(
            sum(case["expected_skill"] == "cudaq-algorithms"
                for case in cases), CATALOG_POSITIVE_COUNT)

    def test_fixtures_are_complete_local_regular_and_byte_identical(self):
        path = CATALOG_EVALS / "evals.json"
        self.assertTrue(path.is_file(),
                        "runtime evals need their local fixtures")
        cases = json.loads(path.read_text())["evals"]
        referenced = {name for case in cases for name in case["files"]}
        self.assertEqual(len(referenced), CATALOG_FIXTURE_COUNT)
        actual = {
            p.relative_to(CATALOG_EVALS).as_posix()
            for p in (CATALOG_EVALS / "files").rglob("*") if p.is_file()
        }
        self.assertEqual(actual, referenced)
        for name in referenced:
            with self.subTest(fixture=name):
                relative = PurePosixPath(name)
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertEqual(relative.parts[0], "files")
                path = CATALOG_EVALS / name
                self.assertFalse(path.is_symlink())
                self.assertTrue(path.resolve().is_relative_to(CATALOG_EVALS))
                self.assertEqual(path.read_bytes(),
                                 (DEV_EVALS / name).read_bytes())

    def test_eval_directory_works_without_development_skill(self):
        self.assertTrue(CATALOG_EVALS.is_dir(),
                        "runtime eval package is missing")
        self.assertFalse(CATALOG_EVALS.is_symlink())
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "cudaq-algorithms" / "evals"
            shutil.copytree(CATALOG_EVALS, copied, symlinks=True)
            self.assertTrue((copied / "config.yml").is_file())
            self.assertTrue((copied / "EVAL.md").is_file())
            for case in json.loads(
                (copied / "evals.json").read_text())["evals"]:
                for name in case["files"]:
                    path = copied / name
                    self.assertTrue(path.is_file())
                    self.assertTrue(path.resolve().is_relative_to(copied))


if __name__ == "__main__":
    unittest.main()
