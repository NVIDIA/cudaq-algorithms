# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "evals.json"
REGRESSION = ROOT / "regression-evals.json"
EXPECTED_REGRESSION_SHA256 = ("131d9d5dda81a24c653e4faa5d012dd5"
                              "c9f5f5fe6fb21e502c7e6f85adec22d7")
EXPECTED_IDS = {
    "lean-p01-walk-state-preparation",
    "lean-p02-structural-block-encoding-action",
    "lean-p03-qsvt-real-time-recovery",
    "lean-p04-trotter-identity-phase",
    "lean-p05-catalyst-configuration-comparison",
    "lean-p06-double-factorization-diagnostics",
    "lean-p07-missing-researcher-feedback",
}
CASE_KEYS = {
    "id", "prompt", "expected_output", "expected_skill", "files", "assertions"
}


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


class ActiveEvalContractTests(unittest.TestCase):

    def test_active_suite_has_exactly_the_seven_review_cases(self):
        suite = load_json(ACTIVE)
        self.assertEqual(set(suite), {"skill_name", "evals"})
        self.assertEqual(suite["skill_name"], "cudaq-algorithms")
        self.assertEqual(len(suite["evals"]), 7)
        self.assertEqual({case["id"] for case in suite["evals"]}, EXPECTED_IDS)
        for case in suite["evals"]:
            self.assertEqual(set(case), CASE_KEYS)
            self.assertEqual(case["expected_skill"], "cudaq-algorithms")
            self.assertIsInstance(case["files"], list)
            self.assertTrue(case["prompt"].strip())
            self.assertTrue(case["expected_output"].strip())
            self.assertGreaterEqual(len(case["assertions"]), 2)

    def test_every_worker_fixture_is_an_existing_safe_relative_file(self):
        suite = load_json(ACTIVE)
        for case in suite["evals"]:
            for name in case["files"]:
                relative = Path(name)
                self.assertFalse(relative.is_absolute(), name)
                self.assertNotIn("..", relative.parts, name)
                target = (ROOT / relative).resolve()
                self.assertTrue(target.is_relative_to(ROOT.resolve()), name)
                self.assertTrue(target.is_file(), name)

    def test_historical_suite_is_preserved_byte_for_byte(self):
        digest = hashlib.sha256(REGRESSION.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_REGRESSION_SHA256)
        self.assertEqual(len(load_json(REGRESSION)["evals"]), 42)

    def test_shared_science_fixture_has_independent_sanity_controls(self):
        fixture = load_json(ROOT / "files" / "lean-seven-science.json")
        self.assertEqual(set(fixture), {"p01", "p03", "p04", "p05", "p06"})

        p01 = fixture["p01"]
        self.assertEqual(len(p01["amplitudes"]), 4)
        self.assertAlmostEqual(
            sum(value * value for value in p01["amplitudes"]), 1.0)

        p03 = fixture["p03"]
        h = p03["hamiltonian"]
        self.assertEqual(h[0][1], h[1][0])
        self.assertAlmostEqual(sum(value * value for value in p03["ket"]), 1.0)
        radius = math.hypot(p03["a_x"], p03["b_z"])
        alpha = abs(p03["a_x"]) + abs(p03["b_z"])
        cosine = math.acos(math.cos(radius * p03["time"]) / 2)
        sine_sum = math.asin(alpha * math.sin(radius * p03["time"]) /
                             (2 * radius))
        self.assertAlmostEqual(p03["cos_phases"][0], cosine, places=14)
        self.assertAlmostEqual(sum(p03["sin_phases"]), sine_sum, places=14)

        p04 = fixture["p04"]
        terms = dict((word, coefficient) for coefficient, word in p04["terms"])
        self.assertNotEqual(terms["X"], 0)
        self.assertNotEqual(terms["Z"], 0)
        self.assertNotEqual(terms["I"], 0)

        p05 = fixture["p05"]
        self.assertEqual(len(p05["configurations"]), 2)
        for key in ("cluster", "electron_count", "orbital_order",
                    "integral_units"):
            self.assertEqual(p05["configurations"][0][key],
                             p05["configurations"][1][key])
        self.assertNotIn("active_space_energy_offset", p05)

        p06 = fixture["p06"]
        self.assertEqual(len(p06["target_eri"]), 2)
        self.assertEqual(len(p06["rotations"]), len(p06["cores"]))
        for rotation in p06["rotations"]:
            dot00 = sum(row[0] * row[0] for row in rotation)
            dot11 = sum(row[1] * row[1] for row in rotation)
            dot01 = sum(row[0] * row[1] for row in rotation)
            self.assertAlmostEqual(dot00, 1.0, places=14)
            self.assertAlmostEqual(dot11, 1.0, places=14)
            self.assertAlmostEqual(dot01, 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
