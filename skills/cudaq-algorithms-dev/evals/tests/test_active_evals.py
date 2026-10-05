"""Validate the delivered suite without loading archived source suites.

The manifest records source provenance and pins every complete case and fixture.
Intentional content changes require an explicit review and manifest rebaseline.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "evals.json"
MANIFEST = ROOT / "manifest.json"
CASE_KEYS = {
    "id", "prompt", "expected_output", "expected_skill", "files", "assertions"
}
CANONICALIZATION = (
    'UTF-8 json.dumps(record, sort_keys=True, separators=(",", ":"), '
    'ensure_ascii=False, allow_nan=False)')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def record_sha256(record):
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ActiveEvalContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.suite = load_json(ACTIVE)
        cls.manifest = load_json(MANIFEST)

    def test_suite_schema_and_case_fields(self):
        self.assertEqual(set(self.suite), {"skill_name", "evals"})
        self.assertEqual(self.suite["skill_name"], "cudaq-algorithms")
        self.assertIsInstance(self.suite["evals"], list)
        for case in self.suite["evals"]:
            with self.subTest(case=case.get("id")):
                self.assertEqual(set(case), CASE_KEYS)
                for key in ("id", "prompt", "expected_output"):
                    self.assertIsInstance(case[key], str)
                    self.assertTrue(case[key].strip(), key)
                self.assertIn(case["expected_skill"],
                              ("cudaq-algorithms", None))
                for key in ("files", "assertions"):
                    self.assertIsInstance(case[key], list)
                    for value in case[key]:
                        self.assertIsInstance(value, str)
                        self.assertTrue(value.strip(), key)
                self.assertTrue(case["assertions"])
                self.assertEqual(len(case["files"]), len(set(case["files"])))

    def test_case_counts_and_unique_ids(self):
        cases = self.suite["evals"]
        self.assertEqual(len(cases), 62)
        self.assertEqual(len({case["id"] for case in cases}), 62)
        self.assertEqual(
            Counter(case["expected_skill"] for case in cases),
            {
                "cudaq-algorithms": 59,
                None: 3
            },
        )

    def test_manifest_contract_and_source_provenance(self):
        manifest = self.manifest
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["canonicalization"], CANONICALIZATION)
        self.assertEqual(
            manifest["suite"], {
                "file": "evals.json",
                "skill_name": "cudaq-algorithms",
                "count": 62,
                "positive": 59,
                "negative": 3,
            })
        self.assertEqual(
            [(group["name"], group["source"], group["count"])
             for group in manifest["groups"]],
            [("regression", "regression-evals.json", 42),
             ("science", "eval_science.json", 20)],
        )
        for group in manifest["groups"]:
            self.assertRegex(group["source_sha256"], r"\A[0-9a-f]{64}\Z")
            self.assertEqual(len(group["cases"]), group["count"])
            for case in group["cases"]:
                self.assertEqual(set(case), {"id", "sha256"})
                self.assertRegex(case["sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertEqual(len(manifest["fixtures"]), 10)

    def test_ordered_groups_and_complete_record_integrity(self):
        expected = [
            case for group in self.manifest["groups"]
            for case in group["cases"]
        ]
        actual = self.suite["evals"]
        self.assertEqual([case["id"] for case in actual],
                         [case["id"] for case in expected])
        for case, pinned in zip(actual, expected):
            with self.subTest(case=case["id"]):
                self.assertEqual(record_sha256(case), pinned["sha256"])

    def test_fixture_inventory_and_safe_existing_paths(self):
        referenced = {
            name
            for case in self.suite["evals"]
            for name in case["files"]
        }
        self.assertEqual(len(referenced), 10)
        self.assertEqual(referenced, set(self.manifest["fixtures"]))
        for name in referenced:
            with self.subTest(fixture=name):
                relative = PurePosixPath(name)
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertNotIn("\\", name)
                self.assertEqual(relative.parts[0], "files")
                self.assertEqual(relative.as_posix(), name)
                target = (ROOT / name).resolve()
                self.assertTrue(
                    target.is_relative_to((ROOT / "files").resolve()))
                self.assertTrue(target.is_relative_to(ROOT.resolve()))
                self.assertTrue(target.is_file(), name)

    def test_fixture_byte_integrity(self):
        for name, digest in self.manifest["fixtures"].items():
            with self.subTest(fixture=name):
                self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
                self.assertEqual(
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                    digest)


if __name__ == "__main__":
    unittest.main()
