"""Keep the independently installable runtime eval package in canonical sync."""

import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unittest

DEV_EVALS = Path(__file__).resolve().parents[1]
CATALOG_EVALS = DEV_EVALS.parents[1] / "cudaq-algorithms" / "evals"
NEGATIVE_IDS = {
    "negative-cudaq-install",
    "negative-generic-quantum-concept",
    "negative-generic-cudaq-kernel",
}


class CatalogEvalTests(unittest.TestCase):

    def test_runtime_dataset_preserves_every_canonical_record(self):
        canonical = json.loads((DEV_EVALS / "evals.json").read_text())
        path = CATALOG_EVALS / "evals.json"
        self.assertTrue(path.is_file(),
                        "runtime skill needs its own eval dataset")
        self.assertFalse(path.is_symlink())
        self.assertTrue(path.read_bytes().isascii())
        self.assertEqual(json.loads(path.read_text()), canonical)

    def test_runtime_negatives_are_explicit_and_unchanged(self):
        path = CATALOG_EVALS / "evals.json"
        self.assertTrue(path.is_file(),
                        "runtime evals must include negative cases")
        cases = json.loads(path.read_text())["evals"]
        self.assertEqual(len(cases), 62)
        self.assertEqual(len({case["id"] for case in cases}), 62)
        self.assertTrue(all("expected_skill" in case for case in cases))
        self.assertEqual(
            {case["id"]
             for case in cases if case["expected_skill"] is None},
            NEGATIVE_IDS)
        self.assertEqual(
            sum(case["expected_skill"] == "cudaq-algorithms"
                for case in cases), 59)

    def test_fixtures_are_complete_local_regular_and_byte_identical(self):
        path = CATALOG_EVALS / "evals.json"
        self.assertTrue(path.is_file(),
                        "runtime evals need their local fixtures")
        cases = json.loads(path.read_text())["evals"]
        referenced = {name for case in cases for name in case["files"]}
        self.assertEqual(len(referenced), 10)
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
