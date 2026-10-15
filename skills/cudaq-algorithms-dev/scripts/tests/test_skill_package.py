"""Guard the runtime skill's public catalog packaging requirements."""
from pathlib import Path
import unittest

import yaml

SKILL = Path(__file__).resolve().parents[3] / "cudaq-algorithms"


class SkillPackageTests(unittest.TestCase):

    def test_discovery_metadata_is_short_and_tagged(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        header = yaml.safe_load(text.split("---", 2)[1])
        self.assertEqual(header["name"], SKILL.name)
        self.assertIsInstance(header["description"], str)
        self.assertGreaterEqual(len(header["description"]), 50)
        self.assertLessEqual(len(header["description"]), 150)
        tags = header.get("tags")
        self.assertIsInstance(tags, list)
        self.assertTrue(tags)
        self.assertTrue(
            all(isinstance(tag, str) and tag.strip() for tag in tags))
        self.assertEqual(header["metadata"]["tags"], tags)

    def test_runtime_package_text_is_ascii(self):
        violations = []
        for path in SKILL.rglob("*"):
            if path.suffix not in {".md", ".py", ".json", ".yml", ".yaml"}:
                continue
            if not path.read_text(encoding="utf-8").isascii():
                violations.append(str(path.relative_to(SKILL)))
        self.assertEqual(violations, [], "Non-ASCII runtime package files")

    def test_catalog_card_and_standalone_evaluation_are_present(self):
        for name in ("skill-card.md", "evals/evals.json", "evals/config.yml"):
            with self.subTest(name=name):
                path = SKILL / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
