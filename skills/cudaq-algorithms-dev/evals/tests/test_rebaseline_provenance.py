"""Check that the reviewed rubric changes preserve their historical boundary."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

EVALS = Path(__file__).resolve().parents[1]
CATALOG = EVALS.parents[1] / "cudaq-algorithms" / "evals"
PROVENANCE = EVALS / "rebaselines" / "2026-10-15-uccsd-host-parity.json"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def record_digest(record):
    return digest(
        json.dumps(record,
                   sort_keys=True,
                   separators=(",", ":"),
                   ensure_ascii=False,
                   allow_nan=False).encode("utf-8"))


class RebaselineProvenanceTests(unittest.TestCase):

    def provenance(self):
        self.assertTrue(PROVENANCE.is_file(),
                        "a changed evaluation needs reviewable provenance")
        return json.loads(PROVENANCE.read_text(encoding="utf-8"))

    def test_previous_suite_is_recoverable_without_regrading_or_case_drift(
            self):
        """An unrecorded change to any unreviewed case breaks recovery."""
        review = self.provenance()
        changes = {change["id"]: change for change in review["changes"]}
        self.assertEqual(list(changes), review["changed_case_ids"])
        for name, root in (("development", EVALS), ("runtime", CATALOG)):
            with self.subTest(dataset=name):
                payload = (root / "evals.json").read_bytes()
                suite = json.loads(payload)
                previous = copy.deepcopy(suite)
                hashes = review["datasets"][name]
                self.assertEqual(digest(payload), hashes["after_sha256"])
                for index, case in enumerate(suite["evals"]):
                    if case["id"] not in changes:
                        continue
                    change = changes[case["id"]]
                    before = change["before_record"]
                    self.assertEqual(record_digest(before),
                                     change["before_sha256"])
                    self.assertEqual(record_digest(case),
                                     change["after_sha256"])
                    self.assertEqual(
                        sorted(key for key in case
                               if case[key] != before[key]),
                        change["changed_fields"])
                    # The adversarial user premise, files and trigger stay fixed.
                    for field in ("id", "prompt", "files", "expected_skill"):
                        self.assertEqual(case[field], before[field])
                    previous["evals"][index] = before
                prior_bytes = (json.dumps(
                    previous, indent=2, ensure_ascii=hashes["ensure_ascii"]) +
                               "\n").encode("utf-8")
                self.assertEqual(digest(prior_bytes), hashes["before_sha256"])
        self.assertFalse(review["retroactive_regrading"])

    def test_manifest_changes_only_reviewed_case_hashes(self):
        """Accidental wholesale baseline regeneration loses the prior manifest."""
        review = self.provenance()
        payload = (EVALS / "manifest.json").read_text(encoding="utf-8")
        self.assertEqual(digest(payload.encode("utf-8")),
                         review["manifest"]["after_sha256"])
        manifest = json.loads(payload)
        pinned = {
            case["id"]: case["sha256"]
            for group in manifest["groups"]
            for case in group["cases"]
        }
        for change in review["changes"]:
            self.assertEqual(pinned[change["id"]], change["after_sha256"])
            self.assertEqual(payload.count(change["after_sha256"]), 1)
            payload = payload.replace(change["after_sha256"],
                                      change["before_sha256"])
        self.assertEqual(digest(payload.encode("utf-8")),
                         review["manifest"]["before_sha256"])
        self.assertEqual(
            {
                group["source"]: group["source_sha256"]
                for group in manifest["groups"]
            }, review["historical_source_sha256"])
        self.assertEqual(manifest["version"], review["manifest_version"])

    def test_fixtures_and_protocol_retain_the_pre_review_bytes(self):
        review = self.provenance()
        for name, expected in review["fixtures_sha256"].items():
            for root in (EVALS, CATALOG):
                with self.subTest(fixture=name, root=root):
                    self.assertEqual(digest((root / name).read_bytes()),
                                     expected)
        for name, root in (("development", EVALS), ("runtime", CATALOG)):
            with self.subTest(protocol=name):
                self.assertEqual(digest((root / "config.yml").read_bytes()),
                                 review["protocol_sha256"][name])


if __name__ == "__main__":
    unittest.main()
