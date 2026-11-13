# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the reviewed rubric changes preserve their historical boundary.

Every reviewed change to the development suite is recorded under
``rebaselines/`` as a dated JSON record. The records form a chain: each one's
``before`` hashes equal the previous record's ``after`` hashes, and the newest
record's ``after`` hashes match the files on disk. An unrecorded edit to any
case, the manifest, a fixture or a protocol file breaks the chain.
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

EVALS = Path(__file__).resolve().parents[1]
CATALOG = EVALS.parents[2] / "skills" / "cudaq-algorithms" / "evals"
REBASELINES = sorted((EVALS / "rebaselines").glob("*.json"))
# Fields a reviewed change may never alter unless the record says otherwise.
DEFAULT_IMMUTABLE_FIELDS = ("id", "prompt", "files", "expected_skill")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def record_digest(record):
    return digest(
        json.dumps(record,
                   sort_keys=True,
                   separators=(",", ":"),
                   ensure_ascii=False,
                   allow_nan=False).encode("utf-8"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RebaselineProvenanceTests(unittest.TestCase):

    def reviews(self):
        self.assertTrue(REBASELINES,
                        "a changed evaluation needs reviewable provenance")
        return [load(path) for path in REBASELINES]

    def test_records_form_an_unbroken_chain_ending_at_the_current_suite(self):
        reviews = self.reviews()
        for previous, review in zip(reviews, reviews[1:]):
            with self.subTest(record=review["date"]):
                self.assertEqual(
                    review["datasets"]["development"]["before_sha256"],
                    previous["datasets"]["development"]["after_sha256"])
                self.assertEqual(review["manifest"]["before_sha256"],
                                 previous["manifest"]["after_sha256"])
        latest = reviews[-1]
        self.assertEqual(digest((EVALS / "evals.json").read_bytes()),
                         latest["datasets"]["development"]["after_sha256"])
        self.assertEqual(digest((CATALOG / "evals.json").read_bytes()),
                         latest["datasets"]["runtime"]["after_sha256"])
        self.assertEqual(digest((EVALS / "manifest.json").read_bytes()),
                         latest["manifest"]["after_sha256"])

    def test_each_review_recovers_its_prior_suite_without_regrading(self):
        """An unrecorded change to any unreviewed case breaks recovery."""
        reviews = self.reviews()
        suite_bytes = (EVALS / "evals.json").read_bytes()
        # Walk the chain backwards, recovering each prior development suite
        # from the recorded before-records and checking its hash.
        for review in reversed(reviews):
            with self.subTest(record=review["date"]):
                changes = {
                    change["id"]: change
                    for change in review["changes"]
                }
                self.assertEqual(list(changes), review["changed_case_ids"])
                hashes = review["datasets"]["development"]
                self.assertEqual(digest(suite_bytes), hashes["after_sha256"])
                suite = json.loads(suite_bytes)
                previous = copy.deepcopy(suite)
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
                    immutable = change.get("immutable_fields",
                                           DEFAULT_IMMUTABLE_FIELDS)
                    self.assertFalse(
                        set(immutable) & set(change["changed_fields"]))
                    for field in immutable:
                        self.assertEqual(case[field], before[field])
                    previous["evals"][index] = before
                suite_bytes = (json.dumps(
                    previous, indent=2, ensure_ascii=hashes["ensure_ascii"]) +
                               "\n").encode("utf-8")
                self.assertEqual(digest(suite_bytes), hashes["before_sha256"])
                self.assertFalse(review["retroactive_regrading"])

    def test_manifest_changes_only_reviewed_case_hashes(self):
        """Accidental wholesale baseline regeneration loses the prior manifest."""
        reviews = self.reviews()
        payload = (EVALS / "manifest.json").read_text(encoding="utf-8")
        manifest = json.loads(payload)
        pinned = {
            case["id"]: case["sha256"]
            for group in manifest["groups"]
            for case in group["cases"]
        }
        for review in reversed(reviews):
            with self.subTest(record=review["date"]):
                self.assertEqual(digest(payload.encode("utf-8")),
                                 review["manifest"]["after_sha256"])
                for change in review["changes"]:
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
                self.assertEqual(manifest["version"],
                                 review["manifest_version"])
        latest = reviews[-1]
        for change in latest["changes"]:
            self.assertEqual(pinned[change["id"]], change["after_sha256"])

    def test_fixtures_and_protocol_retain_the_reviewed_bytes(self):
        latest = self.reviews()[-1]
        # The runtime dataset ships only the fixtures its subset references;
        # test_catalog_evals.py checks those are byte-identical copies.
        for name, expected in latest["fixtures_sha256"].items():
            with self.subTest(fixture=name, root=EVALS):
                self.assertEqual(digest((EVALS / name).read_bytes()), expected)
            if (CATALOG / name).exists():
                with self.subTest(fixture=name, root=CATALOG):
                    self.assertEqual(digest((CATALOG / name).read_bytes()),
                                     expected)
        for name, root in (("development", EVALS), ("runtime", CATALOG)):
            with self.subTest(protocol=name):
                self.assertEqual(digest((root / "config.yml").read_bytes()),
                                 latest["protocol_sha256"][name])


if __name__ == "__main__":
    unittest.main()
