#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check skill links, feature mappings and scoped run evidence without models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


def inside(root, name, scope="skill"):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"outside {scope} root: {name}")
    return path


def inside_declared_roots(runtime_root, support_root, name):
    path = Path(name).resolve()
    if not (path.is_relative_to(runtime_root)
            or path.is_relative_to(support_root)):
        raise ValueError(f"outside declared roots: {name}")
    return path


def display_path(path, runtime_root, support_root):
    if runtime_root == support_root:
        return path.relative_to(runtime_root)
    if path.is_relative_to(runtime_root):
        return Path("runtime") / path.relative_to(runtime_root)
    return Path("support") / path.relative_to(support_root)


def read_json(path):
    return json.loads(path.read_text())


def behavioral_eval_ids(registry, support_root, support_scope):
    suites = registry.get("behavioral_eval_suites", ["evals/evals.json"])
    if (not isinstance(suites, list) or not suites
            or any(not isinstance(name, str) or not name.strip()
                   for name in suites)):
        raise ValueError(
            "behavioral_eval_suites must be a nonempty list of paths")
    known = set()
    for name in suites:
        path = inside(support_root, name, support_scope)
        if not path.is_file():
            raise ValueError(f"missing behavioral eval suite: {name}")
        for case in read_json(path)["evals"]:
            identity = case["id"]
            if identity in known:
                raise ValueError(f"duplicate behavioral eval id: {identity}")
            known.add(identity)
    return known


def markdown_links(path):
    """Ignore example fences; return local Markdown targets only."""
    fenced = False
    for line in path.read_text().splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
            continue
        if not fenced:
            for target in re.findall(r"\]\(([^)]+)\)", line):
                target = target.strip().strip("<>")
                if not urlsplit(target).scheme and not target.startswith("#"):
                    yield unquote(target.split("#", 1)[0])


def scientific_pass(record):
    checks = record.get("grading", {}).get("checks", [])
    return (record.get("attempted") is True and len(checks) == 2
            and {check.get("variant")
                 for check in checks} == {0, 1} and all(
                     check.get("returncode") == 0 and all(
                         check.get(kind, {}).get("passed") is True
                         for kind in ("numeric", "host_api", "device_api"))
                     for check in checks))


def verify_execution(runtime_root, support_root, feature, evidence, campaigns):
    """Return whether this is scientific evidence for this exact record text."""
    runtime_scope = "skill" if runtime_root == support_root else "runtime"
    support_scope = "skill" if runtime_root == support_root else "support"
    manifest_path, manifest = campaigns[evidence["campaign"]]
    case_id, arm = evidence["case"], evidence["arm"]
    specs = {case["id"]: case for case in manifest["cases"]}
    if case_id not in specs:
        raise ValueError(f"unknown execution case: {case_id}")
    required = specs[case_id].get("required_public_apis", [])
    if not set(feature["symbols"]).intersection(required):
        raise ValueError(
            f"no required API matches feature {feature['id']}: {case_id}")
    if not isinstance(evidence.get("scope"),
                      str) or not evidence["scope"].strip():
        raise ValueError(f"missing evidence scope: {feature['id']}/{case_id}")
    expected = evidence["repetitions"]
    if type(expected) is not int or expected < 1:
        raise ValueError("repetition count must be a positive integer")
    planned = [
        run for run in manifest["runs"]
        if run["case"] == case_id and run["arm"] == arm
    ]
    if len(planned) != expected or {run["repetition"]
                                    for run in planned} != set(
                                        range(1, expected + 1)):
        raise ValueError(
            f"repetition count mismatch: {feature['id']}/{case_id}/{arm}")
    status = evidence["status"]
    if status not in ("scientific_pass", "executed", "blocked"):
        raise ValueError(f"unknown evidence status: {status}")
    for plan in planned:
        result_path = inside(
            support_root,
            str(manifest_path.parent / "runs" / plan["id"] / "result.json"),
            support_scope)
        result = read_json(result_path)
        if any(
                result.get(key) != plan[key]
                for key in ("id", "case", "arm", "repetition")):
            raise ValueError(f"run identity mismatch: {result_path}")
        if status == "scientific_pass" and not scientific_pass(result):
            raise ValueError(
                f"scientific evidence does not pass: {plan['id']}")
        if status == "executed" and result.get("attempted") is not True:
            raise ValueError(f"execution evidence is blocked: {plan['id']}")
        if status == "blocked" and not (result.get("attempted") is False
                                        and result.get("status")
                                        == "infrastructure_failure"):
            raise ValueError(f"blocked evidence was attempted: {plan['id']}")
    if status != "scientific_pass":
        return False, False
    if arm == "baseline":
        raise ValueError("baseline has no skill record evidence")
    key = "skills/cudaq-algorithms/" + feature["record"]
    inventory = (manifest["arms"].get(arm, {}).get("inventory", {})
                 if "arms" in manifest else manifest.get(
                     "source_inventory", {}) if arm == "skill" else {})
    recorded = inventory.get(key)
    if not recorded:
        raise ValueError(
            f"no recorded skill version for evidence: {feature['id']}/{arm}")
    current = hashlib.sha256(
        inside(runtime_root, feature["record"],
               runtime_scope).read_bytes()).hexdigest()
    return True, recorded == current


def check(root, support_root=None):
    root = Path(root).resolve()
    support_root = Path(
        support_root).resolve() if support_root is not None else root
    runtime_scope = "skill" if root == support_root else "runtime"
    support_scope = "skill" if root == support_root else "support"
    errors, scientific, current = [], set(), set()
    registry = read_json(support_root / "coverage/features.json")
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported coverage schema_version")
    features = registry["features"]
    historical = []
    if registry.get("historical_metadata_inventory"):
        historical = read_json(
            inside(support_root, registry["historical_metadata_inventory"],
                   support_scope))["removed_metadata"]
    behavioral = behavioral_eval_ids(registry, support_root, support_scope)
    campaigns = {}
    for campaign in registry.get("campaigns", []):
        try:
            path = inside(support_root, campaign["manifest"], support_scope)
            if campaign["id"] in campaigns:
                raise ValueError(f"duplicate campaign id: {campaign['id']}")
            campaigns[campaign["id"]] = path, read_json(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    ids, registered = set(), set()
    for feature in features:
        try:
            identity = feature["id"]
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("missing feature id")
            if identity in ids:
                raise ValueError(f"duplicate feature id: {identity}")
            ids.add(identity)
            record = inside(root, feature["record"], runtime_scope)
            if not record.is_file():
                raise ValueError(f"missing record: {feature['record']}")
            if record in registered:
                raise ValueError(
                    f"duplicate feature record: {feature['record']}")
            registered.add(record)
            parts = record.relative_to(root).parts
            if len(parts) != 3 or parts[0] != "references" or parts[
                    1] != feature.get("family"):
                raise ValueError(f"family does not match record: {identity}")
            symbols = feature.get("symbols")
            if not isinstance(symbols, list) or not symbols or any(
                    not isinstance(s, str) or not s.strip() for s in symbols):
                raise ValueError(
                    f"public symbols must be a nonempty list: {identity}")
            for index in feature.get("historical_metadata_indices", []):
                if type(index) is not int or not 0 <= index < len(historical):
                    raise ValueError(
                        f"historical metadata index out of range: {identity}/{index}"
                    )
                if historical[index]["source_record"] != feature["record"]:
                    raise ValueError(
                        f"historical metadata source mismatch: {identity}/{index}"
                    )
            for case in feature.get("behavioral_evals", []):
                if case not in behavioral:
                    errors.append(
                        f"unknown behavioral eval: {identity}/{case}")
            for evidence in feature.get("executions", []):
                try:
                    observed, fresh = verify_execution(root, support_root,
                                                       feature, evidence,
                                                       campaigns)
                    if observed:
                        scientific.add(identity)
                    if fresh:
                        current.add(identity)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{identity}: {exc}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    feature_records = registered.copy()
    for name in registry.get("support_records", []):
        try:
            path = inside(root, name, runtime_scope)
            if not path.is_file():
                errors.append(f"missing support record: {name}")
            if path in registered:
                errors.append(f"duplicate registered record: {name}")
            registered.add(path)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    operational = set((root / "references").rglob("*.md"))
    # Detailed family tables are the live source of independently selectable
    # contracts. A registry edit must not silently reclassify one as support.
    selectable = set()
    for document in operational:
        in_table = fenced = False
        for line in document.read_text().splitlines():
            if re.match(r"^\s*(```|~~~)", line):
                fenced = not fenced
            if fenced:
                continue
            if line.startswith("| Operation + object | Public entry point |"):
                in_table = True
                continue
            if in_table and not line.startswith("|"):
                in_table = False
            if in_table:
                matches = re.findall(r"\]\(([^)]+)\)", line)
                if matches:
                    try:
                        selectable.add(
                            inside(
                                root,
                                str(document.parent /
                                    matches[-1].split("#", 1)[0]),
                                runtime_scope))
                    except ValueError as exc:
                        errors.append(str(exc))
    for path in sorted(selectable - feature_records):
        errors.append(
            f"selectable catalog contract is not a feature: {path.relative_to(root)}"
        )
    if selectable:
        for path in sorted(feature_records - selectable):
            errors.append(
                f"feature missing from selectable catalog: {path.relative_to(root)}"
            )
    for path in sorted(operational - registered):
        errors.append(
            f"unregistered operational record: {path.relative_to(root)}")
    documents = {root / "SKILL.md", *operational}
    for directory in ("authoring", "coverage"):
        documents.update((support_root / directory).rglob("*.md"))
    documents.update(path for path in (support_root / "evals/EVAL.md",
                                       support_root / "evals/e2e/README.md")
                     if path.exists())
    graph = {}
    for document in sorted(documents):
        edges = set()
        if not document.is_file():
            errors.append(
                f"missing document: {display_path(document, root, support_root)}"
            )
            continue
        for link in markdown_links(document):
            try:
                target = inside_declared_roots(root, support_root,
                                               document.parent / link)
                if not target.exists():
                    errors.append(
                        f"broken link: {display_path(document, root, support_root)} -> {link}"
                    )
                edges.add(target)
            except ValueError as exc:
                errors.append(
                    f"{display_path(document, root, support_root)}: {exc}")
        graph[document] = edges
    seen, pending = set(), [root / "SKILL.md"]
    while pending:
        path = pending.pop()
        if path not in seen:
            seen.add(path)
            pending.extend((graph.get(path, set()) & operational) - seen)
    for path in sorted(operational - seen):
        errors.append(
            f"unreachable operational record: {path.relative_to(root)}")
    return {
        "passed": not errors,
        "features": len(features),
        "support_records": len(registry.get("support_records", [])),
        "features_with_scientific_evidence": len(scientific),
        "features_with_current_record_evidence": len(current),
        "errors": errors
    }, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    support_default = Path(__file__).resolve().parents[1]
    runtime_default = support_default.parent / "cudaq-algorithms"
    parser.add_argument(
        "--root",
        type=Path,
        help=
        "runtime skill root (an explicit root also supplies combined-layout support)"
    )
    parser.add_argument("--support-root",
                        type=Path,
                        help="maintainer coverage/evaluation root")
    parser.add_argument("--json",
                        action="store_true",
                        help="emit machine-readable diagnostics")
    parser.add_argument(
        "--feature", help="show one feature entry after checking consistency")
    args = parser.parse_args()
    if args.root is None:
        runtime_root = runtime_default
        support_root = args.support_root or support_default
    else:
        runtime_root = args.root
        support_root = args.support_root or args.root
    try:
        report, registry = check(runtime_root, support_root)
        if args.feature:
            matches = [
                feature for feature in registry["features"]
                if feature["id"] == args.feature
            ]
            if not matches:
                raise ValueError(f"unknown feature: {args.feature}")
            if report["passed"]:
                print(json.dumps(matches[0], indent=2))
                return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"passed": False, "errors": [str(exc)]}
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["passed"]:
        print(
            f"Coverage consistency PASS: {report['features']} features; "
            f"{report['features_with_scientific_evidence']} have scoped scientific evidence, "
            f"{report['features_with_current_record_evidence']} match current record text."
        )
    else:
        for error in report["errors"]:
            print("ERROR: " + error, file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
