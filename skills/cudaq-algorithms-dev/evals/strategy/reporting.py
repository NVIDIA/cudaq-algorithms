#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline current-contract reporting from collected evidence and saved grades.

No workers, checks, judges or credential reads run here. Explicit grade directories
must match fresh payloads under an existing blinding mapping. Historical grades
from different judge contracts are rejected, never silently migrated. File
consistency is not proof of freeze chronology or scientific judgment validity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import judging
import posthoc
import suite

FAILED_STATUSES = {
    "transport_error", "invalid_grade", "client_error", "evidence_oversized",
    "invalid_token_bound"
}


def _canonical(value):
    # Unlike Python container equality, this distinguishes true from 1.
    return json.dumps(value,
                      sort_keys=True,
                      ensure_ascii=False,
                      allow_nan=False)


def _verify_request(record, result, payload):
    if (record.get("schema_version") != 1 or result.get("schema_version") != 1
            or record.get("provider") != judging.PROVIDER
            or result.get("provider") != judging.PROVIDER
            or result.get("model") != judging.MODEL
            or _canonical(record.get("payload")) != _canonical(payload)):
        raise ValueError(
            "saved grade does not match the current payload/provider")
    presentation = record.get("evidence_presentation")
    structured = record.get("structured_output")
    if (type(structured) is not bool or presentation
            not in ("original_lines_v1", "numbered_blocks_8_v1")
            or record.get("budget_mode")
            not in ("utf8_bytes", "input_token_bound") or any(
                _canonical(record.get(name)) != _canonical(result.get(name))
                for name in ("evidence_presentation", "budget_mode",
                             "input_token_bound"))):
        raise ValueError("unsupported or inconsistent saved request metadata")
    wire = deepcopy(payload)
    if presentation == "numbered_blocks_8_v1":
        for item in wire["evidence"]:
            lines = item.pop("lines")
            if any("\n" in line or "\r" in line for line in lines):
                raise ValueError(
                    "numbered evidence requires single-line entries")
            item["line_count"] = len(lines)
            item["numbered_text"] = "\n".join(
                f"[lines {start+1}-{min(start+8, len(lines))}]\n" +
                "\n".join(lines[start:start + 8])
                for start in range(0, len(lines), 8))
    expected = {
        "model":
        judging.MODEL,
        "messages": [{
            "role": "system",
            "content": judging.JUDGE_SYSTEM
        }, {
            "role":
            "user",
            "content":
            json.dumps(wire, ensure_ascii=False, allow_nan=False)
        }],
        "max_tokens":
        judging.MAX_OUTPUT_TOKENS,
        "temperature":
        0,
        "timeout":
        300,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    }
    if structured:
        expected["response_format"] = judging.grade_response_format(payload)
    request = record.get("request")
    if (_canonical(request) != _canonical(expected)
            or result.get("request_sha256") != hashlib.sha256(
                _canonical(request).encode()).hexdigest()):
        raise ValueError(
            "wire request differs from the supported current judge contract")
    if any(
            type(record.get(name)) is not int or record[name] != value
            for name, value in (("request_byte_ceiling",
                                 judging.MAX_REQUEST_BYTES),
                                ("context_token_capacity",
                                 64000), ("output_token_limit",
                                          judging.MAX_OUTPUT_TOKENS))):
        raise ValueError("unsupported saved request capacity")
    bound, mode = record.get("input_token_bound"), record["budget_mode"]
    if ((bound is not None and judging._count(bound) is None)
            or mode == "utf8_bytes" and bound is not None):
        raise ValueError("invalid recorded input-token bound")
    blocked = None
    if mode == "input_token_bound" and bound is None:
        blocked = "invalid_token_bound"
    elif (mode == "utf8_bytes"
          and len(json.dumps(request, ensure_ascii=False).encode())
          > judging.MAX_REQUEST_BYTES
          or bound is not None and bound + judging.MAX_OUTPUT_TOKENS > 64000):
        blocked = "evidence_oversized"
    if (blocked is not None and result.get("status") != blocked
            or blocked is None and result.get("status")
            in ("invalid_token_bound", "evidence_oversized")):
        raise ValueError("saved status contradicts request admission")


def _merge_grade(directory, rows, payloads, hashes, seen):
    result = posthoc._json(directory / "result.json", hashes)
    key = result.get("attempt_key")
    if not isinstance(key, str) or key not in payloads or key in seen:
        raise ValueError("duplicate, foreign, or non-execution grade key")
    seen.add(key)
    request = posthoc._json(directory / "request.json", hashes)
    _verify_request(request, result, payloads[key])
    if not isinstance(result.get("judge_usage"), dict):
        raise ValueError("invalid retained judge-cost container")
    row = rows[key]
    status = result.get("status")
    if status == "graded":
        response = posthoc._json(directory / "response.json", hashes)
        try:
            if response.get("model") not in (None, judging.MODEL):
                raise ValueError("unexpected response model")
            choice = response["choices"][0]
            if not isinstance(choice, dict) or not isinstance(
                    choice.get("message"), dict):
                raise ValueError("invalid saved response shape")
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete judge response")
            raw = judging._strict_json(choice["message"]["content"])
            validated = judging.validate_grade(
                raw,
                payloads[key],
                execution_status=row["execution_status"],
                checks=row["independent_checks"])
        except (KeyError, TypeError, IndexError, RecursionError) as error:
            raise ValueError("invalid saved judge response") from error
        if _canonical(validated) != _canonical(result.get("grade")):
            raise ValueError(
                "saved grade disagrees with fresh evidence/checks/status")
        row["grade"] = validated
    elif status not in FAILED_STATUSES or result.get("grade") is not None:
        raise ValueError("unsupported result or failed judge carrying a grade")
    else:
        # Retain/hash a failed reply if present; never turn it into a success.
        try:
            posthoc._read(directory / "response.json", hashes)
        except FileNotFoundError:
            pass
    row["judge"] = result


def _recheck(hashes):
    for path, expected in hashes.items():
        if hashlib.sha256(posthoc._read(path, {})).hexdigest() != expected:
            raise ValueError("an observed input changed while reporting")


def report(run_dir,
           *,
           mapping_path,
           grade_dirs=(),
           checks_dir=None,
           capture_dirs=None,
           tool_projection="full"):
    """Validate selected saved grades; preserve all manifest attempts/unknowns.

    The manifest supplies both purpose and seed. No external rows, KPI values,
    seed override, inference-policy switch, or implicit directory glob is used.
    Judge costs remain retained controller measurements, not worker efficiency.
    """
    if mapping_path is None:
        raise ValueError("a saved blinding mapping is required")
    hashes = {}
    here = Path(__file__).absolute().parent
    for name in ("reporting.py", "posthoc.py", "judging.py", "confidence.py",
                 "suite.py", "numerical.py", "capture.py", "BRIEF.md"):
        posthoc._read(here / name, hashes)
    collected = posthoc.collect(run_dir,
                                mapping_path=mapping_path,
                                checks_dir=checks_dir,
                                capture_dirs=capture_dirs,
                                tool_projection=tool_projection)
    freeze = collected["manifest_freeze"]
    if freeze["status"] != "verified":
        raise ValueError("a verified local manifest freeze is required")
    hashes.update(collected["input_hashes"])
    manifest_path = Path(run_dir).absolute() / "manifest.json"
    purpose_hashes = {}
    manifest = posthoc._json(manifest_path, purpose_hashes)
    if purpose_hashes[str(manifest_path)] != freeze["manifest_sha256"]:
        raise ValueError("manifest changed after collection")
    purpose = manifest.get("purpose")
    if purpose not in ("readiness-pilot", "disclosed-study"):
        raise ValueError("unsupported manifest purpose")
    expected_schedule = ([[a.case_id, a.arm, a.repetition]
                          for a in suite.build_schedule(suite.load_cases())]
                         if purpose == "disclosed-study" else
                         [[case, arm, 1]
                          for case in ("T02", "N07", "E01", "E03")
                          for arm in ("baseline", "candidate")])
    observed_schedule = [[a["case_id"], a["arm"], a["repetition"]]
                         for a in manifest["attempts"]]
    if _canonical(observed_schedule) != _canonical(expected_schedule):
        raise ValueError(
            "manifest schedule does not match its declared purpose")
    rows = collected["rows"]
    by_key = {r["attempt_key"]: r for r in rows}
    payloads = {p["attempt_key"]: p for p in collected["payloads"]}
    seen = set()
    for directory in grade_dirs:
        _merge_grade(
            Path(directory).absolute(), by_key, payloads, hashes, seen)
    summary = judging.summarize(rows,
                                pilot=purpose == "readiness-pilot",
                                manifest_sha256=freeze["manifest_sha256"])
    _recheck(hashes)
    return {
        "rows": rows,
        "summary": summary,
        "provenance": {
            "schema_version":
            1,
            "manifest_freeze":
            freeze,
            "purpose":
            purpose,
            "input_hashes":
            hashes,
            "tool_projection":
            collected["tool_projection"],
            "bound_judge_attempts":
            sorted(seen),
            "validated_grade_attempts":
            sorted(r["attempt_key"] for r in rows if r["grade"] is not None),
            "judge_cost_provenance":
            "Retained controller measurements; not independently remeasured or authenticated.",
            "limitations":
            collected["limitations"] + [
                "Local artifact consistency does not establish pre-observation freeze chronology or scientific judge reliability.",
                "Tokenizer assets and input-token bounds are retained metadata, not independently verified here.",
                "Missing inputs/grades remain snapshot unknowns; no worker/check/judge was rerun.",
                "Diagnostic confidence remains uncalibrated and cannot establish statistical acceptance."
            ]
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--grade-dir", type=Path, action="append", default=[])
    parser.add_argument("--checks-dir", type=Path)
    parser.add_argument("--capture-dirs", type=Path)
    parser.add_argument("--tool-projection",
                        choices=tuple(posthoc.TOOL_POLICIES),
                        default="full")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(args.output_dir)
        override_hashes = {}
        overrides = posthoc._json(
            args.capture_dirs, override_hashes) if args.capture_dirs else None
        value = report(args.run_dir,
                       mapping_path=args.mapping,
                       grade_dirs=args.grade_dir,
                       checks_dir=args.checks_dir,
                       capture_dirs=overrides,
                       tool_projection=args.tool_projection)
        value["provenance"]["input_hashes"].update(override_hashes)
        _recheck(value["provenance"]["input_hashes"])
        args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        outputs = {}
        for name in ("rows", "summary", "provenance"):
            path = args.output_dir / (name + ".json")
            judging._private_json(path, value[name])
            outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        judging._private_json(args.output_dir / "hashes.json", {
            "inputs": value["provenance"]["input_hashes"],
            "outputs": outputs
        })
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        print(
            json.dumps({
                "status": "report_failed",
                "reason": "Inspect private inputs; output must be new."
            }))
        return 1
    print(
        json.dumps({
            "status": "reported",
            "attempts": len(value["rows"]),
            "statistical_pass": None
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
