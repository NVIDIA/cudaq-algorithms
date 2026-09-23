#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble private pilot evidence without running workers, checks, or judges.

collect(run_dir) reads only, returning rows, execution payloads, complete
projected evidence, checks_mapping, blinding, input_hashes, and manifest_freeze.
The last field distinguishes a matching local freeze record from no record;
it does not prove pre-observation chronology or certify confidence. Pass a previous
blinding.json as mapping_path to reuse opaque keys and the randomized order.
The CLI requires a NEW private output directory and writes those artifacts and
hashes. Original trials, captures, sessions, and check logs remain untouched.

Capture overrides map manifest attempt_id to a recovered capture directory.
Checks default to <run>/posthoc/<attempt_id>/{targeted,regression}/result.json
and pytest.log. Registered coverage comes from numerical.CHECK_SUITES; E02,
E05 and E08 targeted results additionally require frozen explicit bindings.
An artifact_checks.json recovery record may select the capture directory.
--tool-projection ledger-v1 preserves every call/argument, replacing full tool
result bodies with byte/hash/status descriptors. Full projections remain in
evidence/. Final responses, repository diffs and independent test logs remain
complete in either policy; over-limit requests are explicitly unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import capture
import judging
import numerical
import suite

MAX_ARTIFACT_BYTES = 256 * 1024**2
MAX_TRANSCRIPT_BYTES = 256 * 1024**2
TOOL_POLICIES = {
    "full": {
        "name": "full",
        "limitation": "Full normalized tool result bodies are included."
    },
    "ledger-v1": {
        "name":
        "ledger-v1",
        "scope":
        "All normalized calls and complete arguments, in original order.",
        "result_projection":
        "Each observation/wrapper has UTF-8 byte count, SHA-256, observed exit code and native success interpretation.",
        "limitation":
        "The judge cannot access full result bodies; these remain in private evidence and original logs. Unknown status is not success.",
        "unchanged_channels":
        ["final", "diff", "independent test summary and logs"]
    },
}


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, hashes, limit=MAX_ARTIFACT_BYTES):
    path = Path(path).absolute()
    directory = capture._directory(path.parent)
    try:
        raw = capture._read(directory, path.name, limit)
    finally:
        os.close(directory)
    hashes[str(path)] = _digest(raw)
    return raw


def _json(path, hashes):
    value = judging._strict_json(_read(path, hashes).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _optional_json(path, hashes):
    try:
        return _json(path, hashes), "complete"
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError, UnicodeError):
        return None, "invalid"


def _relative(root, name):
    return root.joinpath(*capture._relative_parts(name))


def _mapping(manifest, manifest_hash, mapping_path, hashes):
    attempts = manifest["attempts"]
    identifiers = [a["attempt_id"] for a in attempts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate manifest attempt IDs")
    execution = {
        a["attempt_id"]
        for a in attempts if suite.case_by_id(a["case_id"]).kind == "execution"
    }
    if mapping_path is not None:
        result = _json(mapping_path, hashes)
        keys = result.get("attempt_keys", {})
        order = result.get("grading_order", [])
        if (result.get("manifest_sha256") != manifest_hash
                or set(keys) != set(identifiers)
                or any(not isinstance(k, str)
                       or not re.fullmatch(r"[0-9a-f]{32}", k)
                       for k in keys.values())
                or len(set(keys.values())) != len(keys)
                or not isinstance(order, list) or len(order) != len(execution)
                or set(order) != {keys[a]
                                  for a in execution}):
            raise ValueError("blinding mapping does not match this manifest")
        return result
    keys = {identifier: uuid.uuid4().hex for identifier in identifiers}
    order = judging.grading_order([{
        "attempt_key": keys[a]
    } for a in identifiers if a in execution])
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "attempt_keys": keys,
        "grading_order": [item["attempt_key"] for item in order]
    }


def _capture(directory, source, hashes):
    manifest, status = _optional_json(directory / "capture.json", hashes)
    result = {"status": status, "manifest": manifest, "diff": None}
    if manifest is None:
        return result
    try:
        if (manifest.get("schema_version") != 1
                or manifest.get("status") != "complete"
                or manifest.get("rejections") != []
                or manifest.get("unverified_deletions") != []):
            result["status"] = "incomplete"
            return result
        old = capture._validated_inventory(manifest.get("source_inventory"))
        new = capture._validated_inventory(manifest.get("worktree_inventory"))
        original = capture._snapshot(
            source, "source", {
                "max_file_bytes": capture.MAX_FILE_BYTES,
                "max_tree_bytes": capture.MAX_TREE_BYTES,
                "max_entries": capture.MAX_ENTRIES,
                "max_judge_bytes": capture.MAX_JUDGE_BYTES
            })
        if (original["rejections"] or original["inventory"] != old or _digest(
                capture._json(old)) != manifest.get("source_inventory_sha256")
                or _digest(capture._json(new))
                != manifest.get("worktree_inventory_sha256")):
            raise ValueError("source or captured inventory mismatch")
        changed = sorted(name for name in new if old.get(name) != new[name])
        if (manifest.get("changed_paths") != changed
                or manifest.get("added_paths") != sorted(set(new) - set(old))
                or manifest.get("deleted_paths")
                != sorted(set(old) - set(new))):
            raise ValueError("capture change lists mismatch")
        expected = {"files/" + name
                    for name in changed
                    } | {"changes.diff", "judge_evidence.json"}
        artifacts = manifest.get("artifacts")
        if (not isinstance(artifacts, list) or len(artifacts) != len(expected)
                or any(not isinstance(a, dict) for a in artifacts)
                or {a.get("path")
                    for a in artifacts} != expected):
            raise ValueError("capture artifact inventory mismatch")
        for artifact in artifacts:
            name = artifact["path"]
            raw = _read(_relative(directory, name), hashes)
            if len(raw) != artifact.get("size_bytes") or _digest(
                    raw) != artifact.get("sha256"):
                raise ValueError("capture artifact hash mismatch")
            if name.startswith("files/"):
                entry = new[name.removeprefix("files/")]
                if len(raw) != entry["size_bytes"] or _digest(
                        raw) != entry["sha256"]:
                    raise ValueError("captured blob mismatch")
            elif name == "changes.diff":
                result["diff"] = raw.decode("utf-8")
            else:
                projection = judging._strict_json(raw.decode("utf-8"))
                if projection.get("status") == "complete" and projection.get(
                        "diff") != result["diff"]:
                    # Artifact order is not semantically meaningful; check below.
                    result["projection_diff"] = projection.get("diff")
        if "projection_diff" in result and result.pop(
                "projection_diff") != result["diff"]:
            raise ValueError("capture judge projection mismatch")
        result["status"] = "complete"
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        result.update(status="invalid", diff=None)
    return result


def _check(directory, kind, *, case_id, applicable, captured, hashes,
           artifact_record):
    result, result_status = _optional_json(directory / "result.json", hashes)
    log, log_status = None, "missing"
    try:
        raw = _read(directory / "pytest.log", hashes)
        log = raw.decode("utf-8", errors="replace")
        log_status = "complete" if result and result.get(
            "output_sha256") == _digest(raw) and result.get(
                "output_bytes") == len(raw) else "invalid"
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        log_status = "invalid"
    verdict = "unknown" if applicable else "not_applicable"
    recorded_hashes = artifact_record.get("hashes", {}) if isinstance(
        artifact_record, dict) else {}
    if not isinstance(recorded_hashes, dict):
        recorded_hashes = {}
    check_suite = numerical.CHECK_SUITES.get(case_id)
    bound = (check_suite is not None and captured
             and result_status == "complete" and log_status == "complete"
             and recorded_hashes.get(kind + "_result") == hashes.get(
                 str(directory / "result.json"))
             and recorded_hashes.get(kind + "_log") == hashes.get(
                 str(directory / "pytest.log")))
    binding = "not_applicable"
    inputs_status = "not_applicable"
    if case_id in numerical.BOUND_CHECKS and kind == "targeted":
        prefix = case_id.lower()
        freeze_path = directory / f"{prefix}-inputs.json"
        frozen, inputs_status = _optional_json(freeze_path, hashes)
        digest = recorded_hashes.get(prefix + "_binding")
        reported = result.get(prefix +
                              "_inputs") if isinstance(result, dict) else None
        freeze_hash = hashes.get(str(freeze_path))
        source_files = recorded_hashes.get("source_files")
        configured_consumers = numerical.BOUND_CHECKS[case_id][2]
        expected_consumers = ({
            name: source_files.get(name)
            for name in configured_consumers
        } if isinstance(source_files, dict) else None)
        staged_binding_bound = True
        # E02's established evidence wire did not retain staged binding bytes.
        # Newer bound suites require those bytes to remain safe and unchanged.
        if case_id != "E02":
            try:
                staged_binding_bound = (_digest(
                    _read(directory / f"{prefix}-input/binding.json", hashes,
                          8192)) == digest)
            except (OSError, ValueError):
                staged_binding_bound = False
        inputs_bound = (
            inputs_status == "complete" and isinstance(reported, dict)
            and reported.get("unchanged") is True
            and reported.get("manifest_sha256") == freeze_hash
            and recorded_hashes.get(prefix + "_inputs_manifest") == freeze_hash
            and all(
                reported.get(key) == value for key, value in frozen.items())
            and frozen.get("binding_sha256") == digest
            and frozen.get("captured_inventory_sha256")
            == recorded_hashes.get(prefix + "_captured_inventory")
            and frozen.get("checks_inventory_sha256")
            == recorded_hashes.get("checks_inventory") and
            frozen.get("checks_files") == recorded_hashes.get("checks_files")
            and frozen.get("consumer_files") == expected_consumers
            and expected_consumers is not None
            and all(value is not None for value in expected_consumers.values())
            and staged_binding_bound)
        binding = ("verified" if isinstance(digest, str) and re.fullmatch(
            r"[0-9a-f]{64}", digest) and result is not None
                   and result.get("binding_sha256") == digest and inputs_bound
                   else "unknown")
        bound = bound and binding == "verified"
    if (applicable and bound and result.get("kind") == kind
            and result.get("case_id") == case_id):
        count, skips = check_suite[kind]
        verdict = numerical.judge_check(result,
                                        expected_collected=count,
                                        allowed_skips=skips)
        if artifact_record.get(kind) != verdict:
            verdict = "unknown"
    provenance = {"capture_binding": "verified" if bound else "unknown"}
    binding_evidence = f"capture binding={provenance['capture_binding']}"
    if case_id in suite.CONDITIONAL_OUTCOME_IDS:
        # A verified capture can establish no implementation without numerical
        # results. Distinguish the two bindings in this case's judge evidence.
        provenance = {
            "capture_binding":
            "verified" if captured else "unknown",
            "result_binding": ("not_applicable" if not applicable else
                               "verified" if bound else "unknown")
        }
        binding_evidence = (
            f"capture binding={provenance['capture_binding']}; "
            f"numerical result binding={provenance['result_binding']}")
    evidence = [
        f"Independent {kind}: {verdict}; result={result_status}; log={log_status}; {binding_evidence}."
    ]
    if result is not None:
        # Keep the scientific checker summary complete; paths are blinded later.
        evidence.append(json.dumps(result, sort_keys=True, ensure_ascii=False))
    if log is not None:
        evidence.extend(log.splitlines())
    return verdict, {
        "result_status": result_status,
        "log_status": log_status,
        **provenance, "binding": binding,
        "inputs_status": inputs_status,
        "result": result
    }, evidence


def _capture_binding(record, captured, *, identity, case_id, manifest_hash,
                     result_hash, capture_hash):
    if not isinstance(record, dict) or captured["status"] != "complete":
        return False
    reconstruction, hashes = record.get("reconstruction"), record.get("hashes")
    if not isinstance(reconstruction, dict) or not isinstance(hashes, dict):
        return False
    tree_hash = captured["manifest"]["worktree_inventory_sha256"]
    # Capture provenance is independent of numerical-suite registration.
    # _check separately requires a registered suite before accepting test results.
    return (record.get("attempt_id") == identity
            and record.get("case_id") == case_id
            and record.get("capture_status") == "complete"
            and record.get("reconstruction_status") == "complete"
            and reconstruction.get("status") == "complete"
            and reconstruction.get("capture_json_sha256") == capture_hash
            and reconstruction.get("worktree_inventory_sha256") == tree_hash
            and hashes.get("capture_manifest") == capture_hash
            and hashes.get("worktree_inventory") == tree_hash
            and hashes.get("manifest") == manifest_hash
            and hashes.get("trial_result") == result_hash
            and capture_hash is not None and result_hash is not None)


def _request_bytes(payload):
    # Mirror the frozen grade() serialization, including the system/rubric cost.
    kwargs = {
        "model":
        judging.MODEL,
        "messages": [{
            "role": "system",
            "content": judging.JUDGE_SYSTEM
        }, {
            "role":
            "user",
            "content":
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
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
    return len(json.dumps(kwargs, ensure_ascii=False).encode())


def _ledger(lines):
    projected = []
    for line in lines:
        label, encoded = line.split(": ", 1)
        call = json.loads(encoded)
        entry = dict(call)
        for field in ("observation", "wrapper_observation"):
            if field not in call:
                continue
            value = call[field]
            text = value if isinstance(value, str) else json.dumps(
                value, sort_keys=True, ensure_ascii=False)
            structured = value
            if isinstance(value, str):
                try:
                    structured = json.loads(value)
                except ValueError:
                    structured = None
            code, is_error = None, None
            if isinstance(structured, dict):
                metadata = structured.get("metadata")
                code = structured.get(
                    "exit_code",
                    metadata.get("exit_code")
                    if isinstance(metadata, dict) else None)
                is_error = structured.get("isError") if isinstance(
                    structured.get("isError"), bool) else None
            if code is None:
                match = re.search(
                    r"(?i)(?:Process exited with code|exit(?:_code| code))\s*[:=]?\s*(-?\d+)",
                    text)
                code = int(match[1]) if match else None
            entry[field] = {
                "utf8_bytes":
                len(text.encode()),
                "sha256":
                _digest(text.encode()),
                "exit_code":
                code,
                "is_error":
                is_error,
                "success":
                judging._successful_observation({
                    **call, "observation": value
                })
            }
        projected.append(label + ": " +
                         json.dumps(entry, sort_keys=True, ensure_ascii=False))
    return projected


def collect(run_dir,
            *,
            mapping_path=None,
            capture_dirs=None,
            checks_dir=None,
            tool_projection="full"):
    """Read all manifest attempts, preserving unknowns and complete evidence.

    No model/API calls, worker execution, reconstruction, or source writes.
    Save and reuse returned blinding to keep keys/order stable across snapshots.
    """
    root = Path(run_dir).absolute()
    if tool_projection not in TOOL_POLICIES:
        raise ValueError("unsupported tool projection")
    hashes = {}
    manifest = _json(root / "manifest.json", hashes)
    manifest_hash = hashes[str(root / "manifest.json")]
    frozen, frozen_status = _optional_json(root / "manifest.sha256.json",
                                           hashes)
    if (frozen_status == "invalid" or frozen_status == "complete"
            and frozen.get("sha256") != manifest_hash):
        raise ValueError("manifest hash mismatch")
    manifest_freeze = {
        "status": "verified" if frozen_status == "complete" else "missing",
        "manifest_sha256": manifest_hash,
        "record_sha256": hashes.get(str(root / "manifest.sha256.json"))
    }
    blinding = _mapping(manifest, manifest_hash, mapping_path, hashes)
    checks_root = Path(
        checks_dir).absolute() if checks_dir else root / "posthoc"
    rows, payloads, checks_mapping, all_evidence = [], {}, {}, {}
    for attempt in manifest["attempts"]:
        identity = attempt["attempt_id"]
        key = blinding["attempt_keys"][identity]
        case = suite.case_by_id(attempt["case_id"])
        job = _relative(root / "jobs", attempt["job_name"])
        trials = sorted(
            p for p in job.iterdir()
            if p.is_dir() and p.name.startswith(identity +
                                                "__")) if job.is_dir() else []
        trial = trials[0] if len(trials) == 1 else None
        result, result_status = _optional_json(
            trial / "result.json", hashes) if trial else (None, "missing")
        if result and (result.get("task_name", identity)
                       not in (identity, "nvidia/skillevaluator-" + identity)
                       or result.get("trial_name", trial.name) != trial.name):
            result, result_status = None, "invalid"
        trajectory, trajectory_status = _optional_json(
            trial / "agent/trajectory.json", hashes) if trial else (None,
                                                                    "missing")
        sessions = sorted(
            (trial / "agent/sessions").rglob("*.jsonl")) if trial else []
        safe_sessions = []
        for session in sessions:
            try:
                _read(session, hashes)
                safe_sessions.append(session)
            except (OSError, ValueError):
                pass
        unsafe_sessions = len(sessions) - len(safe_sessions)
        measured = judging.usage(
            safe_sessions,
            trajectory=trajectory,
            result=result,
            trace_complete=False if unsafe_sessions else None)
        transcript = judging.transcript_evidence(
            safe_sessions,
            trajectory=trajectory,
            max_bytes=MAX_TRANSCRIPT_BYTES)
        if unsafe_sessions:
            measured["unreadable_sessions"] += unsafe_sessions
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                if measured[name] is not None:
                    measured["observed_" + name] = measured[name]
                    measured[name] = None
            measured["usage_coverage_status"] = "partial" if measured[
                "observed_total_tokens"] is not None else "missing"
            if measured["peak_request_input_tokens_proxy"] is not None:
                measured["peak_coverage_status"] = "partial"
            if transcript["status"] != "oversized":
                transcript["status"] = "incomplete"
        failure = "ambiguous_trials" if len(
            trials
        ) > 1 else "missing_result" if result_status == "missing" else "invalid_result" if result_status == "invalid" else None
        if result and result.get("exception_info"):
            exception = result["exception_info"]
            category = exception.get("exception_type", exception.get(
                "type", "")) if isinstance(exception, dict) else ""
            failure = "timeout" if "timeout" in category.casefold(
            ) else "worker_exception"
        source = _relative(
            root, attempt["dataset_dir"]) / identity / "environment/repo"
        artifact_record, artifact_record_status = _optional_json(
            _relative(checks_root, identity) / "artifact_checks.json", hashes)
        override = (capture_dirs or {}).get(identity) or (
            artifact_record or {}).get("capture_selected_path")
        capture_dir = Path(override).absolute(
        ) if override else trial / "verifier/artifacts/capture" if trial else None
        captured = _capture(capture_dir, source, hashes) if capture_dir else {
            "status": "missing",
            "manifest": None,
            "diff": None
        }
        capture_status = "complete" if captured[
            "status"] == "complete" else "incomplete" if captured[
                "status"] == "incomplete" else "unknown"
        binding = result_status == "complete" and _capture_binding(
            artifact_record,
            captured,
            identity=identity,
            case_id=case.case_id,
            manifest_hash=manifest_hash,
            result_hash=hashes.get(str(trial /
                                       "result.json")) if trial else None,
            capture_hash=hashes.get(str(capture_dir / "capture.json"))
            if capture_dir else None)
        checks = {"capture": capture_status}
        if case.conditional_outcome:
            checks["unchanged_capture"] = "unknown"
            if binding:
                captured_manifest = captured["manifest"]
                checks["unchanged_capture"] = (
                    "verified" if captured_manifest["source_inventory"]
                    == captured_manifest["worktree_inventory"] else "changed")
        implementation_required = (case.mutation_request
                                   and checks.get("unchanged_capture")
                                   != "verified")
        tests = [
            f"Execution status: {measured['execution_status']}; failure category: {failure or 'none'}.",
            f"Transcript evidence: {transcript['status']}; trace: {measured['trace_status']}.",
            f"Repository capture: {captured['status']}."
        ]
        if case.conditional_outcome:
            tests.append(
                f"Unchanged capture: {checks['unchanged_capture']}; "
                f"implementation checks required: {implementation_required}.")
        check_records = {}
        for kind in ("targeted", "regression"):
            checks[kind], check_records[kind], lines = _check(
                _relative(checks_root, identity) / kind,
                kind,
                case_id=case.case_id,
                applicable=implementation_required,
                captured=binding,
                hashes=hashes,
                artifact_record=artifact_record)
            tests.extend(lines)
        evidence = dict(
            transcript.get("evidence") or {
                "final": [],
                "tools": []
            })
        evidence.update(tests=tests,
                        diff=(captured["diff"] or "").splitlines())
        prefixes = [
            str(root),
            str(checks_root), "/logs/agent/worktree", "/logs",
            "/workspace/project", "/workspace/repo"
        ]
        if capture_dir:
            prefixes.append(str(capture_dir))
        from skillevaluator.utils.redaction import redact_sensitive_text
        evidence = {
            channel: [
                redact_sensitive_text(judging._blind(line, prefixes))
                for line in lines
            ]
            for channel, lines in evidence.items()
        }
        all_evidence[key] = evidence
        payload_status, request_bytes = "not_applicable", None
        if case.kind == "execution" and transcript["status"] == "oversized":
            payload_status = "evidence_oversized"
        elif case.kind == "execution":
            try:
                judge_evidence = dict(evidence)
                if tool_projection == "ledger-v1":
                    judge_evidence["tools"] = _ledger(evidence["tools"])
                payload = judging.judge_payload(case.case_id,
                                                attempt_key=key,
                                                evidence=judge_evidence)
                payload["tool_projection"] = TOOL_POLICIES[tool_projection]
                request_bytes = _request_bytes(payload)
                payload_status = (
                    "evidence_oversized"
                    if request_bytes > judging.MAX_REQUEST_BYTES else
                    "execution_incomplete" if result_status != "complete"
                    or not (result or {}).get("finished_at") else
                    "evidence_incomplete"
                    if transcript["status"] != "complete" else "ready")
                payloads[key] = payload
            except ValueError:
                payload_status = "evidence_oversized"
        checks_mapping[key] = checks
        rows.append({
            "attempt_id": identity,
            "attempt_key": key,
            "case_id": case.case_id,
            "arm": attempt["arm"],
            "repetition": attempt["repetition"],
            "kind": case.kind,
            "expected_activation": case.expected_activation,
            "execution_status": measured["execution_status"],
            "failure_category": failure,
            "usage": measured,
            "grade": None,
            "judge": None,
            "independent_checks": checks,
            "check_records": check_records,
            "result_status": result_status,
            "trajectory_status": trajectory_status,
            "transcript_status": transcript["status"],
            "capture_status": captured["status"],
            "payload_status": payload_status,
            "request_bytes": request_bytes,
            "tool_projection": tool_projection,
            "artifact_record_status": artifact_record_status,
            "artifact_checks": artifact_record,
            "trial_paths": [str(p) for p in trials],
            "capture_dir": str(capture_dir) if capture_dir else None,
            "source_reference": str(source),
            "session_paths": [str(p) for p in sessions]
        })
    return {
        "schema_version":
        1,
        "rows":
        rows,
        "payloads": [
            payloads[key] for key in blinding["grading_order"]
            if key in payloads
        ],
        "checks_mapping":
        checks_mapping,
        "evidence":
        all_evidence,
        "blinding":
        blinding,
        "input_hashes":
        hashes,
        "manifest_freeze":
        manifest_freeze,
        "tool_projection":
        TOOL_POLICIES[tool_projection],
        "limitations": [
            "No worker, checker or judge was run by this collector.",
            "Original private artifacts remain at the recorded source paths; retain the entire run.",
            "Oversized full evidence is retained, never clipped or silently graded.",
            "Independent checker coverage is defined only for registered E02, E03, E05 and E08 cases."
        ]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping",
                        type=Path,
                        help="Reuse an earlier private blinding.json")
    parser.add_argument(
        "--capture-dirs",
        type=Path,
        help="Private JSON mapping attempt_id to recovered capture directory")
    parser.add_argument("--checks-dir", type=Path)
    parser.add_argument("--tool-projection",
                        choices=tuple(TOOL_POLICIES),
                        default="full")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        overrides = _json(args.capture_dirs, {}) if args.capture_dirs else None
        result = collect(args.run_dir,
                         mapping_path=args.mapping,
                         capture_dirs=overrides,
                         checks_dir=args.checks_dir,
                         tool_projection=args.tool_projection)
        args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        output_hashes = {}

        def write(name, value):
            path = args.output_dir / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            judging._private_json(path, value)
            output_hashes[name] = _digest(path.read_bytes())

        for name in ("rows", "blinding", "checks_mapping"):
            write(name + ".json", result[name])
        for payload in result["payloads"]:
            write("payloads/" + payload["attempt_key"] + ".json", payload)
        for key, evidence in result["evidence"].items():
            write("evidence/" + key + ".json", evidence)
        write(
            "provenance.json", {
                "schema_version": 1,
                "input_hashes": result["input_hashes"],
                "manifest_freeze": result["manifest_freeze"],
                "tool_projection": result["tool_projection"],
                "limitations": result["limitations"]
            })
        write("hashes.json", {
            "inputs": result["input_hashes"],
            "outputs": dict(output_hashes)
        })
    except (OSError, ValueError, KeyError, TypeError):
        print(
            json.dumps({
                "status": "collection_failed",
                "reason": "Inspect private inputs; output must be new."
            }))
        return 1
    print(
        json.dumps({
            "status": "collected",
            "attempts": len(result["rows"]),
            "payloads": len(result["payloads"])
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
