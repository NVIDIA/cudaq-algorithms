#!/usr/bin/env python3
"""Read-only campaign progress snapshots; never run, grade, or retry attempts.

Usage: monitor_eval.py ROOT --output ROOT/status.json [--interval 30] [--once]
Campaigns can be ROOT/campaigns/<name>, ROOT/<name>, or ROOT itself.
Optional ROOT/launches.json: {"models": {"alias": {"pid": 123,
"controller_pid": 124, "exit_code_file": "logs/alias.exit"}}}. ``pid`` can
identify a launcher; supply ``controller_pid`` only when its identity is known.
An exit-code file contains one integer, or a JSON object with ``pid``,
``exit_code``, and optional ``finished_at``, written when the launcher exits.
JSON exit records must match the current launcher PID.

All counts are operational file snapshots, not valid delivery metrics. They do
not authenticate grading provenance or replace the canonical evidence reporter.
Only fixed status fields, identifiers, counts, and process metadata are emitted;
transcript contents, final answers, prompts, and error messages are never emitted.
"""

import argparse
from collections import Counter
import datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
OUTCOMES = {
    "answered", "backend_error", "budget_timeout", "tool_limit", "no_answer",
    "error"
}
CHECKS = {"passed", "failed", "not_run", "not_applicable"}
NOTICE = "Operational file snapshots; not delivery metrics."


def read_json(path):
    """Never treat a partially written or concurrently replaced file as final."""
    try:
        before = path.stat()
        if before.st_size > 16 * 1024 * 1024:
            return None, "unreadable"
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size,
                before.st_mtime_ns) != (after.st_ino, after.st_size,
                                        after.st_mtime_ns):
            return None, "unreadable"
        value = json.loads(raw)
        return (value, "ok") if isinstance(value, dict) else (None,
                                                              "unreadable")
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError, RecursionError):
        return None, "unreadable"


def pid_alive(pid):
    """Signal zero is a process existence probe, never a delivered signal."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        try:
            # An unreaped exited launcher can still answer signal zero.
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(
                ")", 1)[1].split()[0]
            return state not in {"Z", "X"}
        except (OSError, IndexError):
            return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False


def controller_status(root, launch):
    if not isinstance(launch, dict):
        launch = {}
    pid = launch.get("controller_pid", launch.get("pid"))
    alive = pid_alive(pid)
    code = None
    reference = launch.get("exit_code_file")
    if isinstance(reference, str):
        path = Path(reference)
        if not path.is_absolute():
            path = root / path
        record, _ = read_json(path)
        if record is not None:
            launch_pid = launch.get("pid", pid)
            if (type(launch_pid) is int and launch_pid > 0
                    and type(record.get("pid")) is int
                    and record["pid"] == launch_pid
                    and type(record.get("exit_code")) is int
                    and -9999 <= record["exit_code"] <= 9999):
                code = record["exit_code"]
        else:
            try:
                value = path.read_text().strip()
                if re.fullmatch(r"-?\d{1,4}", value):
                    code = int(value)
            except (OSError, UnicodeError):
                pass
    supervisor_state = launch.get("status")
    if supervisor_state not in ("paused", "cooldown"):
        supervisor_state = None
    return {
        "state":
        "exited" if code is not None else
        "running" if alive else "not_running" if alive is False else "unknown",
        "pid":
        pid if type(pid) is int and pid > 0 else None,
        "exit_code":
        code,
        # Explicit supervisor metadata does not establish controller liveness.
        "supervisor_state":
        supervisor_state,
    }


def campaigns(root):
    if (root / "campaign.json").is_file():
        return [root]
    base = root / "campaigns" if (root / "campaigns").is_dir() else root
    try:
        return sorted(path for path in base.iterdir()
                      if path.is_dir() and (path / "campaign.json").is_file())
    except OSError:
        return []


def mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def safe_name(value):
    return isinstance(value, str) and bool(SAFE_NAME.fullmatch(value))


def model_snapshot(root, directory, spec, alias, launch, now):
    cases = spec["selected_cases"]
    seeds = spec["protocol"]["seeds"]
    outcomes, checks, alerts = Counter(), Counter(), Counter()
    counts = Counter()
    latest = mtime(directory / "campaign.json")
    controller = controller_status(root, launch)
    if controller["supervisor_state"] is not None:
        alerts["supervisor_" + controller["supervisor_state"]] += 1
    controller_pid = launch.get("controller_pid") if isinstance(launch,
                                                                dict) else None
    for case in cases:
        contracts = spec.get("case_contracts")
        contract = contracts.get(case) if isinstance(contracts, dict) else None
        no_executable_check = (isinstance(contract, dict)
                               and contract.get("executable_check") is False)
        for seed in seeds:
            for arm in ("baseline", "skill"):
                path = directory / "attempts" / alias / case / str(seed) / arm
                files = {}
                states = {}
                for name in ("started", "result", "verification", "assessment",
                             "assessment.provenance"):
                    file = path / (name + ".json")
                    files[name], states[name] = read_json(file)
                    counts["partial_or_unreadable_files"] += states[
                        name] == "unreadable"
                    latest = max(latest, mtime(file))
                # Activity requires only metadata, never transcript contents.
                latest = max(latest, mtime(path / "transcript.jsonl"))
                result = files["result"]
                complete = False
                if result is not None:
                    if (result.get("case_id"), result.get("seed"),
                            result.get("arm")) != (case, seed, arm):
                        alerts["result_identity_mismatch"] += 1
                    elif (not isinstance(result.get("outcome"), str)
                          or result["outcome"] not in OUTCOMES):
                        alerts["invalid_outcome"] += 1
                    else:
                        complete = True
                if complete:
                    counts["completed"] += 1
                    outcome = result["outcome"]
                    outcomes[outcome] += 1
                    if outcome == "backend_error":
                        alerts["backend_error"] += 1
                    answer = result.get("final_answer")
                    if not isinstance(answer, str) or not answer.strip():
                        alerts["no_final_answer"] += 1
                    check = files["verification"]
                    if (check is not None
                            and isinstance(check.get("status"), str)
                            and check["status"] in CHECKS):
                        checks[check["status"]] += 1
                    elif states[
                            "verification"] == "missing" and no_executable_check:
                        # The frozen contract makes absence expected for discussion cases.
                        checks["not_applicable"] += 1
                    else:
                        checks["missing_or_unreadable"] += 1
                    assessment = files["assessment"]
                    if assessment is not None and assessment.get(
                            "case_id") == case:
                        counts["assessment_files"] += 1
                        # Presence of provenance is observed, not authenticated.
                        counts["graded"] += files[
                            "assessment.provenance"] is not None
                elif states["started"] != "missing":
                    started = files["started"] or {}
                    alive = pid_alive(started.get("pid"))
                    mismatch = (type(controller_pid) is int
                                and started.get("pid") != controller_pid)
                    if mismatch and alive:
                        alerts["reservation_pid_mismatch"] += 1
                    if alive and not mismatch:
                        counts["active"] += 1
                    else:
                        counts["unfinished"] += 1
                        if not alive:
                            alerts["unfinished_without_live_pid"] += 1
                elif states["result"] != "missing":
                    counts["unfinished"] += 1
                    alerts["result_without_readable_reservation"] += 1
                else:
                    counts["pending"] += 1
    scheduled = len(cases) * len(seeds) * 2
    if counts["completed"] < scheduled and controller["state"] in {
            "exited", "not_running"
    }:
        alerts["controller_stopped_before_completion"] += 1
    preflight, _ = read_json(directory / ("preflight-" + alias + ".json"))
    if preflight is not None and preflight.get("ok") is False:
        alerts["endpoint_unavailable"] += 1
    return {
        "scheduled": scheduled,
        **{
            key: counts[key]
            for key in ("completed", "active", "unfinished", "pending", "graded", "assessment_files", "partial_or_unreadable_files")
        },
        "outcomes": dict(sorted(outcomes.items())),
        "numerical_checks": dict(sorted(checks.items())),
        "last_activity_age_seconds":
        round(max(0, now - latest), 1) if latest else None,
        "controller": controller,
        "run_lock_present": (directory / ".run-lock").exists(),
        "alerts": dict(sorted(alerts.items())),
    }


def snapshot(root, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    launches, launch_state = read_json(root / "launches.json")
    launches = launches or {}
    launches = launches.get("models", launches)
    if not isinstance(launches, dict):
        launches = {}
    data = {
        "schema_version":
        1,
        "observed_at":
        datetime.datetime.fromtimestamp(now,
                                        datetime.timezone.utc).isoformat(),
        "notice":
        NOTICE,
        "models": {},
        "alerts": {},
    }
    alerts = Counter()
    if launch_state == "unreadable":
        alerts["unreadable_launch_metadata"] += 1
    directories = campaigns(root)
    if not directories:
        alerts["no_campaigns_discovered"] += 1
    for directory in directories:
        spec, _ = read_json(directory / "campaign.json")
        if spec is None:
            alerts["unreadable_campaign"] += 1
            continue
        cases = spec.get("selected_cases")
        protocol = spec.get("protocol")
        seeds = protocol.get("seeds") if isinstance(protocol, dict) else None
        models = spec.get("models")
        if not (isinstance(cases, list) and all(safe_name(c) for c in cases)
                and len(set(cases)) == len(cases) and isinstance(seeds, list)
                and all(type(s) is int
                        for s in seeds) and len(set(seeds)) == len(seeds)
                and isinstance(models, list)):
            alerts["invalid_campaign_schedule"] += 1
            continue
        for model in models:
            alias = model.get("alias") if isinstance(model, dict) else None
            if not safe_name(alias):
                alerts["invalid_model_identifier"] += 1
                continue
            if alias in data["models"]:
                alerts["duplicate_model_alias"] += 1
                continue
            data["models"][alias] = model_snapshot(root, directory, spec,
                                                   alias,
                                                   launches.get(alias,
                                                                {}), now)
    data["alerts"] = dict(sorted(alerts.items()))
    return data


def render(data):
    lines = [data["observed_at"], NOTICE]
    for alias, row in data["models"].items():
        checks = ",".join(
            f"{k}={v}" for k, v in row["numerical_checks"].items()) or "none"
        lines.append(
            f"{alias}: completed={row['completed']}/{row['scheduled']} "
            f"active={row['active']} unfinished={row['unfinished']} pending={row['pending']} "
            f"graded={row['graded']} checks[{checks}] "
            f"controller={row['controller']['state']} "
            f"supervisor={row['controller']['supervisor_state'] or 'unknown'} "
            f"activity_age={row['last_activity_age_seconds']}s")
        if row["alerts"]:
            lines.append("  alerts: " +
                         ", ".join(f"{k}={v}"
                                   for k, v in row["alerts"].items()))
    if data["alerts"]:
        lines.append("alerts: " +
                     ", ".join(f"{k}={v}" for k, v in data["alerts"].items()))
    return "\n".join(lines) + "\n"


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".",
                                             dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    output = args.output.resolve()
    text_output = output.with_suffix(".txt")
    events = output.with_name(output.stem + ".events.jsonl")
    destinations = (output, text_output, events)
    if len(set(destinations)) != 3:
        parser.error("--output must have a distinct JSON filename")
    for path in destinations:
        if path.resolve() == (args.root / "launches.json").resolve() or any(
                path.resolve().is_relative_to(directory.resolve())
                for directory in campaigns(args.root)):
            parser.error(
                "monitor output must be outside campaign evidence directories")
    try:
        while True:
            data = snapshot(args.root)
            human = render(data)
            atomic_write(output,
                         json.dumps(data, indent=2, allow_nan=False) + "\n")
            atomic_write(text_output, human)
            with events.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(data, separators=(",", ":"), allow_nan=False) +
                    "\n")
            if not args.quiet:
                print(human, end="", flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
