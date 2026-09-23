# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Codex end-to-end paired experiment; no skill mutations or API keys."""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import classical_cases
import quantum_cases
from grading import compare_output, required_calls_seen, read_regular
from comparison import distribution, paired_comparisons, scientific_api_passed
from runtime import (CODEX, MODEL, REPO, configuration, environment, inventory,
                     isolation_probe, runtime_metadata, sandbox_command,
                     scope_changes, search_tool_metadata,
                     stage_runtime_artifacts, stage_workspace)
from telemetry import TelemetryCollector

HERE = Path(__file__).resolve().parent
CASES = {
    s["id"]: (module, s)
    for module in (quantum_cases, classical_cases)
    for s in module.case_specs()
}
SUITE = "historical_api_informed"


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluator_inventory():
    return {
        name: digest
        for name, digest in inventory(HERE).items() if name.endswith(".py")
    }


def classify_attempt(execution, totals, passes_contract):
    if execution["timed_out"]:
        return "model_failure", []
    infrastructure = []
    if totals.get("input_tokens") is None or totals.get(
            "output_tokens") is None:
        infrastructure.append("final native usage unavailable")
    if execution["returncode"] != 0:
        infrastructure.append("native agent process failed")
    return ("infrastructure_failure" if infrastructure else
            "pass" if passes_contract else "model_failure"), infrastructure


def prompt_for(spec, python, arm):
    api_hint = (
        f"Required package workflow entry points: {json.dumps(spec.get('required_public_apis', []))}.\n"
        if spec.get("disclose_required_apis", True) else "")
    prompt = f'''Implement a real, executable CUDA-Q Algorithms application in app.py.
{spec['task']}

Interface: {python} -B app.py --input input.json --output result.npz
Read scientific parameters from JSON. Write the requested numeric scalar/array
fields into a NumPy NPZ (no object arrays). The same application must work on
another input of the same scientific form without editing code. Public input
is input.json. Requested fields: {json.dumps(spec['outputs'])}.
{api_hint}Use genuine packaged primitives, not reimplementations or mocked modules.
Package source is python/cudaq_algorithms, ordinary documentation/examples are
in docs/, and ordinary package tests are in tests/python/. PYTHONPATH is set.
Interpreter: {python}. CUDA-Q target qpp-cpu, precision fp64. Dependencies are
already installed. Do not install anything or use network, other projects,
hidden files, global skills, or subagents. Do not modify source/docs/tests/input.
Write only app.py, result.npz, and temporary work under .tmp/. Keep the final
application self-contained in app.py. Run it on the public input and inspect
its numerical output. A prose answer is not an executable deliverable.
'''
    if arm in ("previous", "skill"):
        prompt += (
            "\nAvailable task-specific skill: cudaq-algorithms. Read "
            "skills/cudaq-algorithms/SKILL.md and follow relevant guidance, "
            "loading references incrementally as needed. Skill files are read-only.\n"
        )
    return prompt


def campaign_arms(manifest):
    return list(manifest.get("arms", ("baseline", "skill")))


def arm_source(manifest, arm):
    return Path(manifest["arms"][arm]["snapshot"] if "arms" in
                manifest else manifest["source_snapshot"])


def verify_snapshots(manifest):
    """Check frozen content, and independently require equal non-skill inputs."""
    if inventory(Path(
            manifest["source_snapshot"])) != manifest["source_inventory"]:
        raise RuntimeError("source snapshot changed since prepare")
    common = {
        name: digest
        for name, digest in manifest["source_inventory"].items()
        if not name.startswith("skills/")
    }
    for arm, entry in manifest.get("arms", {}).items():
        actual = inventory(Path(entry["snapshot"]))
        if actual != entry["inventory"]:
            raise RuntimeError(f"{arm} snapshot changed since prepare")
        shared = {
            name: digest
            for name, digest in actual.items()
            if not name.startswith("skills/")
        }
        if shared != common:
            raise RuntimeError(
                "common source snapshot differs from canonical gold source")
        if arm == "baseline" and any(
                name.startswith("skills/") for name in actual):
            raise RuntimeError("baseline snapshot contains skill content")


def execute_app(source, spec, app_source, python, destination, limit=180):
    """Two independent sandboxes, one public and one held-out; no answer access."""
    module = CASES[spec["id"]][0]
    destination.mkdir(parents=True, exist_ok=False)
    checks = []
    for variant in (0, 1):
        evidence = destination / str(variant)
        evidence.mkdir()
        params = module.parameters(spec["id"], variant)
        reference = module.expected(spec["id"], params)
        workspace = Path(
            tempfile.mkdtemp(prefix="cudaq-e2e-check-")) / "workspace"
        stage_workspace(source, workspace, "baseline", params)
        stage_runtime_artifacts(workspace, spec["id"])
        (workspace / "app.py").write_text(app_source)
        shutil.copy2(HERE / "grading.py", workspace / "_trace_app.py")
        initial = inventory(workspace)
        cmd = sandbox_command(workspace, python, [
            workspace / "_trace_app.py", "--app", workspace / "app.py",
            "--input", workspace / "input.json", "--output",
            workspace / "result.npz", "--trace", workspace / ".tmp/trace.json",
            "--package-root", workspace / "python/cudaq_algorithms"
        ])
        started = time.monotonic()
        try:
            proc = subprocess.Popen(cmd,
                                    cwd=workspace,
                                    env=environment(workspace),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    start_new_session=True)
            stdout, stderr = proc.communicate(timeout=limit)
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            rc = None
        elapsed = time.monotonic() - started
        (evidence / "stdout.txt").write_text(stdout)
        (evidence / "stderr.txt").write_text(stderr)
        save(evidence / "input.json", params)
        save(evidence / "command.json", cmd)
        numeric = compare_output(workspace / "result.npz", reference, spec)
        try:
            trace = json.loads(
                read_regular(workspace / ".tmp/trace.json", limit=1_000_000))
            if not isinstance(trace, dict):
                trace = {}
        except (OSError, ValueError, TypeError):
            trace = {}
        host = required_calls_seen(spec.get("required_symbols", []),
                                   trace.get("calls", []))
        device = required_calls_seen(spec.get("required_kernels", []),
                                     trace.get("compiled_kernels", []))
        changes = scope_changes(initial, inventory(workspace))
        check = {
            "variant":
            variant,
            "returncode":
            rc,
            "elapsed_s":
            elapsed,
            "numeric":
            numeric,
            "host_api":
            host,
            "device_api":
            device,
            "scope_changes":
            changes,
            "passed":
            rc == 0 and numeric["passed"] and host["passed"]
            and device["passed"] and not changes
        }
        save(evidence / "trace.json", trace)
        try:
            (evidence / "result.npz").write_bytes(
                read_regular(workspace / "result.npz"))
        except (OSError, ValueError):
            pass
        save(evidence / "check.json", check)
        checks.append(check)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def prepare(args):
    if args.repetitions < 1 or args.parallel < 1 or args.timeout < 1:
        raise ValueError(
            "repetitions, parallel pairs, and timeout must be positive")
    selected = list(CASES) if not args.cases else args.cases.split(",")
    if set(selected) - CASES.keys() or len(selected) != len(set(selected)):
        raise ValueError("unknown or duplicate case IDs")
    previous = getattr(args, "previous_skill", None)
    if previous is not None and not (Path(previous) / "SKILL.md").is_file():
        raise ValueError("previous skill must contain SKILL.md")
    search = search_tool_metadata(getattr(args, "rg", None))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    staging_root = Path(tempfile.mkdtemp(prefix="cudaq-e2e-campaign-"))
    source = staging_root / "source"
    stage_workspace(REPO,
                    source,
                    "baseline", {},
                    rg_path=search["source_path"] if search else None)
    arm_names = ["baseline", "previous", "skill"
                 ] if previous is not None else ["baseline", "skill"]
    snapshots = {}
    for arm in arm_names:
        snapshot = staging_root / "arms" / arm
        skill_root = Path(
            previous
        ) if arm == "previous" else REPO / "skills/cudaq-algorithms"
        stage_workspace(source, snapshot, arm, {}, skill_root=skill_root)
        snapshots[arm] = {
            "snapshot":
            str(snapshot),
            "inventory":
            inventory(snapshot),
            "skill_origin":
            str(skill_root.resolve()) if arm != "baseline" else None
        }
    rng = random.Random(args.seed)
    pairs = [(cid, rep) for rep in range(1, args.repetitions + 1)
             for cid in selected]
    rng.shuffle(pairs)
    runs = []
    # Rotate a seeded arm order: every position is balanced to within one
    # block, and each case/repetition has one fresh run of every arm.
    order = list(arm_names)
    if previous is not None:
        rng.shuffle(order)
    for index, (cid, rep) in enumerate(pairs):
        shift = index % len(order)
        arms = order[shift:] + order[:shift]
        for arm in arms:
            runs.append({
                "id": f"{cid}--{rep}--{arm}",
                "case": cid,
                "repetition": rep,
                "arm": arm
            })
    interpreters = {
        cid:
        str(
            Path(args.psi4_python if cid == "psi4_energy" and args.psi4_python
                 else args.python).absolute())
        for cid in selected
    }
    manifest = {
        "kind":
        "native Codex end-to-end paired benchmark, not NVIDIA SkillEvaluator",
        "suite":
        SUITE,
        "source_revision":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                cwd=REPO,
                                text=True).strip(),
        "source_inventory":
        inventory(source),
        "evaluator_inventory":
        evaluator_inventory(),
        "source_snapshot":
        str(source),
        "staging_root":
        str(staging_root),
        "arms":
        snapshots,
        "search_tool":
        search,
        "model":
        MODEL,
        "reasoning_effort":
        "low",
        "seed":
        args.seed,
        "cudaq_target":
        "qpp-cpu",
        "precision":
        "fp64",
        "repetitions":
        args.repetitions,
        "timeout_s":
        args.timeout,
        "parallel_pairs":
        args.parallel,
        "ordering":
        "seed-shuffled case/repetition blocks, balanced rotating arm order; serial within each block",
        "stop_on_pass":
        False,
        "interpreters":
        interpreters,
        "runtimes": {
            p: runtime_metadata(p)
            for p in sorted(set(interpreters.values()))
        },
        "cases": [CASES[cid][1] for cid in selected],
        "runs":
        runs
    }
    verify_snapshots(manifest)
    save(output / "manifest.json", manifest)
    print(json.dumps({
        "prepared": len(runs),
        "output": str(output)
    }),
          flush=True)


def preflight(output):
    if (output / "runs").exists() and any((output / "runs").iterdir()):
        raise RuntimeError(
            "cannot re-preflight a campaign with attempt records; prepare a new campaign"
        )
    start_evaluator = evaluator_inventory()
    manifest = json.loads((output / "manifest.json").read_text())
    verify_snapshots(manifest)
    source = Path(manifest["source_snapshot"])
    root = output / ("preflight-" +
                     time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    root.mkdir(exist_ok=False)
    manifest["preflight_path"] = str(root)
    manifest["cases"] = [CASES[s["id"]][1] for s in manifest["cases"]]
    manifest["runtimes"] = {
        p: runtime_metadata(p)
        for p in sorted(set(manifest["interpreters"].values()))
    }
    manifest["cudaq_target"], manifest["precision"] = "qpp-cpu", "fp64"
    manifest["evaluator_inventory"] = evaluator_inventory()
    save(output / "manifest.json", manifest)
    isolation = []
    probe_root = Path(tempfile.mkdtemp(prefix="cudaq-e2e-isolation-"))
    arms = campaign_arms(manifest)
    for arm in arms:
        workspace = probe_root / arm
        stage_workspace(arm_source(manifest, arm), workspace, arm,
                        {"probe": 1})
    # Both sibling workspaces must already exist, otherwise ENOENT would be a
    # meaningless access-control pass. Also test every case-specific runtime.
    for arm in arms:
        workspace = probe_root / arm
        forbidden = [
            REPO / "skills/cudaq-algorithms/SKILL.md",
            HERE / "quantum_cases.py", output / "manifest.json",
            str(Path.home() / ".codex/skills/.system/openai-docs/SKILL.md")
        ]
        for other in arms:
            if other != arm:
                forbidden.append(probe_root / other / "input.json")
                if other != "baseline":
                    forbidden.append(probe_root / other /
                                     "skills/cudaq-algorithms/SKILL.md")
            if other != "baseline":
                forbidden.append(
                    arm_source(manifest, other) /
                    "skills/cudaq-algorithms/SKILL.md")
        for python in sorted(set(manifest["interpreters"].values())):
            isolation.append(isolation_probe(workspace, python, forbidden))
    save(root / "isolation.json", isolation)
    if not all(i["passed"] for i in isolation):
        raise RuntimeError("isolation preflight failed; see isolation.json")
    results = {}
    for spec in manifest["cases"]:
        cid = spec["id"]
        try:
            result = execute_app(source, spec,
                                 CASES[cid][0].reference_source(cid),
                                 manifest["interpreters"][cid], root / cid)
        except Exception as exc:
            result = {
                "passed": False,
                "infrastructure_error": type(exc).__name__ + ": " + str(exc)
            }
        results[cid] = result
        print(json.dumps({
            "preflight": cid,
            "passed": result["passed"]
        }),
              flush=True)
        save(root / "summary.json", results)
    if evaluator_inventory() != start_evaluator:
        raise RuntimeError(
            "evaluator changed during preflight; rerun before any model attempts"
        )
    verify_snapshots(manifest)
    manifest["gold_verified_evaluator_inventory"] = start_evaluator
    save(output / "manifest.json", manifest)


def run_native(command, workspace, prompt, destination, timeout, collector):
    """Binary selector reads avoid buffered readline timestamp artefacts."""
    events, pending = [], b""
    start = time.monotonic()
    timed_out = False
    with (destination /
          "stderr.txt").open("wb") as err, (destination /
                                            "events.jsonl").open("w") as log:
        proc = subprocess.Popen(command,
                                cwd=workspace,
                                env=environment(workspace),
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=err,
                                bufsize=0,
                                start_new_session=True)
        proc.stdin.write(prompt.encode())
        proc.stdin.close()
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)

        def record(line):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                event = {
                    "type": "unparsed_stdout",
                    "line": line.decode(errors="replace")
                }
            collector.record_event(event)
            item = {"elapsed_s": time.monotonic() - start, "event": event}
            events.append(item)
            log.write(json.dumps(item) + "\n")
            log.flush()

        while sel.get_map() or proc.poll() is None:
            if time.monotonic() - start > timeout:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            for key, _ in sel.select(timeout=0.5):
                data = os.read(key.fd, 65536)
                if not data:
                    sel.unregister(key.fileobj)
                    continue
                pending += data
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line:
                        record(line)
        if pending.strip():
            record(pending)
        rc = proc.wait()
        sel.close()
    return {
        "returncode":
        rc,
        "timed_out":
        timed_out,
        "elapsed_s":
        time.monotonic() - start,
        "commands":
        sum(e["event"].get("type") == "item.completed"
            and e["event"].get("item", {}).get("type") == "command_execution"
            for e in events),
        "tool_output_bytes":
        sum(
            len(e["event"].get("item", {}).get("aggregated_output",
                                               "").encode()) for e in events
            if e["event"].get("type") == "item.completed")
    }


def run_one(run, manifest, output):
    destination = output / "runs" / run["id"]
    if (destination / "result.json").exists():
        return json.loads((destination / "result.json").read_text())
    # A partial attempt is retained and marked interrupted, never retried.
    if destination.exists():
        return blocked_result(run,
                              output,
                              "interrupted prior attempt; evidence preserved",
                              attempted=True)
    destination.mkdir(parents=True)
    source = Path(manifest["source_snapshot"])
    workspace = Path(
        manifest["staging_root"]) / "runs" / run["id"] / "workspace"
    module, spec = CASES[run["case"]]
    stage_workspace(arm_source(manifest, run["arm"]), workspace, run["arm"],
                    module.parameters(run["case"], 0))
    stage_runtime_artifacts(workspace, run["case"])
    before = inventory(workspace)
    python = manifest["interpreters"][run["case"]]
    prompt = prompt_for(spec, python, run["arm"])
    (destination / "prompt.txt").write_text(prompt)
    save(destination / "initial-inventory.json", before)
    print(json.dumps({"started": run["id"]}), flush=True)
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with TelemetryCollector() as collector:
        command = [
            str(CODEX), "exec", *configuration(workspace, python),
            *collector.config_args, "--model", manifest["model"],
            "--ephemeral", "--skip-git-repo-check", "--json", "-C",
            str(workspace), "-o",
            str(destination / "final.txt"), "-"
        ]
        save(destination / "command.json", command)
        execution = run_native(command, workspace, prompt, destination,
                               manifest["timeout_s"], collector)
    telemetry = collector.summary()
    save(destination / "telemetry-events.json", collector.events)
    save(destination / "telemetry-summary.json", telemetry)
    after = inventory(workspace)
    changes = scope_changes(before, after)
    save(destination / "final-inventory.json", after)
    application = workspace / "app.py"
    grading = {"passed": False, "reason": "missing or symlink application"}
    try:
        code = read_regular(application, limit=5_000_000).decode()
    except (OSError, ValueError, UnicodeError):
        code = None
    if code is not None:
        (destination / "app.py").write_text(code)
        grading = execute_app(source, spec, code, python,
                              destination / "grading")
    cumulative = telemetry["totals"]
    status, infrastructure = classify_attempt(
        execution, cumulative, not changes and grading["passed"])
    passed = status == "pass"
    result = {
        **run,
        **execution, "started_utc":
        start_utc,
        "telemetry":
        telemetry,
        "grading":
        grading,
        "scope_changes":
        changes,
        "passed":
        passed,
        "scientific_api_passed":
        scientific_api_passed({"grading": grading}),
        "status":
        status,
        "attempted":
        True,
        "infrastructure_reasons":
        infrastructure,
        "time_to_verified_solution_s":
        execution["elapsed_s"] +
        sum(c["elapsed_s"]
            for c in grading.get("checks", [])) if passed else None,
        "billed_dollars":
        None
    }
    save(destination / "result.json", result)
    print(json.dumps({
        "finished": run["id"],
        "status": result["status"],
        "seconds": round(result["elapsed_s"], 2),
        "tokens": cumulative
    }),
          flush=True)
    return result


def blocked_result(run, output, reason, attempted=False):
    destination = output / "runs" / run["id"]
    destination.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetryCollector().summary()
    if (destination / "telemetry-summary.json").exists():
        telemetry = json.loads(
            (destination / "telemetry-summary.json").read_text())
    result = {
        **run, "status": "infrastructure_failure",
        "passed": False,
        "attempted": attempted,
        "infrastructure_reasons": [reason],
        "elapsed_s": None,
        "time_to_verified_solution_s": None,
        "telemetry": telemetry,
        "scope_changes": [],
        "billed_dollars": None
    }
    save(destination / "result.json", result)
    return result


def run_all(output):
    # Protect resume from another controller; file is not an agent credential.
    with (output / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_all_locked(output)


def _run_all_locked(output):
    manifest = json.loads((output / "manifest.json").read_text())
    verify_snapshots(manifest)
    preflight_path = Path(manifest["preflight_path"])
    preflights = json.loads((preflight_path / "summary.json").read_text())
    if not all(i["passed"]
               for i in json.loads((preflight_path /
                                    "isolation.json").read_text())):
        raise RuntimeError("isolation preflight must pass")
    if evaluator_inventory() != manifest.get(
            "gold_verified_evaluator_inventory"):
        raise RuntimeError(
            "evaluator code changed after gold preflight; use a new campaign after any agent runs"
        )
    width = len(campaign_arms(manifest))
    pairs = [
        manifest["runs"][i:i + width]
        for i in range(0, len(manifest["runs"]), width)
    ]
    for block in pairs:
        if (len(block) != width
                or {r["arm"]
                    for r in block} != set(campaign_arms(manifest))
                or len({(r["case"], r["repetition"])
                        for r in block}) != 1):
            raise RuntimeError("invalid case/repetition arm block")

    def run_pair(pair):
        results = []
        for run in pair:
            if not preflights.get(run["case"], {}).get("passed"):
                results.append(
                    blocked_result(run, output,
                                   "real gold preflight failed for this task"))
                continue
            try:
                results.append(run_one(run, manifest, output))
            except Exception as exc:
                results.append(
                    blocked_result(run,
                                   output,
                                   "controller exception: " +
                                   type(exc).__name__ + ": " + str(exc),
                                   attempted=True))
        return results

    with ThreadPoolExecutor(
            max_workers=manifest["parallel_pairs"]) as executor:
        results = [
            result for pair in executor.map(run_pair, pairs) for result in pair
        ]
    save(output / "results.json", results)
    report(output)


def report(output):
    manifest = json.loads((output / "manifest.json").read_text())
    results = [
        json.loads(p.read_text())
        for p in sorted((output / "runs").glob("*/result.json"))
    ]
    summary = {
        "completed_records":
        len(results),
        "executed_attempts":
        sum(r.get("attempted", True) for r in results),
        "suite":
        manifest.get("suite", "historical_api_informed"),
        "preflight_blocked_slots":
        sum(not r.get("attempted", True) for r in results),
        "interrupted_attempts":
        sum(
            any("interrupted" in reason
                for reason in r.get("infrastructure_reasons", []))
            for r in results),
        "planned_attempts":
        len(manifest["runs"]),
        "arms": {},
        "families": {}
    }
    arms = campaign_arms(manifest)
    summary["paired_comparisons"] = paired_comparisons(
        results, arms, planned_runs=manifest["runs"])
    for arm in arms:
        subset = [r for r in results if r["arm"] == arm]
        valid = [r for r in subset if r["status"] != "infrastructure_failure"]
        summary["arms"][arm] = {
            "completed":
            len(subset),
            "valid":
            len(valid),
            "passes":
            sum(r["passed"] for r in valid),
            "infrastructure_failures":
            len(subset) - len(valid),
            "strict_passes":
            sum(r["passed"] for r in subset),
            "scientific_api_passes":
            sum(scientific_api_passed(r) for r in subset),
            "executed_attempts":
            sum(r.get("attempted", True) for r in subset),
            "blocked_slots":
            sum(not r.get("attempted", True) for r in subset),
            "tokens_by_field": {
                field:
                distribution(
                    [r["telemetry"]["totals"].get(field) for r in subset])
                for field in ("input_tokens", "cached_input_tokens",
                              "output_tokens", "reasoning_output_tokens",
                              "total_tokens")
            },
            "tokens_all_attempts":
            distribution([
                r["telemetry"]["totals"].get("total_tokens") for r in subset
            ]),
            "observed_request_input_peaks":
            distribution([
                r["telemetry"].get("observed_peak_request_input_tokens")
                for r in subset
            ]),
            "compactions":
            distribution([r["telemetry"].get("compactions") for r in subset]),
            "metric_completeness_counts": {
                field:
                sum(
                    bool(r["telemetry"]["completeness"].get(field))
                    for r in subset)
                for field in ("totals", "observed_peak_request_input_tokens",
                              "compactions", "cumulative_crosscheck")
            },
            "counter_mismatches":
            sum(r["telemetry"]["counter_crosscheck"].get("matches") is False
                for r in subset),
            "attempt_seconds":
            distribution([r["elapsed_s"] for r in subset]),
            "successful_solution_seconds":
            distribution([
                r["time_to_verified_solution_s"] for r in valid if r["passed"]
            ]),
            "failure_seconds":
            distribution([r["elapsed_s"] for r in valid if not r["passed"]])
        }
    for family in sorted(
        {f
         for spec in manifest["cases"]
         for f in spec["families"]}):
        ids = {s["id"] for s in manifest["cases"] if family in s["families"]}
        summary["families"][family] = {
            arm: {
                "passes":
                sum(r["passed"] for r in results
                    if r["arm"] == arm and r["case"] in ids),
                "scientific_api_passes":
                sum(
                    scientific_api_passed(r) for r in results
                    if r["arm"] == arm and r["case"] in ids),
                "executed_attempts":
                sum(
                    r.get("attempted", True) for r in results
                    if r["arm"] == arm and r["case"] in ids),
                "valid":
                sum(r["status"] != "infrastructure_failure" for r in results
                    if r["arm"] == arm and r["case"] in ids)
            }
            for arm in arms
        }
    save(output / "summary.json", summary)
    lines = [
        "# End-to-end paired benchmark", "",
        f"Executed {summary['executed_attempts']}/{len(manifest['runs'])} planned agent attempts; {summary['preflight_blocked_slots']} slots infrastructure-blocked before execution; {len(results)} finalized records.",
        "",
        f"Suite: {summary['suite']}. Results belong to this suite only; do not pool them with another corpus.",
        "",
        "Native Codex, gpt-5.5 low; actual CUDA-Q 0.15.1 qpp-cpu applications, public and held-out numerical checks. This is not NVIDIA SkillEvaluator.",
        "", "| Task (families) | " +
        " | ".join(arm + " strict/scientific/executed"
                   for arm in arms) + " | Infrastructure failures |",
        "| --- | " + " | ".join("---:" for _ in arms) + " | ---: |"
    ]
    for spec in manifest["cases"]:
        group = [r for r in results if r["case"] == spec["id"]]
        cells = []
        for arm in arms:
            subset = [r for r in group if r["arm"] == arm]
            cells.append(
                f"{sum(r['passed'] for r in subset)}/{sum(scientific_api_passed(r) for r in subset)}/{sum(r.get('attempted', True) for r in subset)}"
            )
        lines.append(
            f"| {spec['id']} ({', '.join(spec['families'])}) | " +
            " | ".join(cells) +
            f" | {sum(r['status']=='infrastructure_failure' for r in group)} |"
        )
    lines += [
        "",
        "Paired differences below are candidate minus control. Every executed matched pair is eligible, including failures; each metric reports its own observed-pair count. Verified-solution timing is conditional on both arms passing the strict contract. Blocked and missing pairs remain explicit in summary.json.",
        "",
        "| Comparison | Token delta median (n) | Observed request-input peak delta median (n) | Attempt seconds delta median (n) | Verified seconds delta median (n) |",
        "| --- | ---: | ---: | ---: | ---: |"
    ]
    for name, comparison in summary["paired_comparisons"].items():
        cells = [
            f"{comparison[field]['median']} ({comparison[field]['n']})"
            for field in ("total_tokens", "observed_peak_request_input_tokens",
                          "elapsed_s", "time_to_verified_solution_s")
        ]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "", "## Measurements and limitations", "",
        "Final native usage counters supply cumulative input/output/cache totals. Cache and reasoning are subsets, not extra tokens. Request peaks are maxima of observed native response-completion input counters—not model capacity, not cumulative input, and not a proven complete measurement of all internal context. Native input-only completion events can disagree with final counters; every run preserves the cross-check. No dollar pricing is inferred from a ChatGPT login.",
        "",
        "Time to verified solution = full agent attempt wall time plus independent public/hidden execution time; it is not an unobserved first-correct-edit timestamp. Failed attempts retain their time and tokens. Five repetitions estimate this fixed corpus/model configuration, not all algorithms, hardware paths or future models. API profiling is execution evidence, not adversarial proof of output dataflow.",
        "",
        ("Required callable names are withheld in every focused prompt. This measures unaided API selection from natural-language scientific tasks with ordinary source/docs/tests available, and optional skill guidance. It is not discovery without package context. Scientific inputs differ from the historical corpus."
         if summary["suite"] == "focused_natural_language" else
         "Prompts disclose required public workflow entry points equally to every arm: this measures API-informed end-to-end implementation, not unaided keyword routing."
         ), "",
        "See summary.json for distributions and runs/* for raw events, applications, traces and grading evidence.",
        ""
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=("prepare", "preflight", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--psi4-python")
    parser.add_argument(
        "--previous-skill",
        type=Path,
        help=
        "preserved previous skill root; enables baseline/previous/skill arms")
    parser.add_argument(
        "--rg",
        type=Path,
        help=
        "real ripgrep binary to stage identically for every arm; defaults to PATH lookup"
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--cases",
        help=
        "optional comma-separated task IDs; defaults to every case in the selected suite"
    )
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args)
    else:
        manifest = json.loads(
            (args.output.resolve() / "manifest.json").read_text())
        if manifest.get("suite", "historical_api_informed") != SUITE:
            raise RuntimeError(
                "campaign suite does not match this runner; use the matching suite CLI"
            )
        {
            "preflight": preflight,
            "run": run_all,
            "report": report
        }[args.action](args.output.resolve())


if __name__ == "__main__":
    main()
