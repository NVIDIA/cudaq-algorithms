"""One command surface for endpoint checks, execution, assessment and reports."""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import assessment, campaign, providers

HEARTBEAT_SECONDS = 15


def models_from(args):
    catalog = campaign.read_json(args.catalog)["models"]
    requested = args.models.split(",") if args.models else [
        m["id"] for m in catalog
    ]
    by_id = {m["id"]: m for m in catalog}
    missing = set(requested) - set(by_id)
    if missing:
        raise ValueError("Unknown model aliases: " +
                         ", ".join(sorted(missing)))
    result = [dict(by_id[name]) for name in requested]
    for model in result:
        if args.key_file:
            model["api_key_file"] = str(Path(args.key_file).resolve())
    return result


def preflight_command(args):
    """Publish phase results immediately and show activity during blocking I/O."""
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model in models_from(args):
        rows.append({"id": model["id"], "status": "running", "ok": None})
        active = {"phase": None, "started": time.monotonic()}
        completed = set()
        stop = threading.Event()
        lock = threading.Lock()

        def update(evidence):
            rows[-1] = json.loads(json.dumps(evidence))
            campaign.write_json(args.output, {
                "schema_version": 1,
                "models": rows
            })
            with lock:
                for phase in ("completion", "tool_call"):
                    if phase in evidence and phase not in completed:
                        result = evidence[phase]
                        status = "passed" if result["ok"] else "failed"
                        reason = (result.get("error") or {}).get("kind")
                        detail = f" ({reason})" if reason else ""
                        print(
                            f"[{model['id']}] {phase}: {status}{detail} after {result['latency_seconds']:.1f}s",
                            file=sys.stderr,
                            flush=True)
                        completed.add(phase)
                phase = evidence.get("active_phase")
                if evidence.get("status") != "running":
                    phase = None
                if phase != active["phase"]:
                    active.update(phase=phase, started=time.monotonic())
                    if phase:
                        print(
                            f"[{model['id']}] {phase}: waiting; configured phase timeout {args.timeout:g}s",
                            file=sys.stderr,
                            flush=True)

        def heartbeat():
            while not stop.wait(HEARTBEAT_SECONDS):
                with lock:
                    if active["phase"]:
                        elapsed = time.monotonic() - active["started"]
                        print(
                            f"[{model['id']}] {active['phase']}: still waiting ({elapsed:.0f}s elapsed)",
                            file=sys.stderr,
                            flush=True)

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            update(providers.preflight(model, args.timeout, progress=update))
        except KeyboardInterrupt:
            interrupted = {**rows[-1], "status": "interrupted", "ok": False}
            update(interrupted)
            print(
                f"Interrupted. Partial endpoint results saved to {args.output}.",
                file=sys.stderr,
                flush=True)
            return 130
        finally:
            stop.set()
            worker.join()
    print(json.dumps(rows, indent=2))
    return 0 if all(row["ok"] for row in rows) else 1


def grade(root, judge, timeout, limit=None):
    root = Path(root).resolve()
    campaign.verify_inputs(root)
    cases = {
        c["id"]: c
        for c in campaign.read_json(root / "suite.json")["evals"]
    }
    count = 0
    for path in sorted((root / "attempts").glob("*/*/*/*/result.json")):
        attempt = path.parent
        if (attempt / "assessment.json").exists() and (
                attempt / "assessment.provenance.json").exists():
            continue
        result = campaign.read_json(path)
        if limit is not None and count >= limit:
            break
        transcript_path = attempt / "transcript.jsonl"
        transcript = transcript_path.read_text()
        start = time.monotonic()
        raw_path = attempt / "judge-response.json"
        provenance_path = attempt / "judge-response.provenance.json"
        if raw_path.exists():
            reply = campaign.read_json(raw_path)
            provenance = campaign.read_json(provenance_path)
            if provenance["result_sha256"] != campaign.digest(
                    path
            ) or provenance["transcript_sha256"] != campaign.digest(
                    transcript_path):
                raise ValueError(
                    f"Retained judge response refers to changed evidence: {attempt}"
                )
        else:
            messages = assessment.grade_prompt(cases[result["case_id"]],
                                               transcript)
            if result["outcome"] != "answered":
                messages[0][
                    "content"] += " This attempt produced no final answer: assertion values, expected_output and claimed_success must be false or null. Still assess critical failures and tripwires from the full transcript."
            reply = providers.complete(judge, messages, None, timeout)
            provenance = {
                "result_sha256": campaign.digest(path),
                "transcript_sha256": campaign.digest(transcript_path),
                "judge_model": judge["model"],
                "returned_model": reply.get("model"),
                "wall_seconds": time.monotonic() - start,
                "usage": reply.get("usage")
            }
            campaign.write_json(provenance_path, provenance)
            campaign.write_json(raw_path, reply)
        text = reply["choices"][0]["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Judge returned no assessment for {attempt}; retained response requires review"
            )
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        validated = assessment.validate_assessment(cases[result["case_id"]],
                                                   text, transcript)
        campaign.write_json(attempt / "assessment.json", validated)
        campaign.write_json(attempt / "assessment.provenance.json", provenance)
        count += 1
        print(f"Graded {attempt.relative_to(root)}", flush=True)
    return {"new_assessments": count}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def catalog(command):
        command.add_argument("--catalog",
                             type=Path,
                             default=campaign.EVALS / "models.json")
        command.add_argument(
            "--models",
            help="Comma-separated catalog aliases (default: all catalog models)"
        )
        command.add_argument(
            "--key-file",
            help=
            "Optional credential file; otherwise the model's configured environment variable"
        )

    ls = sub.add_parser(
        "models",
        help="Show configured model aliases; this does not probe endpoints")
    catalog(ls)
    pre = sub.add_parser(
        "preflight",
        help="Make a small completion and tool-call probe per model")
    catalog(pre)
    pre.add_argument(
        "--timeout",
        type=float,
        default=45,
        help="Total deadline per preflight phase, including retries and backoff"
    )
    pre.add_argument("--output", type=Path, required=True)
    init = sub.add_parser(
        "init", help="Freeze a full campaign or explicitly labelled pilot")
    catalog(init)
    init.add_argument("directory", type=Path)
    init.add_argument("--source", type=Path, default=campaign.EVALS.parents[2])
    init.add_argument("--pilot", action="store_true")
    init.add_argument("--cases",
                      help="Comma-separated canonical IDs; only in pilot mode")
    init.add_argument("--budget", type=float, default=900)
    init.add_argument("--tool-limit", type=int, default=100)
    init.add_argument("--request-timeout", type=float, default=180)
    init.add_argument(
        '--backend-wait-budget-seconds',
        type=float,
        default=1800,
        help='Maximum measured failed-request/backoff allowance per attempt; '
        'hard wall cap is task budget plus this allowance (default 1800)')
    init.add_argument(
        '--transport-max-retries',
        type=int,
        default=None,
        help='Optional maximum retries per failed transport request (0-10); '
        'by default retries are bounded by the backend wait allowance')
    init.add_argument("--runtime-python", type=Path, required=True)
    init.add_argument("--exposure",
                      choices=["listed", "injected"],
                      default="listed")
    init.add_argument("--skill-label", default="current")
    init.add_argument("--time-regression-limit", type=float)
    init.add_argument("--token-regression-limit", type=float)
    run = sub.add_parser(
        "run",
        help="Preflight and execute pending attempts; never replace an attempt"
    )
    run.add_argument("directory", type=Path)
    run.add_argument("--models",
                     help="Comma-separated aliases from this campaign")
    run.add_argument("--limit",
                     type=int,
                     help="Execute at most this many new attempts")
    status = sub.add_parser("status")
    status.add_argument("directory", type=Path)
    g = sub.add_parser(
        "grade",
        help="Privately assess original rubrics against retained transcripts")
    catalog(g)
    g.add_argument("directory", type=Path)
    g.add_argument("--timeout", type=float, default=180)
    g.add_argument("--limit", type=int)
    export = sub.add_parser(
        "export", help="Build and validate a complete canonical result bundle")
    export.add_argument("directory", type=Path)
    export.add_argument("--report",
                        type=Path,
                        help="Also generate the two-table report per model")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "limit", None) is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.command == "models":
            print(
                json.dumps([{
                    k: m.get(k)
                    for k in ("id", "model", "base_url")
                } for m in models_from(args)],
                           indent=2))
            return 0
        if args.command == "preflight":
            return preflight_command(args)
        if args.command == "init":
            spec = campaign.prepare(
                args.directory,
                args.source,
                models_from(args),
                case_ids=args.cases.split(",") if args.cases else None,
                pilot=args.pilot,
                budget=args.budget,
                tool_limit=args.tool_limit,
                runtime_python=args.runtime_python,
                exposure=args.exposure,
                request_timeout=args.request_timeout,
                skill_label=args.skill_label,
                time_limit=args.time_regression_limit,
                token_limit=args.token_regression_limit,
                transport_max_retries=args.transport_max_retries,
                backend_wait_budget_seconds=args.backend_wait_budget_seconds)
            print(
                json.dumps(
                    {
                        "directory":
                        str(args.directory),
                        "mode":
                        spec["mode"],
                        "models":
                        len(spec["models"]),
                        "attempts_per_model":
                        len(spec["selected_cases"]) *
                        len(spec["protocol"]["seeds"]) * 2
                    },
                    indent=2))
            return 0
        if args.command == "run":
            result = campaign.run(
                args.directory,
                aliases=args.models.split(",") if args.models else None,
                limit=args.limit,
                emit=lambda value: print(value, flush=True))
            print(json.dumps(result, indent=2))
            selected = [
                counts for alias, counts in result.items()
                if not args.models or alias in args.models.split(",")
            ]
            if any(
                    any(
                        counts.get(key, 0)
                        for key in ("backend_error", "error", "unfinished",
                                    "endpoint_unavailable"))
                    for counts in selected):
                return 1
            if any(counts.get("pending", 0)
                   for counts in selected) and (args.limit is None or not any(
                       counts.get("answered", 0) for counts in selected)):
                return 1
            return 0
        if args.command == "status":
            print(
                json.dumps(campaign.campaign_status(args.directory), indent=2))
            return 0
        if args.command == "grade":
            models = models_from(args)
            if len(models) != 1:
                raise ValueError("Select exactly one judge with --models")
            print(
                json.dumps(
                    grade(args.directory, models[0], args.timeout,
                          args.limit)))
            return 0
        if args.command == "export":
            campaign.verify_inputs(args.directory.resolve())
            bundle = assessment.build_bundle(args.directory)
            path = args.directory / "results.json"
            campaign.write_json(path, bundle)
            reporter = campaign.EVALS.parent / "scripts" / "report_eval.py"
            command = [sys.executable, str(reporter), str(path)]
            command += ["--output", str(args.report)
                        ] if args.report else ["--validate-only"]
            return subprocess.run(command, check=False).returncode
    except (ValueError, OSError, KeyError, RuntimeError,
            providers.ProviderError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
