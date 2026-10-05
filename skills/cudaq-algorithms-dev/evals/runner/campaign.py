"""Shared scheduling and evidence collection for tool-using model evaluations."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import providers, workspace, verification, checkers

EVALS = Path(__file__).resolve().parents[1]
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file in /workspace or /skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description":
            "Run a bash command inside the isolated workspace; write Python kernels to files before executing them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string"
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    },
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def safe_name(value):
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"Invalid path identifier: {value!r}")
    return value


def schedule(cases, seeds):
    """Keep pairs adjacent, counterbalancing the first arm across cases and seeds."""
    rows = []
    for seed_index, seed in enumerate(seeds):
        for index, case in enumerate(cases):
            arms = ("baseline",
                    "skill") if (seed_index + index) % 2 == 0 else ("skill",
                                                                    "baseline")
            rows.extend({
                "case_id": case["id"],
                "seed": seed,
                "arm": arm
            } for arm in arms)
    return rows


def attempt_path(root, alias, row):
    if row["arm"] not in ("baseline", "skill") or type(row["seed"]) is not int:
        raise ValueError("Invalid attempt seed/arm")
    return Path(root) / "attempts" / safe_name(alias) / safe_name(
        row["case_id"]) / str(row["seed"]) / row["arm"]


def reserve(path):
    path.mkdir(parents=True, exist_ok=True)
    if (path / "result.json").exists():
        return False
    try:
        with (path / "started.json").open("x") as handle:
            json.dump({"started_at": time.time(), "pid": os.getpid()}, handle)
    except FileExistsError:
        return False
    return True


def total_tokens(responses):
    counts = [
        r["usage"].get("total_tokens")
        if isinstance(r.get("usage"), dict) else None for r in responses
    ]
    if not counts or any(type(c) is not int or c < 0 for c in counts):
        return None
    return sum(counts)


def inventory(root):
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def snapshot(source, dest):
    """Copy authored regular files, excluding generated caches and all symlinks."""
    if source.is_symlink():
        raise ValueError(f"Snapshot symlink is not allowed: {source}")
    if source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in {"__pycache__", ".pytest_cache", ".git"
                          } or child.suffix in {".pyc", ".pyo"}:
            continue
        snapshot(child, dest / child.name)


def prepare(root,
            source,
            selected_models,
            *,
            case_ids=None,
            pilot=False,
            budget=900,
            tool_limit=100,
            runtime_python=None,
            exposure="listed",
            request_timeout=180,
            skill_label="current",
            time_limit=None,
            token_limit=None,
            transport_max_retries=None,
            backend_wait_budget_seconds=1800):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise ValueError(
            "Campaign directory already exists; use run to resume it")
    if root.is_relative_to(source):
        raise ValueError("Store campaign evidence outside the source checkout")
    if not selected_models or budget <= 0 or request_timeout <= 0 or tool_limit < 1:
        raise ValueError("Models and positive resource limits are required")
    if exposure not in ("listed", "injected"):
        raise ValueError("Invalid skill exposure")
    if transport_max_retries is not None and (
            type(transport_max_retries) is not int
            or not 0 <= transport_max_retries <= 10):
        raise ValueError(
            'transport_max_retries must be null or an integer from 0 to 10')
    if (isinstance(backend_wait_budget_seconds, bool)
            or not isinstance(backend_wait_budget_seconds, (int, float))
            or not math.isfinite(backend_wait_budget_seconds)
            or backend_wait_budget_seconds < 0):
        raise ValueError(
            'backend_wait_budget_seconds must be a finite nonnegative number')
    suite = read_json(EVALS / "evals.json")
    contracts = read_json(EVALS / 'case_contracts.json')
    checker_contracts = checkers.specifications()
    applicable = {
        key
        for key, value in contracts.items() if value['executable_check']
    }
    if set(checker_contracts) != applicable:
        raise ValueError(
            'Executable checker coverage differs from case contracts')
    known = {c["id"] for c in suite["evals"]}
    if case_ids and (not pilot or not set(case_ids) <= known):
        raise ValueError(
            "Case selection requires pilot mode and known canonical IDs")
    aliases = [safe_name(m["id"]) for m in selected_models]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Duplicate model aliases")
    # Refuse embedded credentials: only environment names and explicit file references.
    if any(
            any(k in m for k in ("api_key", "token", "authorization"))
            for m in selected_models):
        raise ValueError(
            "Use credential environment/file references, not inline credentials"
        )
    root.mkdir(parents=True)
    shutil.copyfile(EVALS / "evals.json", root / "suite.json")
    shutil.copyfile(EVALS / "case_contracts.json",
                    root / "case_contracts.json")
    snapshot(EVALS / "files", root / "fixtures" / "files")
    snapshot(source / "skills" / "cudaq-algorithms", root / "skill")
    for name in ("python", "tests", "docs/sphinx", "README.md",
                 "pyproject.toml", ".cudaq_version"):
        if (source / name).exists():
            snapshot(source / name, root / "source" / name)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    source_hash = hashlib.sha256(
        json.dumps(inventory(root / "source"),
                   sort_keys=True).encode()).hexdigest()
    revision = f"{revision}+tree-sha256:{source_hash}"
    env = {
        "simulator": "qpp-cpu",
        "precision": "fp64",
        "threads": "1",
        "runner": "http-tools-v1",
        # Environment strings survive canonical export without extending its
        # fixed schema. The settings below remain the execution authority.
        "transport_policy": json.dumps({
            'name': 'measured_failed_requests_and_backoff_v1',
            'task_budget_seconds': budget,
            'backend_wait_budget_seconds': backend_wait_budget_seconds,
            'wall_cap_seconds': budget + backend_wait_budget_seconds,
            'request_timeout_seconds': request_timeout,
            'max_retries_per_request': transport_max_retries,
            'fallback_backoff_seconds': [2, 4, 8, 16, 32, 60],
            'honor_retry_after': True,
            'retry_after_policy': 'maximum_of_header_and_fallback',
            'successful_request_latency': 'task_time',
            'zero_wait_allowance': 'no_retries_all_latency_charged'
        }, sort_keys=True, allow_nan=False)
    }
    python = str(runtime_python or "python3")
    probe = subprocess.run([
        python, "-c",
        "import json,platform,importlib.metadata as m; print(json.dumps({'python':platform.python_version(),'cudaq':m.version('cudaq'),'numpy':m.version('numpy'),'scipy':m.version('scipy')}))"
    ],
                           capture_output=True,
                           text=True,
                           timeout=30)
    if probe.returncode == 0:
        env.update(json.loads(probe.stdout))
    else:
        env["runtime_probe"] = "unavailable; run preflight before executing scientific tasks"
    env["cudaq_algorithms_source"] = revision
    models = []
    for model in selected_models:
        models.append({
            **model, "alias": model["id"],
            "id": model["model"],
            "tier": model.get("tier", "mid"),
            "wall_seconds": {
                "baseline": 0.0,
                "skill": 0.0
            },
            "notes": {
                group: "No graded evidence yet; inspect attempt transcripts."
                for group in ("regression", "science", "controls")
            }
        })
    spec = {
        "schema_version":
        1,
        "mode":
        "pilot" if pilot else "full",
        "suite_sha256":
        digest(root / "suite.json"),
        "source_revision":
        revision,
        "skill_revision":
        "sha256:" + hashlib.sha256(
            json.dumps(inventory(root / "skill"),
                       sort_keys=True).encode()).hexdigest(),
        "skill_label":
        skill_label,
        "environment":
        env,
        "protocol": {
            "seeds": [0] if pilot else [0, 1, 2, 3, 4],
            "budget_seconds": budget,
            "tool_call_limit": tool_limit,
            "skill_exposure": exposure,
            "time_regression_limit": time_limit,
            "token_regression_limit": token_limit
        },
        "models":
        models,
        "case_contracts":
        read_json(root / "case_contracts.json"),
        "checker_contracts":
        checker_contracts,
        "verification_policy":
        "after_final_answer_v1",
        "selected_cases":
        case_ids or [c["id"] for c in suite["evals"]],
        "settings": {
            "backend_wait_budget_seconds":
            backend_wait_budget_seconds,
            "transport_max_retries":
            transport_max_retries,
            "transport_backoff":
            "max(Retry-After, 2**(retry_index+1) seconds capped at 60)",
            "runtime_python":
            str(Path(runtime_python).absolute()) if runtime_python else None,
            "request_timeout":
            request_timeout,
            "isolation":
            "unshare"
        }
    }
    write_json(root / "campaign.json", spec)
    write_json(root / "definition.json", immutable_definition(spec))
    write_json(root / "evaluator.json", evaluator_inventory())
    write_json(root / "inputs.json", {
        name: inventory(root / name)
        for name in ("source", "skill", "fixtures")
    })
    return spec


def immutable_definition(spec):
    definition = json.loads(json.dumps(spec))
    for model in definition["models"]:
        model.pop("wall_seconds", None)
        model.pop("notes", None)
    return definition


def evaluator_inventory():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((Path(__file__).parent / "checkers").glob("*.py"))
    paths += [
        EVALS / "results.schema.json", EVALS.parent / "scripts/report_eval.py",
        EVALS.parent / "scripts/run_eval.py"
    ]
    return {str(p.relative_to(EVALS.parent)): digest(p) for p in sorted(paths)}


def verify_inputs(root):
    spec = read_json(root / "campaign.json")
    if immutable_definition(spec) != read_json(root / "definition.json"):
        raise ValueError(
            "Frozen execution configuration changed; start a new campaign")
    if evaluator_inventory() != read_json(root / "evaluator.json"):
        raise ValueError(
            "Evaluator code changed; start a new campaign with the new runner")
    if digest(root / "suite.json") != spec["suite_sha256"]:
        raise ValueError("Frozen suite changed")
    if read_json(root / "case_contracts.json") != spec["case_contracts"]:
        raise ValueError("Frozen case contracts changed")
    expected = read_json(root / "inputs.json")
    for name, files in expected.items():
        if inventory(root / name) != files:
            raise ValueError(f"Frozen {name} changed")
    return spec


def worker_messages(case, arm, exposure, skill_body):
    system = (
        "You are helping a researcher use CUDA-Q Algorithms. Work in /workspace. "
        "Use the available tools to inspect source, implement and verify the requested work. "
        "Give a final answer with the actual result and evidence. Tool commands run without network "
        "inside an isolated workspace. Python and CUDA-Q are available on PATH. "
        "Do not claim a calculation or test ran unless you observed its output."
    )
    system += verification.instructions(case['id'])
    if case.get("files"):
        system += " Supplied input files: " + ", ".join(
            "/workspace/.eval-inputs/" + f for f in case["files"]) + "."
    if arm == "skill":
        description = next((
            line[len("description:"):].strip()
            for line in skill_body.splitlines()
            if line.startswith("description:")
        ), "Guidance for CUDA-Q Algorithms primitives, workflows and validation."
                           )
        system += f" Available skill: cudaq-algorithms — {description} Read /skill/SKILL.md when relevant."
        if exposure == "injected":
            system += "\n" + skill_body
    return [{
        "role": "system",
        "content": system
    }, {
        "role": "user",
        "content": case["prompt"]
    }]


def read_worker_file(path, staged):
    requested = Path(path)
    if not requested.is_absolute():
        requested = Path("/workspace") / requested
    for prefix, host in (("/workspace", staged["workspace_path"]),
                         ("/skill", staged.get("skill_path"))):
        if host and requested.is_relative_to(prefix):
            base = Path(host).resolve()
            target = base / requested.relative_to(prefix)
            if target.is_symlink() or not target.resolve().is_relative_to(
                    base) or not target.is_file():
                raise ValueError(
                    "Only regular files inside the worker roots can be read")
            if target.stat().st_size > 2_000_000:
                raise ValueError(
                    "File exceeds read limit; use a scoped shell command")
            return target.read_text(
            ), prefix == "/skill" and target.name == "SKILL.md"
    raise ValueError("File is outside /workspace and /skill")


def run_attempt(root, attempt, case, model, row, spec, *, progress=None):
    start = time.monotonic()
    started_at = time.time()
    deadline = start + spec["protocol"]["budget_seconds"]
    settings = spec['settings']
    measured_wait_policy = 'backend_wait_budget_seconds' in settings
    wait_budget = settings.get('backend_wait_budget_seconds', 0)
    wall_deadline = deadline + wait_budget
    backend_wait, request_count = 0., 0
    transport_stop_reason = None
    responses, calls, opened = [], 0, None if row[
        "arm"] == "baseline" else False
    outcome, final, error = "error", None, None
    usage_incomplete = False
    initial_files = None
    transcript = attempt / "transcript.jsonl"

    def event(kind, **data):
        with transcript.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": kind,
                        "elapsed": time.monotonic() - start,
                        **data
                    },
                    allow_nan=False) + "\n")

    def record_wait(seconds):
        nonlocal backend_wait, deadline
        if wait_budget > 0:
            # Preserve actual elapsed time even if OS scheduling overslept.
            # The wall cap still prevents further work after such an overrun.
            backend_wait += seconds
            deadline = min(start + spec['protocol']['budget_seconds'] +
                           backend_wait, wall_deadline)

    event("started", case_id=case["id"], seed=row["seed"], arm=row["arm"])
    try:
        staged = workspace.stage_workspace(root / "source", root / "skill",
                                           root / "fixtures", case, attempt,
                                           row["arm"])
        if case['id'] in ('source-version-drift',
                          'implementation-scope-overreach'):
            initial_files = verification.file_inventory(
                Path(staged['workspace_path']))
        body = (root / "skill" /
                "SKILL.md").read_text() if row["arm"] == "skill" else ""
        messages = worker_messages(case, row["arm"],
                                   spec["protocol"]["skill_exposure"], body)
        event("input", messages=messages)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcome = "budget_timeout"
                break
            max_retries = settings.get('transport_max_retries',
                                       None if measured_wait_policy else 2)
            retry, last_error = 0, None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    transport_stop_reason = ('wall_budget' if time.monotonic()
                                             >= wall_deadline else 'task_budget')
                    raise last_error or providers.ProviderError(
                        "timeout", None, "Attempt deadline reached")
                if retry and measured_wait_policy and backend_wait >= wait_budget:
                    transport_stop_reason = 'backend_wait_budget'
                    raise last_error
                timeout = min(remaining, settings.get('request_timeout', 180))
                if wait_budget > 0:
                    # A failed request must fit the remaining wait allowance,
                    # while a successful one must fit the remaining task time.
                    timeout = min(timeout, wait_budget - backend_wait)
                    if timeout <= 0:
                        transport_stop_reason = 'backend_wait_budget'
                        raise providers.ProviderError(
                            'backend_wait_budget', None,
                            'Measured backend wait allowance exhausted.')
                request_count += 1
                event('request_started', request_index=request_count,
                      retry_index=retry, timeout_seconds=timeout)
                request_started = time.monotonic()
                try:
                    response = providers.complete(model, messages, TOOLS,
                                                  timeout)
                    break
                except KeyboardInterrupt:
                    latency = time.monotonic() - request_started
                    record_wait(latency)
                    usage_incomplete = True
                    event('transport_error', request_index=request_count,
                          retry_index=retry, latency_seconds=latency,
                          error={'kind': 'interrupted', 'status': None,
                                 'message': 'Provider request interrupted.',
                                 'retryable': False})
                    raise
                except providers.ProviderError as exc:
                    latency = time.monotonic() - request_started
                    record_wait(latency)
                    usage_incomplete = True
                    event("transport_error",
                          error=exc.as_dict(),
                          request_index=request_count,
                          retry_index=retry, latency_seconds=latency)
                    last_error = exc
                    fallback_delay = min(60., 2**min(retry + 1, 6))
                    delay = getattr(exc, "retry_after_seconds", None)
                    delay = delay if delay is not None else fallback_delay
                    if measured_wait_policy:
                        delay = max(delay, fallback_delay)
                    if not exc.retryable:
                        transport_stop_reason = 'not_retryable'
                    elif max_retries is not None and retry >= max_retries:
                        transport_stop_reason = 'retry_limit'
                    elif measured_wait_policy and wait_budget - backend_wait <= delay:
                        transport_stop_reason = 'backend_wait_budget'
                    elif wall_deadline - time.monotonic() <= delay:
                        transport_stop_reason = 'wall_budget'
                    elif not measured_wait_policy and deadline - time.monotonic() <= delay:
                        transport_stop_reason = 'task_budget'
                    else:
                        transport_stop_reason = None
                    if transport_stop_reason is not None:
                        event('transport_stopped', reason=transport_stop_reason,
                              request_index=request_count)
                        raise
                    event('transport_backoff_started', scheduled_seconds=delay,
                          request_index=request_count)
                    if progress:
                        label = f'HTTP {exc.status}' if exc.status else exc.kind
                        limit = f'/{max_retries}' if max_retries is not None else ''
                        charged = 'backend wait allowance' if wait_budget > 0 else 'task budget'
                        progress(
                            f"{model.get('alias', model.get('id'))}: {label}; retry {retry+1}{limit} in {delay:g}s ({charged})"
                        )
                    sleep_started = time.monotonic()
                    try:
                        time.sleep(delay)
                    finally:
                        waited = time.monotonic() - sleep_started
                        record_wait(waited)
                        event('transport_backoff', seconds=waited,
                              scheduled_seconds=delay,
                              request_index=request_count)
                    retry += 1
            responses.append(response)
            event("response", response=response, request_index=request_count,
                  latency_seconds=time.monotonic() - request_started)
            if measured_wait_policy and time.monotonic() > deadline:
                outcome = 'budget_timeout'
                break
            message = response["choices"][0]["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content")
                if isinstance(content, str) and content.strip(
                ) and response["choices"][0].get("finish_reason") not in (
                        "length", "content_filter"):
                    outcome, final = "answered", content
                else:
                    outcome = "no_answer"
                break
            for call in tool_calls:
                if calls >= spec["protocol"]["tool_call_limit"]:
                    outcome = "tool_limit"
                    break
                if time.monotonic() >= deadline:
                    outcome = "budget_timeout"
                    break
                calls += 1
                name = call.get("function", {}).get("name")
                try:
                    args = json.loads(
                        call.get("function", {}).get("arguments", "{}"))
                    if name == "read_file":
                        content, read_skill = read_worker_file(
                            args["path"], staged)
                        if read_skill:
                            opened = True
                        output = {"content": content}
                    elif name == "shell":
                        output = workspace.run_command(
                            args["command"],
                            Path(staged["workspace_path"]),
                            timeout=max(0.001, deadline - time.monotonic()),
                            runtime_python=Path(
                                spec["settings"]["runtime_python"]) if
                            spec["settings"].get("runtime_python") else None,
                            isolation=spec["settings"].get(
                                "isolation", "unshare"),
                            log_dir=attempt / "tool-logs")
                        # Shell reads can be arbitrarily indirect; absence of read_file evidence is unknown.
                        if opened is False:
                            opened = None
                    else:
                        output = {"error": f"Unknown tool: {name}"}
                except (ValueError, KeyError, OSError) as exc:
                    output = {"error": str(exc)}
                event("tool",
                      call_id=call.get("id"),
                      name=name,
                      arguments=call.get("function", {}).get("arguments"),
                      output=output)
                content = json.dumps(output)
                if len(content) > 48000:
                    content = content[:48000] + "\n[truncated in model context; full output retained in transcript]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content
                })
            if outcome in ("tool_limit", "budget_timeout"):
                break
    except providers.ProviderError as exc:
        usage_incomplete = True
        outcome = "budget_timeout" if time.monotonic(
        ) >= deadline else "backend_error"
        error = exc.as_dict()
        if transport_stop_reason is not None:
            error['stop_reason'] = transport_stop_reason
    except KeyboardInterrupt:
        usage_incomplete = True
        outcome, error = "error", {
            "kind": "interrupted",
            "message": "Interrupted; attempt retained and will not be rerun"
        }
    except Exception as exc:
        outcome, error = "error", {
            "kind": type(exc).__name__,
            "message": str(exc)
        }
    observed_check = None
    if (outcome == 'answered' and spec.get('case_contracts', {}).get(
            case['id'], {}).get('executable_check')):
        try:
            observed_check = verification.execute(
                case['id'],
                Path(staged['workspace_path']),
                attempt,
                runtime_python=Path(spec['settings']['runtime_python'])
                if spec['settings'].get('runtime_python') else None,
                isolation=spec['settings'].get('isolation', 'unshare'),
                deadline=deadline,
                initial_files=initial_files)
            # The judge can cite the executed artifact output and checker result.
            # This event is written after the worker has finished and is never
            # included in a subsequent worker request.
            event('verification',
                  observation=observed_check,
                  evidence=read_json(attempt /
                                     observed_check['evidence_file']))
        except KeyboardInterrupt:
            outcome, error = 'error', {
                'kind': 'interrupted',
                'message': 'Interrupted during verification'
            }
        except Exception as exc:
            # Preserve the model result even if verifier infrastructure fails.
            # The absence of trusted verification remains unassessed on export.
            event('verification_error',
                  error=type(exc).__name__,
                  message=str(exc))
    duration = time.monotonic() - start
    # Successful request latency remains charged; queue time is never estimated.
    result = {
        **row, "outcome": outcome,
        "skill_opened": opened,
        "resources": {
            "task_seconds": max(0., duration - backend_wait),
            "wall_seconds": duration,
            "backend_wait_seconds": backend_wait,
            "tokens": None if usage_incomplete else total_tokens(responses),
            "tool_calls": calls,
            "cost_usd": None
        },
        "started_at": started_at,
        "finished_at": time.time(),
        "transport": {
            "request_count": request_count,
            "successful_request_count": len(responses),
            "stop_reason": transport_stop_reason
        },
        "final_answer": final,
        "error": error
    }
    event("finished", result=result)
    write_json(attempt / "result.json", result)
    if observed_check is not None:
        checked = observed_check['status'] in ('passed', 'failed')
        evidence = attempt / observed_check['evidence_file']
        completion = None
        if observed_check['status'] == 'passed':
            completion = {
                k: result['resources'][k]
                for k in ('task_seconds', 'wall_seconds', 'tokens',
                          'tool_calls')
            }
            # Exactly one trusted check is performed after the final answer.
            # These are measured cumulative costs at that first successful check.
            completion['failed_attempts'] = 0
        write_json(
            attempt / 'verification.json', {
                'status': observed_check['status'],
                'result_sha256': digest(attempt / 'result.json'),
                'transcript_sha256': digest(transcript),
                'evidence':
                str(evidence.relative_to(root)) if checked else None,
                'evidence_sha256': digest(evidence) if checked else None,
                'verified_completion': completion
            })
    return result


def campaign_status(root):
    root = Path(root)
    spec = read_json(root / "campaign.json")
    cases = [
        c for c in read_json(root / "suite.json")["evals"]
        if c["id"] in spec["selected_cases"]
    ]
    rows = schedule(cases, spec["protocol"]["seeds"])
    status = {}
    for model in spec["models"]:
        counts = {"pending": 0, "unfinished": 0}
        for row in rows:
            path = attempt_path(root, model["alias"], row)
            if (path / "result.json").exists():
                key = read_json(path / "result.json")["outcome"]
            elif (path / "started.json").exists():
                key = "unfinished"
            else:
                key = "pending"
            counts[key] = counts.get(key, 0) + 1
        status[model["alias"]] = counts
        preflight = root / ("preflight-" + model["alias"] + ".json")
        if preflight.exists() and not read_json(preflight).get("ok"):
            counts["endpoint_unavailable"] = 1
    return status


def run(root, *, aliases=None, limit=None, emit=print):
    root = Path(root).resolve()
    spec = verify_inputs(root)
    if aliases and set(aliases) - {m["alias"] for m in spec["models"]}:
        raise ValueError("Requested model is not in this frozen campaign")
    settings = spec["settings"]
    capability = workspace.preflight(
        isolation=settings["isolation"],
        runtime_python=Path(settings["runtime_python"])
        if settings.get("runtime_python") else None)
    if not capability.get("available") or not capability.get("isolated"):
        raise ValueError(f"Execution preflight failed: {capability}")
    if any(contract['executable_check']
           for key, contract in spec['case_contracts'].items()
           if key in spec['selected_cases']):
        verification.preflight(settings.get('runtime_python'))
    cases = [
        c for c in read_json(root / "suite.json")["evals"]
        if c["id"] in spec["selected_cases"]
    ]
    by_id = {c["id"]: c for c in cases}
    completed = 0
    lock = root / ".run-lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "Campaign is locked; do not run concurrent controllers. Inspect a stale lock before removing it."
        ) from exc
    os.close(descriptor)
    try:
        for model in spec["models"]:
            if aliases and model["alias"] not in aliases:
                continue
            rows = schedule(cases, spec["protocol"]["seeds"])
            if all((attempt_path(root, model["alias"], row) / "started.json"
                    ).exists() or (attempt_path(root, model["alias"], row) /
                                   "result.json").exists() for row in rows):
                continue
            health = providers.preflight(model,
                                         timeout=settings["request_timeout"])
            write_json(root / ("preflight-" + model["alias"] + ".json"),
                       health)
            if not health["ok"]:
                emit(
                    f"{model['alias']}: endpoint unavailable; attempts remain pending"
                )
                continue
            for row in rows:
                if limit is not None and completed >= limit:
                    return campaign_status(root)
                path = attempt_path(root, model["alias"], row)
                if not reserve(path):
                    continue
                emit(
                    f"{model['alias']} {row['case_id']} seed={row['seed']} {row['arm']}: started"
                )
                result = run_attempt(root,
                                     path,
                                     by_id[row["case_id"]],
                                     model,
                                     row,
                                     spec,
                                     progress=emit)
                arm_results = [
                    read_json(p) for p in (
                        root / "attempts" /
                        model["alias"]).glob(f"*/*/{row['arm']}/result.json")
                ]
                model["wall_seconds"][row["arm"]] = max(
                    r["finished_at"]
                    for r in arm_results) - min(r["started_at"]
                                                for r in arm_results)
                write_json(root / "campaign.json", spec)
                diagnostic = result.get('error') or {}
                detail = (
                    f" ({diagnostic.get('kind')}, HTTP {diagnostic['status']})"
                    if diagnostic.get('status') else f" ({diagnostic['kind']})"
                    if diagnostic.get('kind') else '')
                emit(
                    f"{model['alias']} {row['case_id']}: {result['outcome']}{detail}"
                )
                completed += 1
                if (result.get("error") or {}).get("kind") == "interrupted":
                    return campaign_status(root)
                if result["outcome"] == "backend_error":
                    emit(
                        f"{model['alias']}: stopping this model after error; completed evidence retained"
                    )
                    break
    finally:
        lock.unlink()
    return campaign_status(root)
