# Multi-model evaluation runner

Use `python3 skills/cudaq-algorithms-dev/scripts/run_eval.py` from the repository
root. The same commands run locally and in CI. Python 3.11+, `pytest`,
`jsonschema`, NumPy, SciPy and PySCF support the offline tests/reporting and
independent scientific references. Scientific execution additionally
requires a dedicated CUDA-Q environment and Linux mount/PID/network namespace
capabilities (`unshare`, `mount`, `chroot`, `setpriv`). Execution fails closed if
isolation is unavailable. On this development host the controller requires
approved elevated execution; no unrestricted execution fallback is offered by
the CLI.

The initial backend is an OpenAI-compatible Chat Completions tool loop. The
catalog contains ten NVIDIA Build model configurations, including Nemotron,
Kimi, Muse, GLM, DeepSeek and GPT-OSS. A catalog entry is a configuration, not a
claim that its endpoint is healthy. Native Codex and Claude CLI adapters are not
included in this first consolidated backend.

## Check endpoints first

Provide `NVIDIA_API_KEY` through your environment or CI secret store. Alternatively,
pass `--key-file /path/to/credential` to `preflight` and `init`. Only the reference
to a credential is saved; its value is never written into campaign metadata or
mounted into a worker. No endpoint or model substitution happens automatically.

```bash
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py models
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py preflight \
  --models nemotron-ultra,kimi --timeout 180 --output /tmp/endpoint-check.json
```

Each endpoint must return a completion and a valid tool call. The probe records
the requested/returned model, usage, latency and classified errors. A failed probe
returns a nonzero exit status. Reasoning models can need substantially longer
than a short connectivity probe; choose an explicit deadline. Bounded transport
retries retain errors and share the deadline. Authentication and semantic failures
are not retried. HTTP 429/503 and timeouts remain provider incidents, not model
correctness failures or permission to switch to a different model.

The Nemotron Ultra catalog entry includes the reasoning and nonempty-content
chat-template flags recommended by [NVIDIA's agent integration guide](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b).
Passing preflight checks text and tool-call emission; it cannot guarantee endpoint
availability throughout a longer conversation. Inspect saved transport errors
when a campaign request fails after a successful probe.

`--timeout` applies separately to the completion and tool-call phases, including
each phase's retries. Two slow phases can therefore take roughly twice the
configured timeout per model. The CLI prints phase progress and a waiting update
every 15 seconds. It saves results to `--output` as each phase finishes; Ctrl+C
preserves completed checks and records the interruption before exiting with 130.

## Freeze, run and resume

Start with a diagnostic pilot. Supply the path to the **virtual environment's**
Python executable, not its resolved system-interpreter target.

```bash
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py init /tmp/cudaq-pilot \
  --models nemotron-ultra,kimi --runtime-python /path/to/venv/bin/python \
  --pilot --cases science-block-encoding-heralded-observables \
  --budget 900 --request-timeout 180
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py run /tmp/cudaq-pilot
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py status /tmp/cudaq-pilot
```

A full campaign omits `--pilot` and `--cases`. It schedules the unchanged 62
cases, five paired repetitions and two arms: **620 attempts per model**. Repetition
IDs do not imply deterministic provider seeds. Skill exposure defaults to
`listed`; `--exposure injected` creates a different, explicitly labelled protocol.
Register material-regression ratios using `--time-regression-limit` and
`--token-regression-limit`; unspecified targets remain unassessed.

Run again to resume **pending** attempts. Started, interrupted and terminal
attempts are never replaced. `--models` selects campaign aliases, and `--limit`
bounds the number of new attempts in a batch. Execution is sequential within one
controller; separate campaign directories can be scheduled independently. A model
queue stops after a terminal backend error. Later pending attempts can resume
after the endpoint recovers; the failed observation remains in the campaign.

New campaigns freeze `--backend-wait-budget-seconds` (default 1800). Failed
request durations and actual retry sleeps consume this measured wait allowance;
successful request latency, tools and verification consume `--budget` (default
900). The attempt's wall cap is the sum of both budgets. Request timeouts fit
the remaining task, wait and wall allowances. With a zero wait allowance,
initial requests remain available and charged to task time, with no retries.

Retries resend the same failed request and never restart the scientific attempt.
By default their count is bounded by the wait allowance; an explicit
`--transport-max-retries 0` through `10` adds a count limit per request.
Fallback delays are 2, 4, 8, 16, 32, then 60 seconds. A server `Retry-After`,
expressed as seconds or an HTTP date, is honored without shortening it; the
larger of that value and the fallback delay applies so a zero header cannot
disable retry pacing. A delay that cannot fit the remaining wait and wall
allowances stops the attempt.
Request counts, failed-request latency, scheduled and measured sleeps, and the
reason for stopping are retained with the provider error metadata. Interrupted
requests and partial sleeps retain their measured wait. Scheduling overruns are
recorded at their actual duration and cannot authorize further work. The CLI
prints the error code and upcoming retry. Previously frozen campaigns without
the wait setting retain their original task-time accounting and retry count
(two retries when their count setting is absent).

One exclusive controller lock protects the campaign. A hard-killed process may
leave a lock and an unfinished reservation: inspect them before removing a stale
lock. The unfinished attempt remains incomplete evidence; the runner does not
invent an end time or rerun it to obtain a better result.

Source, skill, declared fixtures, suite, case contracts, model settings, protocol
and evaluator code (including every private checker) are frozen or fingerprinted.
The complete checker output contracts, oracles and numerical tolerances are saved
in `campaign.json` before execution. Changed inputs/configuration/code
require a new campaign. Every worker gets the original prompt and only its
declared inputs. Rubrics, expected outputs, sibling attempts and credentials are
not exposed to workers. Source is writable per attempt; fixtures and the skill
are read-only. Worker commands have no network access.

There is no request after the task or wall deadline to rescue a missing answer.
Unknown token usage stays null, including any failed transport after earlier
measured requests; USD stays null when no measured cost is available. Model-arm
campaign wall time spans that arm's first started to last finished attempt,
including gaps and interleaved work. Run wall time equals task time plus measured
backend waiting. Successful request latency always remains task time; queue
waiting is never estimated from token counts or generation rates.
The frozen policy is also recorded as the JSON string
`environment.transport_policy`, which survives canonical export and reporting.

Skill opening is directly observed by the `read_file` tool. A shell can perform
indirect reads; when that makes opening uncertain it remains unknown rather than
being guessed from a command string. Provider-native responses, tool calls,
complete command output and final answers are retained outside the worker mounts.

## Grade and report

Use an explicitly selected judge model, ideally different from the worker. The
judge receives the full transcript plus the original private rubric. Each
non-null assessment must cite an exact transcript excerpt. Assessment is not a
substitute for independent executable verification.

```bash
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py grade /tmp/cudaq-campaign \
  --models nemotron-ultra --timeout 180
python3 skills/cudaq-algorithms-dev/scripts/run_eval.py export /tmp/cudaq-campaign \
  --report /tmp/cudaq-report
```

Responses and original judge provenance are retained before parsing. Invalid
responses fail visibly and can be revalidated on the next grading invocation;
they are not silently replaced by another judgment. Grades remain bound to exact
result/transcript hashes.

`case_contracts.json` identifies 24 executable-eligible cases: 20 scientific cases
and four executable regression cases. All have registered private checks in
`checkers/`. Numerical references use independent NumPy/SciPy/PySCF calculations;
the four regression probes execute the repaired application or kernel.

The original user question is unchanged. Both arms receive the same additional
system instructions describing a runnable artifact and its output fields. Science
tasks save `evaluation.py`, whose last stdout line is a JSON object in the declared
format. Regression tasks retain their specified application filename; the generic
Bell task exposes a zero-argument `bell` kernel in `evaluation.py`. Neither arm
receives reference answers, private checker source or original grading assertions.

After the final answer, the harness executes that artifact inside the worker
sandbox, then sends only its bounded JSON output to a separate clean reference
process. The check shares the task's remaining budget. Verification output is
limited to 16 MiB; full logs up to termination remain evidence. Scope-restricted
client/example tasks also check file inventories against the initial workspace.

This measures **numerical artifact correctness**. The original rubric separately
assesses substantive library use, circuit provenance, resource-count assumptions,
scientific interpretation and consistency with the final answer. A numerical pass
alone is not a full rubric pass or evidence of a hardware execution.

The policy is `after_final_answer_v1`: exactly one independent verification is
performed per answered applicable attempt. Its completion costs are measured at
the check, including artifact execution and reference computation. Earlier worker
self-tests remain in transcripts; they are not treated as trusted verification.
Thus a successful first independent check has zero preceding independent failed
checks. We do not infer when the solution first became correct.

Missing/malformed artifacts or incorrect outputs fail verification. A budget
exhausted before comparison, unavailable checker dependencies or an internal
checker failure leaves verification `not_run`, with the reason retained.
Unanswered attempts also remain `not_run`. Dependency checks happen before model
requests. A completed check writes `verification.json` beside `result.json`:

- `status`: `passed`, `failed` or `not_run`;
- `result_sha256` and `transcript_sha256` binding the checked observation;
- for a performed check, `evidence`: a nonempty campaign-relative check-log path,
  and `evidence_sha256` binding that log;
- `verified_completion`: the canonical measured completion object when passed,
  otherwise null. Do not invent first-completion costs from the final total.

The exporter validates those bindings, original assertion positions, evidence
files and complete schedule, then calls the existing `report_eval.py` validator.
Only complete full campaigns can produce the canonical result bundle and its
two tables per model. A pilot is explicitly diagnostic and cannot be relabelled
as the delivery evaluation. Valid bundles may still contain failed or unassessed
targets; schema validity is not a scientific pass.

## CI and maintenance

`.github/workflows/skill_evaluation.yaml` runs offline suite-integrity, coverage,
runner, grading and reporting tests on relevant PRs. It also supports manual and
reusable invocation. It requires no model credentials and sends no data to model
providers. Real model execution remains an explicit local or separately configured
CI campaign with the same CLI, credential references and external evidence directory.

```bash
python3 -B -m unittest discover -s skills/cudaq-algorithms-dev/evals/tests
python3 -B -m pytest -q -p no:cacheprovider skills/cudaq-algorithms-dev/evals/runner/tests
python3 -B -m pytest -q -p no:cacheprovider skills/cudaq-algorithms-dev/scripts/tests
```

Opt-in tests exercise actual Linux isolation, CUDA-Q kernels, repaired regression
artifacts and a Pauli-LCU scientific calculation. Set `CUDAQ_RUNNER_TEST_ISOLATION=1` and
`CUDAQ_RUNNER_TEST_PYTHON=/path/to/venv/bin/python` on a capable runner. The trusted
executor mode used by offline lifecycle tests is deliberately absent from the
production CLI. Scientific model outcomes require real campaigns; offline tests
do not establish them.

The implementation consolidates staging, process deadlines, usage handling,
evidence validation and namespace isolation lessons from the historical
`behavioral/`, `e2e/`, `strategy/` and v18.2 runners. It has no dependency on their
temporary paths, old schedules, reset commands or report generators.
