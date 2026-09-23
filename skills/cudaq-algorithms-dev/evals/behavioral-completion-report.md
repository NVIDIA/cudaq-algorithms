# Behavioral and authoring evaluator completion

Date: 2026-09-13

## Outcome

A separate native suite now loads all 42 unchanged authored cases and adds two
bounded record-addition tasks plus one no-source roadmap advisory task. The
frozen default plan is 45 cases × two arms × three fresh repetitions = 270
worker attempts, followed by one fresh grader attempt per completed worker
attempt. No scored model campaign was run while implementing the controller;
campaign execution and final review remain root-owned.

The treatment is prompt-catalog discovery: candidate workers can see the
frozen skill name, description, and local path, while baseline workers see an
empty catalog. Neither arm is told to force a read. Explicit arm identifiers
and staging paths are masked from graders, but scientifically relevant answer,
citation, and file-read evidence is preserved, so treatment may be inferred
from content. This is not a claim of full blinding or native automatic skill
activation.

## Contracts and adaptations

- Original `evals.json` bytes, prompts, fixtures, expected outputs, and all 220
  authored assertions remain unchanged. Adaptations live only in
  `behavioral/adaptations.json` and are frozen before attempts.
- Forty-three read/activation checks have a separate command-evidence
  denominator. Four combined assertions are explicitly split: treatment-only
  skill reading stays diagnostic, while ambiguity recognition, unavailable
  contract handling, and shared fixture inspection remain semantic outcomes.
  Only the applicable `id`/`text` projection reaches the grader.
- Fixture-free tasks receive no package source, tests, runtime, or hidden
  evidence. The worker venv must not discover/import `cudaq` or
  `cudaq_algorithms`; the preflight checks that premise inside both sandboxes.
- The three implementation cases have explicit required root targets; the QPU
  case has an optional local `app.py`; all other original cases are no-write.
  Authoring targets are disposable `target_skill/` files in both arms.
- Authoring candidate guidance is an explicitly incomplete read-only subset:
  authoring policy/templates, coverage policy, and shared convention/source/
  validation guidance. Completed operational records, coverage history, and
  the feature registry are absent, preventing answer leakage.
- Operational authoring records must not contain lifecycle/run-history stamps.
  Honest execution and lifecycle statements are graded from the final answer,
  not required inside the resulting record.

## Grading and execution boundaries

Shared semantic assertions are independently graded `PASS`, `FAIL`, or
`UNCLEAR` with exact numbered evidence citations. Invalid schemas, missing or
duplicate assertions, unsupported verdicts, bad citations, and uncertainty do
not pass. Overall pass also requires authorized scope, bounded regular/parseable
artifacts or resolving local authoring routes, a successful complete worker
native turn, and a successful complete grader native turn.

Code artifact checks deliberately do not execute untrusted generated Python in
the controller. They check bounded regular-file syntax; equivalent valid forms
remain eligible for semantic grading. The staged QSVT focused test does not
execute the client, and CUDA-Q/package/scientific runtime is absent. Therefore
the suite provides behavioral artifact evidence, not scientific package
validation. Execution claims require matching successful command evidence.

Workers run serially within rotating paired blocks and blocks run in bounded
parallel. Graders start only after all paired worker results exist, run in a
separate bounded pool, and never feed results back to workers. Worker/grader
exceptions finalize infrastructure evidence and allow other attempts to
continue. Existing partial worker/grader directories are finalized without a
retry; available telemetry and files are retained.

## Test-first evidence

Representative red observations included missing modules before implementation,
over-permissive semantic AST/token gates rejecting equivalent code/records,
missing mixed read kinds, authoring staging that omitted required policy or
leaked broad references, invalid lifecycle requirements, resource pairing on
missing counters, nested output paths, interrupted attempt bookkeeping, and a
provenance comparison polluted by an environment-dependent CLI warning.

Final controller test command:

```bash
BEHAVIORAL_WORKER_PYTHON=/tmp/cudaq-behavioral-runtime.K8q5lC/venv/bin/python \
PYTHONPATH=skills/cudaq-algorithms/evals/behavioral:skills/cudaq-algorithms/evals/e2e \
  /tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python -m pytest -q \
  -p no:cacheprovider skills/cudaq-algorithms/evals/behavioral/tests
```

Result after the relative-output regression: **39 passed in 0.92 seconds**.
The suite includes real nested-sandbox tests; they must run outside an already
restrictive outer sandbox. Those
runtime-adapter tests cover clean PATH selection plus sibling, evaluator,
controller/global package-root, fixture-write, and network denial.

A fresh 270-attempt manifest was then prepared without model calls and its
two-arm strict preflight passed. Both arms proved evaluator/manifest denial,
`cudaq`/`cudaq_algorithms` non-discovery and import failure, fixture overwrite
and chmod denial, scratch write access, and network denial. The sibling and
clean-PATH probes belong to the runtime-adapter test result above, not this
campaign preflight result.

## Campaign commands and reporting

Use the five commands in `behavioral/README.md` with an explicit disposable
stdlib-only worker venv. `prepare` freezes the candidate bytes, original and
adapted contracts, exact fixtures, local evaluator modules including imported
`e2e` dependencies, runtime/CLI provenance, budgets, and shuffled plan.
`preflight` must pass before `run`; `grade` requires every planned worker result;
`report` keeps original/authoring workflows, cases, assertions, read diagnostics,
failures, observed input peaks, paired worker counters, and grader counters in
separate denominators.

Observed request-input peaks are native observed counters with completeness
metadata, not full context-window occupancy. Missing counters remain missing;
grader costs are never added to worker KPIs, and no dollar estimate is inferred.

## Pre-attempt relative-output correction

The first repository-local prepared campaign used a relative `--output` path.
Its preflight therefore tested a relative, nonexistent manifest path, and the
same root would have produced worker/grader `-o` paths inside their workspaces.
No model attempt started. That zero-attempt harness artifact and its status note
remain untouched and are not valid scoring evidence.

Before the v2 freeze, every callable campaign entrypoint and the CLI boundary
were changed to canonicalize the output root with `Path.resolve()`. TDD
regressions observed both relative-path failures, then proved an absolute
prepared/run root, absolute preflight manifest denial, and absolute worker
final-answer path. The fresh v2 manifest is expected to contain 45 cases, 270
runs, 19 fixture hashes, 18 evaluator hashes (seven local suite files plus 11
imported `e2e` Python dependencies), and three source-contract hashes.
