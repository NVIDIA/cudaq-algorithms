# cudaq-algorithms-dev

This folder is the development companion of the `cudaq-algorithms` agent skill in
[`../cudaq-algorithms/`](../cudaq-algorithms/). That skill (`SKILL.md`, `references/`,
`scripts/route.py`) is what an agent loads when it designs, implements, debugs, reviews
or composes `cudaq_algorithms` APIs and fault-tolerant primitives. Nothing here is loaded
by an agent or staged into that skill. This folder holds what maintainers use to author
skill records, track coverage evidence and evaluate the skill.

It is licensed under Apache-2.0, like the rest of the repository.

## Folder layout

| Folder | What it holds |
| --- | --- |
| `authoring/` | The authoring method: architecture, record design, capability design, source review, extension workflow and open design decisions. `templates/` has skeletons for primitive, capability, representation and convention records. |
| `coverage/` | `features.json`, the registry of the 52 operation/object contracts and their evaluation and run-evidence mappings; `policy.md`, the rules for what counts as evidence; `history.md`, the review and migration audit trail. |
| `scripts/` | `check_coverage.py`, which checks Markdown links, feature mappings and scoped run evidence without calling a model, plus its tests. |
| `evals/` | `EVAL.md`, the seven active cases in `evals.json`, the 42 preserved cases in `regression-evals.json`, `config.yml`, fixtures in `files/`, schema tests in `tests/`, the previous evaluation notes in `archive/`, and the task briefs and completion reports of past evaluation work. |
| `evals/behavioral/` | A native Codex suite for the authored behavioral and authoring cases: controller, case staging, execution, grading, regrading, selective discovery, a pruning pilot and tests. Read its `README.md` first. |
| `evals/e2e/` | An end-to-end benchmark in which fresh Codex sessions write and run applications against the real package, with and without the skill: controllers, case definitions, independent numerical references, grading, telemetry, comparison, reports and tests. Read its `README.md` first. |
| `evals/strategy/` | A strategy evaluation built on the SkillEvaluator harness: brief, protocol, implementation plan, capture, judging, post-hoc checks, reporting, numerical checkers in `checks/` and tests. |

## What is not in the repository

Three things were left out on purpose. The run evidence under `evals/results/` is
ignored through the top-level `.gitignore`; result trees are hundreds of megabytes of
per-attempt captures. The machine-bound experiment controllers that imported helpers
from temporary directories were removed. A local bridge module that depended on a
package not available publicly was removed as well. Several historical documents in
this folder still cite paths under `evals/results/`. Those paths refer to unpublished
evidence, and the text around them describes the state at the time it was written.

## Environment variables

Scripts take machine-specific paths from the environment instead of carrying them in
source. Each variable is optional unless the script's own preflight says otherwise.

- `CUDAQ_E2E_PYTHON`: interpreter with CUDA-Q and the package installed. Default worker interpreter for `evals/behavioral/behavioral_run.py` and `evals/behavioral/pruning_pilot.py`; also enables the opt-in real-runtime tests under `evals/e2e/tests/`.
- `CUDAQ_E2E_CODEX`: path to the Codex CLI executable used by `evals/e2e/runtime.py`; falls back to `codex` on `PATH`.
- `CUDAQ_PRUNING_OLD_EVALS`: directory holding the preserved previous evaluation tree from which `evals/behavioral/pruning_pilot.py` reads its helper modules.
- `CUDAQ_PRUNING_E03_RUNNER`: path to the preserved paired E03 runner script used by the pruning pilot.
- `CUDAQ_PRUNING_E03_CHECKER`: path to the native E03 checker script used by the pruning pilot.
- `BEHAVIORAL_WORKER_PYTHON` (pre-existing): clean, stdlib-only worker interpreter that enables the isolation probes in `evals/behavioral/tests/`.

The pruning pilot checks the SHA-256 of every external dependency at preflight and
reports a missing one instead of guessing. Further opt-in variables for the end-to-end
suite are listed in `evals/e2e/README.md`.

## Running the tests

From the repository root, run the tests that need no model, no Codex installation and
no run evidence:

```bash
PYTHONPATH=python python3 -m pytest -q \
  skills/cudaq-algorithms-dev/scripts \
  skills/cudaq-algorithms-dev/evals/tests \
  skills/cudaq-algorithms-dev/evals/strategy/tests
```

Two caveats. The strategy tests for judging, live runs, post-hoc processing and
reporting import the `skillevaluator` package (two of them also `openai`) when they run;
without those packages they fail rather than skip. `scripts/check_coverage.py` reads the
historical metadata inventory that `coverage/features.json` points to under
`evals/results/`, so a default run stops at that missing file when the evidence tree is
absent.

The behavioral and end-to-end suites need a Codex installation and clean, dedicated
interpreters. Their own `README.md` files describe the prepare, preflight, run and
report steps and the isolation rules.
