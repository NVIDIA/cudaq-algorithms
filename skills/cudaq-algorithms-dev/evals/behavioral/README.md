# Native behavioral completion suite

This is a separately labeled native Codex benchmark for the 42 unchanged
authored behavioral cases plus three separately scored authoring cases. It is
not NVIDIA SkillEvaluator and does not use `config.yml`.

The candidate arm receives a frozen, task-local prompt catalog; the baseline
does not. Catalog exposure does not force a skill read. Read/activation
outcomes are command-output diagnostics and are not shared semantic score
requirements. Grading masks explicit arm identifiers and staging paths, but
treatment can still be inferred from preserved answer and read evidence.

Workers receive only declared fixtures. `input/` and candidate guidance are
sandbox read-only and `.tmp/` is scratch. The sandbox permits task workspace
writes, but only listed output paths are authorized; before/after inventory and
bounded artifact checks fail every other authored or nonregular change.
Authoring cases receive incomplete authoring/policy/conventions guidance
without completed operational record exemplars. A clean, explicit stdlib-only
worker venv is required; CUDA-Q Python packages must be absent from it. This
does not prove the machine lacks a core CUDA-Q C++ installation. The recorded
campaign exposed that toolchain; see [results and limits](../results/behavioral-20260913-v2/ANALYSIS.md).

## Controller commands

The commands below preserve the historical combined application/authoring
protocol. Supply `--skill-root` explicitly: use a preserved combined skill
snapshot with `--include-authoring`. For an application-only comparison of the
split layout, omit `--include-authoring` and pass
`--skill-root skills/cudaq-algorithms`; the old default now points to the
development directory and is not an application skill. Authoring-suite staging
for the new sibling layout is not part of the pruning pilot.

Run the controller itself with the established evaluation interpreter. Set
`WORKER_PYTHON` to a disposable venv created with `python3 -m venv --without-pip`.
Preparation rejects a worker interpreter that can discover `cudaq` or
`cudaq_algorithms`.

```bash
CONTROLLER=/path/to/e2e-runtime/venv/bin/python
WORKER_PYTHON=/path/to/behavioral-runtime/venv/bin/python
RUN_DIR=/path/to/behavioral-campaign

PYTHONPATH=skills/cudaq-algorithms-dev/evals/behavioral "$CONTROLLER" \
  skills/cudaq-algorithms-dev/evals/behavioral/behavioral_run.py prepare \
  --output "$RUN_DIR" --include-authoring --python "$WORKER_PYTHON"
PYTHONPATH=skills/cudaq-algorithms-dev/evals/behavioral "$CONTROLLER" \
  skills/cudaq-algorithms-dev/evals/behavioral/behavioral_run.py preflight \
  --output "$RUN_DIR" --include-authoring --python "$WORKER_PYTHON"
PYTHONPATH=skills/cudaq-algorithms-dev/evals/behavioral "$CONTROLLER" \
  skills/cudaq-algorithms-dev/evals/behavioral/behavioral_run.py run \
  --output "$RUN_DIR" --include-authoring --python "$WORKER_PYTHON"
PYTHONPATH=skills/cudaq-algorithms-dev/evals/behavioral "$CONTROLLER" \
  skills/cudaq-algorithms-dev/evals/behavioral/behavioral_run.py grade \
  --output "$RUN_DIR" --include-authoring --python "$WORKER_PYTHON"
PYTHONPATH=skills/cudaq-algorithms-dev/evals/behavioral "$CONTROLLER" \
  skills/cudaq-algorithms-dev/evals/behavioral/behavioral_run.py report \
  --output "$RUN_DIR" --include-authoring --python "$WORKER_PYTHON"
```

Preparation fixes three repetitions, rotating serial arms, no stop-on-pass,
480-second worker/grader limits, and eight parallel pair/grader slots by
default. Worker and grader token/time telemetry are retained and reported
separately. Started or partial attempts are evidence, never silently retried.

## Evaluator regression

From the repository root, use the same controller and clean worker paths as
above. Both variables below are required: the module path enables collection,
and the worker path enables the real runtime/isolation probes instead of
skipping them.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=skills/cudaq-algorithms-dev/evals/behavioral \
BEHAVIORAL_WORKER_PYTHON="$WORKER_PYTHON" \
"$CONTROLLER" -B -m pytest -q -p no:cacheprovider \
  skills/cudaq-algorithms-dev/evals/behavioral/tests
```

The native sandbox probes require an environment that can start the Codex
child sandbox. Controller launch permission does not relax its tested child
filesystem or network restrictions.

## Separate corrected grading-only protocol

`behavioral_regrade.py` reuses all 270 workers from the completed original
campaign. It never repairs/reruns workers or overwrites original grades. The
same assertions and bounded evidence content receive explicit stable citation
IDs; blank entries remain preserved but cannot be cited. Unknown citations,
missing verdicts, timeouts and incomplete execution are measurement failures.

Use a new, distinct output directory, after every original grader has
finalized. Preparation freezes all consumed source files, initial/final
artifacts, projected evidence, assertions, helpers and runtime provenance.

```bash
SOURCE_RUN=skills/cudaq-algorithms-dev/evals/results/behavioral-20260913-v2
REGRADING_RUN=skills/cudaq-algorithms-dev/evals/results/behavioral-regrade-20260913

"$CONTROLLER" -B skills/cudaq-algorithms-dev/evals/behavioral/behavioral_regrade.py \
  prepare --source "$SOURCE_RUN" --output "$REGRADING_RUN" \
  --python "$WORKER_PYTHON"
"$CONTROLLER" -B skills/cudaq-algorithms-dev/evals/behavioral/behavioral_regrade.py \
  run --output "$REGRADING_RUN" --python "$WORKER_PYTHON"
"$CONTROLLER" -B skills/cudaq-algorithms-dev/evals/behavioral/behavioral_regrade.py \
  report --output "$REGRADING_RUN" --python "$WORKER_PYTHON"
```

The fixed pass uses at most eight graders and 480 seconds each, with no retry.
Reservations, launch requests, observed native attempts, native completion and
valid grades are distinct. Interrupted attempts retain available native usage;
an event timestamp without a process summary is only an elapsed-time lower
bound. Worker KPIs are imported unchanged, original grader costs stay in the
source campaign, and corrected grader costs are separate.

This is a grader sensitivity study, not an independent worker replication or
a human-gold quality score. Valid evidence IDs prevent a citation-format
failure; they do not establish consistent interpretation or correct judgment.

## Selective discovery comparison

`selective_discovery.py` compares frozen `current` and `keywords` skill copies
on an explicit subset of cases. Both arms expose a catalog without forcing a
read. It does not edit `evals.json`, rewrite the skill, grade scientific
correctness, or test native IDE installation/discovery.

The input JSON supplies frozen case dictionaries, `expected_activation`,
`expected_records` relative to the skill root, the two snapshot paths, and an
allowlist of changed files. `activation_required: false` and
`routing_required: false` identify fixture-backed cases where skill reads are
optional diagnostics. Controls are three repetitions, GPT-5.5 low, a fixed
seed, at most four concurrent pairs, and a 180-second timeout. Failures are
retained without retry. Use a new output directory for every comparison:

```bash
SELECTIVE_INPUT=skills/cudaq-algorithms-dev/evals/results/keyword-discovery-20260914/experiment.json
SELECTIVE_OUTPUT=skills/cudaq-algorithms-dev/evals/results/keyword-discovery-repeat
SELECTIVE_RUNNER=skills/cudaq-algorithms-dev/evals/behavioral/selective_discovery.py

"$CONTROLLER" -B "$SELECTIVE_RUNNER" prepare --input "$SELECTIVE_INPUT" \
  --output "$SELECTIVE_OUTPUT" --python "$WORKER_PYTHON"
"$CONTROLLER" -B "$SELECTIVE_RUNNER" preflight \
  --output "$SELECTIVE_OUTPUT" --python "$WORKER_PYTHON"
"$CONTROLLER" -B "$SELECTIVE_RUNNER" run \
  --output "$SELECTIVE_OUTPUT" --python "$WORKER_PYTHON"
"$CONTROLLER" -B "$SELECTIVE_RUNNER" report --output "$SELECTIVE_OUTPUT"
```

Read-time/order observations come from controller-received command events,
not internal reasoning or causal attribution. Native totals describe the
whole attempt; per-file billed tokens and exact context occupancy are not
available. Review final answers/artifacts separately before promoting wording.

If a read-observation parser defect is found after a completed comparison,
reprocess its raw events into a new directory without rerunning workers or
overwriting the original summaries:

```bash
"$CONTROLLER" -B "$SELECTIVE_RUNNER" reanalyze --output "$SELECTIVE_OUTPUT" \
  --destination "$SELECTIVE_OUTPUT/parser-reanalysis-v2"
```

The reanalysis retains original native usage and elapsed times, includes old
and corrected observations, and hashes every consumed result/event file plus
the manifest and both parser versions. It is still not a correctness grader.
