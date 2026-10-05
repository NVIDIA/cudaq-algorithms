# Delivery evaluation

[`evals.json`](evals.json) is the single maintained suite: **62 unchanged cases**,
comprising 42 regression and 20 science cases. It has 59 positive skill cases
and 3 negative controls. [`manifest.json`](manifest.json) pins the original
case order, complete case records, source provenance and fixture hashes.
Intentional future changes require explicit manifest review and rebaselining.
Old 42- or 7-case measurements are not results for this delivery suite.

## Paired campaign protocol

Use the canonical suite, [`config.yml`](config.yml), declared `files/` fixtures,
and runtime skill [`../../cudaq-algorithms`](../../cudaq-algorithms). An external
runner must execute every case with and without the skill, for **exactly five
independent paired repetitions**: 62 × 5 × 2 = 620 records per model. Include
records for failures, unanswered runs and ungraded runs. Never stop on pass,
retry until a pass, or select the best attempt. A seed identifies a pair; it
does not promise deterministic model-provider behavior.

The harness configuration uses `n_attempts: 1`, `stop_on_pass: false`, and the
unchanged `pass_threshold: 0.50`. The separate `evaluation_protocol` section is
reporting metadata, **not a supported harness seed-loop feature**. The runner
must loop `[0, 1, 2, 3, 4]` itself, select budgets before running, and copy the
actual protocol into the results. Do not pass custom reporting metadata to a
harness unless its configuration adapter supports it.

A full campaign includes a frontier model and a mid-tier model. A single-model
bundle is valid and visibly reports missing tier coverage. Keep each model's
results separate. Use isolated throwaway worktrees for implementation tasks.
Both arms receive the same prompts, scientific inputs, repository/library
access, dependencies, tool limits and time budgets; only skill availability
changes. Keep assertions, expected outputs and other answer material out of
worker inputs. Record whether the skill was merely `listed` or explicitly
`injected`; observed opening after injection is not natural trigger recall.

Before execution, register one `case_contracts` entry for each canonical ID:
`executable_check`, `implementation`, and a nonempty `rationale` describing the
check and applicability. Use identical contracts across arms and models.
Implementation cases must have an executable check. Other checks apply only
where the authored task warrants them; advice and predictions do not acquire
an extra universal simulator or hardware requirement. In science cases, grade
substantive CUDA-Q Algorithms use against the original rubric; an unused
import is insufficient, and permitted independent classical references remain
valid.

For applicable scientific checks, preregister the oracle, inputs, tolerance,
target and precision. Check intermediate stages and pin the full statevector
when needed to distinguish register order, phase, normalization or partial
state errors. Follow the requested scientific quantity and the source-grounded
[validation guidance](../../cudaq-algorithms/references/validation.md). Preserve
full transcripts, executable commands, artifacts, numerical outputs and check
logs, including failed attempts. Classify the five tripwires from transcript
evidence: controlled measurement, sample feedback, kernel definitions in a
heredoc, unnecessary kernel reminting in loops, and misuse of partial
statevectors. Assess their meaning against the task and inspected source;
do not count an intentional demonstration or a valid requested operation as
a failure merely because a keyword appears.

The three existing negative cases are the control subgroup. They are not
necessarily unrelated maintenance controls from a broader evaluation guide;
do not change the 62 cases or claim coverage they do not provide.

## Result bundle and evidence

[`results.schema.json`](results.schema.json) defines the required JSON shape
using JSON Schema draft 2020-12. All record fields are required and fixed
objects reject extra fields. `null` records missing evidence; it is not a
passing grade, a zero count, or free resource usage.

Top-level fields are `schema_version: 1`, `suite_sha256` (SHA-256 of the raw
`evals.json` bytes), `source_revision`, `skill_revision`, `skill_label`,
`environment`, `protocol`, `case_contracts`, and `models`. Store immutable
source/skill revisions and record CUDA-Q, CUDA-Q Algorithms, Python,
dependencies, simulator and precision as string-valued environment entries.
The protocol requires five distinct integer `seeds`, positive `budget_seconds`,
a positive integer or null `tool_call_limit`, `skill_exposure`, and nullable
`time_regression_limit`/`token_regression_limit` ratios. Each stated tolerance
must be at least 1; null means the regression target was not assessed.

Each model has a unique `id`, a `tier` of `frontier` or `mid`, independently
measured `wall_seconds` for each arm, `notes` for regression/science/controls,
and `runs`. Notes should identify task IDs and evidence showing where the
skill changed the work or misled the model. Each run contains:

- `case_id`, `seed`, `arm` (`baseline` or `skill`), and `outcome`: `answered`,
  `budget_timeout`, `tool_limit`, `backend_error`, `error`, or `no_answer`.
- `skill_opened` (baseline must be null), one boolean/null per original
  `assertions` entry in order, and boolean/null `expected_output` and
  `claimed_success`. Unanswered outcomes cannot have true assertions,
  expected output or claimed success; answered runs may still be ungraded.
- `verification`: `passed`, `failed`, `not_run`, or `not_applicable`.
  `not_applicable` is mandatory exactly when the registered case has no
  executable check. `verification_evidence` is a relative path to an existing,
  nonempty check-log file inside the bundle directory for passed/failed checks
  and null otherwise.
- `transcript`: a relative path to an existing, nonempty file inside the bundle
  directory. `grading_evidence` follows the same rule and is required when
  any assertion, expected output, critical failure, tripwire or executable
  check has been assessed. These paths must still resolve inside the bundle
  after resolving symlinks. Save full transcripts and grading/check logs;
  references may share a file when it contains the relevant evidence.
- Nullable nonnegative `critical_failures` and the five `tripwires` counts:
  `controlled_measurement`, `sample_feedback`, `heredoc_kernel`,
  `remint_in_loop`, `partial_statevector`. Zero requires an actual transcript
  assessment; null means unknown.
- `resources`: nonnegative `task_seconds`, `wall_seconds`,
  `backend_wait_seconds`, nullable nonnegative integer `tokens`/`tool_calls`,
  and nullable nonnegative measured `cost_usd`. Run wall time equals task
  time plus backend waiting within floating tolerance. Record total token
  usage consistently, including skill reading and unsuccessful work.
- `verified_completion`: present as an object exactly when verification
  passed, otherwise null. Its `task_seconds`, `wall_seconds`, nullable
  `tokens`/`tool_calls`, and integer `failed_attempts` describe cumulative
  work through the first successful executable check, including earlier
  failed attempts within that run. Completion costs cannot exceed known
  corresponding run totals; completion wall time cannot be below task time.
  Waiting accumulated before completion cannot exceed the entire run's wait.

The reporter rejects duplicate JSON keys, nonfinite numbers, wrong suite
hashes, missing/extra/duplicate case-seed-arm records, wrong assertion counts,
unsafe, absent or empty evidence files, and inconsistent evidence/resource fields.
Each arm's measured campaign wall time must cover its longest run. Campaign
wall time is not the sum of task times; concurrent tasks and backend waits can
overlap. Schema and reporter checks enforce recording and reporting contracts;
they do not establish scientific truth or execute the campaign.

## Generate the report

Run from the repository root, after collecting real evidence:

```bash
python3 -m pip install jsonschema
python3 skills/cudaq-algorithms-dev/scripts/report_eval.py /path/to/campaign/results.json --validate-only
python3 skills/cudaq-algorithms-dev/scripts/report_eval.py /path/to/campaign/results.json --output /tmp/cudaq-eval-report
```

Invalid input exits with status 2 before creating output. The output directory
must be absent or empty. Keep both the bundle and generated reports outside
the runtime and development skill trees. The reporter does not run models
or invent missing measurements. Its tests use
synthetic records only.

An evaluation handoff is incomplete until this command succeeds and its
generated reports, details and source evidence are supplied. A valid report
may contain failed or unassessed targets; validation is not a claim that
delivery targets were met. Regenerate the Markdown from the JSON instead of
editing calculated cells by hand.

The output `index.md` links one `model-01/`, `model-02/`, … directory per model.
Each `report.md` contains **exactly two Markdown tables** for the full suite:

1. Quality and target results: `Metric | Baseline (no skill) | With skill
   (<label>) | Target | Status`. The original eleven rows cover negative
   controls, positive opening, assertion score, strict pass, expected output,
   uplift, implementation checks, critical failures, no final answer, median
   task time and median tokens. Additional rows expose verified pass, silent
   wrongness, unverified success claims, tripwires and budget timeouts.
2. Effort and completion evidence: `Metric | Baseline (no skill) | With skill`. Include
   judged count; median/mean/total task time; median/total tokens and tool calls;
   measured campaign wall time; cumulative backend waits (overlaps possible);
   no-answer causes; measured USD; per-type tripwire counts; verified
   completion counts and median task/wall time, tokens, tool calls and failed
   attempts; and paired verified completion ratios with their sample counts.

There is no cross-model aggregate table. A link to each model's `details.json`
provides the entire validated model run data, campaign provenance, per-task
seed-paired comparisons, and regression/science/control summaries using the
same metrics, including control time/token ratios. Subgroups and task evidence
remain available without adding tables to the two-table model report.

## Metric definitions and targets

All rates use the complete selected case × seed population for the relevant
arm and applicability subset. Unknown or ungraded records are never removed
to improve a rate. Show known failures and unknowns separately; unknown
evidence cannot satisfy a target.

A fully judged observation has all original assertions and its expected-output
rubric assessed. This count is separate from executable verification.

- **Negative controls staying closed:** skill-arm `skill_opened: false` over
  all negative-control pairs. This is not conventional precision. **Positive
  opening:** `skill_opened: true` over all positive-case pairs. Each target is
  at least 90%. Baseline activation is N/A; unobserved opening remains unknown.
- **Mean assertion score:** average of each case-run's fraction of original
  assertions satisfied, multiplied by 10; each case-run has equal weight.
  Target at least 8.5/10. If any assertions are unknown, show not assessed
  rather than quietly substituting zero or averaging only graded runs.
- **Strict pass:** answered and every original assertion true. **Expected
  output:** original expected-output rubric satisfied. Each target is at
  least 85%. Strict pass is separate from harness `pass_threshold: 0.50`.
  **Uplift:** report skill minus baseline in percentage points for both rates;
  the at-least-15-point target is met only when both qualify. Some unchanged
  assertions and expected-output rubrics require using the skill, so either
  uplift can include treatment compliance. Neither alone establishes a gain
  in scientific correctness; executable checks and paired costs are separate
  evidence.
- **Verified pass:** passed executable checks over all executable-eligible
  pairs; not-run checks remain unknown. **Implementation passing:** the same
  measure restricted to registered implementation cases; target 100%. A
  failed executable artifact stays a verified failure even when its rubric
  assertions pass.
- **Silent wrongness:** `claimed_success: true` with verification `failed`,
  reported as a count over the full executable-eligible denominator; target
  zero. Report success claims without a run check separately as unverified,
  retaining unknown claim/check evidence instead of calling it correct.
- **Critical failures** and **tripwires:** report occurrence totals and
  per-tripwire counts; targets zero. Missing assessments mean incomplete
  coverage, not a zero-incident pass. Retain known incident counts even when
  other assessments are missing; a known incident fails the zero target.
- **No final answer:** every outcome other than `answered`; target zero.
  **Budget timeouts** are a separate subset, also with target zero. Report
  other no-answer causes rather than absorbing them into timeouts.
- **Time and tokens:** compare skill/baseline full-population medians against
  the preregistered regression limits. Null tolerance means not assessed;
  do not invent a numerical meaning for “no material regression.” Unknown
  token measurements make aggregate token costs unavailable. A zero baseline
  makes a ratio undefined (a displayed `0 → 0` can describe no change).

Completion medians apply to verified completions only. Paired completion
ratios are medians of per-pair skill/baseline ratios for pairs where both arms
passed, not ratios of aggregate medians. Display both the eligible pair count
and usable ratio count; unknown and zero-baseline costs must not be silently
dropped. Show failed and unanswered counts alongside these conditional
comparisons to expose survivor selection. Null USD means unavailable; no
prices are inferred.

## Local validation and maintenance

```bash
python3 -B -m unittest discover -s skills/cudaq-algorithms-dev/evals/tests
python3 -B skills/cudaq-algorithms-dev/scripts/check_coverage.py
python3 -B -m pytest -q -p no:cacheprovider skills/cudaq-algorithms-dev/scripts/tests
```

These checks validate packaging, integrity, routing and reporting; they do
not establish model-evaluation targets. Preserve suite/source/skill revisions,
environment, evidence and outputs with each campaign. Repeat paired model
evaluation after model updates and quarterly. A small golden-path CI check
and pointer/link checks are recommended maintenance policy, not a claim that
scheduled jobs are installed.

An optional ablation is a separate, clearly labelled campaign: hold cases,
models, budgets and seeds fixed, identify the removed skill component, and
use its own skill label/revision and bundle. Do not mix ablations into the
no-skill baseline or select ablations after inspecting favorable scores.
