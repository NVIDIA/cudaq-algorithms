# Strategy evaluation: execution contract

## Current checkpoint — 2026-09-16

The [E08 captured-artifact numerical integration](../results/strategy-20260916-e08-integration/REPORT.md)
registers a separate 12-case worker-targeted suite and connects source/binding
freezes through capture and post-hoc verification. Four native valid controls
pass all 12 tests; seven broken controls fail and four unsupported/missing
controls remain unknown. Independent reviews approved all three components.
Fresh controller and mathematical verification covers 866 harness tests across
three runs; seven initial tokenizer skips were resolved using the existing
pinned assets. E02/E03/E05/E08 now have registered numerical suites. No model,
worker, KPI, original-regression or acceptance pass is established. Actual
example documentation and entrypoint/independent-validator review remain
separate obligations; synthetic fixtures do not satisfy them.

The [E08 binding/runtime checkpoint](../results/strategy-20260916-e08-binding/REPORT.md)
adds explicit artifact bindings, actual public-QSVT tracing, sampleability,
independent reconstruction checks and perturbation of the artifact's validator.
All 23 native calibration controls and 326 tests across eight related modules
pass. Independent review closed two adapter gaps and approved the runtime.
An earlier native failure remains preserved. These are checker controls, not
worker coverage, model success or efficiency measurements. E08 is still not
registered; captured-artifact/documentation binding, entrypoint validation use,
worker-targeted integration and original regression remain separate work.

The [E08 reference/native checkpoint](../results/strategy-20260916-e08-reference/REPORT.md)
adds a bounded independent dense-evolution oracle: 75 focused tests and 143
tests across four related mathematical modules pass. Actual public QSVT kernels
were sampled without SDK extraction during construction/execution, then checked
against independent dense references. Two supported real-domain controls agree
at roughly 2e-15; eight numerical/domain-limit controls are rejected. These are
calibration outputs, not worker results, and **E08 is not registered**. Explicit
artifact binding, the example's own self-validation, and full regression remain.
The outer tool sandbox prevented the initial isolation canary from starting;
an approved repeat outside it succeeded with the inner restrictions unchanged.
A separate approved private-proc capability probe still failed with mount EPERM,
so that scientific-regression prerequisite is not resolved. No model call,
operational-skill edit, policy relaxation or acceptance pass occurred.

The [E05 captured-artifact integration](../results/strategy-20260916-e05-integration/REPORT.md)
registers a separate 13-case targeted suite and binds it through numerical,
artifact and post-hoc verification. Four source-bound positive controls pass
all 13 cases; broken and unsupported controls produce failure and unknown as
appropriate. Independent review closed a stale staged-binding evidence gap.
The final twelve-module offline regression passes 561 tests. E02/E03/E05 now
have registered suites; this is not an E05 worker or model-quality pass.
The full frozen scientific regression remains unverified: a 120-second run
timed out, and a separate first-failure diagnostic confirms PySCF cannot read
its own `/proc/2/statm` in this native sandbox. Prior failed proc remedies were
not repeated. No threshold, model, permission or original-test requirement was
changed; the approved container/runtime route remains an infrastructure need.

The [E05 native-runtime checkpoint](../results/strategy-20260916-e05-runtime/REPORT.md)
implements and calibrates actual preparation, sampling and Walk-block checks:
17 native tests and 526 tests across twelve offline harness modules pass.
The compiled-operation guard inspects captured helpers and rejects measurement,
reset, state initialization and explicit noise. Internal allocation remains
unknown until an ancilla-layout contract is bound, not an automatic failure.
No model or worker was evaluated. That checkpoint's pending artifact binding
and registration are implemented by the subsequent integration above. Resource
evidence, broader ancilla contracts and original regression remain unfinished;
neither checkpoint is an acceptance result.

The [E05 reference/binding checkpoint](../results/strategy-20260916-e05-reference/REPORT.md)
adds an independent four-determinant state oracle and explicit API mapping,
with 515 passing tests across eleven offline harness modules. Two isolated
native CUDA-Q characterizations demonstrate why individually phase-equivalent
Slater states do not guarantee a correct coherent sum: the constructed negative
control has fidelity `4/9`. The first non-reproducing basis is also preserved.
These are checker-calibration inputs, not worker solutions. E05 is not yet
registered: the subsequent native calibration above supplies bounded
preparation/Walk checks; captured-artifact binding, broader contracts,
resources and full original regression remain to be verified.

The [offline reporting implementation](../results/strategy-20260916-verified-reporting/REPORT.md)
now joins saved current-contract judge records to freshly collected evidence
using the saved blinding mapping. It revalidates raw responses against current
independent checks/status, retains all missing/failed attempts, and derives the
diagnostic seed from a verified matching manifest. This establishes artifact
consistency, not judge reliability, pre-observation chronology or calibrated
confidence. No live model call or statistical acceptance pass occurred.

The [manifest-freeze repair](../results/strategy-20260916-manifest-freeze/REPORT.md)
rejects empty/malformed freeze records and exports explicit verified-versus-missing
local record status. That earlier eight-module regression passed 387 tests.
The missing saved-grade join identified at that checkpoint is implemented
above; matching files still do not establish chronology or calibrated confidence.

The [harness readiness audit](../results/strategy-20260916-harness-audit/REPORT.md)
identified an eight-attempt-only preparation gap. The subsequent
[full-schedule repair](../results/strategy-20260916-full-schedule/REPORT.md)
now supports all 100 disclosed attempts, with explicit per-case budgets and
verified case-to-task bindings. That seven-module offline regression passed
354 tests; no live campaign ran. Only E02/E03 have registered independent check
suites; six mutation cases do not. The
[E07 grading repair](../results/strategy-20260916-e07-gating/REPORT.md) implements
the already-approved no-implementation exception using verified unchanged
capture evidence. Changed or unbound captures remain unverified without
independent checks. Its six-module regression passed 315 tests, including
the fixture probe with its required UID-switch permission. This establishes
no agent-quality or full-campaign result. Actual campaign budgets/freezes,
checker coverage, primary grading and confidence readiness remain prerequisites,
not completed work.

The [Walk contract correction](../results/strategy-20260916-walk-readiness/REPORT.md)
qualifies control-off identity by `uncompute`: without uncomputation the ancillas
remain prepared. Four numerical counterexamples refute the old unqualified
statement; 16 independent branch characterizations plus nine existing
orchestration tests passed. Only that focused record changed; no product code
or public tests changed. This is not a model evaluation or acceptance result,
and earlier model runs do not cover the revised record.

The [40-trial workflow-verb microtest](../results/strategy-20260916-routing-verb/REPORT.md)
is complete: neither `Choose` nor `Open` produced an expected workflow-file
read. The one-word variant is not retained. This virtual read-only Qwen probe
has complete usage but no end-to-end efficacy or acceptance result. Some
focused references were read in both conditions; no workflow read alone is
not proof of a routing defect. That experiment did not change the operational skill.

The [local-native E03 pair](../results/strategy-20260916-local-e03/REPORT.md)
has finished. Neither arm solved the task: baseline failed 16 of 32 targeted
tests; candidate introduced an import-time syntax error. Exact candidate skill
delivery passed, but no reference-file reads occurred. Full regression remains
unverified in the local native environment. There is no successful matched
efficiency cohort, and the local Qwen diagnostic does not replace NVIDIA
acceptance. See the final checkpoint below for the preserved evidence.

### Earlier E01 routing checkpoint

The [native E01 routing trial](../results/strategy-20260916-app-e01-v4/REPORT.md)
has complete delivery and token accounting but an unfinished candidate answer.
Its lower usage is not a successful-task efficiency improvement. The five-line
routing change is preserved with that trial, not retained in the active skill.
The [preceding corrected pair](../results/strategy-20260916-app-e01-v3/REPORT.md)
performed real source reads but produced scientifically inadequate answers and
no quality-successful matched cohort. No acceptance criterion is established.
The subsequent harness change retained sanitized response diagnostics; it did
not change worker behavior, the parser, grading or any success threshold.

### Earlier local-delivery checkpoint

The [local E01 diagnostic](../results/strategy-20260916-local-e01/REPORT.md)
completed both workers with cross-checked token and timing evidence. The
candidate did not read the skill, and native delivery was not established.
These measurements are not a quality-verified skill comparison. A subsequent
[native delivery smoke](../results/strategy-20260916-native-skill-delivery/REPORT.md)
registered the existing `skills/` path and verified the exact full skill text
in one actual local Qwen request. Scientific task efficacy remains untested on
that route. App-server completion accounting has since been corrected and tested;
the next matched run must verify it against actual native events. The earlier
CLI and delivery-smoke results are unchanged.
The separate [confidence calibration proposal](CONFIDENCE_CALIBRATION.md)
has not been executed and does not alter acceptance or certify uncertainty.

Acceptance remains unverified. The [revised-skill pilot](../results/strategy-20260915-pilot03/REPORT.md)
finished with four completed trigger attempts and four interrupted execution
attempts; it establishes no successful matched execution comparison. Its
measurements and all earlier failures are preserved separately.

The [separate regression-staging recovery](../results/strategy-20260915-regression-recovery/REPORT.md)
checked both saved E03 captures with readable, byte-identical public source:
307 passed and four documented skips each. Original source/captures and prior
results remain unchanged. This resolves the regression infrastructure error,
not the 16 targeted failures in each interrupted artifact.

The [rate-limit repair and pinned-client check](RATE_LIMIT_REPAIR.md) establish
that forwarding HTTP 429 correctly does not make the client retry it. This
repair is not a provider-capacity solution and has not replaced the frozen
pilot runtime. Primary rubric grading remains suspended after false passes.

One configuration hypothesis remains untested: the local judge request uses
`temperature=0` and disables thinking, while the model's [official model card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard)
recommends `temperature=1.0` and `top_p=0.95`. This is not evidence that sampling
caused the false passes. Any future calibration must declare its settings
before running; no judge setting or historical grade is changed here.

The [E02 checker calibration](../results/strategy-20260915-e02-calibration/REPORT.md)
records 30 passing calibration tests. The subsequent
[positive-binding checks](../results/strategy-20260915-e02-positive-controls/REPORT.md)
record all 60 targeted tests passing separately for two valid normalization
policies. Neither is a worker acceptance result: the controls use evaluator-only
tomography. Captured-source binding and E02 registration now have local
integration tests; the revised path still needs pinned-runtime validation, including negative and
unknown-result controls. These readiness checks do not change any success
threshold below.

The separately authorized [local-model readiness probes](../results/strategy-20260915-local-inference-smoke/REPORT.md)
loaded a private Qwen model and exercised four bounded tool-call requests.
Direct Responses custom-tool semantics failed despite HTTP 200; translating
through the existing Chat conversion helpers preserved the requested custom
input and usage. The subsequent [local agent/tool smoke](../results/strategy-20260915-local-agent-smoke/REPORT.md)
verified file creation, successful tool readback and matching usage counters
after an explicit shell-PATH repair. The first partial attempt and its weak
test oracle are retained. This is diagnostic infrastructure only, not a replacement for the fixed NVIDIA
acceptance model or evidence of skill uplift. No NVIDIA evaluation calls were
made in these probes. The E02 negative-control upload remains stopped pending
explicit user approval required by the environment.

## Earlier pilot and grading-recovery history

The first readiness pilot finished with implementation failures and incomplete
rubric-grade coverage; not a completed full benchmark or a shipping verdict.
See the [pilot report](../results/strategy-20260915-pilot/REPORT.md).

A [separate grading recovery](../results/strategy-20260915-grade-recovery/REPORT.md)
subsequently repaired output-schema, citation-presentation and token-budget
handling without rerunning workers or changing this acceptance policy. It
retains every prior grading version. Three final grades were structurally valid;
one request exhausted transport retries. Independent implementation failures
override the judge's inflated raw scores. No full-study pass is established.

[BRIEF.md](BRIEF.md) is the user-supplied strategy brief, preserved verbatim
(SHA-256 `bcb948bcb5d36b85c19d2366af95880cecb0784cec2cffca34268c2161dfc6c2`).
This file records how to operationalize it; it does not replace its criteria.
The existing 42-case fixture suite and all historical results remain separate.

## Two tracks

- **Trigger:** T01–T12 and N01–N10. Candidate has the skill available through
  normal discovery, without explicit invocation. Baseline has no target skill.
  Observe actual skill activation, not merely whether its name appears in an
  answer. Missing traces are missing evidence, not negative activation.
  Precision and recall apply to the candidate's natural activation decisions;
  a baseline unable to access the skill is not a meaningful trigger comparator.
- **Execution:** E01–E12. Baseline gets the exact task; candidate gets the same
  task with an explicit skill-invocation directive. Both get the same source,
  public tests, tools, dependencies, and execution budget. Score outcomes using
  the five 0–2 dimensions, not SkillEvaluator's default aggregate or its
  qualitative `skill_efficiency` metric.

The development schedule follows the brief's baseline-then-candidate order.
This is an order confound; report it. Do not silently substitute the balanced
ordering of earlier campaigns. There is no retry-until-pass or early stopping.

Repeat mutation requests E02, E03, E04, E05, E07, E08, E10 and E12 three times
in each condition; other cases once. This conservative classification also
tests consistency of scope restraint and documentation edits: 100 development
worker attempts (44 trigger and 56 execution), before holdouts or pilots.
E07's correct boundary response need not modify core code; tests are required
for any implementation it actually undertakes, not for a correct refusal.

E04 and E12 allow planning in their expected behavior. Record a plan as a plan,
not as a completed, test-passing implementation. The implementation acceptance
gate remains unverified if the requested implementation is not delivered.
E11 supplies no underlying researcher message: do not invent its inventory or
mandatory set. Preserve the prompt and mark which expectations lack inputs.

## Isolation and evidence

Use the existing NVIDIA SkillEvaluator/Harbor integration, fixed to NVIDIA
Build model `nvidia/nemotron-3-super-120b-a12b`. Do not silently substitute the
native Codex benchmark, a model, or a mock CUDA-Q runtime.

Freeze the repository commit, source inventory, candidate operational-skill
inventory, prompts, grading rubric, independent checks, tool/runtime versions,
budgets, target, precision and tolerances before observing results. The current
source commit is `a3fdbb926e47bdfd8c86f9df11f0ed6f9e7b3eff`; pre-existing
uncommitted skill changes require a separate hash manifest, not a claim that
the commit describes the evaluated skill.

Workers receive current repository source/tests/examples, not the evaluator
repository. Exclude all `evals/`, generated evidence, rubric, holdouts, secrets,
unrelated instructions and git history that could expose them. Candidate gets
the frozen operational skill; baseline cannot read another copy of it.
`worker_payload` is a strict projection, never the complete case dictionary.

Preserve the standard skill package minus evaluator/cache artifacts, including
its authoring link and support files. Only the application task determines
which references the worker reads. A reduced SKILL-plus-references package is
a different ablation, not an unannounced change to this candidate treatment.

Reuse Harbor's repository staging and custom environment. Before running an
implementation case, verify a fresh, clean task-local worktree from an
evaluation-free source snapshot; do not expose the original repository's git
database. A plain container copy is isolation, but is not by itself the clean
Git worktree requested in the brief. Preserve source-commit provenance across
this sanitization. Never mutate the user's working tree or branches.

Capture final response, tool trace, source changes/untracked files, test
commands and exit codes, and independent numerical outputs. Capture must run
even after grading failure. Worker session logs must survive cancellation
without exporting auth/configuration files. A missing final usage record is
not zero; partial request usage is a lower bound.

Grade opaque attempt IDs in randomized order without condition labels or
staging-path hints. Preserve evidence needed to check claims, and acknowledge
that references to the skill may still reveal treatment. Retain the mapping
privately. Failed targeted tests and verified critical failures override the
numeric score. Missing/infrastructure-broken tests block certification rather
than masquerading as numerical correctness or a scientific failure.

Run targeted and independent checks on the captured patch, not only on a
worker's claim. Establish the pre-change existing-suite result in the same
environment, then test each relevant implementation; report pre-existing
failures separately and do not call an incomplete suite regression-free.

## Holdouts and acceptance

All 34 supplied prompts are disclosed development cases. None becomes an
unseen holdout merely by being renamed. Before changing the skill, reserve at
least 12 independently authored private prompts: eight trigger and four
execution cases. This gives 12/46 overall, 8/30 trigger and 4/16 execution
holdouts. Keep their wording and grading keys outside runtime copies and
skill-authoring context. Freeze before scoring; do not tune to holdout failures
and then reuse them as unseen evidence.

A separate private twelve-case freeze is now recorded in the
[preparation report](../results/strategy-20260915-preflight/REPORT.md).
Its prompts/keys are not in this directory. `suite.py --summary` intentionally
counts only the 34 disclosed cases; it does not load the private holdout set.

Quality gates from the brief are all required:

- Trigger precision and recall each at least 90%.
- Mean execution score at least 8.5/10 and at least 1.5/10 above baseline
  (15 percentage points, not 15% relative improvement).
- Zero critical failures, passing targeted implementation tests, and no
  regression in the existing test suite.
- Improved holdout performance and preserved test integrity.

Average repetitions within each execution case before averaging across cases,
so extra implementation repetitions do not silently overweight those cases.
Report each repetition, raw and critical-failure-adjusted scores, coverage and
uncertainty. Do not select only passing repetitions or treat missing grades as
measured zeros. If the baseline leaves insufficient headroom for 15 points,
report that result; do not harden already-observed holdouts to manufacture uplift.

Efficiency is a co-primary endpoint: paired task time, worker tokens and peak
request-input tokens as a context-size proxy. Report judge tokens/time
separately, and failures plus measurement coverage alongside successful-pair
ratios. Never call early failures cheaper solutions. Exact occupied context,
per-file billed tokens and monetary cost require measurements not supplied by
the proxy.

The user-defined maximum efficiency regression is **0.1%**, replacing the
unaccepted 10% proposal. Apply this limit independently to median task time,
median worker tokens per task and median peak request-input tokens, comparing
the same successfully completed matched tasks across repeated runs. For each
metric, `with_skill_median <= 1.001 * baseline_median`; equivalently,
`100 * (with_skill_median / baseline_median - 1) <= 0.1` for a positive baseline.
Improvement in one metric or in quality does not excuse another metric exceeding
its limit. Report case-level outcomes, failures, coverage and uncertainty;
missing evidence or insufficient measurement precision cannot establish a pass.

The user selected **95% one-sided confidence bounds** on 2026-09-15. Use one
common cohort of successfully completed matched tasks with complete time,
token, and peak-input measurements for all three metrics. For each metric,
the upper confidence bound on the ratio of arm medians must be at most
`1.001`; an observed ratio alone is insufficient. Report exclusions and
coverage, and return inconclusive when the evidence cannot establish the
limit. This does not relax the 0.1% threshold or authorize a larger campaign.

The confidence level and common-cohort policy are now chosen. The resampling
or other inference method, dependence treatment, informative-sample checks,
and fixed sampling/stopping plan must still be specified and verified before
scoring. The local reducer now applies common-cohort selection and records
pair IDs and measurement-exclusion reasons; frozen historical reducers and
results are unchanged. It has no implemented confidence-bound decision, so
this policy is not a measured statistical pass.

The user subsequently selected **similar future tasks**, using paired task-level
confidence bounds, rather than repeat-run inference for this exact finite suite.
The intended resampling unit is the case: preserve both arms and all of that
case's eligible repetitions together, and retain the existing ratio of arm
medians. This is model-based generalization to comparable tasks, conditional on
the complete jointly successful cohort; it assumes suitable independence and
exchangeability of cases, not randomized sampling merely because prompts were
authored. It does not remove provider/order confounding or missing-outcome bias.
The exact resampling strata, fixed seed/count, quantile rule and coverage
calibration remain to be frozen and verified before acceptance scoring.

`confidence.py` now provides a diagnostic-only paired case-cluster
percentile calculator, exposed by `judging.summarize` on the same common
measurement cohort. It retains all eligible repetitions of each sampled
case, uses shared draws for the three metrics, and returns `statistical_pass:
null` even when calculated bounds are below the limit. Malformed schedule
metadata, nonfinite values and overflow fail closed. The reducer derives
planned schedule strata from the suite contract, not the observed repetition
count; unknown/private IDs or conflicting metadata leave confidence unknown
without suppressing descriptive measurements. Its optional `manifest_sha256`
argument supplies a reproducible seed, not proof of provenance: the caller
must verify the actual frozen manifest. Missing hashes leave confidence unknown.
Independent scoped review passed and all 395 combined harness tests passed
after integration. These tests verify implementation behavior, not 95%
statistical coverage. Both statistical-pass fields remain null. The supported
`reporting.py` path now verifies the manifest/grade join; calibration and the
frozen inference specification remain prerequisites to an acceptance decision.

### Offline reporting command

Use the same source collection, saved mapping, capture/check selections and
tool projection that produced the grade payloads. Select grade directories
explicitly; omitted grades remain missing. Run with the existing evaluator
interpreter and SkillEvaluator source on `PYTHONPATH`:

```bash
python skills/cudaq-algorithms/evals/strategy/reporting.py \
  --run-dir /path/to/frozen-run \
  --mapping /path/to/collection/blinding.json \
  --grade-dir /path/to/first-saved-grade \
  --grade-dir /path/to/second-saved-grade \
  --output-dir /path/to/new-private-report
```

Optional `--checks-dir`, `--capture-dirs` and `--tool-projection ledger-v1`
match the collector's options. The output is private `rows.json`, `summary.json`,
`provenance.json` and `hashes.json`; no existing output is overwritten. Request
budgets, fixed model/system, exact payloads and fresh check/status gates must
agree. Requests from older judge contracts are rejected, not silently migrated.
Judge usage/time remain labelled retained controller measurements, separate
from worker efficiency; they are not independently remeasured or authenticated.
The wrapper accepts only the exact pilot or disclosed-study schedule, including
all absent outcomes. It never launches models, checks, or private holdouts.

## Pilot gate before a full campaign

First verify repairs with regression tests and a deliberately cancelled
synthetic worker. Then use a small, separately labeled live paired pilot with
natural trigger boundaries, a repository-grounded advisory task, and a real
implementation task. Fix equal worker budgets for both arms before launch;
the old 300-second fixture limit is not automatically suitable for repository
implementation. Check provider availability, complete trace/usage retention,
artifact capture, blinded grading and executable checks before spending on
the full development/holdout campaign. Preserve every pilot failure.

Historical five-trajectory regrading is grade recovery only, not this pilot or
evidence that these new success thresholds have been reached.

## Implemented pilot integration

`suite.py` is a tested case/payload contract, **not a live runner or grader**.
The companion `live_run.py`, capture/check helpers, `posthoc.py` and `judging.py`
implement the following integration with the repaired NVIDIA evaluator. Live
grading failures and remaining readiness work are recorded in the pilot report:

1. Stage one private, single-entry dataset per arm through
   `adapter.generate_harbor_tasks`, with `custom_only` capture and a separately
   supplied evaluator snapshot. Entries contain only an opaque ID and worker
   instruction, never the case rubric. Preserve the complete runtime skill
   minus evaluator artifacts. Keep its source anchor outside the staged
   repository projection, with baseline alias checks enabled.
2. Use `/workspace/project` as the editable agent workdir; the adapter reserves
   `/workspace/repo` for source projection. A one-time, sentinel-guarded setup
   makes a sanitized Git seed and a fresh detached worktree. Setup is a recurring
   healthcheck, so it must never recommit or reject a worker's later edits.
3. Invoke `runner._run_harbor` serially for each baseline/candidate pair, using
   the repaired interpreter's own Harbor launcher and the explicit
   `SkillEvaluatorNvidiaBuildCodex` agent wrapper. The high-level paired runner
   orders with-skill first and therefore does not implement this brief's order.
   This private-API integration must be covered by version-pinned tests.
4. Capture bounded final response, trace, modified/untracked files, diff and
   test outcomes before cleanup. Do not rely on the stock collector to copy
   arbitrary verifier files. Retain raw jobs until capture is verified; grade
   opaque, arm-redacted evidence separately using the five-dimensional rubric
   and all critical checks. Independently test the captured implementation in
   a secret-free numerical environment, not only the worker's claimed tests.
5. Preflight every generated task for source/skill isolation and no private
   rubric exposure; prove the clean starting worktree and actual subprocess
   module hashes. Run T02, N07, E01 and E03 once per arm, baseline first,
   serially: eight readiness attempts. Fix equal 600-second worker limits for
   trigger/advisory pairs and 1,200 seconds for the implementation pair, with
   equal per-pair resources and model/output policy. Keep verifier limits and
   costs separate; changing a global timeout multiplier also changes them.

This pilot is not the required three-repetition implementation study and does
not consume private holdouts. Do not launch the full 100-attempt disclosed
campaign until the pilot demonstrates usable paired execution and evidence.

### Disclosed-study preparation (offline support)

`live_run.prepare(..., mode="disclosed-study", worker_timeouts_s=budgets)` now
uses the existing `suite.build_schedule` for all 100 disclosed attempts: 44
trigger and 56 execution. It accepts no custom case list or private holdout
source. `budgets` must map exactly T01–T12, N01–N10 and E01–E12 to positive
integer seconds below `2**53` (Harbor's float-seconds representation). There
are no implied full-study limits. Each case's arms and repetitions share its
declared limit; resources and the independent 120-second verifier limit stay
unchanged. Invalid budgets fail before runtime inspection or run creation.

The CLI equivalent adds `--mode disclosed-study --worker-timeouts FILE.json`
to `prepare`. The JSON object contains only case IDs and integer seconds; it
is not a new YAML configuration. Archive hashes remain mandatory freezes:
the CLI retains the pinned defaults, while the Python API accepts explicitly
recorded `expected_source_sha256` and `expected_skill_sha256` for a new freeze.
Do not rewrite historical manifests to use new archives or controller code.

The study manifest records the expanded budgets and ordered schedule digest.
Preflight verifies the selected purpose, exact typed schedule, staged entry ID,
worker prompt bytes, and instruction digest as well as existing source/skill,
task and helper hashes. A preparation or staging-only preflight success means
that staging is consistent, not that the scientific acceptance gates passed.
The pilot remains the default eight-attempt path with its original budgets.

The [offline evidence](../results/strategy-20260916-full-schedule/REPORT.md)
uses synthetic repositories with the real adapter. Its 100 missing worker
outcomes remain missing through collection and summary, yielding no successful
pair or statistical pass. This establishes preparation/collection support, not
successful model execution, holdout coverage, new numerical checks or permission
to skip the pilot and primary-judge readiness gates.

## Local E03 diagnostic checkpoint (2026-09-16)

The [local-native E03 pair](../results/strategy-20260916-local-e03/REPORT.md)
finished with both captures preserved. Baseline failed 16 of 32 independent
targeted cases; candidate introduced an import-time `IndentationError`.
Its native transport `completed` status is not scientific task success.
Exact candidate skill delivery passed but no reference-file reads occurred.
Full regression remains unverified because the original-source native
environment cannot supply PySCF's correct own-process memory metadata.

These are local Qwen development diagnostics, not acceptance runs. Failed-task
time/token observations cannot enter the jointly successful efficiency cohort.
There is no measured quality uplift, confidence pass, trigger result or holdout
result from this pair. The pinned NVIDIA protocol and all thresholds above
remain unchanged; neither worker is retried and no new hosted call is implied.

## Local diagnostic checkpoint: workflow verb (2026-09-16)

The [bounded routing microtest](../results/strategy-20260916-routing-verb/REPORT.md)
preserves all 40 attempts and complete usage. Expected workflow reads were
0/20 in each wording condition; 26 attempts exhausted the eight-request budget.
The predeclared benefit condition was not met, so the active entry still says
`Choose`. No content was removed. Task-owned local server 07 was stopped.

These are not with/without-skill comparisons or native implementation outcomes:
both arms received the full entry, and the harness permits only virtual
list/read tools. Its missing search tool and read-only framing limit transfer
to native tasks. Terminal answers include factual errors, so no quality or
efficiency pass is inferred. Do not expand routing wording from this result;
future content fixes require source/test evidence and performance claims need
real end-to-end checks. Existing acceptance and infrastructure prerequisites
remain unchanged.

## Latest content checkpoint: Walk control-off contract (2026-09-16)

The [source-backed reference correction](../results/strategy-20260916-walk-readiness/REPORT.md)
retains the distinction between system identity and prepared ancillas. Its
numerical checks cover both state-input modes, two encodings, powers zero/two
and both uncompute choices. Original assertion failures and all isolation
attempts remain recorded. The corrected deny-file probe distinguishes empty
sandbox masks from host-file exposure without reading credential contents.
It does not solve the separate native PySCF regression limitation.

No measured skill uplift or successful matched KPI cohort follows from these
checks. The primary NVIDIA evaluation, grading/calibration prerequisites and
full-campaign requirements above remain unchanged.
