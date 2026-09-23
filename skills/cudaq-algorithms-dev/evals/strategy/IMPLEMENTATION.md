# Live NVIDIA strategy evaluation implementation plan

> For agentic workers: execute the approved protocol with test-first development,
> independent bounded tasks, and review before live calls. Do not restart design
> approval or create commits; the user has authorized continuation.

**Goal:** Run the real NVIDIA SkillEvaluator paired readiness pilot and retain
usable outcome, time, token and context evidence before the larger study.

**Architecture:** A thin controller stages private single-arm Harbor tasks from
the existing case contract. Generic capture preserves the worker's repository;
independent checks and the private five-dimension judge run after capture.

**Tech stack:** Python 3.12, repaired SkillEvaluator 0.2.1, Harbor 0.13.2,
Codex agent with NVIDIA Build, CUDA-Q 0.15.1 qpp-cpu Docker runtime.

**Spec:** [PROTOCOL.md](PROTOCOL.md), interpreting the unchanged [BRIEF.md](BRIEF.md).

## Global constraints

- NVIDIA Build model `nvidia/nemotron-3-super-120b-a12b`; no provider/model switch.
- Baseline first within each pair, serial attempts, no worker retry or stop-on-pass.
- Pilot: T02, N07, E01, E03 once per arm (eight attempts), separate from main study.
- Worker limits: 600 seconds for T02/N07/E01, 1,200 seconds for E03; identical
  resources within each pair: 2 CPUs, 4,096 MiB RAM, 10,240 MiB storage.
- Verifier/capture limit: 120 seconds, independent of worker limit. Judge:
  300 seconds per request, at most three transport attempts, no semantic retries.
- Frozen CPU image tag `cudaq-strategy-runtime:20260915-1z7jNt-qsp`, verified
  local image ID `sha256:0390e31ee0dbb81f0ab54919cf88c5440c2ff88fe5e276f7001419aff6fa48cd`.
  Do not misinterpret that config/image ID as a registry `tag@digest` reference.
- No private grading material in worker prompts, source, skill or image.
- No secret values in argv, stdout, logs, repo, images or captured worker files.
- Preserve every attempt and original result. Missing usage is unknown, not zero.
- Skill/content unchanged; all repository edits stay under this `evals/strategy`
  area and generated `evals/results`; no YAML, CI, commit, push or `.agents` edits.
- Efficiency ceiling: 0.1% independently for matched median time, worker tokens,
  and peak request-input proxy. Pilot cannot certify this repeated-run criterion.

## Task 1: Private task staging and serial live execution

**Files:** `live_run.py`, `tests/test_live_run.py`.

**Consumes:** `suite.worker_payload(case_id, arm)`, frozen source/skill archives,
and standalone `capture.py`. **Produces:** `prepare(...)`, `preflight(...)`,
`run(...)` and CLI subcommands; `manifest.json` with attempts containing
`attempt_id`, `case_id`, `arm`, `repetition`, `dataset_dir`, `job_name`,
`worker_timeout_s`, and stable input hashes. Jobs remain in `<run>/jobs`.

- [ ] Write tests for projection, baseline-first ordering and one-shot reservation:

  ```python
  assert [(a['case_id'], a['arm']) for a in pilot_attempts()][:2] == [
      ('T02', 'baseline'), ('T02', 'candidate')]
  assert len(pilot_attempts()) == 8
  assert 'Explicitly use' not in worker_instruction('T02', 'candidate')
  assert worker_instruction('E03', 'candidate').endswith(
      'Explicitly use the available cudaq-algorithms skill for this task.')
  ```

- [ ] Run tests and confirm the missing behavior fails, then implement the
  smallest controller using public `generate_harbor_tasks` and private
  `_run_harbor`; do not use its with-first high-level paired runner.
- [ ] Create a synthetic source-only Git seed with the runtime skill in an
  excluded anchor. Verify the generated repo projections are identical and
  baseline has no target skill. Use a private evaluator snapshot for the
  opaque-ID/question-only entry and generic capture grader.
- [ ] Before each worker, create a clean detached task-local Git worktree.
  Preserve the actual editable tree under `/logs/agent/worktree` so cancellation
  does not destroy it; `/workspace/project` may link to that real worktree.
  Guard initialization with a sentinel: recurring healthchecks must not commit
  or reset later worker edits. Keep initial cleanliness/source evidence.
- [ ] Freeze task TOML timeouts directly; do not use a global multiplier that
  also changes verifier deadlines. Pin/record the actual Codex agent version.
- [ ] Test setup twice around a real file edit and prove the second call
  preserves it. Test that a reserved attempt cannot be relaunched. Call the
  repaired interpreter's own Harbor entrypoint with secure key stdin handoff.
- [ ] Run the focused tests and expose a no-model-call preflight before `run`.

## Task 2: Durable repository capture and independent implementation checks

**Files:** `capture.py`, `tests/test_capture.py`,
`checks/test_protocol_action.py` (private to the evaluator, never worker-staged).

**Consumes:** source-only `/workspace/repo`, actual `/logs/agent/worktree`;
generic Harbor capture environment. **Produces:** `capture(source, worktree,
output)` writing `capture.json`, bounded changed-file blobs and a unified diff.
Standalone `grader.py` staging of this module writes a capture-only reward
(not a scientific/quality grade). A host-side recovery call supports timeouts.

- [ ] Write real temporary-filesystem tests before implementation:

  ```python
  result = capture(source, worktree, output)
  assert result['changed_paths'] == ['python/example.py']
  assert result['deleted_paths'] == ['docs/removed.md']
  assert result['status'] == 'complete'
  ```

  Also test an added file, unchanged tree, symlink/hardlink/nonregular rejection,
  oversized-file handling and duplicate-output refusal. No unsafe entry may be
  followed or silently called a complete capture.
- [ ] Compare file content against the immutable source inventory, not only
  mutable Git status. Exclude Git databases and generated Python caches;
  report exclusions. Maximum 2 MiB/file, 50 MiB/tree and explicit 1 MiB judge
  evidence ceiling; do not truncate silently or execute worker code to capture.
- [ ] Preserve complete patch/artifacts privately even when judge projection
  cannot fit. Capture before any outcome tests or model grading.
- [ ] For E03, author exact dense references for one/two-qubit real and complex
  states, both built-in and foreign structural encodings with no encode_kernel.
  Ensure unchanged source fails the foreign-action check for the actual defect,
  not a missing runtime. Include unchanged downstream Walk/QSVT tests.
- [ ] Verify the oracle using a temporary reference repair, not changes to
  repository product code; preserve the original failing reference check.
  Run any worker code only in a separate secret-free, network-disabled bounded
  Docker numerical environment. Check the captured patch, not the live worker.
- [ ] Run focused capture tests and report the standalone capture interface.

## Task 3: Blinded grading and honest KPI reduction

**Files:** `judging.py`, `tests/test_judging.py`.

**Consumes:** `suite.grader_payload`, opaque attempt evidence, actual native
trajectory/session logs and capture/check records; approved key at execution
only. **Produces:** `judge_payload(...)`, `validate_grade(...)`, `usage(...)`,
`summarize(...)` and a private grade CLI. Keep these helpers independently
usable by the live controller; do not import its code.

- [ ] Write tests for valid/invalid five-dimension grades, unknown evidence IDs,
  missing totals, critical failures, failed execution, and threshold boundaries:

  ```python
  assert efficiency_change(100, 100.1)['within_limit'] is True
  assert efficiency_change(100, 100.2)['within_limit'] is False
  assert efficiency_change(None, 100)['within_limit'] is None
  ```

- [ ] Build private payloads from exact task + rubric + stable cited evidence;
  strip explicit arm labels/staging prefixes and the treatment directive.
  Retain task-relevant content, record residual inference risk, and randomize
  opaque grade order. Validate returned IDs/ranges and critical verdicts.
- [ ] Reuse the official provider/client interface where possible; fix model,
  bound requests, retry only 429/5xx/transport up to three total attempts, and
  preserve failures plus judge usage/time separately. Never expose an API key
  to worker code or serialize it in payloads/configuration.
- [ ] Reduce native reported token totals and request peaks without inventing
  usage. Missing final totals are incomplete; partial usage is a lower bound.
  Trigger activation requires a successful observed skill read, not a mention
  or a listing. Keep trigger scoring separate from forced execution.
- [ ] Summarize all outcomes/coverage, successful matched comparisons and raw
  per-attempt observations. Do not label a single-pilot ratio a statistical
  pass at 0.1%; no calibrated uncertainty procedure exists yet.
- [ ] Run focused tests and report actual callable contracts to the controller.

## Task 4: Integration, live pilot, and campaign gate

**Owner:** controller; evidence under a new task-local run root and a separately
named `evals/results` directory. No edits to historical campaign evidence.

- [ ] Run all strategy tests with the repaired evaluator source on PYTHONPATH.
- [ ] Review the three changes together; test real task generation, secret-free
  setup/cancellation/capture and the exact repaired Harbor child imports.
- [ ] Freeze source, candidate, tasks, helpers, image, budgets and judge contract.
- [ ] Launch the eight attempts exactly once; monitor with bounded waits and
  preserve controlled progress plus all raw private artifacts. Diagnose any
  infrastructure failure without relaunching until success.
- [ ] Capture/recover artifacts, run independent numerical checks, blind-grade,
  and report actual quality/time/tokens/context/coverage. Record every missing
  channel or provider failure separately from scientific failure.
- [ ] Proceed to the predeclared larger study only when usable paired outcomes
  and complete capture are demonstrated. Add the other cases' independent
  implementation checks before launching them; never claim full certification
  from an E03-only numerical checker or from the readiness pilot.

## Execution ledger

- Design/protocol approved by user; continuation explicitly requested.
- Existing runtime/source preflight remains valid: 307 pass, four documented
  skips; fresh remote identity/resource check passed before implementation.
- Tasks 1–3 have disjoint owned files; interfaces are named above. Root owns
  integration and records any interface correction before launch.
- Existing user edits and all earlier evidence are preserved.
- Tasks 1–3 and the post-hoc capture/check/assembly integration are implemented.
  Fresh combined verification: 167 strategy tests passed. These test counts
  verify the harness, not the skill's scientific success criteria.
- Task 4 reached the campaign gate: all eight workers were launched once;
  seven completed and the E03 candidate timed out. All captures were verified.
  Both E03 patches failed 16/32 independent targeted tests and passed the
  frozen existing suite (307 passes/four skips). Two judge replies were invalid;
  two implementation payloads exceeded the frozen size limit. No valid rubric
  grades or quality-verified execution efficiency pairs are available.
  Actual CUDA-Q Docker import/capture/recovery probes passed, and the corrected
  independent original-suite wrapper reproduced 307 passes/four skips.
- The checked-in [pilot report](../results/strategy-20260915-pilot/REPORT.md)
  distinguishes pre-launch freezes, preserved failed setup probes, later
  infrastructure-only corrections, and measured worker observations.
- Worker, checker and judge schedules have finished; the report and sanitized
  measurements preserve every attempt, missing grade, provider error and KPI
  coverage limitation. The expansion gate is not satisfied.
- Remaining: correct and test the judge-response/evidence-size integration for
  a new versioned run, establish reliable provider accounting, and review the
  trigger/protocol failure modes. Preserve the existing pilot without retries.
  Do not launch the main 140-attempt study or claim certification from this
  pilot without meeting the documented expansion gate.
- Supplemental grading repair is implemented and verified by 221 passing
  harness tests. It adds opt-in structured output, lossless eight-line evidence
  blocks, conditional citation requirements, and a pinned-tokenizer guard.
  The final separately recorded recovery accepted three grades and exhausted
  transport retries for one. Independent checks forced both implementation
  scores to zero despite inflated raw judge scores. See the
  [recovery report](../results/strategy-20260915-grade-recovery/REPORT.md).
  Remaining: provider reliability and judge calibration, then the original
  trigger/protocol failures. Worker KPIs and the 0.1% criterion are unchanged.
- Local-diagnostic checkpoint: preserved the versioned local tool-loop
  [smoke evidence](../results/strategy-20260915-local-agent-smoke/REPORT.md),
  including the first partial attempt and corrected readback check. This is
  not a skill-quality result or a replacement for the NVIDIA campaign.
- After the user-requested break, the old local server's missing process
  handle, refused health connection and clean shutdown log were checked before
  restarting only the task-owned offline loopback service. No container or
  driver was restarted, and no existing attempt was relaunched.
- Diagnostic confidence output is now integrated into the reducer's exact
  common cohort and independently reviewed; 395 harness tests passed. It
  explicitly cannot establish acceptance. The next disclosed local E01 pair
  is still in preflight, with no worker launched as of this checkpoint.
- The separate E02 remote checker upload remains stopped pending the explicit
  approval required by the environment. Resume permission is not treated as
  approval for that transfer. No new hosted NVIDIA worker/judge calls were made.
- Local E01 prelaunch checkpoint (2026-09-16): both arms passed sandbox
  isolation preflight, but the first launch stopped before reservation or model
  contact. The frozen-input check rejected a zero-byte bubblewrap lock created
  by preflight under each arm's declared `.tmp/` scratch directory. All protected
  source and skill hashes still matched. The failed launch and preparation
  versions are being retained; a narrow, tested scratch-inventory correction
  must pass fresh verification before any worker attempt.
- Local E01 completed after that scratch correction: both workers exited zero,
  all 23 model requests returned HTTP 200, and 334,802 total tokens were
  cross-checked. The candidate read no skill file despite the explicit path
  directive; native registration/delivery had not been established. This is
  a delivery failure to investigate, not an observed content-effect comparison.
  See the [versioned report](../results/strategy-20260916-local-e01/REPORT.md).
  No acceptance criterion is met by these local measurements.
- The existing local CUDA-Q 0.15.1 environment imported on `qpp-cpu` and passed
  a one-qubit Hadamard state check against `[1, 1] / sqrt(2)` at absolute
  tolerance 1e-12. This is local diagnostic readiness, not validation under the
  pinned NVIDIA Docker runtime. No remote E02 upload was performed.
- The [local E03 original-source readiness check](../results/strategy-20260916-local-e03-readiness/REPORT.md)
  collected all 32 targeted cases: 16 built-in cases passed and all 16 foreign
  cases exposed the known `encode_kernel` coupling. No worker or regression
  suite ran. Its command reassigned `HOME` to scratch contrary to the preserved-
  environment constraint; the evidence is unchanged and future runs must never
  reassign `HOME` or `CODEX_HOME`. This is diagnostic failure evidence only.

- The [native skill-delivery diagnostic](../results/strategy-20260916-native-skill-delivery/REPORT.md)
  registered the existing `skills/` path and proved exact full skill-text
  injection into one local Qwen request. A subsequent app-server completion
  accounting repair passed independent review and combined harness tests
  (494 passed, 23 environment-gated skips). The historical smoke retains its
  original incomplete reducer output; a fresh scientific pair still needs to
  establish quality, routing and live KPI coverage on this native transport.

- The subsequent [one-shot native E01 pair](../results/strategy-20260916-app-e01/REPORT.md)
  completed both arms with exact candidate-only skill delivery and matching
  backend/native/OTLP accounting (364,572 total worker tokens). Neither arm
  executed a repository-read command: rejected `run_command`/`execute_command`
  calls led to ungrounded answers. Delivery/accounting now work together, but
  no quality-successful pair exists. Preserve both attempts; inspect the actual
  tool declarations with sanitized metadata before another run. No runtime
  skill edit or NVIDIA acceptance claim follows from this tooling failure.
- A subsequent no-model tool-schema contrast confirmed the E01 wrapper's
  `thread/start.environments=[]` removed execution tools. Omitting that field
  exposes `exec_command` and `write_stdin`, as the pinned schema documents.
  A fresh version removes only that setting, retaining every scientific input
  and budget. Ten focused tests and its independent isolation preflight passed
  before launch; the original failed pair and synthetic probe remain distinct.
- The [corrected native E01 pair](../results/strategy-20260916-app-e01-v3/REPORT.md)
  completed real repository commands with verified delivery/accounting. The
  candidate read no skill references, repeated whole-module reads, and produced
  an underspecified composite recommendation. It used 97.85% more time and
  92.77% more tokens despite a 4.05% lower peak-input proxy. These are single-case
  failed-task observations, not a quality-success cohort. The next narrow
  intervention is an explicit shared-contract entrypoint route, measured
  separately; no acceptance threshold is relaxed.
- The [routing trial](../results/strategy-20260916-app-e01-v4/REPORT.md)
  preserved both one-shot arms and exact candidate text. The candidate stopped
  after one filename-discovery command with an unfinished tool-call marker;
  the pair used 157,580 tokens, but there is no successful-task efficiency
  comparison. Only the trial's five-line addition was removed from the active
  skill, restoring its pre-trial hash; user edits and archived evidence remain.
  The next change is diagnostic response metadata, not a parser workaround or
  more skill wording. Backend finish reason was not retained in the old trace,
  so model-versus-parser cause remains unknown. No acceptance gate is met.
- The [offline response-boundary follow-up](../results/strategy-20260916-app-e01-v4/diagnostics/REPORT.md)
  reproduced the bare-marker pass-through in the installed parser/converter,
  while a complete synthetic call parsed correctly. The local bridge now
  retains only allowlisted finish reason, stop-reason type, structured-call
  count and a marker boolean. It preserves conversion, validation, usage and
  grading behavior. Independent review passed; root verification passed all
  429 strategy-plus-telemetry tests with the cached pinned tokenizer. No model
  generation or inference about the historical stop cause was made. Next:
  freeze an E03 implementation pair on an editable, isolated source worktree,
  capture each result before independent targeted/regression checks, and use
  the new metadata on its first actual requests. This is still diagnostic-only.

## Local-native E03 checker reuse

Original-source targeted-oracle readiness is complete as recorded above. The
local worker pair has now finished; see the completion checkpoint below. The
worker path remains diagnostic-only
after skill delivery is independently fixed and does not replace or change the
pinned NVIDIA acceptance runtime. The E01 profile makes `python/` read-only, so
it cannot host E03's source mutation. Run the mutation in a fresh editable
strategy worktree, stop the worker, then verify and reconstruct its capture
before any checker imports it.

Build a fresh post-worker checker workspace from the reconstructed `python/`
plus immutable original `tests/python`, docs and metadata. Stage the unchanged
evaluator-owned oracle only then; neither worker may see it or modified tests.
Use the existing network-disabled native sandbox and these pytest selections:

`python -B -m pytest -q -p no:cacheprovider --confcutdir=<checks> <checks>/test_protocol_action.py`

`python -B -m pytest -q -p no:cacheprovider --confcutdir=<frozen>/tests/python <frozen>/tests/python`

E03 targeted requires exactly 32 collected with zero skips; regression requires
exactly 311 with at most four skips (historically 307 passed, four skipped).
The local venv has CUDA-Q 0.15.1, NumPy 2.5.3, SciPy 1.18.1, PySCF 2.14.0,
QSPPACK 0.4.0 and pytest 9.1.1; OpenFermion and Psi4 are absent. Preserve
`PYTHONPATH` to captured `python/`, `qpp-cpu`, disabled pytest plugin autoload,
thread limits, traversable read-only inputs, and writable executable scratch
for caches and CUDA-Q JIT. Count or dependency drift is unknown, never pass.

### Next local E03 diagnostic pair: bounded implementation plan

Use the existing app-server runner's transport, typed candidate skill input,
delivery checks, telemetry and process-drain logic. This is a local Qwen
development diagnostic, not a change to the fixed NVIDIA acceptance protocol.
No model call is authorized by preparation alone: root must review the runner,
checker and successful isolation preflight before reserving a fresh pair.

The E01 inventory's 83 files comprise 81 repository files plus harness-only
`input.json` (an E01 advisory label) and `tools/rg` (a 5.2 MiB binary). Preserve
the exact 81 scientific-source/docs/tests/metadata files as the E03 public
source; stage the same hashed search binary as a separate read-only tool.
Do not carry the advisory label into E03 or raise the 2 MiB capture limit.

- [x] **Runner preparation:** create `paired_app_e03.py` and focused tests in
  one fresh temporary root. Use `suite.worker_payload("E03", arm)` only for
  worker task content; no rubric, expected answer, oracle or holdout material.
  Make independent source-only Git seeds and real worktrees for the two arms,
  with no common Git directory across arms. Candidate gets the exact frozen
  100-file operational skill as a read-only sibling outside captured source;
  baseline registers zero skills. Keep tools, scratch and JIT caches outside
  captured source so they cannot inflate or contaminate repository evidence.
- [x] **Test-first isolation:** demonstrate editable worker `python/` and
  `tests/`, ordinary private Git operations, read-only docs/metadata/skill,
  denied original repository/evaluator/other arm/credentials/network, executable
  scratch and unchanged actual HOME/CODEX_HOME. Expose the existing qpp-cpu
  environment's Python first on PATH with `PYTHONPATH=<worktree>/python`.
  Test both allowed and denied operations, restore any preflight edits, verify
  equal source inventories, and confirm no reservation or generation occurred.
- [x] **Checker preparation:** independently test a small native adapter around
  `capture.reconstruct`, `runtime.sandbox_command`, `numerical.classify`, and
  `numerical.judge_check`. Require both complete captures and their frozen
  manifest hashes before staging or executing any check. Preserve the full
  reconstruction, including worker-authored tests, as evidence; the checker
  projection uses only reconstructed `python/` plus immutable original tests,
  docs and metadata. Stage the unchanged private oracle only after both
  captures are verified. Never collect worker tests as grader tests.
- [x] **Review and freeze:** independently review both adapters, run focused
  tests and real no-generation sandbox preflight, then freeze all helper,
  prompt, source, skill, model, tool/runtime and budget hashes. Record which
  resource limits are enforced; native diagnostics must not imply the pinned
  Docker runtime's guarantees. Verify the live server lifetime covers the
  complete scheduled worker budget before launch.
- [x] **Run once:** baseline then candidate, at most 1,200 seconds each, no
  retries or stop-on-pass. Stop each app-server and drain events before checking
  protected inputs and calling
  `capture.capture(frozen_public, stopped_worktree, capture_dir)`. Do not use
  mutable Git status as source evidence. Preserve failures and incomplete
  captures; neither becomes a successful task.
- [ ] **Check captured outcomes:** execute each of the targeted and original
  regression selections above once per arm, with 120-second targeted and
  900-second regression limits (corrected after original-source readiness).
  Use `judge_check(result, expected_collected=32, allowed_skips=0)` and
  `judge_check(result, expected_collected=311, allowed_skips=4)` respectively.
  Hash commands, outputs and source binding; separate numerical failures from
  unavailable, incomplete or drifted checks. No hosted judge call is included.
- [x] **Report and decide:** retain final responses, actual paths read, patch,
  independent checks and time/token/peak-input measurements. Inspect the new
  response diagnostics when calls are unfinished. Assess whether both tasks
  actually succeeded before comparing efficiency; no single local pair can
  satisfy the original trigger, quality-uplift, holdout or confidence gates.

Historical preparation checkpoint: the runner was being tested at
`/tmp/cudaq-local-e03-app-pair.TLzA7k`, with a separate checker adapter at
`/tmp/cudaq-e03-native-checker.AOgb2f`. Neither has launched a worker or quantum
checker. The runner's first isolation probe had a false-positive missing-path
check and is preserved for correction; the checker needs scratch-independent
inventory and bounded output before live use. Do not launch from their initial
green test counts or initial preflight alone. The idle task-owned Qwen server
05 was gracefully stopped at approximately 00:20 UTC (terminal process exit 0)
because its remaining lifetime could not cover the full next 40-minute worker
budget. Restart only after the fresh reviewed freeze and preflight pass.

Runner preparation follow-up: preparation revision 3 corrects the initial
false-positive probe and subsequent review findings, preserving revisions 1
and 2 separately. Its actual no-model sandbox preflight passed for both arms,
including edits, private Git operations, executable scratch, denied cross-arm
and repository reads, denied network access, and read-only candidate skill.
Root independently verified all 13 focused runner tests and the frozen-input
readiness check. Runner SHA-256:
`2b2f2f0b2cffd9342cfde1fd3990c8dfadff1d7d9a126a99acc016a1df5aec4a`.
Independent static review found no remaining important runner issue. This is
not evidence of task success: no worker has launched, and the separate
checker review and complete launch freeze remain outstanding.

Checker preparation follow-up: independent review and root's 24-test run
passed for the adapter; a real sandbox probe then exposed a permission
precedence issue missed by static checks. The corrected read-only workspace
profile passed the complete harmless probe, while preserving both the failed
probe and reviewed predecessor. The [checker archive](../results/strategy-20260916-local-e03/checker-preparation/README.md)
records exact versions. This is infrastructure evidence, not task efficacy.

The [original-source regression readiness](../results/strategy-20260916-local-e03/original-source-readiness/README.md)
then hit the initial 120-second budget with incomplete results. A separate
stop-at-first-failure diagnostic identified PySCF's denied own-process memory
read at `/proc/2/statm` after 41 passing tests. Narrow process-metadata access
is under investigation; no scientific tests or source are being weakened.
Historical complete runs of this 311-case suite took 412–529 seconds under
the existing 900-second checker allowance. The local regression allowance was
therefore corrected to 900 seconds before any worker launched;
targeted checks remain at 120 seconds. The initial timeout is retained and
no acceptance threshold or worker KPI definition changes. At that checkpoint,
task-owned local server 06 was running without worker requests.

### Local E03 completion checkpoint

The [completed diagnostic report](../results/strategy-20260916-local-e03/REPORT.md)
preserves one attempt per arm, exact source/skill/capture evidence, and the
post-capture targeted checks. The baseline exhausted its 40-request budget;
the previously unstarted candidate ran once through the frozen runner in an
explicitly recorded controller continuation. No worker was retried.

Neither solved the task: baseline 16 passed/16 failed; candidate introduced
an `IndentationError` and collected no numerical cases. Candidate native
transport completion with an unfinished tool marker is not task completion.
Exact skill delivery passed, but no skill reference was read. Existing
references already contain the missed runtime-interface guidance.

Observed worker totals were 616,099/595,278 tokens, 282.561/245.072 seconds,
and 24,189/24,283 peak input tokens (baseline/candidate). Baseline telemetry
completeness remains unverified after the failed turn. There is no jointly
successful pair and no efficiency or quality-uplift claim. Full regression
on either capture was not run: the original-source process-metadata issue
remains, and the tested private-proc isolation attempt was denied. No shim or
relaxed test was applied. The targeted checks do not replace regression.

Server 06 was gracefully stopped after both captures. The outcome-check item
above intentionally remains incomplete for full regression. The subsequent
independent result-gate audit found no acceptance escape; root verified 11
focused numerical/judging tests. No grading or parser behavior was changed.
Any future local-runner integration must distinguish its native transport
`completed` field from independently checked task success. Primary NVIDIA
acceptance and remote execution retain their existing prerequisites.

### Workflow-verb microtest completion checkpoint

The [40-trial local microtest](../results/strategy-20260916-routing-verb/REPORT.md)
is archived with its prelaunch freeze, both entry variants' byte provenance,
full traces, exact file-read paths/timing and complete provider usage. The
controller completed its fixed schedule once; fresh probe/controller tests
passed 19/19 and all 387 frozen input/code hashes were reverified.

Neither `Choose` nor `Open` produced a workflow-file read (0/20 each), and the
one-word intervention is not retained. Family-reference reads occurred in
both variants. There were 14 terminal responses and 26 request-budget endings;
terminal responses include source-contradicting answers and are not successful
implementations. The eight-request virtual list/read harness lacks `rg` and
reframes implementation as discovery, so it does not establish native routing
failure, implementation efficacy or an efficiency improvement.

The operational skill remains unchanged; task-owned Qwen server 07 is stopped.
No new hosted call, acceptance adjustment, source change or retry occurred.
Do not fund further wording variants from these observations alone. The next
end-to-end comparison still requires actual task completion, independent
targeted/regression checks and complete matched measurements; NVIDIA grading,
runtime/remote prerequisites and statistical calibration remain outstanding.

Historical next content candidate (completed below, separate from the frozen wording trial): the Walk
record's control-off identity statement lacks an `uncompute=True` qualifier.
Current `qubitization.py:253–316` applies PREPARE in both branches but UNPREPARE
only when requested; the existing orchestration mock uses nontrivial `ry(0.81)`
preparation. Independently test the control-off/no-uncompute combination before
qualifying the record. Do not treat this source-grounded discrepancy as proof
of a measured skill uplift or change product code to fit the documentation.

### Walk reference correction completion checkpoint

The [contract correction and evidence](../results/strategy-20260916-walk-readiness/REPORT.md)
are now recorded. The old unqualified identity assertion fails in all four
control-off/no-uncompute combinations tested; correct branch expectations and
the original orchestration suite give 25 passes. Only the focused Walk record
changed among the prior 100 operational files. Product source/tests and the
entry point are unchanged by this correction.

The numerical sandbox's exact-file deny masks initially fooled an exception-only
readability probe. Positive/negative harmless canaries and protected-file
metadata established the masking behavior; the corrected gate and numerical
commands use the same profile. Preserve all attempts. This does not resolve
the separate full-regression PySCF process-metadata blocker or qualify a model
run for acceptance. No new model call, CI change, threshold change, or retry of
a failed worker occurred.

### Earlier diagnosis: E07 conditional outcome

Read-only diagnosis reproduced a mismatch with the approved protocol:
`suite.case_by_id("E07").conditional_outcome` is true, but `judge_payload`
sets `implementation_requested` from `mutation_request` alone. With a
structurally valid synthetic grade, complete capture, and both test verdicts
`not_applicable`, `validate_grade` returns `unverified` / `successful=False`.
The collector also uses static mutation status for test applicability.

This is a gate reproduction, not an evaluated agent answer. Proposed bounded
repair: derive E07's no-implementation exemption only from a verified complete
unchanged capture; changed or incomplete captures still require independent
checks. The initial approval pause was unnecessary: the approved protocol
already specifies this behavior. The completed repair is recorded below.
Other mutation cases lack case-specific check suites; this exception must not
make E02/E03/E04/E05/E08/E10/E12 automatically successful or certify untested
implementations. Hosted judging and the primary campaign remain suspended.

### Full-campaign readiness audit

The [offline audit](../results/strategy-20260916-harness-audit/REPORT.md)
confirmed that the supported preparer still stages only eight pilot attempts:
the 100-entry `suite.build_schedule` is not wired into `live_run.prepare` or
its CLI. Do not mistake a generic execution loop or `--main-study` summary flag
for full-campaign staging support. Add the supported schedule/freeze path and
declare budgets before any main-study launch; E04/E05/E07/E08/E10/E12 also lack
registered independent checks. E07's conditional refusal requires separate
capture-grounded applicability, not an unconditional bypass.

Six current harness modules passed 274/274 in the final offline run. The initial
273-pass/one-failure result is retained: the outer sandbox prohibited the test's
UID-65534 child; the exact probe and then the complete selection passed with
the required permission, without changing tests. This is harness evidence,
not a scientific regression run, model comparison, or confidence certification.
No grading implementation or policy changed in this audit.

### E07 no-implementation gate repaired

The [capture-grounded repair](../results/strategy-20260916-e07-gating/REPORT.md)
implements the existing E07 exception without changing acceptance policy.
Complete, hash-bound unchanged captures can make implementation tests
not-applicable; modified or unverifiable captures cannot. Unknown/failed
critical verdicts and incomplete execution still prevent certification.
E07's evidence now distinguishes capture provenance from numerical-result
provenance. Other execution cases receive byte-identical judge payloads for
identical evidence, as checked against the saved pre-repair implementation.

The final six-module offline regression passed 315 tests. Red tests and all
intermediate runs are preserved, including a corrected test lookup that had
ignored randomized payload order. Synthetic grade fixtures establish gate
behavior only, not agent quality, efficiency improvement or a statistical pass.
No hosted judgment, model execution, private holdout, product-source change,
CI change or threshold change occurred. Full-study preparation and independent
check coverage remain unfinished, as do primary grading and confidence readiness.

### Disclosed-study preparation: bounded implementation steps

Implement the already-approved 100-attempt schedule without launching it.
Files: `live_run.py`, `tests/test_live_run.py`, this plan and `PROTOCOL.md`.
No new worker budget is selected: callers must supply every disclosed case's
limit in positive integer seconds. The pilot defaults remain byte-for-byte
equivalent in schedule and budget; no arbitrary case list or holdout input is
accepted. Tests use synthetic source/skill archives and the real local adapter.

- [x] Add failing tests for `prepare(..., mode="disclosed-study",
  worker_timeouts_s=budgets)`, requiring exactly the 34 disclosed case IDs.
  Check all 100 ordered attempts (44 trigger, 56 execution), three repetitions
  for E02/E03/E04/E05/E07/E08/E10/E12, and baseline first in every pair.
- [x] Implement explicit schedule selection using `suite.build_schedule`.
  Validate all budgets before creating a run or inspecting runtime. Freeze
  expanded budgets and a digest of case/arm/repetition/timeout tuples in the
  manifest; keep the 120-second verifier limit and existing paired resources.
- [x] Exercise real adapter staging and preflight: prompt-only worker entries,
  no private rubric, exact source copies, candidate-only full skill, unique
  opaque identities, generated task budgets and schedule consistency.
- [x] Expose `prepare --mode disclosed-study --worker-timeouts FILE.json`.
  Verify CLI plumbing and rejection of incompatible pilot budget overrides.
  Invalid/missing budgets, schedule tampering and attempts skipped out of order
  must fail before execution. Keep all run/reservation semantics unchanged.
- [x] Run focused and combined offline regression, review the diff, preserve
  failures and final evidence. Document preparation support separately from
  pilot/primary-judge/checker/confidence readiness. Do not launch models or
  alter acceptance thresholds.

Completed with [full-schedule evidence](../results/strategy-20260916-full-schedule/REPORT.md):
39 controller tests and 354 combined harness tests pass. Review-driven negative
controls additionally reject coerced schedule values, purpose relabeling, and
case/task swaps even when ordinary byte hashes are recomputed. The actual
instruction file and private task entry must both match the declared case/arm.
The real collector and reducer retain all 100 absent outcomes as missing,
without a successful pair or statistical pass. No live budget selection,
campaign, model call or holdout execution occurred. Remaining numerical checker,
primary-judge and confidence readiness gaps are unchanged.

### Manifest freeze validation before summary provenance

The collector currently accepts an empty `manifest.sha256.json` object because
the mismatch condition relies on dictionary truthiness. Repair this with a
failing empty-object test, retain diagnostic collection for a genuinely absent
freeze file, and export explicit matching/missing freeze status in the existing
`provenance.json` (no new output type). Add real collector/CLI tests for correct,
missing, malformed and mismatched records before editing `posthoc.py`.

This is local file consistency, not proof that freezing happened before results
were seen. A later supported offline merge must bind saved judge requests and
responses to collected payloads, revalidate grades using current independent
checks/execution status, and only then seed the diagnostic summary from the
verified manifest. That merge does not exist yet; a caller-supplied hash or
manually attached grade is not its substitute. Statistical acceptance remains
null until the separately proposed calibration is approved and verified.

The [freeze validation repair](../results/strategy-20260916-manifest-freeze/REPORT.md)
is complete: 63 collector tests and 387 tests across eight harness modules pass.
The preserved red run reproduced the empty-object acceptance and missing status
export. No judge/scoring/calculator behavior changed; rows remain ungraded until
the separate supported merge is implemented. That merge must also bind raw
responses to the exact wire request: `request_sha256` currently hashes request
kwargs, not independently supplied checks or execution status. Revalidate from
the retained raw response against freshly collected evidence; do not trust a
copied `result.grade` or permit omitted/duplicate rows to change the cohort.

### Offline saved-grade reporting

Implement `reporting.py` and `tests/test_reporting.py` as Task 3's missing
supported join, without model calls or inference-policy changes. Require an
existing blinding mapping and verified local manifest freeze. Recollect every
attempt; bind each explicitly selected saved request to its fresh payload and
the supported wire settings, then revalidate the raw response using current
execution status and independent checks. Reject stale, duplicate, foreign or
inconsistent grades; retain missing grades and failed attempts. Derive the
diagnostic seed from the same observed manifest bytes, not a caller argument.

Test the four presentation/schema combinations, failed/missing grades, changed
evidence/checks/status, unsafe paths, tampered requests/responses and output
no-overwrite permissions before implementation. Publish rows, summary and
hashed provenance only after validation, into a new private output directory.
Keep judge costs explicitly labelled retained controller measurements. Matching
artifacts prove consistency, not chronology, valid model judgment, or calibrated
confidence; both statistical acceptance decisions remain null.

Completed in the [verified-reporting checkpoint](../results/strategy-20260916-verified-reporting/REPORT.md):
44 focused tests and 431 tests across nine harness modules pass. Independent
review exposed invalid budget admissions and malformed-container error paths;
negative controls reproduced both before repair. Exact schedule validation,
helper/brief provenance, common-cohort exclusion and no-overwrite publication
are covered. No live current-contract grade collection exists locally, so this
is tested implementation support, not a replayed primary campaign or acceptance
result. Historical prompt/validator versions are intentionally not migrated.

### E05 independent state-preparation checks

Implement the missing E05 checker in bounded stages. First add an independent
four-determinant reference and full-state assertions (`checks/ci_oracle.py`),
plus an evaluator-owned direct callable mapping (`checks/ci_adapter.py`).
Use determinants from one common orthonormal orbital basis, with real and
complex coefficients; allow only one global phase for the complete sum.
Test the reference against a separate creation-operator construction and
explicit sparse amplitudes. Negative controls must catch independent branch
phases, lost imaginary coefficients, wrong bit order, nonunit norm and garbage
ancillas. The binder must never implement missing preparation or repair output.

This slice covers common-basis CI, not arbitrary nonorthogonal families or
postselected preparation. Unmapped API shapes remain unknown. Next, calibrate
the actual kernel/Walk checker in an isolated frozen runtime and bind it to
captured worker artifacts before registering any E05 pass. Do not advertise
math/binding unit tests as agent success or as the complete E05 outcome suite.
No product edits, hosted model calls, holdout reads or threshold changes.

The [E05 reference/binding checkpoint](../results/strategy-20260916-e05-reference/REPORT.md)
implements this first slice: 35 reference tests, 49 binding tests, and 515
combined harness tests pass. Native public-kernel characterization confirms
branch phases `(pi, pi, 0, 0)` can yield a unit-norm, correct-particle-number
host-formed sum with fidelity `4/9`; both real and complex coefficient controls
are caught. The earlier dense-basis hypothesis did not reproduce the defect
and remains preserved. Evaluator-only phase compensation is not a proposed
production implementation. The actual E05 captured-artifact/kernel/Walk checker
and registration are still pending; no acceptance criterion was certified.

The [E05 native-runtime checkpoint](../results/strategy-20260916-e05-runtime/REPORT.md)
now implements actual system-only preparation/Walk calibration: 17 native
tests and 526 tests across twelve offline harness modules pass. The probe
samples with SDK extraction disabled, checks the full prepared/identity states
and one-step block, and inspects resolved captured kernels. Preserved failing
controls drove corrections for builder/decorator composition, already-compiled
module inspection, hidden reset and explicit noise. The dense positive
calibration is not a worker solution or evidence of Slater/Givens reuse.
Internal ancilla allocation is unknown without an explicit layout binding;
the brief permits defined ancilla behavior, so do not classify it as inherently
incorrect. Next bind complete artifact/source provenance, input/resource and
ancilla contracts, exact outcome counts and original regression before E05
registration. No model call, holdout read or acceptance change occurred.

### E05 captured-artifact integration (in progress)

Use separate worker-targeted tests, source/binding freezes and the existing
post-hoc grading path; never reuse calibration counts as worker coverage.

| Task | Produces / consumes | Boundary check |
| --- | --- | --- |
| Targeted suite | Explicit E05 binding, actual kernels, validity/policy probes | Unsupported mappings/layouts must be unknown; genuine outcomes fail normally |
| Numerical runner | E05 selection, binding-only mount, e05-inputs.json | Preserve existing E02 wire keys; count frozen only after native calibration |
| Artifact/posthoc integration | Revalidates E05 freeze against capture/source/checker | Targeted result cannot bypass capture binding or original regression |
| Native calibration/review | Positive, broken, missing/foreign binding controls | Does not establish worker reuse/resources or full campaign acceptance |

Ruling: keep this ledger in the existing skills plan and scratch briefs/review
packages in `/tmp`, not new `.superpowers` files, because permanent edits are
restricted to `skills/`; no commits or cleanup of historical work. Parallel
owners have disjoint files and the E02-compatible freeze interface above.

Completed the bounded [E05 integration](../results/strategy-20260916-e05-integration/REPORT.md):
13 worker-targeted tests registered, numerical/source freezes wired through
artifact and post-hoc verification, and four positive native controls pass
the exact inventory (both input representations and coefficient policies).
Numerical phase controls fail; missing/foreign bindings, wrong consumer origin
and unsupported ancillas remain unknown. Review fix round 1 added exact staged
E05 binding rereads at both acceptance layers; four negative tests reproduced
the gap before repair and scoped re-review approved it. Final combined harness
verification: 561 passes across twelve modules, without exclusions/skips.

Full original regression was attempted, not waived: it timed out at 120 seconds.
The separate first-failure diagnostic reproduced PySCF's missing own-PID statm
file after 41 passes. Existing evidence already rejects a stale exact-file
counter and records private procfs mount EPERM; no broader permission or fake
memory workaround was attempted. A namespace-correct private procfs / approved
container route remains necessary. E05 reuse/resource evidence and internal
ancilla contracts are still incomplete; positive dense calibration is not an
agent solution. Continue remaining case-checker and primary-evaluation work
without treating this registration or its test counts as acceptance.

### E08 independent evolution reference and native characterization (in progress)

Continue Task 4's missing independent case coverage without registering an
unvalidated worker suite. The E08 brief requires a public-API, hardware-shaped
example and independent dense-exponential checks; the current recovery helper
explicitly supports real Hamiltonians and real inputs only. A documented,
enforced real-only example must not be failed merely for declining complex
inputs. The checker must not repair that limitation on the artifact's behalf.

| Task | Produces / consumes | Boundary check |
| --- | --- | --- |
| Independent oracle | Small Hermitian matrix, normalized input, real time; dense evolution and raw output metrics | No artifact/CUDA-Q imports, implicit normalization, or hidden phase adjustment |
| Native characterization | Original QSVT/PhaseSequence, QSP phases and evaluator-only state extraction | Sampling a data-free kernel is separate from validation; no worker or acceptance claim |
| Explicit binding design | Actual artifact build/run/recovery seams | Unbindable examples remain unknown, not failures invented from evaluator API preferences |

Ruling: implement the mathematical reference and characterize the native public
path before selecting an E08 artifact contract. Exact-vector versus physical-ray
comparison must be explicit, not silently chosen after results. Native controls
and oracle unit tests cannot stand in for worker provenance, worker-authored
validation, original regression, or the full E08 suite. The existing skills-only
ledger, temporary briefs, no-commit and unchanged-acceptance constraints remain.

Completed the bounded [E08 reference/native slice](../results/strategy-20260916-e08-reference/REPORT.md):
the independent NumPy oracle has 75 passing tests. Review fix round 1 reproduced
and corrected huge-integer exception classification and optimization-removable
output validation; scoped re-review found both addressed with no new important
issue. Final related mathematical regression: 143 passes in the existing native
runtime interpreter, with no skips. Earlier setup/test failures are preserved.

Ten native public-QSVT controls ran under the unchanged inner isolation profile
after explicit outer-sandbox approval. Both real-domain controls agree with
SciPy expm at about 2e-15; the final oracle independently agrees with that
reference and rejects eight wrong/domain-limited controls. This verifies
calibration, not E08 workers, self-validation, binding or original regression.
The prior direct SDK version-string comparison failed before kernel execution;
its correction checks the same pinned distribution version as existing suites.

A new approved boundary-only private-proc probe still failed with mount EPERM.
It mounted no host procfs, home or repository, and ran no quantum/model code.
No further retry or permission workaround was used. Full regression remains
blocked; other case-checker work can continue. E08's next slice is explicit
construction/recovery bindings, actual public-call tracing and perturbation of
the artifact's own validation before runner/posthoc registration. Unbindable
monolithic examples must stay unknown rather than be failed for an invented API.

### E08 artifact binding and runtime probes (in progress)

Implement explicit build/recover/validate mappings, without supplying missing
phase generation, preparation, reconstruction or validation. The first binding
supports exactly two plain public QSVT kernel results and real-linear classical
recovery. Four-run complex extensions, wrapped/controlled layouts, monolithic
scripts and nonlinear recovery remain unknown until explicitly supported.

| Task | Produces / consumes | Boundary check |
| --- | --- | --- |
| `evolution_adapter.py` + binding tests | Strict JSON roles and selectors; opaque artifact results unchanged | Imports/preflight only inside runtime isolation; no API guessing or expression evaluation |
| `evolution_runtime.py` + pure tests | Actual public-call trace, sampleability, oracle comparison, dependency and validation controls | No host execution of artifact code; no statevector initialization; no silent convention fixes |
| Native source-bound calibration | Positive examples and deliberate broken/unsupported controls | Calibration is not worker coverage or evidence that the example entry point invokes its validator |

Ruling: explicitly support nonzero SystemExit validation failures because the
existing public example uses them. A negative self-validation probe must exceed
the independent oracle's tolerance under the selected phase convention while
preserving shape and norm. Recovery checks include zero, half, negative and
component-separated inputs; these are bounded homogeneity/additivity evidence,
not a proof against an adversarial artifact. Sequence snapshots must reflect
what public QSVT actually consumed, without iterating generator phases twice.
Metadata declarations must eventually be bound to artifact documentation.

The two implementers own disjoint files; root owns native controls and evidence.
No E08 registration, model call, source/product edit, holdout inspection, remote
upload, criterion change or full-regression waiver is part of this slice.

Ruling: no observed QSVT construction yields unknown, not an automatic scientific
failure, because a legitimate cached kernel may predate the probe. It cannot
pass without additional origin/use evidence. This may leave more examples
unverified, but avoids treating absence of tracing evidence as proof of absence.

Completed this bounded [E08 binding/runtime slice](../results/strategy-20260916-e08-binding/REPORT.md):
23 native controls pass with actual CUDA-Q kernels, and root independently ran
326 passing tests across eight related modules. Native v1's omitted-preparation
classification failure is preserved; the corrected runtime passes v2/v3.
Adapter review exposed missing-key synthesis and non-string equality dispatch;
four red regressions, targeted fixes and scoped re-review closed both gaps.
Independent runtime review approved the bounded contract without important or
critical findings. No worker suite was registered and no acceptance claim made.

Next: build a distinct E08 worker-targeted inventory, bind its declared metadata
to artifact documentation and validation-use/source evidence, then connect the
calibrated checks to numerical/capture/post-hoc provenance. Do not reuse the 23
calibration counts as worker coverage or infer that a mapped validator is called
by the example entry point. Unsupported layouts remain unknown. Original
regression infrastructure, other case checkers, reliable primary grading and
the complete paired study/confidence requirements remain unchanged.

### E08 captured-artifact numerical integration (in progress)

Continue the approved outcome-checking work with a separate worker-targeted
suite, not the earlier 23 calibration controls. Source-level review of actual
worker documentation and use/independence of its validator cannot be certified
using evaluator fixtures; retain those obligations explicitly in grading.

| Task | Produces / consumes | Boundary check |
| --- | --- | --- |
| Worker suite | Explicit E08 binding; 8 real-domain and 4 domain-dependent cases | Tests invoke captured APIs; limitations skip to unknown, faults fail normally |
| Numerical registration | Suite selection and 12-case inventory, pending native collection; E08 source/binding freeze | Existing E02/E03/E05 channels remain compatible; no calibration counts reused |
| Capture/post-hoc join | E08 binding option and exact freeze/result/log/staged-binding checks | Missing or changed evidence remains unknown; original regression still runs |
| Root calibration/review | Exact suite against positive, broken and unbindable fixtures | No model/worker/quality/KPI claim; every count must be observed before final registration |

Ruling: retain the reviewed E08 binding schema unchanged and use the established
numerical evidence channel for numerical/sampleability/self-validation results.
Do not invent a machine attestation that would imply documentation or validator
independence was reviewed without an actual worker artifact. Those source-level
requirements remain distinct, explicit judge/reviewer obligations. This costs
additional review before any E08 quality acceptance, but avoids a false pass.
The three implementers own disjoint files; root owns isolated calibration and
evidence. Existing skills-only/no-commit/no-cleanup constraints remain in force.

Completed the bounded [E08 numerical integration](../results/strategy-20260916-e08-integration/REPORT.md):
native collection confirms the separate 12-case inventory. Four valid controls
pass it fully, seven broken controls fail and four unsupported/missing controls
stay unknown. Runner and provenance changes preserve other registered cases and
reread actual staged bindings. Independent reviews approved all three components
with no important/critical findings. Root verification: 533 controller passes
plus seven initially skipped tokenizer tests that passed with pinned assets,
and 326 related mathematical passes; 866 total across 19 modules/three runs.
Original scientific regression was not rerun or waived; no model/KPI/acceptance
claim was made. Source-level E08 obligations on real worker artifacts remain.

Next independent-case gap: E04 chemistry integration, followed by changed E07,
E10 and E12. Preserve the overall primary-evaluation, original-regression,
quality, efficiency, holdout and confidence requirements while completing these
gaps; registered checker coverage is not a substitute for paired agent outcomes.

### E04 independent chemistry reference and contract audit (in progress)

Current source already provides `chemistry.from_pyscf`, with a lazy provider
import, chemist-order spatial tensors and separate nuclear repulsion. Build
independent convention evidence for the requested adapter outcome rather than
assuming an absent public feature or requiring a duplicate API.

| Task | Produces / consumes | Boundary check |
| --- | --- | --- |
| Pure chemistry oracle | AO-to-MO tensors and occupation-basis dense operator from chemist integrals | No CUDA-Q/PySCF/artifact imports; signs, half factor, spin/orbital order and offset independently tested |
| Contract audit | E04 brief versus proposal/implementation outcome policy | Do not silently relax the frozen acceptance interpretation |
| Native characterization | Actual current chemistry consumer with independent tensors; provider-absent import boundary | Not a live mean-field conversion, original regression, worker solution or registered E04 suite |

Ruling: the existing policy explicitly leaves E04/E12 plans unverified as
implementations. Audit that distinction, but do not change it without resolving
the original brief/policy conflict. Real PySCF mean-field execution still needs
the known regression-runtime prerequisite; no fake mean field or patched memory
counter will substitute for it. Pure references and downstream characterization
are necessary preparation, not completion of E04. Existing file-scope and
no-commit/no-remote/no-holdout constraints continue.
