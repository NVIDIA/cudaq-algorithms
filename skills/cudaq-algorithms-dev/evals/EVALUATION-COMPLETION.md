# Evaluation follow-up — 2026-09-13

> Execution ledger for the user-approved follow-up: finish the remaining
> evaluation work, excluding CI. Use scoped implementation, test-first changes
> and independent review. Keep this ledger when handing off or compacting.

## Final outcome

The requested non-CI implementation and evaluation follow-up completed on
2026-09-13. The chronological notes below retain intermediate failures and
decisions; their earlier pending/running statements are historical.

- Scientific: **70/70 strict and numerical/API passes**, including real Psi4;
  **52/52 records now have scoped current-record execution mappings**. This
  does not validate every parameter, alias, hardware target or workflow.
- Behavioral: **270/270 workers completed**, with complete native token/time
  measurements. Original grades are preserved. A separate all-worker regrade
  completed **270/270 valid grades**, but independent review found semantic
  inconsistencies. All 192 original and 160 corrected strict failures were
  audited without changing their recorded verdicts.
- Corrected recorded scores: original cases 73/126 candidate versus 27/126
  baseline; separate authoring 6/9 versus 4/9. These are provisional judge
  outcomes, not calibrated quality or scientific-validation rates.
- Efficiency is mixed: the new scientific suite is near-neutral in total
  tokens/time; candidate behavioral totals are **+164.17% tokens / +69.34%
  mean attempt time**, authoring **+60.25% / +62.05%**. Full context occupancy,
  dollar charges and behavioral time-to-verified-solution remain unmeasured.
- All 85 operational skill files match the previous refactor candidate
  byte-for-byte. Original fixtures, YAML and historical results are retained.
  No CI integration or Git history/index changes were made; no NVIDIA
  SkillEvaluator run is claimed.

See [scientific results](results/coverage-20260913/ANALYSIS.md) and
[behavioral/authoring results](results/behavioral-20260913-v2/ANALYSIS.md) for
protocol, scope, limitations and next-step recommendations.

## Goal and specification

Resolve the real Psi4 runtime blocker and add meaningful execution evidence
for the 25 scientific records not covered by the completed refactor campaigns.
Also evaluate the previously authored behavioral cases and the separate
authoring workflow. Record successes, failures, blocked cases, tokens and time
honestly. The user's latest instruction and the gaps in
[the completed refactor report](results/refactor-20260911/RESULTS.md) define
scope; this does not authorize CI work or package-source fixes.

## Global constraints

- All maintained changes stay under `skills/cudaq-algorithms/`.
- No CI work, new YAML, `.agents/` migration, Git mutation, commit or push.
- Preserve all existing skill content and all completed campaign artifacts.
  Do not alter operational guidance without a demonstrated, reviewed defect.
- Use actual installed CUDA-Q/Psi4 providers, not mocks. Never relax scientific
  tolerances or sandbox boundaries to turn failures into passes.
- Preserve independent numerical oracles, actual host/device execution checks,
  native token accounting, no-skill controls and isolated fresh applications.
- Freeze every new campaign before model attempts; retain failures and
  interrupted evidence. No repairs to submitted applications or score tuning.
- Scientific, behavioral and authoring evidence have separate denominators.
  Individual API coverage is not universal scientific validation.
- Existing login only; no API credential reads/copies, global configuration
  changes or additional remote services. No NVIDIA SkillEvaluator claim.

## Architecture and task boundaries

Extend the existing native evaluator through separately labeled suites.
New scientific case modules expose the established `case_specs()`,
`parameters(case_id, variant)`, `expected(case_id, params)` and
`reference_source(case_id)` interface. Their wrapper temporarily installs
its registry into `run.CASES` and restores it, like `focused.py`.
Behavioral evaluation consumes the existing authored prompts/assertions and
case-specific fixtures; it must not reuse scientific pass fields for prose.
Authoring applications operate only on disposable skill copies.

The scientific comparison uses five fresh repetitions, baseline/candidate
arms, 480-second attempts and at most eight concurrent paired blocks. New
scientific tasks disclose required APIs equally. Behavioral attempt policy
must be frozen and explicitly distinguished from the existing NVIDIA-format
configuration before it runs. Missing or ambiguous evidence never defaults
to pass; grader tokens/time are separate from task-agent KPIs.

## Task 1: reproduce and fix Psi4 loading

Owner: `psi4_loader`, initially diagnosis only.

- [x] Trace the failing strict-sandbox gold execution to its exact loader or
  runtime dependency; compare the working and failing environments.
- [x] Prove a minimal repair with a failing regression before implementation.
  Keep network, sibling workspaces, evaluator and credential paths denied.
- [x] Run both real Psi4 gold inputs and isolation probes with the repair.
- [x] Independently review the repair, then execute fresh paired Psi4 attempts.

## Task 2: uncovered scientific contracts

Owner: `coverage_gap_design` initially audits contracts; implementation tasks
will have disjoint case-module/test ownership after that audit.

- [x] Map all 25 missing records to concrete outcome checks and package symbols.
- [x] Add small application groups, with two nontrivial public/held-out inputs,
  independent expectations and real compiled/API provenance where applicable.
- [x] Test wrong output, missing API execution, malformed inputs and relevant
  sign/ordering/phase/resource distinctions; observe expected failures first.
- [x] Validate private gold applications under the actual worker profile.
- [x] Independently review scientific contracts and integration; freeze and run
  all runnable new cases with and without the skill.
- [x] Link only supported outcomes to the registry, preserving narrow scopes.

## Task 3: behavioral and authoring evaluation

Owner: `behavioral_eval_design` initially audits the 42 authored cases.

- [x] Identify fixture/expectation defects before any scored attempts. Preserve
  old authored cases and document necessary adaptations in a separate suite.
- [x] Stage only each case's fixtures; distinguish advisory/no-write and
  authorized implementation. Do not expose assertions or prior answers to
  task agents. Activation claims must match actual skill-discovery mechanics.
- [x] Add a small separate authoring set that creates/refines a real record or
  family route in disposable copies and checks the resulting artifacts.
- [x] Freeze outcome-based rubrics, including an arm-identifier-masked review protocol;
  test scoring against positive and negative evidence, not keyword matching.
- [x] Run repeated no-skill/candidate attempts; preserve task and grading
  evidence and separate their token/time accounting. Review every failure.

## Task 4: evidence and final verification

Owner: root, with independent review.

- [x] Audit exact attempt counts, frozen inventories, all outcomes and KPIs.
- [x] Update coverage/evaluation documentation with precise scopes and blockers.
- [x] Run relevant evaluator regression tests, coverage tests/checker, structural
  skill validation, links and whitespace checks; confirm CI remains unchanged.
- [x] Report completed work and any externally blocked remainder without
  promising full scientific validity or improved latency.

## Interface review

| Tasks | Shared interface | Decision |
| --- | --- | --- |
| 1 / 2 | `runtime.py` permission/environment helpers | One runtime owner; new cases consume the reviewed profile without custom broad permissions. |
| 2 / 3 | Native execution and telemetry | Reuse stable helpers; do not change historical numerical cases or accounting. |
| 2 / 4 | Frozen case required APIs and per-feature evidence | Registry updates happen after execution, never from family labels alone. |
| 3 / 4 | Semantic outcomes vs numerical checks | Keep separate evidence fields/reports; never inflate scientific coverage from prose. |
| All | Dirty checkout and earlier results | Work in place as previously requested; preserve unrelated work and use before-snapshots for review. |

## Progress and decisions

- Initial state: coverage consistency passes with 52 records, 32 support
  references and 27 current-record scoped scientific mappings. Prior work is
  still uncommitted; no new Git actions are authorized.
- Preserved the pre-follow-up evaluator in
  `/tmp/cudaq-e2e-completion-before-s2jpyf_x/e2e` for review comparisons.
- Three independent read-only audits dispatched: Psi4 root cause, missing
  scientific contracts, and authored behavioral fixture feasibility.
- Baseline evaluator regression is running before implementation. The original
  supported runtimes and native CLI still exist; ripgrep is supplied through
  the same bundled executable used by prior campaigns.
- Process adaptation: user-requested skills-only edits and no Git mutation
  override default worktree/commit/`.superpowers` artifact locations. This
  ledger is the durable execution record; it must not be deleted on completion.
- Baseline regression completed: 83 passed, 2 optional tests skipped, 40
  subtests passed in 124.24 seconds. No baseline failure to repair.
- Gap audit grouped the 25 missing records into three state-preparation cases,
  three host/simulator/Trotter cases and the existing real Psi4 case. The sole
  registry path error, root `make_trotter_terms`, was corrected to its actual
  `cudaq_algorithms.trotter.make_trotter_terms` path. The operational reference
  was already correct and is unchanged.
- Implementation ownership: `coverage_gap_design` owns
  [state-preparation cases](stateprep-completion-task.md); root owns
  `e2e/extended_classical.py`, its tests and the combined wrapper;
  `behavioral_eval_design` owns the [separate behavioral suite](behavioral-completion-task.md);
  `psi4_loader` owns the [minimal runtime repair](psi4-completion-task.md).
- Psi4 diagnosis proved oneMKL's `/proc/self/exe` lookup is blocked. A narrow
  self-process read allows both gold geometries (numeric error at most
  2.65e-14); other-process, sibling, evaluator and symlink-escape probes remain
  denied. Implementation limits the grant to detected Psi4 conda runtimes.
  The sanitized self environment becomes readable; credentials are not added.
  Its fixed cwd `timer.dat` will be routed by a checked relative symlink into
  `.tmp`, with link tampering still treated as a scope failure.
- Host/simulation/Trotter test-first evidence: three independent hand-reference
  tests failed with missing `extended_classical`; after implementation, four
  deterministic tests pass. Final real-runtime tests: seven passed, including
  six isolated gold executions. The wrapper integration tests also passed (2).
- The three state-preparation groups passed ten tests, including six real
  isolated gold executions, and map exactly 15 previously uncovered records.
  Independent review is underway; no scored attempts have started.
- Independent runtime/host review found no severity findings and approved the
  narrow Psi4 repair and new host/simulation/Trotter suite for fresh preflight.
  The full evaluator regression suite is being rerun with both real runtimes.
- Behavioral pre-freeze review is checking equivalent-artifact fairness,
  immutable fixture permissions, imported-evaluator inventory and interrupted
  attempt handling. Arm identifiers are masked for semantic grading, but
  treatment may remain inferable from preserved answer/read evidence.
- State-preparation review found the original single-excitation inputs did
  not distinguish fermionic parity from parity-free ladders. The implementer
  is adding an occupied intermediate spectator and an explicit mutant-rejection
  test before any scored attempt. This is an evaluator fixture repair, not a
  skill/package change or a relaxed tolerance.
- An initial follow-up regression run reported 99 passed, 14 skipped, 28
  subtests. Some runtime switches were absent; it is not an all-provider
  verification. A new run enables every documented real-runtime switch.
- The existing coverage checker still passes with 27/52 current-record
  scientific mappings. No new mapping is being promoted from gold tests.
  Git inspection found no changes in CI configuration or `config.yml`.
- The parity correction passed 11 tests including six real gold applications;
  independent re-review resolved both findings and approved final preflight.
  The new `coverage-20260913` campaign is prepared with 70 fixed attempts,
  and all-case numerical/isolation preflight is running. Evaluator Python
  sources are now frozen; do not edit them while this campaign is active.
- All seven campaign gold preflights passed, including both real Psi4 inputs.
  The fixed 70-attempt scientific comparison has started. Do not alter its
  cases, evaluator Python, frozen snapshots or denominators while it runs.
- Full fresh evaluator verification with every documented runtime switch:
  **114 passed, 50 subtests passed in 272.67 seconds**, no skips. This includes
  real CUDA-Q, PySCF and Psi4 under their restrictive child profiles.
- Preservation check compared the current application entry and all live
  references to the completed refactor campaign's frozen candidate: all
  **85 files are byte-identical**, with no missing operational files.
- Independent registry mapping audit confirms exactly 25 unique target records
  across the seven new campaign cases. Trotter factory records also list root
  aliases, but this campaign explicitly traces the module-qualified aliases;
  promotion must not claim every spelling was separately exercised. Retain
  earlier Psi4 blocked evidence. Behavioral results will be linked from
  evaluation documentation, never inserted as scientific feature executions.
- Behavioral contract review found two unfair assertion partitions and
  implicit authoring output paths. Explicit splits now keep ambiguity/source
  honesty and shared fixture inspection scored, separate only skill access
  diagnostics, and name authorized paths identically in both arms. Independent
  re-review resolved all three findings. Full runner review is still pending.
- Existing coverage consistency regression: all 17 tests pass. No checker or
  CI implementation has been modified.
- Root independently ran the behavioral input-write/chmod denial test under
  normal outer execution approval: 1 passed in 0.31 seconds. The same strict
  profile is retained. Structural skill validation also passed using the
  system interpreter (the scientific venv lacks the validator's PyYAML).
- Full behavioral review found additional pre-freeze defects: interrupted
  attempt accounting, overly literal authoring links, unvalidated optional
  artifacts, overclaimed content-read observations, and staging complete
  target records into authoring tasks. These are being repaired before any
  behavioral model attempt, with equivalent/negative regression cases.
- The behavioral worker must not use the scientific interpreter: some authored
  tasks explicitly require CUDA-Q to be unavailable. A clean stdlib-only venv
  was created at `/tmp/cudaq-behavioral-runtime.K8q5lC/venv`. Root proved its
  imports exclude both packages, but the shared minimal filesystem still
  exposed the global CUDA-Q source. `coverage_gap_design` now owns a separate
  behavioral-only runtime adapter/test to deny global package roots and pin
  PATH-resolved Python to the clean interpreter. No frozen e2e code changes.
- `coverage-20260913` completed: 70/70 strict and scientific/API passes,
  35/35 per arm, zero blocked/interrupted attempts. Native final counters are
  complete for all 70. Candidate total tokens +0.6372%, mean attempt time
  −1.3428%, mean independently verified time −1.1463% versus no skill.
  Aggregate KPI results are essentially neutral, not a general speedup claim.
- New 25-record evidence links preserve all old executions and narrow scopes;
  the unchanged checker passes with **52/52 current-record scientific
  mappings**. See [scientific analysis](results/coverage-20260913/ANALYSIS.md).
  Independent final campaign/mapping audit is underway.
- Independent scientific audit confirmed all 70 attempts, 140 checks, native
  final counters, frozen inventories, 25 new mappings and preserved historical
  blockers; no findings. The scientific work is complete.
- Behavioral implementation and clean-runtime review passed; root freshly ran
  all 36 tests, including real sandbox probes. The first preparation
  `behavioral-20260913` exposed a relative-output-path defect during inspection
  before any model attempt. Its manifest/preflight are retained as zero-attempt
  harness evidence, not passing campaign isolation evidence. Callable/CLI paths
  are being normalized and regression-tested; use a fresh
  `behavioral-20260913-v2` freeze after independent re-review. No scored
  behavioral attempts have been discarded or retried.
- The absolute-output repair passed 39 tests in root's fresh run (0.89 seconds)
  and independent re-review. A fresh `behavioral-20260913-v2` preparation and
  both strict preflights passed with real absolute manifest-denial paths.
  A separate native grader-format smoke correctly produced one expected PASS
  and one expected FAIL with valid citations in 4.83 seconds. It is recorded
  separately and excluded from all campaign denominators.
- The fixed **270 behavioral/authoring task attempts are running** in v2,
  with 18 evaluator hashes, three source-contract hashes and 19 fixture hashes
  frozen. Do not edit evaluator Python, case/adaptation JSON or staged skill
  content until both workers and graders finish. Preserve every attempt.
- Independent live audit of five finalized behavioral workers across both
  arms found correct absolute output paths, complete matching native counters,
  clean-runtime/guide isolation and unchanged scope. `rg` is not staged in
  this separate suite; workers use fallbacks after failed searches. Retain that
  environment limitation when interpreting discovery/token/time results, and
  do not change tools midway through the frozen campaign.
- Independent scientific cost-counter review found uncached input increased
  5.54% despite total tokens increasing only 0.64%. The analysis now separates
  cached/uncached/output counters and calls the aggregate result near-neutral
  total-token volume/time, not monetary cost. Actual billing rates/charges
  remain unmeasured.
- Scientific telemetry wording was also corrected after raw-counter review:
  summed observed request-level counts differ from final input/total by
  5,586; observed cumulative totals agree with final counters.
- Final behavioral regression: **39 passed in 0.94 seconds**, including all
  five clean-runtime probes. Two preceding verification invocations omitted
  required environment settings (collection failed without the module path;
  then 34 passed/5 skipped without the worker path). These were command setup
  mistakes, not campaign attempts or ignored test failures. The README now
  gives the complete invocation. Structural skill validation and tracked
  whitespace checks also pass; final campaign grading/audit remains pending.
- Independent preservation audit: all 85 operational hashes match the prior
  refactor candidate and new scientific snapshot; the original `evals.json`,
  `config.yml` and ten tracked fixtures are unchanged from HEAD. CI paths have
  no worktree changes. Historical campaigns retain every planned result
  identity (150, 225 and 45 respectively), summaries and prior snapshots.
  Their generated output trees are untracked and lack detached full-tree
  checksums, so this proves internal completeness, not independent byte-level
  immutability of every historical output. Coverage consistency passes with
  52 current-record scientific mappings and no errors.
- All 270 behavioral/authoring workers finalized with native completion;
  268 passed deterministic scope/artifact checks. Two baseline attempts left
  compiler backend-probe files, with one also leaving a frontend core dump.
  Independent review traced these to actual `nvq++` subprocess side effects
  outside authorized `.tmp/`, not evaluator fabrication. Keep both strict
  failures and their tokens/time.
- The clean default Python premise holds, but `/usr/local/cudaq` and inherited
  `nvq++` remained accessible: three baseline trajectories (3/270 overall,
  3/135 baseline, 0/135 candidate) inspected core CUDA-Q tools/headers/libraries.
  No target `cudaq_algorithms` source access or successful provider execution
  was observed. The generic staging prompt's package-source absence wording
  is overbroad; disclose this environment limitation and avoid attributing
  the two scope failures solely to model behavior. Do not alter the frozen
  campaign or silently retry. All 270 fresh separate graders are now running.
- Original grading exposed frequent invalid evidence citations and some
  inconsistent semantic interpretations. The original prompt supplies arrays
  without explicit line IDs, and its schema allows arbitrary citation sources
  and out-of-range indices. Strict raw failures therefore mix worker outcomes
  with measurement failures and must not be advertised as quality uplift.
- Corrective measurement plan: add a standalone grading-only protocol in new
  files, preserving every original worker and grade. Freeze all 270 saved
  attempts after original grading finishes, retain identical assertions and
  evidence content, render explicit evidence IDs and constrain citations to
  those IDs. Regrade **all** attempts once in a separately labeled campaign;
  never select only failures, repair workers or overwrite original scores.
  Validate protocol changes before new grades; keep new grader costs separate.
  This is a diagnostic measurement sensitivity study, not a fresh worker
  comparison or confirmation of general judge accuracy. Invalid judgments
  remain measurement failures, never worker failures or implicit passes.
- Original behavioral grading completed: 270 attempts, 269 native completions,
  201 schema/citation-valid grades, 68 citation-invalid grades and one native
  480.21-second timeout. Raw strict score is 78/270, but its failure label
  conflates measurement and worker failures; do not report it as calibrated
  quality. Original workflows remain separate (71/252, 7/18). All 192 raw
  failures were reviewed; [the audit](results/behavioral-20260913-v2/AUDIT.md)
  preserves exact coverage and interpretation disagreements without rescoring.
- Independent worker accounting confirms all 270 final counters exactly match
  raw native events. Original-case candidate totals are +164.17% and mean
  attempt time +69.34%; authoring +60.25% and +62.05%. These are all-attempt
  resource costs, not time-to-verified-solution, dollars or context occupancy.
  All 270 peak-completeness fields are false. The six generated provider
  record artifacts per arm have median 39 versus 147 lines; selector edits
  and the advisory authoring case are excluded from that size comparison.
- Corrected-grader code is implemented in new files only. Initial eight tests
  passed; independent review requested prelaunch/interruption accounting
  repairs before freezing. Root additionally requires blank-content citations
  to be rejected while preserving every original evidence row. A synthetic
  native schema smoke is separate from all task/grading campaign counts.
- Corrected grader passed independent re-review and **48/48 behavioral tests
  in 1.44 seconds**, including all clean-runtime probes. Its synthetic native
  schema control returned the expected PASS/FAIL in 7.63 seconds. The frozen
  `behavioral-regrade-20260913` inventory contains 270 retained workers, 900
  source-evidence files, 17 evaluator files and 194 unchanged assertions.
- Exact corrected-workspace isolation preflight passed: both campaign
  manifests, a prior candidate answer, evaluator source and outside canary
  are denied; scratch works, package imports and network are denied. The
  initial probe mistakenly placed socket creation outside its exception
  check; its expected permission-denial failure is retained separately.
  Correcting that probe did not alter runtime permissions or model attempts.
- The corrected **270 one-shot grading attempts are running**. Source workers,
  original grades, all evaluator Python and projected assertions/evidence are
  frozen. Do not edit these until completion. These are additional diagnostic
  judgments, not 270 new worker attempts. Record grader costs separately.
- Corrected grading completed: **270/270 native-complete valid grades**, zero
  measurement failures, 110/270 recorded strict passes (100/252 original,
  10/18 authoring). All 1,164 assertions are accounted for: 775 PASS, 379 FAIL,
  10 UNCLEAR. Forty prior strict failures become passes and eight passes become
  failures over identical workers; this is grading sensitivity, not worker
  improvement. All 900 frozen evidence files and 17 evaluator files verified.
- Corrected grader costs are **3,074,210 native tokens** and 3,062.38 summed
  seconds across 270 attempts, separately from all original grader and worker
  resources. Root matched all native completion counters independently.
- Three independent reviewers audited all 152 corrected original-case strict
  failures; root audited all eight authoring failures. They also examined
  changed outcomes and selected passes. Citation mechanics are repaired, but
  inconsistent final-versus-consulted evidence, register geometry, execution
  fallback and prompt/rubric requirements remain. No score-aware worker repair,
  human-adjusted score or further post-hoc campaign was performed.
- Fresh final behavioral regression: **48 passed in 1.41 seconds**, including
  all five real clean-runtime probes. Scientific code remains unchanged since
  its **114 tests plus 50 subtests passed with no skips**. Independent final
  coverage regression passed **17/17**; checker reports **52/52 current-record
  scoped mappings**, zero errors. Structural skill validation passes.
- Independent final preservation review again matched all 85 operational
  files to both prior refactor and new scientific snapshots, and all 12
  original fixture/configuration files to HEAD. Read-only Git inspection
  found no CI or YAML changes. No commit, push or Git mutation was performed.
- Final documentation checks resolve every local link in the eight follow-up
  entry/report/audit files. Tracked whitespace checks and a separate check of
  55 maintained evaluator/checker/coverage files pass. The reports disclose
  incomplete context telemetry, toolchain exposure and semantic-judge
  limitations; these are not silently recast as passing validation.
