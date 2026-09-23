# Progressive-disclosure refactor and comparison plan

> For agentic workers: use scoped implementation and independent review; follow the approved chat/tree and the task ownership below. This is evaluation/maintenance documentation, not application guidance.

## Goal and approved specification

Implement the user-approved `references/`, `authoring/`, `coverage/`, and
`scripts/` separation, preserve the information being moved, then measure the
candidate against both the previous skill and no skill. The approved tree in
the conversation is the specification. The existing modular `evals/e2e/`
replaces the sketch's proposed standalone `grader.py`; `discover.py` remains
deferred until routing evidence justifies it.

## Global constraints

- All maintained changes stay under `skills/cudaq-algorithms/`; no extra YAML,
  `.agents/` migration, Git mutations, commits, pushes, or CI-file edits outside
  this directory. Preserve unrelated `.superpowers/` and `docs/superpowers/`.
- Preserve scientific contracts, unsupported/unverified boundaries, source/test
  pointers, review history and open discussions. Move information; do not
  invent package behavior or promote a record from a family-level result.
- Current package source, docs, tests and runtime stay identical across arms.
  Existing results are immutable historical evidence. No skill/evaluator edits
  after a comparison campaign is frozen and gold-validated.
- Use the native Codex model `gpt-5.5`, low reasoning, CPU CUDA-Q 0.15.1
  `qpp-cpu` fp64 and the existing private interpreters. No mock providers,
  relaxed worker isolation, credential copying, global config changes or
  unapproved remote services.
- Keep five fresh repetitions, fixed budgets, balanced randomized ordering and
  all failures. Native final counters are token authority. Cached input and
  reasoning are subsets. Observed peaks are not complete context occupancy;
  unavailable telemetry remains null. Do not promise KPI improvement.

## Preservation and baseline

The unchanged operational package is copied to
`results/refactor-20260911/previous-skill/`; the unchanged evaluator is copied
to `results/refactor-20260911/previous-evaluator/`. The completed baseline is
`results/e2e-20260911-full/`. These are generated audit/comparison artifacts,
never routed to application agents. Preserve them.

The baseline has 140 real attempts, 70/70 scientific/API successes per arm,
64/70 strict successes per arm, higher skill-arm tokens/time, and twelve
bytecode-footprint failures. It motivates an efficiency refactor, not new
scientific rules or a claim that the skill prevents an observed numerical
failure. New retrieval/boundary cases are separate from the historical corpus.

## Task 1: content migration and runtime discovery (root)

Files: `SKILL.md`, `references/**`, `authoring/**`, `assets/` template moves,
`coverage/history.md`, `results/refactor-20260911/migration.json`.

- [x] Capture old files and check baseline inventories before edits.
- [x] Extract advisory/implementation workflows; make the root a small selector
  with essential scientific and authorization boundaries and one conditional
  authoring pointer.
- [x] Make the root catalog a family/operation selector. Retain detailed rows
  in family selectors, with focused records directly discoverable by API name.
- [x] Split conventions into ordering, kernel-boundaries, spectral-processing,
  resource-levels and numerical-comparison; extract the convention template.
- [x] Extract state-preparation injection semantics, HF/UCC detailed validation
  and UCC parameterization. Keep smaller existing records and family folders.
- [x] Split architecture into authoring architecture, record-design,
  capability-design, source-review, extension-workflow and design-decisions.
  Extract primitive/representation/capability/convention templates.
- [x] Relocate lifecycle, historical provenance and eval bookkeeping from all
  operational records. Keep current-source lookup, scientific checking,
  restrictions and semantic uncertainty operational.
- [x] Record each moved section's old/new destination. Check preserved
  scientific text and references against the snapshot and review the diff.

## Task 2: fair three-arm evaluator (one scoped implementer)

Files owned: `evals/e2e/run.py`, `runtime.py`, new `comparison.py` if reporting
needs a separate focused unit, `tests/test_runner.py`, `tests/test_comparison.py`,
and `evals/e2e/README.md`. Do not edit scientific case modules, numerical
grading, telemetry accounting, operational skill or old result artifacts.

Interface: `prepare --previous-skill PATH` selects three arms: `baseline`,
`previous`, `skill` (candidate). With no option, preserve existing two-arm
behavior. Every with-skill prompt uses the same text and local skill path.
Store per-arm snapshots/inventories in the manifest and verify them at freeze,
preflight and run/resume. Stage only application skill content, never authoring,
coverage, evaluator, prior results or the other skill version. Balanced case /
repetition blocks contain one of each arm and run serially within a block.

- [x] Write and run failing tests for three-arm staging, previous-skill
  isolation, balanced ordering, immutable arm inventories and reports.
- [x] Implement the interface above without weakening existing isolation,
  artifact checks, timeout, resume or telemetry behavior.
- [x] Fix bytecode noise equally using a task-local cache under `.tmp/`; test
  real explicit `py_compile` writes stay inside allowed paths. Permit a real
  task-local `rg` executable equally, record its path/version, and test worker
  access. Do not fabricate a replacement `rg` implementation.
- [x] Report strict versus scientific/API outcomes separately, all-attempt
  token/time distributions, paired candidate-minus-previous and
  candidate-minus-baseline metrics, conditional verified-solution timing,
  blocked slots and telemetry completeness.
- [x] Run focused regression tests and provide a concise report with red/green
  evidence. No model campaign is launched by this task.

Example consumer assertions (literal independent fixtures):

```python
assert set(manifest["arms"]) == {"baseline", "previous", "skill"}
assert not (baseline / "skills").exists()
assert (previous / "skills/cudaq-algorithms/SKILL.md").read_text() == "old skill"
assert (candidate / "skills/cudaq-algorithms/SKILL.md").read_text() == "new skill"
assert changes_after_explicit_py_compile == []
# One pair with baseline 100, previous 120, candidate 90 tokens:
assert comparison["candidate_minus_previous"] == -30
assert comparison["candidate_minus_baseline"] == -10
```

## Task 3: feature coverage and consistency (after document paths settle)

Files: `coverage/features.json`, `coverage/policy.md`,
`scripts/check_coverage.py`, `scripts/tests/test_check_coverage.py`.

Registry contract: each feature has a stable `id`, `family`, operational
`record` path, public `symbols`, authored behavioral eval IDs, and explicitly
scoped executed case evidence. Preserve the 52 selectable operation/object
contracts from the old catalog; do not infer all-leaf coverage from nine
family labels. Register shared operational support separately from selectable
features, without pretending a template is a scientific primitive.

- [x] Write failing CLI tests with tiny real fixture trees: valid registry
  exits zero; missing record, duplicate ID, unknown eval/case, broken link,
  fabricated executed evidence and unregistered selectable record exit nonzero.
- [x] Implement a stdlib-only consistency command with readable diagnostics;
  resolve evidence within the skill tree and distinguish authored, executed,
  numerical success and blocked states.
- [x] Populate truthful per-feature mappings from source symbols, old catalog
  and actual grading traces; allow uncovered/partially covered features with
  explicit scope rather than inventing a pass.
- [x] Document the exact non-model CI command. Do not edit external CI YAML.

```text
python skills/cudaq-algorithms/scripts/check_coverage.py
python -m unittest discover -s skills/cudaq-algorithms/scripts/tests
```

## Task 4: focused cases, review and full rerun

Files: new focused eval module/tests under `evals/e2e/` as required,
`evals/EVAL.md`, generated comparison outputs under `evals/results/`.

- [x] Add a bounded, separately scored set of natural-language routing,
  composition and boundary tasks with independent outcome checks; use varied
  scientific inputs where numerical execution applies. Do not disguise these
  as repetitions of the unchanged historical inputs or pool scores together.
- [x] Independently review content preservation, route reachability, scientific
  boundaries, registry/CI checks, and three-arm isolation/reporting.
- [x] Run all deterministic tests, skill structural validation and live gold /
  isolation preflight. Record provider blockers without spending model calls
  on known-broken cases.
- [x] Freeze candidate and evaluator, then run all runnable historical cases
  in all three arms with five repetitions (210 runnable slots, plus 15 Psi4
  slots if its gold runtime remains blocked); run focused cases separately.
- [x] Verify attempt counts, hashes, tokens and output artifacts, then report
  results and regressions without cherry-picking or tuning during the run.

## Interface review and decisions

| Tasks | Shared interface | Finding / ruling |
| --- | --- | --- |
| 1 / 2 | Candidate directory vs preserved previous-skill root | Stage SKILL.md/references and existing assets as the original evaluator did. Candidate templates move to authoring and are not application context; preserve old assets in the previous arm for fairness. |
| 1 / 3 | Stable old-catalog contracts vs new record paths | Registry is populated after paths settle; old snapshot retains the source inventory. |
| 2 / 4 | Three-arm manifest, gold gate, frozen Python hashes | No scientific case changes during the historical comparison. Additional cases freeze before their own runs. |
| 3 / 4 | Feature evidence and current campaign results | Historical evidence remains scoped to its skill version; candidate evidence is linked only after completion. |
| 1 | Moving information vs smaller runtime context | Preserve raw old text in generated snapshot and relocate substantive sections; do not put all historical prose into the new runtime. |
| 2 | Harness corrections vs causal comparison | Apply corrections to all three fresh arms; compare within this new campaign, not old-vs-new wall time as if environment were unchanged. |
| 3 | CI requested vs skills-only writes | Deliver checker/tests/CI command here; external workflow wiring remains outside authorized file scope. |
| 4 | Benefit hoped for vs fixed experiment | Report actual outcomes including worse KPIs; no stop-on-pass or post-hoc score relaxation. |

Ruling: continue in the user-selected `algo-skill` checkout, preserving the
dirty worktree and avoiding skill defaults that create worktrees, `.superpowers`
ledgers or commits. The approved tree and this durable plan replace those
default artifact locations; review uses snapshot/worktree diffs.

## Progress

- 2026-09-11: user approved implementation and rerun. Previous skill and
  evaluator snapshots captured before any operational edit. Task 1 and scoped
  Task 2 have disjoint file ownership; root may implement documentation while
  one implementation agent handles evaluator changes.
- 2026-09-11: migration preserves 52 original leaf paths and catalog rows.
  Root/catalog reduced from 201/124 to 41/22 lines. Independent review found
  two provider-compatibility caveats needed on direct leaf routes; restored
  them with family links. Repaired authoring cross-references and clarified
  template fields' coverage destinations.
- Coverage checker TDD: initial 11 failing tests before implementation, then
  11 passed; four added scientific/hygiene, runtime-only routing, schema and
  historical-index tests failed before fixes; all 15 now pass. Actual registry
  checks 52 features and 32 support references, with 26 scoped historical
  scientific mappings and zero claims for changed current record bytes.
- Evaluator review identified and implementer fixed canonical grading-source
  equality, paired observed-input-peak deltas and all-planned missing-pair
  counts. Independent recheck confirmed all three; focused suite tests and
  live gold execution still precede freezing.
- Shared workflow text remains repeated. Each application mode is independently
  readable and does not route to the other; shared extraction is deferred
  rather than expanding the approved tree before measuring retrieval.
- Fixed campaign execution uses eight concurrent case/repetition blocks on
  this 16-CPU container, with one arm at a time per block and worker BLAS/OMP
  threads fixed to one. The two suites run sequentially, use five repetitions
  and 480-second attempt budgets, and keep identical concurrency for all arms.
  This is a fresh within-campaign comparison, not a causal comparison to the
  earlier campaign's controller wall time.
- Independently rechecked the 77 preserved original files against the old
  campaign inventory: zero hash mismatches. Coverage review's historical
  baseline-hash fallback and selectable-to-support demotion reproductions
  failed as expected, then passed after fixes; all 17 checker tests pass.
- Focused-suite independent review found hidden convenience-helper
  requirements could reject otherwise valid direct packaged circuit paths.
  Fixing only those focused API requirements before preflight; numerical
  oracles and historical tasks remain unchanged. Prepared manifests contain
  225 historical and 45 separate focused slots; no model attempts yet.
- Final focused review also required common compiled-device evidence, so
  unused factories plus dense answers cannot satisfy genuine circuit use.
  Two new negative tests failed then passed with that fix; required compiled
  names match both successful helper and direct traces. Reviewer cleared all
  Important findings. Refreshing preflight after the correction; no model
  attempts were launched against the earlier grading definition.
- Final root regression: 83 passed, 2 optional classical module tests skipped,
  40 subtests passed in 124.24 seconds. All six helper/direct routes pass on
  both focused inputs with genuine circuit traces. Final focused preflight
  passed three isolation checks and all six gold executions. Discovery model
  campaign started first (45 slots); historical suite runs after it. The first
  historical preflight correctly rejected evaluator drift after fourteen
  science cases passed and Psi4 blocked; refreshing it with the final hashes.
- Final all-family preflight passed six isolation checks and 28 gold numerical
  executions; both real Psi4 variants remain MKL-loader blocked. All current
  evaluator files match its successful freeze inventory.
- Discovery campaign completed 45/45 attempts: baseline 15/15, previous 14/15,
  candidate 15/15 strict/scientific successes. The previous-arm failure is an
  incorrect symmetric QSP self-oracle that rejects its own circuit on the
  held-out doubled-phase contract; phase mismatch predicts the observed error
  0.038446698690243586. The failure and application remain untouched. Candidate
  total tokens are 3,290,946 vs previous 3,422,630; summed attempt time is
  1,187.420563s vs 1,145.524029s. No latency improvement is claimed. Main
  all-family campaign started after discovery completed (210 runnable slots).
- Final all-family comparison completed 225/225 records: 210 executed, all
  strict/scientific passes, plus 15 unattempted Psi4 slots. Independent audit
  verified all identities, counters, inventories and report arithmetic in both
  suites. No missing pairs or final freeze mismatches. Current registry links
  27 records to narrowly scoped matching candidate evidence; all 17 checker
  tests and structural skill validation pass. No operational/evaluator changes
  were made during or after the campaigns; final writes only backfill evidence
  and report results. See [completed outcome](results/refactor-20260911/RESULTS.md)
  for the mixed KPI result and explicitly deferred work.
