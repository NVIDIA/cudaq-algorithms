# Skill review and migration history

The pre-refactor records were draft documentation. Their exact historical
metadata and authored evaluation mappings are retained by source record in
[the migration inventory](../evals/results/refactor-20260911/migration.json).
Per-feature current coverage belongs in [features.json](features.json); a family
result never establishes all of its selectable contracts.

The historical source-review anchor was
`61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03), with repository CUDA-Q
development pin `b28cf0f6f2d9387f12ca81b6824b8833a38530af`. These are audit
values, not active API or compatibility promises. The old records' statements
such as “not run” describe their recorded review time and are not current campaign status.

The untouched [previous skill](../evals/results/refactor-20260911/previous-skill/SKILL.md)
preserves original bytes. Substantive guidance has live destinations in
`references/` or `authoring/`; this snapshot is not an application route.
The existing [evaluation procedure and history](../evals/EVAL.md) owns prior runs.
The completed baseline `results/e2e-20260911-full/` has 140 real attempts:
70/70 scientific/API successes and 64/70 strict successes per arm, with higher
skill-arm tokens/time and twelve bytecode-footprint failures. This motivates
retrieval efficiency, not new numerical correctness claims.

## Completed progressive-disclosure comparison

The [2026-09-11 refactor outcome](../evals/results/refactor-20260911/RESULTS.md)
records preserved live content and two fresh three-arm campaigns: 210 real
all-family attempts plus 15 blocked Psi4 slots, and 45 separate focused
attempts. Candidate passed 70/70 and 15/15, respectively. Token usage improved
versus the previous layout; latency was mixed. Historical results are retained,
not replaced or reinterpreted as current-version evidence.

The registry now includes exact required-API mappings for both candidate
campaigns. Twenty-seven individual records have scoped scientific evidence
matching their current bytes; the other contracts are not silently promoted.
These runs do not validate every parameter, all 52 contracts, authoring or
the entire authored behavioral fixture suite. See [policy](policy.md).

## Coverage-completion follow-up — 2026-09-13

The [separate seven-case scientific comparison](../evals/results/coverage-20260913/ANALYSIS.md)
completed 70 strict/scientific passes (35 per arm), including real Psi4 under
the repaired bounded runtime. It adds scoped execution evidence for the
remaining 25 records; the registry now has 52/52 individual records with
scientific evidence matching their current bytes. Historical blockers remain
recorded. This does not establish universal parameter, alias, hardware or
workflow validation.

The [behavioral and separate authoring comparison](../evals/results/behavioral-20260913-v2/ANALYSIS.md)
records 270 completed workers, all-attempt token/time measurements and grader
reliability limits. Its original and corrected grading protocols retain
separate evidence and costs. No behavioral judgment is inserted as scientific
feature execution. All 85 operational entry/reference files stayed unchanged;
this follow-up adds evaluation evidence, not new application lifecycle labels.

CI integration remains outside this follow-up's scope.

## Seven-question local diagnostics and harness-fix follow-up — 2026-09-17/18

The [consolidated seven-question report](../evals/results/seven-local-20260917-final.md)
records the 28-attempt local Qwen campaign (0 PASS in both arms), the 12-attempt routing
retest, and the verified diagnosis: a floor effect from the model population plus harness
faults (Codex `apply_patch` mandate consuming ~39% of 30B wall time, a 32k window with an
8.7k-token prefix, an advertised-but-rejected ask tool), and a SKILL.md body carrying no
import, symbol, signature or example payload while 76 references were never opened.

The [harness-fix experiment](../evals/results/harness-fix-20260918/) (36 attempts,
Qwen3-Coder-30B, compact base prompt, 600 s) compared no skill, the current SKILL.md, and the
current SKILL.md plus a verified quick-contract block. Strict scientific passes: 0/9, 0/9, 5/9
(P03 3/3, P04 2/3, P01 0/3); the enhanced arm imported the package in 9/9 attempts and was
faster and cheaper than the current skill in 9/12 and 10/12 paired attempts. Independent
artifact-level grading agreed with the automatic grader on 26/27 attempts. The quick-contract
block is now part of `skills/cudaq-algorithms/SKILL.md`. This is single-model diagnostic
evidence under a modified harness, not acceptance against the strategy brief; trigger
precision/recall and P02/P05/P06/P07 remain unmeasured under the fixed harness. Sources and
controllers are preserved under `evals/harness-fix-20260918/`.

Round 2 (same day, 29 attempts, two GPUs) retried P01 with a rewritten Walk row, checked
P03/P04 for regression, and ran P02/P05/P06/P07 under the fixed harness for the first time.
Independent verdicts: P01 0/5, P02 0/5, P05 0/5, P07 0/5 in every arm; P03 1/2 and P04 1/2 with
v2; P06 2/2 numerically correct with v2 but provenance undeclared (strict 0/2). One attempt ran
a destructive `find -delete` in its sandboxed workspace, a critical-failure class. The repository
SKILL.md carries the v2 body. Judgment tasks and P01 remain unsolved on this model; the
acceptance question requires a run on the target model. Details and evidence in the
[consolidated report](../evals/results/seven-local-20260917-final.md) and
`evals/results/harness-fix-20260918/round2/`.

Round 3 (same day, 14 attempts, two GPUs) applied the fixable items: Trotter/DF/chemistry
wording, a literal verified Walk snippet, a deliverable rule, a base prompt forbidding deletion
and unsupported claims, and grading checks for both. Independent verdicts with the v3 body: P01
3/3, P03 2/3, P06 2/3, P04 0/3, P05 0/2; zero critical failures. Against the brief: baseline
improvement met (+39 pp), execution score not met (39% strict, 78% partial), P02 oracle not met,
trigger precision/recall and the 42 regression evaluations not measured, efficiency criterion not
evaluable. KPI table and evidence in the [consolidated report](../evals/results/seven-local-20260917-final.md)
and `evals/results/harness-fix-20260918/round3/`. `skills/cudaq-algorithms/SKILL.md` carries v3.

Round 4 and trigger measurement (same day). Serving the 30B at 64k context removed the
compaction failures on P03/P04 (5/6 versus 2/6 at 32k, zero compactions, half the time); the
recommendation is to evaluate locally at 64k from now on. A trigger measurement with the skill
registered but not injected found the 30B never opens a listed skill (0/19 positives, 0/10
negatives), so trigger precision/recall cannot be evaluated on this model. A Nemotron-3 Super run
on NVIDIA Build (none vs v3, seven cases x 3) is in progress; its results are added to the
consolidated report when complete. Sections 7-8 of the report hold the details; evidence under
`evals/results/harness-fix-20260918/round4-64k/` and `.../trigger-30b/`.

Nemotron-3 Super via NVIDIA Build (2026-09-18, declared primary target; the Codex and Claude runs
follow as the roadmap's cross-model adaptability step once the Nemotron work is settled).
Main run, skill v3 versus none, seven cases x 3, independently graded: 8/21 versus 1/21. Ten
skill iterations followed (v5-v12), each written only after a specific observed failure and each
snippet verified by execution before it entered the text: protocol-composition row and the
apply_kernel recipe (P02), never-zero-an-absent-constant, sentence-plus-bold-items request wording,
sector-projection one-liner and pinned chemist index order (P05), bold missing-items list and the
inventory-boundary question (P07), prepared-state read-back (P01). Confirmation run of v10 on all
seven tasks x 3 with independent verdicts: 13/21 = 62% (P01 0/3, P02 2/3, P03 3/3, P04 3/3,
P05 1/3, P06 2/3, P07 2/3), medians 156 s / 133k tokens versus 403 s / 276k without the skill;
round 11 with v12 on P01: 2/3, both passes through the new read-back check. Trigger precision/recall with the skill listed only:
recall 32%, precision 55% (the model follows listings; the description does not discriminate).
Regression suite, 42 evaluations, judged with the evaluations themselves under scrutiny: 10/42
pass, 15 of 32 non-passes are timeouts, 4 evaluations flawed and 14 ambiguous (stale reference
file names, a false premise, a convention conflict with the skill). Two attempted boundary
violations were blocked by the sandbox (a fixture overwrite; a credential hunt plus paid-QPU
submission attempt); the harness key was verified unexposed. Dominant residual failure is a
degenerate bare-`cd` loop (10 of 109 attempts); base prompt v5 against it is drafted, not run.
`skills/cudaq-algorithms/SKILL.md` carries v12. Section 9 and the Nemotron KPI table in the
[consolidated report](../evals/results/seven-local-20260917-final.md); evidence under
`evals/results/harness-fix-20260918/nemotron/`.

Regression suite repair (same evening). The 42 evaluation definitions were validated against the
source, the current skill and the harness by eleven validator agents and eleven adversarial reviewers:
28 changed (stale reference names, role-only file references, one false harness premise, one
convention conflict, vacuous or unanswerable assertions), 14 sound, 0 rejected; the pre-fix file and a
per-case change log are preserved under `evals/results/harness-fix-20260918/nemotron/`. The
validators also found a skill defect (the `stateprep.uccsd` row lacked the required `spin`
argument), fixed as v13. The repaired suite was rerun with v13 at 900 s and judged with
reference-record reads required by command: 6/42 pass (4 of 39 positives, 2 of 3 negatives), 6 timeouts, 41/42 evaluations rated sound, 2 fabricated-result critical failures (operator-pool-selection-boundary, unexecuted-code-evidence); the dominant failure is that the model does not open reference records when the skill body is injected. A two-arm test on the 27 unopened-record failures (skill-text routing rule versus harness-injected family records) raised record reads from 0 to 5/22 and 9/21 but converted no failure into a pass; the remaining regression failures are boundary reasoning the model does not do from any text. Cross-model check on the same 27 evaluations with Codex production (gpt-5.6-sol, skill v14, native delivery): 15/27 pass, records opened in 26/27 attempts, 87% of assertions met (Nemotron: 0/27, 0/27, 39%). Full suite on Codex production (gpt-5.6-sol, skill v14, native; positives injected, negatives listed-only): 28/42 pass (positives 25/39, negatives 3/3), records opened in 38/39 positive attempts, 0 timeouts, 89% of assertions met (Nemotron v13: 6/42, 4/39, 2/3, 0/39, 47%). Skill v15 experiment (2026-09-19, state-preparation premise-check table + four-slot answer shape + scripts/route.py): on the 12 state-preparation evaluations Nemotron arm A (records injected) 0/8 pass, 63% assertions met, 2 false premises accepted, records opened 8/12 (v13: 0/12, 36%, 1, 0); arm B (script mandated, not injected) 0/10, script run 0/12; Codex v15 converted 9/14 of its 14 v14 non-passes (→ 37/42). Text form and answer shape move both models; the script helps only a model that already reads. v15 is the repository text; extension to other families pending. Skill v16 (2026-09-21; diagnosis of the 26 injected-text failures, then unconditional answer headings + evidence rule at the top of SKILL.md, split premise rows with demanded words first and runnable record paths, router listing the focused records): Nemotron on the 12 state-preparation evaluations, arm D (records injected) 4/12 pass, 79% assertions, router run 11/12, median 72 s; arm E (router output under the task) 3/12, 74%, median 85 s (v13: 0/12, 36%, 544 s; v15: 0/8, 63%). First strict Nemotron passes on the boundary suite; residue = omitted specifics, one fabricated result, records listed as consulted but not opened. Codex production KPI campaign (2026-09-21, skill v15): seven build tasks skill 21/21 vs no skill 13/21 (no-skill fails: p05 silent offset cancellation, p07 fabricated feedback + source overwrite), skill median 356 s / 657,624 tokens vs 594 s / 1,192,575; trigger precision 68% recall 100% on 29 listed-only prompts; full 42 under v15 34/42 strict (2 more under corrected readings), 95% assertions, 0 critical. Claude Code KPI campaign (2026-09-22, skill v16, claude-sonnet-5): seven build tasks skill 21/21 vs no skill 8/21, skill median 72 s / 800,162 tokens vs 118 s / 430,229; trigger read precision 95% recall 95% on 29 listed-only prompts; full 42 28/42 strict (1 more under corrected readings), 89% assertions, 0 critical. Skill v17/v17.1 (2026-09-22): premise tables for the double-factorization and Trotter front doors, provider/excitation facts in bold in the state-preparation rows, router prints focused records inline, generated-files rule; boundary suite rerun: Codex 35/42 strict (37 outcome-reached; v15 34/42), Claude 31/42 strict (37 outcome-reached; v16 28/42), critical 0/0. Skill v17.2 (2026-09-23): router prints matched premise rows in full and lists focused records instead of pasting them; full 42 on both models: Codex 35/42 strict, 36/42 outcome, 6 timeouts, median 541 s (v15 34/42, 301 s; v17.1 35/42, 569 s); Claude 30/42 strict, 38/42 outcome, median 22 s, $6.49 (v16 28/42; v17.1 31/42); 0 critical on both. No-skill baseline on the 42 (2026-09-23): Codex no skill 6/42 strict, 27/42 outcome, 3 critical (unrequested rewrites of library sources), 7 timeouts vs v17.2 35/42, 36/42, 0 critical; Claude no skill 5/42 strict, 26/42 outcome, 0 critical vs v17.2 30/42, 38/42; brief table filled with the real baseline.
