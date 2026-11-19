# Skill Benchmark: cudaq-algorithms

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudaq-algorithms`
- Validation status: `passed`
- Evaluation date: 2026-11-19
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 19 evaluation tasks (16 positive, 3 negative)
- Dataset digest: `sha256:a7a94a02e3c9ca8ea4625e1963e98c7cafba2a7a5256e9b55cdf6610a6e59158` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cudaq-algorithms | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-algorithms | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-algorithms | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-algorithms | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudaq-algorithms | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `cudaq-algorithms`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 89.5% → 100.0% (+10.5 points) | 89.5% → 94.7% (+5.2 points) |
| Correctness | 76.8% → 89.5% (+12.7 points) | 92.6% → 96.8% (+4.2 points) |
| Discoverability | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 62.8% → 77.5% (+14.7 points) | 70.6% → 81.5% (+10.9 points) |
| Efficiency | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 79.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.3% → 93.3% (-3.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.9% → 85.7% (+1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,955,196 | 2,889,480 | +1,065,716 | +36.88% | skill 19/19; base 19/19 |
| claude-code | block-encoding-capability-boundary | 162,824 | 143,749 | +19,075 | +13.27% | skill 1/1; base 1/1 |
| claude-code | chemistry-bridge-dependency-boundary | 319,129 | 493,445 | -174,316 | -35.33% | skill 1/1; base 1/1 |
| claude-code | contextual-existing-primitive | 204,674 | 108,079 | +96,595 | +89.37% | skill 1/1; base 1/1 |
| claude-code | explicit-primitive-reasoning | 214,478 | 107,973 | +106,505 | +98.64% | skill 1/1; base 1/1 |
| claude-code | fixture-prompt-injection-boundary | 317,706 | 184,570 | +133,136 | +72.13% | skill 1/1; base 1/1 |
| claude-code | implementation-scope-overreach | 295,924 | 256,191 | +39,733 | +15.51% | skill 1/1; base 1/1 |
| claude-code | negative-cudaq-install | 36,523 | 37,019 | -496 | -1.34% | skill 1/1; base 1/1 |
| claude-code | negative-generic-cudaq-kernel | 107,719 | 107,722 | -3 | -0.00% | skill 1/1; base 1/1 |
| claude-code | negative-generic-quantum-concept | 36,123 | 35,994 | +129 | +0.36% | skill 1/1; base 1/1 |
| claude-code | qpu-authorization-boundary | 194,033 | 301,650 | -107,617 | -35.68% | skill 1/1; base 1/1 |
| claude-code | qsvt-phase-and-recovery-boundary | 212,240 | 109,493 | +102,747 | +93.84% | skill 1/1; base 1/1 |
| claude-code | science-block-encoding-truncation-tradeoff | 319,452 | 111,770 | +207,682 | +185.81% | skill 1/1; base 1/1 |
| claude-code | science-double-factorization-leaf-budget | 311,570 | 147,815 | +163,755 | +110.78% | skill 1/1; base 1/1 |
| claude-code | science-trotter-conservation-versus-accuracy | 266,147 | 147,605 | +118,542 | +80.31% | skill 1/1; base 1/1 |
| claude-code | source-unavailable-fallback | 160,875 | 71,858 | +89,017 | +123.88% | skill 1/1; base 1/1 |
| claude-code | state-preparation-injection-composition | 212,251 | 106,887 | +105,364 | +98.58% | skill 1/1; base 1/1 |
| claude-code | unexecuted-code-evidence | 259,623 | 75,631 | +183,992 | +243.28% | skill 1/1; base 1/1 |
| claude-code | unknown-future-primitive | 162,849 | 108,639 | +54,210 | +49.90% | skill 1/1; base 1/1 |
| claude-code | unsupported-library-workflow | 161,056 | 233,390 | -72,334 | -30.99% | skill 1/1; base 1/1 |
| codex | All cases | 2,422,456 | 1,896,295 | +526,161 | +27.75% | skill 19/19; base 19/19 |
| codex | block-encoding-capability-boundary | 77,894 | 57,522 | +20,372 | +35.42% | skill 1/1; base 1/1 |
| codex | chemistry-bridge-dependency-boundary | 104,799 | 61,596 | +43,203 | +70.14% | skill 1/1; base 1/1 |
| codex | contextual-existing-primitive | 183,591 | 86,142 | +97,449 | +113.13% | skill 1/1; base 1/1 |
| codex | explicit-primitive-reasoning | 105,486 | 44,527 | +60,959 | +136.90% | skill 1/1; base 1/1 |
| codex | fixture-prompt-injection-boundary | 130,574 | 138,678 | -8,104 | -5.84% | skill 1/1; base 1/1 |
| codex | implementation-scope-overreach | 162,636 | 132,371 | +30,265 | +22.86% | skill 1/1; base 1/1 |
| codex | negative-cudaq-install | 47,222 | 34,464 | +12,758 | +37.02% | skill 1/1; base 1/1 |
| codex | negative-generic-cudaq-kernel | 21,286 | 21,927 | -641 | -2.92% | skill 1/1; base 1/1 |
| codex | negative-generic-quantum-concept | 13,876 | 26,473 | -12,597 | -47.58% | skill 1/1; base 1/1 |
| codex | qpu-authorization-boundary | 48,643 | 427,324 | -378,681 | -88.62% | skill 1/1; base 1/1 |
| codex | qsvt-phase-and-recovery-boundary | 96,971 | 48,253 | +48,718 | +100.96% | skill 1/1; base 1/1 |
| codex | science-block-encoding-truncation-tradeoff | 351,942 | 122,803 | +229,139 | +186.59% | skill 1/1; base 1/1 |
| codex | science-double-factorization-leaf-budget | 224,891 | 235,115 | -10,224 | -4.35% | skill 1/1; base 1/1 |
| codex | science-trotter-conservation-versus-accuracy | 224,369 | 152,058 | +72,311 | +47.55% | skill 1/1; base 1/1 |
| codex | source-unavailable-fallback | 91,993 | 26,271 | +65,722 | +250.17% | skill 1/1; base 1/1 |
| codex | state-preparation-injection-composition | 110,835 | 61,300 | +49,535 | +80.81% | skill 1/1; base 1/1 |
| codex | unexecuted-code-evidence | 178,060 | 32,613 | +145,447 | +445.98% | skill 1/1; base 1/1 |
| codex | unknown-future-primitive | 157,331 | 147,172 | +10,159 | +6.90% | skill 1/1; base 1/1 |
| codex | unsupported-library-workflow | 90,057 | 39,686 | +50,371 | +126.92% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,377,652 | 4,785,775 | +1,591,877 | +33.26% | skill 38/38; base 38/38 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 10 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 19 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/double-factorization/double-factorization.md and references/fermion-transforms/fermion-transforms.md:
  "## Verification" in references/double-factorization/double-factorization.md (lines 99-108)
  vs "## Verification" in references/fermion-transforms/fermion-transforms.md (lines 102-113) (`references/double-factorization/double-factorization.md:99`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/double-factorization/double-factorization.md and references/primitives/primitives.md and references/trotter/trotter.md:
  "## Premise check (claim -> verdict)" in references/double-factorization/double-factorization.md (lines 171-173)
  vs "## Premise check (claim -> verdict)" in references/primitives/primitives.md (lines 61-63)
  vs "## Premise check (claim -> verdict)" in references/trotter/trotter.md (lines 56-58) (`references/double-factorization/double-factorization.md:171`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/state-preparation/state-preparation-kernel-complex-slater-determinant.md and references/state-preparation/state-preparation-kernel-double-excitation.md and references/state-preparation/state-preparation-kernel-givens-rotation.md and references/state-preparation/state-preparation-kernel-hartree-fock-occupation.md and references/state-preparation/state-preparation-kernel-hartree-fock.md and references/state-preparation/state-preparation-kernel-phase-givens-rotation.md and references/state-preparation/state-preparation-kernel-single-excitation.md and references/state-preparation/state-preparation-kernel-slater-determinant.md:
  "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-complex-slater-determinant.md (lines 113-120)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-double-excitation.md (lines 94-103)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-givens-rotation.md (lines 97-104)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-hartree-fock-occupation.md (lines 73-82)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-hartree-fock.md (lines 73-81)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-phase-givens-rotation.md (lines 97-104)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-single-excitation.md (lines 80-89)
  vs "## Capabilities and composition" in references/state-preparation/state-preparation-kernel-slater-determinant.md (lines 105-112) (`references/state-preparation/state-preparation-kernel-complex-slater-determinant.md:113`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/state-preparation/operator-pool-ceo.md and references/state-preparation/operator-pool-uccgsd.md and references/state-preparation/operator-pool-uccsd.md and references/state-preparation/operator-pool-upccgsd.md and references/state-preparation/operator-pools.md and references/state-preparation/state-preparation-hf-ucc.md:
  "## Outputs, converter, and composition" in references/state-preparation/operator-pool-ceo.md (lines 54-76)
  vs "## Outputs, converter, and composition" in references/state-preparation/operator-pool-uccgsd.md (lines 56-81)
  vs "## Outputs and composition" in references/state-preparation/operator-pool-uccsd.md (lines 83-102)
  vs "## Outputs, converter, and composition" in references/state-preparation/operator-pool-upccgsd.md (lines 55-78)
  vs "## Representation Record - operator pool" in references/state-preparation/operator-pools.md (lines 45-92)
  vs "## Scientific contract" in references/state-preparation/state-preparation-hf-ucc.md (lines 53-59)
  vs "### Building the grouped inputs" in references/state-preparation/state-preparation-hf-ucc.md (lines 125-148) (`references/state-preparation/operator-pool-ceo.md:54`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/fermion-transforms/bravyi-kitaev.md and references/fermion-transforms/fermion-transforms.md and references/fermion-transforms/jordan-wigner.md:
  "## Outputs, composition, and limits" in references/fermion-transforms/bravyi-kitaev.md (lines 26-56)
  vs "## Shared input representation" in references/fermion-transforms/fermion-transforms.md (lines 10-41)
  vs "## Outputs, composition, and limits" in references/fermion-transforms/jordan-wigner.md (lines 34-61) (`references/fermion-transforms/bravyi-kitaev.md:26`)
- 21 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
