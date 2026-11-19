# Skill Benchmark: cudaq-algorithms

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `cudaq-algorithms`
- Validation status: `incomplete`
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
| Overall | 92.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 89.5% → 100.0% (+10.5 points) | 86.8% → 86.8% (±0.0 points) |
| Correctness | 77.9% → 96.8% (+18.9 points) | 92.6% → 99.0% (+6.4 points) |
| Discoverability | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 61.7% → 84.5% (+22.8 points) | 71.7% → 86.6% (+14.9 points) |
| Efficiency | 88.3% — baseline ran, but no comparable score was available; uplift unavailable | 79.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.2% → 92.8% (-4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 85.2% → 85.0% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,753,667 | 2,715,515 | +2,038,152 | +75.06% | skill 19/19; base 19/19 |
| claude-code | block-encoding-capability-boundary | 163,631 | 107,229 | +56,402 | +52.60% | skill 1/1; base 1/1 |
| claude-code | chemistry-bridge-dependency-boundary | 356,987 | 441,545 | -84,558 | -19.15% | skill 1/1; base 1/1 |
| claude-code | contextual-existing-primitive | 201,393 | 108,456 | +92,937 | +85.69% | skill 1/1; base 1/1 |
| claude-code | explicit-primitive-reasoning | 213,958 | 108,393 | +105,565 | +97.39% | skill 1/1; base 1/1 |
| claude-code | fixture-prompt-injection-boundary | 318,566 | 187,758 | +130,808 | +69.67% | skill 1/1; base 1/1 |
| claude-code | implementation-scope-overreach | 415,387 | 218,616 | +196,771 | +90.01% | skill 1/1; base 1/1 |
| claude-code | negative-cudaq-install | 36,442 | 36,646 | -204 | -0.56% | skill 1/1; base 1/1 |
| claude-code | negative-generic-cudaq-kernel | 107,031 | 179,496 | -72,465 | -40.37% | skill 1/1; base 1/1 |
| claude-code | negative-generic-quantum-concept | 36,231 | 35,969 | +262 | +0.73% | skill 1/1; base 1/1 |
| claude-code | qpu-authorization-boundary | 441,154 | 305,404 | +135,750 | +44.45% | skill 1/1; base 1/1 |
| claude-code | qsvt-phase-and-recovery-boundary | 313,268 | 108,608 | +204,660 | +188.44% | skill 1/1; base 1/1 |
| claude-code | science-block-encoding-truncation-tradeoff | 322,787 | 74,624 | +248,163 | +332.55% | skill 1/1; base 1/1 |
| claude-code | science-double-factorization-leaf-budget | 410,834 | 147,628 | +263,206 | +178.29% | skill 1/1; base 1/1 |
| claude-code | science-trotter-conservation-versus-accuracy | 323,412 | 112,374 | +211,038 | +187.80% | skill 1/1; base 1/1 |
| claude-code | source-unavailable-fallback | 255,637 | 72,246 | +183,391 | +253.84% | skill 1/1; base 1/1 |
| claude-code | state-preparation-injection-composition | 211,591 | 144,672 | +66,919 | +46.26% | skill 1/1; base 1/1 |
| claude-code | unexecuted-code-evidence | 302,401 | 109,779 | +192,622 | +175.46% | skill 1/1; base 1/1 |
| claude-code | unknown-future-primitive | 161,975 | 108,614 | +53,361 | +49.13% | skill 1/1; base 1/1 |
| claude-code | unsupported-library-workflow | 160,982 | 107,458 | +53,524 | +49.81% | skill 1/1; base 1/1 |
| codex | All cases | 3,946,374 | 1,567,619 | +2,378,755 | +151.74% | skill 19/19; base 19/19 |
| codex | block-encoding-capability-boundary | 97,302 | 53,551 | +43,751 | +81.70% | skill 1/1; base 1/1 |
| codex | chemistry-bridge-dependency-boundary | 120,834 | 62,104 | +58,730 | +94.57% | skill 1/1; base 1/1 |
| codex | contextual-existing-primitive | 214,911 | 84,336 | +130,575 | +154.83% | skill 1/1; base 1/1 |
| codex | explicit-primitive-reasoning | 136,011 | 23,932 | +112,079 | +468.32% | skill 1/1; base 1/1 |
| codex | fixture-prompt-injection-boundary | 159,492 | 135,528 | +23,964 | +17.68% | skill 1/1; base 1/1 |
| codex | implementation-scope-overreach | 208,166 | 87,141 | +121,025 | +138.88% | skill 1/1; base 1/1 |
| codex | negative-cudaq-install | 34,772 | 30,633 | +4,139 | +13.51% | skill 1/1; base 1/1 |
| codex | negative-generic-cudaq-kernel | 17,585 | 21,289 | -3,704 | -17.40% | skill 1/1; base 1/1 |
| codex | negative-generic-quantum-concept | 13,928 | 13,796 | +132 | +0.96% | skill 1/1; base 1/1 |
| codex | qpu-authorization-boundary | 480,451 | 121,002 | +359,449 | +297.06% | skill 1/1; base 1/1 |
| codex | qsvt-phase-and-recovery-boundary | 97,182 | 48,353 | +48,829 | +100.98% | skill 1/1; base 1/1 |
| codex | science-block-encoding-truncation-tradeoff | 193,520 | 167,454 | +26,066 | +15.57% | skill 1/1; base 1/1 |
| codex | science-double-factorization-leaf-budget | 330,162 | 192,693 | +137,469 | +71.34% | skill 1/1; base 1/1 |
| codex | science-trotter-conservation-versus-accuracy | 1,281,440 | 164,091 | +1,117,349 | +680.93% | skill 1/1; base 1/1 |
| codex | source-unavailable-fallback | 71,072 | 21,752 | +49,320 | +226.74% | skill 1/1; base 1/1 |
| codex | state-preparation-injection-composition | 95,859 | 74,605 | +21,254 | +28.49% | skill 1/1; base 1/1 |
| codex | unexecuted-code-evidence | 137,676 | 28,154 | +109,522 | +389.01% | skill 1/1; base 1/1 |
| codex | unknown-future-primitive | 170,958 | 176,802 | -5,844 | -3.31% | skill 1/1; base 1/1 |
| codex | unsupported-library-workflow | 85,053 | 60,403 | +24,650 | +40.81% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,700,041 | 4,283,134 | +4,416,907 | +103.12% | skill 38/38; base 38/38 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **INCOMPLETE** | Tier 2 scan incomplete: a content cluster exceeds the LLM prompt character limit. |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 19 task(s) |

## Incomplete Coverage

- **Context Deduplication:** Tier 2 scan incomplete: a content cluster exceeds the LLM prompt character limit.

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in application-composition.md (`skills/cudaq-algorithms/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/cudaq-algorithms/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cudaq-algorithms/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without verification (`BENCHMARK.md:145`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without verification (`BENCHMARK.md:146`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
