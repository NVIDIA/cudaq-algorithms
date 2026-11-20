## Description: <br>
Use CUDA-Q Algorithms contracts: state_prep injection, QSP/QSVT, chemistry, dynamics and primitives. Excludes setup and standalone kernels. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers, researchers, and engineers use this skill to design, implement, debug, review, validate, or compose CUDA-Q Algorithms (cudaq_algorithms) APIs and fault-tolerant primitives, including state preparation, block encoding, qubitization, QSP/QSVT, Suzuki-Trotter evolution, fermion transforms, chemistry, double factorization, and circuit primitives such as QROM and reversible arithmetic. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Catalog](references/catalog.md) <br>
- [Quick contract](references/quick-contract.md) <br>
- [Workflow](references/workflow.md) <br>
- [Validation](references/validation.md) <br>
- [Conventions](references/conventions.md) <br>
- [Application composition](references/application-composition.md) <br>
- [Source provenance](references/source-provenance.md) <br>
- [Block encoding](references/block-encoding/block-encoding.md) <br>
- [Chemistry bridges](references/chemistry/chemistry-bridges.md) <br>
- [Double factorization](references/double-factorization/double-factorization.md) <br>
- [Fermion transforms](references/fermion-transforms/fermion-transforms.md) <br>
- [Primitives](references/primitives/primitives.md) <br>
- [QSVT](references/qsvt/qsvt.md) <br>
- [Qubitization](references/qubitization/qubitization.md) <br>
- [Simulation](references/simulation/simulation-evolve.md) <br>
- [State preparation](references/state-preparation/state-preparation.md) <br>
- [State preparation injection contract](references/state-preparation/injection-contract.md) <br>
- [Trotter](references/trotter/trotter.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline Python and bash code blocks; runnable Python scripts with observed CHECK output] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
19 evaluation tasks (16 positive, 3 negative), 1 attempt per task, run in the `k8s-sandbox` environment with each task attempt in its own isolated sandbox pod; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.3% → 93.3% (-3.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.9% → 85.7% (+1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 10 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 19 task(s) |

## Skill Version(s): <br>
0.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
