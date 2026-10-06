# CUDA-Q Algorithms Skill Card

## Description:

Guides source-backed design, implementation, debugging, review, validation, and
composition of CUDA-Q Algorithms primitives. It covers state preparation, block
encoding, qubitization, QSP/QSVT, Suzuki-Trotter evolution, chemistry, fermion
transforms, double factorization, and statevector analysis.

The skill is distributed under Apache-2.0. Evaluation through nvskills-ci is
pending; this card does not claim evaluation success or deployment readiness.

## Owner

CUDA-Q Algorithms Team, NVIDIA

### License/Terms of Use:

Apache-2.0

## Use Case:

Answer API and capability questions, build or repair scientific workflows, and
validate results against explicit contracts and independent numerical references.
Requests may describe scientific inputs and desired outputs without naming an API.
CUDA-Q installation, backend setup, basic standalone kernels, and unrelated
quantum-computing questions are outside the skill's scope.

### Deployment Geography for Use:

Global. Any selected execution service remains subject to its own terms.

## Requirements / Dependencies:

- A local CUDA-Q Algorithms checkout matching the target version is required to
  verify source and test behavior. The skill does not bundle the package source.
- Python 3 is required for the local routing script. Running generated examples
  also requires CUDA-Q Algorithms and the selected workflow's dependencies.
- Requires API Key or External Credential: No for local guidance and routing.
  Remote execution may require provider credentials and authorization.
- Credential Type(s): None required by the skill itself. Do not place credentials
  in prompts, logs, or generated artifacts.

## Known Risks and Mitigations:

- Incorrect API or version assumptions can produce invalid code. Consult the
  focused record and the matching checkout's public source and tests; distinguish
  unsupported behavior from behavior that has not been verified.
- Scientific conventions, register ordering, phases, or normalization can change
  a result. State the selected conventions and use an independent oracle with an
  explicit precision and tolerance before claiming agreement.
- Unexecuted examples or formula-level cost proxies can be mistaken for measured
  results. Label execution status and evidence, and distinguish circuit proxies
  from hardware measurements.
- Generated code can modify files or consume remote resources. Preserve user
  files and obtain authorization covering dependencies, credentials, and remote
  or paid execution before performing those actions.

## Reference(s):

- [Skill instructions and scope](SKILL.md)
- [API catalog](references/catalog.md)
- [Scientific workflow](references/workflow.md)
- [Validation guide](references/validation.md)
- [Source lookup and provenance](references/source-provenance.md)

## Skill Output:

- Output Type(s): Scientific analysis, Python code, validation instructions, and
  evidence summaries.
- Output Format: Markdown and, when requested, runnable Python files.
- Output Parameters: Determined by the selected API contract and the user's
  scientific inputs, accuracy target, and execution constraints.
- Other Properties Related to Output: Claims identify source evidence, observed
  execution, material limitations, and unresolved inputs.

## Evaluation Agents Used:

Pending nvskills-ci. No evaluation agent or model result is claimed for this
revision.

## Evaluation Tasks:

The [authored dataset](evals/EVAL.md) contains 62 cases: 42 regression and
20 scientific cases, including three negative triggering controls.
Execution through nvskills-ci is pending; actual attempt counts will be recorded
with that evaluation evidence.

## Evaluation Metrics Used:

Pending nvskills-ci. Metrics and their definitions will be reported from the
actual CI evaluation.

## Evaluation Results:

Pending nvskills-ci. No pass rate, correctness score, effectiveness improvement,
or efficiency result is asserted. Local documentation and routing checks do not
establish agent evaluation performance.

## Skill Version(s):

0.2.0, as recorded in [SKILL.md](SKILL.md) metadata.

## Ethical Considerations:

Users remain responsible for reviewing generated scientific claims and code in
their application context. Report limitations and uncertainty, preserve the
provenance of inputs and results, and do not present predictions or unexecuted
examples as observed experiments.
