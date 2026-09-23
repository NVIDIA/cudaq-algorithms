# [Primitive name]

Operation + object: **[operation]** a **[mathematical object]**.

This authoring checklist spans two destinations. Scientific contract fields
belong in the operational record. Lifecycle history, executed-version/run
records and authored evaluation mappings belong in
[coverage bookkeeping](../../coverage/policy.md) and its registry/history,
not in application references. The former `Status: draft.` line is retained
here as historical schema guidance, not text to copy into a new record.

This file documents one independently selectable scientific contract. If two
operations differ in return type, execution layer, validation, approximation,
resources, or composition, create two records and link them from a family front
door.

Use `Not applicable:` when a field cannot apply to this contract. Use
`Deferred:` only when the field is required but evidence is currently missing;
name the evidence or event that will resolve it. Never say to omit a heading and
then put a deferred statement under that omitted heading.

## Identity and provenance

- Owner:
- Public symbols and import paths:
- Contract-specific source paths:
- Authoritative tests:
- Authoritative documentation and runnable examples:
- Source provenance: [source-provenance.md](../../../cudaq-algorithms/references/source-provenance.md)
- Package/CUDA-Q versions executed (coverage run evidence):
- Historical lifecycle (coverage history): draft | verified | deprecated | removed
- Implementation evidence (coverage, with scope): documented | implemented | compiled | executed |
  numerically validated
- Replacement and migration notes:

## Classification

- Operation + mathematical object (primary identity):
- Kind: quantum operation | classical transformation | measurement/readout |
  simulation-only analysis | resource estimator
- Routine role: driver | computational | auxiliary
- Abstraction level: leaf operation | composite protocol
- Parameterization: none | construction-time | runtime
- Execution layers:
- Input representations:
- Output representations:
- Domain: domain-independent | quantum-chemistry | other tag
- Required dependencies:
- Optional dependencies:
- Exactness:
- Uncertainty:
- Method:

## Scientific contract

- Purpose:
- Mathematical definition:
- Why and when to use:
- When not to use:
- Approximation controls:

## Inputs

- Arguments:
- Shapes/ranks:
- Dtypes/domains:
- Units:
- Ordering/layout:
- Normalization:
- Required mathematical properties:
- Validation and rejection behavior:

## Outputs

- Return type or emitted kernel signature:
- Mathematical meaning:
- Shape/register geometry:
- Normalization, sign, and phase:
- Observable or measurement interpretation:
- Error/status information:

## Capabilities and composition

For every provided or required capability:

- Stable ID: `cudaq-algorithms.<dotted-capability-name>.v<major>`
- Capability status: candidate | provisional | stable taxonomy contract
- Direction: provides | requires
- Owning record:
- Boundary representation and exact signature:
- Semantic invariants:
- Shape/register geometry:
- Normalization, sign, phase, and ordering:
- Convention requirements:
- Host/device/simulation boundary:
- Unsupported conditions:

If no capability applies, say `Not applicable:` and explain the direct concrete
composition boundary. A capability ID is a documentation identifier unless the
record names a source-level protocol or public symbol.

## Composite protocol

For `composite protocol` records:

- Required lower-level capabilities:
- Canonical reference composition:
- Default recipe and applicability conditions:
- Materially different alternatives:
- Propagated conventions:
- Propagated errors:
- Propagated resources:
- Component-substitution requirements:

For leaf records, write `Not applicable: leaf operation.`

## Accuracy and limitations

- Error behavior or bounds:
- Precision sensitivity:
- Unsupported inputs:
- Known implementation limitations:
- Unsupported, absent, and unverified behavior:

## Resources

For every quantity, state metric/unit, abstraction level,
architecture/execution assumptions, exact/bounded/estimated/measured status,
controlling parameters, confidence/limitations, and composition rule. If no
estimator exists, say so and document only exact structural facts.

## Validation

- Independent oracle:
- Invariants:
- Representative cases:
- Predeclared tolerances:
- Expected failure/adversarial case:
- Runnable example or usage test:
- Execution record (coverage evidence): command, date, package/CUDA-Q version, target, precision,
  result
- Evidence status per claim: derived | source-checked | compiled | executed |
  numerically validated | measured | assumed | unverified. Add `unexecuted` as
  an execution-state qualifier when no successful run occurred in the current
  task. Reserve `source-checked` for current-task inspection; describe durable
  historical evidence as `derived` and link its review context through
  [coverage history](../../coverage/history.md). Source provenance provides
  current-source lookup, not historical run authority.

## Evaluation coverage

Store this section's mappings and results in the coverage registry, not in the
operational record.

- Positive selection/application:
- Convention or misconception:
- Capability composition:
- Invalid/unsupported boundary:
- Negative activation:
- Eval status: authored | baseline run | with-skill run | compared

## External alignment

- Literature conventions:
- External package translations:
- Known semantic differences:
