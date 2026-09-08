# [Primitive family]

Status: draft

This is the canonical contract schema. Do not create a second one. The
**Primitive Record** below is the default shape. Two optional schemas follow it
— **Representation Record** and **Capability Record** — so the three-record
model in `references/architecture.md` is expressible inside one family file
without a parallel template.

Omit a heading only when it does not apply, and say so with one line beginning
`Deferred:` plus the follow-up trigger. A visibly deferred slot is honest; a
silently dropped one looks like the contract does not exist.

**One family file, several providers.** When a family file covers more than one
provider, instantiate each Primitive-Record heading **once at family level** and
put the per-provider material in visibly separate subsections beneath it. Do not
repeat the heading set per provider, and do not rename or drop a heading. The
optional schemas stay separate sections of their own, so the three record types
remain visibly distinct inside the file.

**Relationship to the catalog.** `references/catalog.md` carries a subset of
these field names as a one-line summary projection per family. Shared field
names keep the spelling and meaning defined here; the catalog never redefines a
field and is never the authority for a contract.

## Identity and provenance

- Owner:
- Public symbols and import paths:
- Source paths:
- Authoritative tests:
- Authoritative documentation:
- Package/CUDA-Q versions verified:
- Commit/date last verified:
- Lifecycle: draft | verified | deprecated | removed
  (the only maturity vocabulary; do not add a separate maturity field)
- Replacement and migration notes:

## Classification

- Operation + mathematical object (primary identity):
- Kind: quantum operation | classical transformation | measurement/readout |
  simulation-only analysis | resource estimator
- Routine role: driver | computational | auxiliary
  (problem completeness, not execution location)
- Abstraction level: leaf operation | composite protocol
- Parameterization: none | construction-time | runtime
- Execution layers:
- Input representations:
- Output representations:
- Domain: domain-independent | quantum-chemistry | other tag
- Required dependencies:
- Optional dependencies:

Provisional metadata — record the value, do not route on it:

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

Repeat this block for every provided or required capability:

- Stable ID: `cudaq-algorithms.<capability-name>.v<major>`
- Direction: provides | requires
- Owning family record:
- Boundary representation:
- Semantic invariants:
- Shape/register geometry:
- Normalization, sign, phase, and ordering:
- Convention requirements:
- Host/device/simulation boundary:
- Unsupported conditions:

Compatibility requires matching IDs and compatible major versions. Providers
must satisfy every consumer invariant and constraint. A capability ID is a
documentation contract, not a Python protocol or public symbol.

## Composite protocol

Required when Abstraction level is `composite protocol`; omit otherwise.

- Required lower-level capabilities:
- Canonical reference composition:
- Default recipe and its applicability conditions:
- Materially different alternatives:
- Propagated conventions:
- Propagated errors:
- Propagated resources and how they compose:
- Component substitution: how to supply an alternative implementation, and
  what the substitution must preserve:

The default composition prioritizes transparent scientific correctness and
broad validity. It is never silently replaced by a resource-adaptive choice.

## Accuracy and limitations

- Error behavior or bounds:
- Precision sensitivity:
- Unsupported inputs:
- Known implementation limitations:
- Unsupported versus unverified: state which behaviors are documented as
  unsupported and which are merely undocumented in the source.

## Resources

Every executable primitive documents a resource contract. For each quantity,
state:

- metric and unit;
- abstraction level;
- architecture/execution assumptions;
- exact, bounded, estimated, or measured status;
- controlling parameters;
- confidence/limitations;
- composition rule, if known.

A reusable estimator may additionally have its own primitive record.

## Validation

- Independent oracle:
- Invariants:
- Representative cases:
- Predeclared tolerances:
- Expected failure/adversarial case:
- Reference results:
- Evidence status per claim: derived | measured | assumed | unverified.
  A committed repository test assertion that was not executed in this session
  is cited evidence, not a fresh measurement.

## Evaluation coverage

- Positive selection/application:
- Convention or misconception:
- Capability composition:
- Invalid/unsupported boundary:
- Negative activation:

## External alignment

- Literature conventions:
- External package translations:
- Known semantic differences:

---

# Optional schema: Representation Record

Use only when multiple primitives exchange or interpret the same object. Do not
create one per public Python type.

- Object name and canonical symbol:
- Public type or structural form, and source path:
- Mathematical meaning:
- Shape, layout, ordering, dtype, and units:
- Normalization, sign, and phase convention:
- Required mathematical properties (the applicability preconditions):
- Producers (≥2 required to justify this record):
- Consumers:
- Invariants preserved across the boundary:
- Observable symptom of a misinterpretation:
- Unsupported or ambiguous forms:
- Source paths, tests, docs, and last verification:

---

# Optional schema: Capability Record

Use only when multiple independent producers or consumers demonstrate a
reusable boundary. A one-off interface stays inside its primitive record.

- Stable ID: `cudaq-algorithms.<capability-name>.v<major>`
- Status: candidate | provisional | stable taxonomy contract
  (this is the capability-contract axis only; it is not the record `Lifecycle`
  vocabulary, so `draft` never appears here. Use `candidate` while a single
  producer or consumer motivates the boundary, `provisional` once the source
  demonstrates several independent producers or consumers, and `stable
  taxonomy contract` only after the contract has been verified)
- Contract type: documentation/taxonomy only, or backed by a source-level
  protocol (name it)
- Boundary representation and exact signature:
- Semantic invariants:
- Register or shape geometry and ownership:
- Convention requirements:
- Host/device/simulation boundary:
- Providers, with source paths:
- Consumers, with source paths:
- Unsupported and unverified conditions:
- Promotion criteria to a public protocol or compiler IR operation, and the
  explicit decision still required before promotion:
