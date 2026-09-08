# Skill architecture

## Design goal

Represent CUDA-Q Algorithms as a long-lived library of composable scientific
primitives without organizing knowledge around today's applications or
hard-coding tomorrow's taxonomy.

This document defines structural invariants and the approved taxonomy. The
taxonomy is provisional where marked; it is tested by populated records rather
than settled in advance.

## Organization

This file is maintainer and reviewer policy. Answering a scientific question
does not require it; adding, changing, or reviewing a record does.

`SKILL.md` is the compact router and reasoning contract.

`catalog.md` indexes only populated family references. It does not duplicate
their contracts.

Cross-cutting rules live in `conventions.md` and `validation.md`.

Each mature family receives one flat reference file, for example
`state-preparation.md`, `block-encodings.md`, or `qrom.md`. Split a family only
when independent ownership or context size justifies it.

`assets/primitive-record-template.md` is the canonical schema. Do not create a
second contract schema in prose.

## Catalog identity

The primary identity of every record is a two-key pair:

```text
scientific operation + mathematical object
```

Operations include prepare, load, encode, transform, evolve, measure, estimate,
synthesize, and preprocess. Objects include states, fermionic operators, Pauli
operators, tensors, block encodings, polynomials, oracles, tables, phase
sequences, and resource descriptions.

Route by that pair first. Every other dimension below is orthogonal metadata,
not a directory hierarchy and not a routing key. Do not organize the catalog by
Python artifact kind, repository module, or workflow stage.

## Record granularity

One record corresponds to one independently selectable scientific contract.

A separate record is warranted when an operation has its own mathematical
semantics, input/output behavior, approximation or error behavior, resource
behavior, or composition boundary. Public symbols, convenience wrappers, and
implementation helpers do not automatically receive independent records, and
several contracts may live in one module or class.

## Primitive kind

"Primitive" is a broad umbrella: any independently reusable scientific building
block in the library. Every record must declare its **kind** explicitly, because
these artifacts do not share one runtime interface:

- quantum operation or kernel factory;
- classical transformation or preprocessing;
- measurement, observable, or readout protocol;
- simulation-only analysis;
- resource estimator.

The skill's scope is fault-tolerant algorithm primitives. Optimizer-driven NISQ
application loops (VQE, ADAPT-VQE, QAOA, GQE) are out of scope. A primitive may
still accept parameters that such an application would choose; parameterization
is metadata, not a separate taxonomy.

## Routine role is not execution layer

The LAPACK routine layers are adopted by their own criterion — *completeness of
the problem solved* — and not by where the code runs:

- **driver:** solves a complete user problem by sequencing lower-level
  routines; composite protocols and readout protocols belong here;
- **computational:** performs one distinct, well-defined, independently usable
  task; a host classical transform qualifies exactly as much as a device kernel
  does;
- **auxiliary:** block-algorithm subtasks and low-level utilities, documented
  only where needed to interpret a driver or computational contract and
  normally receiving no record of their own.

Do not infer routine role from execution layer. Fermion-to-qubit transforms,
classical factorizations, chemistry input bridges, and kernel factories are
computational routines that happen to run on the host; they are not auxiliaries.
Execution layer stays separate metadata.

This is a fidelity clarification of the LAPACK analogy in the taxonomy design
record, not a new taxonomy branch: that record's sketch — device kernels as
computational routines, composite protocols as drivers, host transforms and
estimators as auxiliaries — read the two axes as one. Role is determined by
problem completeness; host, device, measurement, and simulation remain the
orthogonal execution layer. A host transform that solves one complete,
independently usable task is computational, and only block-algorithm subtasks
and low-level utilities are auxiliary.

CUDA-Q Algorithms follows BLAS/LAPACK principles, not their literal taxonomy.
BLAS Levels 1/2/3 are not adopted: they stratify by the ratio of data movement
to arithmetic, and quantum algorithm primitives have no comparable
stratification available yet. The transferable principles are explicit
contracts before implementations, explicit representations and conventions,
small computational routines plus reusable drivers, visible algorithm and
hardware policy, substitution only through compatible contracts, resource
requirements exposed at the right abstraction, and applications as consumers
and evidence rather than as the taxonomy.

## Three record types

Three complementary record types are adopted provisionally:

1. **Representation records** — what does an exchanged scientific object mean?
   Created **only** when multiple primitives exchange or interpret the same
   object. Not created for every public Python type.
2. **Primitive records** — what concrete scientific operation is performed?
3. **Capability records** — what stable semantic behavior permits composition?
   Extracted **only** when multiple independent producers or consumers
   demonstrate a reusable boundary. One-off interfaces stay inside their
   primitive record. This gate exists to avoid freezing accidental interfaces
   and generating pairwise compatibility records.

These are record types, not three top-level product namespaces. A flat family
reference may hold a representation record, a capability record, and several
primitive records when that is the clearest unit of ownership; the three must
remain visibly separate inside the file.

## Orthogonal metadata dimensions

Classify each record independently along these dimensions. Do not force one
directory hierarchy to encode them.

- **Kind:** as defined above.
- **Routine role:** driver, computational, or auxiliary.
- **Execution layer:** host preprocessing, kernel factory, device kernel,
  observable/measurement, simulation-only analysis, or mixed.
- **Abstraction level:** leaf operation or composite protocol.
- **Parameterization:** none, construction-time, or runtime.
- **Representation:** the objects accepted and produced.
- **Capabilities:** provided and required, as stable contract identifiers
  rather than pairwise compatibility.
- **Exactness:** exact or approximate, with explicit approximation controls.
- **Uncertainty:** deterministic or stochastic, including the source and
  interpretation of randomness.
- **Method:** direct, variational, or heuristic when that affects semantics.
- **Domain:** domain-independent or specialized, such as quantum chemistry.
- **Dependencies:** required and optional packages, hardware, data, or network.
- **Error contract:** validation, rejection, and failure behavior, including
  where a failure is detected.
- **Resource contract:** the quantities under Resource claims below.
- **Lifecycle:** the single maturity vocabulary defined under Lifecycle below.

Exactness, uncertainty, and method are separate axes whose vocabularies remain
provisional until representative records test them. They are provisional
metadata, never routing keys.

Domain specialization is a tag, never a parallel taxonomy. A chemistry-specific
primitive uses the same operation/object identity with `domain:
quantum-chemistry`, and its outputs may still provide domain-independent
capabilities.

## Capability-based composition

Composition is valid when one primitive's provided capability satisfies
another's required capability under the same representation and conventions.

Use stable identifiers of the form
`cudaq-algorithms.<capability-name>.v<major>`. The major version changes only
when the semantic contract is incompatible. A capability declaration must
state:

- identifier and version;
- owning family record;
- provided or required direction;
- boundary representation;
- semantic invariants;
- register or shape constraints;
- normalization, phase, ordering, and convention requirements;
- host/device/simulation boundary;
- unsupported conditions.

Matching requires the same capability identifier and compatible major version.
The provider must satisfy every invariant and constraint required by the
consumer. Do not infer compatibility from similar names, and do not enumerate
every compatible pair.

Capabilities are **documentation and taxonomy contracts first**. They become
public Python protocols or compiler IR operations only after real
implementations demonstrate that the boundary is stable. The skill must not
drive speculative public API design. A capability identifier written in a
record is not a Python `Protocol`, an ABC, or a public symbol.

## Composite primitives

An independently reusable composite protocol may itself be a primitive even
when it is built from lower-level primitives, in the same way a LAPACK driver
remains a legitimate routine. Phase estimation is the canonical example.

A composite record must expose:

- required lower-level capabilities;
- one canonical reference composition;
- a default recipe with explicit applicability conditions;
- materially different alternatives;
- propagated conventions, errors, and resources;
- a way to supply alternative component implementations.

The default composition prioritizes transparent scientific correctness and
broad validity. Target- or resource-adaptive selection may be added once cost
models are trustworthy, but it must never silently redefine the reference
composition.

Applications instantiate and validate primitives; they never define primitive
identity. Roadmap cards are not records — one card may span representations,
primitives, capability candidates, composite protocols, lowering techniques,
domain collections, and application evidence.

## Layered semantics: the QROM pattern

Some names conflate three layers that must remain distinct. "QROM" is the
worked example, and the same split applies elsewhere (for instance, "implement
`e^{-iHt}`" versus Trotter, QSVT, or Taylor-series LCU providers):

1. **Semantic capability:** coherent indexed data access, with distinct
   contracts for XOR lookup, phase lookup, alias sampling, and amplitude or
   state preparation. These are not one contract.
2. **Implementation family:** unary iteration, select-swap, and other concrete
   constructions, each with explicit qubit, depth, T/Toffoli, measurement,
   routing, and feed-forward tradeoffs.
3. **Lowering policy:** a future compiler or synthesis layer selects an
   implementation from target and fault-tolerant resource constraints.

No canonical concrete QROM primitive or public API exists. Coherent indexed
lookup is a candidate taxonomy capability only. Concrete constructions may later
become providers, and compiler lowering may replace them without changing
algorithm-level semantics.

Measurement, readout, and classical post-processing must stay visible as their
own layers. They must not disappear because a roadmap card is named after the
quantum operation alone.

## Source ownership and drift

The codebase remains authoritative for API behavior. Every scientific record
must identify:

- public symbols and source paths;
- authoritative tests and documentation;
- package and CUDA-Q version range verified;
- record owner;
- lifecycle state;
- date or commit last verified;
- known divergence from external literature or packages.

Do not copy volatile API details merely for convenience. When duplication is
needed for agent reliability, preserve provenance so drift can be checked.
Automation may be added later after repeated maintenance pain is observed.

## Lifecycle

`draft | verified | deprecated | removed` is the only maturity vocabulary. A
record starts as `draft`, becomes `verified` only after its contract and
evaluation cases pass, and may later become `deprecated` or `removed`.

A capability record additionally states a capability `Status` of
`candidate | provisional | stable taxonomy contract`. That is a different axis
— how far the *boundary* has been demonstrated, not how far the record has been
verified — and it never borrows a lifecycle value. A `draft` record may declare
a `provisional` capability when the source shows several independent producers
or consumers of the boundary.

Deprecation records must name the replacement, first deprecated version, and
behavioral differences. Removed records remain discoverable only when needed
to interpret old code.

When a contract changes incompatibly, prefer creating a newly named primitive
and deprecating the old one over silently redefining an existing record.

## Resource claims

Every executable primitive documents its own resource contract. A reusable
estimator may additionally be an independent primitive in its own right, with
its own record.

Every resource quantity must identify:

- metric and unit;
- abstraction level, such as logical operation, decomposition proxy,
  transpiled gate, wall time, or memory;
- architecture and execution assumptions;
- exact, bounded, estimated, or measured status;
- confidence or known limitations;
- how the quantity composes, if known.

Never compare quantities from different abstraction levels as though they were
the same metric. Target-specific optimization is never hidden behind an
apparently universal default.

## Growth rule

Add structure in response to real scientific content:

1. populate or add one family record;
2. update the catalog entry;
3. add convention records only for new cross-cutting semantics;
4. add evaluations that fail without the new knowledge;
5. split files or automate indexes only when scale creates demonstrated need.

Document new areas in dependency order — shared representations and
conventions, then proven shared capabilities, then concrete primitives, then
composite protocols, then enabling primitives, then domain specialization, then
application evidence — independent of implementation priority.

This preserves a stable foundation without prematurely designing the complete
future library.
