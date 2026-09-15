# Skill architecture

## Design goal

Represent CUDA-Q Algorithms as a durable library of independently selectable
scientific contracts. Applications consume and validate those contracts; they
do not define the taxonomy.

This file is maintainer and reviewer policy. Scientific tasks normally begin at
[the catalog](catalog.md), not here.

## Organization

- `../SKILL.md` is the compact router and operating contract.
- `catalog.md` has one lightweight entry per independently selectable
  operation/object contract.
- The six shared routers and policies stay at `references/`; one shallow
  family directory groups each domain's front door and focused records.
- A family front door groups related records but contains no primitive record.
- Each primitive record lives in one focused Markdown file and covers every
  applicable canonical schema field once. A short record may combine adjacent
  headings when the field boundaries remain explicit.
- Representation and capability records may live in a family front door when
  several primitive records exchange the same object or behavior.
- `conventions.md`, `validation.md`, and `source-provenance.md` own information
  shared across families.
- `application-composition.md` explains how applications consume records.
- `../assets/primitive-record-template.md` is the only contract schema.

Keep the hierarchy one level deep: `references/<family>/<record>.md`, with no
primitive subdirectories. Every record must be directly linked from
`../SKILL.md`, the root catalog, or its family front door. A front door is a
navigation aid, not a workaround for a multi-contract record.

## Primary identity and granularity

The routing identity is:

```text
scientific operation + mathematical object
```

Operations include prepare, load, encode, transform, evolve, measure, estimate,
synthesize, preprocess, analyze, and reconstruct. Objects include states,
fermionic operators, Pauli operators, integral tensors, block encodings,
polynomials, phase sequences, and resource descriptions.

One primitive record corresponds to one contract a caller can select
independently. Split records when any of these differ materially:

- operation or mathematical object;
- public entry point and input representation;
- return type or emitted kernel signature;
- execution layer or authorization implications;
- validation/rejection behavior or independent oracle;
- approximation/error behavior;
- resource contract;
- required/provided capability or composition boundary.

Several symbols may remain in one record when they are inseparable parts of one
contract. One source module or class may require several records. File size is
evidence of a possible granularity problem, never the routing rule itself.

## Record types

1. **Primitive record:** one concrete, independently selectable operation.
2. **Representation record:** the meaning of an exchanged object; create only
   after multiple producers and consumers interpret the same form.
3. **Capability record:** a reusable semantic composition boundary; create
   only after multiple independent producers or consumers demonstrate it.

These are documentation records, not automatic requests for a Python protocol,
ABC, compiler IR operation, or new public API.

## Orthogonal metadata

Classify, but do not route or organize directories, by:

- **Kind:** quantum operation, classical transformation,
  measurement/readout, simulation-only analysis, resource estimator.
- **Routine role:** driver, computational, auxiliary. Role describes problem
  completeness, not where code executes.
- **Execution layer:** host preprocessing, kernel factory, device kernel,
  observable/measurement, simulation-only host path, or mixed.
- **Abstraction:** leaf operation or composite protocol.
- **Parameterization:** none, construction-time, runtime.
- **Representation and capabilities.**
- **Exactness, uncertainty, and method.**
- **Domain, dependencies, error contract, resource contract, lifecycle.**

Host transforms such as chemistry loaders and factorizations are computational
routines when they solve an independently useful problem. A simulation-only
helper is not a hardware primitive merely because it consumes one.

## Capability composition

Use identifiers of the form
`cudaq-algorithms.<dotted-capability-name>.v<major>`. Dotted capability names
are valid. The major version changes only for an incompatible semantic change.

The currently adopted documentation identifiers are:

- `cudaq-algorithms.state-preparation.unitary.v1`;
- `cudaq-algorithms.block-encoding.zero-flagged.v1`;
- `cudaq-algorithms.chemistry-integrals.v1`.

Their status remains `provisional`; the identifier is resolved even though the
boundary has not been promoted to a stable taxonomy contract.

Every capability record states its ID, status, owner, direction, boundary
representation, exact signature, semantic invariants, geometry, conventions,
execution boundary, providers, consumers, and unsupported conditions.
Composition requires the same ID and compatible major version, plus every
consumer invariant. Similar names and structural member presence are
insufficient.

## Composite protocols

A reusable driver may itself be a primitive. Its record must state required
lower-level capabilities, a source-grounded reference composition, applicability
conditions, alternatives, propagated conventions/errors/resources, and what an
alternative component must preserve. Never silently replace the reference
composition with a target-specific heuristic.

## Source ownership and freshness

Current public code and authoritative tests in the checked-out repository
control API behavior. [Source provenance](source-provenance.md) records one
historical last-review anchor and the common source/test/example locations. The
anchor is an audit trail, not the active contract or a compatibility promise;
each record adds only contract-specific stable symbols and paths.

When a selected record differs from current source or tests:

1. compare the relevant public symbol and tests;
2. treat current public source as authoritative for generated code;
3. report the drift and evidence level;
4. update a maintained record only when that update is in scope;
5. never retain a stale line-number claim merely because the prose is familiar.

Do not copy historical repository hashes or dependency pins into primitive
records. Historical last-review values belong only in the shared provenance
file. A validation or evaluation result instead records the exact revision and
dependencies actually used by that run.

Use line numbers only for a non-obvious invariant that benefits from a precise
anchor. Prefer stable symbol and test names for ordinary provenance.

## Lifecycle and evidence

`draft | verified | deprecated | removed` is the only record lifecycle
vocabulary. A capability separately uses
`candidate | provisional | stable taxonomy contract`.

A record becomes `verified` only after its contract, runnable usage, relevant
scientific tests, and representative evals have actually passed on a recorded,
supported package/CUDA-Q version combination. Store exact source revisions,
dependencies, targets, and commands with that result. Source inspection alone
supports `source-checked` in the task that performs it, not `verified`.

Deprecation records name the replacement, first deprecated version, and
behavioral differences. Incompatible contract changes prefer a new primitive
name and explicit migration over silent redefinition.

## Resource claims

Every executable primitive either gives a resource contract or explicitly says
that none exists. Every quantity identifies metric/unit, abstraction level,
architecture/execution assumptions, exact/bounded/estimated/measured status,
controlling parameters, confidence/limitations, and composition rule if known.

Never compare logical operations, decomposition proxies, transpiled gates,
runtime, memory, or measured hardware cost as if they were one metric. Never
turn a benchmark or source comment into a fresh measurement.

## Growth rule

Add knowledge in this order:

1. shared representation or convention when demonstrated;
2. one independently selectable primitive contract;
3. a capability only when multiple producers or consumers justify it;
4. a composite protocol when its lower-level contracts are populated;
5. catalog routing, runnable usage, validation, and eval coverage in the same
   change.

Do not add records for roadmap concepts or speculative APIs. Split existing
records when real contracts have become independently selectable.
