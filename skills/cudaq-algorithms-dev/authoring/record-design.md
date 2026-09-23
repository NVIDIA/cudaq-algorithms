# Record Design

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

Lifecycle is historical coverage metadata; the other scientific classifications
remain part of the live contract. Follow [coverage policy](../coverage/policy.md)
for that placement distinction.

Host transforms such as chemistry loaders and factorizations are computational
routines when they solve an independently useful problem. A simulation-only
helper is not a hardware primitive merely because it consumes one.

## Composite protocols

A reusable driver may itself be a primitive. Its record must state required
lower-level capabilities, a source-grounded reference composition, applicability
conditions, alternatives, propagated conventions/errors/resources, and what an
alternative component must preserve. Never silently replace the reference
composition with a target-specific heuristic.

## Resource claims

Every executable primitive either gives a resource contract or explicitly says
that none exists. Every quantity identifies metric/unit, abstraction level,
architecture/execution assumptions, exact/bounded/estimated/measured status,
controlling parameters, confidence/limitations, and composition rule if known.

Never compare logical operations, decomposition proxies, transpiled gates,
runtime, memory, or measured hardware cost as if they were one metric. Never
turn a benchmark or source comment into a fresh measurement.

## HF/UCC record application

The [HF/UCC record](../../cudaq-algorithms/references/state-preparation/state-preparation-hf-ucc.md)
is a **concrete primitive record**: it instantiates each applicable
canonical Primitive-Record field from `templates/primitive-record-template.md`
exactly once, for one contract — a Hartree-Fock reference occupation optionally
followed by a UCC product at amplitudes the caller already knows. The shared
seam it plugs into (kernel representation, unitary capability, consumer table,
common boundaries) is [injection contract](../../cudaq-algorithms/references/state-preparation/injection-contract.md);
cross-cutting layout, ownership, and validation conventions are
[the convention selector](../../cudaq-algorithms/references/conventions.md). Shared scientific
detail is linked rather than repeated; lifecycle and run bookkeeping follows
the template's separate coverage destination.

## Provisional HF/UCC classification discussion

The HF/UCC method is a fixed-parameter ansatz product. `ansatz` was a provisional
extension beyond the `direct | variational | heuristic` vocabulary. Exactness,
uncertainty, and method are recorded values, never routing identities.
