---
name: cudaq-algorithms
description: Use when selecting, understanding, composing, or validating reusable fault-tolerant quantum-computing primitives from CUDA-Q Algorithms. Do not use for general CUDA-Q installation, backend setup, or unrelated quantum-computing questions.
license: Apache-2.0
compatibility: Scientific reasoning is environment-independent. Running generated workflows requires Python 3.11+ and repository-compatible CUDA-Q and cudaq-algorithms versions.
allowed-tools: Read Glob Grep
metadata:
  author: CUDA-Q Algorithms Team <cuda-quantum@nvidia.com>
  version: "0.1.0"
  status: early-development
---

# CUDA-Q Algorithms

## Status

Source-grounded draft coverage spans every currently installed scientific
family listed in the catalog. The records describe the repository at commit
`61ac072d`; runtime/version verification and SkillEvaluator uplift remain
unverified. Roadmap-only primitives are explicitly unavailable.

## Purpose

Help agents work with CUDA-Q Algorithms as a BLAS-like library of small,
orthogonal, composable quantum-algorithm primitives.

Start from the requested scientific operation and the mathematical object it
acts on. Select the smallest primitive that satisfies that contract, then check
composition. Applications are compositions of primitives, not the organizing
abstraction of the library.

## Ownership boundary

This skill owns:

- scientific interpretation and primitive selection;
- mathematical and programmatic input/output contracts;
- composition through declared capabilities;
- normalization, indexing, register, phase, and precision conventions;
- approximation, resource, and validation reasoning;
- repository guidance specific to algorithm primitives.

This skill does not own:

- CUDA-Q installation, simulator setup, or QPU-provider onboarding—use
  `cudaq-guide`;
- generic quantum-computing explanations unrelated to this library;
- APIs or workflows not present in CUDA-Q Algorithms;
- NISQ optimizer loops such as VQE, ADAPT-VQE, QAOA, or GQE.

A parameterized construction such as UCC is still in scope as a
state-preparation primitive: `parameters + register -> prepared state`.
Choosing or optimizing those parameters is the excluded workflow.

## Instructions

1. **Define the operation and the object.** Name the scientific operation
   (prepare, load, encode, transform, evolve, measure, estimate, synthesize,
   preprocess) and the mathematical object it acts on. State known inputs,
   requested outputs, acceptable error, and material unknowns.
2. **Route on that pair.** Look up the operation/object pair in
   [the catalog](references/catalog.md), then check the representation,
   capability, and signature constraints of the candidate record. Never choose
   by name similarity, and do not expect a populated capability registry — the
   pair is the routing key, capabilities are the composition check.
3. **Load only populated knowledge.** Read the referenced family record and
   the [conventions](references/conventions.md) needed for the task.
4. **Check composition.** Match provided and required capabilities,
   representations, register geometry, exact kernel signatures, and
   host/device boundaries. Do not maintain or infer pairwise compatibility
   lists.
5. **Validate.** Follow
   [the validation methodology](references/validation.md). Fix the oracle,
   convention translation, precision, and tolerance before judging results.
6. **Report honestly.** Label conclusions as derived, measured, assumed, or
   unverified.

If the catalog has no populated record for a requested primitive, inspect the
repository's current public source, tests, and documentation. Clearly identify
the result as repository-derived rather than skill-grounded; never fill an
unknown contract with a plausible guess.

## Default response contract

For a scientific implementation request, provide:

1. objective and assumptions;
2. selected primitive capabilities and rationale;
3. exact inputs and outputs;
4. normalization and convention map;
5. composition boundaries;
6. runnable code when requested;
7. independent validation and predeclared tolerance;
8. approximation and resource implications;
9. limitations and evidence status.

Scale the response to the task. Do not force a complete application workflow
when the user only needs a primitive contract or comparison.

## Examples

- A request to connect state preparation to a spectral transformation starts
  by matching provided and required capability IDs, representations, register
  geometry, and conventions before code is proposed.
- A request for an anticipated primitive such as QROM is answered from a
  populated record or current repository evidence. If neither exists, report
  the contract as unavailable rather than designing a plausible API.
- A general CUDA-Q installation request routes to `cudaq-guide` without loading
  this skill.

## Durable structure

- [Catalog](references/catalog.md): lightweight index of populated family
  records and capabilities; it is the front door for all ten populated
  scientific families.
- [State preparation](references/state-preparation.md): the unitary
  preparation-kernel injection seam and packaged provider records.
- [Conventions](references/conventions.md): cross-cutting scientific
  translations and invariants.
- [Validation](references/validation.md): evidence hierarchy and evidence
  labels.
- [Primitive record template](assets/primitive-record-template.md): canonical
  schema for each future family record — read it when writing or reviewing a
  record.
- [Architecture](references/architecture.md): **maintainer and reviewer
  policy** — organization, taxonomy dimensions, lifecycle, provenance, and
  maintenance rules. Not required to answer a scientific question; read it
  when adding, changing, or reviewing a record.

Add a further family reference only when source-grounded scientific content is
ready. Keep references directly linked from this file or the catalog; avoid
deep reference chains. Do not create records for roadmap-only QROM, arithmetic,
sparse-oracle, eigensolver, THC, or additional resource-model concepts until
public source establishes a contract.

## Non-negotiable scientific rules

- Separate scientific semantics from execution-target choices.
- State signs, phases, normalizations, ordering, and units explicitly.
- Distinguish host helpers, kernel factories, device kernels, observables, and
  simulation-only utilities.
- Do not call two routines interchangeable without comparing their contracts.
- Do not report numerical agreement without naming the oracle and tolerance.
- Do not present logical proxies as transpiled gates, runtime, or memory.
- Do not claim performance or accuracy that was not measured.
- Treat unsupported or undocumented behavior as unknown.

## Incremental development rule

Each scientific refinement should add one coherent family record or
cross-cutting convention. It must include source provenance, a complete
contract, at least one independent validation method, and evaluation coverage
for both a positive use and a boundary or misconception.
