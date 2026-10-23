---
name: cudaq-algorithms
description: "Use CUDA-Q Algorithms contracts: state_prep injection, QSP/QSVT, chemistry, dynamics and primitives. Excludes setup and standalone kernels."
license: Apache-2.0
version: 0.2.0
permissions: [file_read, file_write, shell]
tags: [cuda-quantum, cudaq-algorithms, quantum-computing, fault-tolerant, simulation, chemistry, nvidia]
metadata:
  author: CUDA-Q Algorithms Team <cuda-quantum@nvidia.com>
  version: 0.2.0
  tags:
    - cuda-quantum
    - cudaq-algorithms
    - quantum-computing
    - fault-tolerant
    - simulation
    - chemistry
    - nvidia
---

# CUDA-Q Algorithms

## When to use

Use when designing, implementing, debugging, reviewing, validating, or composing
cudaq_algorithms / CUDA-Q Algorithms APIs and fault-tolerant primitives, including
scientific requests expressed as molecular geometry/integrals to energies, lattice
Hamiltonians to dynamics, or prepared states to spectral and conditional observables.
Covers state preparation, Pauli/block encoding, qubitization, QSP/QSVT,
Suzuki-Trotter evolution, fermion transforms, chemistry, double factorization,
fault-tolerant circuit primitives (QROM, unary iteration, reversible arithmetic,
alias sampling), and statevector analysis. Not for CUDA-Q installation, backend setup, basic standalone
kernels such as Bell states, or unrelated quantum-computing questions.
For an excluded request, use the ordinary CUDA-Q workflow without loading this
skill's scientific references. A basic state used as an input to a packaged
algorithm remains in scope through that algorithm's contract.

## Before implementation

Keep a short request-to-evidence checklist: requested results, constructed library
objects or preparations, input conventions/physical sector, and output interface.
Before finishing, associate each item with its implementation or answer location
and observed check. Compare the actual constructor settings with the declared
input conventions; a default argument must not silently replace them. Numerical
agreement does not replace an explicitly requested operation or preparation, even
when the remaining calculation is classical.

Route the in-scope task once. From the directory containing this `SKILL.md`,
run `python3 scripts/route.py "<original request>"`, or pass the request on
standard input. Preserve preparation requirements, convention tags and output
fields. Read the whole output; do not pipe it through `head`. It carries the
exact calls, the executable little-endian dense reference for numerical tasks,
the selected contracts and checks, and a resolved fetch command for any section
it had to omit. Open more text only for an unresolved contract. `--paths` gives
navigation only; `--full` prints the complete selected guides.

Deliverable rules:

- **Library participation.** When the request names CUDA-Q Algorithms or a
  library operation, every reported number must come from a library construction
  whose returned data enters the result. A dense or NumPy calculation is a check,
  never the deliverable, even when the remaining computation is classical; an
  unused object or an occupation list implements nothing. See
  [task completion](references/validation.md#requested-task-completion).
- **Ordering.** `q[0]` is the least-significant bit in statevectors and
  `SpinOperator.to_matrix()`. Before trusting any dense Hamiltonian, operator or
  state reference, run the [dense-operator self-test](references/conventions/ordering.md#dense-operator-self-test);
  norms, Hermiticity and agreement between two objects built with the same
  reversed ordering do not detect a reversal. For complex input to
  real-Hamiltonian QSVT recovery follow the
  [reconstruction recipe](references/qsvt/qsvt-recovery.md#complex-input-composition-for-a-real-hamiltonian).
- **Claims.** Derive thresholds, selections and changes from the validated rows.
  State the threshold and its source for every sufficiency claim; without a
  supplied criterion, quantify the change and make sufficiency conditional.
  Print one `CHECK` line per independent comparison and claim only checks that
  ran; distinguish percentage points from percent; include the exact
  reproduction command for executed work. See
  [completion](references/workflow.md#completion-checkpoint).
- **Machine-readable results.** When an output interface is requested, run the
  actual entrypoint once and validate the complete payload at that interface per
  [output validation](references/validation.md#requested-output-artifacts); a
  separate complete file does not validate an incomplete stdout result.

Work discipline:

- Run the artifact once per change and keep its captured output as evidence;
  never rerun unchanged code to read its output again.
- When implementation and reference disagree, make at most two repair rounds,
  then report the discrepancy with the evidence instead of iterating.
- Use the named checkout paths; never search the whole filesystem or parent
  directories. Define kernels in a `.py` file, not a heredoc, and batch related
  shell steps into one call.

## Prerequisites

A local CUDA-Q Algorithms checkout matching the task's target version is required
to verify API behavior against source and tests; establish its root before
following checkout-relative paths such as `python/cudaq_algorithms/`,
`tests/python/`, and `docs/sphinx/examples/python/`. This skill does not include
that source tree; without it, identify the missing evidence and label
version-dependent claims unverified. The routing script needs only Python 3 and
never imports the package; executing examples needs CUDA-Q Algorithms and the
selected workflow's dependencies.

## Instructions

Start from the researcher's inputs and desired output; they need not name APIs,
repository files, or tests. Use the existing API contracts together with the
selected guide's **Workflow** and **Verification** sections. Lead the answer with
the scientific result or actionable conclusion, then give its evidence and material
limitations. Cite the records used, check consequential premises, and identify
unresolved inputs; do not turn every scientific result into a contract inventory.
For a contract audit, organize these as Record consulted, Premise check, Boundaries,
and Unknowns. A record sentence is source evidence, never an observed run.

### Available scripts

| Script | Invocation and inputs | Output |
| --- | --- | --- |
| [route.py](scripts/route.py) | From the skill directory: `python3 scripts/route.py [--paths \| --full] "<request>"`; reads standard input when request arguments are omitted or empty. `--help` shows usage. | Exact calls, the executable dense reference for numerical tasks, bounded relevant sections from the selected guides and records, source pointers, shared checkpoints and a resolved fetch command for each omitted section. `--paths` prints navigation only; `--full` prints complete selected guides. Standalone Bell/basic-kernel requests return a scope note; unmatched requests return the catalog entry point. |

If your agent runtime supports `run_script`, use it to run `scripts/route.py`;
otherwise run it with `python3` as shown. Reuse the emitted sections and follow
their focused source pointers; routing is a navigation aid, not a scientific
decision.

Use supplied data and derive the intermediate inputs the model determines, such as
integrals from geometry, basis, charge and spin. Ask only for missing choices that
change the scientific result or scope, resolve the version/source boundary of a
requested inventory from context first, never invent data or APIs, and proceed
when the available inputs resolve the task.

Use CUDA-Q Algorithms as a BLAS/LAPACK-like set of small scientific building
blocks: start with the requested operation and mathematical object, choose the
smallest matching contract, and compose through explicit representations,
signatures, capabilities and conventions. Links resolve relative to their
document; source and test paths are checkout-relative.

- Known API or operation: use its focused record from the router (the
  [catalog](references/catalog.md) lists every path), then the named current
  source and tests. The record carries the contract, boundaries and provenance,
  the source carries behavior; an answer about a contract, capability boundary or
  dependency that cites no record is incomplete, and the checkout controls API
  behavior.
- Unknown family: select the operation and mathematical object in the
  [catalog](references/catalog.md), then open only the matching records.
- Multi-step advice, implementation or repair: apply the emitted shared
  [workflow](references/workflow.md) checkpoints; advice alone does not
  authorize file changes.
- Molecular energies or lattice dynamics without a named algorithm: start from
  the [application composition](references/application-composition.md#workflow)
  sections selected by the router and choose the method from the requested
  output, accuracy and execution constraints; a default route does not mandate a
  circuit.
- Contract or boundary question: check the applicable premise rows against the
  current source and cite the focused record; keyword overlap alone does not
  establish that a row applies, and an unsupported premise is never confirmed.

For an unresolved boundary, consult the relevant [conventions](references/conventions.md),
[composition](references/application-composition.md) or [validation](references/validation.md)
section; [source lookup](references/source-provenance.md) gives shared paths.
Match signatures, representations, capabilities, register geometry,
normalization, ordering, signs, phases and execution layers. Fix an independent
oracle, precision and tolerance before judging agreement; report the evidence
actually gathered and label unexecuted code as such. Unsupported, absent and
unverified mean different things; roadmap-only APIs are unavailable.

## Limitations

Installation, backend setup, and optimizer-driven NISQ workflows are outside this
skill; parameterized UCC construction remains an in-scope state-preparation primitive.
Dependencies, credentials, remote jobs, paid/shared QPUs, publishing, and external
contact require authorization covering those actions.

Before writing CUDA-Q kernels, read the
[kernel boundaries](references/quick-contract.md#kernel-boundaries).
Define kernels in a `.py` file run with `python3 file.py`; CUDA-Q cannot compile
kernels defined in a heredoc or `-c` string. Mint kernels and compute NumPy/host
data outside `@cudaq.kernel`; inside it allocate registers, apply gates, and call
captured kernels. Pass statevector input with `state_from(ket)` as a cudaq.State
argument; there is no `cudaq.prepare_state`. CUDA-Q statevectors are little-endian
(`q[0]` is the least-significant bit); the good subspace is the first
`2**num_system` amplitudes when the system register is allocated first.

For implementation requests, deliver a runnable script and the observed output when
execution is available; keep the requested artifact in the workspace. For predictions
or advice, supply the requested scientific result and sufficient evidence without
claiming circuit execution. Preserve user files. If inputs remain unavailable after
checking the named locations, identify what is missing and which work depends on it.

## Examples

Starting routes, not substitutes for the source and verification checks:

| Request | Starting guide |
| --- | --- |
| Check a PauliLCU block encoding against a dense matrix | [Block encoding](references/block-encoding/block-encoding.md) |
| Compute molecular energies from geometry and a basis | [Chemistry](references/chemistry/chemistry-bridges.md) and [composition](references/application-composition.md) |
| Compare Suzuki-Trotter evolution with exact dynamics | [Trotter](references/trotter/trotter.md) |
