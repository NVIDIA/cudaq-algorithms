---
name: cudaq-algorithms
description: Use for CUDA-Q Algorithms primitives and molecular/lattice workflows. Excludes installation, backend setup, and standalone kernels (e.g. Bell).
license: Apache-2.0
version: "0.2.0"
permissions: [file_read, file_write, shell]
tags: [cuda-quantum, cudaq-algorithms, quantum-computing, fault-tolerant, simulation, chemistry, nvidia]
metadata:
  author: CUDA-Q Algorithms Team <cuda-quantum@nvidia.com>
  version: "0.2.0"
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

Route the in-scope task with
`python3 <skill-dir>/scripts/route.py "<request>"`, retaining named operations,
the desired scientific result, and requested output format. Open the selected
family's Workflow and Verification and its relevant focused records; the router
also points to shared workflow and validation checkpoints. Read only the sections
needed for this task, then follow their current source/test pointers.

Use the relevant [Quick contract](references/quick-contract.md) entry when an
exact call or runnable pattern is needed. The [protocol-only encoding](references/quick-contract.md#consume-a-protocol-only-encoding),
[walk verification](references/quick-contract.md#qubitization-walk-action-and-prepared-state-checks),
and [chemistry conventions](references/quick-contract.md#chemistry-sector-energies-and-integral-conventions)
sections supplement those records. Apply these checks when relevant:

- **Complex input to real-Hamiltonian QSVT recovery:** follow the
  [real/imaginary reconstruction recipe](references/qsvt/qsvt-recovery.md#complex-input-composition-for-a-real-hamiltonian).
  Recover the two real components separately, restore their norms, and recombine.
  Pass contiguous state arrays to CUDA-Q and check the raw complex vector against
  an independent reference before a parameter sweep.
- **States specified by site or qubit:** check amplitudes against the user's
  specification with `q[0]` as the least-significant bit; see
  [ordering](references/conventions/ordering.md). Agreement between a circuit and
  a dense oracle using the same incorrectly constructed input is insufficient.
- **Requested machine-readable results:** before delivering, follow
  [output validation](references/validation.md#requested-output-artifacts).
  Run the actual entrypoint and validate the complete payload at the requested
  interface (stdout, file, or return value), including required fields, types,
  identifiers and references. A separate complete file does not validate an
  incomplete stdout result. Distinguish requested caps from realized sizes.
- **Final scientific claims:** derive threshold decisions and selected candidates
  from the validated rows; reconcile the prose with the observed output. For
  executed work, include the exact reproduction command. Diagnose a conflicting observation
  before describing it as successful; see
  [completion](references/workflow.md#completion-checkpoint).

## Prerequisites

A local CUDA-Q Algorithms repository checkout matching the task's target version
is required to verify API behavior against source and tests. Establish its root
before following checkout-relative paths such as `python/cudaq_algorithms/`,
`tests/python/`, and `docs/sphinx/examples/python/`. An installed copy of this skill
alone does not include that source tree. If the checkout is unavailable, identify
the missing source evidence and label version-dependent claims unverified.

The routing script requires Python 3. Executing examples additionally requires
CUDA-Q Algorithms and the dependencies of the selected workflow in the execution
environment; the routing script itself does not import the package.

## Using the skill

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
| [route.py](scripts/route.py) | `python3 <skill-dir>/scripts/route.py [--full] "<request>"`; reads standard input when request arguments are omitted or empty. `--help` shows usage. | At most three starting guides with section names, plus focused operation records, applicable premise candidates, and shared checkpoints. `--full` prints complete selected guides. Standalone Bell/basic-kernel requests return a scope note; unmatched requests return the catalog entry point. Quantum phase estimation gets a note naming available building blocks. |

Here, `<skill-dir>` is the directory containing this `SKILL.md`, independently of
the repository checkout root. Open the relevant sections of the routed guides and
follow their focused-record/source pointers. Routing is a navigation aid, not a
scientific decision.

Use supplied data and derive intermediate inputs that the available model determines.
For example, specified geometry/basis/charge/spin can determine molecular integrals
and the nuclear constant. Ask only for missing choices that change the scientific
result or scope, and continue independent work. Do not invent data or APIs. Resolve
the version/source boundary of a requested complete inventory from context first;
ask when it remains ambiguous. When the available inputs resolve the task, proceed.

Use CUDA-Q Algorithms as a BLAS/LAPACK-like set of small scientific building
blocks. Start with the requested operation and mathematical object, choose the
smallest matching contract, and compose applications through explicit
representations, signatures, capabilities, and conventions.

Resolve links relative to their containing document, not the working directory.
Source/test paths are checkout-relative. Use these explicit locations, not
parent-directory or whole-filesystem searches for missing inputs.

- Known API or operation: before writing any answer or code, open its focused record
  with a command (`cat <this skill's directory>/references/<family>/<record>.md`; the
  [catalog](references/catalog.md) lists every path) and then the named current public
  source/test. The record carries the contract, boundaries, and provenance; the source
  carries behavior. Reading the package source alone is not a substitute: an answer
  about a contract, a capability boundary, or a dependency that cites no record is
  incomplete. The current checkout controls API behavior.
- Unknown family: select the operation and mathematical object in the
  [catalog](references/catalog.md), then open only the matching records.
- Multi-step advice, implementation, or repair: read the shared
  [workflow](references/workflow.md) before acting. Advice alone does not
  authorize file changes.
- Molecular energies or lattice dynamics without a named algorithm: start with
  [application composition](references/application-composition.md#workflow), then
  open the stage guides it names. Choose a method from the requested output,
  accuracy, and execution constraints; a default route does not mandate a circuit.

- Contract or boundary question (does X validate, is X supported, are X and Y
  interchangeable, what happens on a mismatch): check the applicable premise rows
  against the current source and cite the focused record. Keyword overlap alone
  does not establish that a row applies. Never confirm an unsupported premise.

Open the relevant [conventions](references/conventions.md) when translating
boundaries, [composition guide](references/application-composition.md) when
assembling a chain, and [validation guide](references/validation.md) when
checking claims. [Source lookup](references/source-provenance.md) gives shared paths.

Match signatures, representations, capability requirements, register geometry,
normalization, ordering, signs, phases, and execution layers. Fix an independent
oracle, precision, and tolerance before judging numerical agreement. Report the
evidence actually gathered; label requested code unexecuted when a run is unavailable.
Unsupported, absent, and unverified mean different things. Roadmap-only APIs are unavailable.

Installation, backend setup, and optimizer-driven NISQ workflows are outside this
skill; parameterized UCC construction remains an in-scope state-preparation primitive.
Dependencies, credentials, remote jobs, paid/shared QPUs, publishing, and external
contact require authorization covering those actions.

Before writing CUDA-Q kernels, read the
[kernel boundaries](references/quick-contract.md#kernel-boundaries).
Define kernels in a `.py` file run with `python3 file.py`; CUDA-Q cannot compile
kernels defined in a heredoc or `-c` string. Mint kernels and compute NumPy/host
data outside `@cudaq.kernel`; inside it allocate registers, apply gates, and call
captured kernels. Pass statevector input with `state_from(ket)` as a `cudaq.State`
argument; there is no `cudaq.prepare_state`. CUDA-Q statevectors are little-endian
(`q[0]` is the least-significant bit); the good subspace is the first
`2**num_system` amplitudes when the system register is allocated first.

For implementation requests, deliver a runnable script and the observed output when
execution is available; keep the requested artifact in the workspace. For predictions
or advice, supply the requested scientific result and sufficient evidence without
claiming circuit execution. Preserve user files. If inputs remain unavailable after
checking the named locations, identify what is missing and which work depends on it.
