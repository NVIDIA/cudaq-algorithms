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

Use supplied data and derive intermediate inputs that the available model determines.
For example, specified geometry/basis/charge/spin can determine molecular integrals
and the nuclear constant. Ask only for missing choices that change the scientific
result or scope, and continue independent work. Do not invent data or APIs. Resolve
the version/source boundary of a requested complete inventory from context first;
ask when it remains ambiguous. When the available inputs resolve the task, proceed.

If a missing user choice clearly blocks requested implementation, ask the necessary
clarification before loading implementation routes. Continue independent work with
sufficient inputs, including architectural advice and work supported by inferable
inputs or harmless assumptions.

Keep a short request-to-evidence checklist: requested results, constructed library
objects or preparations, input conventions/physical sector, and output interface.
Before finishing, associate each item with its implementation or answer location
and observed check. Compare the actual constructor settings with the declared
input conventions; a default argument must not silently replace them. Numerical
agreement does not replace an explicitly requested operation or preparation, even
when the remaining calculation is classical.

Route the in-scope task with
`python3 <skill-dir>/scripts/route.py "<original request>"` or pass the original
request on standard input. Do not paraphrase away preparation requirements,
convention tags, or output fields. Read the complete bounded output without
truncating it with `head`: it includes selected exact calls, contracts and checks.
Reuse those excerpts. Follow current source/test pointers for the chosen operation;
open another reference section only to resolve an identified missing contract.
Use `--paths` for navigation only and `--full` when complete guides are needed.
If the user forbids local execution, navigate the [catalog](references/catalog.md)
and relevant records directly; keep the same contract and verification requirements.

The router selects relevant entries from the [Quick contract](references/quick-contract.md).
Additional patterns include [protocol-only encoding](references/quick-contract.md#consume-a-protocol-only-encoding),
[walk verification](references/quick-contract.md#qubitization-walk-action-and-prepared-state-checks),
and [chemistry conventions](references/quick-contract.md#chemistry-sector-energies-and-integral-conventions)
sections. Apply these checks when relevant:

- **Requested machine-readable results:** derive field/type/shape constraints from
  the request before execution, never from output. Follow
  [output validation](references/validation.md#requested-output-artifacts).
  Check all required fields, types, identifiers, references and meanings at the
  actual stdout/file/return-value interface; retain full schema and scientific
  checks. Use `check_output.py` only for requested JSON: `--format json` checks a
  whole document; `--format last-json-line` applies only to a declared final-line
  stdout contract. For JSON stdout, chain the actual producer into a fresh permitted
  capture, the checker with that format, then `cat` of the unchanged capture with
  `&&` in one shell call. Check the actual freshly produced file in its requested
  format; assert return values directly. Sidecars do not validate another interface.
  Honor execution/file-creation restrictions; use permitted existing output or
  report the check unavailable. Metadata must describe the actual computation and
  requested convention; distinguish requested caps from realized sizes.
- **Generated library calls:** before finalizing Python code, including advisory
  snippets, run `check_calls.py --summary` against the target checkout source using
  an existing `.py` file or a temporary file containing the complete snippet and
  imports, as shown below. Batch this with planned file creation or validation in
  the same tool call. If user restrictions prohibit the required file creation
  or local execution, or matching source is unavailable, use available contract
  evidence and state that this check was not run.
  Correct definite argument-binding errors. Investigate unresolved CUDA-Q
  Algorithms calls needed by the task; retain other unknowns without chasing
  every external call or requiring exit `0`. Rerun without `--summary` only when
  individual call evidence is needed. Keep the numerical, schema, and smoke
  checks required by the selected workflow: source binding cannot establish
  types, scientific correctness, CUDA-Q compilation, or runtime behavior.
- **Requested deliverables:** check each item against the actual artifact or
  answer using [task completion](references/validation.md#requested-task-completion).
  Trace results requested through a library object to its actual returned data
  or operation. An unused object does not establish that calculation, and an
  occupation list does not implement a requested state preparation. Keep
  independent reference calculations separate from the requested implementation.
- **Complex input to real-Hamiltonian QSVT recovery:** follow the
  [real/imaginary reconstruction recipe](references/qsvt/qsvt-recovery.md#complex-input-composition-for-a-real-hamiltonian).
  Recover the two real components separately, restore their norms, and recombine.
  Pass contiguous state arrays to CUDA-Q and check the raw complex vector against
  an independent reference before a parameter sweep.
- **States specified by site or qubit:** check amplitudes against the user's
  specification with `q[0]` as the least-significant bit; see
  [ordering](references/conventions/ordering.md). Agreement between a circuit and
  a dense oracle using the same incorrectly constructed input is insufficient.
- **Final scientific claims:** derive threshold decisions and selected candidates
  from the validated rows; reconcile the prose with the observed output. If no
  usefulness criterion is supplied, report the improvement and make sufficiency
  conditional instead of inventing a threshold. For
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

The bundled scripts use Python 3 and its standard library. Executing examples additionally requires
CUDA-Q Algorithms and the dependencies of the selected workflow in the execution
environment; the bundled scripts do not import scientific packages. The call checker
parses source without executing or rewriting it; the output checker reads data only.

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
| [route.py](scripts/route.py) | `python3 <skill-dir>/scripts/route.py [--paths \| --full] "<request>"`; reads standard input when request arguments are omitted or empty. `--help` shows usage. | Bounded relevant sections from selected guides and records, plus source pointers and shared checkpoints. `--paths` prints navigation only; `--full` prints complete selected guides. Standalone Bell/basic-kernel requests return a scope note; unmatched requests return the catalog entry point. Quantum phase estimation gets a note naming available building blocks. |
| [check_calls.py](scripts/check_calls.py) | `python3 <skill-dir>/scripts/check_calls.py --summary --package-root <checkout-root>/python/cudaq_algorithms snippet.py` | JSON counts of checked, erroneous and unknown calls, all errors with source evidence, and unknown reason counts. Omit `--summary` for individual diagnostics. Exit `1`: definite binding errors; `2`: unresolved calls or no checked calls; `0`: reported bindings checked without errors or unknowns, not overall validity. |
| [check_output.py](scripts/check_output.py) | `python3 <skill-dir>/scripts/check_output.py --input PATH --format json --field /values number '[3,4]'`; repeat `--field` for requested constraints; see `--help`. | stderr diagnostics; exit `0`: supplied structure only; `1`: malformed output/mismatch; `2`: invalid or incomplete check. |

If your agent runtime supports `run_script`, run these script paths through it
with the same arguments; otherwise use `python3` as shown. For the call checker,
pass `--summary --package-root <checkout-root>/python/cudaq_algorithms snippet.py` explicitly:
the source must match the task's target version, not a different installed copy.

Here, `<skill-dir>` is the directory containing this `SKILL.md`, independently of
the repository checkout root. Reuse the emitted sections and follow their
focused source pointers. Routing is a navigation aid, not a
scientific decision.

Use CUDA-Q Algorithms as a BLAS/LAPACK-like set of small scientific building
blocks. Start with the requested operation and mathematical object, choose the
smallest matching contract, and compose applications through explicit
representations, signatures, capabilities, and conventions.

Resolve links relative to their containing document, not the working directory.
Source/test paths are checkout-relative. Use these explicit locations, not
parent-directory or whole-filesystem searches for missing inputs.

- Known API or operation: use the relevant sections of its focused record from the
  router (or open the missing section; the [catalog](references/catalog.md) lists
  every path), then check the named current public
  source/test. The record carries the contract, boundaries, and provenance; the source
  carries behavior. Reading the package source alone is not a substitute: an answer
  about a contract, a capability boundary, or a dependency that cites no record is
  incomplete. The current checkout controls API behavior.
- Unknown family: select the operation and mathematical object in the
  [catalog](references/catalog.md), then open only the matching records.
- Multi-step advice, implementation, or repair: apply the emitted shared
  [workflow](references/workflow.md) checkpoints. Advice alone does not authorize
  changes to project files.
- Molecular energies or lattice dynamics without a named algorithm: start with
  [application composition](references/application-composition.md#workflow)
  sections selected by the router. Choose a method from the requested output,
  accuracy, and execution constraints; a default route does not mandate a circuit.

- Contract or boundary question (does X validate, is X supported, are X and Y
  interchangeable, what happens on a mismatch): check the applicable premise rows
  against the current source and cite the focused record. Keyword overlap alone
  does not establish that a row applies. Never confirm an unsupported premise.

For an unresolved boundary, consult the relevant [conventions](references/conventions.md),
[composition](references/application-composition.md), or [validation](references/validation.md)
section. [Source lookup](references/source-provenance.md) gives shared paths.

Match signatures, representations, capability requirements, register geometry,
normalization, ordering, signs, phases, and execution layers. Fix an independent
oracle, precision, and tolerance before judging numerical agreement. Report the
evidence actually gathered; label requested code unexecuted when a run is unavailable.
Unsupported, absent, and unverified mean different things. Roadmap-only APIs are unavailable.

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

Pass the request to `route.py` and follow the selected records. These examples
illustrate starting routes; they do not replace the source and verification checks.

| Request | Starting guide |
| --- | --- |
| Check a PauliLCU block encoding against a dense matrix | [Block encoding](references/block-encoding/block-encoding.md) |
| Compute molecular ground-state energies from geometry and a chosen basis | [Chemistry](references/chemistry/chemistry-bridges.md) and [application composition](references/application-composition.md) |
| Compare Suzuki-Trotter time evolution with exact dynamics | [Trotter](references/trotter/trotter.md) |
