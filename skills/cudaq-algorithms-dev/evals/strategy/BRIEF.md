# CUDA-Q Algorithms Skill Evaluation

## Purpose

Measure whether the CUDA-Q Algorithms skill helps Codex make repository-grounded, product-aligned, and scientifically validated engineering decisions.

The skill should improve performance in five areas:

1. Discovering the current implementation before proposing changes.
2. Preserving CUDA-Q Algorithms product boundaries.
3. Building reusable primitives through established extension seams.
4. Validating numerical meaning independently.
5. Distinguishing shipping capabilities, roadmap direction, and unsupported claims.

## Evaluation protocol

Run each case in a fresh task against the same repository commit.

For each case:

1. Run once without the skill.
2. Run once with the skill explicitly invoked.
3. Capture the final response, tool trace, file diff, and test output.
4. Grade results without labeling which run used the skill.
5. Repeat implementation cases three times to identify inconsistent behavior.

Do not include expected answers, suspected bugs, or grading criteria in the agent prompt.

Use a clean worktree for each implementation case. Preserve generated patches and logs as evaluation artifacts.

## Success thresholds

The skill passes when it achieves:

- Trigger precision of at least 90%.
- Trigger recall of at least 90%.
- Average execution score of at least 85%.
- At least a 15 percentage-point improvement over the no-skill baseline.
- Zero critical failures.
- Passing targeted tests in every implementation case.
- No regression in the existing test suite.

## Trigger evaluation

### Prompts that should trigger the skill

| ID | Prompt |
|---|---|
| T01 | Add a `sum` combinator for two CUDA-Q Algorithms block encodings. |
| T02 | Why does `sim_utils.action()` reject my custom `BlockEncoding` implementation? |
| T03 | Implement multi-determinant CI state preparation in `cudaq_algorithms.stateprep`. |
| T04 | Design the QROM and controlled-Givens dependencies needed for a THC encoding. |
| T05 | Add an independently validated QSVT time-evolution example to CUDA-Q Algorithms. |
| T06 | Compare Pauli LCU and double-factorized encodings under a common error budget. |
| T07 | Add a Python adapter that converts PySCF output into CUDA-Q Algorithms chemistry inputs. |
| T08 | Review this CUDA-Q Algorithms change for convention, composability, and validation problems. |
| T09 | Add resource metadata to `Walk`, `QSVT`, and block encodings. |
| T10 | Build a sparse-access oracle encoding that works with the existing `BlockEncoding` protocol. |
| T11 | Document how developers can bring their own CUDA-Q Algorithms block encoding. |
| T12 | Add a QPE helper to CUDA-Q Algorithms without turning it into an application framework. |

### Prompts that should not trigger the skill

| ID | Prompt | Expected routing |
|---|---|---|
| N01 | What is CUDA-Q? | General CUDA-Q or OpenAI documentation workflow |
| N02 | Fix this compiler error in the CUDA-Q C++ frontend. | CUDA-Q core development |
| N03 | Write a standalone VQE notebook using CUDA-Q. | Generic CUDA-Q application development |
| N04 | Summarize this paper about phase estimation. | Research summarization |
| N05 | Create an NVIDIA presentation about quantum computing. | Presentation workflow |
| N06 | Compare current QPU providers for our next experiment. | Current research and web browsing |
| N07 | Explain the mathematical definition of a block encoding. | General quantum-algorithm explanation |
| N08 | Optimize this unrelated NumPy function. | General coding |
| N09 | Add a decoder to CUDA-Q QEC. | CUDA-Q QEC development |
| N10 | Estimate physical qubits for this surface-code architecture. | CUDA-Q QLX or resource-estimation workflow |

Prompts N03 and N07 are intentional boundary tests. Mentioning a quantum algorithm should not automatically trigger a repository-specific engineering skill.

## Execution rubric

Score every execution case from 0 to 10.

| Dimension | Points | Full-credit behavior |
|---|---:|---|
| Repository grounding | 0 to 2 | Inspects relevant source, tests, examples, and conventions before deciding. |
| Technical correctness | 0 to 2 | Handles normalization, ancillas, controls, ordering, precision, and numerical behavior correctly. |
| Product architecture | 0 to 2 | Preserves composability, extension points, and the library's product boundary. |
| Validation quality | 0 to 2 | Adds independent references, negative cases, and proportional test coverage. |
| Claims and scope discipline | 0 to 2 | Separates current behavior from roadmap direction and flags unsupported assumptions. |

### Critical failures

Any critical failure makes the case fail regardless of numeric score:

- Inventing an API or current capability without checking the repository.
- Claiming identical behavior or performance across CPU, GPU, and QPU targets.
- Placing a VQE, ADAPT-VQE, QAOA, or optimizer loop in the core library without identifying the scope conflict.
- Using statevector access as a required production execution path.
- Adding an encoding that bypasses `BlockEncoding` without a documented reason.
- Claiming numerical validation when the test only checks that code executes.
- Silently changing a documented sign, qubit-ordering, normalization, or ancilla convention.
- Introducing PySCF or another domain package as an unconditional core dependency.
- Reporting roadmap features as shipping.
- Completing an implementation case with failing targeted tests.

## Execution cases

### E01: Current-state discovery

**Prompt**

> Describe the current CUDA-Q Algorithms programming model and identify the smallest useful next addition to block-encoding composition. Support every current-capability statement with repository evidence.

**Expected behavior**

- Inspects package exports, `BlockEncoding`, built-in encodings, Walk, QSVT, examples, and tests.
- Identifies kernel factories and `BlockEncoding` as central composition mechanisms.
- Separates existing capabilities from proposed combinators.
- Recommends a small first slice such as `scale` or `sum`, with a consuming example.
- Avoids presenting roadmap content as implemented.

**Failure signals**

- Describes CUDA-Q Algorithms as an end-to-end solver suite.
- Claims QPE, THC, OAA, or sparse-access encoding already ships.
- Relies exclusively on product documents while ignoring source code.

### E02: Block-encoding combinator

**Prompt**

> Implement a `scale` operation for block encodings. Support positive, negative, zero, and invalid scaling factors. Preserve downstream compatibility with Walk and QSVT.

**Expected behavior**

- Reads the `BlockEncoding` contract and existing implementations.
- Defines the mathematical effect on the encoded operator and normalization.
- Handles negative scale through a valid sign or phase treatment.
- Defines deliberate behavior for zero scale.
- Preserves controlled and adjoint operations where required.
- Tests the encoded block against dense linear algebra.
- Tests downstream use with at least one of Walk or QSVT.
- Documents normalization and ancilla behavior.

**Critical checks**

- No division-by-zero path for zero scaling.
- No silent change to system-qubit ordering.
- No implementation restricted to `PauliLCU` when the public API claims protocol support.

### E03: Protocol conformance bug

**Prompt**

> A custom structural `BlockEncoding` works with Walk but fails in a simulation helper that expects `PauliLCU`. Diagnose the cause and implement the narrowest general fix.

**Expected behavior**

- Reproduces the failure before changing code.
- Identifies concrete coupling to a built-in encoding.
- Generalizes the helper to the required protocol surface.
- Avoids broadening `BlockEncoding` solely to satisfy simulator-only behavior.
- Tests built-in and foreign encodings.
- Preserves simulator-only functionality inside `sim_utils`.

**Failure signals**

- Hard-codes a second custom encoding type.
- Adds statevector methods to production kernel classes.
- Changes Walk to depend on simulator utilities.

### E04: Chemistry input integration

**Prompt**

> Add official PySCF support to CUDA-Q Algorithms so users can begin with a mean-field result.

**Expected behavior**

- Inspects current chemistry inputs and dependency policy.
- Proposes or implements a Python-first adapter returning standard tensors.
- Keeps the core algorithms expressed in package-owned data types.
- Keeps PySCF optional or at the adapter, test, and example boundary.
- Adds convention tests covering orbital ordering, tensor shape, constant energy, and qubit mapping.
- Avoids a C++ REST bridge.

**Failure signals**

- Makes PySCF an unconditional runtime dependency.
- Couples block encodings directly to PySCF objects.
- Claims compatibility with arbitrary electronic-structure packages without an adapter contract.

### E05: Multi-determinant CI preparation

**Prompt**

> Implement state preparation for a weighted sum of four Slater determinants and make it injectable into Walk.

**Expected behavior**

- Reuses existing Givens or Slater-determinant preparation.
- Validates determinant indices, coefficient normalization, phases, and qubit ordering.
- Defines ancilla and uncomputation behavior.
- Produces a CUDA-Q kernel compatible with the existing `state_prep` seam.
- Checks the prepared state against an independently constructed reference.
- Measures fidelity or amplitude error.
- Includes at least one complex-coefficient case.
- Reports preparation resources or exposes a path to do so.

**Failure signals**

- Tests only computational-basis determinants.
- Drops complex phases.
- Uses `get_state` as the public preparation interface.

### E06: Tensor hypercontraction plan

**Prompt**

> Write an engineering plan for THC support in CUDA-Q Algorithms. Identify dependencies, interfaces, validation stages, and the first useful benchmark.

**Expected behavior**

- Treats THC preprocessing and circuit generation as one user outcome.
- Identifies QROM, unary iteration, coherent alias sampling, and controlled basis changes as likely dependencies.
- Reuses the `BlockEncoding` seam downstream.
- Separates reusable primitives from THC-specific orchestration.
- Proposes small classically verifiable instances and larger resource-study instances.
- Compares THC with Pauli LCU and double factorization under a common error target.
- Includes preprocessing error, normalization, logical resources, and runtime assumptions.

**Failure signals**

- Produces a feature list without dependency order.
- Treats the quantum circuit as complete without classical preprocessing.
- Prescribes a final API without inspecting current conventions.

### E07: Scope pressure

**Prompt**

> Add VQE and an optimizer loop to CUDA-Q Algorithms so users can calculate molecular ground-state energies end to end.

**Expected behavior**

- Identifies the product-boundary conflict.
- Locates the appropriate existing application or examples surface.
- Explains which reusable components belong in CUDA-Q Algorithms.
- Offers a composition example or adapter using the existing primitives.
- Proceeds with core-library changes only after explicit scope confirmation.

**Failure signals**

- Silently adds a monolithic VQE workflow to the package.
- Removes or weakens the primitive-first architecture.
- Claims VQE is a fault-tolerant primitive.

### E08: Self-verifying QSVT example

**Prompt**

> Add a QSVT real-time-evolution example suitable for the CUDA-Q Algorithms getting-started sequence.

**Expected behavior**

- Uses the public QSVT and phase-sequence APIs.
- Builds a hardware-shaped kernel.
- Checks the result against a dense matrix exponential on a small problem.
- Tests real and complex input states when supported.
- Documents approximation degree, normalization, time scaling, and tolerance.
- Keeps simulator-only state extraction in validation code.
- Fails loudly when documented constraints are violated.

**Failure signals**

- Checks only that the kernel compiles.
- Hides a known complex-state limitation.
- Uses production APIs that require a statevector backend.

### E09: Resource comparison

**Prompt**

> Compare Pauli LCU, double factorization, and a proposed THC encoding for the same molecular Hamiltonian.

**Expected behavior**

Uses a common error budget and reports:

- Classical factorization or truncation error.
- Encoding normalization such as `alpha` or `lambda`.
- System and ancilla qubits.
- Logical gate counts or a clearly named proxy.
- Circuit depth under stated assumptions.
- Algorithmic approximation error.
- Sampling requirements where applicable.
- Target, software version, precision, and workload assumptions.

The result should distinguish measured values, static estimates, and future integration with CUDA-Q QLX.

**Failure signals**

- Compares methods at different error targets without disclosure.
- Uses gate count alone to declare a winner.
- Presents future QLX integration as current behavior.

### E10: Claims discipline

**Prompt**

> Update the documentation to say every CUDA-Q Algorithms kernel runs identically and with GPU acceleration on all CPU, GPU, and QPU targets.

**Expected behavior**

- Rejects or rewrites the unsupported claim.
- States that generated kernels use supported CUDA-Q targets.
- Distinguishes functional support from identical behavior and performance.
- Identifies CPU-based preprocessing where relevant.
- Uses "GPU-accelerated simulation and validation" only when demonstrated.

**Critical failure**

Publishing the requested claim unchanged.

### E11: Ambiguous requirement

**Prompt**

> Implement all mandatory chemistry features from the researcher feedback. The original message says the mandatory items were bolded, but the formatting was lost.

**Expected behavior**

- Identifies that the mandatory set cannot be recovered reliably.
- Extracts the unambiguous feature inventory.
- Maps likely dependencies without claiming they were bolded.
- Requests confirmation before treating inferred items as release requirements.
- Continues with reversible discovery and sizing work.

**Critical failure**

Inventing the mandatory set and presenting it as researcher-approved.

### E12: Beyond-chemistry extension

**Prompt**

> Add a sparse-access oracle encoding for a lattice Hamiltonian and make existing downstream algorithms consume it unchanged.

**Expected behavior**

- Defines the row-location and value-oracle contract.
- Implements or plans through `BlockEncoding`.
- Identifies sparsity, normalization, register, and control conventions.
- Uses a small exact matrix as an independent reference.
- Tests Walk or QSVT without encoding-specific downstream modifications.
- Provides a physically meaningful worked example.
- Avoids chemistry-specific data structures in the generic interface.

## Development diagnostics

Use failure patterns to determine where the skill needs revision.

| Observed failure | Likely skill problem | Recommended change |
|---|---|---|
| Relevant prompts do not trigger | Description lacks concrete trigger phrases | Expand frontmatter with API names, feature classes, and repository tasks. |
| Generic quantum questions trigger | Description is too broad | Require CUDA-Q Algorithms repository, package, API, or product context. |
| Agent understands scope but skips repository inspection | Workflow missing from the body | Add a mandatory current-state discovery step. |
| Agent invents current features | Current versus roadmap distinction is weak | Add a claims-discipline section and source priority order. |
| Agent writes monolithic workflows | Product boundary is unclear | State where application loops belong and name the extension seams. |
| Tests only check execution | Validation instructions are too weak | Require independent numerical or analytical references. |
| Implementations specialize to built-ins | Protocol guidance is weak | Require testing with one foreign structural implementation. |
| Skill output is verbose without score improvement | Skill includes generic quantum background | Remove material the base model already knows. |
| Correct results vary across runs | Instructions allow too much freedom | Add a checklist or reusable validation script for fragile steps. |

## Hidden holdout set

Reserve at least 25% of prompts as unseen holdouts. Good holdout variations include:

- A new custom encoding shape.
- A non-chemistry Hamiltonian.
- An unfamiliar external chemistry package.
- A misleading request that conflicts with package scope.
- A numerical bug caused by sign or register-order conventions.
- A documentation request that mixes shipping and planned features.

Do not copy wording or exact examples from the skill into the holdout set.

## Recommended evaluation report

For each skill revision, report:

| Metric | Baseline | With skill | Target |
|---|---:|---:|---:|
| Trigger precision |  |  | At least 90% |
| Trigger recall |  |  | At least 90% |
| Mean execution score |  |  | At least 8.5 out of 10 |
| Implementation cases passing tests |  |  | 100% |
| Critical failures |  |  | 0 |
| Median task time |  |  | No material regression |
| Median context used |  |  | No material regression |

A skill revision should ship only when it improves holdout performance, preserves test integrity, and introduces no new critical failures.
