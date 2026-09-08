# Primitive catalog

This is a lightweight discovery index. It must not become a database of full
contracts or an application cookbook.

## Catalog policy

Route by the primary two-key identity — scientific operation plus mathematical
object — and only then check representation, capability, and signature
constraints. See `architecture.md` before changing this structure.

Add an entry only when its family reference contains source-grounded,
reviewable scientific content. Do not list anticipated work as an available
primitive.

**An entry is a lightweight summary projection of its family record, not a
second contract schema.** Where an entry's field name also exists in
`assets/primitive-record-template.md`, it carries the template's spelling and
meaning, so the two cannot say different things about the same field. Three
entry fields exist only in the catalog, because they compress several template
fields onto one line:

| Catalog-only field | What it projects |
| --- | --- |
| `Reference` | the family file to read; navigation only, no template counterpart |
| `Provides capability IDs` / `Requires capability IDs` | the template's `Stable ID` plus `Direction` from Capabilities and composition |
| `Provenance` | the template's `Source paths` plus `Commit/date last verified` |

An entry also deliberately omits every template field that does not compress
into one line — the scientific contract, inputs, outputs, dependencies,
accuracy, resources, and validation. The record owns those. The catalog is
never the authority for a contract; it only says which record to open.

Vocabularies are shared, not parallel: `Lifecycle` uses the single
`draft | verified | deprecated | removed` maturity vocabulary of
`architecture.md`, and a capability's own
`candidate | provisional | stable taxonomy contract` status is stated in its
capability record, never in this file.

## Populated family records

### State preparation

- Reference: [state-preparation.md](state-preparation.md)
- Operation + mathematical object (primary identity): prepare / quantum state
- Kind: quantum operation (packaged kernel factories) with host classical
  planning, validation, and resource-estimator helpers in the same family — a
  provisional compound of the template's `Kind` vocabulary
- Routine role: computational
- Abstraction level: leaf operation
- Parameterization: construction-time (both packaged factories bake their
  data and width in at factory time); runtime-parameterized device kernels
  exist in the same namespace but are not the injection seam
- Input representations: orbital-coefficient matrix; Givens rotation
  schedule; Hartree-Fock occupation; grouped fixed-parameter UCC Pauli words
  and coefficients; the system register
- Output representations: one-argument `(qubits: cudaq.qview)` preparation
  kernel
- Provides capability IDs: `cudaq-algorithms.state-preparation.unitary.v1`
  (provisional documentation/taxonomy contract, not a Python protocol)
- Requires capability IDs: none
- Execution layers: host preprocessing and validation, kernel factory,
  device kernel
- Domain: domain-independent seam; both packaged providers are tagged
  `domain: quantum-chemistry`
- Exactness: exact for the emitted circuit contract, subject to numerical
  synthesis and host input tolerances
- Uncertainty: deterministic
- Method: direct, plus a fixed-parameter ansatz form — `ansatz` is a
  provisional extension of the `direct | variational | heuristic` vocabulary
- Lifecycle: draft
- Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03),
  `python/cudaq_algorithms/stateprep/`, `docs/sphinx/guide/state_prep.rst`

### Excitation enumeration and operator pools

- Reference: [operator-pools.md](operator-pools.md)
- Operation + mathematical object (primary identity): preprocess /
  fermionic-excitation operator pool
- Kind: classical transformation
- Routine role: computational
- Abstraction level: leaf operation
- Parameterization: none
- Input representations: orbital and electron counts, spin, subset switches
- Output representations: ordered `list[cudaq.SpinOperator]`; grouped Pauli data
- Provides capability IDs: none
- Requires capability IDs: none
- Execution layers: host preprocessing
- Domain: quantum-chemistry
- Exactness: exact; Uncertainty: deterministic; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/stateprep/_pools.py`

### Block encoding

- Reference: [block-encoding.md](block-encoding.md)
- Operation + mathematical object (primary identity): encode / operator as a
  zero-flagged block of a unitary
- Kind: quantum operation (`PauliLCU`); adjacent representation and capability
  records define its composition boundary but are not primitive kinds
- Routine role: computational
- Abstraction level: leaf operation (`PauliLCU`)
- Parameterization: construction-time
- Input representations: `BlockEncoding` protocol; Pauli-sum Hamiltonian
- Output representations: protocol-conforming encoding and composable kernels
- Provides capability IDs: `cudaq-algorithms.block-encoding.zero-flagged.v1`
  (provisional documentation/taxonomy contract)
- Requires capability IDs: optionally
  `cudaq-algorithms.state-preparation.unitary.v1`
- Execution layers: host construction, kernel factory, device kernel
- Domain: domain-independent
- Exactness: exact encoding contract subject to coefficient pruning;
  Uncertainty: deterministic; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `block_encoding.py`, `pauli_lcu.py`

### Qubitization

- Reference: [qubitization.md](qubitization.md)
- Operation + mathematical object (primary identity): evolve / quantum state
  by a qubitization walk; measure / Chebyshev spectral moments
- Kind: quantum operation and measurement/readout protocol
- Routine role: computational
- Abstraction level: composite protocol
- Parameterization: construction-time
- Input representations: `BlockEncoding`; optional preparation kernel or state
- Output representations: compiled kernels; classical moment values
- Provides capability IDs: none
- Requires capability IDs:
  `cudaq-algorithms.block-encoding.zero-flagged.v1`; optionally
  `cudaq-algorithms.state-preparation.unitary.v1`
- Execution layers: host orchestration, kernel factory, device kernel,
  `cudaq.observe`
- Domain: domain-independent
- Exactness: exact walk circuit and estimator semantics; Uncertainty:
  deterministic circuit or shot/statistics dependent; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/qubitization.py`

### QSP and QSVT

- Reference: [qsvt.md](qsvt.md)
- Operation + mathematical object (primary identity): transform / encoded
  spectrum by a polynomial; reconstruct / good-subspace statevector pair
- Kind: quantum operation and host classical transformation
- Routine role: driver and computational
- Abstraction level: composite protocol and leaf companions
- Parameterization: construction-time and runtime, by provider
- Input representations: `PhaseSequence`; `BlockEncoding`; optional preparation
- Output representations: compiled kernels; `numpy.ndarray`
- Provides capability IDs: none
- Requires capability IDs:
  `cudaq-algorithms.block-encoding.zero-flagged.v1`; optionally
  `cudaq-algorithms.state-preparation.unitary.v1`
- Execution layers: host validation, kernel factory, device kernel,
  host post-processing
- Domain: domain-independent
- Exactness: exact for supplied phases; polynomial approximation is
  caller-owned; Uncertainty: deterministic; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/qsvt.py`

### Suzuki-Trotter evolution

- Reference: [trotter.md](trotter.md)
- Operation + mathematical object (primary identity): evolve / quantum state
  under a Pauli-sum Hamiltonian
- Kind: classical transformation, quantum operation, resource estimator, and
  simulation-only companion
- Routine role: computational and driver
- Abstraction level: leaf and composite, by provider
- Parameterization: construction-time and runtime, by provider
- Input representations: Pauli-sum Hamiltonian, state, time, order, steps
- Output representations: compiled kernels, resource estimate, or statevector
- Provides capability IDs: none
- Requires capability IDs: optionally
  `cudaq-algorithms.state-preparation.unitary.v1`
- Execution layers: host, kernel factory, device kernel, simulation-only helper
- Domain: domain-independent
- Exactness: approximate product formula; Uncertainty: deterministic;
  Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/trotter.py`

### Fermion-to-qubit transforms

- Reference: [fermion-transforms.md](fermion-transforms.md)
- Operation + mathematical object (primary identity): transform /
  fermionic ladder-coefficient tensors to a Pauli operator
- Kind: classical transformation
- Routine role: computational
- Abstraction level: leaf operation
- Parameterization: construction-time
- Input representations: one- and two-body ladder-coefficient tensors
- Output representations: `cudaq.SpinOperator`
- Provides capability IDs: none
- Requires capability IDs: none
- Execution layers: host preprocessing
- Domain: domain-independent
- Exactness: exact apart from tolerance pruning and floating point;
  Uncertainty: deterministic; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/fermion/`

### Chemistry bridges

- Reference: [chemistry-bridges.md](chemistry-bridges.md)
- Operation + mathematical object (primary identity): load and transform /
  fermionic integral tensors and Pauli operators
- Kind: classical transformation
- Routine role: computational and driver
- Abstraction level: leaf operations and one composite bridge
- Parameterization: construction-time
- Input representations: FCIDUMP, PySCF/Psi4 objects, chemist-notation tensors
- Output representations: integral triples, spin-orbital tensors,
  `cudaq.SpinOperator`
- Provides capability IDs: `cudaq-algorithms.chemistry-integrals.v1`
  (provisional documentation/taxonomy contract)
- Requires capability IDs: none
- Execution layers: host preprocessing
- Domain: quantum-chemistry
- Exactness: exact transformations subject to validation and tolerance pruning;
  Uncertainty: deterministic; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/chemistry.py`

### Double factorization

- Reference: [double-factorization.md](double-factorization.md)
- Operation + mathematical object (primary identity): factorize or compress /
  fermionic two-electron integral tensor
- Kind: classical transformation
- Routine role: computational
- Abstraction level: leaf operation
- Parameterization: construction-time
- Input representations: dense chemist-notation ERI tensor
- Output representations: `DoubleFactorization`, tensors, matrices, scalars
- Provides capability IDs: none
- Requires capability IDs: none
- Execution layers: host preprocessing, optionally CuPy-backed linear algebra
- Domain: quantum-chemistry
- Exactness: exact at full explicit rank or approximate under truncation and
  compression; Uncertainty: deterministic; Method: direct or optimization
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`,
  `python/cudaq_algorithms/double_factorization/`

### Simulation-only analysis

- Reference: [simulation-analysis.md](simulation-analysis.md)
- Operation + mathematical object (primary identity): analyze / simulated
  statevector
- Kind: simulation-only analysis
- Routine role: computational and driver
- Abstraction level: leaf and composite, by helper
- Parameterization: per call
- Input representations: block encoding, QSVT or Trotter object, statevector
- Output representations: unnormalized good-subspace block or full statevector
- Provides capability IDs: none
- Requires capability IDs: none; consumes concrete documented representations
- Execution layers: simulation-only host path using `cudaq.get_state`
- Domain: domain-independent
- Exactness: exact extraction or subject to the underlying algorithm;
  Uncertainty: deterministic; Method: direct
- Lifecycle: draft; Package/CUDA-Q versions verified: unverified
- Provenance: commit `61ac072d`, `python/cudaq_algorithms/sim_utils.py`

## Roadmap-only, not available

QROM and coherent alias sampling, encoding combinators, quantum arithmetic,
eigensolver protocols, sparse-oracle encodings, THC, and additional resource
models remain roadmap concepts. They are not available primitives and must not
be assigned plausible APIs.

## Entry format

`Reference` is a relative markdown link whose target is the family file name;
paths are relative to this file, which lives in `references/`.

```markdown
### Family name

- Reference: relative link to `family-name.md`
- Operation + mathematical object (primary identity):
- Kind:
- Routine role:
- Abstraction level:
- Parameterization:
- Input representations:
- Output representations:
- Provides capability IDs:
- Requires capability IDs:
- Execution layers:
- Domain:
- Exactness:
- Uncertainty:
- Method:
- Lifecycle:
- Package/CUDA-Q versions verified:
- Provenance:
```
