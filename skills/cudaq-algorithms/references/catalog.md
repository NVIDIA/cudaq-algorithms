# Primitive catalog

Route first by operation plus mathematical object, then open the linked record
to check inputs, outputs, signatures, capabilities, conventions, error behavior,
resources, and evidence. This is a discovery index, not a second contract.

All populated records have lifecycle `draft` and were historically
source-reviewed at the last-review anchor in
[source-provenance.md](source-provenance.md). Current public source and tests
remain authoritative. No record has a freshly executed package/CUDA-Q version
or SkillEvaluator result unless it says otherwise.

## State preparation and operator pools

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| preprocess / orbital-coefficient matrix into a Givens schedule | `cudaq_algorithms.stateprep.make_givens_rotation_schedule` | classical transformation; host | none | [Givens schedule](state-preparation/state-preparation-givens-schedule.md) |
| prepare / Slater-determinant state from a Givens schedule | `cudaq_algorithms.stateprep.slater_determinant_kernel` | quantum operation; host validation + kernel factory/device kernel | provides `cudaq-algorithms.state-preparation.unitary.v1` | [injectable Slater determinant](state-preparation/state-preparation-slater-determinant-kernel.md) |
| prepare / Hartree–Fock state with optional fixed-parameter UCC product | `cudaq_algorithms.stateprep.hartree_fock_ucc_kernel` | quantum operation; host validation + kernel factory/device kernel | provides `cudaq-algorithms.state-preparation.unitary.v1` | [HF + fixed UCC](state-preparation/state-preparation-hf-ucc.md) |
| prepare / contiguous Hartree–Fock occupation on a live register | `cudaq_algorithms.stateprep.hartree_fock` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [canonical HF device kernel](state-preparation/state-preparation-kernel-hartree-fock.md) |
| prepare / explicit occupation on a live register | `cudaq_algorithms.stateprep.hartree_fock_occupation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [occupation HF device kernel](state-preparation/state-preparation-kernel-hartree-fock-occupation.md) |
| apply / one UCCSD single excitation to a live register | `cudaq_algorithms.stateprep.single_excitation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [single-excitation device kernel](state-preparation/state-preparation-kernel-single-excitation.md) |
| apply / one UCCSD double excitation to a live register | `cudaq_algorithms.stateprep.double_excitation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [double-excitation device kernel](state-preparation/state-preparation-kernel-double-excitation.md) |
| apply / occupied-to-virtual UCCSD product to a live register | `cudaq_algorithms.stateprep.uccsd` | quantum operation; device kernel | runtime amplitudes and concrete caller-owned `cudaq.qview`; no capability ID | [UCCSD device kernel](state-preparation/state-preparation-kernel-uccsd.md) |
| apply / generalized UCCGSD product to a live register | `cudaq_algorithms.stateprep.uccgsd` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [UCCGSD device kernel](state-preparation/state-preparation-kernel-uccgsd.md) |
| apply / paired UpCCGSD product to a live register | `cudaq_algorithms.stateprep.upccgsd` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [UpCCGSD device kernel](state-preparation/state-preparation-kernel-upccgsd.md) |
| apply / coupled-exchange product to a live register | `cudaq_algorithms.stateprep.ceo` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [CEO device kernel](state-preparation/state-preparation-kernel-ceo.md) |
| apply / arbitrary fixed-parameter UCC product to a live register | `cudaq_algorithms.stateprep.fixed_parameter_ucc` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [fixed-UCC device kernel](state-preparation/state-preparation-kernel-fixed-parameter-ucc.md) |
| apply / adjacent real fermionic Givens rotation | `cudaq_algorithms.stateprep.givens_rotation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [Givens-rotation device kernel](state-preparation/state-preparation-kernel-givens-rotation.md) |
| apply / adjacent phase-aware fermionic Givens rotation | `cudaq_algorithms.stateprep.phase_givens_rotation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [phase-Givens device kernel](state-preparation/state-preparation-kernel-phase-givens-rotation.md) |
| prepare / real Slater determinant from flattened arrays | `cudaq_algorithms.stateprep.slater_determinant` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [real Slater device kernel](state-preparation/state-preparation-kernel-slater-determinant.md) |
| prepare / complex Slater determinant from flattened arrays | `cudaq_algorithms.stateprep.complex_slater_determinant` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [complex Slater device kernel](state-preparation/state-preparation-kernel-complex-slater-determinant.md) |
| preprocess / occupied-to-virtual UCCSD excitation pool | `cudaq_algorithms.stateprep.make_uccsd_operator_pool` | classical transformation; host | none | [UCCSD operator pool](state-preparation/operator-pool-uccsd.md) |
| preprocess / generalized UCCGSD excitation pool | `cudaq_algorithms.stateprep.make_uccgsd_operator_pool` | classical transformation; host | none | [UCCGSD operator pool](state-preparation/operator-pool-uccgsd.md) |
| preprocess / paired UpCCGSD excitation pool | `cudaq_algorithms.stateprep.make_upccgsd_operator_pool` | classical transformation; host | none | [UpCCGSD operator pool](state-preparation/operator-pool-upccgsd.md) |
| preprocess / coupled-exchange operator pool | `cudaq_algorithms.stateprep.make_ceo_operator_pool` | classical transformation; host | none | [CEO operator pool](state-preparation/operator-pool-ceo.md) |
| estimate / Givens determinant logical operations | `cudaq_algorithms.stateprep.estimate_givens_resources` | formula-level resource estimator; host | consumes a Givens schedule | [Givens resources](state-preparation/state-preparation-resources-givens.md) |
| estimate / canonical Hartree–Fock logical operations | `cudaq_algorithms.stateprep.estimate_hartree_fock_resources` | formula-level resource estimator; host | consumes qubit/electron counts | [canonical HF resources](state-preparation/state-preparation-resources-hartree-fock.md) |
| estimate / occupation-list Hartree–Fock logical operations | `cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources` | formula-level resource estimator; host | consumes an occupation list | [occupation HF resources](state-preparation/state-preparation-resources-hartree-fock-occupation.md) |
| estimate / fixed-parameter UCC logical operations | `cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources` | formula-level resource estimator; host | consumes grouped Pauli words (structural counts only) | [fixed-UCC resources](state-preparation/state-preparation-resources-fixed-parameter-ucc.md) |

The shared injection representation, exact one-register signature, consumer
table, and unsupported boundaries live in the
[state-preparation front door](state-preparation/state-preparation.md). Shared operator-pool
representation and provider routing live in the
[operator-pool front door](state-preparation/operator-pools.md); state-preparation estimator
routing lives in the [resource front door](state-preparation/state-preparation-resources.md), and
live-register kernels are collected by the
[device-kernel front door](state-preparation/state-preparation-device-kernels.md).

## Block encoding and spectral processing

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| encode / Pauli-sum operator as a zero-flagged unitary block | `PauliLCU` | quantum operation; host construction + kernel factory/device kernel | provides `cudaq-algorithms.block-encoding.zero-flagged.v1`; optionally requires `cudaq-algorithms.state-preparation.unitary.v1` | [Pauli LCU](block-encoding/pauli-lcu.md) |
| evolve / state by a qubitization walk | `Walk.kernel`, adjoint/controlled variants | quantum operation; kernel factory/device kernel | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; optional state preparation | [walk kernels](qubitization/qubitization-walk.md) |
| measure / Chebyshev moment | `Walk.moment`, `Walk.moments` | measurement/readout; kernel + observable + `cudaq.observe` | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; odd orders also require `select_observable` | [moments](qubitization/qubitization-moments.md) |
| transform / encoded spectrum with a phase sequence | `PhaseSequence`, `QSVT.kernel`, `QSVT.controlled_kernel` | quantum driver; host validation + kernel factory/device kernel | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; optional state preparation | [QSVT sequence](qsvt/qsvt-sequence.md) |
| reconstruct / real-time state from cosine and sine QSVT blocks | `recover_real_time_evolution` | classical transformation; host | concrete QSVT/good-block inputs | [QSVT recovery](qsvt/qsvt-recovery.md) |

The shared structural protocol and zero-flagged capability are in the
[block-encoding front door](block-encoding/block-encoding.md). QSVT phase generation and
degree selection are not populated capabilities.

## Product-formula evolution

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| preprocess / Pauli Hamiltonian into ordered product-formula terms | `make_trotter_terms`, `TrotterOrdering` | classical transformation; host | none | [Trotter planning](trotter/trotter-planning.md) |
| evolve / newly allocated all-zero or injected-preparation state by a stored Suzuki–Trotter plan | `cudaq_algorithms.Trotter.kernel` | quantum operation; kernel factory/device kernel | optionally consumes `cudaq-algorithms.state-preparation.unitary.v1` | [zero-argument Trotter factory](trotter/trotter-kernel-factory.md) |
| evolve / caller-supplied `cudaq.State` by a stored Suzuki–Trotter plan | `cudaq_algorithms.Trotter.state_kernel` | quantum operation; kernel factory/device kernel | concrete state input; no state-preparation capability | [state-input Trotter factory](trotter/trotter-state-kernel-factory.md) |
| apply / Suzuki–Trotter formula to a live qubit register | `cudaq_algorithms.trotter.apply_trotter` | quantum operation; device kernel | concrete flattened lists and caller-owned `cudaq.qview`; no capability ID | [low-level Trotter apply kernel](trotter/trotter-apply-kernel.md) |
| estimate / logical resources for a validated, pruned Trotter plan | `Trotter.resources` | resource estimator; host | consumes stored `Trotter` plan | [planned Trotter resources](trotter/trotter-resources-planned.md) |
| estimate / logical resources from caller-supplied flattened lists | `trotter.estimate_trotter_resources` | resource estimator; host | raw lists; no Hamiltonian capability inferred | [raw Trotter resources](trotter/trotter-resources-raw.md) |

Shared routing and provenance are summarized in the
[Trotter family front door](trotter/trotter.md); the two stored-plan factories are
compared by the [evolution front door](trotter/trotter-evolution.md), and the estimator
choices are collected by the [resource front door](trotter/trotter-resources.md).

## Fermion transforms and chemistry bridges

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| transform / ladder tensors to Jordan–Wigner Pauli operator | `fermion.jordan_wigner` | classical transformation; host | none | [Jordan–Wigner](fermion-transforms/jordan-wigner.md) |
| transform / ladder tensors to Bravyi–Kitaev Pauli operator | `fermion.bravyi_kitaev` | classical transformation; host | none | [Bravyi–Kitaev](fermion-transforms/bravyi-kitaev.md) |
| load / FCIDUMP text to chemist spatial integral triple | `chemistry.from_fcidump` | classical transformation; host parser | provides `cudaq-algorithms.chemistry-integrals.v1` | [FCIDUMP loader](chemistry/chemistry-from-fcidump.md) |
| load / restricted PySCF mean field to chemist spatial integral triple | `chemistry.from_pyscf` | classical transformation; host/provider bridge; PySCF at call time | provides `cudaq-algorithms.chemistry-integrals.v1` | [PySCF loader](chemistry/chemistry-from-pyscf.md) |
| load / restricted C1 Psi4 wavefunction to chemist spatial integral triple | `chemistry.from_psi4` | classical transformation; host/provider bridge; Psi4 at call time | provides `cudaq-algorithms.chemistry-integrals.v1` | [Psi4 loader](chemistry/chemistry-from-psi4.md) |
| transform / spatial integrals to spin-orbital tensors | `chemistry.spin_orbital_tensors` | classical transformation; host | requires `cudaq-algorithms.chemistry-integrals.v1` | [spin expansion](chemistry/chemistry-spin-orbital-tensors.md) |
| transform / spatial integrals to Jordan–Wigner qubit Hamiltonian | `chemistry.qubit_hamiltonian` | classical driver; host | requires `cudaq-algorithms.chemistry-integrals.v1` | [qubit Hamiltonian bridge](chemistry/chemistry-qubit-hamiltonian.md) |

The shared provider choice is collected in the
[integral-loader front door](chemistry/chemistry-integral-loaders.md). The chemistry
integral representation and capability are in the
[chemistry family front door](chemistry/chemistry-bridges.md).

## Double factorization

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| factorize / chemist ERI tensor by X-DF | `double_factorization.explicit_double_factorization` | classical transformation; host NumPy/optional CuPy | consumes chemist-integral representation | [explicit DF](double-factorization/double-factorization-explicit.md) |
| compress / chemist ERI tensor by C-DF or RC-DF | `double_factorization.compressed_double_factorization` | classical optimization; host NumPy/optional CuPy | consumes chemist-integral representation | [compressed DF](double-factorization/double-factorization-compressed.md) |
| reconstruct / dense chemist ERI tensor | `double_factorization.reconstruct_eri` | classical transformation; host | consumes `double_factorization.DoubleFactorization` | [DF reconstruction](double-factorization/double-factorization-reconstruction.md) |
| compare / ERI tensor with a factorization | `double_factorization.factorization_error` | classical reduction; host | consumes chemist ERI plus `double_factorization.DoubleFactorization` | [DF residual error](double-factorization/double-factorization-error.md) |
| transform / one-body and ERI tensors to corrected one-body matrix | `double_factorization.modified_one_body_integrals` | classical transformation; host | consumes two dense chemist-basis tensors, not `DoubleFactorization` | [modified one-body integrals](double-factorization/double-factorization-modified-one-body.md) |
| estimate / double-factorized Hamiltonian one-norm | `double_factorization.double_factorization_one_norm` | formula-level estimator; host | consumes `double_factorization.DoubleFactorization` plus Fock-like eigenvalues | [DF one-norm](double-factorization/double-factorization-one-norm.md) |

The shared `DoubleFactorization` representation and example-only quantum
encoding boundary are in the [family front door](double-factorization/double-factorization.md);
the four helper choices are also collected by the
[analysis front door](double-factorization/double-factorization-analysis.md).

## Simulation-only analysis

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| extract / zero-ancilla amplitude block | `sim_utils.good_subspace` | simulation analysis; host array slicing | concrete geometry | [good subspace](simulation/simulation-good-subspace.md) |
| analyze / `(H/alpha)|ket>` | `sim_utils.action` | simulation analysis; `cudaq.get_state` | concrete `PauliLCU.encode_kernel` | [action](simulation/simulation-action.md) |
| analyze / QSVT good-subspace vector | `sim_utils.transform` | simulation analysis; `cudaq.get_state` | concrete QSVT | [transform](simulation/simulation-transform.md) |
| evolve / Trotter statevector | `sim_utils.evolve` | simulation analysis; `cudaq.get_state` | concrete Trotter | [evolve](simulation/simulation-evolve.md) |

These return or manipulate statevectors. They are not device kernels or
shot-based QPU protocols.
