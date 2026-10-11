# Primitive catalog

Known operation: open its direct link below. Unknown operation: choose the family
for the mathematical object, then its focused contract; family selectors retain
the complete API rows. For a named API not linked here, search its exact public
name only within this directory and open the matching record. Links are relative
to this catalog, not the project's working directory. These are routing cues,
not a complete API inventory or evidence that unlisted APIs exist.

For a researcher who supplies physical inputs and asks for a result, start with
[the molecular or lattice workflow](application-composition.md#workflow), then
read **Workflow** and **Verification** in the selected family guides below.
These sections supply the scientific handoffs and numerical checkpoints;
focused records retain the API signatures. For LCU workflow guidance open
[Pauli LCU](block-encoding/pauli-lcu.md#workflow), while the block-encoding family
selector remains the entry point for protocol/provider questions.

| Operation / object | Family selector |
| --- | --- |
| Prepare states; [inject a preparation kernel into a consumer](state-preparation/injection-contract.md); construct [excitation pools](state-preparation/operator-pools.md); estimate preparation resources | [State preparation](state-preparation/state-preparation.md) |
| Encode a Pauli-sum operator in a zero-flagged unitary block with [PauliLCU](block-encoding/pauli-lcu.md); use or implement the shared `BlockEncoding` protocol and custom providers | [Block encoding](block-encoding/block-encoding.md) |
| Apply a [qubitization walk / Walk.kernel](qubitization/qubitization-walk.md) or measure [Chebyshev moments / Walk.moment](qubitization/qubitization-moments.md) | [Qubitization](qubitization/qubitization.md) |
| Apply [QSP/QSVT phase sequences](qsvt/qsvt-sequence.md) to an encoded spectrum or [recover real-time evolution](qsvt/qsvt-recovery.md) from cosine/sine blocks | [QSVT](qsvt/qsvt.md) |
| [Plan](trotter/trotter-planning.md) or apply Suzuki-Trotter evolution to a [prepared state](trotter/trotter-kernel-factory.md), [supplied cudaq.State](trotter/trotter-state-kernel-factory.md), or [live qview](trotter/trotter-apply-kernel.md); estimate [planned](trotter/trotter-resources-planned.md) or [raw flattened-list](trotter/trotter-resources-raw.md) logical resources | [Trotter](trotter/trotter.md) |
| Transform fermion ladder tensors to Pauli operators: [Jordan-Wigner](fermion-transforms/jordan-wigner.md) or [Bravyi-Kitaev](fermion-transforms/bravyi-kitaev.md) | [Fermion transforms](fermion-transforms/fermion-transforms.md) |
| Load molecular integrals from [FCIDUMP](chemistry/chemistry-from-fcidump.md), [PySCF](chemistry/chemistry-from-pyscf.md), or [Psi4](chemistry/chemistry-from-psi4.md); build [spin-orbital tensors](chemistry/chemistry-spin-orbital-tensors.md) or [qubit Hamiltonians](chemistry/chemistry-qubit-hamiltonian.md) | [Chemistry bridges](chemistry/chemistry-bridges.md) |
| [Double-factorize](double-factorization/double-factorization-explicit.md)/[compress](double-factorization/double-factorization-compressed.md) electron-repulsion integrals (ERIs); [reconstruct](double-factorization/double-factorization-reconstruction.md), [compare](double-factorization/double-factorization-error.md), form the [DF-corrected one-body matrix](double-factorization/double-factorization-modified-one-body.md), or estimate [LCU/Burg one-norms](double-factorization/double-factorization-one-norm.md) | [Double factorization](double-factorization/double-factorization.md) |
| Extract or analyze simulator statevectors with [sim_utils.good_subspace](simulation/simulation-good-subspace.md), [sim_utils.action](simulation/simulation-action.md), [sim_utils.transform](simulation/simulation-transform.md), or [sim_utils.evolve](simulation/simulation-evolve.md) | [Simulation analysis](simulation/simulation-analysis.md) |
| Build fault-tolerant circuit components from `cudaq_algorithms.primitives` (imported directly; nothing is re-exported at the package root): [QROM table lookup](primitives/primitives-qrom.md) (SELECT or SELECT-SWAP/QROAM), [unary-iteration SELECT walk](primitives/primitives-unary-iteration.md), in-place little-endian [reversible integer arithmetic](primitives/primitives-arithmetic.md) (adders, comparators, QFT family), or [alias-sampling PREPARE](primitives/primitives-alias-sampling.md); exact logical Toffoli counts of strictly unitary kernels | [Circuit primitives](primitives/primitives.md) |

QSVT phase generation and degree selection are not populated capabilities.
Quantum phase estimation is not a packaged primitive: its building blocks are
the block encoding's walk-step kernels (qubitization family) and Trotter
evolution, and `docs/sphinx/examples/python/08_quantum_phase_estimation.py` is
the self-verifying example (one QPE shell around exact, Trotter, and
qubitization-walk oracles; walk decoding `E = -alpha*cos(theta)`).
Simulation helpers return or manipulate statevectors; they are not device
kernels or shot-based QPU protocols. Compose selected contracts with the
[application guide](application-composition.md) and relevant [conventions](conventions.md).
