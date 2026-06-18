# CUDA-Q Algorithms Examples

The examples directory is where CUDA-Q Algorithms composes reusable primitives
into full quantum application workflows.

The library itself should stay focused on primitives such as:

- fermion-to-qubit transforms
- state-preparation circuits and operator pools
- Pauli LCU block encodings
- qubitization walk and reflection primitives
- QSVT phase and walk sequencing
- Krylov and moment-processing utilities

Examples may combine those pieces with classical packages, generated phase
tables, exact diagonalization, chemistry drivers, or simulation-only helpers
such as `cudaq.get_state()`.

As a rule of thumb:

- Put reusable algorithmic operations in `include/`, `lib/`, and
  `python/cudaq_algorithms`.
- Put end-to-end workflows, comparisons against NumPy/SciPy, phase-generation
  demos, and domain-specific orchestration here.

## Hamiltonian Simulation

`hamiltonian_simulation/qsvt_pauli_lcu.py` demonstrates real-time Hamiltonian
simulation by composing:

1. `PauliLCU` block encoding,
2. QSPPACK-generated QSP phases,
3. `qsvt.apply_phase_sequence()`, and
4. an exact dense NumPy diagonalization reference.

The example uses `cudaq.get_state()` because it is a simulation validation
workflow. Hardware-oriented application code should measure observables or
sample output distributions instead of returning statevectors.

## Quantum Exact Lanczos

`quantum_exact_lanczos/quantum_exact_lanczos_h2.py` is the recommended starting point.
It is a pedagogical H2 example that prints the intermediate moments, Krylov
matrices, overlap filtering, and final energy reconstruction. The workflow is
based on Kirby, Motta, and Mezzacapo, "Exact and efficient Lanczos method on a
quantum computer" (arXiv:2208.00567).

`quantum_exact_lanczos/quantum_exact_lanczos_molecules.py` is the follow-on runner for
precomputed H2, LiH, N2, and benzene active-space fixtures. Both examples compose:

1. `PauliLCU` block encoding,
2. qubitization observables for Chebyshev moment collection,
3. `krylov.build_chebyshev_matrices()`, and
4. an example-local generalized eigenproblem solve with overlap filtering.

The molecule Hamiltonians under `quantum_exact_lanczos/data/` are example data
fixtures, not tests and not a supported molecule-construction API. The examples
keep molecule construction and overlap conditioning outside the public library
API.
