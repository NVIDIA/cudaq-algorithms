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
