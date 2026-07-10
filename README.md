# CUDA-Q Algorithms

CUDA-Q Algorithms is a primitive-first algorithms library built on CUDA-Q,
focused on fault-tolerant quantum computing (FTQC) primitives.

It imports selected reusable functionality from `cudaqx/libs/solvers`:

- fermion-to-qubit transforms
- state-preparation circuits
- state-preparation excitation/operator-pool helpers

NISQ-era application workflows such as VQE, ADAPT-VQE, QAOA, GQE, PySCF drivers,
and optimizer loops are intentionally out of scope for this library and will not
be ported; they remain available in `cudaqx/libs/solvers`.

## Python primitives

Alongside the compiled bindings, `cudaq_algorithms` ships pure-Python
primitives implemented as CUDA-Q Python kernels (no compiled-extension
dependency):

- Pauli LCU block encoding (`PauliLCU`, plus prepare/select/apply kernels)
- qubitization walks and Chebyshev moment measurement (`Walk`)
- QSVT phase sequences (`QSVT`, `PhaseSequence`)

Simulation-only helpers (statevector access) are isolated in
`cudaq_algorithms.sim_utils`; everything else is hardware-shaped. See
[docs/pauli_lcu_qsvt.md](docs/pauli_lcu_qsvt.md) and
[examples/pauli_lcu_qsvt/](examples/pauli_lcu_qsvt/).

## Chemistry Inputs

CUDA-Q Algorithms does not provide an official bridge to PySCF or any other
electronic-structure package. Chemistry-facing tests and examples may use PySCF
to generate reference data, such as one- and two-electron integrals, but that
dependency stays at the test/example boundary.

The library APIs operate on reusable algorithmic inputs, such as one- and
two-body tensors, qubit Hamiltonians, Pauli words, and state-preparation
operator pools.
