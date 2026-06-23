# CUDA-Q Algorithms

CUDA-Q Algorithms is a primitive-first algorithms library built on CUDA-Q.

The initial transition imports selected reusable functionality from
`cudaqx/libs/solvers`:

- fermion-to-qubit transforms
- state-preparation circuits
- state-preparation excitation/operator-pool helpers

Application workflows such as VQE, ADAPT-VQE, QAOA, GQE, PySCF drivers, and
optimizer loops are intentionally not part of the initial extraction.

## Chemistry Inputs

CUDA-Q Algorithms does not provide an official bridge to PySCF or any other
electronic-structure package. Chemistry-facing tests and examples may use PySCF
to generate reference data, such as one- and two-electron integrals, but that
dependency stays at the test/example boundary.

The library APIs operate on reusable algorithmic inputs, such as one- and
two-body tensors, qubit Hamiltonians, Pauli words, and state-preparation
operator pools.
