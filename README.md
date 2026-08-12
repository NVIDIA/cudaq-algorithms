# CUDA-Q Algorithms

CUDA-Q Algorithms is a primitive-first algorithms library built on CUDA-Q,
focused on fault-tolerant quantum computing (FTQC) primitives: block
encodings, qubitization, quantum singular value transformation, product
formulas, and the state-preparation and fermion-to-qubit building blocks
they compose with.

NISQ-era application workflows such as VQE, ADAPT-VQE, QAOA, GQE, and
optimizer loops are intentionally out of scope.

## Python primitives

`cudaq_algorithms` is a pure-Python package implemented as CUDA-Q Python
kernels and host-side helpers (the only runtime requirement is the
`cudaq` Python package):

- fermion-to-qubit transforms (`fermion.jordan_wigner`,
  `fermion.bravyi_kitaev`)
- state-preparation kernels and operator pools (`stateprep`)
- Pauli LCU block encoding (`PauliLCU`, plus prepare/select/apply kernels)
- qubitization walks and Chebyshev moment measurement (`Walk`)
- QSVT phase sequences (`QSVT`, `PhaseSequence`)
- Suzuki-Trotter product formulas (`trotter.Trotter`, orders 1/2/4)
- chemistry input bridges (`chemistry.from_pyscf`, `chemistry.from_psi4`,
  `chemistry.from_fcidump`)

Simulation-only helpers (statevector access) are isolated in
`cudaq_algorithms.sim_utils`; everything else is hardware-shaped. The
documentation is a Sphinx site under [docs/](docs/) (build instructions in
[docs/README.md](docs/README.md)); the runnable, self-verifying examples
live in [docs/sphinx/examples/python/](docs/sphinx/examples/python/).

Classical chemistry preprocessing ships as a peer pure-Python module:
double factorization of two-electron integrals (X-DF and C-DF/RC-DF) on
NumPy/SciPy with optional CuPy GPU acceleration — see the preprocessing
guide in the Sphinx docs.

## Chemistry Inputs

The library APIs operate on reusable algorithmic inputs: one- and two-body
tensors, qubit Hamiltonians, Pauli words, and state-preparation operator
pools. The `chemistry` module provides bridges that produce those inputs
from electronic-structure packages — `from_pyscf` and `from_psi4` extract
molecular-orbital integrals from a converged mean-field calculation, and
`from_fcidump` parses the standard FCIDUMP interchange format. The
electronic-structure packages themselves are optional and never imported
at package-import time; everything downstream of the integrals runs
without them.

## License

The code in this repository is licensed under the [Apache License 2.0](./LICENSE). Dependency license references and attributions are listed in [NOTICE](./NOTICE).

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution requirements,
including Developer Certificate of Origin (DCO) sign-off.
