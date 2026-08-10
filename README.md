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

CUDA-Q Algorithms does not provide an official bridge to PySCF or any other
electronic-structure package. Chemistry-facing tests and examples may use PySCF
to generate reference data, such as one- and two-electron integrals, but that
dependency stays at the test/example boundary.

The library APIs operate on reusable algorithmic inputs, such as one- and
two-body tensors, qubit Hamiltonians, Pauli words, and state-preparation
operator pools.

## License

The code in this repository is licensed under the [Apache License 2.0](./LICENSE). Dependency license references and attributions are listed in [NOTICE](./NOTICE).

See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution requirements,
including Developer Certificate of Origin (DCO) sign-off.
