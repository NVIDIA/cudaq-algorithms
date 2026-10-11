# Psi4 integral loader

Operation + object: **load** a **chemist-notation spatial
integral triple from a Psi4 wavefunction**.

## Identity and classification

- Public symbol: `cudaq_algorithms.chemistry.from_psi4`.
- Source: `python/cudaq_algorithms/chemistry.py`.
- Tests: `tests/python/test_psi4_conversion.py`.
- Runnable usage: the Psi4 section of
  `docs/sphinx/guide/preprocessing.rst`.
- Kind/role/layer: classical transformation, computational leaf, host
  provider bridge.
- Dependency: Psi4 is imported when this provider is called; it is not needed
  to import or use the other chemistry providers.
- External Psi4 version compatibility is unverified unless checked in the
  current environment; see the [shared integral contract](chemistry-bridges.md).

## Input, rejection, and output

`from_psi4(wavefunction)` expects a converged restricted wavefunction exposing
`nirrep()`, `same_a_b_orbs()`, `Ca()`, `H()`, `basisset()`, and `molecule()`.
The source explicitly rejects `nirrep() != 1` because the dense `Ca` path
requires C1 symmetry, and rejects differing alpha/beta orbitals. It does not
independently check convergence, the named RHF/RKS reference type, real-valued
coefficients, or orbital orthonormality; incompatible provider objects and
Psi4 failures propagate. The rejection message names RKS as accepted, and a
restricted Kohn-Sham wavefunction does pass both checks, but its integrals
give a Hartree-Fock-type second-quantized Hamiltonian in the Kohn-Sham
orbital basis with the exchange-correlation functional dropped, not a DFT
energy; use an RHF reference for a Hamiltonian consistent with the mean-field
energy.

For the alpha-orbital coefficient matrix `C`, the provider computes
`C.T @ wavefunction.H() @ C`. It constructs
`psi4.core.MintsHelper(wavefunction.basisset())` and calls `mo_eri(C,C,C,C)`,
whose result is used as the dense chemist-order spatial ERI.

The result is `(one_body, eri, nuclear_repulsion)`: contiguous NumPy arrays of
shapes `(nmo,nmo)` and `(nmo,nmo,nmo,nmo)` in the C1 MO basis, plus a Python
`float` from
`wavefunction.molecule().nuclear_repulsion_energy()`. It provides
`cudaq-algorithms.chemistry-integrals.v1`. No spin expansion, fermion transform,
factorization, state preparation, or quantum execution occurs.

## Resources and validation

There is no packaged resource estimator or quantum circuit. This call performs
Psi4 integral transformation and materializes a dense rank-4 ERI; runtime and
peak memory are provider- and input-dependent and are not measured here.

The cited test builds restricted C1 H2 and H4 wavefunctions and compares the
resulting qubit-Hamiltonian spectra with independently extracted PySCF tensors
using `np.allclose(..., atol=1e-6)` with NumPy's default relative tolerance.
The two explicit failures, multiple irreps and unequal alpha/beta orbitals,
are source-checked but untested in-repo; validation of this provider must
exercise both.
