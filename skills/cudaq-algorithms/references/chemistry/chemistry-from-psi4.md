# Psi4 integral loader

Status: draft. Operation + object: **load** a **chemist-notation spatial
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
- Lifecycle/evidence: draft; current public source and tests are authoritative
  and must be rechecked at use time; not freshly executed. External Psi4 version
  compatibility is
  unverified. [source-provenance.md](../source-provenance.md) records historical
  last-review audit context.

## Input, rejection, and output

`from_psi4(wavefunction)` expects a converged restricted wavefunction exposing
`nirrep()`, `same_a_b_orbs()`, `Ca()`, `H()`, `basisset()`, and `molecule()`.
The source explicitly rejects `nirrep() != 1` because the dense `Ca` path
requires C1 symmetry, and rejects differing alpha/beta orbitals. It does not
independently check convergence, the named RHF/RKS reference type, real-valued
coefficients, or orbital orthonormality; incompatible provider objects and
Psi4 failures propagate.

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
Validation of this provider must also exercise the two explicit failures:
multiple irreps and unequal alpha/beta orbitals.

Eval coverage: the optional-dependency assertion in authored
`chemistry-bridge-dependency-boundary` covers Psi4 indirectly; no dedicated
positive Psi4-provider eval is present. Baseline and with-skill arms have not
been run.
