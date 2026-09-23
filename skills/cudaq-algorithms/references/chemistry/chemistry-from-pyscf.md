# PySCF integral loader

Operation + object: **load** a **chemist-notation spatial
integral triple from a PySCF mean-field object**.

## Identity and classification

- Public symbol: `cudaq_algorithms.chemistry.from_pyscf`.
- Source: `python/cudaq_algorithms/chemistry.py`.
- Tests: `tests/python/test_psi4_conversion.py`.
- Runnable usage: the PySCF section of
  `docs/sphinx/guide/preprocessing.rst` and
  `docs/sphinx/examples/python/03_chemistry_to_ground_state.py`.
- Kind/role/layer: classical transformation, computational leaf, host
  provider bridge.
- Dependency: PySCF is imported when this provider is called; it is not needed
  to import or use the other chemistry providers.
- External PySCF version compatibility is unverified unless checked in the
  current environment; see the [shared integral contract](chemistry-bridges.md).

## Input, rejection, and output

`from_pyscf(mean_field)` expects a converged restricted mean-field object with
`mol`, a single 2-D `mo_coeff`, `get_hcore()`, and `energy_nuc()`. The source
explicitly rejects only a coefficient array whose rank is not two, including
the usual `(2, nao, nmo)` unrestricted/UHF form. It does not independently
check convergence, real-valued coefficients, orbital orthonormality, or object
class; missing/incompatible provider methods and PySCF failures propagate.

For `C = mean_field.mo_coeff`, the provider computes `C.T @ get_hcore() @ C`,
uses `pyscf.ao2mo.full`, and restores symmetry storage to a dense chemist-order
ERI with `ao2mo.restore(1, ..., nmo)`. Calling `get_hcore()` preserves any ECP,
X2C, or QM/MM contribution present in the mean field instead of rebuilding a
bare core Hamiltonian.

The result is `(one_body, eri, nuclear_repulsion)`: contiguous NumPy arrays of
shapes `(nmo,nmo)` and `(nmo,nmo,nmo,nmo)`, plus a Python `float`. It provides
`cudaq-algorithms.chemistry-integrals.v1`. No spin expansion, fermion transform,
factorization, state preparation, or quantum execution occurs.

## Resources and validation

There is no packaged resource estimator or quantum circuit. This call performs
the PySCF AO-to-MO transformation and materializes a dense `nmo^4` ERI; runtime
and peak memory are provider- and input-dependent and are not measured here.

The cited tests build restricted H2 and H4 mean fields, form qubit
Hamiltonians, and compare their ground energies with PySCF FCI within `1e-8`.
They also compare whole spectra with the Psi4 provider rather than raw MO
coefficients, which may differ by orbital phase or ordering. Add a rank-3
`mo_coeff` fixture for the explicit unrestricted rejection when validating
that boundary independently.
