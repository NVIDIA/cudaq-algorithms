# Modified one-body integrals for double factorization

Operation + object: **transform** **one-body and chemist ERI
tensors into the DF-corrected one-body matrix**.

## Contract

`cudaq_algorithms.double_factorization.modified_one_body_integrals(one_body,
eri)` converts both inputs to real NumPy arrays and returns

```text
kappa[p,q] = one_body[p,q] - 1/2 sum_r eri[p,r,q,r].
```

This helper consumes two dense tensors, not a `DoubleFactorization`. The
intended shapes are `(n,n)` and `(n,n,n,n)` in the same spatial-orbital basis.
The source relies on NumPy subtraction/einsum compatibility and supplies no
complete public shape, Hermiticity, symmetry, or provenance validation; check
those conditions before calling. Complex parts are discarded by `dtype=float`
conversion and are outside this real-integral contract.

The result is host data used when assembling the full double-factorized
Hamiltonian. No resource estimator or quantum kernel is provided.

For the [one-norm helper](double-factorization-one-norm.md), form the distinct
Fock-like matrix `F[p,q] = kappa[p,q] + sum_r eri[p,q,r,r]` and pass its
eigenvalues. The helper above still returns `kappa`, not `F`. When assessing
approximate factors, use their reconstructed ERI in both contractions. Keep
the original physical one-body matrix and scalar when building the ordinary
two-electron Hamiltonian.

## Validation

Use an explicit `p,q,r` loop independent of `einsum`. The runnable pointer is
`tests/python/test_double_factorization.py::test_modified_one_body_matches_independent_loop`;
`test_modified_one_body_integrals` supplies an additional formula check.
Current public source and tests are authoritative and must be rechecked at use
time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
