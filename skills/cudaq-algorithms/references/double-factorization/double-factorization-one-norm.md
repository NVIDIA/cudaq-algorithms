# Double-factorized Hamiltonian one-norm

Operation + object: **estimate** a **double-factorized
Hamiltonian one-norm**.

## Contract

```python
cudaq_algorithms.double_factorization.double_factorization_one_norm(
    factorization, one_body_eigenvalues, convention="lcu"
)
```

returns a nonnegative Python `float` under the named `"lcu"` or `"burg"`
formula; any other convention raises `ValueError`. The formulas differ, so the
convention must accompany every reported value.

For physical one-body integrals `h` and the factorization's reconstructed
chemist ERI, the Fock-like matrix is

```text
F[p,q] = h[p,q] - 1/2 sum_r eri_rec[p,r,q,r] + sum_r eri_rec[p,q,r,r].
```

This is Eq. (2) of [the RC-DF paper](https://arxiv.org/html/2212.07957v3#S1.E2).
Compute it consistently with the factors:

```python
eri_rec = df.reconstruct_eri(factorization)
kappa = df.modified_one_body_integrals(h, eri_rec)
F = kappa + np.einsum("pqrr->pq", eri_rec)
one_body_eigenvalues = np.linalg.eigvalsh(F)
```

Here `df` is `cudaq_algorithms.double_factorization` and `np` is NumPy.
For eigenvalues `F_k` of this matrix and leaf cores `Z^t`, the `lcu` convention is

```text
sum_k |F_k| + sum_t (sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk|).
```

For `Z^t = V^t diag(lambda^t) (V^t)^T`, the gauge-fixed `burg` convention is

```text
sum_k |F_k| + 1/4 sum_t sum_i |lambda^t_i|
                              * (sum_k |V^t_ki|)^2.
```

`one_body_eigenvalues` must be the diagonal/Fock-like one-body eigenvalues
required by this DF normalization convention, rather than eigenvalues of `h`
or `kappa`, a raw matrix, or an arbitrary vector. The helper converts the input
to a real array and sums its
absolute values, but validates neither its provenance nor that its length
equals `factorization.num_orbitals`; a plausible scalar can therefore be
scientifically wrong without raising.

This is a formula-level Hamiltonian normalization estimate. It is not a gate,
depth, runtime, memory, or end-to-end qubitization resource estimate, and the
API states no bound connecting it to algorithmic cost.
The returned value excludes the scalar/identity contribution; preserve that
offset separately when reconstructing energies or defining an encoding.

## Validation

Compute both formulas by hand from a small core with known eigenpairs and
predeclared tolerance. The runnable pointer is
`tests/python/test_double_factorization.py::test_one_norms_match_hand_computation`;
`test_one_norm_conventions` covers both selectors and the invalid value.
These tests supply eigenvalues directly; independently check the contractions
that form `F` when validating the full integral-to-norm recipe.
Current public source and tests are authoritative and must be rechecked at use
time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
