# Double-factorized Hamiltonian one-norm

Status: draft. Operation + object: **estimate** a **double-factorized
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

For Fock-like eigenvalues `F_k` and leaf cores `Z^t`, the `lcu` convention is

```text
sum_k |F_k| + sum_t (sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk|).
```

For `Z^t = V^t diag(lambda^t) (V^t)^T`, the gauge-fixed `burg` convention is

```text
sum_k |F_k| + 1/4 sum_t sum_i |lambda^t_i|
                              * (sum_k |V^t_ki|)^2.
```

`one_body_eigenvalues` must be the diagonal/Fock-like one-body eigenvalues
required by this DF normalization convention—not a raw one-body matrix or an
arbitrary vector. The helper converts the input to a real array and sums its
absolute values, but validates neither its provenance nor that its length
equals `factorization.num_orbitals`; a plausible scalar can therefore be
scientifically wrong without raising.

This is a formula-level Hamiltonian normalization estimate. It is not a gate,
depth, runtime, memory, or end-to-end qubitization resource estimate, and the
API states no bound connecting it to algorithmic cost.

## Validation and evaluation

Compute both formulas by hand from a small core with known eigenpairs and
predeclared tolerance. The runnable pointer is
`tests/python/test_double_factorization.py::test_one_norms_match_hand_computation`;
`test_one_norm_conventions` covers both selectors and the invalid value.
Current public source and tests are authoritative and must be rechecked at use
time; this record was not freshly executed.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.

Declared eval coverage: `double-factorization-host-contracts` exercises the
Fock-like-eigenvalue input boundary, named convention, and resource abstraction.
