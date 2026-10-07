# Double-factorization residual error

Operation + object: **compare** a **chemist ERI tensor with a
`DoubleFactorization` approximation**.

## Contract

`cudaq_algorithms.double_factorization.factorization_error(eri,
factorization)` returns the Python `float`
`||asarray(eri, dtype=float) - factorization.reconstruct_eri()||_F`. The result
is an absolute Frobenius norm: it is not divided by tensor size or input norm
and carries the same units as `eri`.

The helper relies on NumPy conversion and broadcasting rather than a dedicated
shape-validation contract. Callers must require a real chemist-notation tensor
with the exact reconstructed shape; otherwise conversion/broadcast behavior is
not a meaningful factorization metric. It emits no quantum resource or error
bound beyond the scalar computed for the supplied arrays.

## Validation

Independently reconstruct the tensor, form the residual, and compare with
`np.linalg.norm` using a predeclared tolerance. The runnable cases throughout
`tests/python/test_double_factorization.py` use this helper to distinguish
full-rank, thresholded, compressed, and nonconverged results. Current public
source and tests are authoritative and must be rechecked at use time; this
record was not freshly executed.
[Source lookup](../source-provenance.md) gives shared current-source paths.
