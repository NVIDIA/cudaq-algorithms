# Explicit double factorization

Status: draft. Operation + object: **factorize** a **chemist-notation
two-electron integral tensor**.

## Identity and classification

- Public symbol:
  `cudaq_algorithms.double_factorization.explicit_double_factorization`.
- Source: `double_factorization/_factorization.py`; backend selection in
  `_backend.py`.
- Tests: `test_double_factorization.py`.
- Kind/role: classical transformation, computational leaf, host.
- Dependencies: NumPy/SciPy; optional CuPy/GPU backend.
- Exactness: exact at retained numerical rank, approximate under thresholds or
  leaf cap.

## Scientific contract

Input `eri` is a real square rank-4 `(n,n,n,n)` tensor in chemist notation with
the real-orbital eightfold symmetry. The function first factorizes the
`n^2 x n^2` supermatrix, then diagonalizes each symmetric leaf to produce
orthogonal rotations and rank-one cores.

```python
explicit_double_factorization(
    eri,
    threshold=1.0e-8,
    max_num_leaves=None,
    second_factor_threshold=0.0,
    first_factorization="cholesky",
    backend="auto",
)
```

- `cholesky` is pivoted/rank-revealing for positive semidefinite ERIs.
- `eigendecomposition` retains eigenmodes by magnitude and is required for
  indefinite inputs.
- `max_num_leaves` caps the first-factor rank.
- `second_factor_threshold` zeros small second-factor modes using their
  importance-weighted contribution.
- `backend` is `numpy`, `cupy`, or size-aware `auto`; explicit `cupy` errors if
  a real GPU kernel cannot run.

The output is `DoubleFactorization(method="X-DF")` with NumPy arrays,
`first_factorization`, and `leaf_weights`. Cholesky may warn when an indefinite
supermatrix has a material negative part.

## Accuracy, resources, and validation

Thresholds and leaf caps define reconstruction error; no universal error bound
or quantum resource estimate is returned. Dense storage and decomposition
costs are implementation facts, not measured performance. Source comments
about CPU/GPU crossover thresholds are selection heuristics, not a measurement
from the current run.

Independent oracle: reconstruct the ERI and compare Frobenius residual;
cross-check the full-rank result with a direct symmetric eigendecomposition or
OpenFermion reference. Test threshold monotonicity, true numerical rank,
Cholesky/eigendecomposition agreement, asymmetry rejection, indefinite warning,
backend failure, and second-factor pruning. Current public source and tests are
authoritative and must be rechecked at use time; this record was not freshly
executed. Runnable
pointers are the explicit-factorization cases in
`tests/python/test_double_factorization.py` and
`docs/sphinx/examples/python/04_double_factorization_and_the_protocol.py`.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.

Declared eval coverage: `double-factorization-host-contracts` exercises the
explicit ERI input, public operation, host-data output, and encoding boundary.
