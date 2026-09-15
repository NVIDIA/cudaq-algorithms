# Compressed double factorization

Status: draft. Operation + object: **compress** a **chemist-notation
two-electron integral tensor**.

## Identity and classification

- Public symbol:
  `cudaq_algorithms.double_factorization.compressed_double_factorization`.
- Source: `double_factorization/_factorization.py`; backend helpers in
  `_backend.py`.
- Tests: `test_double_factorization.py`.
- Kind/role: classical transformation, computational leaf, host optimization.
- Dependencies: NumPy/SciPy; optional CuPy/GPU backend.
- Exactness/method: approximate, deterministic numerical optimization.

## Scientific contract

C-DF fixes `num_leaves`, writes each rotation as `U[t] = exp(X[t])` for an
antisymmetric generator, solves symmetric cores inside the objective, and uses
L-BFGS-B for the outer variables. It minimizes half the squared Frobenius
reconstruction residual. `regularization > 0` adds the RC-DF core penalty.

Material controls:

- `num_leaves >= 1`;
- `max_iterations` and outer `tolerance`;
- `regularization`;
- inner solver `lstsq` or matrix-free `cg`;
- CG tolerance, iteration cap, warm start, and optional in-loop tolerance;
- optional initial antisymmetric generators;
- backend `auto`, `numpy`, or `cupy`.

The same square real chemist tensor and symmetry checks as explicit DF apply.
The output is `DoubleFactorization(method="C-DF")` with optimizer success,
iteration count, and gradient norm. If optimization does not converge, the best
factorization is returned with `optimizer_success=False` and a `RuntimeWarning`.
Never hide that status or call the result converged.

## Accuracy, resources, and validation

Approximation depends on leaf count, regularization, optimizer termination, and
inner-solve accuracy. Regularization can trade reconstruction error for smaller
cores/one-norm; it is not a guaranteed quantum speedup. No packaged quantum
kernel or end-to-end resource estimator is returned.

Independent oracle: reconstruct the ERI and compare the residual with explicit
DF at the same leaf count and with a known low-rank synthetic tensor. Test exact
true-rank recovery, regularization effects, `cg`/`lstsq` agreement, warm-start
functional consistency, optimizer-limit warnings/status, invalid inputs, and
backend fallback. Runnable pointers are the compressed-factorization cases in
`tests/python/test_double_factorization.py` and
`docs/sphinx/examples/python/df_compression_to_qsvt.py`. Current public source
and tests are authoritative and must be rechecked at use time; this record was
not freshly executed.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.

Declared eval coverage: `double-factorization-encoding-boundary` and
`chemistry-end-to-end-composition`.
