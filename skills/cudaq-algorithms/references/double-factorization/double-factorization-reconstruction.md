# Double-factorization ERI reconstruction

Status: draft. Operation + object: **reconstruct** a **dense chemist-notation
ERI tensor from `DoubleFactorization` data**.

## Contract

`cudaq_algorithms.double_factorization.reconstruct_eri(factorization)` returns
the same dense NumPy tensor as `factorization.reconstruct_eri()`. Its shape is
`(n,n,n,n)` for `factorization.num_orbitals == n`, with the approximation and
chemist convention represented by the stored rotations and cores. It accepts a
`DoubleFactorization`; it does not accept an original ERI or report an error.

This is a deterministic host transformation. It performs no new fitting,
optimizer-status check, quantum encoding, or normalization. Dense allocation
can be material; no runtime, memory, or quantum-resource estimator is provided.

## Validation and evaluation

Compare against a direct four-index loop over the documented factorization
formula, not only the method wrapper. The runnable repository pointer is
`tests/python/test_double_factorization.py::test_reconstruct_eri_matches_helper`;
the full-rank and synthetic cases in that file exercise the reconstructed
tensor. Current public source and tests are authoritative and must be rechecked
at use time; this record was not freshly executed.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.

Declared eval coverage: `double-factorization-encoding-boundary` and
`chemistry-end-to-end-composition` in `../../evals/evals.json`.
