# Double factorization — family front door

Status: draft.

| Operation + object | Record |
| --- | --- |
| factorize / ERI tensor by explicit nested decomposition | [double-factorization-explicit.md](double-factorization-explicit.md) |
| compress / ERI tensor by optimized double factorization | [double-factorization-compressed.md](double-factorization-compressed.md) |
| reconstruct, compare, transform, or estimate / DF-derived arrays and scalars | [analysis front door](double-factorization-analysis.md) |

## Shared representation — `cudaq_algorithms.double_factorization.DoubleFactorization`

The public dataclass contains `num_orbitals`, lists of orthogonal
`leaf_rotations`, symmetric `leaf_cores`, method (`X-DF` or `C-DF`), and
method-specific first-factorization or optimizer status. Its mathematical
meaning is:

```text
(pq|rs) ~= sum_t sum_kl U[t,p,k] U[t,q,k] Z[t,k,l]
                         U[t,r,l] U[t,s,l]
```

It is host-side factorization data, not a quantum kernel or block encoding.
The only double-factorized encodings in this repository are examples.

Shared source is `python/cudaq_algorithms/double_factorization/`; tests are
`test_double_factorization.py`, `test_df_encoding.py`, and
`test_df_qsvt_bridge.py`. Current public source and tests are authoritative and
must be rechecked at use time.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.
