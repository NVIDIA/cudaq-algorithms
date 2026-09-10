# Double-factorization analysis — front door

Status: draft. Choose the record for the exact output the application needs;
these helpers do not share one input or return contract.

| Operation + object | Public entry point | Record |
| --- | --- | --- |
| reconstruct / dense chemist ERI tensor | `double_factorization.reconstruct_eri` | [reconstruction](double-factorization-reconstruction.md) |
| compare / ERI tensor with a factorization | `double_factorization.factorization_error` | [residual error](double-factorization-error.md) |
| transform / one-body and ERI tensors to corrected one-body integrals | `double_factorization.modified_one_body_integrals` | [modified one-body integrals](double-factorization-modified-one-body.md) |
| estimate / double-factorized Hamiltonian one-norm | `double_factorization.double_factorization_one_norm` | [one-norm](double-factorization-one-norm.md) |

All four are host-side NumPy helpers. They emit no kernel, do not choose or
certify a factorization, and do not turn the example-only double-factorized
encodings into package APIs. Shared representation and provenance are in
[double-factorization.md](double-factorization.md).

Current public source and tests are authoritative and must be rechecked at use
time. [source-provenance.md](../source-provenance.md) records historical
last-review audit context.
