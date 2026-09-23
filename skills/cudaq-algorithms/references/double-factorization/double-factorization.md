# Double factorization — family front door

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| factorize / chemist ERI tensor by X-DF | `double_factorization.explicit_double_factorization` | classical transformation; host NumPy/optional CuPy | consumes chemist-integral representation | [explicit DF](double-factorization-explicit.md) |
| compress / chemist ERI tensor by C-DF or RC-DF | `double_factorization.compressed_double_factorization` | classical optimization; host NumPy/optional CuPy | consumes chemist-integral representation | [compressed DF](double-factorization-compressed.md) |
| reconstruct / dense chemist ERI tensor | `double_factorization.reconstruct_eri` | classical transformation; host | consumes `double_factorization.DoubleFactorization` | [DF reconstruction](double-factorization-reconstruction.md) |
| compare / ERI tensor with a factorization | `double_factorization.factorization_error` | classical reduction; host | consumes chemist ERI plus `double_factorization.DoubleFactorization` | [DF residual error](double-factorization-error.md) |
| transform / one-body and ERI tensors to corrected one-body matrix | `double_factorization.modified_one_body_integrals` | classical transformation; host | consumes two dense chemist-basis tensors, not `DoubleFactorization` | [modified one-body integrals](double-factorization-modified-one-body.md) |
| estimate / double-factorized Hamiltonian one-norm | `double_factorization.double_factorization_one_norm` | formula-level estimator; host | consumes `double_factorization.DoubleFactorization` plus Fock-like eigenvalues | [DF one-norm](double-factorization-one-norm.md) |

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

The reconstruction, residual-error, modified-one-body, and one-norm helpers
are all host-side NumPy operations, but they do not share one input or return
contract. They emit no kernel, do not choose or certify a factorization, and do
not turn the example-only double-factorized encodings into package APIs. Select
the focused record for the exact output the application needs.

Shared source is `python/cudaq_algorithms/double_factorization/`; tests are
`test_double_factorization.py`, `test_df_encoding.py`, and
`test_df_qsvt_bridge.py`. Current public source and tests are authoritative and
must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.

## Premise check (claim → verdict)

Test every claim the request makes or assumes against this table before answering; a matching row is
the answer to give, cited by the record path in its last cell, and the records named there are read
with the `cat` command shown.

| If the request claims or assumes | Verdict | Say (demanded words first) |
| --- | --- | --- |
| Compress the tensor: compressed factorization has one knob (`num_leaves`) and returns a converged result | false | **The material controls are `num_leaves >= 1`, `max_iterations` and the outer `tolerance`, `regularization`, the inner solver (`lstsq` or matrix-free `cg`, with its CG tolerance, iteration cap, warm start and optional in-loop tolerance), optional initial antisymmetric generators, and `backend` (`auto`, `numpy`, `cupy`); the returned `DoubleFactorization(method="C-DF")` carries `optimizer_success`, `optimizer_nit`, and `optimizer_grad_norm`, and a non-converged run returns the best factorization with `optimizer_success=False` plus a `RuntimeWarning`.** Report that status with the result; never call it converged or hide the warning. Record: `cat <this skill's directory>/references/double-factorization/double-factorization-compressed.md` |
| Double factorization returns a packaged quantum kernel or `DoubleFactorizedEncoding` | false | **`DoubleFactorization` is host-side factorization data (`num_orbitals`, `leaf_rotations`, `leaf_cores`, method, optimizer status), not a kernel and not a block encoding; the double-factorized encodings in this repository are examples (`df_encoding.py`, `test_df_encoding.py`), not package APIs.** Record: `cat <this skill's directory>/references/double-factorization/double-factorization-compressed.md` |
| The documented GPU speedup can be quoted as a measurement from this run | unverified | **The `cupy` backend is optional and no record states a measured speedup; regularization trades reconstruction error for smaller cores and one-norm and is not a guaranteed quantum or GPU speedup.** Quote only what this run measured, with the backend actually used. Record: `cat <this skill's directory>/references/double-factorization/double-factorization-compressed.md` |
| `reconstruct_eri`, `factorization_error`, or the one-norm helper certify or select a factorization | false | **They are host NumPy operations with separate input contracts: `reconstruct_eri` rebuilds the dense tensor, `factorization_error` returns the residual against the caller's ERI, `double_factorization_one_norm` takes the factorization plus Fock-like eigenvalues; none chooses, certifies, or emits a kernel.** Declare which ERI chain (target or reconstructed) each downstream number uses. Records: `cat <this skill's directory>/references/double-factorization/double-factorization-reconstruction.md`, `cat <this skill's directory>/references/double-factorization/double-factorization-error.md`, `cat <this skill's directory>/references/double-factorization/double-factorization-one-norm.md` |
| `modified_one_body_integrals` consumes a `DoubleFactorization` | false | **It consumes two dense chemist-basis tensors (one-body and ERI), not the factorization object.** Record: `cat <this skill's directory>/references/double-factorization/double-factorization-modified-one-body.md` |
| Explicit and compressed factorization are interchangeable | false | **X-DF is a deterministic explicit factorization whose first stage is Cholesky or eigendecomposition (`first_factorization`, with a warning on an indefinite tensor); C-DF/RC-DF is an L-BFGS-B optimization with the controls above; both take a square real chemist tensor and apply the same symmetry checks.** Records: `cat <this skill's directory>/references/double-factorization/double-factorization-explicit.md`, `cat <this skill's directory>/references/double-factorization/double-factorization-compressed.md` |
