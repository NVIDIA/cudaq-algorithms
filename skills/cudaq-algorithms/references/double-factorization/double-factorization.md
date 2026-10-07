# Double factorization - family front door

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| factorize / chemist ERI tensor by X-DF | `double_factorization.explicit_double_factorization` | classical transformation; host NumPy/optional CuPy | consumes chemist-integral representation | [explicit DF](double-factorization-explicit.md) |
| compress / chemist ERI tensor by C-DF or RC-DF | `double_factorization.compressed_double_factorization` | classical optimization; host NumPy/optional CuPy | consumes chemist-integral representation | [compressed DF](double-factorization-compressed.md) |
| reconstruct / dense chemist ERI tensor | `double_factorization.reconstruct_eri` | classical transformation; host | consumes `double_factorization.DoubleFactorization` | [DF reconstruction](double-factorization-reconstruction.md) |
| compare / ERI tensor with a factorization | `double_factorization.factorization_error` | classical reduction; host | consumes chemist ERI plus `double_factorization.DoubleFactorization` | [DF residual error](double-factorization-error.md) |
| transform / one-body and ERI tensors to corrected one-body matrix | `double_factorization.modified_one_body_integrals` | classical transformation; host | consumes two dense chemist-basis tensors, not `DoubleFactorization` | [modified one-body integrals](double-factorization-modified-one-body.md) |
| estimate / double-factorized Hamiltonian one-norm | `double_factorization.double_factorization_one_norm` | formula-level estimator; host | consumes `double_factorization.DoubleFactorization` plus Fock-like eigenvalues | [DF one-norm](double-factorization-one-norm.md) |

## Shared representation - `cudaq_algorithms.double_factorization.DoubleFactorization`

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

## Workflow

Start from real chemist `eri`, the physical `h`, a separate scalar offset, and
the electron/spin sector. State the scientific accuracy target (energy, gap,
observable or dynamics) and the cost objective before choosing leaves. Tensor
Frobenius error, leaf count and the one-norm answer different questions; none
alone certifies the requested physics or a hardware speedup.

For a leaf-budget study, establish an untruncated reference, then vary
`max_num_leaves` and explicit-factorization thresholds within a declared search
budget. Record the actual leaf count because thresholding can stop before the
cap. Cholesky assumes a positive semidefinite ERI supermatrix; an intentionally
indefinite synthetic tensor requires the eigendecomposition path. Recheck the
scientific observable for every candidate rather than treating a smaller tensor
residual as an automatic ordering of energy or dynamics errors.

For explicit-versus-compressed comparisons, hold the physical inputs and cost
definition fixed. Compare at equal leaf budget or select the cheapest candidates
that meet the same scientific tolerance. Set `num_leaves`, iteration budget,
optimizer tolerance and backend explicitly for bounded studies; record inner
solver settings and regularization when changed. C-DF optimizes a tensor-fit
objective (plus any core penalty), not the energy error. Inspect optimizer
status and retain warnings; a useful unconverged candidate is not a converged
optimum. Additional starts are warranted by optimization uncertainty, not by an
assumption that compression must beat X-DF. A tie or no feasible candidate is a
valid outcome; a claimed minimum needs a stated search domain and evidence.

Keep two one-body roles distinct. The approximate physical Hamiltonian uses
the original `h`, reconstructed `eri_approx`, and the original scalar. The
number-operator DF representation instead uses
`kappa[p,q] = h[p,q] - 0.5*sum_r eri_approx[p,r,q,r]`. Compute its one-body
eigenvalues for the norm helper, but do not feed `kappa` to
`chemistry.qubit_hamiltonian`: that bridge already implements the ordinary
two-electron Hamiltonian. Recompute this correction for each approximation;
mixing the target correction with approximate factors changes the Hamiltonian.

## Verification

For a small system, reconstruct independently as pair-basis matrices and compare
the physical spectrum in a fixed sector. The checkpoint takes `factorization`,
physical spatial `h`, target `eri`, `offset`, and JW occupation indices `sector`.
`reference_levels` are sorted eigenvalues from an independently assembled target
Hamiltonian in that sector. Supply `check_atol` for reconstruction checks and
`prune_tol` for qubit compilation; choose them below the scientific error budget.
Use `fixed_width_matrix` from the
[fermion checkpoint](../fermion-transforms/fermion-transforms.md#verification).

```python
import numpy as np
from cudaq_algorithms import chemistry, double_factorization as df

n = len(h)
pair_matrix = np.zeros((n*n, n*n))
for U, Z in zip(factorization.leaf_rotations, factorization.leaf_cores):
    np.testing.assert_allclose(U.T @ U, np.eye(n), atol=check_atol, rtol=0)
    np.testing.assert_allclose(Z, Z.T, atol=check_atol, rtol=0)
    B = np.column_stack([np.outer(U[:, k], U[:, k]).reshape(-1)
                         for k in range(n)])
    pair_matrix += B @ Z @ B.T
eri_approx = pair_matrix.reshape(n, n, n, n)
np.testing.assert_allclose(eri_approx, df.reconstruct_eri(factorization),
                           atol=check_atol, rtol=0)
operator = chemistry.qubit_hamiltonian(
    h, eri_approx, scalar_offset=offset, tolerance=prune_tol)
matrix = fixed_width_matrix(operator, 2*n)
levels = np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])
kappa = df.modified_one_body_integrals(h, eri_approx)
one_body_levels = np.linalg.eigvalsh(kappa)
lambda_lcu = df.double_factorization_one_norm(
    factorization, one_body_levels, convention="lcu")
manual_lambda = np.abs(one_body_levels).sum() + sum(
    np.abs(np.triu(Z, 1)).sum() + 0.25*np.abs(np.diag(Z)).sum()
    for Z in factorization.leaf_cores)
np.testing.assert_allclose(lambda_lcu, manual_lambda, atol=check_atol, rtol=0)
metrics = dict(leaves=len(factorization.leaf_cores),
               eri_error=float(np.linalg.norm(eri - eri_approx)),
               ground_error=float(abs(levels[0] - reference_levels[0])),
               lambda_lcu=lambda_lcu)
```

Check the sector and Hamiltonian conventions independently before trusting the
reference; rebuilding both sides with the same chemistry bridge only tests the
DF approximation. Compare gap/observable/dynamics errors too when those are the
target: a small ground-energy shift is insufficient. Account for degeneracy
when tracking individual states. For larger systems use a validated reference
solver or explicit bounds, rather than requiring full Fock-space matrices.

Label the norm convention, scalar treatment, backend and actual runtime. The
helper above omits the scalar; its value is a formula-level proxy, not an
automatically valid normalization for an arbitrary block encoding. A resource
claim needs the cost model or constructed encoding used downstream. Read
DF tests (`tests/python/test_double_factorization.py`) for helper and
optimizer checks and the compression example (`docs/sphinx/examples/python/df_compression_to_qsvt.py`)
for composition. Example-only encoding classes remain examples, not package APIs.

Shared source is `python/cudaq_algorithms/double_factorization/`; tests are
`test_double_factorization.py`, `test_df_encoding.py`, and
`test_df_qsvt_bridge.py`. Current public source and tests are authoritative and
must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.

## Premise check (claim -> verdict)

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
