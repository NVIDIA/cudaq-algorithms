# Double factorization — family record

Status: draft. Operation + object: **factorize/compress** a **fermionic
two-electron integral tensor**, plus the reconstruction, error, one-body
correction, and one-norm operations defined on the factorization it returns.

This file is one family reference covering the six public scientific entry
points of `cudaq_algorithms.double_factorization` and its two backend helpers.
Each canonical Primitive-Record heading is instantiated **once at family
level**, with per-provider material in visibly separate subsections beneath it,
as `assets/primitive-record-template.md` requires. One optional
**Representation Record** for the `DoubleFactorization` object follows the
primitive headings. No Capability Record is created; the reason is stated under
Capabilities and composition.

Everything in this family is **classical host preprocessing**. It emits no
quantum kernel, allocates no register, and consumes no qubit. The one
double-factorized *block encoding* in the repository is an **example file, not
an installed primitive** — see "Installed primitives versus the example-only DF
block encoding" under Capabilities and composition, and do not blur the two.

---

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths: from `cudaq_algorithms.double_factorization`
  (also reachable as `cudaq_algorithms.double_factorization` after
  `import cudaq_algorithms`, which imports the subpackage in its `__init__`):
  `DoubleFactorization`, `explicit_double_factorization`,
  `compressed_double_factorization`, `reconstruct_eri`, `factorization_error`,
  `modified_one_body_integrals`, `double_factorization_one_norm`,
  `cupy_gpu_available`, `resolve_backend`. That list is the subpackage
  `__all__`. Nothing from this family is re-exported at the
  `cudaq_algorithms` package root.
- Source paths: `python/cudaq_algorithms/double_factorization/__init__.py`,
  `_factorization.py`, `_backend.py`.
- Authoritative tests: `tests/python/test_double_factorization.py` (the whole
  suite). Cited for boundary behavior of the object, not of this family:
  `tests/python/test_df_encoding.py` and
  `tests/python/test_df_qsvt_bridge.py`.
- Authoritative documentation: `docs/sphinx/guide/preprocessing.rst`
  ("Classical double factorization" and everything below it);
  `docs/sphinx/conventions.rst` ("Spin orbitals: interleaved", "Fermionic
  integral tensors"); `docs/sphinx/api/python_api.rst` (automodule entry).
  Examples: `docs/sphinx/examples/python/double_factorization.py`,
  `df_compression_to_qsvt.py`, `df_block_encoding.py`,
  `04_double_factorization_and_the_protocol.py`. Benchmark:
  `benchmarks/double_factorization/bench_cg_inner_solve.py`.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16`, `numpy`, `scipy`
  (`pyproject.toml`), CUDA-Q pinned at
  `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`). No test,
  example, or benchmark in this family was executed in this session. Note that
  `cudaq` itself is **not** imported by any module in this family.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft.
- Replacement and migration notes: none; no symbol in this family is deprecated
  or renamed at the cited commit.
- **Evidence rule for this family.** Every test, tolerance, benchmark, and
  performance statement named below is *cited repository evidence* read at that
  commit and labeled `derived`. Nothing here is a fresh measurement, a runtime,
  or a throughput claim. The documented GPU crossovers and the in-source
  accuracy note about batched `eigh` are labeled `unverified`: they are
  recorded in the repository as prior empirical findings, and no recorded
  execution backs them here.

---

## Classification

- Operation + mathematical object (primary identity): **factorize/compress** a
  **two-electron integral tensor** (chemist `(pq|rs)`, real spatial orbitals).
  The derived operations share the family because they are defined on the
  factorization object rather than on a new mathematical object:
  reconstruct/tensor (`reconstruct_eri`), estimate/reconstruction-residual
  (`factorization_error`), transform/one-body-integral-matrix
  (`modified_one_body_integrals`), estimate/Hamiltonian-one-norm
  (`double_factorization_one_norm`).
- Kind: **classical transformation / preprocessing** for all six scientific
  entry points. `double_factorization_one_norm` and `factorization_error` are
  scalar diagnostics computed from data, not `estimate_*_resources`-style
  logical-resource proxies; do not classify them with the state-preparation
  resource estimators. `cupy_gpu_available` and `resolve_backend` are
  auxiliary utilities that happen to be public.
- Routine role: **computational** for the six scientific entry points — each
  solves one distinct, well-defined, independently usable task on the host
  (`architecture.md`: a host classical factorization is computational, not
  auxiliary). **Auxiliary** for the two backend helpers.
- Abstraction level: leaf operation. C-DF *internally* calls X-DF for its warm
  start, which is an implementation detail of one routine, not a composition of
  published capabilities.
- Parameterization: construction-time (every control is an argument of the
  single call; nothing is re-parameterized afterwards).
- Execution layers: host preprocessing only. Heavy linear algebra dispatches to
  NumPy/SciPy or to CuPy (cuSOLVER/cuBLAS); results always return as host NumPy
  arrays. No kernel factory, no device kernel, no observable, no
  simulation-only analysis.
- Input representations: dense chemist-notation `(n, n, n, n)` ERI tensor; an
  `(n, n)` one-body matrix (`modified_one_body_integrals`); a
  `DoubleFactorization`; a one-body eigenvalue vector.
- Output representations: `DoubleFactorization`; dense `(n, n, n, n)` tensor;
  `(n, n)` matrix; Python `float`; `(module, name)` backend pair; `bool`.
- Domain: `quantum-chemistry`. The object it produces is domain-specialized,
  but nothing in the module is chemistry-aware beyond the index symmetry it
  validates.
- Required dependencies: `numpy`, `scipy` (`scipy.linalg` for `logm`, `expm`,
  `expm_frechet`; `scipy.optimize.minimize` for L-BFGS-B — C-DF only).
- Optional dependencies: `cupy` with a usable GPU. CuPy is imported in a
  `try/except` and is **not** declared in `pyproject.toml`; its absence forces
  the NumPy path silently. PySCF and OpenFermion appear only in examples,
  benchmarks, and tests, never in the module.

Provisional metadata — record the value, do not route on it:

- Exactness: X-DF is **exact at full rank** and approximate under any
  truncation control; C-DF is **approximate by construction** (a fixed leaf
  budget) and additionally biased when `regularization > 0`.
  `reconstruct_eri`, `factorization_error`, and `modified_one_body_integrals`
  are exact up to floating-point round-off given their inputs.
  `double_factorization_one_norm` is an exact evaluation of a stated formula,
  not a bound on any measured quantity.
- Uncertainty: deterministic. No sampling, no random initialization: C-DF's
  starting point is the deterministic X-DF warm start unless
  `initial_generators` is supplied.
- Method: X-DF is `direct`; C-DF is a deterministic local `variational`
  optimization (L-BFGS-B over rotation generators with a closed-form inner
  core solve), which is a numerical-optimization sense of the word and **not**
  a NISQ variational workflow — no quantum expectation value is involved.

---

## Scientific contract

### Purpose

Rewrite the two-electron integral tensor as a short sum of terms that are each
diagonalizable in a single rotated orbital frame, so that downstream consumers
can either (a) work with a compressed reconstructed tensor, or (b) exploit the
frame structure directly. The family also supplies the scalars needed to judge
a factorization: the reconstruction residual and the Hamiltonian one-norm.

### Mathematical definition

With `eri[p, q, r, s] == (pq|rs)` over `n` real spatial orbitals, double
factorization writes

```text
(pq|rs)  ~=  sum_t sum_{k,l}  U^t_pk U^t_qk  Z^t_kl  U^t_rl U^t_sl
```

with orthogonal **leaf rotations** `U^t` and symmetric **core** matrices `Z^t`
(`_factorization.py` module docstring; `preprocessing.rst`). `t` indexes
leaves; `k`, `l` index rotated spatial orbitals of that leaf.

Per provider:

- **X-DF (`explicit_double_factorization`).** Two nested decompositions.
  *First factorization* of the symmetrized supermatrix
  `M[(pq), (rs)] = (pq|rs)` (built as `eri.reshape(n*n, n*n)` then
  `0.5 * (M + M.T)`):
  - `first_factorization="cholesky"` (default): rank-revealing pivoted
    Cholesky, `(pq|rs) = sum_t L^t_pq L^t_rs`, pivoting on the largest
    residual diagonal entry;
  - `first_factorization="eigendecomposition"`: symmetric eigendecomposition
    `(pq|rs) = sum_t lambda_t V^t_pq V^t_rs`, leaves ordered by descending
    `|lambda_t|`.

  *Second factorization*: each leaf matrix is symmetrized and eigendecomposed,
  `L^t = U^t diag(gamma^t) (U^t)^T`, giving the **rank-one** core
  `Z^t = scale * outer(gamma^t, gamma^t)` with `scale = 1.0` on the Cholesky
  path and `scale = lambda_t` on the eigendecomposition path.
- **C-DF (`compressed_double_factorization`).** For a fixed leaf count `T`,
  minimize

  ```text
  O(X, Z) = 1/2 || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F^2
            + rho * sum_{t,k,l} (Z^t_kl)^2
  ```

  over antisymmetric generators `X^t` with `U^t = exp(X^t)` and symmetric cores
  `Z^t`. The cores are solved exactly (linear least squares) at every step and
  the generators are driven by L-BFGS-B; by the envelope theorem the gradient
  is `dO/dU^t_ak = -4 sum_{qrsl} Delta_aqrs U^t_qk Z^t_kl U^t_rl U^t_sl`
  (Eq. 17 of arXiv:2104.08957, cited in source), pulled back through the
  Fréchet derivative of `exp`. Cores are general symmetric, not rank one.
  `rho > 0` is RC-DF (arXiv:2212.07957 Eq. 17) and enters both the objective
  and the inner solve as a ridge.
- **`reconstruct_eri`.** Evaluates the sum above exactly, in float64, on the
  host.
- **`factorization_error`.** `|| eri - reconstruction ||_F` — the **absolute**
  Frobenius norm of the residual, not a relative error and not an energy error.
- **`modified_one_body_integrals`.** `kappa_pq = h_pq - 1/2 sum_r (pr|qr)`
  (Eq. 3 of arXiv:2104.08957, cited in source): the one-body matrix corrected
  for the exchange term produced by normal-ordering the two-body operator.
- **`double_factorization_one_norm`.** With `F` the supplied one-body
  (Fock-like) eigenvalues:
  - `convention="lcu"` (RC-DF Eq. 13):
    `lambda = sum_k |F_k| + sum_t ( sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk| )`;
  - `convention="burg"`: each core is eigendecomposed
    `Z^t = sum_i lambda^t_i v^t_i (v^t_i)^T` and
    `lambda = sum_k |F_k| + 1/4 sum_t sum_i |lambda^t_i| (sum_k |v^t_ki|)^2`.
    The source states the gauge argument explicitly: `Z = W W^T` fixes `W` only
    up to a right orthogonal gauge and the column-norm formula is not gauge
    invariant, so the eigenfactor is the gauge-fixed choice; it reduces to
    RC-DF Eq. 15 / von Burg's `(1/4)(sum_k |gamma_k|)^2` for rank-one cores.

### Why and when to use

- Use **X-DF** when an exact or threshold-controlled factorization is wanted
  cheaply, when the rank itself is the question, or as the reference against
  which a compression is judged. It is also the warm start C-DF uses.
- Use **C-DF** when the leaf count is the binding constraint and the extra
  L-BFGS cost is acceptable: at equal leaf count it is never worse than
  truncated X-DF in the repository's tests, and the guide states it reaches a
  target accuracy with substantially fewer leaves.
- Use **RC-DF** (`regularization > 0`) when the Hamiltonian one-norm matters
  more than the last digit of tensor accuracy — the ridge shrinks the cores and
  lowers `lambda`, and it also conditions an otherwise rank-deficient inner
  system.
- Use `reconstruct_eri` when a downstream consumer wants a *tensor*: the output
  is in the same chemist convention `chemistry.qubit_hamiltonian` and
  `chemistry.spin_orbital_tensors` consume.
- Use `double_factorization_one_norm` to compare factorizations of the same
  system, having first decided which convention the downstream algorithm
  actually uses.

### When not to use

- Do not use this family to obtain a quantum circuit, a block encoding, a
  qubit Hamiltonian, or any register-level object. It produces none of those.
- Do not use it on complex, spin-resolved, or non-8-fold-symmetric integrals:
  the validator rejects the asymmetric case, and the complex case is unverified
  (see Accuracy and limitations).
- Do not use `factorization_error` as a proxy for energy error, spectral error,
  or `alpha`; nothing in this family bounds those.
- Do not use `double_factorization_one_norm` as a resource estimate for a
  packaged primitive. `PauliLCU.alpha` is a different quantity computed from a
  Pauli expansion, and the DF-encoding `alpha` that *does* match this one-norm
  belongs to an example file (see below).

### Approximation controls

| Control | Provider | What it acts on |
| --- | --- | --- |
| `threshold` (default `1e-8`) | X-DF | Cholesky path: floor on the residual **pivot** (a diagonal entry of the residual supermatrix). Eigendecomposition path: floor on `\|lambda_t\|`. These are different quantities; a value tuned on one path does not transfer to the other. In both paths the effective floor is `max(threshold, scale * 1e-14)` with `scale` the largest initial pivot / largest `\|lambda\|`, so `threshold=0.0` stops at the numerical rank instead of emitting null leaves. |
| `max_num_leaves` (default `None`) | X-DF | Hard cap on the leaf count, applied in both paths. |
| `second_factor_threshold` (default `0.0`, inactive) | X-DF | Zeros individual `gamma^t_k` whose *importance-weighted* magnitude `\|scale\| * (sum_k \|gamma^t_k\|) * \|gamma^t_k\|` does not exceed the threshold. The docstring states this matches OpenFermion's convention, and that `\|lambda_t\|` is included on the eigendecomposition path so both paths compare the same absolute quantity (on the Cholesky path the pivot scale already lives in the leaf vector's norm). |
| `first_factorization` | X-DF | `"cholesky"` (default) or `"eigendecomposition"`. Not an accuracy knob at full rank, but it is the only correct choice for an indefinite supermatrix. |
| `num_leaves` (required) | C-DF | The leaf budget being optimized. This is the compression dial. |
| `regularization` (`rho`, default `0.0`) | C-DF | RC-DF ridge, an **absolute** coefficient whose useful scale depends on the integral magnitude; the guide cites `~1e-6` to `1e-3` from the paper. Trades reconstruction accuracy for smaller cores and a smaller `lambda`. |
| `max_iterations` (2000), `tolerance` (`1e-10`) | C-DF | L-BFGS-B `maxiter`, and the value passed to **both** `ftol` and `gtol`. |
| `inner_solver`, `cg_tolerance` (`1e-10`), `cg_max_iterations`, `cg_warm_start` (`True`), `cg_optimization_tolerance` (default `max(cg_tolerance, 1e-6)`) | C-DF | Inner core solve. Documented as accuracy-neutral for the returned cores because the single final solve is always tightened to `cg_tolerance`; the in-loop solves are deliberately inexact. |
| `initial_generators` | C-DF | Replaces the X-DF warm start. Changes which local minimum is reached; the objective is not convex in the rotations. |

No control in this family is documented as delivering a bound on the resulting
Frobenius error, on an eigenvalue shift, or on a downstream `alpha`. Truncation
error is monotone in the leaf count for nested X-DF truncation (asserted in
tests); `alpha` of the example encoding is explicitly **not** monotone in the
leaf count (`df_block_encoding.py`).

---

## Inputs

### Shared: the ERI tensor and its validation

- Arguments: `eri` as `ArrayLike`, accepted by `explicit_double_factorization`,
  `compressed_double_factorization`, `factorization_error`, and
  `modified_one_body_integrals`.
- Shapes/ranks: rank 4, all four axes equal: `(n, n, n, n)`. `_validate_eri`
  rejects anything else with
  `ValueError("eri must be a square rank-4 tensor (n, n, n, n) in chemist
  notation (pq|rs).")`.
- Dtypes/domains: converted with `np.asarray(eri, dtype=float)` after
  validation, i.e. **float64 throughout**. There is no float32 path and no
  complex path.
- Units: whatever the caller's integrals carry (Hartree in every repository
  example). Nothing is rescaled.
- Ordering/layout: chemist notation `eri[p, q, r, s] == (pq|rs)` over **real
  spatial** orbitals — for example `pyscf.ao2mo.restore("s1", ...)`. Indices
  are spatial; **no spin index appears anywhere in this family.** Spin
  expansion to the interleaved `2p` alpha / `2p+1` beta layout happens later,
  in `chemistry.spin_orbital_tensors` or in the example encoding
  (`conventions.md`, "Spin orbitals").
- Normalization: none applied. No `1/2` factor is folded in or out here; the
  chemist-to-`adag adag a a` reordering and the `1/2` belong to
  `chemistry.spin_orbital_tensors` (`conventions.rst`, "Fermionic integral
  tensors").
- Required mathematical properties: the three generators of the real-orbital
  8-fold symmetry, `(pq|rs) == (qp|rs) == (pq|sr) == (rs|pq)`. Positive
  semidefiniteness of the supermatrix is *assumed by the default Cholesky
  path* but not required by the validator — an indefinite input warns rather
  than raising.
- Validation and rejection behavior: `_validate_eri` runs first in both
  factorizers and checks the three transposes with `np.allclose` at **NumPy's
  default tolerances** (`rtol=1e-5`, `atol=1e-8`), raising
  `ValueError("eri must have the real-orbital chemist symmetries (pq|rs) ==
  (qp|rs) == (pq|sr) == (rs|pq).")`. The source states why pair swap is
  included even though X-DF symmetrizes the supermatrix: the C-DF gradient
  folds four residual terms into a single `-4` prefactor, which assumes the
  full symmetry on the *unsymmetrized* tensor, so without the check C-DF would
  silently optimize the wrong objective. Note the looser default `rtol` here
  versus the explicit `atol=1e-8` used by `chemistry.spin_orbital_tensors`.

### `explicit_double_factorization`

```python
explicit_double_factorization(eri, threshold=1.0e-8, max_num_leaves=None,
                              second_factor_threshold=0.0,
                              first_factorization="cholesky",
                              backend="auto") -> DoubleFactorization
```

`first_factorization` outside `{"cholesky", "eigendecomposition"}` raises
`ValueError`, but the check is the `else` branch of the dispatch, so it fires
only after ERI validation, backend resolution, the device transfer, and the
supermatrix symmetrization have already run. `threshold`, `max_num_leaves`, and
`second_factor_threshold` are not range-checked: a negative `threshold` is
absorbed by the `max(threshold, scale * 1e-14)` floor, and `max_num_leaves=0`
yields zero leaves rather than an error (`unverified` — no test covers either).

### `compressed_double_factorization`

```python
compressed_double_factorization(eri, num_leaves, max_iterations=2000,
                                tolerance=1.0e-10, regularization=0.0,
                                inner_solver="lstsq", cg_tolerance=1.0e-10,
                                cg_max_iterations=None, cg_warm_start=True,
                                cg_optimization_tolerance=None,
                                initial_generators=None,
                                backend="auto") -> DoubleFactorization
```

`num_leaves < 1` raises `ValueError("num_leaves must be >= 1.")`;
`inner_solver` outside `{"lstsq", "cg"}` raises `ValueError("inner_solver must
be 'lstsq' or 'cg'.")` — checked eagerly in the factorizer *and* again in the
inner dispatch. `initial_generators` must be a sequence of `n x n`
antisymmetric arrays of length `num_leaves`; only the strict lower triangle is
read (`_skew_to_vector` uses `np.tril_indices(n, k=-1)`), so a non-antisymmetric
argument is silently reinterpreted as its lower-triangular part — undocumented
and untested (`unverified`). `regularization`, `tolerance`, and the `cg_*`
values are not range-checked.

### `reconstruct_eri`, `factorization_error`

`reconstruct_eri(factorization)` takes a `DoubleFactorization` and delegates to
its method. `factorization_error(eri, factorization)` takes both. Neither
validates that the factorization's `num_orbitals` matches the tensor: a
mismatch surfaces as a NumPy broadcasting error from the subtraction, not as a
library `ValueError` (`derived` from the absence of any check; the exact
exception is `unverified`).

### `modified_one_body_integrals(one_body, eri)`

Both arguments are converted with `np.asarray(..., dtype=float)`. **No shape
validation and no symmetry validation are performed** — this function does not
call `_validate_eri`. `one_body` is expected `(n, n)` and `eri` `(n, n, n, n)`;
a mismatch surfaces from `np.einsum` or the subtraction. The result is
symmetric whenever `one_body` is symmetric and `eri` carries the chemist
symmetry (asserted for one case in tests).

### `double_factorization_one_norm(factorization, one_body_eigenvalues, convention="lcu")`

`one_body_eigenvalues` is a 1-D array of the **eigenvalues of the DF-corrected
one-body matrix** — the "Fock-like" diagonal, not the raw `h_pq` matrix and not
its diagonal entries. Nothing verifies that the caller actually supplied
eigenvalues, that they were computed from the corrected `kappa`, or that their
length equals `factorization.num_orbitals`. A wrong choice here silently shifts
`lambda` by the difference of the two `sum |F_k|` values. `convention` outside
`{"lcu", "burg"}` raises `ValueError("convention must be 'lcu' or 'burg'.")`.

### Backend selection (all factorizers)

`backend` is `"auto"` (default), `"cupy"`, or `"numpy"`; anything else raises
`ValueError(f"unknown backend '{backend}'; expected 'auto', 'cupy', or
'numpy'.")`. `"cupy"` is a hard request and raises `RuntimeError` when
CuPy/GPU is unusable. `"auto"` evaluates the **size check first** so a
below-threshold problem never creates a CUDA context, then runs the kernel
probe. Thresholds are per method: `AUTO_GPU_MIN_ORBITALS_EXPLICIT = 56`,
`AUTO_GPU_MIN_ORBITALS_COMPRESSED = 18` (orbital count `n`). The derived
helpers (`reconstruct_eri`, `factorization_error`,
`modified_one_body_integrals`, `double_factorization_one_norm`) take **no**
`backend` argument and always run on the host in NumPy.

---

## Outputs

### `DoubleFactorization`

A plain (non-frozen, non-validating) dataclass. Full field semantics are in the
Representation Record below. Summary: `num_orbitals: int`,
`leaf_rotations: List[np.ndarray]` of `(n, n)`, `leaf_cores: List[np.ndarray]`
of `(n, n)`, `method: str`, `first_factorization: Optional[str]`,
`leaf_weights: Optional[np.ndarray]`, `optimizer_success: Optional[bool]`,
`optimizer_nit: Optional[int]`, `optimizer_grad_norm: Optional[float]`, the
property `num_leaves`, and the method `reconstruct_eri()`.

- Mathematical meaning: the `{U^t, Z^t}` pair set of the definition above.
- Shape/register geometry: no register. Shapes are `(n, n)` per leaf; the leaf
  count is `len(leaf_rotations)`, which can be **0** for a negative-definite
  input on the Cholesky path.
- Normalization, sign, and phase: `U^t` is orthogonal to machine precision
  (tests assert `|| U^T U - I ||_F` below `1e-10`/`1e-9`). Column signs are a
  free gauge — the reconstruction depends on `U` only through `u_k u_k^T`, so
  flipping a column changes nothing, a fact the source relies on twice
  (determinant repair in `_initial_generators`, sign-agnostic Givens sweep in
  the example encoding). **Determinant differs by provider**: X-DF rotations
  come from `eigh` and may be reflections (`det = -1`); C-DF rotations are
  `exp(antisymmetric)` and are therefore always special orthogonal
  (`det = +1`). Cores are symmetric: exactly so on the X-DF and lstsq paths by
  construction, and explicitly re-symmetrized as `0.5 * (Z + Z^T)` at the end
  of the CG solve.
- Arrays are **always host NumPy**, whichever backend ran: `_second_factorization`
  and the C-DF return path apply `to_numpy` to every rotation and core.
- Error/status information: `method` is `"X-DF"` or `"C-DF"`.
  `first_factorization` and `leaf_weights` are X-DF-only (`None` on C-DF); the
  three `optimizer_*` fields are C-DF-only (`None` on X-DF, asserted in tests).
  `leaf_weights` carries **different quantities per path**: positive Cholesky
  pivots on the default path, signed supermatrix eigenvalues on the
  eigendecomposition path. On the Cholesky path the weight is *not* applied to
  the core (it is already carried by the leaf vector's norm); on the
  eigendecomposition path the core *is* scaled by `lambda_t`. Cores are
  therefore directly comparable across paths, weights are not.

### The other entry points

| Symbol | Return | Meaning |
| --- | --- | --- |
| `reconstruct_eri` / `DoubleFactorization.reconstruct_eri` | `np.ndarray` `(n, n, n, n)`, float64 | The reconstructed chemist tensor. Satisfies the full 8-fold real-orbital symmetry by construction whenever every core is symmetric (`p<->q` and `r<->s` from the repeated `U` columns; `(pq)<->(rs)` from `Z = Z^T`), so it is accepted by `chemistry.spin_orbital_tensors`'s validator. A hand-built non-symmetric core breaks the pair-swap symmetry. |
| `factorization_error` | `float` | Absolute Frobenius residual norm. Examples divide by `\|\|eri\|\|_F` themselves to get a relative figure; the function does not. |
| `modified_one_body_integrals` | `np.ndarray` `(n, n)`, float64 | `kappa`, the exchange-corrected one-body matrix. |
| `double_factorization_one_norm` | `float` | `lambda` in the requested convention. Depends only on the **cores** and the supplied eigenvalues — the leaf rotations are never read — so it is invariant under any re-gauging of `U^t`. |
| `cupy_gpu_available` | `bool` | True only if `getDeviceCount() > 0` **and** a tiny compiled kernel runs and synchronizes; a driver-only install returns False. Exceptions are swallowed. |
| `resolve_backend` | `tuple[module, str]` | The array module and its name, `"numpy"` or `"cupy"`. |

---

## Capabilities and composition

### Provided

None. `DoubleFactorization` is a populated Representation Record below, but no
stable capability ID is minted for it.

### Required

None. This family requires no capability from any other CUDA-Q Algorithms
primitive. Its only external requirements are NumPy/SciPy, and optionally CuPy
plus a working GPU.

### Why no Capability Record is created here

`architecture.md` gates capability extraction on *multiple independent
producers or consumers demonstrating a reusable boundary*. There are two
producers (X-DF, C-DF), but every installed consumer of `DoubleFactorization`
lives inside this same module (`reconstruct_eri`, `factorization_error`,
`double_factorization_one_norm`), and the one structurally independent consumer
is an example file. That is a one-off interface at the cited commit, so it
stays inside this record as the Representation Record. Re-examine if a second
installed package module ever consumes a `DoubleFactorization`.

### Source-grounded composition paths

1. **Compression into the packaged quantum stack, via a tensor.** This is the
   only fully packaged path:

   ```text
   chemistry.from_pyscf / from_psi4 / from_fcidump
     -> eri
     -> explicit_/compressed_double_factorization
     -> reconstruct_eri
     -> chemistry.qubit_hamiltonian   (spin expansion + Jordan-Wigner)
     -> PauliLCU -> Walk / QSVT / Trotter
   ```

   Documented in `preprocessing.rst` ("Feeding a compressed tensor to the
   transform") and `chemistry.qubit_hamiltonian`'s docstring, and exercised by
   `tests/python/test_df_qsvt_bridge.py` and
   `docs/sphinx/examples/python/df_compression_to_qsvt.py`. The compression's
   payoff appears downstream as a smaller LCU `alpha` and fewer Pauli terms, at
   the price of a spectrum shift controlled by the tensor reconstruction error.
   Note the round trip **discards the frame structure**: `PauliLCU` sees only a
   tensor, so nothing of the low-rank factorization survives into the circuit.
2. **Diagnostics.** `factorization_error` and `double_factorization_one_norm`
   compose with either factorizer; the guide describes picking `rho` by trading
   one against the other.
3. **One-body correction.** `modified_one_body_integrals` is offered "when
   assembling the full double-factorized Hamiltonian from the two-body
   factorization". No installed module calls it, and — importantly — the
   example encoding does **not** call it either (see below).

### Installed primitives versus the example-only DF block encoding

| Artifact | Status | Evidence |
| --- | --- | --- |
| `cudaq_algorithms.double_factorization` (the nine symbols above) | **Installed package API.** Importable from the wheel; documented in the Sphinx API reference | `pyproject.toml` packages `cudaq_algorithms*` under `python/`; `python_api.rst` automodule entry |
| `DoubleFactorizedEncoding` and the kernels in `docs/sphinx/examples/python/df_encoding.py` | **Example file, not an installed primitive.** Not in the package tree, not importable from `cudaq_algorithms`, no public-API stability | The package `__init__.py` says it "ships as a runnable example at `docs/sphinx/examples/python/df_encoding.py`"; tests and sibling examples reach it by inserting that directory on `sys.path`; `tests/python/test_examples.py` excludes it from standalone example runs as "not standalone" |

What the example encoding is, stated so it is not mistaken for a contract of
this family: it is a user-level implementation of the structural
`BlockEncoding` protocol that consumes a `DoubleFactorization` **directly**
(keeping the frame structure that path 1 discards), encoding the von Burg form
of the Hamiltonian (von Burg et al., PRX Quantum 2, 030305 (2021),
arXiv:2007.14460). Facts worth carrying, all `derived` from the example and its
test suite, none of them a promise of the installed API:

- It builds its own corrected one-body matrix, contracting the **factorized**
  tensor and additionally absorbing the `sum_l Z^t_kl` singles produced by
  centering the leaf number operators. It does **not** call
  `modified_one_body_integrals`, and the two expressions are not the same
  quantity — the module helper carries only the exchange correction.
- Its `alpha` minus the identity weight equals
  `double_factorization_one_norm(..., convention="lcu")` to `rel=1e-12` in
  `tests/python/test_df_encoding.py::test_alpha_matches_published_lcu_one_norm`.
  That is the sharpest statement available that the `"lcu"` convention means
  what its docstring says — and it is evidence about the example, cited at the
  commit, not a measurement.
- It maps leaf index `k` to interleaved spin qubits `2k`, `2k+1`; the spin
  layout enters there, never in this family.
- It does not provide `select_observable` (raises `NotImplementedError`), so
  odd Chebyshev moments via the observable trick are unavailable for it; even
  moments and the walk/QSVT surface are unaffected.
- `docs/sphinx/examples/python/df_block_encoding.py` records that `alpha` is
  **not** monotone in the leaf count (dropping a leaf reshapes the absorbed
  one-body singles), while the tensor error is monotone under nested
  pivoted-Cholesky truncation. Do not assume "fewer leaves, smaller `alpha`".

If a user asks for a *packaged* double-factorized block encoding, the honest
answer is that none exists at this commit: the installed encoding is
`PauliLCU`, and the DF encoding is an example to copy.

---

## Composite protocol

Deferred: not applicable — every record in this family is a leaf operation, and
`architecture.md` requires this section only for a `composite protocol`.
Follow-up trigger: if a packaged driver ever sequences factorization,
reconstruction, and encoding into one call, that driver gets this section.

---

## Accuracy and limitations

### Error behavior

- **X-DF at full rank is exact to round-off.** The repository asserts
  `factorization_error < 1e-9` on H2O/STO-3G at `threshold=0.0` for both first
  factorizations, and both paths agree on the reconstructed tensor to
  `atol=1e-9`.
- **X-DF truncation error is monotone** in the retained leaf count under nested
  Cholesky truncation (asserted with a `1e-12` slack), and `max_num_leaves`
  caps the count exactly.
- `threshold` bounds a *pivot* or an *eigenvalue* of the supermatrix, not the
  resulting Frobenius error of the reconstruction. Do not present it as an
  error bound.
- **C-DF is a local optimization.** At the true rank of a synthetic tensor it
  reaches `factorization_error < 1e-4` in tests and reports
  `optimizer_success` with `optimizer_grad_norm < 1e-8`; at reduced leaf counts
  it is asserted only to be no worse than truncated X-DF within `1e-6`. There
  is no global-optimality claim anywhere in source, tests, or docs.
- **RC-DF trades accuracy for one-norm.** Tests assert that `rho = 1e-2`
  shrinks the summed core norms, lowers `lambda` in *both* conventions, and
  does not improve the reconstruction error. The example
  `df_block_encoding.py` adds the practical warning that unregularized C-DF can
  exploit gauge freedom to fit better with enormous core entries, blowing up
  `alpha` by orders of magnitude — the pathology the ridge exists to prevent.
- **Both inner solvers reach the same reconstruction.** Tests assert the
  CG and lstsq reconstruction errors agree within `1e-6` with and without
  regularization; the guide adds that when the inner system is full rank (e.g.
  `rho > 0`) the cores themselves agree to machine precision. The CG
  accelerators (warm start, inexact in-loop tolerance) are asserted
  accuracy-neutral within `1e-4` against the cold, tight configuration.

### Precision sensitivity

Float64 only, everywhere. The relative rank floor `scale * 1e-14` exists
specifically because dividing by `sqrt(pivot)` at a denormal pivot yields
garbage rotations. `_initial_generators` carries a second numerical guard:
`scipy.linalg.logm` sits on a branch cut for rotations with a `(-1, -1)`
eigenvalue pair, where taking `.real` silently gives `exp(skew) != rotation`;
the code checks with `atol=1e-8` and falls back to a zero generator, which
degrades the warm start rather than corrupting it.

### Unsupported inputs

| Input | Behavior | Label |
| --- | --- | --- |
| Non-rank-4 or non-square tensor | `ValueError` from `_validate_eri` | derived (tested) |
| Tensor violating any of the three chemist symmetry generators | `ValueError` matching `"chemist symmetries"` | derived (two tests: a `(pq\|rs) == (qp\|rs)` break and a pair-swap-only break) |
| `num_leaves < 1` | `ValueError` | derived (tested) |
| `inner_solver` / `first_factorization` / `convention` / `backend` outside their enumerations | `ValueError` | derived (`inner_solver`, `convention`, and unusable `backend="cupy"` are tested; the `first_factorization` and unknown-`backend` branches are read from source, untested) |
| `backend="cupy"` with no usable GPU | `RuntimeError("the 'cupy' backend was requested but CuPy/GPU is not usable (not installed, no device, or the kernel probe failed).")` | derived (tested, including the "enumerates but dies on the first kernel" case) |
| Indefinite supermatrix on the Cholesky path | `RuntimeWarning` "not positive semidefinite", the negative part is **dropped**, and the reconstruction error can far exceed `threshold`; the message directs the caller to `first_factorization="eigendecomposition"` | derived (tested: error `> 1e-3` on Cholesky versus `< 1e-10` on the eigendecomposition path for the same input) |
| Negative-definite supermatrix | Same warning, **zero leaves returned**, `factorization_error > 1` | derived (tested) |
| C-DF hitting `max_iterations` or otherwise failing | `RuntimeWarning` "did not converge" with status/nit/nfev/`\|\|g\|\|`; the best factorization found is still returned with `optimizer_success is False` | derived (tested at `max_iterations=1`) |
| Complex-valued `eri` | Not validated for dtype; the tensor is later forced through `np.asarray(eri, dtype=float)`. No test covers it | **unverified** — do not state what happens |
| Non-antisymmetric `initial_generators` | Only the strict lower triangle is read; no check | unverified |
| `num_orbitals` mismatch between `eri` and a `DoubleFactorization` in `factorization_error` | No library check; NumPy raises | unverified (exact exception) |
| Wrong-length or wrong-origin `one_body_eigenvalues` | No check; `lambda` silently shifts | unverified |

### Known implementation limitations

- `explicit_double_factorization` loops leaves in Python and calls
  `to_numpy` per leaf inside `_second_factorization`, so even on the CuPy
  backend the second factorization round-trips each leaf to the host.
- C-DF keeps `scipy.linalg.expm_frechet` and `scipy.optimize.minimize` on the
  host, so each objective evaluation includes a device-to-host transfer of the
  gradient (the source calls the Fréchet step "~2% of the step cost" — an
  in-source claim, `unverified` here).
- The CG warm-start cache is a 3-slot LRU keyed on the parameter vector's
  bytes. This exists because SciPy requires `(f, g)` to be a function of `x`
  alone; a regression here is exactly what
  `test_cdf_objective_is_a_function_of_x` guards, with a `1e-12` / `1e-9`
  identity on a rank-deficient H2O case.
- The dataclass performs no validation, so a hand-built `DoubleFactorization`
  with inconsistent shapes, non-orthogonal rotations, or non-symmetric cores is
  accepted by every consumer in this family. Tests deliberately build one.
- `double_factorization_one_norm` reads only the **strict upper triangle**
  (`np.triu(core, k=1)`) in the `"lcu"` convention, while `np.linalg.eigh` in
  the `"burg"` convention reads the **lower** triangle by default. For the
  symmetric cores the packaged factorizers produce this is immaterial; for a
  hand-built non-symmetric core the two conventions would read different data.
  Undocumented; `unverified`.

### Unsupported versus unverified

Documented as unsupported: non-square/non-rank-4 tensors, symmetry-violating
tensors, `num_leaves < 1`, out-of-enumeration string arguments, and
`backend="cupy"` without a usable GPU. Documented as *degraded but permitted*:
indefinite input on the Cholesky path (warning, not rejection) and
non-converged C-DF (warning, result still returned). Merely undocumented, hence
unknown: complex integrals, non-antisymmetric `initial_generators`, negative or
zero `max_num_leaves`, shape mismatches in the helper functions, non-symmetric
hand-built cores, and every behavior of `modified_one_body_integrals` outside
matched `(n, n)` / `(n, n, n, n)` inputs.

---

## Resources

All quantities are **classical host resources**. This family emits no circuit,
so it has no qubit count, gate count, depth, or T-count of its own, and nothing
here may be compared with a logical-operation proxy from another family
(`conventions.md`, "Resource abstraction levels must not be conflated").

| Quantity | Metric and unit | Abstraction level | Assumptions | Status | Controlling parameters |
| --- | --- | --- | --- | --- | --- |
| X-DF supermatrix | `n^4` float64 entries (`n^2 x n^2`) | host memory | dense storage | exact, derived from source | `n` |
| X-DF leaf count | `<= n(n+1)/2` for a PSD supermatrix, `<= n^2` in general, `<= max_num_leaves` | count | pivoted Cholesky is rank-revealing on a PSD Gram matrix | bounded; the `n(n+1)/2` bound is asserted in tests for the molecular case | `threshold`, `max_num_leaves` |
| X-DF second factorization | one `n x n` symmetric `eigh` per leaf | host/device linear algebra | — | exact, derived | leaf count |
| C-DF parameters | `num_leaves * n(n-1)/2` real generator parameters | count | L-BFGS-B state scales with this | exact, derived | `num_leaves`, `n` |
| C-DF `lstsq` design matrix | `n^4` rows (plus `num_params` penalty rows when `rho > 0`) by `num_leaves * n(n+1)/2` columns, float64 | host/device memory | dense storage; the benchmark computes `n^4 * num_leaves * n(n+1)/2 * 8 / 1e9` GB and skips this solver above 8 GB | exact formula, derived from source; the "H2O/6-311G (`n = 19`) design matrix is ~4 GB while the ERI is ~1 MB" figure is a benchmark docstring claim, **unverified** here | `n`, `num_leaves`, `rho` |
| C-DF CG metric tensor | `num_leaves^2 * n^2` float64 entries | host/device memory | the `(t, u, k, l)` stacked metric | exact, derived | `num_leaves`, `n` |
| C-DF CG iterations | `<= max(50, num_leaves * n * n)`, terminating on `\|\|r\|\|^2 <= tol^2 * max(\|\|r_0\|\|^2, 1)` or non-positive curvature | count | matrix-free operator | bounded, derived; the guide's "cost is iteration-bound, not matvec-bound" is a documented empirical characterization, **unverified** here | `cg_tolerance`, `cg_optimization_tolerance`, `cg_max_iterations`, `rho`, `cg_warm_start` |
| Reconstruction | one `n^4` output tensor; one `n^4`-scale `einsum` per leaf, looped in Python on the host | host memory and arithmetic | float64 | exact, derived | `num_leaves`, `n` |
| GPU crossover | `n >= 18` (C-DF), `n >= 56` (X-DF) for `backend="auto"` | orbital count | "empirical CPU/GPU crossovers" per the source comment, measured on hardware not identified in the repository | **unverified** — recorded prior finding, no execution here | `backend`, `problem_size` |

Composition rule: none is documented. In particular, no source states how a
factorization's leaf count maps to a downstream circuit cost. The example
encoding exposes `num_terms`, `num_frames == 1 + num_leaves`,
`num_givens_rotations`, and `num_ancilla`, but those are properties of the
example, at the example's own abstraction level.

---

## Validation

### Independent oracles available in the repository

| Oracle | Hierarchy level (`validation.md`) | What it pins | Where |
| --- | --- | --- | --- |
| OpenFermion `resource_estimates.df.factorize` | 3 — trusted external implementation after convention alignment | X-DF's reconstructed H2O/STO-3G tensor to `atol=1e-9`; the reference's own rank `<= n(n+1)/2`. Compared on the **reconstructed tensor**, deliberately, because the two implementations store leaves differently and truncate the second factor differently | `test_explicit_double_factorization_matches_openfermion_reference` |
| Hand-computed one-norms | 1 — exact analytical identity | Both `"lcu"` and `"burg"` formulas on a `2 x 2` core built from chosen eigenpairs (`V(pi/6) diag(2.0, -0.5) V^T`), to `abs=1e-12` | `test_one_norms_match_hand_computation` |
| Independent triple loop | 2 — independently constructed reference | `modified_one_body_integrals` against an explicit `p, q, r` loop, `atol=1e-14`, plus the symmetry of the result | `test_modified_one_body_matches_independent_loop` |
| Synthetic ERI of known rank | 2 — constructed ground truth | `_synthetic_eri(n, num_vectors, seed)` builds `sum_x L^x_pq L^x_rs` from symmetrized random leaves, so the exact leaf rank is known: the eigendecomposition path at `threshold=0.0` recovers exactly 3 leaves for a rank-3 tensor | `test_eigendecomposition_threshold_zero_stops_at_numerical_rank` |
| Invariants | 4 | Orthogonality of every `U^t` (`< 1e-10` X-DF, `< 1e-9` C-DF); symmetry of every core (`atol=1e-12`); monotone truncation error; `lambda >= sum \|F_k\|`; helper/method agreement of `reconstruct_eri` | several tests |
| Cross-solver agreement | 5 — mathematically distinct algorithms | Cholesky versus eigendecomposition first factorization (`atol=1e-9`); CG versus lstsq inner solve (`1e-6`); accelerated versus cold/tight CG (`1e-4`) | three tests |
| Dense fermionic reference | 2 — for the **example** encoding only | `<0\|U\|0> == H/alpha` against a dense Jordan-Wigner Hamiltonian at `atol=1e-12`, including the truncated-factorization case, and the `alpha`-versus-one-norm identity at `rel=1e-12` | `tests/python/test_df_encoding.py` |

### Predeclared tolerances

The tolerances above are the ones committed in the repository at the cited
commit. Reuse them as predeclared values for a comparable case; when moving to
a different molecule, basis, or `n`, fix a tolerance from the argument (float64
round-off scaled by `||eri||_F` and by the leaf count) *before* looking at the
result, and say which.

### Expected failure and adversarial cases already covered

Indefinite input on the Cholesky path (warns, degrades); negative-definite
input (warns, zero leaves); asymmetric tensors (two distinct symmetry
violations); C-DF at `max_iterations=1` (warns, reports failure); an absurd
`second_factor_threshold=1e6` (all cores zeroed); a CuPy install that
enumerates a device but fails the kernel probe (auto falls back, explicit
`"cupy"` raises); the line-search replay that would expose a warm start keyed
on call order rather than on `x`.

### Evidence status

| Claim | Status |
| --- | --- |
| Every signature, default, formula, error message, warning, and code path described in this record | `derived` from the named source files at commit `61ac072d…` |
| Every test assertion and tolerance quoted above | `derived` — cited repository evidence, **not** a measurement; nothing was executed in this session |
| GPU crossovers (`n >= 18`, `n >= 56`), the "~4 GB design matrix" figure, "iteration-bound" CG cost, the "~2%" Fréchet cost, "C-DF reaches a target accuracy with substantially fewer leaves", and the in-source note that batched `eigh` matched `scipy.linalg.expm` to ~1e-12 up to `n = 64` on CuPy 13.6 / CUDA 12.9 | `unverified` — documented prior findings with no recorded execution here |
| That `rho ~ 1e-6..1e-3` is a useful range | `unverified` — attributed by the guide to arXiv:2212.07957, not established in this repository |
| Behavior on complex integrals, non-antisymmetric `initial_generators`, shape mismatches in helpers, non-symmetric hand-built cores, non-positive `max_num_leaves` | `unverified` — not covered by source checks, tests, or docs |
| Any energy-error, spectral-error, or downstream-resource consequence of a given `factorization_error` | `assumed` at best; nothing in this family bounds it |

---

## Evaluation coverage

Declared coverage: `double-factorization-encoding-boundary` in
`evals/evals.json` tests explicit-versus-compressed selection, approximation
controls, the host representation, the example-only encoding boundary, and the
refusal to report prior GPU notes as a fresh measurement. It has not been run
with or without the skill, so no uplift is claimed.

The following are additional coverage opportunities:

| Coverage opportunity | What it should test | Sections that answer it |
| --- | --- | --- |
| Positive selection/application | Choosing X-DF versus C-DF versus RC-DF for a stated accuracy-versus-leaf-budget goal, and naming the controlling parameters | Scientific contract; Approximation controls |
| Convention or misconception | That `threshold` is a pivot/eigenvalue floor rather than a Frobenius error bound, that `factorization_error` is absolute rather than relative, and that no spin index exists in this family | Approximation controls; Outputs; Inputs |
| Capability composition | The reconstruct-then-`qubit_hamiltonian` path, including that it discards the frame structure, and that the derived helpers ignore `backend` | Capabilities and composition; Inputs |
| Invalid/unsupported boundary | Indefinite input on the default Cholesky path — warning, dropped negative part, and the eigendecomposition remedy — and the zero-leaf negative-definite case | Accuracy and limitations |
| Installed versus example | Refusing to present `DoubleFactorizedEncoding` as an installed primitive, while still using it correctly as a copyable example | Installed primitives versus the example-only DF block encoding |
| Negative activation | A general "how do I install CuPy / set a CUDA-Q target" request routing to `cudaq-guide` rather than here | `SKILL.md` ownership boundary |
| Refusal to fabricate | Declining to quote a speedup, a runtime, or a GPU crossover as measured | Resources; Validation evidence status |

---

## External alignment

- **Literature conventions.**
  - Cohn, Motta, Parrish, *Quantum Filter Diagonalization with Compressed
    Double-Factorized Hamiltonians*, PRX Quantum **2**, 040352 (2021),
    arXiv:2104.08957 — the X-DF/C-DF definitions, the two-step optimization
    scheme, the gradient (Eq. 17), and the one-body correction (Eq. 3).
  - Oumarou et al., *Quantum* **8**, 1371 (2024), arXiv:2212.07957 — RC-DF: the
    L2 penalty (Eq. 17), the LCU one-norm (Eq. 13), the rank-one one-norm
    (Eq. 15), and the matrix-free CG inner solve (Eqs. 25–30).
  - von Burg et al., PRX Quantum **2**, 030305 (2021), arXiv:2007.14460 — cited
    by the `"burg"` one-norm convention and by the example encoding.
  Attribution of each formula to its equation number is the source's own; the
  papers were not re-derived here.
- **External package translations.**
  - *OpenFermion* `resource_estimates.df.factorize`: an independent explicit-DF
    implementation used as the cross-check oracle. Known semantic differences,
    stated in the test docstring: different leaf storage and a different
    second-factor truncation convention, which is why the comparison runs on
    the reconstructed tensor. The `second_factor_threshold` importance
    weighting is documented as matching OpenFermion's convention.
  - *PySCF* / *Psi4*: sources of the input tensor via
    `chemistry.from_pyscf` / `from_psi4`; `ao2mo.restore("s1", ...)` /
    `restore(1, ...)` produces exactly the dense chemist layout this family
    requires.
  - *CuPy*: optional array backend standing in for NumPy; `cuSOLVER`/`cuBLAS`
    are reached through it.
- **Known semantic differences to watch.**
  - "Double factorization" in the literature often names the *block encoding*;
    here it names only the classical tensor factorization, with the encoding in
    an example file.
  - `lambda` from `double_factorization_one_norm` is convention-dependent by
    design; `"lcu"` and `"burg"` are different numbers for the same
    factorization, and neither is interchangeable with `PauliLCU.alpha`.
  - Chemist `(pq|rs)` versus physicist `<pq|rs>`, and spatial versus spin
    orbitals, are the two translations most likely to be wrong at this
    boundary; `conventions.md` and `docs/sphinx/conventions.rst` own them.

---

# Optional schema: Representation Record

Justified under `architecture.md`: two independent producers
(`explicit_double_factorization`, `compressed_double_factorization`) and
several consumers exchange this object, including one structurally independent
consumer outside the module. It is a representation, not a Python type worth a
record for its own sake.

- Object name and canonical symbol: the double factorization `{U^t, Z^t}`,
  carried by `DoubleFactorization`.
- Public type or structural form, and source path: a `@dataclass` in
  `python/cudaq_algorithms/double_factorization/_factorization.py`, exported
  from the subpackage `__init__`. Not frozen, not validated on construction, no
  `__post_init__`.
- Mathematical meaning: field by field —

  | Field | Type / shape | Meaning |
  | --- | --- | --- |
  | `num_orbitals` | `int` | `n`, the spatial-orbital count |
  | `leaf_rotations` | `List[np.ndarray]`, each `(n, n)` float64 | `U^t`, orthogonal leaf rotations; column `k` is the leaf's rotated orbital `k` |
  | `leaf_cores` | `List[np.ndarray]`, each `(n, n)` float64 | `Z^t`, symmetric cores; rank one from X-DF, general symmetric from C-DF |
  | `method` | `str` | `"X-DF"` or `"C-DF"` |
  | `first_factorization` | `Optional[str]` | `"cholesky"` or `"eigendecomposition"`; `None` on C-DF |
  | `leaf_weights` | `Optional[np.ndarray]`, `(num_leaves,)` | Cholesky pivots (positive) or supermatrix eigenvalues (signed); `None` on C-DF |
  | `optimizer_success` / `optimizer_nit` / `optimizer_grad_norm` | `Optional[bool]` / `Optional[int]` / `Optional[float]` | L-BFGS-B status; all `None` on X-DF |
  | `num_leaves` (property) | `int` | `len(leaf_rotations)`; can be `0` |
  | `reconstruct_eri()` (method) | `np.ndarray` `(n, n, n, n)` | the reconstruction |

- Shape, layout, ordering, dtype, and units: spatial orbitals only, float64,
  host NumPy regardless of backend. Leaf order is pivot order, descending
  `|lambda|`, or warm-start order, per provider. Units are inherited from the
  input integrals.
- Normalization, sign, and phase convention: `U^t` orthogonal; column signs are
  a free gauge; `det U^t = +1` guaranteed only for C-DF. Cores symmetric. No
  normalization is applied to either.
- Required mathematical properties (applicability preconditions for a consumer):
  `len(leaf_rotations) == len(leaf_cores)`; every array `(n, n)` with
  `n == num_orbitals`; rotations orthogonal; cores symmetric. **None of these is
  enforced** — a consumer that needs them must check or must accept
  garbage-in/garbage-out.
- Producers (>= 2 required): `explicit_double_factorization`,
  `compressed_double_factorization`, and hand construction (used by
  `test_one_norms_match_hand_computation`).
- Consumers: `reconstruct_eri`, `factorization_error`,
  `double_factorization_one_norm` (installed); `DoubleFactorizedEncoding` in
  `docs/sphinx/examples/python/df_encoding.py` (example only), which reads
  `num_orbitals`, `leaf_rotations`, and `leaf_cores` and re-exposes the object
  through its `factorization` property.
- Invariants preserved across the boundary: the reconstruction identity, leaf
  orthogonality, core symmetry, and the rotation gauge freedom (any consumer
  must be invariant under a column sign flip; the example encoding's Givens
  sweep explicitly is).
- Observable symptom of a misinterpretation: reading `leaf_weights` as a
  common scale across the two first-factorization paths double-counts
  `lambda_t` on the eigendecomposition path; assuming `det U^t = +1` breaks on
  X-DF leaves; assuming rank-one cores breaks on C-DF; assuming the object
  carries spin structure produces a factor-of-two orbital-count error.
- Unsupported or ambiguous forms: a zero-leaf factorization (legal, produced by
  the negative-definite case, and reconstructs to zero); a hand-built object
  with non-symmetric cores (accepted, but the reconstruction loses pair-swap
  symmetry and the two one-norm conventions read different triangles); any
  complex-valued content (unverified).
- Source paths, tests, docs, and last verification: "Identity and provenance"
  above.
