# Double Factorization (X-DF and C-DF)

`cudaq_algorithms.double_factorization` factorizes the two-electron integrals of
an electronic-structure Hamiltonian into a sum of low-rank, diagonalizable pieces
— the representation used by double-factorized block encodings, qubitization, and
measurement schemes. It implements the explicit (X-DF) and compressed (C-DF)
variants of Cohn, Motta, and Parrish, *Quantum Filter Diagonalization with
Compressed Double-Factorized Hamiltonians*, PRX Quantum **2**, 040352 (2021)
([arXiv:2104.08957](https://arxiv.org/abs/2104.08957)).

## Convention

The two-electron integrals are supplied in **chemist notation** `(pq|rs)` over
real spatial orbitals (8-fold symmetric), as an `(n, n, n, n)` array `eri` with
`eri[p, q, r, s] == (pq|rs)` (e.g. from `pyscf.ao2mo.restore("s1", ...)`).

Double factorization writes

```
(pq|rs)  ~=  sum_t sum_{k,l} U^t_pk U^t_qk  Z^t_kl  U^t_rl U^t_sl
```

with orthogonal **leaf rotations** `U^t` (which compile into a Givens-rotation
fabric) and symmetric **core** matrices `Z^t`.

## Backends (NVIDIA math libraries)

Heavy linear algebra (eigendecompositions, tensor contractions, least squares)
runs on the NVIDIA math libraries — cuSOLVER and cuBLAS via **CuPy** — when a GPU
is available, and falls back to **NumPy/SciPy** otherwise. Every entry point
accepts `backend="auto"` (default), `"cupy"`, or `"numpy"`.

## API

| function | description |
|----------|-------------|
| `explicit_double_factorization(eri, threshold=1e-8, max_num_leaves=None, second_factor_threshold=0.0, first_factorization="cholesky", backend="auto")` | X-DF (rank-one cores). First factorization defaults to **pivoted Cholesky**; pass `first_factorization="eigendecomposition"` for the eigendecomposition variant. |
| `compressed_double_factorization(eri, num_leaves, max_iterations=2000, tolerance=1e-10, regularization=0.0, backend="auto")` | C-DF by least-squares optimization (warm-started from X-DF). `regularization=rho>0` enables **RC-DF** (see below). |
| `reconstruct_eri(factorization)` | Rebuild the `(pq\|rs)` tensor from a factorization. |
| `factorization_error(eri, factorization)` | Frobenius norm of the reconstruction residual. |
| `modified_one_body_integrals(one_body, eri)` | The DF one-body correction `kappa_pq = h_pq - 1/2 sum_r (pr\|qr)`. |
| `double_factorization_one_norm(factorization, one_body_eigenvalues, convention="lcu")` | One-norm `lambda` of the DF Hamiltonian; `convention="lcu"` (Pauli-rotation) or `"burg"` (qubitization). |

All functions return/operate on a `DoubleFactorization` dataclass with
`num_orbitals`, `leaf_rotations` (`U^t`), `leaf_cores` (`Z^t`), `num_leaves`, and
`reconstruct_eri()`.

## X-DF vs C-DF

- **X-DF** is exact and cheap. The first factorization of the ERI supermatrix
  into symmetric leaves `(pq|rs) = sum_t L^t_pq L^t_rs` defaults to **pivoted
  Cholesky**: the ERI matrix is positive semidefinite (a Gram matrix of orbital
  densities), so Cholesky is rank-revealing — it keeps leaves while the residual
  pivot exceeds `threshold` and stops at the true rank (at most the symmetric-pair
  dimension `n(n+1)/2`), which is cheaper than a full eigendecomposition. (Pass
  `first_factorization="eigendecomposition"` for the symmetric-eigendecomposition
  variant, `(pq|rs) = sum_t lambda_t V^t_pq V^t_rs`, required for indefinite
  inputs.) Each symmetric leaf is then eigendecomposed to give `U^t` and the
  rank-one core `Z^t`. At full rank the reconstruction is exact to machine
  precision, independent of the first-factor method.
- **C-DF** lifts the rank-one restriction on `Z^t` and minimizes
  `O = 1/2 || eri - reconstruction ||_F^2` over a fixed number of leaves. The leaf
  rotations are parameterized as `U^t = exp(X^t)` (antisymmetric `X^t`) and
  optimized with L-BFGS, while the symmetric cores are solved in closed form at
  each step (the two-step scheme of the paper, warm-started from X-DF). C-DF
  reaches a target accuracy with substantially fewer leaves than X-DF.

## RC-DF regularization

Setting `regularization=rho > 0` enables **regularized C-DF** (RC-DF, Oumarou
et al., *Quantum* **8**, 1371 (2024), [arXiv:2212.07957](https://arxiv.org/abs/2212.07957)),
adding the L2 penalty `rho * sum_{t,k,l} (Z^t_kl)^2` to the objective (Eq. 17).
The penalty is folded into the inner core solve as a ridge term, so it actually
shrinks the cores `Z^t` (it also conditions the otherwise rank-deficient inner
system). Smaller cores lower the Hamiltonian one-norm `lambda` — and hence the
qubitization runtime and measurement variance — at a modest cost in
reconstruction accuracy. `rho` is an absolute coefficient whose useful scale
depends on the integral magnitude (the paper uses `~1e-6` to `1e-3`); pick it by
trading off `factorization_error` against `double_factorization_one_norm`. By the
envelope theorem the analytic optimization gradient is unchanged in form, so
RC-DF runs at the same per-iteration cost as plain C-DF.

## Example

See [`examples/double_factorization/double_factorization.py`](../examples/double_factorization/double_factorization.py),
which factorizes H2O/STO-3G integrals from PySCF and compares X-DF and C-DF
reconstruction errors at equal leaf counts.
