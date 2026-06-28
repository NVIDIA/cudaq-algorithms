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
| `explicit_double_factorization(eri, eigenvalue_threshold=1e-8, max_num_leaves=None, second_factor_threshold=0.0, backend="auto")` | X-DF via nested eigendecompositions (rank-one cores). |
| `compressed_double_factorization(eri, num_leaves, max_iterations=2000, tolerance=1e-10, regularization=0.0, backend="auto")` | C-DF by least-squares optimization (warm-started from X-DF). |
| `reconstruct_eri(factorization)` | Rebuild the `(pq\|rs)` tensor from a factorization. |
| `factorization_error(eri, factorization)` | Frobenius norm of the reconstruction residual. |
| `modified_one_body_integrals(one_body, eri)` | The DF one-body correction `kappa_pq = h_pq - 1/2 sum_r (pr\|qr)`. |
| `double_factorization_one_norm(factorization, one_body_eigenvalues)` | LCU one-norm `lambda` of the DF Hamiltonian. |

All functions return/operate on a `DoubleFactorization` dataclass with
`num_orbitals`, `leaf_rotations` (`U^t`), `leaf_cores` (`Z^t`), `num_leaves`, and
`reconstruct_eri()`.

## X-DF vs C-DF

- **X-DF** is exact and cheap: the ERI supermatrix is eigendecomposed into leaves
  `(pq|rs) = sum_t lambda_t V^t_pq V^t_rs` (truncating small `|lambda_t|`), then
  each symmetric leaf `V^t` is eigendecomposed to give `U^t` and the rank-one core
  `Z^t_kl = lambda_t gamma^t_k gamma^t_l`. At full rank the reconstruction is
  exact to machine precision.
- **C-DF** lifts the rank-one restriction on `Z^t` and minimizes
  `O = 1/2 || eri - reconstruction ||_F^2` over a fixed number of leaves. The leaf
  rotations are parameterized as `U^t = exp(X^t)` (antisymmetric `X^t`) and
  optimized with L-BFGS, while the symmetric cores are solved in closed form at
  each step (the two-step scheme of the paper, warm-started from X-DF). C-DF
  reaches a target accuracy with substantially fewer leaves than X-DF.

## Example

See [`examples/double_factorization/double_factorization.py`](../examples/double_factorization/double_factorization.py),
which factorizes H2O/STO-3G integrals from PySCF and compares X-DF and C-DF
reconstruction errors at equal leaf counts.
