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

`"auto"` is **size-aware**: below an empirical orbital-count crossover the GPU's
per-kernel launch and host-sync latency makes it slower than NumPy, so `auto`
stays on the CPU for small problems and switches to the GPU only once the work is
large enough to amortize that overhead (and the GPU lead then grows with `n`). The
crossovers differ by method — C-DF (contraction-heavy) crosses around `n ≈ 18`,
while X-DF (a cheap Cholesky loop + tiny per-leaf eigh) stays CPU-favorable until
`n ≈ 56`. Pass `backend="cupy"` or `"numpy"` to force a backend regardless of size.

## API

| function | description |
|----------|-------------|
| `explicit_double_factorization(eri, threshold=1e-8, max_num_leaves=None, second_factor_threshold=0.0, first_factorization="cholesky", backend="auto")` | X-DF (rank-one cores). First factorization defaults to **pivoted Cholesky**; pass `first_factorization="eigendecomposition"` for the eigendecomposition variant. |
| `compressed_double_factorization(eri, num_leaves, max_iterations=2000, tolerance=1e-10, regularization=0.0, inner_solver="lstsq", cg_tolerance=1e-10, cg_max_iterations=None, cg_warm_start=True, cg_optimization_tolerance=None, backend="auto")` | C-DF by least-squares optimization (warm-started from X-DF). `regularization=rho>0` enables **RC-DF** (see below). `inner_solver="cg"` selects the matrix-free inner core solve, with `cg_warm_start` / `cg_optimization_tolerance` accelerators (see below). |
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

## Inner core solve: explicit vs matrix-free

At each L-BFGS step the symmetric cores `Z^t` are recovered exactly by a linear
solve for fixed rotations. Two solvers are available via `inner_solver`:

- `"lstsq"` (default) forms an explicit `n^4 x num_params` design matrix and
  solves it with least squares. Simple and robust for small/medium systems, but
  the design matrix does not scale.
- `"cg"` is the **matrix-free conjugate-gradient** solve (RC-DF,
  [arXiv:2212.07957](https://arxiv.org/abs/2212.07957), Eqs. 25-30). It applies
  the normal-equations operator `A(Z)^t = sum_{t'} M_{tt'} Z^{t'} M_{tt'}^T` with
  `M_{tt'} = (U^t^T U^{t'})` elementwise-squared — only `n x n` matrix products,
  never materializing the design matrix — so it scales to large orbital counts.
  `cg_tolerance` and `cg_max_iterations` control the CG iteration.

Both solvers produce the same reconstruction; when the inner system is full rank
(e.g. `regularization > 0`) they produce the same cores to machine precision.

The CG inner solve is **iteration-bound**: its cost is roughly linear in the
number of CG iterations, which is set by the requested tolerance and the
conditioning, not by the matvec. Two accelerators (active only for
`inner_solver="cg"`) cut the per-step cost without changing the final accuracy:

- **`cg_warm_start=True`** seeds each L-BFGS step's CG from the previous step's
  cores. The cores move slowly between steps, so the warm start collapses the
  iteration count.
- **`cg_optimization_tolerance`** (default `max(cg_tolerance, 1e-6)`) solves the
  *in-loop* systems only loosely — an inexact inner solve, which is sound here
  because the envelope theorem only needs an approximate gradient — while the
  single **final** solve is tightened to `cg_tolerance`. So the returned cores
  are accurate even though the optimization steps used a cheap inner solve.

## Example

See [`examples/double_factorization/double_factorization.py`](../examples/double_factorization/double_factorization.py),
which factorizes H2O/STO-3G integrals from PySCF and compares X-DF and C-DF
reconstruction errors at equal leaf counts.

## From factorization to circuits (the chemistry bridge)

`cudaq_algorithms.chemistry` closes the loop between the classical
preprocessing above and the quantum primitives. It consumes the same
conventions this module does — an `(n, n)` core Hamiltonian and an
`(n, n, n, n)` chemist-notation `(pq|rs)` tensor over real spatial
orbitals — spin-expands them (interleaved spins: `2p` up, `2p + 1` down),
and applies the Jordan-Wigner transform:

```python
from cudaq_algorithms import PauliLCU, chemistry
from cudaq_algorithms import double_factorization as df

factorization = df.compressed_double_factorization(eri, num_leaves=T)
h = chemistry.qubit_hamiltonian(one_body,
                                df.reconstruct_eri(factorization),
                                scalar_offset=nuclear_repulsion)
encoding = PauliLCU(h)      # block encoding of h / alpha -> Walk / QSVT
```

`chemistry.qubit_hamiltonian` returns a `cudaq.SpinOperator`, so the
truncated Hamiltonian feeds `PauliLCU`, `Walk`, and `QSVT` directly. The
payoff of compression shows up on the quantum side as a smaller LCU
normalization `alpha` (QSVT degree scales like `alpha * t` for time
evolution) and fewer Pauli terms (SELECT cost), at the price of a
spectrum shift controlled by the tensor reconstruction error.
`chemistry.spin_orbital_tensors` exposes the spin expansion by itself for
users who need the fermionic tensors.

`chemistry.qubit_hamiltonian` uses `fermion.jordan_wigner`; the classical
factorization itself does not. See
[`examples/double_factorization/df_compression_to_qsvt.py`](../examples/double_factorization/df_compression_to_qsvt.py),
which sweeps X-DF leaf counts on H2/STO-3G (hardcoded integrals, no PySCF
needed) and tabulates tensor error, `alpha`, term count, and the exact
ground-state shift at each truncation.

### Loading integrals

Integrals can come from a mean-field object (`chemistry.from_pyscf` /
`chemistry.from_psi4`) or from an **FCIDUMP** file — the de-facto
interchange format written by Molpro, PySCF, Psi4, and Block/DMRG codes.
`chemistry.from_fcidump` parses the *text* of such a file (the caller does
the file I/O, so the parse stays pure and testable) and returns the same
`(one_body, eri, core_energy)` triple the other loaders do, already in
chemist `(pq|rs)` notation over real spatial orbitals:

```python
from pathlib import Path
from cudaq_algorithms import chemistry

one_body, eri, core = chemistry.from_fcidump(
    Path("molecule.fcidump").read_text())
h = chemistry.qubit_hamiltonian(one_body, eri, scalar_offset=core)
# or hand the same tensors to DoubleFactorizedEncoding(one_body, eri, ...)
```

The reader has no third-party dependency (pure text + NumPy) and fills the
eight-fold index symmetry from the symmetry-unique records FCIDUMP stores.
Only real (RHF/ROHF) FCIDUMP files are supported; complex/UHF variants
(`IUHF=1`) carry a different index symmetry and are rejected by the
`validate_symmetry` check downstream.
