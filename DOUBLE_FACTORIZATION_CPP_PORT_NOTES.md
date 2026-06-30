# Double Factorization: C++ Port, Composition, and Distributed-Memory Notes

Design/decision notes (no code). Context: the `double_factorization` feature
(X-DF + C-DF, RC-DF regularization) is currently **pure Python** (NumPy/SciPy +
CuPy for the NVIDIA cuSOLVER/cuBLAS path). Questions considered:

1. How much effort to port it to C++ with nanobind bindings?
2. Does that future-proof it for distributed memory (MPI / NCCL)?
3. Does the existence of the separate **C++ qubitization / LCU block-encoding /
   QSVT** feature (QPU kernels as C++ primitives) change the calculus?

---

## TL;DR

- A faithful **full C++ port of the factorization is ~1 month**, dominated by
  C-DF (the L-BFGS optimizer + the matrix-exponential Fréchet-derivative gradient
  + the inner least-squares). X-DF alone is ~days.
- **C++ is not the lever for distributed memory.** The compute is already native
  (cuSOLVER/cuBLAS via CuPy / LAPACK). Distribution is about *data layout and the
  distributed library*, and most of it can be driven from Python.
- **Composition with the C++ qubitization stack happens at the data boundary, not
  the language boundary.** Python DF can already feed the C++ kernels by passing
  arrays as kernel arguments.
- The **highest-leverage C++ work is a new DF block-encoding `__qpu__` kernel**
  (which does not exist yet), not porting the classical factorization.
- On the *classical* side, the highest-leverage native target is the **matrix-free
  CG inner matvec**, not the optimizer. The cheap Python wins (batched-tensor
  matvec, on-device CG scalars) are **done**; profile at scale before going
  native. See §6.
- Port the classical factorization to C++ **only** for a Python-free end-to-end
  pipeline or library-wide C++ consistency — not as a prerequisite for
  composition or distribution.

---

## 1. Effort to port DF → C++ with nanobind

The repo already has the scaffolding (`lib/`, `include/`, `python/bindings/`,
CMake, nanobind). DF is *classical* dense linear algebra (no `__qpu__` kernels),
so it would be a new standalone C++ module. The two halves differ sharply:

### X-DF — easy (~days)
- Pivoted Cholesky: a short loop, or LAPACK `?pstrf` / cuSOLVER.
- Second factorization: `dsyevd` / cuSOLVER `Xsyevd`.
- Reshape / symmetrize: trivial.
- Maps cleanly onto BLAS/LAPACK/cuSOLVER (already linked via CUDA-Q).

### C-DF — hard (~2–3 weeks) and the real cost
Three pieces have no drop-in equivalent in the codebase today:
1. **Optimizer.** `scipy.optimize.minimize(L-BFGS-B)` → add a C++ L-BFGS
   (LBFGS++, liblbfgs, or hand-rolled). New dependency.
2. **`scipy.linalg.expm_frechet`** (Fréchet derivative of the matrix exponential,
   used in the analytic gradient). Biggest correctness risk — either port
   scipy's scaling-and-squaring algorithm or use the closed form in the skew
   eigenbasis (Daleckii–Krein divided differences). The Python gradient was
   verified to ~4e-10; reproducing that in C++ takes care.
3. **`expm` / `logm` of skew matrices, `lstsq`** — doable via eigh/LAPACK,
   moderate.

### Bindings + tests + re-validation (~1 week)
nanobind returning the factorization structs is straightforward; re-establishing
parity with the Python reference and the OpenFermion cross-check is the bulk.

**Total: ~1 month**, concentrated in C-DF. On its own this buys nothing for
distribution.

---

## 2. Language ≠ distribution

Where DF gets big is **data**, not Python overhead:
- ERI tensor `~n^4` and the supermatrix `n^2 x n^2` are the memory walls
  (n ≈ 1000 → ~8 TB ERI). That is the motivation for distributing.
- Compute is already native (CuPy → cuSOLVER/cuBLAS); Python only orchestrates.

Distribution decomposes as:
- **Tensor contractions** (C-DF einsums, the inner solve): distribute ERI blocks
  over ranks → local GEMMs + `allreduce`. **Doable from Python** with CuPy +
  CUDA-aware `mpi4py`, or CuPy's NCCL bindings (`cupy.cuda.nccl`). No C++ needed.
- **Distributed dense eigendecomposition / Cholesky** of the supermatrix: the one
  part with no good pure-Python option. Needs **cuSOLVERMp / cuBLASMp**
  (multi-node multi-GPU, NCCL/NVSHMEM underneath) or ELPA / ScaLAPACK / SLATE —
  all C/C++. This warrants a **thin C++/nanobind shim**, but it is glue around an
  existing library, not a DF rewrite.
- **The L-BFGS optimizer itself stays serial and tiny** — it acts on the small
  rotation-generator vector (`num_leaves * n(n-1)/2`), not the big tensors. Only
  the per-iteration objective/gradient (contractions + inner solve) needs
  distributing.

So a C++ port does **not** deliver distribution for free; the distribution lever
is the backend + comm library either way.

---

## 3. Composition with the C++ qubitization / LCU / QSVT feature

There is a separate feature where qubitization, LCU/PauliLCU block encoding, and
QSVT are C++ `__qpu__` primitives. This is the strongest argument for C++ DF —
but it points at a different deliverable.

### Composition is at the data boundary
DF outputs numbers: leaf rotations `U^t` (→ Givens angles), symmetric cores `Z^t`
(→ controlled-phase angles), one-body eigenvalues. The C++ kernels consume arrays
of doubles as kernel arguments. The repo *already* uses this pattern: in the QSVT
and Givens features, **Python computes the plan/angles and passes them into the
C++ `__qpu__` kernels**. So Python DF can already feed C++ qubitization today —
the seam is `numpy array → nanobind kernel arg`, independent of where the array
was computed. "C++ qubitization exists" does not, by itself, force DF into C++.

### The real C++ gap is a DF *block-encoding kernel*
The qubitization feature has a **generic Pauli-LCU** block encoding. A
double-factorized Hamiltonian has a *much cheaper, structured* block encoding (the
von Burg / Lee construction): apply the leaf Givens rotation `U^t`, controlled
phase rotations on the diagonalized cores, reflect, uncompute. That is the **point
of DF for FTQC** (lower one-norm λ → fewer Toffolis) and it is a **new `__qpu__`
primitive that does not exist yet**.

- It is naturally **C++** (a quantum-circuit primitive like the existing ones).
- It **reuses the existing C++ Givens-rotation kernels** (from the Givens
  Slater-determinant feature) and slots into the existing qubitization walk + QSVT.
- It is **small and well-scoped** (hundreds of lines), not a month-long port.
- It consumes Python-computed `U^t` / `Z^t` as kernel arguments.

This is the high-leverage move: build the **DF block-encoding QPU kernel in C++**,
without touching the classical factorization.

### When porting the *factorization* to C++ is justified
One specific scenario: a **fully C++-native, Python-free end-to-end pipeline** —
"integrals → DF → block encoding → qubitization → resource estimate" in one C++
process (a C++ costing/driver tool, or library consistency where every solver is
C++ with thin bindings). Then C++ DF hands `U^t` / `Z^t` directly to the C++
block-encoding kernel with no marshaling — the cleanest native story, and the
strongest reason to absorb the ~1-month cost.

If Python stays in the orchestration loop (the current repo pattern), keeping DF
in Python is fine and the factorization port is low-value relative to its cost.

---

## 4. Scalability caveats independent of language

These bite before the language does, and are needed in *either* Python or C++:

- **The inner C-DF solve.** The default `inner_solver="lstsq"` builds an explicit
  `n^4 x num_params` design matrix — fine for small/medium systems, but it does
  **not** scale. The scalable form, the paper's **matrix-free conjugate-gradient**
  solve (RC-DF Eqs. 25–30) applying the `A` operator via `n x n` contractions, is
  now available as `inner_solver="cg"`. This is the form to carry into any C++ /
  distributed port; the explicit lstsq path is the small-system reference. The CG
  *matvec* is in fact the single highest-leverage native target in DF — see §6.
- **Forming the full `n^4` ERI / `n^2 x n^2` supermatrix** is itself the memory
  wall; at scale, work directly from Cholesky / density-fitting vectors and never
  materialize the dense tensor.

---

## 5. Recommended sequencing

1. **First / highest value:** a **C++ DF block-encoding `__qpu__` kernel**,
   consuming Python-computed `U^t` / `Z^t`, reusing the existing Givens +
   qubitization primitives. This is what "compose with the C++ stack" actually
   requires, and it is small.
2. **Keep classical DF in Python** behind the existing backend abstraction
   (`_backend.py`). If/when it becomes the bottleneck, distribute via CuPy +
   NCCL/MPI and add a thin shim around a distributed eigensolver
   (cuSOLVERMp / ELPA). Also switch the inner solve to matrix-free CG.
3. **Port classical DF to C++ only** if committing to a Python-free end-to-end
   pipeline or library-wide C++ consistency — then it composes in-process and the
   ~1-month cost buys a clean native stack.

**Net:** the C++ qubitization feature makes a **C++ DF block-encoding kernel**
clearly worth building; it makes a **C++ port of the factorization** a real option
but still a "native pipeline / consistency" decision, not a technical prerequisite
for composition or distribution.

---

## 6. The CG inner matvec: the highest-leverage native target

Of everything in the classical factorization, the **matrix-free CG matvec**
(`apply_operator` in `_solve_inner_cores_cg`) is the best C++/CUDA candidate —
more so than the optimizer or `expm_frechet` flagged in §1. It is the one piece
that is simultaneously **hot, regular, and small-matrix-dominated**, which is
exactly where Python + CuPy leaves the most on the table and where hand control
helps most.

### Why this piece specifically

- **It is the innermost loop.** `apply_operator` runs every CG iteration, CG runs
  to convergence, and the whole solve runs every L-BFGS step — millions of
  invocations over a full optimization.
- **It is structured.** `A(Z)^t = sum_{t'} M_{tt'} Z^{t'} M_{tt'}^T` with
  `M_{tt'} = (U^t^T U^{t'})` elementwise-squared is the *same* congruence batched
  over `num_leaves^2` pairs — it maps cleanly onto batched GEMM or one fused
  kernel.
- **The matrices are small** (`n x n`, `n` = orbital count). Small GEMMs are
  latency-bound, not throughput-bound, so the current code is dominated by
  *overhead*, not arithmetic:
  - `2 * num_leaves^2` tiny cuBLAS launches per matvec (`M@z`, then `…@M^T`),
    driven from a **Python double `for t / for u` loop** that serializes launches
    on one stream.
  - CuPy allocates a temporary for every `@` and every `+`.
  - `inner_product` calls `float(to_numpy(...))` — a **device→host sync every CG
    iteration** — which stalls the pipeline and prevents overlap.

### Do the cheap Python wins first (profile-gated)

Most of that overhead is fixable without leaving Python, and the bottleneck
should be *measured* before any port (at small/medium sizes the whole solve is
sub-second; a port only pays off at the active-space sizes where DF matters):

1. **[DONE]** **Vectorize the matvec into a batched `einsum`** over the leaf
   index. `metric` is stacked as a `(t, t', k, l)` tensor and the operator is the
   single contraction `einsum("tukl,ulm,tunm->tkn", metric, z, metric)` (a
   strided-batched GEMM under cuBLAS), with the cores stacked as one
   `(num_leaves, n, n)` array — replacing the `num_leaves^2` Python-driven launches
   and the double `for t / for u` loop.
2. **[DONE]** **Cut the per-iteration sync** — the CG scalars (`residual_norm_sq`,
   `curvature`, `step`, `beta`) stay as on-device 0-d arrays; only the convergence
   test and the curvature guard convert to host (one scalar each per iteration),
   instead of the old full-reduction-returning-`float` on every inner product.
3. **[BLOCKED from Python]** wrap the CG iteration in a **CUDA graph** to amortize
   remaining launch overhead. Not feasible from CuPy 13.6 — the matvec is a cuBLAS
   call and CuPy errors on cuBLAS during stream capture. See the algorithmic
   wins list below (item 5) for details and the native route.

Items 1–2 are implemented in `_solve_inner_cores_cg`; they are reversible and
preserve the NumPy/CuPy duality (verified identical to the lstsq solve, recon to
~1e-14 and cores to ~1e-10 when the inner system is full rank).

### Measured (benchmarks/double_factorization/bench_cg_inner_solve.py)

One inner core solve, warm-started X-DF rotations, `rho` = regularization.

- **Tier-0 win (batched vs old list-based CG), H2O/STO-3G, n=7, 8 leaves,
  rho=1e-3:** NumPy 64 → 39 ms (**1.7x**), CuPy 1921 → 204 ms (**9.4x**). The win
  is GPU-weighted, as expected — batching removes the `num_leaves^2` micro-launches
  and the per-iteration syncs that dominate on the GPU.
- **H2O/6-311G, n=19, 20 leaves, rho=1e-3, tol=1e-10:** the lstsq design matrix is
  **~4.0 GB** (vs a ~1 MB ERI) — this is the OOM that motivated CG. Batched CG:
  NumPy 2557 ms, **CuPy 432 ms (~5.9x)**. So at the real problem size the GPU wins
  and CG is the only solver that fits.
- **At small sizes the GPU loses** (n=7: CuPy CG 204 ms vs CPU lstsq 30 ms). The
  matvec is too small to saturate; per-iteration launch + sync latency dominates.
- **Crossover and the GPU advantage** (synthetic, leaves=n, rho=1e-2, tol=1e-4):
  GPU time stays nearly flat while CPU explodes, so the GPU lead compounds with
  size — this is the regime where this code outruns CPU-bound implementations:

  | n  | NumPy ms | CuPy ms | GPU speedup |
  |----|----------|---------|-------------|
  | 8  | 4.4      | 20.9    | 0.2x        |
  | 12 | 18.2     | 24.8    | 0.7x        |
  | 16 | 61.1     | 28.4    | 2.2x        |
  | 20 | 164.8    | 31.2    | 5.3x        |
  | 28 | 778.2    | 36.1    | 21.6x       |

**The dominant cost is iteration count, not the matvec.** Time is ~linear in CG
iterations (~0.6 ms/iter on GPU at n=19), and iterations are set by the inner
tolerance and conditioning, not by matvec efficiency:

| rho  | tol   | iters | GPU ms |
|------|-------|-------|--------|
| 1e-3 | 1e-10 | 700   | 435    |
| 1e-3 | 1e-6  | 316   | 200    |
| 1e-3 | 1e-4  | 119   | 81     |
| 1e-2 | 1e-10 | 245   | 159    |
| 1e-2 | 1e-4  | 60    | 45     |

So the next wins are **algorithmic**, and they compound the Tier-0 kernel win:

1. **[DONE] Inexact inner solve / forcing sequence + warm start.** The in-loop CG
   no longer solves to `tol=1e-10` every L-BFGS step (`cg_optimization_tolerance`,
   default `max(cg_tolerance, 1e-6)`) and warm-starts from the previous step's
   cores (`cg_warm_start`, default `True`); only the single final solve is tight.
   By the envelope theorem the gradient only needs to be approximate, so final
   accuracy is unchanged. **End-to-end (synthetic n=16, 8 leaves, rho=1e-3,
   maxiter=150): NumPy 11.2 -> 5.9 s (1.9x), CuPy 17.5 -> 9.6 s (1.8x)**, with the
   final reconstruction error matching the cold/tight path.
2. **[DONE] Size-aware `auto` backend.** Since the GPU loses below the crossover,
   `backend="auto"` now picks NumPy for small problems and CuPy only at/above an
   empirical orbital-count threshold (C-DF `n >= 18`, X-DF `n >= 56`; the X-DF
   crossover is higher — a cheap Cholesky loop + tiny per-leaf eigh stays
   CPU-favorable until ~n=64). `"cupy"`/`"numpy"` still force a backend. This is
   the "best of both" the benchmarks call for: no GPU penalty on small inputs, GPU
   speedup (growing with `n`) on large ones.
3. **[DONE] Batched GPU-resident reconstruct/gradient + eigh.** The per-step work
   outside the inner solve is now fully batched over leaves: `_reconstruct_dev` is
   a single `einsum("tpk,tqk,tkl,trl,tsl->pqrs", ...)` (leaf sum folded in) instead
   of a Python loop with a separate `n^4` einsum per leaf; the gradient `dO/dU` is
   one `einsum("aqrs,tqk,tkl,trl,tsl->tak", ...)` with a single device->host
   transfer; and `unpack` exponentiates all leaf generators in one *batched*
   Hermitian eigh (`expm_skew_symmetric_batched`, a single cuSOLVER call) instead
   of `num_leaves` separate ones. Numerically identical (all 25 tests pass).
   GPU-weighted as expected (it removes per-leaf launch overhead): end-to-end
   accel path at n=16, 8 leaves, maxiter=150 drops **CuPy 9.6 -> 7.9 s (1.21x),
   NumPy 5.9 -> 5.3 s (1.10x)**; the gain grows with leaf count.
4. **[WONT — Jacobi is a no-op here]** A Jacobi/diagonal (or block-Jacobi-by-leaf)
   preconditioner does nothing for this operator: orthonormal leaves give
   `M_tt = (U_t^T U_t)^{circ 2} = I`, so the operator is `A = (1+2rho) I + C` with
   `C` the pure inter-leaf coupling (zero diagonal). The diagonal is *exactly
   constant* (`1+2rho`; verified numerically), so Jacobi is a uniform rescale and
   leaves CG convergence unchanged. The conditioning (`~3/(2rho)`: `lambda_min =
   2rho` exactly, `lambda_max ~ 3`) lives entirely in `C`. The effective option is
   a **polynomial (Chebyshev) accelerator** on the known interval `[2rho, lambda_max]`
   -- which is also sync-free (no per-iteration dot products, so GPU-friendly) but
   needs `rho > 0` (falls back to CG at `rho = 0`). Deferred.
5. **[BLOCKED from Python — needs native cuBLAS] CUDA graph capture** (Tier-0
   item 3). Capturing the CG iteration as a CUDA graph would amortize the
   per-iteration launch + Python overhead that makes the GPU lose at small sizes.
   But the iteration's dominant op is the matvec, which is a cuBLAS GEMM (via
   `einsum`/`matmul`), and **CuPy 13.6 errors with "calling cuBLAS API during
   stream capture is currently unsupported"** — verified by prototype. So the one
   op worth capturing is exactly the one CuPy forbids during capture; capturing
   only the elementwise axpy/reductions (which *are* capture-safe) and leaving the
   matvec outside saves almost nothing. Additional friction: `cupy.einsum` has no
   `out=` (so the matvec result needs a `copyto` into a fixed buffer for stable
   graph addresses anyway). The viable route is raw cuBLAS with a
   capture-compatible workspace (`cublasSetStream`/`cublasSetWorkspace`) under the
   CUDA driver graph API — i.e. the native C++/CUDA port (§1, §6 "port only the
   matvec"), where the matvec kernel and the CG driver are captured together. Not
   pursued from Python.

**Where the end-to-end time went** (profiled *before* item 3, full C-DF, CuPy,
n=16, 8 leaves): inner CG solve ~53%, the `n^4` reconstruct/gradient `einsum`s +
the per-leaf eigh in `unpack` (`expm_skew_symmetric`) ~45%, and
`scipy.linalg.expm_frechet` (host) **only ~2%**. So the Fréchet derivative is
*not* the bottleneck — contrary to the §1/§2 worry — and a GPU port of it buys
little on its own. That ~45% reconstruct/gradient/eigh share is what item 3 above
batched; `expm_frechet` is deliberately left on the host.

**End-to-end CPU/GPU crossover is ~n=18-20**, higher than the inner-solve
crossover (~n=14-16), because the full step also runs L-BFGS, the per-leaf
eigh/Fréchet, and per-step host syncs. Measured (accel path, identical problem per
backend at each `n`, leaves as noted, rho=1e-3, maxiter=100):

| n (leaves) | NumPy accel | CuPy accel | GPU vs CPU |
|------------|-------------|------------|------------|
| 16 (8)     | 5.9 s       | 9.6 s      | 0.6x (GPU loses) |
| 20 (10)    | 12.0 s      | 8.1 s      | 1.5x       |
| 24 (10)    | 11.2 s      | 5.4 s      | 2.1x       |

(Errors differ across `n` because each is a different random ERI, unconverged at
maxiter=100; the CPU-vs-GPU comparison is valid since both backends solve the same
problem at each size.) The accelerator speedup itself holds steady at ~1.8-1.9x on
both backends regardless of size. Only once `n` is large enough that the
reconstruct/gradient `einsum`s *and* the inner solve are all real GPU work (and
iteration count is controlled, as it now is) does the GPU win end-to-end — and the
lead grows with size (2.1x by n=24). That large-`n`, controlled-iteration regime
is where this code outruns CPU-bound implementations of the same algorithm.

### If/when it goes native, port only the matvec

The right seam is surgical: push **just `apply_operator`** (ideally with the
dot/axpy reductions) across the nanobind boundary and **keep the CG driver and
L-BFGS in Python**. The driver only touches tiny data (scalars, the small
generator vector); the heavy, regular work is the congruence. This is the repo's
existing pattern (Python orchestrates, C++/CUDA does the primitive) and it
sidesteps everything expensive in a full port — the L-BFGS dependency,
`expm_frechet`, and the OpenFermion re-validation.

A fused kernel can keep `z`-blocks in shared memory across the `num_leaves^2`
congruence terms, do the accumulation in registers, and run the reductions
on-device — control CuPy cannot express.

### This is also the distribution seam

Once the matvec is a kernel over per-leaf / per-ERI-block partial sums,
distributing it is an **NCCL allreduce of the partial `A(Z)^t`** across ranks.
That — not single-GPU speed — is the real reason to own this in native code, and
it lines up with the distributed-eigensolver shim in §2.

### Sequencing

- **Tier 0 (DONE, Python):** batched-tensor matvec + on-device CG scalars
  (per-iteration host sync removed except the convergence/curvature guards).
  Cheap, reversible, probably the bulk of the win.
- **Tier 1 (C++/CUDA, when profiled as the bottleneck):** fused `apply_operator`
  kernel via nanobind; CG driver and L-BFGS stay in Python.
- **Tier 2 (C++/CUDA + NCCL):** partial-sum matvec with allreduce for distributed
  ERIs/leaves.

---

## 7. X-DF vs C-DF: accuracy and cost vs leaf count (H2O/6-311G)

Measured with `benchmarks/double_factorization/bench_cg_inner_solve.py --leaf-sweep`
(n=19, rho=1e-3, C-DF maxiter=300; X-DF auto->NumPy in ms, C-DF auto->GPU):

| leaves | X-DF rel err | C-DF rel err | C-DF/X-DF | X-DF time | C-DF time |
|--------|--------------|--------------|-----------|-----------|-----------|
| 4      | 1.58e-1      | 4.17e-2      | 0.26x     | 0.001 s   | 28.6 s    |
| 16     | 4.93e-2      | 7.08e-3      | 0.14x     | 0.001 s   | 65.4 s    |
| 32     | 2.08e-2      | 3.11e-3      | 0.15x     | 0.002 s   | 91.1 s    |
| 64     | 2.12e-3      | 9.50e-4      | 0.45x     | 0.005 s   | 125.7 s   |
| 96     | 9.72e-5      | 5.59e-4      | 5.75x     | 0.009 s   | 155.9 s   |

Takeaways:
- **C-DF's value is compression at low/moderate rank.** At 16 leaves C-DF reaches
  ~7e-3, an accuracy X-DF needs ~3-4x more leaves to match -- fewer leaves means a
  cheaper Givens fabric / lower block-encoding cost. That is the point of C-DF for
  FTQC, and it holds for leaves up to ~64 here.
- **X-DF wins at high leaf count.** By 96 leaves X-DF (9.7e-5) is approaching the
  true rank (~177) and marches toward machine precision, while regularized C-DF is
  *floored* by the `rho` penalty (and the maxiter budget), so X-DF overtakes it.
  The `rho=1e-3` term is a confound for a pure accuracy comparison at high rank;
  `rho=0` lifts the C-DF floor (at the cost of one-norm / conditioning).
- **Cost asymmetry is enormous.** X-DF is milliseconds; C-DF is 30-160 s (the
  iterative optimization), growing ~linearly in leaf count. C-DF is a one-time
  classical pre-processing investment justified only when the leaf savings matter
  downstream (circuit depth, measurement variance), not for raw reconstruction.
- **Regularization is load-bearing for CG tractability, not just the one-norm.**
  A `rho=0` control run timed out before finishing a single leaf row. Cause
  (measured, one inner solve at 64 leaves, n=19, tol=1e-6): the inner operator is
  rank-deficient at `rho=0` (`lambda_min -> 0`), so CG iterations explode --
  `rho=1e-3`: 407 iters / 0.27 s; `rho=1e-4`: 908 / 0.56 s; **`rho=0`: 4876 /
  2.83 s (~12x)** -- and that is *per L-BFGS step*. So `rho>0` conditions the CG
  solve (an RC-DF benefit beyond shrinking the one-norm); the explicit lstsq solver
  tolerates `rho=0` (direct min-norm) but its design matrix OOMs at these leaf
  counts (>12 GB). Net: **C-DF-with-CG is inherently a regularized, low-to-moderate
  rank tool**; for near-full-rank accuracy X-DF is both exact-ish and free, so the
  96-leaf reversal above is the expected regime boundary, not a deficiency.
