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
3. **[TODO, optional]** wrap the CG iteration in a **CUDA graph** (CuPy supports
   capture) to amortize remaining launch overhead.

Items 1–2 are implemented in `_solve_inner_cores_cg`; they are reversible and
preserve the NumPy/CuPy duality (verified identical to the lstsq solve, recon to
~1e-14 and cores to ~1e-10 when the inner system is full rank).

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
