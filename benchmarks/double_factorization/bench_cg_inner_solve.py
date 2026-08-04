# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmarks for the C-DF matrix-free CG inner core solve.

Modes:
  (default)     new batched CG vs explicit lstsq (where the design matrix fits),
                on NumPy and CuPy.
  --ab          also time the original list-based CG to measure the Tier-0
                vectorization win. SLOW at n>=19 -- the list-based CG is exactly
                what this work replaced.
  --tol-sweep   CG iterations and time vs inner tolerance / regularization.
                Shows that cost is iteration-bound, not matvec-bound.
  --scaling     synthetic sweep in n to locate the CPU/GPU crossover.

Default problem is H2O/6-311G (n = 19), the case that motivated the matrix-free
path: there the lstsq design matrix is ~4 GB while the ERI itself is ~1 MB.

    PYTHONPATH=build/python:/usr/local/cudaq python3 \
        benchmarks/double_factorization/bench_cg_inner_solve.py --tol-sweep
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import cudaq_algorithms as algorithms

df = algorithms.double_factorization
from cudaq_algorithms.double_factorization import _factorization as F
from cudaq_algorithms.double_factorization._backend import (
    AUTO_GPU_MIN_ORBITALS_COMPRESSED, AUTO_GPU_MIN_ORBITALS_EXPLICIT,
    resolve_backend, to_device, to_numpy)


def _sync(xp):
    """Block until queued device work finishes (no-op on NumPy)."""
    if xp.__name__ == "cupy":
        xp.cuda.runtime.deviceSynchronize()


def molecular_eri(basis):
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.957; H 0 0.926 -0.24",
                basis=basis,
                verbose=0)
    mf = scf.RHF(mol).run()
    n = mf.mo_coeff.shape[1]
    return np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))


def synthetic_eri(n, num_vectors, seed=0):
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((num_vectors, n, n))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    return np.einsum("xpq,xrs->pqrs", leaves, leaves)


def warm_start_rotations(eri, num_leaves, xp):
    """Realistic leaf rotations: the top ``num_leaves`` X-DF leaves on device."""
    xdf = df.explicit_double_factorization(eri,
                                           threshold=0.0,
                                           max_num_leaves=num_leaves)
    rotations = xdf.leaf_rotations[:num_leaves]
    while len(rotations) < num_leaves:  # pad if rank-deficient
        rotations.append(np.eye(eri.shape[0]))
    return [to_device(np.asarray(r, dtype=float), xp) for r in rotations]


def cg_iterations(eri_dev, rotations_dev, xp, regularization, tolerance):
    """Iteration count for the batched CG (mirrors _solve_inner_cores_cg)."""
    n = eri_dev.shape[0]
    num_leaves = len(rotations_dev)
    rotations = xp.stack(rotations_dev)
    metric = xp.einsum("tpk,upl->tukl", rotations, rotations)
    metric = metric * metric
    rhs = xp.stack(
        [F._project_eri_into_leaf(eri_dev, u, xp) for u in rotations_dev])

    def apply_operator(z):
        out = xp.einsum("tukl,ulm,tunm->tkn", metric, z, metric, optimize=True)
        return out + (2.0 * regularization) * z if regularization > 0 else out

    ip = lambda x, y: xp.sum(x * y)
    z = xp.zeros((num_leaves, n, n))
    residual = rhs.copy()
    direction = residual.copy()
    rs = ip(residual, residual)
    threshold = (tolerance**2) * max(float(to_numpy(rs)), 1.0)
    iterations = 0
    for _ in range(max(50, num_leaves * n * n)):
        if float(to_numpy(rs)) <= threshold:
            break
        iterations += 1
        ad = apply_operator(direction)
        curvature = ip(direction, ad)
        if float(to_numpy(curvature)) <= 0:
            break
        step = rs / curvature
        z = z + step * direction
        residual = residual - step * ad
        new = ip(residual, residual)
        direction = residual + (new / rs) * direction
        rs = new
    return iterations


# --------------------------------------------------------------------------- #
# The ORIGINAL list-based CG, reconstructed here so we can A/B the Tier-0 win.
# (Double for-loop matvec; per-iteration host sync on every inner product.)
# --------------------------------------------------------------------------- #
def cg_list_based(eri_dev, rotations_dev, xp, regularization, tolerance):
    n = eri_dev.shape[0]
    num_leaves = len(rotations_dev)
    overlaps = [[
        rotations_dev[t].T @ rotations_dev[u] for u in range(num_leaves)
    ] for t in range(num_leaves)]
    metric = [[overlaps[t][u] * overlaps[t][u] for u in range(num_leaves)]
              for t in range(num_leaves)]
    rhs = [F._project_eri_into_leaf(eri_dev, u, xp) for u in rotations_dev]

    def apply_operator(z):
        result = []
        for t in range(num_leaves):
            acc = xp.zeros((n, n))
            for u in range(num_leaves):
                acc = acc + metric[t][u] @ z[u] @ metric[t][u].T
            if regularization > 0.0:
                acc = acc + 2.0 * regularization * z[t]
            result.append(acc)
        return result

    def inner_product(x, y):
        return float(to_numpy(sum(xp.sum(xi * yi) for xi, yi in zip(x, y))))

    z = [xp.zeros((n, n)) for _ in range(num_leaves)]
    residual = [r.copy() for r in rhs]
    direction = [r.copy() for r in residual]
    residual_norm_sq = inner_product(residual, residual)
    threshold = (tolerance**2) * max(residual_norm_sq, 1.0)
    for _ in range(max(50, num_leaves * n * n)):
        if residual_norm_sq <= threshold:
            break
        operator_direction = apply_operator(direction)
        curvature = inner_product(direction, operator_direction)
        if curvature <= 0.0:
            break
        step = residual_norm_sq / curvature
        z = [zi + step * di for zi, di in zip(z, direction)]
        residual = [
            ri - step * oi for ri, oi in zip(residual, operator_direction)
        ]
        new_norm_sq = inner_product(residual, residual)
        beta = new_norm_sq / residual_norm_sq
        direction = [ri + beta * di for ri, di in zip(residual, direction)]
        residual_norm_sq = new_norm_sq
    return [0.5 * (zi + zi.T) for zi in z]


def time_call(fn, repeats, xp):
    fn()  # warm up (allocations, cuBLAS handles)
    _sync(xp)
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(xp)
        best = min(best, time.perf_counter() - start)
    return best, result


def lstsq_design_gb(n, num_leaves):
    return (n**4) * (num_leaves * n * (n + 1) // 2) * 8 / 1e9


def run_default(eri, num_leaves, rho, tol, repeats, ab):
    n = eri.shape[0]
    for backend in _backends():
        xp, name = resolve_backend(backend)
        eri_dev = to_device(np.asarray(eri, dtype=float), xp)
        rotations = warm_start_rotations(eri, num_leaves, xp)
        print(f"\n=== {name}  n={n}  leaves={num_leaves}  rho={rho:g}  "
              f"tol={tol:g} ===")

        iters = cg_iterations(eri_dev, rotations, xp, rho, tol)
        new_time, new_cores = time_call(
            lambda: F._solve_inner_cores_cg(eri_dev, rotations, xp, rho, tol,
                                            None), repeats, xp)
        print(
            f"  CG (batched)       : {new_time*1e3:9.2f} ms   ({iters} iters)")

        if ab:
            old_time, old_cores = time_call(
                lambda: cg_list_based(eri_dev, rotations, xp, rho, tol),
                repeats, xp)
            diff = max(
                float(np.linalg.norm(to_numpy(a) - to_numpy(b)))
                for a, b in zip(new_cores, old_cores))
            print(f"  CG (old list-based): {old_time*1e3:9.2f} ms   "
                  f"speedup x{old_time/new_time:.1f}   |dZ|={diff:.1e}")

        gb = lstsq_design_gb(n, num_leaves)
        if gb <= 8.0:
            ls_time, ls_cores = time_call(
                lambda: F._solve_inner_cores_lstsq(eri_dev, rotations, xp, rho
                                                   ), repeats, xp)
            rn = to_numpy(F._reconstruct_dev(rotations, new_cores, xp))
            rl = to_numpy(F._reconstruct_dev(rotations, ls_cores, xp))
            print(f"  lstsq (design {gb:.2f} GB): {ls_time*1e3:9.2f} ms   "
                  f"|recon dCG-lstsq|={np.linalg.norm(rn-rl):.1e}")
        else:
            print(f"  lstsq: SKIPPED -- design matrix {gb:.1f} GB (> 8 GB); "
                  f"this is the case CG exists for.")


def run_tol_sweep(eri, num_leaves, repeats):
    n = eri.shape[0]
    for backend in _backends():
        xp, name = resolve_backend(backend)
        eri_dev = to_device(np.asarray(eri, dtype=float), xp)
        rotations = warm_start_rotations(eri, num_leaves, xp)
        print(f"\n=== tol/rho sweep  {name}  n={n}  leaves={num_leaves} ===")
        print(f"  {'rho':>8} {'tol':>8} {'iters':>6} {'ms':>9}")
        for rho in (1e-3, 1e-2):
            for tol in (1e-10, 1e-6, 1e-4):
                iters = cg_iterations(eri_dev, rotations, xp, rho, tol)
                t, _ = time_call(
                    lambda: F._solve_inner_cores_cg(
                        eri_dev, rotations, xp, rho, tol, None), repeats, xp)
                print(f"  {rho:>8g} {tol:>8g} {iters:>6d} {t*1e3:>9.1f}")


def run_scaling(rho, tol, repeats):
    for backend in _backends():
        xp, name = resolve_backend(backend)
        print(f"\n=== scaling sweep (synthetic)  {name}  rho={rho:g} ===")
        print(f"  {'n':>4} {'leaves':>7} {'iters':>6} {'ms':>9}")
        for n in (8, 16, 24, 32, 48):
            eri = synthetic_eri(n, num_vectors=max(2, n // 2), seed=1)
            eri_dev = to_device(eri, xp)
            rotations = warm_start_rotations(eri, n, xp)
            iters = cg_iterations(eri_dev, rotations, xp, rho, tol)
            t, _ = time_call(
                lambda: F._solve_inner_cores_cg(eri_dev, rotations, xp, rho,
                                                tol, None), repeats, xp)
            print(f"  {n:>4} {n:>7} {iters:>6d} {t*1e3:>9.1f}")


def run_full(n, num_leaves, rho, max_iterations, repeats):
    """End-to-end compressed_double_factorization: warm-start + inexact in-loop
    CG ('accel', the new defaults) vs the cold/tight in-loop solve ('plain'),
    on a synthetic ERI. Verifies the final reconstruction error matches."""
    eri = synthetic_eri(n, num_vectors=max(2, num_leaves - 1), seed=5)
    for backend in _backends():
        print(f"\n=== full C-DF  {backend}  n={n}  leaves={num_leaves}  "
              f"rho={rho:g}  maxiter={max_iterations} ===")

        def solve(accel):
            kw = dict(num_leaves=num_leaves,
                      max_iterations=max_iterations,
                      regularization=rho,
                      inner_solver="cg",
                      backend=backend)
            if not accel:  # original behavior: cold start, tight in-loop solve
                kw.update(cg_warm_start=False, cg_optimization_tolerance=1e-10)
            return df.compressed_double_factorization(eri, **kw)

        best_plain = best_accel = float("inf")
        err_plain = err_accel = None
        for _ in range(repeats):
            t = time.perf_counter()
            f = solve(False)
            best_plain = min(best_plain, time.perf_counter() - t)
            err_plain = df.factorization_error(eri, f)
            t = time.perf_counter()
            f = solve(True)
            best_accel = min(best_accel, time.perf_counter() - t)
            err_accel = df.factorization_error(eri, f)
        print(
            f"  plain (cold, tol=1e-10): {best_plain:7.2f} s  err={err_plain:.3e}"
        )
        print(
            f"  accel (warm + inexact) : {best_accel:7.2f} s  err={err_accel:.3e}"
            f"   speedup x{best_plain/best_accel:.1f}")


def run_leaf_sweep(eri, leaf_counts, rho, max_iterations, repeats):
    """Time and accuracy vs leaf count for X-DF (Cholesky first factorization)
    and C-DF (matrix-free CG inner solve), on a molecular ERI. Rows print as they
    are computed so partial results survive if the large leaf counts run long."""
    n = eri.shape[0]
    norm = float(np.linalg.norm(eri))
    xdf_backend = resolve_backend(
        "auto", problem_size=n, gpu_min_size=AUTO_GPU_MIN_ORBITALS_EXPLICIT)[1]
    cdf_backend = resolve_backend(
        "auto", problem_size=n,
        gpu_min_size=AUTO_GPU_MIN_ORBITALS_COMPRESSED)[1]
    print(
        f"\n=== leaf sweep  n={n}  rho={rho:g}  C-DF maxiter={max_iterations} "
        f"  (auto backends: X-DF={xdf_backend}, C-DF={cdf_backend}) ===")
    print(f"  {'leaves':>6} | {'X-DF s':>8} {'X-DF rel':>10} | "
          f"{'C-DF s':>8} {'C-DF rel':>10}  {'C-DF/X-DF err':>13}")

    for num_leaves in leaf_counts:

        def do_xdf():
            return df.explicit_double_factorization(
                eri,
                threshold=0.0,
                max_num_leaves=num_leaves,
                first_factorization="cholesky",
                backend="auto")

        def do_cdf():
            return df.compressed_double_factorization(
                eri,
                num_leaves=num_leaves,
                max_iterations=max_iterations,
                regularization=rho,
                inner_solver="cg",
                backend="auto")

        t_x, xdf = _best_time(do_xdf, repeats, xdf_backend)
        x_rel = df.factorization_error(eri, xdf) / norm
        t_c, cdf = _best_time(do_cdf, repeats, cdf_backend)
        c_rel = df.factorization_error(eri, cdf) / norm
        ratio = c_rel / x_rel if x_rel > 0 else float("nan")
        print(f"  {num_leaves:>6} | {t_x:>8.3f} {x_rel:>10.3e} | "
              f"{t_c:>8.3f} {c_rel:>10.3e}  {ratio:>12.2f}x")


def _best_time(fn, repeats, backend_name):
    xp, _ = resolve_backend(backend_name)
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(xp)
        best = min(best, time.perf_counter() - start)
    return best, result


def _backends():
    backends = ["numpy"]
    if df.cupy_gpu_available():
        backends.append("cupy")
    return backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-311g")
    parser.add_argument("--leaves", type=int, default=20)
    parser.add_argument("--rho", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ab",
                        action="store_true",
                        help="also time the original list-based CG (slow)")
    parser.add_argument("--tol-sweep", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--full",
                        action="store_true",
                        help="end-to-end C-DF: accel vs plain in-loop CG")
    parser.add_argument("--n",
                        type=int,
                        default=16,
                        help="orbital count for --full (synthetic ERI)")
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument("--leaf-sweep",
                        action="store_true",
                        help="X-DF (Cholesky) and C-DF (CG) time/accuracy vs "
                        "leaf count, on the molecular ERI")
    parser.add_argument("--leaves-list",
                        default="4,16,32,64,96",
                        help="comma-separated leaf counts for --leaf-sweep")
    args = parser.parse_args()

    if not df.cupy_gpu_available():
        print("(CuPy GPU not available -- NumPy only)")

    if args.full:  # synthetic ERI; no PySCF needed
        run_full(args.n, args.leaves, args.rho, args.max_iterations,
                 args.repeats)
    elif args.scaling:  # synthetic sweep; no PySCF needed
        run_scaling(args.rho, args.tol, args.repeats)
    else:
        eri = molecular_eri(args.basis)
        print(f"H2O/{args.basis}: n={eri.shape[0]}  "
              f"||eri||_F={np.linalg.norm(eri):.4f}")
        if args.leaf_sweep:
            leaf_counts = [int(x) for x in args.leaves_list.split(",")]
            run_leaf_sweep(eri, leaf_counts, args.rho, args.max_iterations,
                           args.repeats)
        elif args.tol_sweep:
            run_tol_sweep(eri, args.leaves, args.repeats)
        else:
            run_default(eri, args.leaves, args.rho, args.tol, args.repeats,
                        args.ab)


if __name__ == "__main__":
    main()
