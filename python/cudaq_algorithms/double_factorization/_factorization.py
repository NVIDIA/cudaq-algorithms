# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Explicit (X-DF) and compressed (C-DF) double factorization.

Follows Cohn, Motta, and Parrish, "Quantum Filter Diagonalization with
Compressed Double-Factorized Hamiltonians," PRX Quantum 2, 040352 (2021)
(arXiv:2104.08957).

The two-electron integrals are taken in chemist notation ``(pq|rs)`` over real
spatial orbitals (8-fold symmetry), i.e. an ``(n, n, n, n)`` array ``eri`` with
``eri[p, q, r, s] == (pq|rs)``. Double factorization writes

    (pq|rs) ~= sum_t sum_kl U^t_pk U^t_qk Z^t_kl U^t_rl U^t_sl

with orthogonal "leaf" rotations ``U^t`` (the Givens-rotation fabric) and
symmetric "core" matrices ``Z^t``. X-DF obtains these from nested
eigendecompositions (each ``Z^t`` is rank one). C-DF instead minimizes the
least-squares reconstruction error, allowing general symmetric ``Z^t`` and far
fewer leaves at equal accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.linalg
import scipy.optimize

from ._backend import (AUTO_GPU_MIN_ORBITALS_COMPRESSED,
                       AUTO_GPU_MIN_ORBITALS_EXPLICIT,
                       expm_skew_symmetric_batched, resolve_backend, to_device,
                       to_numpy)


@dataclass
class DoubleFactorization:
    """Result of a double factorization of the two-electron integrals.

    ``leaf_rotations[t]`` is the orthogonal matrix ``U^t`` (shape
    ``(num_orbitals, num_orbitals)``) and ``leaf_cores[t]`` is the symmetric
    core ``Z^t`` for leaf ``t``. ``method`` is ``"X-DF"`` or ``"C-DF"``.
    """

    num_orbitals: int
    leaf_rotations: List[np.ndarray]
    leaf_cores: List[np.ndarray]
    method: str
    first_factorization: Optional[
        str] = None  # "cholesky" or "eigendecomposition"
    leaf_weights: Optional[
        np.ndarray] = None  # first-factor pivots/eigenvalues

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_rotations)

    def reconstruct_eri(self) -> np.ndarray:
        """Reconstruct the ``(n, n, n, n)`` chemist-notation ERI tensor."""
        n = self.num_orbitals
        eri = np.zeros((n, n, n, n), dtype=float)
        for rotation, core in zip(self.leaf_rotations, self.leaf_cores):
            eri += np.einsum("pk,qk,kl,rl,sl->pqrs",
                             rotation,
                             rotation,
                             core,
                             rotation,
                             rotation,
                             optimize=True)
        return eri


def _sorted_symmetric_eigendecomposition(matrix, xp):
    """Eigenpairs of a symmetric matrix, ordered by descending |eigenvalue|."""
    eigenvalues, eigenvectors = xp.linalg.eigh(matrix)
    order = xp.argsort(xp.abs(eigenvalues))[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _pivoted_cholesky(matrix, threshold, max_rank, xp):
    """Pivoted Cholesky factorization of a symmetric positive-semidefinite
    matrix: returns ``(vectors, pivots)`` with ``matrix ~= sum_t v_t v_t^T``.

    The largest residual-diagonal entry is pivoted on at each step, so the
    factorization is rank-revealing and stops once that pivot drops to or below
    ``threshold`` (or after ``max_rank`` vectors). This is the standard
    Cholesky/density-fitting decomposition of the (PSD) electron-repulsion
    integrals; non-positive residual pivots (numerical null space) terminate it.
    """
    m = matrix.shape[0]
    residual = xp.real(xp.diag(matrix)).astype(float).copy()
    initial_max = float(xp.max(residual)) if m else 0.0
    # Rank-revealing floor: stop once the residual pivot reaches the numerical
    # null space. Without this, threshold == 0 would keep extracting vectors with
    # vanishing (eventually denormal) pivots, and dividing by sqrt(pivot) yields
    # garbage rotations.
    floor = max(float(threshold), initial_max * 1.0e-14)
    vectors: List = []
    pivots: List[float] = []
    limit = m if max_rank is None else min(int(max_rank), m)
    for _ in range(limit):
        pivot_index = int(xp.argmax(residual))
        pivot = float(residual[pivot_index])
        if pivot <= floor:
            break
        column = matrix[:, pivot_index].astype(float).copy()
        for previous in vectors:
            column = column - previous * float(previous[pivot_index])
        vector = column / xp.sqrt(xp.asarray(pivot))
        vectors.append(vector)
        pivots.append(pivot)
        residual = residual - vector * vector
        residual = xp.where(residual > 0.0, residual, 0.0)
    return vectors, pivots


def _second_factorization(leaf, scale, second_factor_threshold, xp):
    """Eigendecompose a symmetric leaf ``L = U diag(gamma) U^T`` and return
    ``(U, Z)`` with the symmetric core ``Z = scale * outer(gamma, gamma)``."""
    leaf = 0.5 * (leaf + leaf.T)
    gamma, rotation = xp.linalg.eigh(leaf)
    if second_factor_threshold > 0.0:
        importance = xp.sum(xp.abs(gamma))
        gamma = xp.where(importance * xp.abs(gamma) > second_factor_threshold,
                         gamma, 0.0)
    core = scale * xp.outer(gamma, gamma)
    return to_numpy(rotation), to_numpy(core)


def _validate_eri(eri) -> int:
    eri = np.asarray(eri)
    if eri.ndim != 4 or len(set(eri.shape)) != 1:
        raise ValueError(
            "double_factorization error - eri must be a square rank-4 tensor "
            "(n, n, n, n) in chemist notation (pq|rs).")
    return eri.shape[0]


def explicit_double_factorization(
        eri,
        threshold: float = 1.0e-8,
        max_num_leaves: Optional[int] = None,
        second_factor_threshold: float = 0.0,
        first_factorization: str = "cholesky",
        backend: str = "auto") -> DoubleFactorization:
    """Explicit double factorization (X-DF).

    First factorization of the ERI supermatrix ``(pq|rs) = sum_t L^t_pq L^t_rs``
    into symmetric leaves ``L^t``:

    * ``first_factorization="cholesky"`` (default) -- pivoted Cholesky of the
      positive-semidefinite ERI matrix. Rank-revealing: it keeps leaves while the
      residual-diagonal pivot exceeds ``threshold`` (the numerical null space
      terminates it), so it stops at the true factorization rank.
    * ``first_factorization="eigendecomposition"`` -- symmetric eigendecomposition
      ``(pq|rs) = sum_t lambda_t V^t_pq V^t_rs`` keeping ``|lambda_t| > threshold``.
      Required for indefinite inputs; the ERI is PSD so Cholesky is preferred.

    ``max_num_leaves`` caps the leaf count. Second factorization: each symmetric
    leaf is eigendecomposed, ``L^t = U^t diag(gamma^t) (U^t)^T``, giving the
    rank-one core ``Z^t = outer(gamma^t, gamma^t)`` (scaled by ``lambda_t`` in the
    eigendecomposition case). ``second_factor_threshold`` optionally zeros small
    ``gamma^t_k`` (importance-weighted, matching OpenFermion's convention).

    Returns a :class:`DoubleFactorization` with NumPy arrays.
    """
    n = _validate_eri(eri)
    xp, _ = resolve_backend(backend,
                            problem_size=n,
                            gpu_min_size=AUTO_GPU_MIN_ORBITALS_EXPLICIT)
    eri_dev = to_device(np.asarray(eri, dtype=float), xp)
    supermatrix = eri_dev.reshape(n * n, n * n)
    supermatrix = 0.5 * (supermatrix + supermatrix.T)

    rotations: List[np.ndarray] = []
    cores: List[np.ndarray] = []
    weights: List[float] = []

    if first_factorization == "cholesky":
        vectors, pivots = _pivoted_cholesky(supermatrix, threshold,
                                            max_num_leaves, xp)
        for vector, pivot in zip(vectors, pivots):
            rotation, core = _second_factorization(vector.reshape(n, n), 1.0,
                                                   second_factor_threshold, xp)
            rotations.append(rotation)
            cores.append(core)
            weights.append(pivot)
    elif first_factorization == "eigendecomposition":
        eigenvalues, eigenvectors = _sorted_symmetric_eigendecomposition(
            supermatrix, xp)
        abs_eigenvalues = to_numpy(xp.abs(eigenvalues))
        for index in range(n * n):
            if abs_eigenvalues[index] <= threshold:
                break
            if max_num_leaves is not None and len(rotations) >= max_num_leaves:
                break
            lam = eigenvalues[index]
            rotation, core = _second_factorization(
                eigenvectors[:, index].reshape(n, n), lam,
                second_factor_threshold, xp)
            rotations.append(rotation)
            cores.append(core)
            weights.append(float(to_numpy(lam)))
    else:
        raise ValueError(
            "double_factorization error - first_factorization must be "
            "'cholesky' or 'eigendecomposition'.")

    return DoubleFactorization(num_orbitals=n,
                               leaf_rotations=rotations,
                               leaf_cores=cores,
                               method="X-DF",
                               first_factorization=first_factorization,
                               leaf_weights=np.asarray(weights))


def _leaf_outer_columns(rotation, xp):
    """Return ``A`` with ``A[:, k] = vec(u_k u_k^T)`` for rotation columns u_k."""
    n = rotation.shape[0]
    outer = xp.einsum("pk,qk->pqk", rotation, rotation)
    return outer.reshape(n * n, n)


def _project_eri_into_leaf(eri_dev, rotation, xp):
    """R^t_kl = sum_pqrs U^t_pk U^t_qk (pq|rs) U^t_rl U^t_sl (the RHS of the inner
    normal equations / projection of the ERIs into a leaf's rotated basis)."""
    a = xp.einsum("pk,qk->pqk", rotation, rotation)
    return xp.einsum("pqk,pqrs,rsl->kl", a, eri_dev, a, optimize=True)


def _solve_inner_cores_cg(eri_dev,
                          rotations_dev,
                          xp,
                          regularization=0.0,
                          tolerance=1.0e-10,
                          max_iterations=None,
                          initial_guess=None):
    """Matrix-free conjugate-gradient solve for the symmetric cores ``{Z^t}``.

    Solves the same normal equations as the least-squares solver,
    ``(A + 2 rho I) Z = R``, but never forms the ``n^4 x num_params`` design
    matrix (RC-DF Eqs. 25-30). The operator factorizes as

        A(Z)^t = sum_{t'} M_{tt'} Z^{t'} M_{tt'}^T ,
        M_{tt'} = (U^t^T U^{t'}) elementwise-squared      (Eq. 27)

    so each matvec is just ``n x n`` matrix products; the ERIs are contracted
    only once to build the RHS ``R^t``. The cores are stacked into a single
    ``(num_leaves, n, n)`` array so the whole operator is one batched ``einsum``
    (a strided-batched GEMM under cuBLAS) rather than ``num_leaves^2`` Python-
    driven launches, and the CG scalars stay on device -- only the convergence and
    curvature guards sync to host. This scales to large orbital counts where the
    explicit design matrix is infeasible.

    ``initial_guess`` (a stacked ``(num_leaves, n, n)`` array) warm-starts CG. In
    the C-DF optimizer the rotations -- and hence the cores -- change slowly
    between L-BFGS steps, so seeding from the previous step's solution collapses
    the iteration count.
    """
    n = eri_dev.shape[0]
    num_leaves = len(rotations_dev)

    # M_{tt'} = (U^t^T U^{t'}) elementwise-squared, stacked as (t, t', k, l).
    rotations = xp.stack(rotations_dev)
    metric = xp.einsum("tpk,upl->tukl", rotations, rotations)
    metric = metric * metric
    rhs = xp.stack(
        [_project_eri_into_leaf(eri_dev, u, xp) for u in rotations_dev])

    def apply_operator(z):
        # A(Z)^t = sum_{t'} M_{tt'} Z^{t'} M_{tt'}^T, all leaves at once.
        out = xp.einsum("tukl,ulm,tunm->tkn", metric, z, metric, optimize=True)
        if regularization > 0.0:
            out = out + (2.0 * regularization) * z
        return out

    def inner_product(x, y):
        # A device scalar (0-d array); converted to host only where needed.
        return xp.sum(x * y)

    if initial_guess is None:
        z = xp.zeros((num_leaves, n, n))
        residual = rhs.copy()  # r = rhs - A(0) = rhs since z = 0
    else:
        z = xp.asarray(initial_guess).copy()
        residual = rhs - apply_operator(z)
    direction = residual.copy()
    residual_norm_sq = inner_product(residual, residual)
    threshold = (tolerance**2) * max(float(to_numpy(residual_norm_sq)), 1.0)
    limit = max_iterations if max_iterations is not None else max(
        50, num_leaves * n * n)

    for _ in range(limit):
        if float(to_numpy(residual_norm_sq)) <= threshold:
            break
        operator_direction = apply_operator(direction)
        curvature = inner_product(direction, operator_direction)
        if float(to_numpy(curvature)) <= 0.0:
            break
        step = residual_norm_sq / curvature  # device scalar
        z = z + step * direction
        residual = residual - step * operator_direction
        new_norm_sq = inner_product(residual, residual)
        beta = new_norm_sq / residual_norm_sq  # device scalar
        direction = residual + beta * direction
        residual_norm_sq = new_norm_sq

    z = 0.5 * (z + z.transpose(0, 2, 1))
    return [z[t] for t in range(num_leaves)]


def _solve_inner_cores_lstsq(eri_dev, rotations_dev, xp, regularization=0.0):
    """Least-squares optimal symmetric cores ``{Z^t}`` for fixed rotations.

    Solves, for fixed ``U^t``,

        min_Z  1/2 || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F^2
                  + rho * sum_{t,k,l} (Z^t_kl)^2

    exactly (linear in the symmetric ``Z^t``) via an explicit design matrix. The
    ``rho`` (``regularization``) term is the RC-DF L2 penalty
    (arXiv:2212.07957, Eq. 17); it shrinks the cores and conditions the otherwise
    rank-deficient inner system. See ``_solve_inner_cores_cg`` for the matrix-free
    alternative that scales to large orbital counts.
    """
    n = eri_dev.shape[0]
    target = eri_dev.reshape(-1)

    columns = []
    metadata = []  # (leaf, k, l) with k <= l
    for leaf_index, rotation in enumerate(rotations_dev):
        a = _leaf_outer_columns(rotation, xp)  # (n*n, n)
        for k in range(n):
            for l in range(k, n):
                column = xp.outer(a[:, k], a[:, l])
                if l != k:
                    column = column + xp.outer(a[:, l], a[:, k])
                columns.append(column.reshape(-1))
                metadata.append((leaf_index, k, l))

    design = xp.stack(columns, axis=1)

    if regularization > 0.0:
        # Augmented rows enforce the L2 penalty rho * ||Z||_F^2 (full k, l). For
        # the k <= l parameterization, off-diagonal entries appear twice in
        # ||Z||_F^2, so they carry weight 2 (diagonal weight 1). The factor 2
        # below comes from the 1/2 on the least-squares term in Eq. 17.
        weights = xp.asarray(
            [1.0 if k == l else 2.0 for (_, k, l) in metadata])
        penalty = xp.sqrt(2.0 * regularization * weights)
        design = xp.concatenate([design, xp.diag(penalty)], axis=0)
        target = xp.concatenate([target, xp.zeros(penalty.shape[0])], axis=0)

    solution = xp.linalg.lstsq(design, target, rcond=None)[0]

    cores = [xp.zeros((n, n)) for _ in rotations_dev]
    for index, (leaf_index, k, l) in enumerate(metadata):
        value = solution[index]
        cores[leaf_index][k, l] = value
        cores[leaf_index][l, k] = value
    return cores


def _solve_inner_cores(eri_dev,
                       rotations_dev,
                       xp,
                       regularization=0.0,
                       solver="lstsq",
                       cg_tolerance=1.0e-10,
                       cg_max_iterations=None,
                       initial_guess=None):
    """Dispatch the inner core solve to the explicit ``"lstsq"`` solver or the
    matrix-free ``"cg"`` solver. ``initial_guess`` warm-starts CG and is ignored
    by the direct lstsq solver."""
    if solver == "lstsq":
        return _solve_inner_cores_lstsq(eri_dev, rotations_dev, xp,
                                        regularization)
    if solver == "cg":
        return _solve_inner_cores_cg(eri_dev, rotations_dev, xp,
                                     regularization, cg_tolerance,
                                     cg_max_iterations, initial_guess)
    raise ValueError(
        "double_factorization error - inner_solver must be 'lstsq' or 'cg'.")


def _reconstruct_dev(rotations_dev, cores_dev, xp):
    # Stack the leaves and reconstruct in a single batched contraction (the sum
    # over leaves t is folded into the einsum), instead of a Python loop with a
    # separate n^4 einsum and accumulation per leaf.
    rotations = rotations_dev if _is_stacked(rotations_dev) else xp.stack(
        list(rotations_dev))
    cores = cores_dev if _is_stacked(cores_dev) else xp.stack(list(cores_dev))
    return xp.einsum("tpk,tqk,tkl,trl,tsl->pqrs",
                     rotations,
                     rotations,
                     cores,
                     rotations,
                     rotations,
                     optimize=True)


def _is_stacked(arrays):
    """True for a single stacked ``(num_leaves, n, n)`` array, False for a list
    of ``(n, n)`` leaf arrays."""
    return hasattr(arrays, "ndim") and arrays.ndim == 3


def _skew_to_vector(generator, n):
    rows, cols = np.tril_indices(n, k=-1)
    return generator[rows, cols]


def _vector_to_skew(vector, n, xp):
    generator = xp.zeros((n, n))
    rows, cols = np.tril_indices(n, k=-1)
    rows = xp.asarray(rows)
    cols = xp.asarray(cols)
    generator[rows, cols] = vector
    generator[cols, rows] = -vector
    return generator


def _initial_generators(eri, num_leaves, n, backend):
    """Warm-start antisymmetric generators from a truncated X-DF (identity pads
    any missing leaves)."""
    xdf = explicit_double_factorization(eri,
                                        threshold=0.0,
                                        max_num_leaves=num_leaves,
                                        backend=backend)
    generators = []
    for rotation in xdf.leaf_rotations:
        rotation = np.asarray(rotation, dtype=float).copy()
        # exp(antisymmetric) is always special orthogonal (det +1), but eigh
        # leaf rotations can be reflections (det -1), which have no real
        # antisymmetric logarithm. Flipping one column flips the determinant
        # without changing u_k u_k^T (hence without changing the factorization),
        # so force det +1 before taking the logarithm.
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] *= -1.0
        skew = scipy.linalg.logm(rotation).real
        skew = 0.5 * (skew - skew.T)  # clean numerical asymmetry
        generators.append(skew)
    while len(generators) < num_leaves:
        generators.append(np.zeros((n, n)))
    return generators[:num_leaves]


def compressed_double_factorization(
        eri,
        num_leaves: int,
        max_iterations: int = 2000,
        tolerance: float = 1.0e-10,
        regularization: float = 0.0,
        inner_solver: str = "lstsq",
        cg_tolerance: float = 1.0e-10,
        cg_max_iterations: Optional[int] = None,
        cg_warm_start: bool = True,
        cg_optimization_tolerance: Optional[float] = None,
        initial_generators: Optional[List[np.ndarray]] = None,
        backend: str = "auto") -> DoubleFactorization:
    """Compressed double factorization (C-DF) by least-squares optimization.

    Minimizes ``O = 1/2 || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F^2`` over
    a fixed number of leaves. Uses the two-step scheme of arXiv:2104.08957: the
    leaf rotations are parameterized as ``U^t = exp(X^t)`` with antisymmetric
    ``X^t`` and optimized with L-BFGS (warm-started from X-DF), while the
    symmetric cores ``Z^t`` are solved exactly in closed form at each step.

    ``regularization`` (``rho``) enables RC-DF (arXiv:2212.07957, Eq. 17): the
    L2 penalty ``rho * sum_{t,k,l} (Z^t_kl)^2`` is added to the objective and,
    crucially, folded into the inner core solve as a ridge term. It shrinks the
    cores -- lowering the Hamiltonian one-norm ``lambda`` and the measurement
    variance -- and conditions the inner system. ``rho`` is an absolute
    coefficient (its useful scale depends on the integral magnitude; the paper
    uses ~1e-6 to 1e-3). ``rho = 0`` reproduces plain C-DF.

    ``inner_solver`` selects how the cores are solved each step: ``"lstsq"``
    (default) forms an explicit design matrix, while ``"cg"`` is the matrix-free
    conjugate-gradient solve (RC-DF Eqs. 25-30) that avoids the ``n^4``-row design
    matrix and scales to large orbital counts. ``cg_tolerance`` and
    ``cg_max_iterations`` control the CG solve.

    For ``inner_solver="cg"`` two accelerators cut the per-step CG cost without
    changing the final accuracy: ``cg_warm_start`` (default ``True``) seeds each
    step's CG from the previous step's cores -- which move slowly between L-BFGS
    steps -- and ``cg_optimization_tolerance`` (default ``max(cg_tolerance,
    1e-6)``) solves the *in-loop* systems only loosely (an inexact inner solve;
    the gradient need only be approximate by the envelope theorem) while the
    single final solve is tightened to ``cg_tolerance``.

    Returns a :class:`DoubleFactorization` with NumPy arrays.
    """
    n = _validate_eri(eri)
    if num_leaves < 1:
        raise ValueError(
            "double_factorization error - num_leaves must be >= 1.")
    if inner_solver not in ("lstsq", "cg"):
        raise ValueError(
            "double_factorization error - inner_solver must be 'lstsq' or 'cg'."
        )
    xp, _ = resolve_backend(backend,
                            problem_size=n,
                            gpu_min_size=AUTO_GPU_MIN_ORBITALS_COMPRESSED)
    eri_dev = to_device(np.asarray(eri, dtype=float), xp)

    if initial_generators is None:
        initial_generators = _initial_generators(eri, num_leaves, n, backend)

    x0 = np.concatenate(
        [_skew_to_vector(np.asarray(g), n) for g in initial_generators])

    per_leaf = n * (n - 1) // 2
    lower = np.tril_indices(n, k=-1)

    # Inexact in-loop CG (forcing sequence) + warm start across L-BFGS steps.
    use_warm = inner_solver == "cg" and cg_warm_start
    loop_tolerance = (cg_optimization_tolerance if cg_optimization_tolerance
                      is not None else max(cg_tolerance, 1.0e-6))
    warm_state = {"z": None}

    def unpack(parameter_vector):
        # Build all leaf generators, then exponentiate them in one batched
        # Hermitian eigendecomposition (single cuSOLVER call on GPU). Returns
        # stacked (num_leaves, n, n) arrays.
        skews = xp.stack([
            _vector_to_skew(
                to_device(parameter_vector[i * per_leaf:(i + 1) * per_leaf],
                          xp), n, xp) for i in range(num_leaves)
        ])
        rotations = expm_skew_symmetric_batched(skews, xp)
        return skews, rotations

    def objective_and_gradient(parameter_vector):
        # Two-step objective O(X) with the cores Z at their (regularized)
        # least-squares optimum. Because dO/dZ = 0 there, the envelope theorem
        # gives dO/dX = (dO/dU at fixed Z) chained through dU/dX; the L2 penalty
        # has no explicit X-dependence, so the gradient keeps the same form for
        # any regularization. dO/dU^t_ak = -4 sum_qrsl Delta_aqrs U_qk Z_kl U_rl
        # U_sl (Eq. 17 of arXiv:2104.08957).
        skews, rotations = unpack(parameter_vector)
        cores = _solve_inner_cores(eri_dev, rotations, xp, regularization,
                                   inner_solver, loop_tolerance,
                                   cg_max_iterations,
                                   warm_state["z"] if use_warm else None)
        if use_warm:
            warm_state["z"] = xp.stack(cores)
        residual = eri_dev - _reconstruct_dev(rotations, cores, xp)
        loss = 0.5 * xp.sum(residual * residual)
        if regularization > 0.0:
            loss = loss + regularization * sum(
                xp.sum(core * core) for core in cores)
        loss = float(to_numpy(loss))

        # dO/dU for all leaves in one batched contraction, then a single
        # device->host transfer (the leaf index t is the batch axis).
        cores_dev = cores if _is_stacked(cores) else xp.stack(list(cores))
        grad_u_all = to_numpy(-4.0 * xp.einsum("aqrs,tqk,tkl,trl,tsl->tak",
                                               residual,
                                               rotations,
                                               cores_dev,
                                               rotations,
                                               rotations,
                                               optimize=True))
        skews_host = to_numpy(skews)
        gradient_chunks = []
        for leaf_index in range(num_leaves):
            # Adjoint of d(exp)/dX: grad_X = L_exp(X^T, grad_U) (Frechet
            # derivative), then project onto the antisymmetric parameters. The
            # Frechet step stays on the host (it is ~2% of the step cost).
            grad_x = scipy.linalg.expm_frechet(skews_host[leaf_index].T,
                                               grad_u_all[leaf_index],
                                               compute_expm=False)
            grad_x = grad_x - grad_x.T
            gradient_chunks.append(grad_x[lower])
        return loss, np.concatenate(gradient_chunks)

    result = scipy.optimize.minimize(objective_and_gradient,
                                     x0,
                                     method="L-BFGS-B",
                                     jac=True,
                                     options={
                                         "maxiter": max_iterations,
                                         "ftol": tolerance,
                                         "gtol": tolerance,
                                     })

    # Final cores at the tight cg_tolerance (warm-started from the last step).
    _, rotations = unpack(result.x)
    cores = _solve_inner_cores(eri_dev, rotations, xp, regularization,
                               inner_solver, cg_tolerance, cg_max_iterations,
                               warm_state["z"] if use_warm else None)
    return DoubleFactorization(num_orbitals=n,
                               leaf_rotations=[to_numpy(r) for r in rotations],
                               leaf_cores=[to_numpy(c) for c in cores],
                               method="C-DF")


def reconstruct_eri(factorization: DoubleFactorization) -> np.ndarray:
    """Reconstruct the chemist-notation ERI tensor from a factorization."""
    return factorization.reconstruct_eri()


def factorization_error(eri, factorization: DoubleFactorization) -> float:
    """Frobenius norm of the ERI reconstruction residual."""
    return float(
        np.linalg.norm(
            np.asarray(eri, dtype=float) - factorization.reconstruct_eri()))


def modified_one_body_integrals(one_body, eri) -> np.ndarray:
    """Return the DF-corrected one-body matrix ``kappa_pq = h_pq - 1/2 sum_r
    (pr|qr)`` (Eq. 3 of arXiv:2104.08957), used when assembling the full
    double-factorized Hamiltonian from the two-body factorization."""
    one_body = np.asarray(one_body, dtype=float)
    eri = np.asarray(eri, dtype=float)
    return one_body - 0.5 * np.einsum("prqr->pq", eri, optimize=True)


def double_factorization_one_norm(factorization: DoubleFactorization,
                                  one_body_eigenvalues,
                                  convention: str = "lcu") -> float:
    """One-norm ``lambda`` of the double-factorized Hamiltonian (RC-DF,
    arXiv:2212.07957), used to assess factorization quality.

    ``convention="lcu"`` (Eq. 13) -- the LCU / Pauli-rotation norm
    ``sum_k |F_k| + sum_t (sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk|)``.

    ``convention="burg"`` (Eq. 15) -- the qubitization norm
    ``sum_k |F_k| + 1/4 sum_t sum_i (sum_k |W^t_ki|)^2`` with ``W^t = sqrt(Z^t)``
    (the matrix square root, which may be complex for indefinite cores).

    ``one_body_eigenvalues`` are the diagonal one-body (Fock-like) eigenvalues.
    """
    one_body_eigenvalues = np.asarray(one_body_eigenvalues, dtype=float)
    lam = float(np.sum(np.abs(one_body_eigenvalues)))

    if convention == "lcu":
        for core in factorization.leaf_cores:
            core = np.asarray(core)
            # sum_{k<l} |Z_kl| (strict upper triangle) + 1/4 sum_k |Z_kk|
            lam += float(np.sum(np.abs(np.triu(core, k=1))))
            lam += 0.25 * float(np.sum(np.abs(np.diag(core))))
        return lam

    if convention == "burg":
        for core in factorization.leaf_cores:
            eigenvalues, vectors = np.linalg.eigh(np.asarray(core,
                                                             dtype=float))
            root = (vectors * np.sqrt(eigenvalues.astype(complex))) @ \
                vectors.conj().T  # W = sqrt(Z), possibly complex
            column_abs_sum = np.sum(np.abs(root), axis=0)  # sum_k |W_ki| per i
            lam += 0.25 * float(np.sum(column_abs_sum**2))
        return lam

    raise ValueError(
        "double_factorization error - convention must be 'lcu' or 'burg'.")
