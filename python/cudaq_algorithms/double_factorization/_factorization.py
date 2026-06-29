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

from ._backend import (expm_skew_symmetric, resolve_backend, to_device,
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
    first_factorization: Optional[str] = None  # "cholesky" or "eigendecomposition"
    leaf_weights: Optional[np.ndarray] = None  # first-factor pivots/eigenvalues

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
    xp, _ = resolve_backend(backend)
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


def _solve_inner_cores(eri_dev, rotations_dev, xp):
    """Least-squares optimal symmetric cores ``{Z^t}`` for fixed rotations.

    Solves ``min_Z || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F`` exactly
    (linear in the symmetric ``Z^t``) via a pseudoinverse / least squares, the
    inner step of the two-step C-DF scheme.
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
    solution = xp.linalg.lstsq(design, target, rcond=None)[0]

    cores = [xp.zeros((n, n)) for _ in rotations_dev]
    for index, (leaf_index, k, l) in enumerate(metadata):
        value = solution[index]
        cores[leaf_index][k, l] = value
        cores[leaf_index][l, k] = value
    return cores


def _reconstruct_dev(rotations_dev, cores_dev, xp):
    n = rotations_dev[0].shape[0]
    eri = xp.zeros((n, n, n, n))
    for rotation, core in zip(rotations_dev, cores_dev):
        eri = eri + xp.einsum("pk,qk,kl,rl,sl->pqrs",
                              rotation,
                              rotation,
                              core,
                              rotation,
                              rotation,
                              optimize=True)
    return eri


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
        initial_generators: Optional[List[np.ndarray]] = None,
        backend: str = "auto") -> DoubleFactorization:
    """Compressed double factorization (C-DF) by least-squares optimization.

    Minimizes ``O = 1/2 || eri - sum_t U^t Z^t (U^t)^T (congruence) ||_F^2`` over
    a fixed number of leaves. Uses the two-step scheme of arXiv:2104.08957: the
    leaf rotations are parameterized as ``U^t = exp(X^t)`` with antisymmetric
    ``X^t`` and optimized with L-BFGS (warm-started from X-DF), while the
    symmetric cores ``Z^t`` are solved exactly in closed form at each step.

    ``regularization`` adds an optional ``rho * sum_t ||Z^t||_F^2`` penalty
    (the RC-DF extension, arXiv:2212.07957); default 0 reproduces plain C-DF.

    Returns a :class:`DoubleFactorization` with NumPy arrays.
    """
    n = _validate_eri(eri)
    if num_leaves < 1:
        raise ValueError(
            "double_factorization error - num_leaves must be >= 1.")
    xp, _ = resolve_backend(backend)
    eri_dev = to_device(np.asarray(eri, dtype=float), xp)

    if initial_generators is None:
        initial_generators = _initial_generators(eri, num_leaves, n, backend)

    x0 = np.concatenate(
        [_skew_to_vector(np.asarray(g), n) for g in initial_generators])

    per_leaf = n * (n - 1) // 2
    lower = np.tril_indices(n, k=-1)

    def unpack(parameter_vector):
        skews, rotations = [], []
        for leaf_index in range(num_leaves):
            chunk = parameter_vector[leaf_index * per_leaf:(leaf_index + 1) *
                                     per_leaf]
            skew = _vector_to_skew(to_device(chunk, xp), n, xp)
            skews.append(skew)
            rotations.append(expm_skew_symmetric(skew, xp))
        return skews, rotations

    def objective(parameter_vector):
        _, rotations = unpack(parameter_vector)
        cores = _solve_inner_cores(eri_dev, rotations, xp)
        residual = eri_dev - _reconstruct_dev(rotations, cores, xp)
        loss = 0.5 * xp.sum(residual * residual)
        if regularization > 0.0:
            loss = loss + regularization * sum(
                xp.sum(core * core) for core in cores)
        return float(to_numpy(loss))

    def objective_and_gradient(parameter_vector):
        # Analytic gradient (regularization == 0 only): with the inner cores at
        # their least-squares optimum, dO/dZ = 0, so by the envelope theorem the
        # gradient is dO/dU chained through dU/dX (the matrix-exponential
        # derivative). dO/dU^t_ak = -4 sum_qrsl Delta_aqrs U_qk Z_kl U_rl U_sl
        # (Eq. 17 of arXiv:2104.08957).
        skews, rotations = unpack(parameter_vector)
        cores = _solve_inner_cores(eri_dev, rotations, xp)
        residual = eri_dev - _reconstruct_dev(rotations, cores, xp)
        loss = float(to_numpy(0.5 * xp.sum(residual * residual)))

        gradient_chunks = []
        for skew, rotation, core in zip(skews, rotations, cores):
            grad_u = -4.0 * xp.einsum("aqrs,qk,kl,rl,sl->ak",
                                      residual,
                                      rotation,
                                      core,
                                      rotation,
                                      rotation,
                                      optimize=True)
            # Adjoint of d(exp)/dX: grad_X = L_exp(X^T, grad_U) (Frechet
            # derivative), then project onto the antisymmetric parameters.
            grad_u_host = to_numpy(grad_u)
            skew_host = to_numpy(skew)
            grad_x = scipy.linalg.expm_frechet(skew_host.T,
                                               grad_u_host,
                                               compute_expm=False)
            grad_x = grad_x - grad_x.T
            gradient_chunks.append(grad_x[lower])
        return loss, np.concatenate(gradient_chunks)

    use_analytic = regularization == 0.0
    result = scipy.optimize.minimize(
        objective_and_gradient if use_analytic else objective,
        x0,
        method="L-BFGS-B",
        jac=use_analytic,
        options={
            "maxiter": max_iterations,
            "ftol": tolerance,
            "gtol": tolerance,
        })

    _, rotations = unpack(result.x)
    cores = _solve_inner_cores(eri_dev, rotations, xp)
    return DoubleFactorization(
        num_orbitals=n,
        leaf_rotations=[to_numpy(r) for r in rotations],
        leaf_cores=[to_numpy(c) for c in cores],
        method="C-DF")


def reconstruct_eri(factorization: DoubleFactorization) -> np.ndarray:
    """Reconstruct the chemist-notation ERI tensor from a factorization."""
    return factorization.reconstruct_eri()


def factorization_error(eri, factorization: DoubleFactorization) -> float:
    """Frobenius norm of the ERI reconstruction residual."""
    return float(
        np.linalg.norm(np.asarray(eri, dtype=float) -
                       factorization.reconstruct_eri()))


def modified_one_body_integrals(one_body, eri) -> np.ndarray:
    """Return the DF-corrected one-body matrix ``kappa_pq = h_pq - 1/2 sum_r
    (pr|qr)`` (Eq. 3 of arXiv:2104.08957), used when assembling the full
    double-factorized Hamiltonian from the two-body factorization."""
    one_body = np.asarray(one_body, dtype=float)
    eri = np.asarray(eri, dtype=float)
    return one_body - 0.5 * np.einsum("prqr->pq", eri, optimize=True)


def double_factorization_one_norm(factorization: DoubleFactorization,
                                  one_body_eigenvalues) -> float:
    """LCU one-norm ``lambda`` of the double-factorized Hamiltonian (RC-DF
    Eq. 13): ``sum_k |F_k| + sum_t (sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk|)``."""
    one_body_eigenvalues = np.asarray(one_body_eigenvalues, dtype=float)
    lam = float(np.sum(np.abs(one_body_eigenvalues)))
    for core in factorization.leaf_cores:
        core = np.asarray(core)
        # sum_{k<l} |Z_kl| (strict upper triangle) + 1/4 sum_k |Z_kk|
        lam += float(np.sum(np.abs(np.triu(core, k=1))))
        lam += 0.25 * float(np.sum(np.abs(np.diag(core))))
    return lam
