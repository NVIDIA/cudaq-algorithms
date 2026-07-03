# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq_algorithms as algorithms


def test_required_chebyshev_moments():
    assert algorithms.krylov.required_chebyshev_moments(0) == 0
    assert algorithms.krylov.required_chebyshev_moments(1) == 2
    assert algorithms.krylov.required_chebyshev_moments(4) == 8


def test_build_chebyshev_matrices():
    matrices = algorithms.krylov.build_chebyshev_matrices(
        [1.0, 0.2, -0.1, 0.05], 2)

    assert matrices.dimension == 2
    assert np.allclose(matrices.overlap_matrix(), [[1.0, 0.2], [0.2, 0.45]])
    assert np.allclose(matrices.hamiltonian_matrix(),
                       [[0.2, 0.45], [0.45, 0.1625]])
    assert matrices.overlap_data == pytest.approx([1.0, 0.2, 0.2, 0.45])
    assert matrices.hamiltonian_data == pytest.approx(
        [0.2, 0.45, 0.45, 0.1625])


def test_build_chebyshev_matrices_validation():
    with pytest.raises(RuntimeError):
        algorithms.krylov.build_chebyshev_matrices([1.0, 0.0], 0)
    with pytest.raises(RuntimeError):
        algorithms.krylov.build_chebyshev_matrices([1.0, 0.0, 0.0], 2)


# Test purpose: defense-in-depth reference that is independent of the C++
# implementation's own moment-combination formulas.
def test_chebyshev_matrices_reproduce_exact_eigenvalues():
    """Moments from an explicit Hermitian matrix must recover its spectrum.

    The C++ unit test checks build_chebyshev_matrices against values produced
    by the same formulas it implements. Here the moments mu_k = v^T T_k(A) v
    come from an explicit matrix, and solving the generalized eigenproblem for
    a full-rank Krylov space must reproduce numpy's eigenvalues of A.
    """

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(3, 3))
    matrix = 0.5 * (raw + raw.T)
    # Chebyshev moments need the spectrum inside [-1, 1].
    matrix /= 1.25 * np.max(np.abs(np.linalg.eigvalsh(matrix)))

    vector = rng.normal(size=3)
    vector /= np.linalg.norm(vector)

    # dimension equals the matrix size, so the Chebyshev Krylov space
    # span{T_0(A)v, ..., T_{d-1}(A)v} is the full space for a generic v.
    dimension = 3
    num_moments = algorithms.krylov.required_chebyshev_moments(dimension)
    chebyshev = [np.eye(3), matrix.copy()]
    while len(chebyshev) < num_moments:
        chebyshev.append(2.0 * matrix @ chebyshev[-1] - chebyshev[-2])
    moments = [
        float(vector @ chebyshev[k] @ vector) for k in range(num_moments)
    ]

    matrices = algorithms.krylov.build_chebyshev_matrices(moments, dimension)
    overlap = np.asarray(matrices.overlap_matrix(), dtype=np.float64)
    hamiltonian = np.asarray(matrices.hamiltonian_matrix(), dtype=np.float64)

    # Solve H c = E S c by symmetric whitening with S^(-1/2).
    overlap_eigenvalues, overlap_vectors = np.linalg.eigh(overlap)
    assert overlap_eigenvalues.min() > 1e-8
    transform = overlap_vectors @ np.diag(overlap_eigenvalues**-0.5)
    eigenvalues = np.linalg.eigvalsh(transform.T @ hamiltonian @ transform)

    assert np.allclose(np.sort(eigenvalues),
                       np.sort(np.linalg.eigvalsh(matrix)),
                       atol=1e-8)
