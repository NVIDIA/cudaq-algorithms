# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for explicit (X-DF) and compressed (C-DF) double factorization."""
import os

import numpy as np
import pytest

import cudaq_algorithms as algorithms

os.environ.setdefault("OMP_NUM_THREADS", "1")

df = algorithms.double_factorization

# Run every test on the NumPy backend, plus the CuPy (GPU) backend when present.
_BACKENDS = ["numpy"]
if df.cupy_gpu_available():
    _BACKENDS.append("cupy")


def _synthetic_eri(n, num_vectors, seed):
    """An 8-fold-symmetric chemist-notation ERI of known low rank."""
    rng = np.random.default_rng(seed)
    leaves = rng.standard_normal((num_vectors, n, n))
    leaves = 0.5 * (leaves + leaves.transpose(0, 2, 1))
    return np.einsum("xpq,xrs->pqrs", leaves, leaves)


def _h2o_eri():
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.957; H 0 0.926 -0.24",
                basis="sto-3g",
                verbose=0)
    mf = scf.RHF(mol).run()
    n = mf.mo_coeff.shape[1]
    return np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))


def _max_orthogonality_error(factorization):
    return max((float(
        np.linalg.norm(u.T @ u - np.eye(u.shape[0])))
                for u in factorization.leaf_rotations),
               default=0.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_explicit_double_factorization_is_exact_at_full_rank(backend):
    eri = _h2o_eri()
    factorization = df.explicit_double_factorization(eri,
                                                     eigenvalue_threshold=0.0,
                                                     backend=backend)
    assert factorization.method == "X-DF"
    assert df.factorization_error(eri, factorization) < 1.0e-9
    assert _max_orthogonality_error(factorization) < 1.0e-10
    # Cores are symmetric.
    for core in factorization.leaf_cores:
        assert np.allclose(core, core.T, atol=1.0e-12)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_explicit_double_factorization_truncation_monotone(backend):
    eri = _h2o_eri()
    full = df.explicit_double_factorization(eri, eigenvalue_threshold=0.0,
                                            backend=backend)
    coarse = df.explicit_double_factorization(eri, eigenvalue_threshold=1.0e-2,
                                              backend=backend)
    fine = df.explicit_double_factorization(eri, eigenvalue_threshold=1.0e-4,
                                            backend=backend)
    # More aggressive truncation -> fewer leaves and larger (or equal) error.
    assert coarse.num_leaves <= fine.num_leaves <= full.num_leaves
    assert (df.factorization_error(eri, coarse) >=
            df.factorization_error(eri, fine) - 1.0e-12)
    # max_num_leaves caps the leaf count.
    capped = df.explicit_double_factorization(eri, eigenvalue_threshold=0.0,
                                              max_num_leaves=3, backend=backend)
    assert capped.num_leaves == 3


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_double_factorization_beats_explicit(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=7)
    for num_leaves in (1, 2):
        explicit = df.explicit_double_factorization(eri,
                                                    eigenvalue_threshold=0.0,
                                                    max_num_leaves=num_leaves,
                                                    backend=backend)
        compressed = df.compressed_double_factorization(eri,
                                                        num_leaves=num_leaves,
                                                        max_iterations=500,
                                                        backend=backend)
        assert compressed.method == "C-DF"
        assert compressed.num_leaves == num_leaves
        # Compression is never worse than the explicit factorization.
        assert (df.factorization_error(eri, compressed) <=
                df.factorization_error(eri, explicit) + 1.0e-6)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_double_factorization_exact_at_true_rank(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    compressed = df.compressed_double_factorization(eri, num_leaves=3,
                                                    max_iterations=1500,
                                                    backend=backend)
    assert df.factorization_error(eri, compressed) < 1.0e-4
    assert _max_orthogonality_error(compressed) < 1.0e-9


def test_reconstruct_eri_matches_helper():
    eri = _synthetic_eri(n=4, num_vectors=2, seed=3)
    factorization = df.explicit_double_factorization(eri,
                                                     eigenvalue_threshold=0.0)
    np.testing.assert_allclose(df.reconstruct_eri(factorization),
                               factorization.reconstruct_eri())


def test_modified_one_body_integrals():
    eri = _synthetic_eri(n=4, num_vectors=2, seed=5)
    one_body = np.diag([0.1, -0.2, 0.3, -0.4])
    kappa = df.modified_one_body_integrals(one_body, eri)
    expected = one_body - 0.5 * np.einsum("prqr->pq", eri)
    np.testing.assert_allclose(kappa, expected)
    assert np.allclose(kappa, kappa.T)


def test_double_factorization_one_norm_nonnegative_and_additive():
    eri = _synthetic_eri(n=4, num_vectors=3, seed=9)
    factorization = df.explicit_double_factorization(eri,
                                                     eigenvalue_threshold=0.0)
    one_body_eigenvalues = np.array([0.5, -1.0, 0.25, 0.75])
    lam = df.double_factorization_one_norm(factorization, one_body_eigenvalues)
    assert lam >= float(np.sum(np.abs(one_body_eigenvalues)))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        df.explicit_double_factorization(np.zeros((3, 3)))  # not rank-4
    with pytest.raises(ValueError):
        df.compressed_double_factorization(_synthetic_eri(4, 2, 1),
                                           num_leaves=0)
