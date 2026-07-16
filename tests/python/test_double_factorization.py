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
    return max((float(np.linalg.norm(u.T @ u - np.eye(u.shape[0])))
                for u in factorization.leaf_rotations),
               default=0.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_explicit_double_factorization_is_exact_at_full_rank(backend):
    eri = _h2o_eri()
    factorization = df.explicit_double_factorization(eri,
                                                     threshold=0.0,
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
    full = df.explicit_double_factorization(eri,
                                            threshold=0.0,
                                            backend=backend)
    coarse = df.explicit_double_factorization(eri,
                                              threshold=1.0e-2,
                                              backend=backend)
    fine = df.explicit_double_factorization(eri,
                                            threshold=1.0e-4,
                                            backend=backend)
    # More aggressive truncation -> fewer leaves and larger (or equal) error.
    assert coarse.num_leaves <= fine.num_leaves <= full.num_leaves
    assert (df.factorization_error(eri, coarse)
            >= df.factorization_error(eri, fine) - 1.0e-12)
    # max_num_leaves caps the leaf count.
    capped = df.explicit_double_factorization(eri,
                                              threshold=0.0,
                                              max_num_leaves=3,
                                              backend=backend)
    assert capped.num_leaves == 3


@pytest.mark.parametrize("backend", _BACKENDS)
def test_pivoted_cholesky_is_default_and_rank_revealing(backend):
    eri = _h2o_eri()
    n = eri.shape[0]
    factorization = df.explicit_double_factorization(eri,
                                                     threshold=1.0e-12,
                                                     backend=backend)
    # Pivoted Cholesky is the default first factorization.
    assert factorization.first_factorization == "cholesky"
    # The ERI supermatrix is PSD, so Cholesky stops at the true rank
    # (<= the symmetric-pair dimension n(n+1)/2), not n^2.
    assert factorization.num_leaves <= n * (n + 1) // 2
    assert df.factorization_error(eri, factorization) < 1.0e-8


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cholesky_and_eigendecomposition_agree(backend):
    eri = _h2o_eri()
    cholesky = df.explicit_double_factorization(eri,
                                                threshold=0.0,
                                                first_factorization="cholesky",
                                                backend=backend)
    eigen = df.explicit_double_factorization(
        eri,
        threshold=0.0,
        first_factorization="eigendecomposition",
        backend=backend)
    assert eigen.first_factorization == "eigendecomposition"
    # Both reconstruct the same ERI exactly, regardless of first-factor method.
    assert df.factorization_error(eri, cholesky) < 1.0e-9
    assert df.factorization_error(eri, eigen) < 1.0e-9
    np.testing.assert_allclose(cholesky.reconstruct_eri(),
                               eigen.reconstruct_eri(),
                               atol=1.0e-9)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_double_factorization_beats_explicit(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=7)
    for num_leaves in (1, 2):
        explicit = df.explicit_double_factorization(eri,
                                                    threshold=0.0,
                                                    max_num_leaves=num_leaves,
                                                    backend=backend)
        compressed = df.compressed_double_factorization(eri,
                                                        num_leaves=num_leaves,
                                                        max_iterations=500,
                                                        backend=backend)
        assert compressed.method == "C-DF"
        assert compressed.num_leaves == num_leaves
        # Compression is never worse than the explicit factorization.
        assert (df.factorization_error(eri, compressed)
                <= df.factorization_error(eri, explicit) + 1.0e-6)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_double_factorization_exact_at_true_rank(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    compressed = df.compressed_double_factorization(eri,
                                                    num_leaves=3,
                                                    max_iterations=1500,
                                                    backend=backend)
    assert df.factorization_error(eri, compressed) < 1.0e-4
    assert _max_orthogonality_error(compressed) < 1.0e-9


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rcdf_regularization_shrinks_cores_and_one_norm(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=7)
    one_body = np.array([0.4, -0.3, 0.2, -0.1])

    plain = df.compressed_double_factorization(eri,
                                               num_leaves=2,
                                               max_iterations=800,
                                               regularization=0.0,
                                               backend=backend)
    regularized = df.compressed_double_factorization(eri,
                                                     num_leaves=2,
                                                     max_iterations=800,
                                                     regularization=1.0e-2,
                                                     backend=backend)

    def core_norm(factorization):
        return sum(float(np.linalg.norm(z)) for z in factorization.leaf_cores)

    # RC-DF shrinks the cores ...
    assert core_norm(regularized) < core_norm(plain)
    # ... lowering the Hamiltonian one-norm in both conventions ...
    for convention in ("lcu", "burg"):
        assert (df.double_factorization_one_norm(regularized,
                                                 one_body,
                                                 convention=convention)
                < df.double_factorization_one_norm(plain,
                                                   one_body,
                                                   convention=convention))
    # ... at the cost of reconstruction accuracy.
    assert (df.factorization_error(eri, regularized)
            >= df.factorization_error(eri, plain) - 1.0e-9)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compressed_matrix_free_cg_matches_lstsq(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    # The matrix-free conjugate-gradient inner solve (RC-DF Eqs. 25-30) is an
    # alternative to the explicit design-matrix least squares; both must reach
    # the same reconstruction accuracy, with and without regularization.
    for regularization in (0.0, 1.0e-2):
        lstsq = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=800,
            regularization=regularization,
            inner_solver="lstsq",
            backend=backend)
        cg = df.compressed_double_factorization(eri,
                                                num_leaves=2,
                                                max_iterations=800,
                                                regularization=regularization,
                                                inner_solver="cg",
                                                backend=backend)
        assert cg.method == "C-DF"
        assert abs(
            df.factorization_error(eri, cg) -
            df.factorization_error(eri, lstsq)) < 1.0e-6
    # CG is exact at the true rank, like the least-squares solver.
    exact = df.compressed_double_factorization(eri,
                                               num_leaves=3,
                                               max_iterations=1500,
                                               inner_solver="cg",
                                               backend=backend)
    assert df.factorization_error(eri, exact) < 1.0e-4
    with pytest.raises(ValueError):
        df.compressed_double_factorization(eri,
                                           num_leaves=2,
                                           inner_solver="bogus",
                                           backend=backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cg_accelerators_preserve_accuracy(backend):
    eri = _synthetic_eri(n=4, num_vectors=3, seed=11)
    # The warm-start + inexact-in-loop accelerators (defaults) must reach the
    # same final reconstruction as the cold-start, tight-in-loop CG: the single
    # final solve is always tightened to cg_tolerance.
    for regularization in (0.0, 1.0e-2):
        accel = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=800,
            regularization=regularization,
            inner_solver="cg",
            backend=backend)
        plain = df.compressed_double_factorization(
            eri,
            num_leaves=2,
            max_iterations=800,
            regularization=regularization,
            inner_solver="cg",
            cg_warm_start=False,
            cg_optimization_tolerance=1.0e-10,
            backend=backend)
        assert abs(
            df.factorization_error(eri, accel) -
            df.factorization_error(eri, plain)) < 1.0e-4


def test_auto_backend_is_size_aware():
    from cudaq_algorithms.double_factorization._backend import resolve_backend
    # Explicit choices are honored regardless of size.
    assert resolve_backend("numpy")[1] == "numpy"
    # A tiny problem under the GPU threshold always resolves to NumPy (whether or
    # not a GPU is present): below the crossover the GPU loses.
    assert resolve_backend("auto", problem_size=4, gpu_min_size=1000)[1] \
        == "numpy"
    # A large problem resolves to CuPy when a GPU is present, else NumPy.
    expected = "cupy" if df.cupy_gpu_available() else "numpy"
    assert resolve_backend("auto", problem_size=10000, gpu_min_size=1)[1] \
        == expected


def test_one_norm_conventions():
    eri = _synthetic_eri(n=4, num_vectors=3, seed=9)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    one_body = np.array([0.5, -1.0, 0.25, 0.75])
    base = float(np.sum(np.abs(one_body)))

    lcu = df.double_factorization_one_norm(factorization,
                                           one_body,
                                           convention="lcu")
    burg = df.double_factorization_one_norm(factorization,
                                            one_body,
                                            convention="burg")
    assert lcu >= base
    assert burg >= base
    with pytest.raises(ValueError):
        df.double_factorization_one_norm(factorization,
                                         one_body,
                                         convention="bogus")


def test_reconstruct_eri_matches_helper():
    eri = _synthetic_eri(n=4, num_vectors=2, seed=3)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
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
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    one_body_eigenvalues = np.array([0.5, -1.0, 0.25, 0.75])
    lam = df.double_factorization_one_norm(factorization, one_body_eigenvalues)
    assert lam >= float(np.sum(np.abs(one_body_eigenvalues)))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        df.explicit_double_factorization(np.zeros((3, 3)))  # not rank-4
    with pytest.raises(ValueError):
        df.compressed_double_factorization(_synthetic_eri(4, 2, 1),
                                           num_leaves=0)


def _openfermion_factorize():
    """OpenFermion's explicit-DF ``factorize``, or skip if unavailable.

    The public ``openfermion.resource_estimates.df`` API is gated behind an
    optional ``jax`` dependency, but the factorization itself only needs NumPy,
    so fall back to loading the module file directly when the gate blocks the
    normal import.
    """
    pytest.importorskip("openfermion")
    try:
        from openfermion.resource_estimates import df as of_df
        if hasattr(of_df, "factorize"):
            return of_df.factorize
    except Exception:
        pass
    import importlib.util
    import openfermion
    path = os.path.join(os.path.dirname(openfermion.__file__),
                        "resource_estimates", "df", "factorize_df.py")
    if not os.path.exists(path):
        pytest.skip("OpenFermion explicit-DF reference is not available")
    spec = importlib.util.spec_from_file_location("_of_factorize_df", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        pytest.skip("OpenFermion explicit-DF reference could not be loaded")
    return module.factorize


def test_explicit_double_factorization_matches_openfermion_reference():
    """Cross-check X-DF against OpenFermion's independent explicit-DF reference.

    Compared on the reconstructed ERI tensor (convention-independent), since the
    two implementations store leaves differently and apply different second-factor
    truncation conventions.
    """
    factorize = _openfermion_factorize()
    eri = _h2o_eri()
    n = eri.shape[0]
    reference_eri, _factors, rank, _num_eigenvectors = factorize(
        eri, thresh=1.0e-13)

    # The reference reconstructs the ERI, and its first-factor rank cannot exceed
    # the symmetric-pair dimension n(n+1)/2.
    assert np.linalg.norm(reference_eri - eri) < 1.0e-9
    assert rank <= n * (n + 1) // 2

    mine = df.explicit_double_factorization(eri, threshold=0.0)
    np.testing.assert_allclose(mine.reconstruct_eri(),
                               reference_eri,
                               atol=1.0e-9)


def test_asymmetric_eri_is_rejected():
    eri = _synthetic_eri(4, 3, seed=11)
    eri[0, 1, 2, 3] += 0.05  # break (pq|rs) == (qp|rs)
    with pytest.raises(ValueError, match="chemist symmetries"):
        df.explicit_double_factorization(eri)


def test_indefinite_eri_warns_on_cholesky_path():
    # A negative-weight leaf makes the supermatrix indefinite while keeping
    # the within-pair index symmetries intact.
    eri = _synthetic_eri(4, 3, seed=12) - 2.0 * _synthetic_eri(4, 1, seed=13)
    with pytest.warns(RuntimeWarning, match="not positive semidefinite"):
        factorization = df.explicit_double_factorization(eri)
    # The dropped negative part shows up as reconstruction error far above
    # the threshold; the eigendecomposition path handles the same input.
    assert df.factorization_error(eri, factorization) > 1.0e-3
    exact = df.explicit_double_factorization(
        eri, first_factorization="eigendecomposition")
    assert df.factorization_error(eri, exact) < 1.0e-10
