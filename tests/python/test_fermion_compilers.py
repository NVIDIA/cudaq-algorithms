# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python fermion-to-qubit compilers, pinned against ground truth.

References used: dense Jordan-Wigner ladder matrices built in NumPy, the
exact Seeley-Richard-Love per-pair operators and the H2 operator
inherited from the retired C++ implementation's unit tests, exact
Fock-space diagonalization via sparse ladder operators, and spectral
equivalence between encodings.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.fermion import bravyi_kitaev, jordan_wigner

# ----------------------------------------------------------------------
# References
# ----------------------------------------------------------------------

_I2 = np.eye(2)
_Z2 = np.diag([1.0, -1.0])
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])


def _dense_ladders(num_modes):

    def annihilator(mode):
        ops = ([_Z2] * mode + [_LOWER] + [_I2] * (num_modes - mode - 1))[::-1]
        out = np.array([[1.0]])
        for op in ops:
            out = np.kron(out, op)
        return out

    lower = [annihilator(j) for j in range(num_modes)]
    return lower, [a.conj().T for a in lower]


def _dense_hamiltonian(one_body, two_body, scalar_offset=0.0):
    m = one_body.shape[0]
    lower, raise_ = _dense_ladders(m)
    dim = 1 << m
    h = scalar_offset * np.eye(dim, dtype=complex)
    for i in range(m):
        for j in range(m):
            if one_body[i, j]:
                h += one_body[i, j] * raise_[i] @ lower[j]
            for k in range(m):
                for l in range(m):
                    if two_body[i, j, k, l]:
                        h += two_body[i, j, k, l] * (
                            raise_[i] @ raise_[j] @ lower[k] @ lower[l])
    return h


def _terms(op, width):
    return {
        term.get_pauli_word(width): complex(term.evaluate_coefficient())
        for term in op
    }


def _max_term_difference(a, b, width):
    ta, tb = _terms(a, width), _terms(b, width)
    return max(
        abs(ta.get(word, 0.0) - tb.get(word, 0.0))
        for word in set(ta) | set(tb))


def _random_generic_system(seed, m):
    """Hermitian tensors with no other structure.

    The two-body tensor is hermitized ((adag_i adag_j a_k a_l)^dag has
    coefficient conj(V[l, k, j, i])) but deliberately has none of the
    additional ordering symmetries of chemistry-canonical tensors.
    """
    rng = np.random.default_rng(seed)
    one_body = rng.normal(size=(m, m)) + 1.0j * rng.normal(size=(m, m))
    one_body = 0.5 * (one_body + one_body.conj().T)
    two_body = 0.3 * rng.normal(size=(m, m, m, m)).astype(complex)
    two_body = 0.5 * (two_body + two_body.transpose(3, 2, 1, 0).conj())
    return one_body.astype(complex), two_body


def _spectrum(op):
    return np.sort(np.linalg.eigvalsh(np.asarray(op.to_matrix())))


# ----------------------------------------------------------------------
# Jordan-Wigner against dense ladders
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [5, 17])
def test_jordan_wigner_matches_dense_ladders(seed):
    one_body, two_body = _random_generic_system(seed, 4)
    op = jordan_wigner(one_body, two_body, scalar_offset=0.25)
    dense = _dense_hamiltonian(one_body, two_body, scalar_offset=0.25)
    np.testing.assert_allclose(np.asarray(op.to_matrix()), dense, atol=1e-12)


def test_jordan_wigner_nonhermitian_tensor_matches_dense():
    # No hermiticity assumed anywhere: a raw random tensor must still
    # compile to exactly the operator it denotes.
    rng = np.random.default_rng(23)
    one_body = (rng.normal(size=(3, 3)) +
                1.0j * rng.normal(size=(3, 3))).astype(complex)
    two_body = (0.3 * rng.normal(size=(3, 3, 3, 3))).astype(complex)
    op = jordan_wigner(one_body, two_body)
    dense = _dense_hamiltonian(one_body, two_body)
    np.testing.assert_allclose(np.asarray(op.to_matrix()), dense, atol=1e-12)


def test_jordan_wigner_one_body_only_and_two_body_only():
    one_body, two_body = _random_generic_system(3, 3)
    combined = jordan_wigner(one_body, two_body)
    split = jordan_wigner(one_body) + jordan_wigner(two_body)
    assert _max_term_difference(combined, split.canonicalize(), 3) < 1e-12


# ----------------------------------------------------------------------
# Bravyi-Kitaev: exact known answers (from the retired C++ unit tests)
# ----------------------------------------------------------------------


def _single_pair(i, j, coefficient, m):
    one_body = np.zeros((m, m), dtype=complex)
    one_body[i, j] = coefficient
    return bravyi_kitaev(one_body)


def _expected(width, *specs):
    out = {}
    for coefficient, paulis in specs:
        word = "".join(paulis.get(q, "I") for q in range(width))
        out[word] = out.get(word, 0.0) + coefficient
    return out


def _assert_terms(op, width, expected, tol=1e-12):
    got = _terms(op, width)
    for word in set(got) | set(expected):
        assert got.get(word, 0.0) == pytest.approx(expected.get(word, 0.0),
                                                   abs=tol), word


def test_fenwick_matrix_matches_binary_indexed_tree():
    # The permutation and spectral BK tests are invariant to *which*
    # invertible GF(2) matrix _fenwick_matrix returns (BK = P.JW.P^T holds
    # for any invertible P), so they do not pin the matrix itself. The
    # sparse single-pair known-answers only touch modes {0,1,2,6,7,18,19};
    # a single off-by-one at a mid-range mode would leave the suite green.
    # Pin the matrix directly: cross-check against an independent Fenwick
    # formulation (the "update" traversal i += i & -i, versus the
    # implementation's range marking) over mid-range modes, and anchor n=8
    # against a hand-written binary-indexed tree.
    from cudaq_algorithms.fermion._compilers import _fenwick_matrix

    def independent(n):
        matrix = np.zeros((n, n), dtype=np.uint8)
        for mode in range(1, n + 1):  # one-based
            i = mode
            while i <= n:
                matrix[i - 1, mode - 1] = 1
                i += i & (-i)
        return matrix

    for n in (1, 2, 3, 4, 5, 8, 12, 13, 16, 20):
        assert np.array_equal(_fenwick_matrix(n), independent(n)), n

    expected_8 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
                          dtype=np.uint8)
    assert np.array_equal(_fenwick_matrix(8), expected_8)


def test_bravyi_kitaev_number_operator():
    _assert_terms(_single_pair(2, 2, 4.0, 20), 20,
                  _expected(20, (2.0, {}), (-2.0, {
                      2: "Z"
                  })))


def test_bravyi_kitaev_neighbor_pair():
    _assert_terms(
        _single_pair(1, 2, 4.0, 20), 20,
        _expected(20, (1.0, {
            0: "Z",
            1: "Y",
            2: "Y"
        }), (-1.0j, {
            0: "Z",
            1: "Y",
            2: "X"
        }), (1.0, {
            1: "X",
            2: "X"
        }), (1.0j, {
            1: "X",
            2: "Y"
        })))


def test_bravyi_kitaev_cross_branch_pair():
    _assert_terms(
        _single_pair(2, 6, 4.0, 20), 20,
        _expected(20, (1.0, {
            1: "Z",
            2: "X",
            3: "Y",
            5: "Z",
            6: "Y"
        }), (-1.0, {
            1: "Z",
            2: "Y",
            3: "Y",
            5: "Z",
            6: "X"
        }), (-1.0j, {
            1: "Z",
            2: "X",
            3: "Y",
            5: "Z",
            6: "X"
        }), (-1.0j, {
            1: "Z",
            2: "Y",
            3: "Y",
            5: "Z",
            6: "Y"
        })))


def test_bravyi_kitaev_root_pair():
    _assert_terms(
        _single_pair(18, 19, 4.0, 20), 20,
        _expected(20, (1.0, {
            18: "X"
        }), (-1.0j, {
            18: "Y"
        }), (1.0j, {
            17: "Z",
            18: "Y",
            19: "Z"
        }), (-1.0, {
            17: "Z",
            18: "X",
            19: "Z"
        })))


def test_bravyi_kitaev_long_range_pair():
    _assert_terms(
        _single_pair(0, 7, 4.0, 20), 20,
        _expected(20, (-1.0, {
            0: "Y",
            1: "X",
            3: "Y",
            5: "Z",
            6: "Z"
        }), (-1.0j, {
            0: "X",
            1: "X",
            3: "Y",
            5: "Z",
            6: "Z"
        }), (1.0j, {
            0: "Y",
            1: "X",
            3: "X",
            7: "Z"
        }), (-1.0, {
            0: "X",
            1: "X",
            3: "X",
            7: "Z"
        })))


_H2_ONE_BODY = np.diag([-1.2488, -1.2488, -0.47967, -0.47967]).astype(complex)
_H2_OFFSET = 0.7080240981000804


def _h2_two_body():
    a1, a2 = 0.3366719725032414, 0.0908126657382825
    b1, b2 = 0.09081266573828267, 0.33121364716348484
    c1, c2 = 0.3312136471634851, 0.09081266573828246
    d1, d2 = 0.09081266573828264, 0.34814578499360427
    entries = {
        (0, 0, 0, 0): a1,
        (0, 0, 2, 2): a2,
        (0, 1, 1, 0): a1,
        (0, 1, 3, 2): a2,
        (0, 2, 0, 2): b1,
        (0, 2, 2, 0): b2,
        (0, 3, 1, 2): b1,
        (0, 3, 3, 0): b2,
        (1, 0, 0, 1): a1,
        (1, 0, 2, 3): a2,
        (1, 1, 1, 1): a1,
        (1, 1, 3, 3): a2,
        (1, 2, 0, 3): b1,
        (1, 2, 2, 1): b2,
        (1, 3, 1, 3): b1,
        (1, 3, 3, 1): b2,
        (2, 0, 0, 2): c1,
        (2, 0, 2, 0): c2,
        (2, 1, 1, 2): c1,
        (2, 1, 3, 0): c2,
        (2, 2, 0, 0): d1,
        (2, 2, 2, 2): d2,
        (2, 3, 1, 0): d1,
        (2, 3, 3, 2): d2,
        (3, 0, 0, 3): c1,
        (3, 0, 2, 1): c2,
        (3, 1, 1, 3): c1,
        (3, 1, 3, 1): c2,
        (3, 2, 0, 1): d1,
        (3, 2, 2, 3): d2,
        (3, 3, 1, 1): d1,
        (3, 3, 3, 3): d2,
    }
    two_body = np.zeros((4, 4, 4, 4), dtype=complex)
    for index, value in entries.items():
        two_body[index] = value
    return two_body


def test_bravyi_kitaev_h2_known_answer():
    op = bravyi_kitaev(_H2_ONE_BODY, _h2_two_body(), scalar_offset=_H2_OFFSET)
    expected = {
        "IIII": -0.1064770114930045,
        "XZXI": 0.04540633286914125,
        "XZXZ": 0.04540633286914125,
        "YZYI": 0.04540633286914125,
        "YZYZ": 0.04540633286914125,
        "ZIII": 0.17028010135220506,
        "ZZII": 0.1702801013522051,
        "ZZZI": 0.16560682358174256,
        "ZZZZ": 0.16560682358174256,
        "ZIZI": 0.12020049071260128,
        "ZIZZ": 0.12020049071260128,
        "IZII": 0.1683359862516207,
        "IZZZ": -0.22004130022421792,
        "IZIZ": 0.17407289249680227,
        "IIZI": -0.22004130022421792,
    }
    # The C++ unit test's reference coefficients are rounded (its own
    # comparison tolerance is 1e-4).
    _assert_terms(op, 4, expected, tol=1e-4)


# ----------------------------------------------------------------------
# Encoding equivalence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4, 5])
def test_bravyi_kitaev_isospectral_to_jordan_wigner(m):
    one_body, two_body = _random_generic_system(11, m)
    jw = jordan_wigner(one_body, two_body, scalar_offset=0.25)
    bk = bravyi_kitaev(one_body, two_body, scalar_offset=0.25)
    np.testing.assert_allclose(_spectrum(jw), _spectrum(bk), atol=1e-10)


def test_bravyi_kitaev_word_weight_advantage():
    # The point of BK: O(log n) support instead of O(n) Z strings.
    m = 16
    one_body = np.zeros((m, m), dtype=complex)
    one_body[0, m - 1] = 1.0
    one_body[m - 1, 0] = 1.0

    def max_weight(op):
        return max(
            sum(1 for ch in word if ch != "I") for word in _terms(op, m))

    assert max_weight(jordan_wigner(one_body)) == m
    assert max_weight(bravyi_kitaev(one_body)) < m // 2


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------


def test_scalar_offset_and_tolerance():
    one_body = np.array([[0.5, 0.0], [0.0, 1e-9]], dtype=complex)
    op = jordan_wigner(one_body, scalar_offset=2.0, tolerance=1e-6)
    got = _terms(op, 2)
    assert got["II"] == pytest.approx(2.25)  # offset + n_0 identity part
    assert got["ZI"] == pytest.approx(-0.25)
    assert all("I" == word[1] for word in got)  # 1e-9 entry pruned


def test_output_side_tolerance_trims_small_compiled_terms():
    # A one-body diagonal entry c compiles to Pauli-word coefficients of
    # magnitude |c|/2, so an entry that survives the input prune
    # (|c| >= tolerance) can still yield a term below tolerance that the
    # output-side prune in _to_spin_operator must drop. With tolerance 1e-6:
    #   c = 3.0e-6 -> Z coefficient 1.5e-6   (kept),
    #   c = 1.5e-6 -> Z coefficient 0.75e-6  (dropped by the output prune).
    op = jordan_wigner(np.diag([3e-6, 1.5e-6]).astype(complex), tolerance=1e-6)
    got = _terms(op, 2)
    assert "ZI" in got  # 1.5e-6 >= tolerance: kept
    assert "IZ" not in got  # 0.75e-6 < tolerance: output-trimmed
    assert got["ZI"] == pytest.approx(-1.5e-6, abs=1e-12)


def test_operator_width_tracks_touched_qubits():
    # Documented edge: the operator's width tracks the qubits actually
    # touched, not the input dimension n.
    idle_top = np.zeros((3, 3), dtype=complex)
    idle_top[0, 0] = 1.0  # only orbital 0 is coupled
    op = jordan_wigner(idle_top)
    assert np.asarray(op.to_matrix()).shape == (2, 2)  # 2^1, not 2^3

    empty = jordan_wigner(np.zeros((3, 3), dtype=complex))
    assert empty.term_count == 0
    assert np.asarray(empty.to_matrix()).shape == (0, 0)


def test_validation_errors():
    with pytest.raises(ValueError, match="rank"):
        jordan_wigner(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="dimensions differ"):
        jordan_wigner(np.zeros((2, 2)), np.zeros((3, 3, 3, 3)))
    with pytest.raises(ValueError, match="rank 4"):
        bravyi_kitaev(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="wrong rank"):
        jordan_wigner(np.zeros((2, 2)), np.zeros((2, 2, 2)))


# ----------------------------------------------------------------------
# Three-way spectral agreement: exact fermionic vs JW vs BK
# ----------------------------------------------------------------------


def _physical_system(n_spatial, seed):
    """Random Hamiltonian with physical electronic-structure symmetries.

    Spatial one-body: real symmetric. Spatial two-electron integrals:
    8-fold symmetric and positive semidefinite (sum of symmetric rank-one
    squares, like real molecular ERIs). Both are spin-expanded to the
    interleaved spin-orbital convention.
    """
    rng = np.random.default_rng(seed)
    n = n_spatial
    chem = np.zeros((n, n, n, n))
    for _ in range(n + 1):
        s = rng.normal(size=(n, n))
        s = 0.5 * (s + s.T)
        chem += float(rng.uniform(0.1, 1.0)) * np.einsum('pq,rs->pqrs', s, s)
    h_spatial = rng.normal(size=(n, n))
    h_spatial = 0.5 * (h_spatial + h_spatial.T)

    reordered = np.ascontiguousarray(chem.transpose(0, 2, 3, 1))
    m = 2 * n
    one_body = np.zeros((m, m), dtype=complex)
    two_body = np.zeros((m, m, m, m), dtype=complex)
    for p in range(n):
        for q in range(n):
            one_body[2 * p, 2 * q] = h_spatial[p, q]
            one_body[2 * p + 1, 2 * q + 1] = h_spatial[p, q]
            for r in range(n):
                for s in range(n):
                    c = 0.5 * reordered[p, q, r, s]
                    two_body[2 * p, 2 * q, 2 * r, 2 * s] = c
                    two_body[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = c
                    two_body[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = c
                    two_body[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = c
    return one_body, two_body


def _sparse_fermionic_hamiltonian(one_body, two_body, scalar_offset):
    """Exact dense Fock-space Hamiltonian via sparse ladder matrices.

    Independent of both transforms under test; sparse products keep the
    build fast enough to diagonalize 10-qubit systems exactly.
    """
    from scipy import sparse

    m = one_body.shape[0]
    dim = 1 << m
    lower = []
    for j in range(m):
        ops = ([_Z2] * j + [_LOWER] + [_I2] * (m - j - 1))[::-1]
        out = sparse.identity(1, format="csr")
        for op in ops:
            out = sparse.kron(out, sparse.csr_matrix(op), format="csr")
        lower.append(out)
    raise_ = [op.conj().T.tocsr() for op in lower]

    h = sparse.identity(dim, format="csr", dtype=complex) * scalar_offset
    for i, j in np.argwhere(one_body):
        h = h + one_body[i, j] * (raise_[i] @ lower[j])
    for i, j, k, l in np.argwhere(two_body):
        h = h + two_body[i, j, k,
                         l] * (raise_[i] @ raise_[j] @ lower[k] @ lower[l])
    return h.toarray()


@pytest.mark.parametrize("n_spatial", [2, 3, 4, 5])
def test_three_way_spectrum_agreement(n_spatial):
    """Exact fermionic, JW, and BK spectra must agree at every size."""
    one_body, two_body = _physical_system(n_spatial, seed=31 + n_spatial)
    offset = 0.317
    m = 2 * n_spatial
    dim = 1 << m

    exact = np.linalg.eigvalsh(
        _sparse_fermionic_hamiltonian(one_body, two_body, offset))

    jw_matrix = np.asarray(
        jordan_wigner(one_body, two_body, scalar_offset=offset).to_matrix())
    bk_matrix = np.asarray(
        bravyi_kitaev(one_body, two_body, scalar_offset=offset).to_matrix())
    assert jw_matrix.shape == (dim, dim)
    assert bk_matrix.shape == (dim, dim)

    jw_spectrum = np.linalg.eigvalsh(jw_matrix)
    bk_spectrum = np.linalg.eigvalsh(bk_matrix)

    np.testing.assert_allclose(jw_spectrum, exact, atol=1e-10)
    np.testing.assert_allclose(bk_spectrum, exact, atol=1e-10)
    np.testing.assert_allclose(bk_spectrum, jw_spectrum, atol=1e-10)


# ----------------------------------------------------------------------
# Exact permutation equivalence between the encodings
# ----------------------------------------------------------------------


def _encoding_permutation(num_modes):
    """P with P|n> = |A n mod 2>, the basis map from the Jordan-Wigner
    (occupation) to the Bravyi-Kitaev (partial-sum) computational basis,
    little-endian bit order matching ``to_matrix``."""
    from cudaq_algorithms.fermion._compilers import _fenwick_matrix

    encoding = _fenwick_matrix(num_modes)
    dim = 1 << num_modes
    perm = np.zeros((dim, dim))
    for n in range(dim):
        bits = [(n >> j) & 1 for j in range(num_modes)]
        image_bits = encoding.dot(bits) % 2
        image = sum(int(b) << j for j, b in enumerate(image_bits))
        perm[image, n] = 1.0
    return perm


@pytest.mark.parametrize("seed,m", [(11, 3), (12, 4), (13, 5), (14, 6)])
def test_bravyi_kitaev_is_permuted_jordan_wigner(seed, m):
    """BK == P JW P^T as exact matrices, pinning BK term content on
    arbitrary hermitian tensors (stronger than matching spectra)."""
    one_body, two_body = _random_generic_system(seed, m)
    jw = np.asarray(
        jordan_wigner(one_body, two_body, scalar_offset=0.125).to_matrix())
    bk = np.asarray(
        bravyi_kitaev(one_body, two_body, scalar_offset=0.125).to_matrix())
    perm = _encoding_permutation(m)
    np.testing.assert_allclose(bk, perm @ jw @ perm.T, atol=1e-12)


def test_bravyi_kitaev_is_permuted_jordan_wigner_nonhermitian():
    rng = np.random.default_rng(7)
    m = 4
    one_body = rng.normal(size=(m, m)) + 1.0j * rng.normal(size=(m, m))
    two_body = 0.3 * (rng.normal(size=(m, m, m, m)) +
                      1.0j * rng.normal(size=(m, m, m, m)))
    jw = np.asarray(jordan_wigner(one_body, two_body).to_matrix())
    bk = np.asarray(bravyi_kitaev(one_body, two_body).to_matrix())
    perm = _encoding_permutation(m)
    np.testing.assert_allclose(bk, perm @ jw @ perm.T, atol=1e-12)


# ----------------------------------------------------------------------
# H2 against a frozen full-CI reference
# ----------------------------------------------------------------------

# Frozen once from pyscf 2.13.1 (RHF + FCI, H2 at 0.7474 A, STO-3G,
# OMP_NUM_THREADS=1) using the exact integral recipe of
# test_jordan_wigner.py. Freezing the MO-basis integrals makes the
# full-CI cross-check deterministic and dependency-free: the SCF
# orbitals are the only non-reproducible ingredient, and they are baked
# into these numbers together with the reference energy computed from
# the same mean field. test_jordan_wigner.py remains the live-pyscf
# version of this check.
_H2_NUCLEAR_REPULSION = 0.7080240981000804
_H2_FCI_ENERGY = -1.1371757102406845
_H2_H1_SPATIAL = [
    [-1.2488468037963385, 8.867226166437188e-18],
    [9.4085812113517e-17, -0.4796778131338564],
]
_H2_H2E_SPATIAL = [
    [[[0.6733439450064822, -2.0816681711721685e-17],
      [0.0, 0.18162533147656484]],
     [[0.0, 0.18162533147656507], [0.6624272943269697,
                                   8.326672684688674e-17]]],
    [[[2.0816681711721685e-17, 0.6624272943269696], [0.18162533147656484,
                                                     0.0]],
     [[0.18162533147656523, -5.551115123125783e-17],
      [-8.326672684688674e-17, 0.6962915699872075]]],
]


def _h2_frozen_spin_orbital_tensors():
    """Interleaved spin expansion (the test_jordan_wigner.py loop)."""
    h1e = np.array(_H2_H1_SPATIAL)
    h2e = np.array(_H2_H2E_SPATIAL)
    num_spin_orbitals = 2 * h1e.shape[0]
    one_body = np.zeros((num_spin_orbitals, num_spin_orbitals),
                        dtype=np.complex128)
    two_body = np.zeros((num_spin_orbitals, ) * 4, dtype=np.complex128)
    for p in range(num_spin_orbitals // 2):
        for q in range(num_spin_orbitals // 2):
            one_body[2 * p, 2 * q] = h1e[p, q]
            one_body[2 * p + 1, 2 * q + 1] = h1e[p, q]
            for r in range(num_spin_orbitals // 2):
                for s in range(num_spin_orbitals // 2):
                    coefficient = 0.5 * h2e[p, q, r, s]
                    two_body[2 * p, 2 * q, 2 * r, 2 * s] = coefficient
                    two_body[2 * p + 1, 2 * q + 1, 2 * r + 1,
                             2 * s + 1] = coefficient
                    two_body[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = coefficient
                    two_body[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = coefficient
    return one_body, two_body


@pytest.mark.parametrize("transform", [jordan_wigner, bravyi_kitaev])
def test_h2_ground_state_matches_frozen_fci(transform):
    one_body, two_body = _h2_frozen_spin_orbital_tensors()
    op = transform(one_body,
                   two_body,
                   scalar_offset=_H2_NUCLEAR_REPULSION,
                   tolerance=1e-12)
    ground = float(np.min(np.linalg.eigvalsh(np.asarray(op.to_matrix()))))
    assert abs(ground - _H2_FCI_ENERGY) < 1e-10
