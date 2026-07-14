# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure-Python fermion-to-qubit compilers, pinned against ground truth.

References used: dense Jordan-Wigner ladder matrices built in NumPy, the
exact Seeley-Richard-Love per-pair operators and the H2 operator from the
C++ unit tests, spectral equivalence between encodings, and (when the
compiled extension is present) term-by-term parity with the C++
transforms.
"""

import importlib.util

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.fermion import bravyi_kitaev, jordan_wigner

_HAVE_COMPILED = importlib.util.find_spec(
    "cudaq_algorithms._pycudaq_algorithms") is not None

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
# Bravyi-Kitaev: exact known answers from the C++ unit tests
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
# Parity with the compiled extension
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_COMPILED, reason="compiled extension not built")
def test_jordan_wigner_parity_with_compiled():
    from cudaq_algorithms._pycudaq_algorithms import fermion as compiled

    one_body, two_body = _random_generic_system(9, 5)
    ours = jordan_wigner(one_body,
                         two_body,
                         scalar_offset=0.3,
                         tolerance=1e-12)
    theirs = compiled.jordan_wigner(one_body,
                                    two_body,
                                    scalar_offset=0.3,
                                    tolerance=1e-12)
    assert _max_term_difference(ours, theirs, 5) < 1e-12


@pytest.mark.skipif(not _HAVE_COMPILED, reason="compiled extension not built")
def test_bravyi_kitaev_parity_with_compiled_on_h2():
    from cudaq_algorithms._pycudaq_algorithms import fermion as compiled

    two_body = _h2_two_body()
    ours = bravyi_kitaev(_H2_ONE_BODY, two_body, scalar_offset=_H2_OFFSET)
    theirs = compiled.bravyi_kitaev(_H2_ONE_BODY,
                                    two_body,
                                    scalar_offset=_H2_OFFSET)
    assert _max_term_difference(ours, theirs, 4) < 1e-12


@pytest.mark.skipif(not _HAVE_COMPILED, reason="compiled extension not built")
def test_bravyi_kitaev_generalizes_beyond_compiled():
    """The pure transform handles tensor orderings the C++ one assumed away.

    The C++ Bravyi-Kitaev enumerated restricted index patterns and is only
    correct for tensors in its expected canonical ordering; the pure
    transform compiles any tensor exactly as given. Pin that here by
    checking the pure BK against the compiled *Jordan-Wigner* spectrum on
    a tensor ordering outside that canonical form.
    """
    from cudaq_algorithms._pycudaq_algorithms import fermion as compiled

    one_body, two_body = _random_generic_system(7, 4)
    ours = bravyi_kitaev(one_body, two_body, scalar_offset=0.1)
    reference = compiled.jordan_wigner(one_body, two_body, scalar_offset=0.1)
    np.testing.assert_allclose(_spectrum(ours),
                               _spectrum(reference),
                               atol=1e-10)


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


def test_empty_input_yields_empty_operator():
    op = jordan_wigner(np.zeros((3, 3)))
    assert op.term_count == 0


def test_validation_errors():
    with pytest.raises(ValueError, match="rank"):
        jordan_wigner(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="dimensions differ"):
        jordan_wigner(np.zeros((2, 2)), np.zeros((3, 3, 3, 3)))
    with pytest.raises(ValueError, match="rank 4"):
        bravyi_kitaev(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="wrong rank"):
        jordan_wigner(np.zeros((2, 2)), np.zeros((2, 2, 2)))
