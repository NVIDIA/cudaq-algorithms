# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse (mapping) integrals for the fermion transforms.

The three transforms accept the integrals either as dense tensors or as
``{index tuple: coefficient}`` mappings (plus ``num_modes`` when no dense
tensor fixes the mode count). Both forms are reduced to the same sorted
records before compilation, so they must produce the same operator -- bitwise,
since the accumulation order is the same -- and sparse input must never be
densified (a dense two-body tensor at 1000 modes would be 10^12 entries).
"""
import numpy as np
import pytest

from cudaq_algorithms.fermion import (bravyi_kitaev, bravyi_kitaev_superfast,
                                      bravyi_kitaev_superfast_stabilizers,
                                      jordan_wigner)
from cudaq_algorithms.fermion._compilers import _normalize_integrals

_LINEAR = {"jordan_wigner": jordan_wigner, "bravyi_kitaev": bravyi_kitaev}
_ALL = (jordan_wigner, bravyi_kitaev, bravyi_kitaev_superfast)


def _dense(op):
    return np.asarray(op.to_matrix())


def _same_operator(a, b, exact=False):
    if a.qubit_count != b.qubit_count or a.term_count != b.term_count:
        return False
    A, B = _dense(a), _dense(b)
    return np.array_equal(A, B) if exact else np.allclose(A, B, atol=1e-12)


def _sparse(array):
    """Mapping of the nonzero entries of a dense tensor."""
    return {
        tuple(int(v) for v in index): complex(array[tuple(index)])
        for index in np.argwhere(array)
    }


def _random_problem(seed, n=4, two_body_terms=6):
    """Complex Hermitian hopping on all pairs, random on-site terms and a few
    general two-body entries (not symmetrized -- compiled as given)."""
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n), dtype=complex)
    for i in range(n):
        h[i, i] = rng.normal()
        for j in range(i + 1, n):
            v = rng.normal() + 1j * rng.normal()
            h[i, j], h[j, i] = v, np.conj(v)
    V = np.zeros((n, n, n, n), dtype=complex)
    for _ in range(two_body_terms):
        i, j, k, l = rng.integers(0, n, size=4)
        V[i, j, k, l] += rng.normal()
    return h, V


def _ring_problem(seed, n=4):
    """Complex hopping and density-density couplings on a ring (a sparse
    interaction graph, as the superfast encoding wants)."""
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n), dtype=complex)
    V = np.zeros((n, n, n, n), dtype=complex)
    for i in range(n):
        h[i, i] = rng.normal()
    for i in range(n):
        j = (i + 1) % n
        v = rng.normal() + 1j * rng.normal()
        h[i, j], h[j, i] = v, np.conj(v)
        V[i, j, j, i] = rng.normal()  # n_i n_j
    return h, V


_TRIANGLE = {
    (0, 1): -1.0,
    (1, 0): -1.0,
    (1, 2): -1.0,
    (2, 1): -1.0,
    (0, 2): -1.0,
    (2, 0): -1.0
}
_PATH = {(0, 1): -1.0, (1, 0): -1.0, (1, 2): -1.0, (2, 1): -1.0}

# ----------------------------------------------------------------------
# Dense, sparse and mixed inputs compile to the same operator
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_LINEAR))
@pytest.mark.parametrize("seed", [0, 1])
def test_linear_transforms_dense_sparse_mixed_agree(name, seed):
    transform = _LINEAR[name]
    h, V = _random_problem(seed)
    reference = transform(h, V, scalar_offset=0.3)
    sh, sV = _sparse(h), _sparse(V)
    assert _same_operator(reference,
                          transform(sh, sV, scalar_offset=0.3, num_modes=4),
                          exact=True)
    assert _same_operator(reference,
                          transform(h, sV, scalar_offset=0.3),
                          exact=True)
    assert _same_operator(reference,
                          transform(sh, V, scalar_offset=0.3),
                          exact=True)
    assert _same_operator(reference,
                          transform(sh, V, scalar_offset=0.3, num_modes=4),
                          exact=True)


@pytest.mark.parametrize("name", list(_LINEAR))
def test_one_body_only_and_two_body_only_shorthands(name):
    transform = _LINEAR[name]
    h, V = _random_problem(3)
    assert _same_operator(transform(h),
                          transform(_sparse(h), num_modes=4),
                          exact=True)
    # A four-index mapping alone is the two-body-only shorthand.
    assert _same_operator(transform(V),
                          transform(_sparse(V), num_modes=4),
                          exact=True)
    with pytest.raises(ValueError, match="rank 4"):
        transform(_sparse(V), _sparse(V), num_modes=4)


@pytest.mark.parametrize("local_modes", ["standard", "binary_tree"])
@pytest.mark.parametrize("parity", [0, 1])
def test_superfast_dense_sparse_agree_with_stabilizers(local_modes, parity):
    h, V = _ring_problem(5)
    kwargs = dict(local_modes=local_modes, parity=parity)
    reference = bravyi_kitaev_superfast(h, V, **kwargs)
    sh, sV = _sparse(h), _sparse(V)
    assert _same_operator(reference,
                          bravyi_kitaev_superfast(sh,
                                                  sV,
                                                  num_modes=4,
                                                  **kwargs),
                          exact=True)
    assert _same_operator(reference,
                          bravyi_kitaev_superfast(h, sV, **kwargs),
                          exact=True)
    reference_stabilizers = bravyi_kitaev_superfast_stabilizers(h, V, **kwargs)
    sparse_stabilizers = bravyi_kitaev_superfast_stabilizers(sh,
                                                             sV,
                                                             num_modes=4,
                                                             **kwargs)
    assert len(reference_stabilizers) == len(sparse_stabilizers) == 1
    for a, b in zip(reference_stabilizers, sparse_stabilizers):
        assert _same_operator(a, b, exact=True)


def test_mapping_insertion_order_does_not_matter():
    h, V = _ring_problem(7)
    sh, sV = _sparse(h), _sparse(V)
    reversed_h = dict(reversed(list(sh.items())))
    reversed_V = dict(reversed(list(sV.items())))
    for transform in _ALL:
        assert _same_operator(transform(sh, sV, num_modes=4),
                              transform(reversed_h, reversed_V, num_modes=4),
                              exact=True)
    for a, b in zip(
            bravyi_kitaev_superfast_stabilizers(sh, sV, num_modes=4),
            bravyi_kitaev_superfast_stabilizers(reversed_h,
                                                reversed_V,
                                                num_modes=4)):
        assert _same_operator(a, b, exact=True)


def test_numpy_integer_keys_are_accepted():
    hop = {(np.int64(0), np.int32(1)): -1.0, (np.int64(1), np.int32(0)): -1.0}
    for transform in _LINEAR.values():
        assert _same_operator(transform(hop, num_modes=2),
                              transform({
                                  (0, 1): -1.0,
                                  (1, 0): -1.0
                              },
                                        num_modes=2),
                              exact=True)


# ----------------------------------------------------------------------
# Zeros, tolerance and declared modes
# ----------------------------------------------------------------------


def test_explicit_zero_entries_behave_as_missing():
    with_zero_edge = {**_PATH, (0, 2): 0.0, (2, 0): 0.0}
    for transform in _ALL:
        assert _same_operator(transform(_PATH, num_modes=3),
                              transform(with_zero_edge, num_modes=3),
                              exact=True)
    # The zero entry creates no edge: the path is still a tree.
    assert bravyi_kitaev_superfast_stabilizers(with_zero_edge,
                                               num_modes=3,
                                               local_modes="standard") == []
    dense = np.zeros((3, 3))
    for (i, j), v in _PATH.items():
        dense[i, j] = v
    assert _same_operator(bravyi_kitaev_superfast(dense,
                                                  local_modes="standard"),
                          bravyi_kitaev_superfast(with_zero_edge,
                                                  num_modes=3,
                                                  local_modes="standard"),
                          exact=True)


def test_tolerance_applies_to_mapping_entries():
    tiny = {**_PATH, (0, 2): 1e-20, (2, 0): 1e-20}
    for transform in _ALL:
        base = transform(_PATH, num_modes=3)
        assert _same_operator(base, transform(tiny, num_modes=3), exact=True)
        kept = transform(tiny, num_modes=3, tolerance=1e-30)
        assert kept.term_count > base.term_count


def test_num_modes_rules():
    hop = {(0, 1): 1.0, (1, 0): 1.0}
    with pytest.raises(ValueError, match="num_modes is required"):
        jordan_wigner(hop)
    with pytest.raises(ValueError, match="num_modes is required"):
        bravyi_kitaev_superfast_stabilizers(hop)
    dense = np.zeros((3, 3))
    dense[0, 1] = dense[1, 0] = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        jordan_wigner(dense, num_modes=4)
    assert _same_operator(jordan_wigner(dense, num_modes=3),
                          jordan_wigner(dense),
                          exact=True)
    with pytest.raises(ValueError, match="positive integer"):
        jordan_wigner(hop, num_modes=0)
    with pytest.raises(ValueError, match="positive integer"):
        jordan_wigner(hop, num_modes=True)
    # The count is part of the operator: Bravyi-Kitaev words can depend on
    # it (the Fenwick ancestors of modes 0 and 4 differ at 5 and 8 modes),
    # Jordan-Wigner words never do.
    far = {(0, 4): 1.0, (4, 0): 1.0}
    assert not _same_operator(bravyi_kitaev(far, num_modes=5),
                              bravyi_kitaev(far, num_modes=8))
    assert _same_operator(jordan_wigner(far, num_modes=5),
                          jordan_wigner(far, num_modes=8),
                          exact=True)


def test_declared_modes_without_terms_are_allowed():
    dense = np.zeros((6, 6))
    for (i, j), v in _TRIANGLE.items():
        dense[i, j] = v
    for transform in _LINEAR.values():
        assert _same_operator(transform(dense),
                              transform(_TRIANGLE, num_modes=6),
                              exact=True)
    # Superfast: modes 3..5 carry no term, so they are not in the graph.
    sparse = bravyi_kitaev_superfast(_TRIANGLE, num_modes=6)
    assert _same_operator(bravyi_kitaev_superfast(dense), sparse, exact=True)
    assert sparse.qubit_count == 3


def test_sparse_input_is_never_densified():
    """1000 declared modes, three couplings: a dense two-body tensor would be
    10^12 entries, so completing at all is the proof."""
    n = 1000
    coulomb = {(0, 1, 1, 0): 2.0}
    terms = _normalize_integrals(_TRIANGLE, coulomb, num_modes=n)
    assert terms.num_modes == n
    assert len(terms.one_body_values) == 6
    assert len(terms.two_body_values) == 1
    assert bravyi_kitaev_superfast(_TRIANGLE, coulomb,
                                   num_modes=n).qubit_count == 3
    assert jordan_wigner(_TRIANGLE, coulomb, num_modes=n).qubit_count == 3
    # A far hop's Fenwick words reach up the 1000-mode tree (a dozen-plus
    # qubits), still without anything dense.
    far = {(0, 512): 1.0, (512, 0): 1.0}
    assert bravyi_kitaev(far, num_modes=n).qubit_count > 10


def test_empty_mappings():
    assert jordan_wigner({}, num_modes=2).term_count == 0
    coulomb = {(0, 1, 1, 0): 0.5}
    dense_V = np.zeros((2, 2, 2, 2))
    dense_V[0, 1, 1, 0] = 0.5
    assert _same_operator(jordan_wigner({}, coulomb, num_modes=2),
                          jordan_wigner(dense_V),
                          exact=True)
    with pytest.raises(ValueError, match="no fermionic terms"):
        bravyi_kitaev_superfast({}, num_modes=2)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_mapping_validation_errors():
    with pytest.raises(ValueError, match="index tuples"):
        jordan_wigner({0: 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="rank"):
        jordan_wigner({(0, 1, 2): 1.0}, num_modes=3)
    with pytest.raises(ValueError, match="mixes keys"):
        jordan_wigner({(0, 1): 1.0, (0, 1, 1, 0): 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="out of range"):
        jordan_wigner({(0, 2): 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="non-integer"):
        jordan_wigner({(0, 1.0): 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="non-integer"):
        jordan_wigner({(0, True): 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="non-numeric"):
        jordan_wigner({(0, 1): "x"}, num_modes=2)
    with pytest.raises(ValueError, match="non-finite"):
        jordan_wigner({(0, 1): float("nan")}, num_modes=2)
    with pytest.raises(ValueError, match="wrong rank"):
        jordan_wigner({(0, 1): 1.0}, {(0, 1): 1.0}, num_modes=2)
    with pytest.raises(ValueError, match="out of range"):
        bravyi_kitaev_superfast({
            (0, 1): 1.0,
            (1, 0): 1.0
        }, {(0, 5, 5, 0): 1.0},
                                num_modes=2)
    with pytest.raises(ValueError, match="non-finite"):
        jordan_wigner(np.array([[0.0, np.inf], [0.0, 0.0]]))
