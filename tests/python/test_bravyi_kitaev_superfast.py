# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Bravyi-Kitaev Superfast (BKSF) edge encoding.

BKSF is not a basis permutation of Jordan-Wigner (different qubit count, a
stabilizer code subspace), so the linear-encoding equivalence test does not
transfer. Instead we validate by *code-subspace spectrum equality*: BKSF maps
onto ``|E|`` edge qubits whose physical states are a joint eigenspace of the
loop stabilizers; restricted there, the mapped Hamiltonian has the same
spectrum as Jordan-Wigner restricted to a fixed fermion-parity sector.

All references are built from first principles (no OpenFermion), on qpp-cpu
(fp64), matching the rest of the fermion suite.
"""
import numpy as np
import pytest

import cudaq

from cudaq_algorithms.fermion import (jordan_wigner, bravyi_kitaev_superfast,
                                      bravyi_kitaev_superfast_stabilizers)
from cudaq_algorithms.fermion._superfast import (_Graph, _b_word, _a_word,
                                                 _anticommute, _build_graph)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _dense(op):
    return np.asarray(op.to_matrix())


def _parity_sector_specs(h, V=None):
    """Sorted eigenvalues of Jordan-Wigner in each fermion-parity sector."""
    op = jordan_wigner(h) if V is None else jordan_wigner(h, V)
    M = _dense(op)
    dim = M.shape[0]
    parity = np.array([(-1) ** bin(k).count("1") for k in range(dim)])
    out = {}
    for sval in (+1, -1):
        idx = np.where(parity == sval)[0]
        out[sval] = np.sort(np.linalg.eigvalsh(M[np.ix_(idx, idx)]))
    return out


def _codespace_specs(h, V=None):
    """Spectra of BKSF on each joint eigenspace of the loop stabilizers."""
    op = bravyi_kitaev_superfast(h) if V is None else \
        bravyi_kitaev_superfast(h, V)
    Hb = _dense(op)
    dim = Hb.shape[0]
    stabs = bravyi_kitaev_superfast_stabilizers(h) if V is None else \
        bravyi_kitaev_superfast_stabilizers(h, V)

    def embed(mat):
        return np.kron(np.eye(dim // mat.shape[0]), mat) \
            if mat.shape[0] < dim else mat

    specs = []

    def recurse(cols, rest):
        if not rest:
            Hp = cols.conj().T @ Hb @ cols
            specs.append(np.sort(np.linalg.eigvalsh(Hp)))
            return
        S = cols.conj().T @ embed(_dense(rest[0])) @ cols
        w, U = np.linalg.eigh(S)
        for sval in (+1, -1):
            sub = cols @ U[:, np.abs(w - sval) < 1e-9]
            if sub.shape[1]:
                recurse(sub, rest[1:])

    recurse(np.eye(dim, dtype=complex), stabs)
    return specs


def _matches_a_jw_sector(h, V=None, atol=1e-10):
    sectors = _parity_sector_specs(h, V)
    best = np.inf
    for spec in _codespace_specs(h, V):
        for ref in sectors.values():
            if len(spec) == len(ref):
                best = min(best, float(np.max(np.abs(spec - ref))))
    return best


def _max_pauli_weight(op):
    nq = op.qubit_count
    return max(sum(c != "I" for c in term.get_pauli_word(nq)) for term in op)


# Graph presets (edges only; on-site / hopping / Coulomb filled per test)
_GRAPHS = {
    "path-4": (4, [(0, 1), (1, 2), (2, 3)]),
    "ring-4": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    "ring-5": (5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
    "2x2-lattice": (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
    "star-4": (4, [(0, 1), (0, 2), (0, 3)]),
}


def _random_tight_binding(seed, n, edges, coulomb=False):
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n))
    for i in range(n):
        h[i, i] = rng.normal()
    for (i, j) in edges:
        h[i, j] = h[j, i] = rng.normal()
    V = None
    if coulomb:
        V = np.zeros((n, n, n, n))
        for (p, q) in edges:
            V[p, q, q, p] += rng.normal()   # (i,j,j,i) -> + n_i n_j
    return h, V


# ----------------------------------------------------------------------
# Operator algebra
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", list(_GRAPHS))
def test_edge_vertex_algebra(name):
    """A/B operators satisfy the BKSF (anti)commutation relations."""
    n, edges = _GRAPHS[name]
    g = _Graph(n, edges)
    B = [_b_word(g, i) for i in range(n)]
    A = {(i, j): _a_word(g, i, j) for (i, j) in edges}
    for (i, j), a in A.items():
        assert _anticommute(a, B[i]) and _anticommute(a, B[j])
        for k in range(n):
            if k not in (i, j):
                assert not _anticommute(a, B[k])
    es = list(A)
    for u in range(len(es)):
        for v in range(u + 1, len(es)):
            shared = len(set(es[u]) & set(es[v]))
            anti = _anticommute(A[es[u]], A[es[v]])
            assert anti == (shared == 1)


# ----------------------------------------------------------------------
# Code-subspace spectrum equivalence with Jordan-Wigner
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", list(_GRAPHS))
@pytest.mark.parametrize("seed", [1, 7])
def test_onebody_spectrum_matches_jordan_wigner(name, seed):
    n, edges = _GRAPHS[name]
    h, _ = _random_tight_binding(seed, n, edges)
    assert _matches_a_jw_sector(h) < 1e-10


@pytest.mark.parametrize("name", ["path-4", "ring-4", "ring-5", "2x2-lattice"])
@pytest.mark.parametrize("seed", [3, 9])
def test_density_density_spectrum_matches_jordan_wigner(name, seed):
    n, edges = _GRAPHS[name]
    h, V = _random_tight_binding(seed, n, edges, coulomb=True)
    assert _matches_a_jw_sector(h, V) < 1e-10


def test_hubbard_dimer_ground_state():
    """Hubbard dimer (2 sites, 4 spin-orbitals): the BKSF code-space ground
    energy equals the closed form E0 = (U - sqrt(U^2 + 16 t^2)) / 2.

    The hopping graph is two disconnected spin chains; the on-site Coulomb
    edges connect it, so the code subspace is a single global-parity sector
    (the half-filled singlet lives there)."""
    # spin orbitals: 0=1up 1=1dn 2=2up 3=2dn; hop within a spin, U on-site.
    t, U = 1.3, 4.0
    h = np.zeros((4, 4))
    for a, b in [(0, 2), (1, 3)]:               # up-up, dn-dn hopping
        h[a, b] = h[b, a] = -t
    V = np.zeros((4, 4, 4, 4))
    for (p, q) in [(0, 1), (2, 3)]:             # U n_up n_dn per site
        V[p, q, q, p] += U
    e0 = 0.5 * (U - np.sqrt(U ** 2 + 16 * t ** 2))
    # the half-filled singlet lives in the even-parity sector (the code space)
    even = _parity_sector_specs(h, V)[+1]
    assert abs(float(even[0]) - e0) < 1e-10     # tensor / convention sanity
    bksf_ground = min(float(spec[0]) for spec in _codespace_specs(h, V))
    assert abs(bksf_ground - e0) < 1e-10


# ----------------------------------------------------------------------
# Stabilizers
# ----------------------------------------------------------------------

def test_stabilizers_count_and_properties():
    n, edges = _GRAPHS["2x2-lattice"]
    h, _ = _random_tight_binding(0, n, edges)
    stabs = bravyi_kitaev_superfast_stabilizers(h)
    # |E| - (N - 1) independent cycles for a connected graph
    assert len(stabs) == len(edges) - (n - 1)
    Hb = _dense(bravyi_kitaev_superfast(h))
    dim = Hb.shape[0]
    for s in stabs:
        S = _dense(s)
        S = np.kron(np.eye(dim // S.shape[0]), S) if S.shape[0] < dim else S
        assert np.allclose(S @ S, np.eye(dim), atol=1e-12)      # involutory
        assert np.allclose(S, S.conj().T, atol=1e-12)           # Hermitian
        assert np.allclose(S @ Hb, Hb @ S, atol=1e-12)          # commutes with H


def test_tree_has_no_stabilizers():
    n, edges = _GRAPHS["path-4"]
    h, _ = _random_tight_binding(0, n, edges)
    assert bravyi_kitaev_superfast_stabilizers(h) == []


# ----------------------------------------------------------------------
# Locality (the point of BKSF)
# ----------------------------------------------------------------------

def test_bounded_weight_beats_jordan_wigner():
    """A graph-adjacent but index-distant hopping is O(N) weight under JW and
    degree-bounded under BKSF."""
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    h = np.zeros((n, n))
    for (i, j) in edges:
        h[i, j] = h[j, i] = 1.0
    assert _max_pauli_weight(jordan_wigner(h)) >= n - 1     # long Z-string
    assert _max_pauli_weight(bravyi_kitaev_superfast(h)) <= 3


# ----------------------------------------------------------------------
# Structure / edge cases
# ----------------------------------------------------------------------

def test_qubit_count_is_edge_count_with_self_loops():
    n, edges = _GRAPHS["ring-4"]
    h, _ = _random_tight_binding(0, n, edges)
    assert bravyi_kitaev_superfast(h).qubit_count == len(edges)


def test_isolated_number_mode_gets_self_loop():
    """A single isolated mode carrying only a number term is represented via a
    self-loop qubit: n_0 = (I - Z)/2 has the single-mode spectrum {0, eps}."""
    eps = 0.7
    h = np.array([[eps]])
    graph = _build_graph(np.asarray(h, dtype=complex),
                         np.zeros((0, 0, 0, 0)), 1e-15)
    assert graph.edges == [(0, 0)]             # a self-loop qubit for n_0
    op = bravyi_kitaev_superfast(h)
    assert op.qubit_count == 1
    spec = np.sort(np.linalg.eigvalsh(_dense(op)))
    assert np.allclose(spec, [0.0, eps], atol=1e-12)


def test_scalar_offset_is_identity_term():
    h, _ = _random_tight_binding(0, *_GRAPHS["path-4"])
    base = bravyi_kitaev_superfast(h)
    shifted = bravyi_kitaev_superfast(h, scalar_offset=2.5)
    diff = _dense(shifted) - _dense(base)
    assert np.allclose(diff, 2.5 * np.eye(diff.shape[0]), atol=1e-12)


def test_returns_spin_operator():
    h, _ = _random_tight_binding(0, *_GRAPHS["path-4"])
    assert isinstance(bravyi_kitaev_superfast(h), cudaq.SpinOperator)


# ----------------------------------------------------------------------
# Rejected inputs (Tier 1 scope)
# ----------------------------------------------------------------------

def test_complex_one_body_rejected():
    h = np.zeros((2, 2), dtype=complex)
    h[0, 1] = 1j
    h[1, 0] = -1j
    with pytest.raises(ValueError, match="real"):
        bravyi_kitaev_superfast(h)


def test_asymmetric_one_body_rejected():
    h = np.array([[0.0, 1.0], [0.5, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        bravyi_kitaev_superfast(h)


def test_nondiagonal_two_body_rejected():
    h = np.zeros((3, 3))
    h[0, 1] = h[1, 0] = 1.0
    V = np.zeros((3, 3, 3, 3))
    V[0, 1, 2, 0] = 1.0                         # not density-density
    with pytest.raises(NotImplementedError, match="density-density"):
        bravyi_kitaev_superfast(h, V)
