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
                                                 _anticommute, _build_graph,
                                                 _stabilizer_words)

_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _word_matrix(x, z, nq):
    """Dense matrix of the Pauli word (x, z) on ``nq`` qubits, little-endian."""
    m = np.array([[1.0 + 0j]])
    for q in range(nq):
        bit = 1 << q
        p = ("Y" if (x & bit and z & bit) else "X" if x & bit
             else "Z" if z & bit else "I")
        m = np.kron(_PAULI[p], m)
    return m


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


def _codespace_spec(h, V=None, scalar_offset=0.0, interaction_graph=None):
    """Spectrum of BKSF on the code subspace -- the *joint +1 eigenspace* of
    the sign-fixed loop stabilizers (no sector search: the stabilizers are
    sign-fixed so +1 is the physical sector).

    Stabilizer matrices are built explicitly on ``nq`` qubits from their
    (coeff, x, z) words -- cudaq's ``to_matrix()`` compacts unused qubit
    indices, which would misalign the projection on graphs whose stabilizers
    do not touch qubit 0 (e.g. dense graphs)."""
    args = (h,) if V is None else (h, V)
    graph = _build_graph(np.asarray(h, dtype=complex),
                         np.zeros((0, 0, 0, 0)) if V is None
                         else np.asarray(V, dtype=complex), 1e-15,
                         interaction_graph)
    nq = graph.num_qubits
    dim = 1 << nq
    Hb = _dense(bravyi_kitaev_superfast(*args, scalar_offset=scalar_offset,
                                        interaction_graph=interaction_graph))
    if Hb.shape[0] < dim:                       # op did not touch every qubit
        Hb = np.kron(np.eye(dim // Hb.shape[0]), Hb)
    cols = np.eye(dim, dtype=complex)
    for coeff, (x, z) in _stabilizer_words(graph):
        S = cols.conj().T @ (coeff * _word_matrix(x, z, nq)) @ cols
        w, U = np.linalg.eigh(S)
        cols = cols @ U[:, np.abs(w - 1) < 1e-7]   # project onto the +1 sector
    return np.sort(np.linalg.eigvalsh(cols.conj().T @ Hb @ cols))


def _matches_jw_even(h, V=None, interaction_graph=None):
    """Error between the BKSF code space and JW's even (code) parity sector."""
    even = _parity_sector_specs(h, V)[+1]
    spec = _codespace_spec(h, V, interaction_graph=interaction_graph)
    if len(spec) != len(even):
        return np.inf
    return float(np.max(np.abs(spec - even)))


def _max_pauli_weight(op):
    nq = op.qubit_count
    return max(sum(c != "I" for c in term.get_pauli_word(nq)) for term in op)


# Independent Fock-space reference (no transform in the loop) --------------

_LOWER = np.array([[0, 1], [0, 0]], dtype=complex)


def _dense_fermion_hamiltonian(one_body, two_body, scalar_offset=0.0):
    """Exact dense Fock-space Hamiltonian via Jordan-Wigner-strung ladder
    matrices -- independent of the transform under test."""
    m = one_body.shape[0]

    def annihilator(mode):
        ops = ([_PAULI["Z"]] * mode + [_LOWER]
               + [_PAULI["I"]] * (m - mode - 1))[::-1]
        out = np.array([[1.0 + 0j]])
        for op in ops:
            out = np.kron(out, op)
        return out

    lower = [annihilator(j) for j in range(m)]
    raise_ = [a.conj().T for a in lower]
    dim = 1 << m
    h = scalar_offset * np.eye(dim, dtype=complex)
    for i, j in np.argwhere(one_body):
        h += one_body[i, j] * (raise_[i] @ lower[j])
    for i, j, k, l in np.argwhere(two_body):
        h += two_body[i, j, k, l] * (raise_[i] @ raise_[j]
                                     @ lower[k] @ lower[l])
    return h


def _physical_system(n_spatial, seed):
    """Random Hamiltonian with physical electronic-structure symmetries: real
    symmetric spatial one-body, 8-fold-symmetric positive-semidefinite spatial
    two-electron integrals, spin-expanded to interleaved spin orbitals (as in
    the Jordan-Wigner / Bravyi-Kitaev three-way spectrum test)."""
    rng = np.random.default_rng(seed)
    n = n_spatial
    chem = np.zeros((n, n, n, n))
    for _ in range(n + 1):
        s = rng.normal(size=(n, n))
        s = 0.5 * (s + s.T)
        chem += float(rng.uniform(0.1, 1.0)) * np.einsum("pq,rs->pqrs", s, s)
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


def _even_sector_spectrum(matrix):
    dim = matrix.shape[0]
    parity = np.array([(-1) ** bin(k).count("1") for k in range(dim)])
    idx = np.where(parity == +1)[0]
    return np.sort(np.linalg.eigvalsh(matrix[np.ix_(idx, idx)]))


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
    assert _matches_jw_even(h) < 1e-10


@pytest.mark.parametrize("name", ["path-4", "ring-4", "ring-5", "2x2-lattice"])
@pytest.mark.parametrize("seed", [3, 9])
def test_density_density_spectrum_matches_jordan_wigner(name, seed):
    n, edges = _GRAPHS[name]
    h, V = _random_tight_binding(seed, n, edges, coulomb=True)
    assert _matches_jw_even(h, V) < 1e-10


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_molecular_hamiltonian_matches_exact_diagonalization(seed):
    """The analog of the JW/BK three-way spectrum test: build a molecular-
    symmetric Hamiltonian, diagonalize the exact fermionic operator, and
    compare to BKSF.

    Unlike the linear encodings -- whose full qubit spectrum equals the whole
    Fock spectrum -- BKSF represents one fermion-parity sector on its code
    subspace, so the comparison is BKSF's code space (joint +1 stabilizer
    eigenspace) against the exact Hamiltonian's even-parity sector. The general
    two-body integrals make a complete interaction graph, so this runs at 4
    spin-orbitals (6 qubits)."""
    one_body, two_body = _physical_system(2, seed)      # 4 spin-orbitals
    offset = 0.317
    exact_even = _even_sector_spectrum(
        _dense_fermion_hamiltonian(one_body, two_body, offset))
    jw_even = _even_sector_spectrum(
        _dense(jordan_wigner(one_body, two_body, scalar_offset=offset)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # dense (K4) graph warns
        code = _codespace_spec(one_body, two_body, scalar_offset=offset)
    np.testing.assert_allclose(code, exact_even, atol=1e-10)
    np.testing.assert_allclose(code, jw_even, atol=1e-10)


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
    bksf_ground = float(_codespace_spec(h, V)[0])
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


@pytest.mark.parametrize("name", ["ring-4", "ring-5", "2x2-lattice"])
def test_code_space_is_the_joint_plus_one_eigenspace(name):
    """The sign-fixed stabilizers put the code space at their joint +1
    eigenspace, which equals JW's even (vacuum) parity sector -- no sector
    search needed."""
    n, edges = _GRAPHS[name]
    h, V = _random_tight_binding(5, n, edges, coulomb=True)
    Hb = _dense(bravyi_kitaev_superfast(h, V))
    dim = Hb.shape[0]
    cols = np.eye(dim, dtype=complex)
    for s in bravyi_kitaev_superfast_stabilizers(h, V):
        S = _dense(s)
        S = np.kron(np.eye(dim // S.shape[0]), S) if S.shape[0] < dim else S
        w, U = np.linalg.eigh(S)
        cols = cols @ U[:, np.abs(w - 1) < 1e-7]        # +1 eigenspace
    spec = np.sort(np.linalg.eigvalsh(cols.conj().T @ Hb @ cols))
    even = _parity_sector_specs(h, V)[+1]
    assert len(spec) == len(even)
    assert np.max(np.abs(spec - even)) < 1e-10


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

def test_qubit_count_is_edge_count():
    n, edges = _GRAPHS["ring-4"]
    h, _ = _random_tight_binding(0, n, edges)
    assert bravyi_kitaev_superfast(h).qubit_count == len(edges)


def test_interaction_graph_superset_override():
    """A caller-supplied edge set (a superset with an extra edge) fixes the
    qubit layout and still reproduces the fermionic spectrum."""
    n, edges = _GRAPHS["path-4"]
    h, _ = _random_tight_binding(1, n, edges)
    extra = edges + [(0, 3)]                      # add a chord -> +1 qubit, +1 loop
    op = bravyi_kitaev_superfast(h, interaction_graph=extra)
    assert op.qubit_count == len(extra)
    assert len(bravyi_kitaev_superfast_stabilizers(
        h, interaction_graph=extra)) == 1
    # spectrum on the (now cyclic) code space still matches JW's even sector
    Hb = _dense(op)
    dim = Hb.shape[0]
    cols = np.eye(dim, dtype=complex)
    for s in bravyi_kitaev_superfast_stabilizers(h, interaction_graph=extra):
        S = _dense(s)
        S = np.kron(np.eye(dim // S.shape[0]), S) if S.shape[0] < dim else S
        w, U = np.linalg.eigh(S)
        cols = cols @ U[:, np.abs(w - 1) < 1e-7]
    spec = np.sort(np.linalg.eigvalsh(cols.conj().T @ Hb @ cols))
    assert np.max(np.abs(spec - _parity_sector_specs(h)[+1])) < 1e-10


def test_interaction_graph_missing_required_edge_raises():
    h = np.zeros((3, 3))
    h[0, 1] = h[1, 0] = 1.0
    h[1, 2] = h[2, 1] = 1.0
    with pytest.raises(ValueError, match="missing edges"):
        bravyi_kitaev_superfast(h, interaction_graph=[(0, 1)])   # (1,2) absent


def test_interaction_graph_out_of_range_raises():
    h, _ = _random_tight_binding(0, *_GRAPHS["path-4"])
    with pytest.raises(ValueError, match="out of range"):
        bravyi_kitaev_superfast(h, interaction_graph=[(0, 1), (1, 2),
                                                      (2, 3), (3, 9)])


def test_isolated_number_mode_raises():
    """A mode carrying only a number term but no coupling is a disconnected
    (isolated) component -- BKSF cannot represent it in a global parity sector,
    so it raises rather than returning a wrong-sector operator."""
    h = np.zeros((4, 4))
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:      # connected triangle 0-1-2
        h[i, j] = h[j, i] = 1.0
    h[3, 3] = 0.7                                 # isolated mode 3
    with pytest.raises(ValueError, match="connected|isolated"):
        bravyi_kitaev_superfast(h)


def test_disconnected_graph_raises():
    """Two decoupled components (e.g. two molecules) raise: BKSF fixes parity
    per component, which is not a single global-parity sector."""
    h = np.zeros((6, 6))
    for (i, j) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        h[i, j] = h[j, i] = 1.0                   # two disjoint triangles
    with pytest.raises(ValueError, match="connected|disconnected"):
        bravyi_kitaev_superfast(h)


def test_disconnected_graph_reconnected_via_override():
    """interaction_graph= can add a connecting edge to make a decoupled
    Hamiltonian's graph connected; the result then matches JW's even sector."""
    n = 4
    h = np.zeros((n, n))
    for (i, j) in [(0, 1), (2, 3)]:               # two disjoint edges
        h[i, j] = h[j, i] = 1.0
    for i in range(n):
        h[i, i] = 0.2 * (i + 1)
    with pytest.raises(ValueError, match="connected"):
        bravyi_kitaev_superfast(h)                 # disconnected as-is
    connected = [(0, 1), (2, 3), (1, 2)]           # bridge the two edges
    assert _matches_jw_even(h, interaction_graph=connected) < 1e-10


@pytest.mark.parametrize("edges", [
    [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)],     # two triangles sharing edge (0,1)
    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],     # 4-ring with a chord (two cycles)
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],  # K4 (three cycles)
])
def test_multicycle_graphs_match_jordan_wigner(edges):
    """Connected graphs with several (possibly edge-sharing) independent cycles
    still match JW's even sector."""
    n = 4
    rng = np.random.default_rng(3)
    h = np.zeros((n, n))
    for i in range(n):
        h[i, i] = rng.normal()
    for (i, j) in edges:
        h[i, j] = h[j, i] = rng.normal()
    assert _matches_jw_even(h) < 1e-10


def test_single_isolated_mode_raises():
    """A one-mode Hamiltonian has no edge to encode B_0 on and raises."""
    with pytest.raises(ValueError, match="connected|isolated"):
        bravyi_kitaev_superfast(np.array([[0.7]]))


def test_empty_hamiltonian_raises():
    """An all-zero Hamiltonian has no interaction graph and no qubits."""
    with pytest.raises(ValueError, match="no fermionic terms"):
        bravyi_kitaev_superfast(np.zeros((3, 3)), scalar_offset=1.5)


def test_interaction_graph_self_loop_rejected():
    """A self-loop in interaction_graph cannot smuggle in an isolated mode
    (it would leave that mode's parity unconstrained)."""
    with pytest.raises(ValueError, match="self-loop"):
        bravyi_kitaev_superfast(np.array([[0.7]]), interaction_graph=[(0, 0)])


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
# General one-body and two-body (the Majorana compiler)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ring-4", "ring-5", "2x2-lattice"])
@pytest.mark.parametrize("seed", [2, 8])
def test_complex_hopping_matches_jordan_wigner(name, seed):
    """Peierls / flux phases: Hermitian complex hopping on a lattice."""
    n, edges = _GRAPHS[name]
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n), dtype=complex)
    for i in range(n):
        h[i, i] = rng.normal()
    for (i, j) in edges:
        z = rng.normal() + 1j * rng.normal()
        h[i, j] = z
        h[j, i] = np.conj(z)
    assert _matches_jw_even(h) < 1e-10


@pytest.mark.parametrize("m", [3, 4])
@pytest.mark.parametrize("seed", [4, 15])
def test_general_two_body_matches_jordan_wigner(m, seed):
    """Arbitrary Hermitian two-body integrals (dense graph)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    h = 0.5 * (h + h.conj().T)
    V = 0.3 * (rng.normal(size=(m,) * 4) + 1j * rng.normal(size=(m,) * 4))
    V = 0.5 * (V + V.transpose(3, 2, 1, 0).conj())
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # dense-graph warning expected
        assert _matches_jw_even(h, V) < 1e-10


def test_pair_hopping_on_a_lattice():
    """A single Hermitian pair-hopping term adag_0 adag_1 a_2 a_3 + h.c. on a
    chain -- a non-density two-body term."""
    n = 4
    h = np.zeros((n, n))
    for i in range(n):
        h[i, i] = 0.2 * (i + 1)
    for (i, j) in [(0, 1), (1, 2), (2, 3)]:
        h[i, j] = h[j, i] = -0.7
    V = np.zeros((n, n, n, n), dtype=complex)
    c = 0.4 + 0.1j
    V[0, 1, 2, 3] += c
    V[3, 2, 1, 0] += np.conj(c)                 # Hermitian pair
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _matches_jw_even(h, V) < 1e-10


def test_dense_graph_warns():
    """A dense two-body tensor triggers the no-advantage UserWarning."""
    m = 4
    rng = np.random.default_rng(0)
    h = np.zeros((m, m))
    V = 0.3 * rng.normal(size=(m,) * 4)
    V = 0.5 * (V + V.transpose(3, 2, 1, 0))     # real symmetric-ish, dense
    with pytest.warns(UserWarning, match="dense interaction graph"):
        bravyi_kitaev_superfast(h, V)
