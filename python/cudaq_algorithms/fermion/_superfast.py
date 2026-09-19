# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bravyi-Kitaev Superfast (BKSF) fermion-to-qubit mapping (Setia-Whitfield,
arXiv:1712.00446).

An *edge* encoding: one qubit per edge of the Hamiltonian's interaction graph,
with a code subspace fixed by loop (cycle) stabilizers. Local terms map to
bounded-weight qubit operators (weight set by the graph degree, not the system
size), unlike Jordan-Wigner's O(N) strings -- the payoff when the interaction
graph is sparse (lattice / Hubbard models).

Unlike the *linear* encodings in ``_compilers`` (Jordan-Wigner, Bravyi-Kitaev),
BKSF has no local per-mode ladder operator -- a single ``a_i`` is odd and not
gauge invariant on the edge code space -- so this module cannot reuse the
``a_i``/``a_i^dagger`` compiler. It supplies its own term compiler that maps the
*even* physical operators directly onto edge and vertex operators:

    n_i                     = (I - B_i) / 2
    a_i^dag a_j + a_j^dag a_i = (i/2) A_ij (B_i - B_j)        (edge (i,j))
    n_i n_j                 = (I - B_i)(I - B_j) / 4

with the vertex operator ``B_i`` (product of Z over edges incident to i) and the
edge operator ``A_ij`` (X on edge (i,j) dressed with Z on lower-indexed incident
edges). The low-level (x, z)-word algebra and the ``SpinOperator`` assembly are
reused verbatim from ``_compilers``.

**Scope (Tier 1).** Real, symmetric one-body integrals plus density-density
(``n_i n_j``) two-body integrals -- Fermi-Hubbard, extended Hubbard, spinless
lattice fermions, and the diagonal Coulomb part of molecular Hamiltonians.
Complex / non-symmetric one-body input and non-density (exchange, pair-hopping,
general ``a^dag a^dag a a``) two-body input are rejected loudly; use
``jordan_wigner`` / ``bravyi_kitaev`` for those, or a later BKSF tier.
"""
from __future__ import annotations

import numpy as np

from ._compilers import _word_product, _to_spin_operator, _validate_tensors


# ---------------------------------------------------------------------------
# (x, z)-word helpers  (a word is (x_mask, z_mask); coefficients are separate)
# ---------------------------------------------------------------------------

def _mask(qubits):
    m = 0
    for q in qubits:
        m |= 1 << q
    return m


def _wmul(w1, w2):
    """Product of two Pauli words -> (phase, word)."""
    phase, x, z = _word_product(w1[0], w1[1], w2[0], w2[1])
    return phase, (x, z)


def _anticommute(w1, w2):
    x1, z1 = w1
    x2, z2 = w2
    return ((x1 & z2).bit_count() + (z1 & x2).bit_count()) & 1 == 1


# ---------------------------------------------------------------------------
# Interaction graph
# ---------------------------------------------------------------------------

class _Graph:
    """Interaction graph with a fixed edge->qubit indexing and incidence.

    ``edges[q]`` is the mode pair carried by qubit ``q``. A self-loop ``(i, i)``
    is a dedicated qubit that makes ``B_i`` non-trivial for a mode that carries
    a number term but no coupling (an isolated vertex)."""

    def __init__(self, num_modes, edges):
        self.num_modes = num_modes
        self.edges = list(edges)
        self.qubit_of = {e: q for q, e in enumerate(self.edges)}
        self.incident = [[] for _ in range(num_modes)]
        for q, (i, j) in enumerate(self.edges):
            self.incident[i].append(q)
            if j != i:
                self.incident[j].append(q)

    @property
    def num_qubits(self):
        return len(self.edges)

    def edge_qubit(self, i, j):
        return self.qubit_of[(min(i, j), max(i, j))]


def _build_graph(one_body, two_body, tolerance):
    """Interaction graph from the nonzero terms, with self-loops added for
    isolated modes that still carry a number term."""
    n = one_body.shape[0]
    edges = set()
    number_modes = set()

    for i in range(n):
        if abs(one_body[i, i]) > tolerance:
            number_modes.add(i)
        for j in range(n):
            if i < j and (abs(one_body[i, j]) > tolerance or
                          abs(one_body[j, i]) > tolerance):
                edges.add((i, j))

    if two_body is not None and two_body.size:
        for i, j, k, l in np.argwhere(np.abs(two_body) > tolerance):
            i, j, k, l = int(i), int(j), int(k), int(l)
            if i == j or k == l:
                continue                       # a^dag a^dag = 0 (or a a = 0)
            if sorted((i, j)) != sorted((k, l)):
                raise NotImplementedError(
                    "bravyi_kitaev_superfast (Tier 1) supports only "
                    "density-density (n_i n_j) two-body terms; entry "
                    f"({i},{j},{k},{l}) is a non-diagonal interaction. Use "
                    "jordan_wigner / bravyi_kitaev for general two-body "
                    "integrals.")
            number_modes.update((i, j))

    graph_edges = sorted(edges)
    incident_modes = {m for e in graph_edges for m in e}
    for m in sorted(number_modes - incident_modes):
        graph_edges.append((m, m))             # self-loop qubit for B_m
    return _Graph(n, graph_edges)


# ---------------------------------------------------------------------------
# Edge / vertex operators
# ---------------------------------------------------------------------------

def _b_word(graph, i):
    """Vertex operator B_i = product of Z over edges incident to i."""
    return (0, _mask(graph.incident[i]))


def _a_word(graph, i, j):
    """Edge operator A_ij (i<j) = X_{(i,j)} dressed with Z on incident edges
    (to i and to j) of smaller qubit index, so A's sharing a vertex
    anticommute."""
    i, j = min(i, j), max(i, j)
    p = graph.edge_qubit(i, j)
    z_edges = [q for q in graph.incident[i] if q < p]
    z_edges += [q for q in graph.incident[j] if q < p]
    return (1 << p, _mask(z_edges))


# ---------------------------------------------------------------------------
# Loop (cycle) stabilizers -- they fix the BKSF code subspace
# ---------------------------------------------------------------------------

def _spanning_forest(graph):
    """Union-find spanning forest over non-self-loop edges; returns the tree
    edge set and the list of non-tree (chord) edges."""
    parent = list(range(graph.num_modes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree, chords = set(), []
    for (i, j) in graph.edges:
        if i == j:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.add((i, j))
        else:
            chords.append((i, j))
    return tree, chords


def _tree_path(tree, num_modes, src, dst):
    """Vertex path src->dst through the (forest) tree edges."""
    adj = {v: [] for v in range(num_modes)}
    for (i, j) in tree:
        adj[i].append(j)
        adj[j].append(i)
    prev = {src: src}
    stack = [src]
    while stack:
        v = stack.pop()
        if v == dst:
            break
        for w in adj[v]:
            if w not in prev:
                prev[w] = v
                stack.append(w)
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _stabilizer_words(graph):
    """One loop stabilizer per independent cycle: the ordered product of edge
    operators A around the cycle. Returns a list of (coefficient, word)."""
    tree, chords = _spanning_forest(graph)
    stabilizers = []
    for (i, j) in chords:
        cycle = _tree_path(tree, graph.num_modes, j, i) + [j]  # close the loop
        phase, word = 1.0 + 0j, (0, 0)
        for a, b in zip(cycle, cycle[1:]):
            ph, word = _wmul(word, _a_word(graph, a, b))
            phase *= ph
        stabilizers.append((phase, word))
    return stabilizers


# ---------------------------------------------------------------------------
# Term compiler
# ---------------------------------------------------------------------------

def _require_real_symmetric(one_body, tolerance):
    if np.max(np.abs(one_body.imag)) > tolerance:
        raise ValueError(
            "bravyi_kitaev_superfast requires a real one-body tensor "
            "(the edge encoding represents the real-symmetric / electronic-"
            "structure case); use jordan_wigner / bravyi_kitaev for complex "
            "integrals.")
    if np.max(np.abs(one_body - one_body.T)) > tolerance:
        raise ValueError(
            "bravyi_kitaev_superfast requires a symmetric one-body tensor "
            "(h[i,j] == h[j,i]); use jordan_wigner / bravyi_kitaev for a "
            "non-symmetric one-body Hamiltonian.")


def _compile(graph, one_body, two_body, scalar_offset, tolerance):
    accumulator: dict = {(0, 0): complex(scalar_offset)}

    def add(coeff, word):
        accumulator[word] = accumulator.get(word, 0j) + coeff

    n = one_body.shape[0]

    # one-body: eps_i n_i  and  h_ij (a^dag_i a_j + a^dag_j a_i)
    for i in range(n):
        eps = one_body[i, i].real
        if abs(eps) > tolerance:
            add(0.5 * eps, (0, 0))
            add(-0.5 * eps, _b_word(graph, i))
        for j in range(i + 1, n):
            t = one_body[i, j].real
            if abs(t) > tolerance:
                A = _a_word(graph, i, j)
                p_i = _wmul(A, _b_word(graph, i))
                p_j = _wmul(A, _b_word(graph, j))
                add(0.5j * t * p_i[0], p_i[1])
                add(-0.5j * t * p_j[0], p_j[1])

    # two-body: density-density  c * n_i n_j  (validated diagonal in _build_graph)
    if two_body is not None and two_body.size:
        for i, j, k, l in np.argwhere(np.abs(two_body) > tolerance):
            i, j, k, l = int(i), int(j), int(k), int(l)
            if i == j or k == l:
                continue
            coeff = two_body[i, j, k, l].real
            # (i,j,j,i) -> +n_i n_j ; (i,j,i,j) -> -n_i n_j
            sign = 1.0 if (k, l) == (j, i) else -1.0
            c = sign * coeff
            Bi, Bj = _b_word(graph, i), _b_word(graph, j)
            BiBj = _wmul(Bi, Bj)
            add(0.25 * c, (0, 0))
            add(-0.25 * c, Bi)
            add(-0.25 * c, Bj)
            add(0.25 * c * BiBj[0], BiBj[1])

    return _to_spin_operator(accumulator, tolerance)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bravyi_kitaev_superfast(one_body_or_two_body,
                            two_body=None,
                            scalar_offset: float = 0.0,
                            tolerance: float = 1e-15):
    """Bravyi-Kitaev Superfast transform of fermionic integrals (Setia-Whitfield,
    arXiv:1712.00446).

    A locality-preserving *edge* encoding: one qubit per edge of the
    Hamiltonian's interaction graph. Local terms compile to bounded-weight
    Pauli operators (set by the graph degree), the payoff over Jordan-Wigner's
    O(N) strings when the graph is sparse. The qubit count is the number of
    graph edges (``> N`` for dense Hamiltonians -- BKSF is for sparse ones).

    Accepts an ``(n, n)`` one-body tensor, optionally with an ``(n, n, n, n)``
    two-body tensor, or a two-body tensor alone; entries are the coefficients
    of ``adag_i a_j`` and ``adag_i adag_j a_k a_l``. ``scalar_offset`` is added
    as an identity term; entries and compiled terms below ``tolerance`` are
    dropped. Returns a ``cudaq.SpinOperator`` acting on the edge qubits.

    Tier 1 supports real, symmetric one-body integrals and density-density
    (``n_i n_j``) two-body integrals. Other inputs raise (see the module
    docstring). Use :func:`bravyi_kitaev_superfast_stabilizers` for the loop
    stabilizers that fix the code subspace.
    """
    one_body, two_body_arr, _ = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    _require_real_symmetric(one_body, max(tolerance, 1e-12))
    graph = _build_graph(one_body, two_body_arr, tolerance)
    return _compile(graph, one_body, two_body_arr, scalar_offset, tolerance)


def bravyi_kitaev_superfast_stabilizers(one_body_or_two_body,
                                        two_body=None,
                                        tolerance: float = 1e-15):
    """Loop stabilizers of the BKSF code subspace for the given integrals.

    Returns one ``cudaq.SpinOperator`` per independent cycle of the interaction
    graph (empty for a tree graph). Each is Hermitian, squares to the identity,
    and commutes with the mapped Hamiltonian; the physical code subspace is a
    joint eigenspace of these operators. Reference-state (occupation) fixing of
    the eigenvalues is a later tier.
    """
    one_body, two_body_arr, _ = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    _require_real_symmetric(one_body, max(tolerance, 1e-12))
    graph = _build_graph(one_body, two_body_arr, tolerance)
    return [_to_spin_operator({word: coeff}, tolerance)
            for coeff, word in _stabilizer_words(graph)]
