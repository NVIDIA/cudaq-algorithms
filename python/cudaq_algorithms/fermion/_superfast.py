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

**Construction.** The compiler works in the Majorana algebra: a fermionic term
is expanded into Majorana monomials, normal-ordered to products of distinct
Majoranas, and each consecutive Majorana pair is mapped to an edge/vertex
operator via a fixed dictionary (below). Every pair of modes coupled by a term
is made a graph edge, so each bilinear is a direct edge operator. This supports
arbitrary (complex, non-symmetric) one-body and arbitrary two-body integrals --
Fermi-Hubbard and extended Hubbard, Peierls / flux (complex hopping), exchange,
pair-hopping, and general ``a^dag a^dag a a`` terms.

A two-body tensor densifies the graph (every pair of a term's modes is an
edge); on a dense graph BKSF uses more qubits than modes with no locality
advantage, and a ``UserWarning`` is emitted -- Jordan-Wigner / Bravyi-Kitaev
are cheaper there. BKSF pays off for *sparse* couplings (lattice models),
where terms stay bounded-weight regardless of system size.
"""
from __future__ import annotations

import itertools
import warnings

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


def _required_structure(one_body, two_body, tolerance):
    """Edges every Majorana bilinear needs, and modes carrying a number term."""
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
            modes = sorted({int(i), int(j), int(k), int(l)})
            number_modes.update(modes)
            # Every pair of modes coupled by a two-body term must be an edge,
            # so each Majorana bilinear arising from that term is a direct edge
            # operator (no path routing needed for correctness). This also
            # keeps the graph connected -- e.g. Fermi-Hubbard, whose hopping
            # graph is two disconnected spin chains joined only by the on-site
            # Coulomb term -- so the code subspace is a single global-parity
            # sector rather than one parity per component. A dense two-body
            # tensor therefore needs a dense graph (~N^2/2 edges); BKSF's
            # locality advantage is for sparse couplings.
            for a in range(len(modes)):
                for b in range(a + 1, len(modes)):
                    edges.add((modes[a], modes[b]))
    return edges, number_modes


def _connected_components(active, edges):
    """Connected components (as root sets) of ``active`` modes over ``edges``."""
    parent = {m: m for m in active}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in edges:
        if i in parent and j in parent:
            parent[find(i)] = find(j)
    return {find(m) for m in active}


def _build_graph(one_body, two_body, tolerance, interaction_graph=None):
    """Interaction graph from the nonzero terms (or a caller-supplied edge
    list).

    A supplied ``interaction_graph`` must be a *superset* of the edges the
    terms require (every pair of modes coupled by a term); extra edges are
    allowed (they add qubits and stabilizers). A subset that would need
    routing a bilinear through a path is rejected.

    The graph over the modes that carry a term must be **connected**: BKSF
    encodes each connected component's even-parity subalgebra independently,
    so a disconnected graph would not represent a single global fermion-parity
    sector (and an isolated mode carrying only a number term has no faithful
    ``B_i``). Both cases raise, pointing at ``interaction_graph`` to add
    connecting edges (or map the components separately)."""
    n = one_body.shape[0]
    required, number_modes = _required_structure(one_body, two_body, tolerance)

    if interaction_graph is None:
        edges = set(required)
    else:
        edges = set()
        for pair in interaction_graph:
            i, j = int(pair[0]), int(pair[1])
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(
                    f"interaction_graph edge {(i, j)} is out of range for "
                    f"{n} modes.")
            edges.add((min(i, j), max(i, j)))
        missing = required - edges
        if missing:
            raise ValueError(
                "interaction_graph is missing edges required by the "
                f"Hamiltonian: {sorted(missing)}. Every pair of modes coupled "
                "by a term must be an edge (routing a bilinear through a path "
                "is not supported); supply a superset or omit "
                "interaction_graph.")

    active = set(number_modes) | {m for e in edges for m in e}
    if not active:
        raise ValueError(
            "bravyi_kitaev_superfast has no fermionic terms to encode (the "
            "integrals are all below tolerance); there is no interaction graph "
            "and no qubits. Add a scalar_offset to jordan_wigner instead if a "
            "constant is all that is needed.")

    components = _connected_components(active, edges)
    isolated = sorted(m for m in active if all(m not in e for e in edges))
    if len(components) > 1 or isolated:
        detail = (f"isolated modes {isolated}" if isolated else
                  "multiple disconnected components")
        raise ValueError(
            "bravyi_kitaev_superfast requires a connected interaction "
            f"graph, but the Hamiltonian's graph has {detail}. BKSF fixes "
            "fermion parity per connected component, so a disconnected "
            "graph does not map to a single global-parity sector. Add "
            "connecting edges via interaction_graph=, or transform each "
            "component separately.")

    return _Graph(n, sorted(edges))


def _warn_if_dense(graph):
    """BKSF is worthwhile only for sparse interaction graphs; warn otherwise."""
    active = {m for e in graph.edges for m in e if e[0] != e[1]}
    coupling = [e for e in graph.edges if e[0] != e[1]]
    k = len(active)
    if k >= 4 and len(coupling) > 0.5 * k * (k - 1) / 2:
        warnings.warn(
            f"bravyi_kitaev_superfast: dense interaction graph "
            f"({len(coupling)} edges over {k} modes, > half of complete); "
            "BKSF uses more qubits than modes with no locality advantage "
            "here -- jordan_wigner / bravyi_kitaev are cheaper for dense "
            "Hamiltonians.", UserWarning, stacklevel=3)


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
    """One loop stabilizer per independent cycle. Returns a list of
    (coefficient, word) with the coefficient sign-fixed so that the code
    subspace is the joint ``+1`` eigenspace.

    The stabilizer is the (unphased) Pauli word of the product of edge
    operators around a fundamental cycle, times ``(-1)^b`` where ``b`` is the
    number of cycle edges traversed from a higher to a lower mode index. The
    fermionic loop operator is a scalar on the code space; this orientation
    count is exactly the sign that makes that scalar ``+1`` (the raw product's
    ``i``/``-i`` phases and the cycle length drop out). The resulting operator
    is a Hermitian involution whose ``+1`` eigenspace is the physical (even
    fermion-parity / vacuum) sector."""
    tree, chords = _spanning_forest(graph)
    stabilizers = []
    for (i, j) in chords:
        cycle = _tree_path(tree, graph.num_modes, j, i) + [j]  # close the loop
        backward = sum(1 for a, b in zip(cycle, cycle[1:]) if a > b)
        word = (0, 0)
        for a, b in zip(cycle, cycle[1:]):
            _, word = _wmul(word, _a_word(graph, a, b))
        stabilizers.append(((-1.0) ** backward, word))
    return stabilizers


# ---------------------------------------------------------------------------
# Term compiler
# ---------------------------------------------------------------------------

# The compiler works in the Majorana algebra. Each mode i carries two
# Majorana operators, gamma_{2i} and gamma_{2i+1}, with
#   a_i = (gamma_{2i} + i gamma_{2i+1})/2,  a^dag_i = (gamma_{2i} - i gamma_{2i+1})/2.
# A fermionic term is expanded into a sum of Majorana monomials, normal-
# ordered to a sorted product of *distinct* Majoranas (gamma^2 = I), then each
# monomial's consecutive pairs are mapped to edge/vertex operators via the
# BKSF dictionary (derived from B_i = -i gamma_{2i} gamma_{2i+1} and
# A_ij = -i gamma_{2i} gamma_{2j}):
#   gamma_{2i}   gamma_{2j}     = i A_ij            (i < j, edge)
#   gamma_{2i}   gamma_{2j+1}   = - A_ij B_j
#   gamma_{2i+1} gamma_{2j}     = - A_ij B_i
#   gamma_{2i+1} gamma_{2j+1}   = -i A_ij B_i B_j
#   gamma_{2i}   gamma_{2i+1}   =  i B_i            (same mode)
# Every bilinear that arises has both modes coupled by the originating term,
# hence an edge (see _build_graph), so no path routing is needed.

def _ladder_majorana(mode, dagger):
    """a_i / a^dag_i as [(coeff, majorana_index), ...]."""
    return [(0.5, 2 * mode),
            (-0.5j if dagger else 0.5j, 2 * mode + 1)]


def _normal_order(sequence):
    """Reduce a raw product of Majoranas to (sign, sorted distinct tuple)."""
    reduced: list = []
    sign = 1.0
    for mu in sequence:
        pos = len(reduced) - 1
        while pos >= 0 and reduced[pos] > mu:
            sign = -sign
            pos -= 1
        if pos >= 0 and reduced[pos] == mu:
            del reduced[pos]                   # gamma_mu^2 = I
        else:
            reduced.insert(pos + 1, mu)
    return sign, tuple(reduced)


def _majorana_terms(one_body, two_body, tolerance):
    """Full Hamiltonian as {sorted-Majorana-tuple: complex coefficient}."""
    accumulator: dict = {}

    def expand(coefficient, ladders):
        factors = [_ladder_majorana(mode, dagger) for mode, dagger in ladders]
        for choice in itertools.product(*factors):
            coeff = coefficient
            indices = []
            for factor_coeff, index in choice:
                coeff *= factor_coeff
                indices.append(index)
            sign, monomial = _normal_order(indices)
            accumulator[monomial] = accumulator.get(monomial, 0j) + coeff * sign

    n = one_body.shape[0]
    for i, j in np.argwhere(np.abs(one_body) > tolerance):
        expand(complex(one_body[i, j]), [(int(i), True), (int(j), False)])
    if two_body is not None and two_body.size:
        for i, j, k, l in np.argwhere(np.abs(two_body) > tolerance):
            expand(complex(two_body[i, j, k, l]),
                   [(int(i), True), (int(j), True),
                    (int(k), False), (int(l), False)])
    return {m: c for m, c in accumulator.items() if abs(c) > tolerance}


def _bilinear_word(mu, nu, graph):
    """gamma_mu gamma_nu (mu < nu) as (coefficient, word)."""
    # The four branches are the dictionary in the module docstring; a mode's
    # secondary Majorana gamma_{2a+1} = i gamma_{2a} B_a introduces a B factor.
    a, b = mu // 2, nu // 2
    if a == b:                                 # gamma_{2a} gamma_{2a+1} =  i B_a
        return 1j, _b_word(graph, a)
    A = _a_word(graph, a, b)                    # a < b since mu < nu
    mu_primary, nu_primary = (mu % 2 == 0), (nu % 2 == 0)
    if mu_primary and nu_primary:              # gamma_{2a} gamma_{2b}   =  i A_ab
        return 1j, A
    if mu_primary and not nu_primary:          # gamma_{2a} gamma_{2b+1} = -A_ab B_b
        phase, word = _wmul(A, _b_word(graph, b))
        return -phase, word
    if not mu_primary and nu_primary:          # gamma_{2a+1} gamma_{2b} = -A_ab B_a
        phase, word = _wmul(A, _b_word(graph, a))
        return -phase, word
    phase, word = _wmul(A, _b_word(graph, a))   # gamma_{2a+1} gamma_{2b+1}
    phase2, word = _wmul(word, _b_word(graph, b))  #             = -i A_ab B_a B_b
    return -1j * phase * phase2, word


def _monomial_word(monomial, graph):
    """A sorted product of distinct Majoranas as (coefficient, word)."""
    coeff, word = 1.0 + 0j, (0, 0)
    for k in range(0, len(monomial), 2):
        c, w = _bilinear_word(monomial[k], monomial[k + 1], graph)
        coeff *= c
        phase, word = _wmul(word, w)
        coeff *= phase
    return coeff, word


def _compile(graph, one_body, two_body, scalar_offset, tolerance):
    accumulator: dict = {(0, 0): complex(scalar_offset)}
    for monomial, coefficient in _majorana_terms(one_body, two_body,
                                                 tolerance).items():
        c, word = _monomial_word(monomial, graph)
        accumulator[word] = accumulator.get(word, 0j) + coefficient * c
    return _to_spin_operator(accumulator, tolerance)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bravyi_kitaev_superfast(one_body_or_two_body,
                            two_body=None,
                            scalar_offset: float = 0.0,
                            tolerance: float = 1e-15,
                            interaction_graph=None):
    """Bravyi-Kitaev Superfast transform of fermionic integrals (Setia-Whitfield,
    arXiv:1712.00446).

    A locality-preserving *edge* encoding: one qubit per edge of the
    Hamiltonian's interaction graph. Local terms compile to bounded-weight
    Pauli operators (set by the graph degree), the payoff over Jordan-Wigner's
    O(N) strings when the graph is sparse. The qubit count is the number of
    graph edges (``> N`` for dense Hamiltonians -- BKSF is for sparse ones).

    Accepts an ``(n, n)`` one-body tensor, optionally with an ``(n, n, n, n)``
    two-body tensor, or a two-body tensor alone; entries are the coefficients
    of ``adag_i a_j`` and ``adag_i adag_j a_k a_l``. Arbitrary (complex,
    non-symmetric) one-body and arbitrary two-body integrals are supported.
    ``scalar_offset`` is added as an identity term; entries and compiled terms
    below ``tolerance`` are dropped. Returns a ``cudaq.SpinOperator`` acting on
    the edge qubits.

    ``interaction_graph`` optionally pins the edge set (an iterable of
    ``(i, j)`` mode pairs) instead of inferring it from the nonzero terms --
    to control the qubit layout, or to add extra edges. It must be a superset
    of the edges the Hamiltonian requires; the ordering of the edges fixes the
    qubit indexing.

    The interaction graph over the modes that carry a term must be
    **connected** (BKSF fixes fermion parity per connected component, so a
    disconnected graph would not map to a single global-parity sector); a
    disconnected graph or an isolated number-only mode raises, pointing at
    ``interaction_graph`` to add connecting edges. Hermiticity of the input is
    the caller's responsibility -- non-Hermitian integrals compile to a
    non-Hermitian operator, as for :func:`jordan_wigner`.

    Two-body couplings densify the interaction graph (every pair of a term's
    modes becomes an edge); a dense tensor gives a dense graph on which BKSF
    has no locality advantage over Jordan-Wigner and emits a ``UserWarning``.
    Use :func:`bravyi_kitaev_superfast_stabilizers` for the loop stabilizers
    that fix the code subspace.
    """
    one_body, two_body_arr, _ = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    graph = _build_graph(one_body, two_body_arr, tolerance, interaction_graph)
    _warn_if_dense(graph)
    return _compile(graph, one_body, two_body_arr, scalar_offset, tolerance)


def bravyi_kitaev_superfast_stabilizers(one_body_or_two_body,
                                        two_body=None,
                                        tolerance: float = 1e-15,
                                        interaction_graph=None):
    """Loop stabilizers of the BKSF code subspace for the given integrals.

    Returns one ``cudaq.SpinOperator`` per independent cycle of the interaction
    graph (empty for a tree graph). Each is a Hermitian involution, commutes
    with the mapped Hamiltonian, and is sign-fixed so that the **code subspace
    is their joint +1 eigenspace** -- the physical (even fermion-parity /
    vacuum) sector, matching the same sector under Jordan-Wigner. Restricting
    the mapped Hamiltonian to that eigenspace recovers the fermionic spectrum.

    ``interaction_graph`` pins the edge set as in :func:`bravyi_kitaev_superfast`
    (use the same value for both so the stabilizers act on the matching qubits).
    """
    one_body, two_body_arr, _ = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    graph = _build_graph(one_body, two_body_arr, tolerance, interaction_graph)
    return [_to_spin_operator({word: coeff}, tolerance)
            for coeff, word in _stabilizer_words(graph)]
