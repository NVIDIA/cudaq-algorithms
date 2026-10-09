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

from ._compilers import (_Encoding, _fenwick_matrix, _word_product,
                         _to_spin_operator, _validate_tensors)

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

    ``edges[q]`` is the ``(i, j)`` mode pair (``i < j``) carried by qubit
    ``q``. The graph is required to be connected with no self-loops (enforced
    in :func:`_build_graph`)."""

    def __init__(self, num_modes, edges):
        self.num_modes = num_modes
        self.edges = list(edges)
        self.qubit_of = {e: q for q, e in enumerate(self.edges)}
        self.incident = [[] for _ in range(num_modes)]
        for q, (i, j) in enumerate(self.edges):
            self.incident[i].append(q)
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
            if i < j and (abs(one_body[i, j]) > tolerance
                          or abs(one_body[j, i]) > tolerance):
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
            if i == j:
                raise ValueError(
                    f"interaction_graph contains a self-loop {(i, j)}; BKSF "
                    "edges couple two distinct modes. A mode carrying only a "
                    "number term must be coupled to the rest of the graph.")
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
        detail = (f"isolated modes {isolated}"
                  if isolated else "multiple disconnected components")
        raise ValueError(
            "bravyi_kitaev_superfast requires a connected interaction "
            f"graph, but the Hamiltonian's graph has {detail}. BKSF fixes "
            "fermion parity per connected component, so a disconnected "
            "graph does not map to a single global-parity sector. Add "
            "connecting edges via interaction_graph=, or transform each "
            "component separately.")

    return _Graph(n, sorted(edges))


def _warn_if_dense(graph):
    """BKSF is worthwhile only for sparse interaction graphs; warn otherwise.

    The threshold is average degree > k/2 (i.e. edges > k^2/4), so genuinely
    sparse graphs -- rings and lattices, whose average degree is O(1) -- do
    not trip it, while near-complete graphs do."""
    active = {m for e in graph.edges for m in e}
    k = len(active)
    edges = len(graph.edges)
    if k >= 4 and edges > k * k / 4:
        warnings.warn(
            f"bravyi_kitaev_superfast: dense interaction graph "
            f"({edges} edges over {k} modes, average degree > k/2); "
            "BKSF uses more qubits than modes with no locality advantage "
            "here -- jordan_wigner / bravyi_kitaev are cheaper for dense "
            "Hamiltonians.",
            UserWarning,
            stacklevel=3)


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
    """Union-find spanning tree; returns the tree edge set and the list of
    non-tree (chord) edges (one fundamental cycle per chord)."""
    parent = list(range(graph.num_modes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree, chords = set(), []
    for (i, j) in graph.edges:
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
        stabilizers.append(((-1.0)**backward, word))
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
    return [(0.5, 2 * mode), (-0.5j if dagger else 0.5j, 2 * mode + 1)]


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
            del reduced[pos]  # gamma_mu^2 = I
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
            accumulator[monomial] = accumulator.get(monomial,
                                                    0j) + coeff * sign

    n = one_body.shape[0]
    for i, j in np.argwhere(np.abs(one_body) > tolerance):
        expand(complex(one_body[i, j]), [(int(i), True), (int(j), False)])
    if two_body is not None and two_body.size:
        for i, j, k, l in np.argwhere(np.abs(two_body) > tolerance):
            expand(complex(two_body[i, j, k, l]), [(int(i), True),
                                                   (int(j), True),
                                                   (int(k), False),
                                                   (int(l), False)])
    return {m: c for m, c in accumulator.items() if abs(c) > tolerance}


def _bilinear_word(mu, nu, backend):
    """gamma_mu gamma_nu (mu < nu) as (coefficient, word).

    The four branches are the dictionary in the module docstring; a mode's
    secondary Majorana gamma_{2a+1} = i gamma_{2a} B_a introduces a B factor.
    ``backend`` supplies the edge/vertex operators as ``(coeff, word)`` --
    ``coeff`` is 1 for the edge encoding but carries the ``(-i)^{d/2}`` and
    local-mode phases for the generalized (vertex) encoding."""
    a, b = mu // 2, nu // 2
    if a == b:  # gamma_{2a} gamma_{2a+1} =  i B_a
        cb, bw = backend.b_word(a)
        return 1j * cb, bw
    ca, A = backend.a_word(a, b)  # a < b since mu < nu
    mu_primary, nu_primary = (mu % 2 == 0), (nu % 2 == 0)
    if mu_primary and nu_primary:  # gamma_{2a} gamma_{2b}   =  i A_ab
        return 1j * ca, A
    if mu_primary and not nu_primary:  # gamma_{2a} gamma_{2b+1} = -A_ab B_b
        cb, bw = backend.b_word(b)
        phase, word = _wmul(A, bw)
        return -ca * cb * phase, word
    if not mu_primary and nu_primary:  # gamma_{2a+1} gamma_{2b} = -A_ab B_a
        cb, bw = backend.b_word(a)
        phase, word = _wmul(A, bw)
        return -ca * cb * phase, word
    ca_a, ba = backend.b_word(a)  # gamma_{2a+1} gamma_{2b+1} = -i A_ab B_a B_b
    cb_b, bb = backend.b_word(b)
    phase, word = _wmul(A, ba)
    phase2, word = _wmul(word, bb)
    return -1j * ca * ca_a * cb_b * phase * phase2, word


def _monomial_word(monomial, backend):
    """A sorted product of distinct Majoranas as (coefficient, word)."""
    coeff, word = 1.0 + 0j, (0, 0)
    for k in range(0, len(monomial), 2):
        c, w = _bilinear_word(monomial[k], monomial[k + 1], backend)
        coeff *= c
        phase, word = _wmul(word, w)
        coeff *= phase
    return coeff, word


def _compile(backend, one_body, two_body, scalar_offset, tolerance):
    accumulator: dict = {(0, 0): complex(scalar_offset)}
    for monomial, coefficient in _majorana_terms(one_body, two_body,
                                                 tolerance).items():
        c, word = _monomial_word(monomial, backend)
        accumulator[word] = accumulator.get(word, 0j) + coefficient * c
    return _to_spin_operator(accumulator, tolerance)


# ---------------------------------------------------------------------------
# Operator backends: edge encoding ("standard") and the vertex-based
# generalized encodings (binary_tree / error_correcting / custom local modes)
# ---------------------------------------------------------------------------
#
# The term compiler and the loop stabilizers touch an encoding only through
# ``a_word(i, j)`` (i < j) and ``b_word(i)``, each returning ``(coeff, word)``.
# A backend also exposes ``num_modes``, ``num_qubits``, and ``edges`` so the
# spanning-forest cycle machinery can build the stabilizers.


class _StandardBackend:
    """The original edge encoding: one qubit per edge (Setia-Whitfield,
    arXiv:1712.00446). A/B are real Hermitian Pauli words (coeff 1)."""

    def __init__(self, graph):
        self.graph = graph
        self.num_modes = graph.num_modes
        self.num_qubits = graph.num_qubits
        self.edges = graph.edges

    def a_word(self, i, j):
        return 1.0, _a_word(self.graph, i, j)

    def b_word(self, i):
        return 1.0, _b_word(self.graph, i)

    def stabilizers(self):
        return _stabilizer_words(self.graph)


def _word_mul(c1, w1, c2, w2):
    """Product of two ``(coeff, word)`` operators."""
    phase, x, z = _word_product(w1[0], w1[1], w2[0], w2[1])
    return c1 * c2 * phase, (x, z)


# --- Local Majorana modes (paper Eqs 18-23, Appendix B) --------------------


def _string_to_word(s):
    """A Pauli string (char ``s[p]`` on local qubit ``p``) as (coeff, (x, z)).

    The string already denotes a tensor of single-qubit Paulis (Hermitian),
    so the canonical coefficient is 1."""
    x = z = 0
    for p, ch in enumerate(s):
        bit = 1 << p
        if ch in "XY":
            x |= bit
        if ch in "ZY":
            z |= bit
    return 1.0, (x, z)


def _fenwick_local_modes(degree):
    """``degree`` local Majorana modes on ``degree/2`` qubits via the
    Fenwick (binary-indexed-tree) encoding -- weight <= ceil(log2 degree),
    with the parity product mapping to a single Z (so ``B_i`` has weight 1)."""
    k = degree // 2
    enc = _Encoding(_fenwick_matrix(k))
    modes = []
    for j in range(k):
        update, parity, flip = (enc.update_masks[j], enc.parity_masks[j],
                                enc.flip_masks[j])
        modes.append((1.0, (update, parity)))  # c_{2j}   = X_U Z_P
        phase, x, z = _word_product(update, parity, 0, flip)
        modes.append((1j * phase, (x, z)))  # c_{2j+1} = i X_U Z_P Z_F
    return modes


def _error_correcting_local_modes(degree):
    """Local modes satisfying the error-correction weight conditions (Eq 22)
    for an even degree ``d >= 6`` (Eq 23 / Appendix B).

    For an odd number of qubits ``d/2 = 2k+1`` the modes are cyclic shifts of
    the ``Z..Z X`` / ``Z..Z Y`` stems (Eq 23 is the ``k = 1`` case); the
    ``d = 8`` case (four qubits) is the explicit construction in Appendix B.
    """
    if degree < 6 or degree % 2:
        raise ValueError(
            f"error_correcting local modes need an even degree >= 6, got "
            f"{degree}; pad the interaction graph (even degree >= 6) or use "
            "local_modes='binary_tree'.")
    nq = degree // 2

    def rot(s, shift):
        shift %= len(s)
        return s[-shift:] + s[:-shift] if shift else s

    if nq % 2 == 1:  # d/2 = 2k + 1
        k = (nq - 1) // 2
        stem_x = "Z" * k + "X" + "I" * k
        stem_y = "Z" * k + "Y" + "I" * k
        strings = []
        for shift in range(nq):
            strings.append(rot(stem_x, shift))
            strings.append(rot(stem_y, shift))
    elif degree == 8:  # explicit Appendix B four-qubit construction
        strings = [
            "ZZXI", "ZZYI", "IZZX", "IZZY", "XIIZ", "YIIZ", "ZXII", "ZYII"
        ]
    else:
        raise ValueError(
            f"error_correcting local modes for degree {degree} (an even "
            "qubit count other than 4) are not built in; supply custom "
            "local_modes satisfying the Majorana algebra, or pad to a "
            "degree whose half is odd.")
    return [_string_to_word(s) for s in strings]


def _custom_local_modes(provider, degree):
    """Normalize a caller-supplied provider to a list of ``(coeff, (x, z))``.

    ``provider`` is a callable ``degree -> sequence`` or a mapping
    ``degree -> sequence``; each element is a Pauli string or a
    ``(coeff, (x, z))`` word on ``degree/2`` qubits."""
    raw = provider(degree) if callable(provider) else provider[degree]
    modes = []
    for item in raw:
        modes.append(_string_to_word(item) if isinstance(item, str) else item)
    return modes


def _validate_local_modes(modes, degree):
    """Enforce the Majorana algebra (Eq 19): ``degree`` Hermitian operators on
    ``degree/2`` qubits that square to I, pairwise anticommute, and are
    GF(2)-independent (so they generate the full local Pauli group)."""
    nq = degree // 2
    if len(modes) != degree:
        raise ValueError(
            f"expected {degree} local Majorana modes for degree {degree}, "
            f"got {len(modes)}.")
    for coeff, (x, z) in modes:
        if abs(coeff.imag) > 1e-9 or abs(abs(coeff) - 1.0) > 1e-9:
            raise ValueError(
                "local Majorana modes must be Hermitian involutions "
                f"(coefficient +-1), got coefficient {coeff}.")
        if (x | z) >> nq:
            raise ValueError(
                f"a local Majorana mode acts outside the {nq} qubits of its "
                "vertex.")
    for u in range(degree):
        for v in range(u + 1, degree):
            if not _anticommute(modes[u][1], modes[v][1]):
                raise ValueError(
                    "local Majorana modes must pairwise anticommute (Eq 19).")
    rows = [modes[u][1][0] | (modes[u][1][1] << nq) for u in range(degree)]
    rank = 0
    basis: list = []
    for r in rows:
        for b in basis:
            r = min(r, r ^ b)
        if r:
            basis.append(r)
            rank += 1
    if rank != degree:
        raise ValueError(
            "local Majorana modes are not independent; they must generate "
            "the full Pauli group on the vertex qubits (Eq 19).")


def _local_mode_provider(local_modes, degree):
    """The validated local Majorana modes for a vertex of the given degree."""
    if local_modes == "binary_tree":
        modes = _fenwick_local_modes(degree)
    elif local_modes == "error_correcting":
        modes = _error_correcting_local_modes(degree)
    elif isinstance(local_modes, str):
        raise ValueError(
            f"unknown local_modes {local_modes!r}; expected 'binary_tree', "
            "'error_correcting', 'standard', or a custom provider.")
    else:
        modes = _custom_local_modes(local_modes, degree)
    _validate_local_modes(modes, degree)
    return modes


class _GSEBackend:
    """Generalized Superfast Encoding (Setia-Bravyi-Mezzacapo-Whitfield,
    Phys. Rev. Research 1, 033033): ``d(i)/2`` qubits per vertex, with a
    choice of local Majorana modes. Same qubit count as the edge encoding
    (``n = |E|``) but a free (non-unique) choice of the per-vertex operators
    -- binary-tree for O(log d) Pauli weight, or the error-correcting modes.

    ``B_i = (-i)^{d/2} gamma_{i,1}...gamma_{i,d}`` (Eq 20);
    ``A_{i,j} = eps_{i,j} gamma_{i,p} gamma_{j,q}`` with ``j = N(i, p)`` and
    ``i = N(j, q)`` (Eq 21). The edge orientation ``eps_{i,j} = +-1`` (for
    ``i < j``; ``A_{j,i} = -A_{i,j}`` fixes the other direction) is the paper's
    "suitable choice": a fixed sign would leave a spurious flux on some loops
    that flips the encoded fermion-parity sector, so each fundamental cycle's
    chord orientation is set (``_fix_orientation``) to make the loop operator
    ``+1`` on the physical (even-parity) code space."""

    def __init__(self, graph, local_modes):
        self.num_modes = graph.num_modes
        self.edges = graph.edges
        neighbors = [[] for _ in range(graph.num_modes)]
        for (i, j) in graph.edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
        self._neighbors = [sorted(ns) for ns in neighbors]
        self._degree = [len(ns) for ns in self._neighbors]

        active = {m for e in graph.edges for m in e}
        offset, running = [0] * graph.num_modes, 0
        for v in range(graph.num_modes):
            offset[v] = running
            running += self._degree[v] // 2
        self.num_qubits = running

        cache: dict = {}
        self._modes: list = [None] * graph.num_modes
        for v in active:
            d = self._degree[v]
            if d not in cache:
                cache[d] = _local_mode_provider(local_modes, d)
            shift = offset[v]
            self._modes[v] = [(c, (x << shift, z << shift))
                              for c, (x, z) in cache[d]]

        self._eps = {edge: 1 for edge in graph.edges}
        self._fix_orientation(graph)

    def _raw_a_word(self, i, j):  # i < j, before the orientation sign
        p = self._neighbors[i].index(j)
        q = self._neighbors[j].index(i)
        ci, wi = self._modes[i][p]
        cj, wj = self._modes[j][q]
        return _word_mul(ci, wi, cj, wj)

    def a_word(self, i, j):  # i < j
        coeff, word = self._raw_a_word(i, j)
        return self._eps[(i, j)] * coeff, word

    def _loop_coeff(self, cycle):
        length = len(cycle) - 1
        coeff, word = 1.0 + 0j, (0, 0)
        for u, v in zip(cycle, cycle[1:]):
            ca, aw = self.a_word(min(u, v), max(u, v))
            if u > v:
                ca = -ca
            coeff, word = _word_mul(coeff, word, ca, aw)
        return coeff * (1j)**length

    def _fix_orientation(self, graph):
        """Set each chord's orientation so its fundamental loop operator is
        ``+1`` on the even-parity code space -- the paper's "suitable choice"
        of edge orientations. Each chord lies in exactly one fundamental cycle,
        so the choices are independent."""
        tree, chords = _spanning_forest(graph)
        for (i, j) in chords:
            cycle = _tree_path(tree, graph.num_modes, j, i) + [j]
            if self._loop_coeff(cycle).real < 0:
                self._eps[(i, j)] = -self._eps[(i, j)]

    def b_word(self, i):
        coeff, word = 1.0 + 0j, (0, 0)
        for c, w in self._modes[i]:
            coeff, word = _word_mul(coeff, word, c, w)
        coeff *= (-1j)**(self._degree[i] // 2)
        return coeff, word

    def stabilizers(self):
        return _gse_stabilizer_words(self)


def _gse_stabilizer_words(backend):
    """Loop stabilizers ``A(zeta) = i^s prod A`` (paper Eq 16) -- one per
    fundamental cycle. Each is Hermitian and the code subspace is its +1
    eigenspace (Eq 17, App A Property 4); no extra sign fix is needed, the
    ``i^s`` of the definition supplies it (``s`` = cycle length)."""
    tree, chords = _spanning_forest(backend)
    stabilizers = []
    for (i, j) in chords:
        cycle = _tree_path(tree, backend.num_modes, j, i) + [j]
        length = len(cycle) - 1
        coeff, word = 1.0 + 0j, (0, 0)
        for u, v in zip(cycle, cycle[1:]):
            ca, aw = backend.a_word(min(u, v), max(u, v))
            if u > v:  # directed edge: A_{u,v} = -A_{v,u}
                ca = -ca
            coeff, word = _word_mul(coeff, word, ca, aw)
        coeff *= (1j)**length
        stabilizers.append((coeff, word))
    return stabilizers


# --- Even-degree padding and the error-correction graph conditions ---------


def _pad_even_degree(num_modes, edges, min_degree):
    """Add zero-coefficient dummy edges so every active vertex has even degree
    (and degree >= ``min_degree``) -- the Generalized Superfast Encoding places
    ``d/2`` qubits per vertex, so every vertex degree must be even."""
    edges = set(edges)
    active = sorted({m for e in edges for m in e})
    degree = {v: 0 for v in active}
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1

    def nonadjacent(u, v):
        return u != v and (min(u, v), max(u, v)) not in edges

    limit = len(active)**2 + 16
    for _ in range(limit):
        need = [v for v in active if degree[v] < min_degree or degree[v] % 2]
        if not need:
            return edges
        need.sort(key=lambda v: (degree[v], v))
        u = need[0]
        partners = [
            v for v in active
            if nonadjacent(u, v) and (degree[v] < min_degree or degree[v] % 2)
        ]
        partners = partners or [v for v in active if nonadjacent(u, v)]
        if not partners:
            break
        partners.sort(key=lambda v: (degree[v], v))
        v = partners[0]
        edges.add((min(u, v), max(u, v)))
        degree[u] += 1
        degree[v] += 1
    raise ValueError(
        f"could not pad the interaction graph to even degree >= {min_degree} "
        "with a simple graph; supply interaction_graph with explicit "
        "(zero-coefficient) dummy edges.")


def _is_three_connected(num_modes, edges):
    """Does the graph stay connected after removing any two vertices?"""
    active = sorted({m for e in edges for m in e})
    if len(active) < 4:
        return False
    adjacency = {v: set() for v in active}
    for (i, j) in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for removed in itertools.combinations(active, 2):
        remaining = [v for v in active if v not in removed]
        seen = {remaining[0]}
        stack = [remaining[0]]
        allowed = set(remaining)
        while stack:
            x = stack.pop()
            for y in adjacency[x]:
                if y in allowed and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != len(remaining):
            return False
    return True


def _validate_error_correcting_graph(graph):
    """Theorem 1 conditions: even vertex degree >= 6 and 3-connected (with at
    most two edges between any pair, which a simple graph satisfies)."""
    active = {m for e in graph.edges for m in e}
    degree = {v: 0 for v in active}
    for (i, j) in graph.edges:
        degree[i] += 1
        degree[j] += 1
    bad = sorted(v for v in active if degree[v] < 6 or degree[v] % 2)
    if bad:
        raise ValueError(
            "error_correcting requires even vertex degree >= 6; vertices "
            f"{bad} do not meet it after padding. Supply interaction_graph "
            "with enough (zero-coefficient) edges.")
    if not _is_three_connected(graph.num_modes, graph.edges):
        raise ValueError(
            "error_correcting requires a 3-connected interaction graph "
            "(Theorem 1); supply interaction_graph to make it 3-connected.")


def _encoding_backend(one_body, two_body, tolerance, interaction_graph,
                      local_modes):
    if local_modes == "standard":
        graph = _build_graph(one_body, two_body, tolerance, interaction_graph)
        _warn_if_dense(graph)
        return _StandardBackend(graph)
    base = _build_graph(one_body, two_body, tolerance, interaction_graph)
    min_degree = 6 if local_modes == "error_correcting" else 2
    graph = _Graph(
        base.num_modes,
        sorted(_pad_even_degree(base.num_modes, set(base.edges), min_degree)))
    if local_modes == "error_correcting":
        _validate_error_correcting_graph(graph)
    _warn_if_dense(graph)
    return _GSEBackend(graph, local_modes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bravyi_kitaev_superfast(one_body_or_two_body,
                            two_body=None,
                            scalar_offset: float = 0.0,
                            tolerance: float = 1e-15,
                            interaction_graph=None,
                            local_modes="binary_tree"):
    """Bravyi-Kitaev Superfast transform of fermionic integrals.

    A locality-preserving fermion-to-qubit mapping that places the qubits by
    the Hamiltonian's interaction graph: local terms compile to bounded-weight
    Pauli operators (set by the graph degree), the payoff over Jordan-Wigner's
    O(N) strings when the graph is sparse. The qubit count is the number of
    graph edges (``> N`` for dense Hamiltonians -- this mapping is for sparse
    ones).

    Accepts an ``(n, n)`` one-body tensor, optionally with an ``(n, n, n, n)``
    two-body tensor, or a two-body tensor alone; entries are the coefficients
    of ``adag_i a_j`` and ``adag_i adag_j a_k a_l``. Arbitrary (complex,
    non-symmetric) one-body and arbitrary two-body integrals are supported.
    ``scalar_offset`` is added as an identity term; entries and compiled terms
    below ``tolerance`` are dropped. Returns a ``cudaq.SpinOperator``.

    ``local_modes`` selects the construction (Setia, Bravyi, Mezzacapo,
    Whitfield, Phys. Rev. Research 1, 033033, generalizing Bravyi-Kitaev,
    Annals of Physics 298):

    - ``"binary_tree"`` (default): the generalized encoding with Fenwick-tree
      local Majorana modes -- ``O(log d)`` Pauli weight in the graph degree
      ``d`` (vertex operators of weight 1). Requires even vertex degree; the
      graph is padded with zero-coefficient dummy edges when needed.
    - ``"error_correcting"``: generalized encoding whose code corrects any
      single-qubit error on a 3-connected graph of even degree ``>= 6`` (the
      graph is validated / padded accordingly).
    - ``"standard"``: the original edge encoding (one qubit per edge; one
      Z-string vertex operator, dressed-X edge operator), of ``O(d)`` weight.
    - a custom provider (callable ``degree -> modes`` or mapping): the caller's
      own local Majorana modes, each a Pauli string or ``(coeff, (x, z))`` word
      on ``degree/2`` qubits, validated to obey the Majorana algebra.

    ``interaction_graph`` optionally pins the edge set (an iterable of
    ``(i, j)`` mode pairs) instead of inferring it from the nonzero terms -- to
    control the qubit layout or to add extra edges. It must be a superset of
    the edges the Hamiltonian requires; the edge ordering fixes the qubit
    indexing.

    The interaction graph over the modes that carry a term must be
    **connected** (fermion parity is fixed per connected component, so a
    disconnected graph would not map to a single global-parity sector); a
    disconnected graph or an isolated number-only mode raises, pointing at
    ``interaction_graph`` to add connecting edges. Hermiticity of the input is
    the caller's responsibility, as for :func:`jordan_wigner`.

    Dense two-body couplings densify the interaction graph and emit a
    ``UserWarning`` (no locality advantage there). Use
    :func:`bravyi_kitaev_superfast_stabilizers` for the loop stabilizers that
    fix the code subspace.
    """
    one_body, two_body_arr, _ = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    backend = _encoding_backend(one_body, two_body_arr, tolerance,
                                interaction_graph, local_modes)
    return _compile(backend, one_body, two_body_arr, scalar_offset, tolerance)


def bravyi_kitaev_superfast_stabilizers(one_body_or_two_body,
                                        two_body=None,
                                        tolerance: float = 1e-15,
                                        interaction_graph=None,
                                        local_modes="binary_tree"):
    """Loop stabilizers of the Superfast code subspace for the given integrals.

    Returns one ``cudaq.SpinOperator`` per independent cycle of the interaction
    graph (empty for a tree graph). Each is a Hermitian involution, commutes
    with the mapped Hamiltonian, and is sign-fixed so that the **code subspace
    is their joint +1 eigenspace** -- the physical (even fermion-parity /
    vacuum) sector, matching the same sector under Jordan-Wigner. Restricting
    the mapped Hamiltonian to that eigenspace recovers the fermionic spectrum.

    ``local_modes`` and ``interaction_graph`` are as in
    :func:`bravyi_kitaev_superfast` (use the same values for both so the
    stabilizers act on the matching qubits).
    """
    one_body, two_body_arr, _ = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    backend = _encoding_backend(one_body, two_body_arr, tolerance,
                                interaction_graph, local_modes)
    return [
        _to_spin_operator({word: coeff}, tolerance)
        for coeff, word in backend.stabilizers()
    ]
