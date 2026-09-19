# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bravyi-Kitaev Superfast (BKSF) fermion-to-qubit mapping (Setia-Whitfield,
arXiv:1712.00446).

An *edge* encoding: one qubit per edge of the interaction graph, with a code
subspace fixed by loop stabilizers. Local (adjacent-mode) fermionic terms map
to bounded-weight qubit operators, unlike Jordan-Wigner's O(N) strings.

This module reuses the encoding-agnostic (x, z)-word algebra and SpinOperator
assembly from ``_compilers`` but supplies its own term compiler, because BKSF
has no local per-mode ladder operator (it encodes the even/bilinear
subalgebra directly onto edge/vertex operators).
"""
from __future__ import annotations

import numpy as np

# The (x, z) symplectic-word algebra and SpinOperator assembly are generic.
from ._compilers import _word_product, _to_spin_operator, _validate_tensors


# ---------------------------------------------------------------------------
# Interaction graph
# ---------------------------------------------------------------------------

def _interaction_edges(one_body, two_body, tolerance):
    """Undirected edges {i, j}, i < j, of modes coupled by a nonzero term."""
    n = one_body.shape[0]
    edges = set()
    for i in range(n):
        for j in range(n):
            if i != j and abs(one_body[i, j]) > tolerance:
                edges.add((min(i, j), max(i, j)))
    if two_body is not None:
        nz = np.argwhere(np.abs(two_body) > tolerance)
        for i, j, k, l in nz:
            for a, b in ((i, j), (i, k), (i, l), (j, k), (j, l), (k, l)):
                if a != b:
                    edges.add((min(a, b), max(a, b)))
    return sorted(edges)


class _Graph:
    """Interaction graph with a fixed edge->qubit indexing and incidence."""

    def __init__(self, num_modes, edges):
        self.num_modes = num_modes
        self.edges = list(edges)                       # qubit q <-> edges[q]
        self.qubit_of = {e: q for q, e in enumerate(self.edges)}
        # incident edge-qubits per vertex, in qubit order
        self.incident = [[] for _ in range(num_modes)]
        for q, (i, j) in enumerate(self.edges):
            self.incident[i].append(q)
            self.incident[j].append(q)

    def edge_qubit(self, i, j):
        return self.qubit_of[(min(i, j), max(i, j))]


# ---------------------------------------------------------------------------
# Edge / vertex operators as (x, z) words
# ---------------------------------------------------------------------------

def _mask(qubits):
    m = 0
    for q in qubits:
        m |= 1 << q
    return m


def _b_word(graph, i):
    """Vertex operator B_i = product of Z over edges incident to i.
    Represents the number: n_i = (I - B_i) / 2."""
    return (0, _mask(graph.incident[i]))            # (x=0, z=incident)


def _a_word(graph, i, j):
    """Edge operator A_ij (i<j) = X_{(i,j)} dressed with Z on incident edges
    (to i and to j) with a smaller qubit index, so shared-vertex A's
    anticommute."""
    i, j = min(i, j), max(i, j)
    p = graph.edge_qubit(i, j)
    z_edges = [q for q in graph.incident[i] if q < p]
    z_edges += [q for q in graph.incident[j] if q < p]
    return (1 << p, _mask(z_edges))                 # (x=edge, z=dressing)


def _anticommute(w1, w2):
    """True if two Pauli words anticommute (symplectic inner product odd)."""
    x1, z1 = w1
    x2, z2 = w2
    return ((x1 & z2).bit_count() + (z1 & x2).bit_count()) & 1 == 1


def _verify_algebra(graph, tolerance=1e-12):
    """Check the BKSF operator relations; returns a list of violations."""
    bad = []
    n = graph.num_modes
    B = [_b_word(graph, i) for i in range(n)]
    A = {(i, j): _a_word(graph, i, j) for (i, j) in graph.edges}
    # 1. {A_ij, B_i} = {A_ij, B_j} = 0 ; [A_ij, B_k] = 0 otherwise
    for (i, j), a in A.items():
        if not _anticommute(a, B[i]):
            bad.append(f"A{i,j} should anticommute B{i}")
        if not _anticommute(a, B[j]):
            bad.append(f"A{i,j} should anticommute B{j}")
        for k in range(n):
            if k not in (i, j) and _anticommute(a, B[k]):
                bad.append(f"A{i,j} should commute B{k}")
    # 2. A_ij, A_kl anticommute iff they share exactly one vertex
    es = list(A.keys())
    for a_idx in range(len(es)):
        for b_idx in range(a_idx + 1, len(es)):
            e1, e2 = es[a_idx], es[b_idx]
            shared = len(set(e1) & set(e2))
            anti = _anticommute(A[e1], A[e2])
            if shared == 1 and not anti:
                bad.append(f"A{e1},A{e2} share a vertex -> should anticommute")
            if shared == 0 and anti:
                bad.append(f"A{e1},A{e2} disjoint -> should commute")
    return bad
