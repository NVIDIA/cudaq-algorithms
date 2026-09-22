# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LCU block encoding of a sparse real symmetric matrix from its data.

``SparseLCUEncoding`` is the data-driven (V2) sparse encoding: the user
supplies the matrix entries and the encoding builds an alias-sampled
PREPARE over the term weights and a unary-iteration SELECT over the term
unitaries. The encoded block is ``H / alpha`` with ``alpha`` the *term
one-norm* (below) — the construction that wins over the oracle-based V1
(``alpha = d_padded * h``) whenever the matrix has one dominant entry and
many small ones, and that handles negative diagonal entries V1 provably
cannot.

The ``U_A = PREPARE-dagger . SELECT . PREPARE`` structure is the standard
linear-combination-of-unitaries (LCU) block encoding (Childs and Wiebe,
"Hamiltonian simulation using linear combinations of unitary operations",
arXiv:1202.5822; Berry, Childs, Cleve, Kothari and Somma, "Simulating
Hamiltonian dynamics with a truncated Taylor series", arXiv:1412.4687):
PREPARE loads the term weights as amplitudes, SELECT applies the indexed
term unitary, and the un-PREPARE projects the weighted sum into the
block. The PREPARE (coherent alias sampling) and SELECT (unary iteration)
are the ``cudaq_algorithms.primitives`` gadgets of Babbush et al.
(arXiv:1805.03662), cited where they are implemented.

Term decomposition and the one-norm bookkeeping
-----------------------------------------------

Every term unitary is Hermitian and involutory, so SELECT is a Hermitian
involution and the whole encoding is exactly self-adjoint (see below).
Over the upper triangle plus diagonal of the (real symmetric) input:

- Off-diagonal pair ``(i, j)``, ``i < j``: the Hermitian *move*
  ``|i><j| + |j><i|`` is **not** unitary (it vanishes off the pair), so
  the pair splits into two unitaries with the identity remainder carried
  with opposite signs and cancelled exactly::

      T   = |i><j| + |j><i| + (I - |i><i| - |j><j|)   (transposition)
      T'  = |i><j| + |j><i| - (I - |i><i| - |j><j|)
      H_ij (|i><j| + |j><i|) = (H_ij / 2) T + (H_ij / 2) T'

  Weight ``|H_ij| / 2`` each, so the pair — i.e. *both* matrix entries
  ``(i, j)`` and ``(j, i)`` — costs ``|H_ij|`` of one-norm: a factor 2
  below the naive elementwise bookkeeping ``2 |H_ij|`` per pair. The
  cancellation is exact for the ideal weights; discretization rounds the
  two weights independently, so the residual identity leaks onto the
  diagonal at the (test-derived) per-bin discretization bound.
- Diagonal entry ``i``: ``H_ii |i><i| = (H_ii / 2) I - (H_ii / 2) Z_i``
  with the reflection ``Z_i = I - 2 |i><i|`` — weight ``|H_ii| / 2``
  each, ``|H_ii|`` per entry. Signs (including negative diagonals) ride
  on the SELECT branch as a leaf-line Z phase, never on the weights.

Total: ``alpha = sum_{i<j} |H_ij| + sum_i |H_ii|``. The alias tables
realize the normalized weights exactly as ``table_probabilities`` (which
sum to 1), so the one-norm of the *discretized* weights equals the ideal
one-norm identically; ``alpha`` reports that common value and
``discretization_bound`` bounds the per-term probability rounding.

Circuit
-------

``U_A = PREPARE-dagger . SELECT . PREPARE`` over ``ancilla = [index(m) |
garbage(2m + 2 mu + 2) | select_work]``:

- PREPARE / PREPARE-dagger: ``AliasSamplingPrepare`` over the term
  weights (with garbage — sound only in this symmetric sandwich; see
  ``cudaq_algorithms.primitives._alias_sampling``). Never placed under
  ``cudaq.control``.
- SELECT: a flat unary-iteration walk
  (``cudaq_algorithms.primitives._unary_iteration``) over the
  term index. Term ``k``'s body realizes its unitary with the extended
  gate vocabulary: an uncontrolled CNOT/X conjugation maps the pair
  ``{i, j}`` to two states differing in one pivot bit, an AND ladder
  folds the remaining ``num_system - 1`` pattern bits into ``work``, and
  the leaf-controlled core is one Toffoli-depth flip
  ``x.ctrl(leaf, match, pivot)`` (plus ``cz`` / leaf-``z`` phases for
  ``T'``, ``Z_i``, and negative terms). Cost per off-diagonal term:
  ``O(num_system)`` CNOTs/Toffolis on top of the walk's own tree
  Toffolis (linear in ``K``; the exact count is the fused-walk figure
  documented in ``cudaq_algorithms.primitives._unary_iteration`` and
  reported by ``toffoli_count``). SELECT's ladder reuses the PREPARE QROM
  ladder qubits (both are clean between uses); the AND-ladder work sits
  in dedicated clean scratch at the ancilla tail (CUDA-Q cannot
  deallocate mid-circuit, so it is folded into ``num_ancilla`` exactly
  as in ``_sparse_oracle``).

Self-adjointness: every term unitary squares to the identity and SELECT
is block-diagonal over the index, so ``SELECT^2 = I`` and ``U_A =
P-dagger S P`` is exactly self-adjoint and involutory **on states whose
ancilla register enters in ``|0...0>``** — the domain every consumer
supplies, and the domain the involution test pins. It is NOT a global
matrix identity: SELECT's unary-iteration walk is an involution only
when its ladder enters clean, so composing ``U_A`` by hand on a dirty
ancilla register silently breaks ``U_A^2 = I`` (``SparseOracleEncoding``'s
identical claim, by contrast, is global). ``Walk``'s Chebyshev powers
and ``QSVT``'s reuse of ``apply_kernel`` for adjoint directions hold
with no separate adjoint factory. Walk reflections cover the **full**
ancilla register, garbage included, and this is a requirement, not a
convenience: the block is the all-``|0...0>`` ancilla subspace, and
because SELECT leaves the PREPARE garbage entangled with the system the
index-``|0>`` sector is *not* garbage-free, so an index-only reflection
would misidentify the block. (This is the one place the LCU differs from
``_sparse_oracle``, whose scratch is a pure function of the system and is
uncomputed to ``|0>`` in every sector, so there the full and block
reflections genuinely coincide.)

Input forms: a dense 2-D array, a ``scipy.sparse`` matrix (anything with
``tocoo``), or a ``(rows, cols, vals)`` tuple of parallel sequences with
an explicit ``dim=`` (duplicate triples sum, COO-style). Entries must be
real (complex input is deferred; loudly rejected) and the matrix
symmetric — for a general square ``A`` use ``from_general``, which
encodes the Hermitian dilation ``[[0, A], [A^T, 0]]`` built at the data
level (every entry of ``A`` lands off-diagonal, so negative diagonals of
``A`` are unproblematic). Non-power-of-two dimensions are zero-padded to
``2^num_system``: the encoded operator is the padded matrix (its
spectrum is H's plus exact zeros), and ``alpha`` is unaffected.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cudaq
import numpy as np

from ..common_kernels import (_validate_power, controlled_reflect_about_zero,
                              reflect_about_zero)
from ..primitives import AliasSamplingPrepare, unary_iteration_kernels
from ._sparse_oracle import _noop, _retain

if TYPE_CHECKING:
    from ..block_encoding import Kernel

__all__ = ["SparseLCUEncoding"]

# Term kinds (the dense definitions live in the module docstring).
_IDENTITY = "identity"
_REFLECTION = "reflection"  # Z_i = I - 2|i><i|
_TRANSPOSITION = "transposition"  # T
_REFLECTED_TRANSPOSITION = "reflected_transposition"  # T' = T (2P - I)


def _real_entry(value, i: int, j: int) -> float:
    entry = complex(value)
    if abs(entry.imag) > 1e-12:
        raise ValueError(
            f"matrix entry ({i}, {j}) is complex ({value!r}): complex "
            "matrices are not supported (deferred; encode real data)")
    if not math.isfinite(entry.real):
        raise ValueError(f"matrix entry ({i}, {j}) must be finite, "
                         f"got {value!r}")
    return entry.real


def _coerce_entries(matrix, dim: int | None) -> tuple[int, dict]:
    """Normalize any accepted input form to ``(dim, {(i, j): value})``.

    Zero entries are dropped; duplicate triples sum (COO semantics).
    """
    if hasattr(matrix, "tocoo"):  # scipy.sparse, without a hard dependency
        coo = matrix.tocoo()
        if coo.shape[0] != coo.shape[1]:
            raise ValueError(f"matrix must be square, got shape {coo.shape}")
        size = int(coo.shape[0])
        rows, cols, vals = coo.row, coo.col, coo.data
    elif isinstance(matrix, tuple) and len(matrix) == 3:
        rows, cols, vals = matrix
        if dim is None:
            raise ValueError(
                "dim is required with (rows, cols, vals) triples (the "
                "matrix dimension cannot be inferred from the entries)")
        if not (len(rows) == len(cols) == len(vals)):
            raise ValueError(
                f"rows, cols, and vals must have the same length, got "
                f"{len(rows)}, {len(cols)}, {len(vals)}")
        size = int(dim)
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError(
                f"matrix must be square, got shape {dense.shape}; accepted "
                "forms are a dense 2-D array, a scipy.sparse matrix, or a "
                "(rows, cols, vals) tuple with dim=")
        size = int(dense.shape[0])
        rows, cols = np.nonzero(dense)
        vals = dense[rows, cols]
    if dim is not None and int(dim) != size:
        raise ValueError(f"dim={dim} does not match the matrix "
                         f"dimension {size}")
    if size < 1:
        raise ValueError("matrix must have dimension >= 1")

    entries: dict = {}
    for i, j, v in zip(rows, cols, vals):
        if int(i) != i or int(j) != j:
            raise ValueError(
                f"row/column indices must be integers, got ({i!r}, {j!r})")
        i, j = int(i), int(j)
        if not (0 <= i < size and 0 <= j < size):
            raise ValueError(f"entry index ({i}, {j}) is out of range for "
                             f"dimension {size}")
        entries[(i, j)] = entries.get((i, j), 0.0) + _real_entry(v, i, j)
    return size, {key: v for key, v in entries.items() if v != 0.0}


def _and_ladder(bits: list) -> tuple[list, int]:
    """Body ops folding the AND of the given target bits into work.

    Returns ``(ops, match)`` with ``work[match]`` holding the AND; the
    caller uncomputes by replaying the ops reversed (all self-inverse).
    Requires ``len(bits) >= 1``.
    """
    if len(bits) == 1:
        return [("copy_tw", bits[0], 0)], 0
    ops = [("and_tt", bits[0], bits[1], 0)]
    for index in range(2, len(bits)):
        ops.append(("and_wt", index - 2, bits[index], index - 1))
    return ops, len(bits) - 2


def _pair_body(i: int, j: int, reflected: bool, sign: int,
               num_system: int) -> list:
    """SELECT body for ``T`` (or ``T'`` when ``reflected``) on pair (i, j).

    A CNOT conjugation off the lowest differing (pivot) bit maps the pair
    to two states differing only in the pivot; the remaining bits then
    share one pattern, X-conjugated to all-ones and AND-folded into
    ``work``. The leaf-controlled core is the pivot flip (plus the
    diagonal phases realizing ``T' = T (2P - I)`` and the term sign).
    """
    differing = i ^ j
    pivot = (differing & -differing).bit_length() - 1
    rest = [
        t for t in range(num_system) if (differing >> t) & 1 and t != pivot
    ]
    conjugated_i = i ^ (sum(1 << t for t in rest) if (i >> pivot) & 1 else 0)
    others = [t for t in range(num_system) if t != pivot]

    pre = [("free_cx", pivot, t) for t in rest]
    pre += [("free_x", t) for t in others if not ((conjugated_i >> t) & 1)]
    if others:
        match_ops, match = _and_ladder(others)
    else:
        match_ops, match = [], -1

    core: list = []
    z_parity = sign < 0
    if reflected and others:
        # T' = T (2P - I): a cz on the pattern match plus a global -1 on
        # the branch. With no pattern bits (num_system == 1) the pair
        # spans the whole space, P = I, and the factor is the identity.
        core.append(("z_w", match))
        z_parity = not z_parity
    if z_parity:
        core.append(("sign", ))
    core.append(("x_w", match, pivot) if others else ("x", pivot))
    return (pre + match_ops + core + list(reversed(match_ops)) +
            list(reversed(pre)))


def _reflection_body(i: int, sign: int, num_system: int) -> list:
    """SELECT body for ``Z_i = I - 2|i><i|`` (a -1 phase on ``|i>`` alone)."""
    pre = [("free_x", t) for t in range(num_system) if not ((i >> t) & 1)]
    if num_system == 1:
        match_ops, core = [], [("z", 0)]
    else:
        match_ops, match = _and_ladder(list(range(num_system)))
        core = [("z_w", match)]
    if sign < 0:
        core.append(("sign", ))
    return pre + match_ops + core + list(reversed(match_ops)) + list(
        reversed(pre))


def _identity_body(sign: int) -> list:
    return [("sign", )] if sign < 0 else []


def _term_body(kind: str, i: int, j: int, sign: int, num_system: int) -> list:
    if kind == _IDENTITY:
        return _identity_body(sign)
    if kind == _REFLECTION:
        return _reflection_body(i, sign, num_system)
    return _pair_body(i, j, kind == _REFLECTED_TRANSPOSITION, sign, num_system)


def _first_asymmetry(entries: dict) -> tuple | None:
    """The first ``(i, j, value, partner)`` violating symmetry, or None.

    Shared by the constructor (which raises on it) and ``encode_sparse``
    (which dilates instead). Symmetry is judged two ways, both of which
    the symmetric constructions require: an off-diagonal entry must have a
    present mirror (missing support is an asymmetry — the constructions
    index ``(j, i)`` directly), and the two values must agree to a
    *matrix-relative* tolerance ``1e-12 * max|H|`` (a fixed absolute floor
    would call a uniformly tiny but lopsided matrix symmetric).
    """
    scale = max((abs(v) for v in entries.values()), default=1.0)
    tol = 1e-12 * scale
    for (i, j), value in sorted(entries.items()):
        if i == j:
            continue
        partner = entries.get((j, i))
        if partner is None or abs(value - partner) > tol:
            return i, j, value, 0.0 if partner is None else partner
    return None


def _build_terms(entries: dict) -> list:
    """The LCU term list ``(kind, i, j, weight, sign)`` for symmetric
    entries — the one-norm bookkeeping of the module docstring. Shared by
    the constructor and ``encode_sparse``'s classical pricing, so the
    priced ``alpha`` (the weight sum) is identically the built one."""
    terms: list = []
    for (i, j) in sorted(entries):
        value = entries[(i, j)]
        weight = abs(value) / 2.0
        sign = 1 if value > 0.0 else -1
        if i == j:
            terms.append((_IDENTITY, i, i, weight, sign))
            terms.append((_REFLECTION, i, i, weight, -sign))
        elif i < j:
            terms.append((_TRANSPOSITION, i, j, weight, sign))
            terms.append((_REFLECTED_TRANSPOSITION, i, j, weight, sign))
    return terms


def _bodies_and_work(terms: list, num_system: int) -> tuple[list, int]:
    """SELECT bodies for the terms plus the required work width (>= 1: an
    empty work view must never cross the kernel boundary,
    cuda-quantum#4847). Classical — shared with ``encode_sparse``'s
    pricing, which needs the exact width without minting kernels."""
    bodies = [
        _term_body(kind, i, j, sign, num_system)
        for kind, i, j, _, sign in terms
    ]
    num_work = 1
    for body in bodies:
        for item in body:
            if item[0] in ("and_tt", "and_wt"):
                num_work = max(num_work, item[3] + 1)
            elif item[0] in ("copy_tw", ):
                num_work = max(num_work, item[2] + 1)
    return bodies, num_work


class SparseLCUEncoding:
    """Block encoding of ``H / alpha`` from the entries of a real
    symmetric matrix ``H`` (see the module docstring).

    Parameters
    ----------
    matrix
        A dense 2-D array, a ``scipy.sparse`` matrix, or a
        ``(rows, cols, vals)`` tuple of parallel sequences.
    mu
        Alias-table keep precision in bits (>= 1); the per-term
        probability rounding is bounded by ``discretization_bound``.
    dim
        Matrix dimension — required with triples, optional (checked)
        otherwise.

    Satisfies the ``BlockEncoding`` protocol except ``select_observable``
    (the term unitaries are not Pauli words, so the odd-moment observable
    trick is unavailable), exactly as ``SparseOracleEncoding``. Measure
    Chebyshev data through ``Walk(...).walk_kernel`` powers, not
    ``Walk.moment``: the even-moment reflection observable expands to
    ``2^num_ancilla`` Pauli terms, astronomical at this encoding's
    ancilla widths (``Walk`` refuses fast rather than hanging).
    """

    def __init__(self, matrix, *, mu: int = 8, dim: int | None = None):
        size, entries = _coerce_entries(matrix, dim)
        if not entries:
            raise ValueError("matrix has no nonzero entries: nothing to "
                             "encode")
        asymmetry = _first_asymmetry(entries)
        if asymmetry is not None:
            i, j, value, partner = asymmetry
            raise ValueError(
                f"matrix is not symmetric at ({i}, {j}): {value} vs "
                f"{partner}; encode a general square matrix through "
                "SparseLCUEncoding.from_general (Hermitian dilation)")

        num_system = max(1, (size - 1).bit_length())
        terms = _build_terms(entries)
        # Pin the select-variant QROM: it freezes num_garbage at
        # 2*num_index + 2*mu + 4 and keeps ladder_offset/num_ladder ==
        # num_index, the geometry sparse_lcu_apply relies on when SELECT
        # reuses the PREPARE QROM ladder (and the closed register-budget
        # formula in _factory). Adopting variant="auto" (select_swap for
        # large tables) would need _factory to price via the QROM cost
        # model instead.
        preparation = AliasSamplingPrepare([t[3] for t in terms],
                                           mu,
                                           variant="select")
        bodies, num_work = _bodies_and_work(terms, num_system)
        select = unary_iteration_kernels(preparation.num_index,
                                         len(terms),
                                         lambda k: bodies[k],
                                         include_adjoint=False,
                                         num_work=num_work)
        controlled_select = unary_iteration_kernels(preparation.num_index,
                                                    len(terms),
                                                    lambda k: bodies[k],
                                                    controlled=True,
                                                    include_adjoint=False,
                                                    num_work=num_work)

        self._dim = size
        self._num_system = num_system
        self._terms = tuple(terms)
        self._mu = int(mu)
        self._preparation = preparation
        self._num_work = num_work
        self._select = select
        self._controlled_select = controlled_select
        self._build_kernels()

    # ------------------------------------------------------------------
    # Kernel construction (all data captured here, factories are data-free)
    # ------------------------------------------------------------------

    def _build_kernels(self) -> None:
        # Unpack into scalar locals: tuples (and self) cannot be
        # closure-captured into kernels.
        m = self._preparation.num_index
        g = self._preparation.num_garbage
        l0 = self._preparation.ladder_offset
        w = self._num_work
        prepare = self._preparation.kernel()
        unprepare = self._preparation.adjoint_kernel()
        select = self._select.kernel
        controlled_select = self._controlled_select.kernel

        @cudaq.kernel
        def sparse_lcu_apply(ancilla: cudaq.qview, system: cudaq.qview):
            """U_A = PREPARE-dagger SELECT PREPARE (self-adjoint,
            involutory). SELECT's ladder reuses the PREPARE QROM ladder
            (clean between the sandwich halves)."""
            prepare(ancilla.front(m), ancilla[m:m + g])
            select(ancilla.front(m), ancilla[m + l0:m + l0 + m], system,
                   ancilla[m + g:m + g + w])
            unprepare(ancilla.front(m), ancilla[m:m + g])

        @cudaq.kernel
        def sparse_lcu_controlled_apply(control_and_ancilla: cudaq.qview,
                                        system: cudaq.qview):
            """Controlled U_A = PREPARE-dagger (controlled SELECT) PREPARE.

            Only the Hermitian middle factor carries the external control
            (the uncontrolled PREPARE pair cancels at control ``|0>``); the
            control reaches SELECT as a one-qubit view, so no control set
            mixes a qview with a bare qubit, and PREPARE — which calls
            sub-kernels — never sits under a control variant.
            """
            prepare(control_and_ancilla[1:1 + m],
                    control_and_ancilla[1 + m:1 + m + g])
            controlled_select(control_and_ancilla.front(1),
                              control_and_ancilla[1:1 + m],
                              control_and_ancilla[1 + m + l0:1 + m + l0 + m],
                              system,
                              control_and_ancilla[1 + m + g:1 + m + g + w])
            unprepare(control_and_ancilla[1:1 + m],
                      control_and_ancilla[1 + m:1 + m + g])

        @cudaq.kernel
        def sparse_lcu_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W = R U_A (block encodes -H/alpha); R covers the full
            ancilla register, garbage included (sound in the sandwiched
            frame — see the module docstring)."""
            sparse_lcu_apply(ancilla, system)
            reflect_about_zero(ancilla)

        @cudaq.kernel
        def sparse_lcu_adjoint_walk_step(ancilla: cudaq.qview,
                                         system: cudaq.qview):
            """W-dagger = U_A R (U_A is self-adjoint)."""
            reflect_about_zero(ancilla)
            sparse_lcu_apply(ancilla, system)

        @cudaq.kernel
        def sparse_lcu_controlled_walk_step(control_and_ancilla: cudaq.qview,
                                            system: cudaq.qview):
            sparse_lcu_controlled_apply(control_and_ancilla, system)
            controlled_reflect_about_zero(control_and_ancilla)

        @cudaq.kernel
        def sparse_lcu_controlled_adjoint_walk_step(
                control_and_ancilla: cudaq.qview, system: cudaq.qview):
            controlled_reflect_about_zero(control_and_ancilla)
            sparse_lcu_controlled_apply(control_and_ancilla, system)

        _retain(sparse_lcu_apply, sparse_lcu_controlled_apply,
                sparse_lcu_walk_step, sparse_lcu_adjoint_walk_step,
                sparse_lcu_controlled_walk_step,
                sparse_lcu_controlled_adjoint_walk_step)
        self._apply = sparse_lcu_apply
        self._controlled_apply = sparse_lcu_controlled_apply
        self._walk_step = sparse_lcu_walk_step
        self._adjoint_walk_step = sparse_lcu_adjoint_walk_step
        self._controlled_walk_step = sparse_lcu_controlled_walk_step
        self._controlled_adjoint_walk_step = (
            sparse_lcu_controlled_adjoint_walk_step)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def num_system(self) -> int:
        return self._num_system

    @property
    def num_ancilla(self) -> int:
        """All ancillas: index + PREPARE garbage + SELECT work scratch.

        The block is the all-ancilla-``|0...0>`` subspace, so consumers
        reflect about the **full** register, not the index alone: after
        SELECT the garbage is entangled with the system, so the
        index-``|0>`` sector still carries nonzero garbage and an
        index-only reflection would misidentify the block. The SELECT work
        scratch *is* ``|0>`` at reflection points but is still counted here
        because CUDA-Q cannot deallocate it mid-circuit.
        """
        return (self._preparation.num_index + self._preparation.num_garbage +
                self._num_work)

    @property
    def num_index(self) -> int:
        """Term-index register width (the block ancillas proper)."""
        return self._preparation.num_index

    @property
    def num_garbage(self) -> int:
        """PREPARE garbage width (entangled between the sandwich halves)."""
        return self._preparation.num_garbage

    @property
    def num_select_work(self) -> int:
        """Clean AND-ladder scratch for SELECT (``|0>`` at reflection points)."""
        return self._num_work

    @property
    def dim(self) -> int:
        """Original matrix dimension (before zero-padding to 2^num_system)."""
        return self._dim

    @property
    def mu(self) -> int:
        return self._mu

    @property
    def terms(self) -> tuple:
        """The LCU terms as ``(kind, i, j, weight, sign)`` tuples, in
        PREPARE bin order (bins ``>= len(terms)`` are zero-weight padding
        acting as the identity)."""
        return self._terms

    @property
    def alpha(self) -> float:
        """The term one-norm ``sum_{i<j} |H_ij| + sum_i |H_ii|``.

        Exact for the discretized weights too: ``table_probabilities``
        sum to 1 by construction, so the discretized one-norm
        ``sum(discretized_weights)`` equals the ideal one identically.
        """
        return self._preparation.lam

    @property
    def discretized_weights(self) -> np.ndarray:
        """The exactly realized per-bin weights ``alpha *
        table_probabilities`` (the dense-reference anchor)."""
        return self.alpha * self._preparation.table_probabilities

    @property
    def discretization_bound(self) -> float:
        """Per-bin probability rounding bound (see
        ``cudaq_algorithms.primitives._alias_sampling``)."""
        return self._preparation.discretization_bound

    @property
    def preparation(self) -> AliasSamplingPrepare:
        """The underlying alias-sampling PREPARE (tables and bounds)."""
        return self._preparation

    def __repr__(self) -> str:
        return (f"SparseLCUEncoding(dim={self.dim}, "
                f"terms={len(self.terms)}, mu={self.mu}, "
                f"system_qubits={self.num_system}, "
                f"ancilla_qubits={self.num_ancilla} "
                f"(index {self.num_index} + garbage {self.num_garbage} + "
                f"work {self.num_select_work}), "
                f"alpha={self.alpha:.6g})")

    # ------------------------------------------------------------------
    # Convenience factories (mirroring SparseOracleEncoding / PauliLCU)
    # ------------------------------------------------------------------

    def encode_kernel(self, state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying the full block encoding.

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state`` and the ancilla register (in
        ``|0...0>``) after it. With ``state_prep`` (a ``(qubits: qview)``
        kernel): a zero-argument kernel that allocates the system register
        in ``|0...0>``, runs ``state_prep`` on it, then applies the encoding.
        """
        apply_u = self._apply
        n_anc = self.num_ancilla
        n_sys = self.num_system

        if state_prep is not None:

            @cudaq.kernel
            def sparse_lcu_prep_encoded():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                apply_u(ancilla, system)

            _retain(sparse_lcu_prep_encoded)
            return sparse_lcu_prep_encoded

        @cudaq.kernel
        def sparse_lcu_encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            apply_u(ancilla, system)

        _retain(sparse_lcu_encoded)
        return sparse_lcu_encoded

    def walk_kernel(self,
                    power: int = 1,
                    state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying ``power`` qubitization walk steps.

        The all-zero-ancilla block of the result is T_power(-H/alpha)
        applied to the input state (the protocol PREPARE hook is trivial:
        the alias-sampling PREPARE lives inside U_A's sandwich). Input
        modes as in ``encode_kernel``.
        """
        step = self._walk_step
        n_anc = self.num_ancilla
        n_sys = self.num_system
        steps = _validate_power(power)

        if state_prep is not None:

            @cudaq.kernel
            def sparse_lcu_prep_walked():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                for _ in range(steps):
                    step(ancilla, system)

            _retain(sparse_lcu_prep_walked)
            return sparse_lcu_prep_walked

        @cudaq.kernel
        def sparse_lcu_walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            for _ in range(steps):
                step(ancilla, system)

        _retain(sparse_lcu_walked)
        return sparse_lcu_walked

    # ------------------------------------------------------------------
    # BlockEncoding protocol: data-free kernel factories
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: trivial — the alias-sampling PREPARE is
        part of ``apply_kernel``'s symmetric sandwich (its garbage is only
        sound there, never left dirty across consumer-visible points)."""
        return _noop

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: trivial (see ``prepare_kernel``)."""
        return _noop

    def apply_kernel(self) -> Kernel:
        """``(ancilla, system)``: the full block encoding U_A."""
        return self._apply

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0."""
        return self._controlled_apply

    def walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one qubitization walk step W = R U_A."""
        return self._walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W-dagger."""
        return self._adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        return self._controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled adjoint step."""
        return self._controlled_adjoint_walk_step

    # ------------------------------------------------------------------
    # Observable hooks
    # ------------------------------------------------------------------

    def select_observable(self) -> Any:
        """Unavailable for this encoding (LCU-over-Pauli-words hook)."""
        raise NotImplementedError(
            "SparseLCUEncoding does not provide select_observable: its "
            "term unitaries are transpositions and reflections, not "
            "computational-frame Pauli words, so the odd-Chebyshev-moment "
            "observable trick is unavailable for this encoding")

    # ------------------------------------------------------------------
    # Hermitian dilation of general (non-symmetric) matrices
    # ------------------------------------------------------------------

    @classmethod
    def from_general(cls,
                     matrix,
                     *,
                     mu: int = 8,
                     dim: int | None = None) -> "SparseLCUEncoding":
        """Encode the Hermitian dilation ``[[0, A], [A^T, 0]]`` of ``A``.

        ``A`` is a general real square matrix in any of the constructor's
        input forms; the dilation is assembled at the data level (each
        entry ``A_ij`` becomes the symmetric pair ``(i, half + j)`` /
        ``(half + j, i)``, ``half`` the power-of-two padding of ``A``'s
        dimension), so every entry — diagonal ones of ``A`` included —
        lands off-diagonal in the dilation, where signs are carried by
        the SELECT branches. The result acts on ``num_system(A) + 1``
        system qubits with the dilation's own term one-norm as ``alpha``.
        """
        size, entries = _coerce_entries(matrix, dim)
        if not entries:
            raise ValueError("matrix has no nonzero entries: nothing to "
                             "encode")
        half = 1 << max(1, (size - 1).bit_length())
        rows: list = []
        cols: list = []
        vals: list = []
        for (i, j), value in sorted(entries.items()):
            rows += [i, half + j]
            cols += [half + j, i]
            vals += [value, value]
        return cls((rows, cols, vals), mu=mu, dim=2 * half)
