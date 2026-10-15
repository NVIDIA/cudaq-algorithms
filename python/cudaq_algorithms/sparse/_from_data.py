# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QROM-backed sparse-access oracles built from matrix data.

``qrom_oracles`` compiles the entries of a real symmetric sparse matrix
into an ``OracleKernels`` bundle for ``SparseOracleEncoding`` — the
generic data-driven route onto the oracle (V1) construction, and the
path ``encode_sparse`` prices against the alias-LCU (V2) encoding.

Slot assignment: matchings, so every slot map is an involution
------------------------------------------------------------------

The oracle contract needs each per-slot column map ``c(s, .)`` to be a
permutation of the whole system space. The off-diagonal sparsity pattern
(an undirected graph — the matrix is symmetric) is greedily edge-colored
into *matchings*: slot ``s`` swaps the endpoints of every edge of
matching ``s`` and fixes everything else, which is an involution — hence
a permutation, its own inverse (``o_loc_adj`` is ``o_loc`` itself), and
its own reverse-direction slot (``slot_flip`` is the identity, every
slot self-paired). Nonzero diagonal entries get one extra slot whose map
is the identity. Greedy coloring may use more slots than the maximum
row sparsity ``d_row`` (never more than ``2 d_row - 1``); the honest
``alpha = d_padded * h`` pays the slots actually used, and
``encode_sparse`` prices with the same assignment.

Oracle circuits: one unary-iteration walk each, slot-dispatched
---------------------------------------------------------------

Both oracles are a single unary-iteration walk
(``cudaq_algorithms.primitives._unary_iteration``)
over the *system* register, with per-address body writes dispatched on
the slot through match qubits: slot-match bits (one multi-controlled X
per real slot) are computed into ``work``, every body gate is
``("x_w", s, t)`` — target flip controlled on the walk's leaf line
(address = this row) *and* match bit ``s`` (slot = ``s``) — and the
matches are uncomputed afterwards.

- ``o_val``: one walk XOR-loads, at row ``x`` under slot ``s``, the
  fixed-point angle of ``|H[x, c(s, x)]|`` (``quantized_angle``, the
  module-shared convention of ``_banded``), the sign bit
  (``H[x, c(s, x)] < 0``), and the upper bit (``x < c(s, x)``). X-only
  writes keyed on registers the walk never changes: an involution, so
  ``o_val_adj`` is ``o_val``.
- ``o_loc``: the walk cannot write into its own address register, so the
  in-place update ``|x> -> |c(s, x)>`` goes through an ``n``-qubit
  ``out`` scratch: walk once (``out ^= x XOR c(s, x)``), CNOT ``out``
  into the system register, walk again — the involution property
  ``delta(s, c(s, x)) = delta(s, x)`` makes the second walk clear
  ``out`` exactly. Applying the whole sequence twice returns ``x``:
  ``o_loc`` is an involution too.

Work register layout (width ``num_work = 2 n + d``): ``[out(n) |
ladder(n) | match(d)]`` — ``o_val`` uses the ladder and match regions
only. Negative *diagonal* entries are rejected loudly: the symmetric
two-sided assembler of ``_sparse_oracle`` provably cannot represent them
(the diagonal phase contribution is always ``+1``); shift the diagonal,
or use the LCU path (``SparseLCUEncoding`` / ``encode_sparse``), or
encode the Hermitian dilation. Non-symmetric input is rejected pointing
at the same escape hatches.
"""

from __future__ import annotations

import cudaq

from ._banded import quantized_angle
from ._sparse_lcu import _coerce_entries, _first_asymmetry
from ..primitives import unary_iteration_kernels
from ._sparse_oracle import OracleKernels, _retain

__all__ = ["qrom_oracles"]


def _assign_slots(entries: dict, size: int) -> tuple[list[dict], bool]:
    """Greedy edge coloring of the off-diagonal pattern into matchings.

    Returns ``(matchings, has_diagonal)``: each matching is a symmetric
    ``{i: j, j: i}`` dict (an involution restricted to its support), and
    ``has_diagonal`` flags whether a dedicated identity slot is needed.
    Deterministic (edges visited in sorted order). Classical and cheap —
    ``encode_sparse`` runs it for pricing without building kernels.
    """
    matchings: list[dict] = []
    for (i, j) in sorted(entries):
        if i < j:
            for matching in matchings:
                if i not in matching and j not in matching:
                    matching[i] = j
                    matching[j] = i
                    break
            else:
                matchings.append({i: j, j: i})
    has_diagonal = any(i == j for (i, j) in entries)
    return matchings, has_diagonal


def qrom_oracles(matrix,
                 value_bits: int,
                 *,
                 h: float | None = None,
                 dim: int | None = None) -> OracleKernels:
    """QROM-backed oracles for a real symmetric sparse matrix.

    Parameters
    ----------
    matrix
        A dense 2-D array, a ``scipy.sparse`` matrix, or a
        ``(rows, cols, vals)`` tuple of parallel sequences (the input
        forms of ``SparseLCUEncoding``). Must be symmetric with a
        non-negative diagonal (see the module docstring for the escape
        hatches).
    value_bits
        Fixed-point bits for the angle register.
    h
        Value normalization (defaults to ``max |H_ij|``); must satisfy
        ``h >= max |H_ij|``.
    dim
        Matrix dimension — required with triples, optional (checked)
        otherwise.
    """
    if int(value_bits) != value_bits or value_bits < 1:
        raise ValueError("value_bits must be a positive integer")
    value_bits = int(value_bits)
    size, entries = _coerce_entries(matrix, dim)
    if not entries:
        raise ValueError("matrix has no nonzero entries: nothing to encode")
    asymmetry = _first_asymmetry(entries)
    if asymmetry is not None:
        i, j, value, partner = asymmetry
        raise ValueError(
            f"matrix is not symmetric at ({i}, {j}): {value} vs {partner}; "
            "qrom_oracles builds direct symmetric oracles — encode a general "
            "square matrix through encode_sparse (which dilates at the data "
            "level) or SparseLCUEncoding.from_general")
    for (i, j), value in sorted(entries.items()):
        if i == j and value < 0.0:
            raise ValueError(
                f"diagonal entry ({i}, {i}) = {value} is negative: the "
                "symmetric two-sided oracle construction cannot represent "
                "negative diagonal elements (their phase contribution is "
                "always +1); shift the diagonal, or use the LCU path "
                "(SparseLCUEncoding / encode_sparse)")
    max_value = max(abs(v) for v in entries.values())
    if h is None:
        h = max_value
    h = float(h)
    if not h >= max_value:
        raise ValueError(f"h={h} is smaller than max |H_ij| = {max_value}: "
                         "the encoding requires h >= max|H_ij|")

    matchings, has_diagonal = _assign_slots(entries, size)
    num_loc = len(matchings)
    d = num_loc + (1 if has_diagonal else 0)
    n = max(1, (size - 1).bit_length())
    m = max(1, (d - 1).bit_length())
    d_padded = 1 << m

    # Classical per-(slot, row) tables: the o_loc XOR deltas and the
    # packed o_val words [angle(value_bits) | sign | upper].
    delta = [[0] * size for _ in range(num_loc)]
    val = [[0] * size for _ in range(d)]
    for s, matching in enumerate(matchings):
        for row, column in matching.items():
            value = entries[(row, column)]
            delta[s][row] = row ^ column
            word = quantized_angle(value, h, value_bits)
            if value < 0.0:
                word |= 1 << value_bits
            if row < column:
                word |= 1 << (value_bits + 1)
            val[s][row] = word
    if has_diagonal:
        for (i, j), value in entries.items():
            if i == j:
                val[num_loc][i] = quantized_angle(value, h, value_bits)

    def val_body(row: int) -> list:
        return [("x_w", s, t) for s in range(d) for t in range(value_bits + 2)
                if (val[s][row] >> t) & 1]

    # X-only writes keyed on the (unchanged) address and match bits:
    # involutions, so no adjoint mint (o_*_adj is o_* itself).
    val_walk = unary_iteration_kernels(n,
                                       size,
                                       val_body,
                                       include_adjoint=False,
                                       num_work=d).kernel
    loc_walk = None
    if num_loc > 0:

        def loc_body(row: int) -> list:
            return [("x_w", s, t) for s in range(num_loc) for t in range(n)
                    if (delta[s][row] >> t) & 1]

        loc_walk = unary_iteration_kernels(n,
                                           size,
                                           loc_body,
                                           include_adjoint=False,
                                           num_work=num_loc).kernel

    mw = 2 * n  # match region offset in the work register
    num_work = 2 * n + d

    if loc_walk is not None:

        @cudaq.kernel
        def qrom_o_loc(slot: cudaq.qview, system: cudaq.qview,
                       work: cudaq.qview):
            """|s>|x> -> |s>|c(s, x)> through the out scratch (module doc).

            Its own inverse: c(s, .) is an involution.
            """
            for s in range(num_loc):
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                x.ctrl(slot, work[mw + s])
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
            loc_walk(system, work[n:2 * n], work[0:n], work[mw:mw + num_loc])
            for k in range(n):
                cx(work[k], system[k])
            loc_walk(system, work[n:2 * n], work[0:n], work[mw:mw + num_loc])
            for s in range(num_loc):
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                x.ctrl(slot, work[mw + s])
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
    else:

        @cudaq.kernel
        def qrom_o_loc(slot: cudaq.qview, system: cudaq.qview,
                       work: cudaq.qview):
            """Diagonal-only matrix: every slot map is the identity."""
            pass

    @cudaq.kernel
    def qrom_o_val(slot: cudaq.qview, system: cudaq.qview,
                   value_and_sign: cudaq.qview, work: cudaq.qview):
        """XOR-load of the (angle, sign, upper) word for (slot, row).

        Its own inverse: X-only writes keyed on unchanged registers.
        """
        for s in range(d):
            for b in range(m):
                if ((s >> b) & 1) == 0:
                    x(slot[b])
            x.ctrl(slot, work[mw + s])
            for b in range(m):
                if ((s >> b) & 1) == 0:
                    x(slot[b])
        val_walk(system, work[n:2 * n], value_and_sign, work[mw:mw + d])
        for s in range(d):
            for b in range(m):
                if ((s >> b) & 1) == 0:
                    x(slot[b])
            x.ctrl(slot, work[mw + s])
            for b in range(m):
                if ((s >> b) & 1) == 0:
                    x(slot[b])

    _retain(qrom_o_loc, qrom_o_val)
    return OracleKernels(o_loc=qrom_o_loc,
                         o_loc_adj=qrom_o_loc,
                         o_val=qrom_o_val,
                         o_val_adj=qrom_o_val,
                         d=d,
                         h=h,
                         value_bits=value_bits,
                         num_work=num_work,
                         slot_flip=list(range(d_padded)))
