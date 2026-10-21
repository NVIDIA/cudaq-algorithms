# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``encode_sparse``: classical pricing and dispatch for sparse matrices.

``encode_sparse(matrix)`` prices the two generic sparse constructions
*classically* — no kernels are minted for the path not taken — and
returns the cheaper encoding:

- **oracle** (``SparseOracleEncoding`` over ``qrom_oracles``):
  ``alpha = d_padded * h`` with ``d`` the slots of the matching-based
  slot assignment (``_from_data``), ``d_padded = 2^ceil(log2 d)`` and
  ``h = max |H_ij|`` — pays the worst element at the padded sparsity, so
  it wins on uniform structured matrices with tiny ``d``.
- **lcu** (``SparseLCUEncoding``): ``alpha = sum_{i<j} |H_ij| +
  sum_i |H_ii|``, the term one-norm — wins whenever the weights are
  skewed (one dominant element, many small ones), and is the only
  generic path for negative diagonal entries (which the symmetric
  two-sided oracle construction provably cannot represent).

Dispatch picks the smaller ``alpha`` among the eligible paths; ties
break to fewer total qubits, then to the LCU path. ``prefer="oracle"`` /
``prefer="lcu"`` overrides the pricing (and raises with the recorded
reason if the preferred path is ineligible).

Non-symmetric input raises by default, naming the first asymmetry: a
symmetric matrix stored as one triangle, or one with rounding noise
above ``1e-12 * max|H|``, must be completed or symmetrized by the caller
rather than silently encoded as something else. With ``dilate=True`` the
input is Hermitian-dilated **once, at the data level** (``[[0, A],
[A^T, 0]]``, exactly as ``SparseLCUEncoding.from_general``), and *both*
paths are priced on the same dilated symmetric matrix — like against
like. Every entry of ``A`` lands off-diagonal in the dilation, so
negative diagonals of ``A`` never disqualify the oracle path there.

``max_terms`` bounds the generic constructions' classical-table sizes —
the LCU term count (each matrix entry costs two alias bins) and the
from-data oracles' per-row QROM entries (``d * dim``). Above it, the
path is marked ineligible; if no path fits, ``encode_sparse`` raises:
a huge dense matrix has no good *generic* route, and the fix is
structure — hand-written oracles for ``SparseOracleEncoding`` (see
``banded_oracles`` for the shape).

The returned encoding carries the pricing as ``encoding.report`` (keys:
``path``, ``alpha``, ``alpha_alternatives``, ``qubits``, ``nnz``,
``max_row_nnz`` (the largest number of nonzeros in a row),
``slots`` (the structure slots the oracle path's matching assignment
uses, which sets its ``alpha``), ``dim``, ``dilated``, ``ineligible``,
``mu``, ``value_bits``) and surfaces a one-line summary in ``repr()``;
``report`` is ``None`` on encodings built directly.
"""

from __future__ import annotations

from ._from_data import _assign_slots, qrom_oracles
from ._sparse_lcu import (SparseLCUEncoding, _bodies_and_work, _build_terms,
                          _coerce_entries, _first_asymmetry)
from ._sparse_oracle import SparseOracleEncoding

__all__ = ["encode_sparse"]


def _dilate_entries(entries: dict, size: int) -> tuple[int, dict]:
    """The Hermitian dilation ``[[0, A], [A^T, 0]]`` at the data level.

    Identical to ``SparseLCUEncoding.from_general``'s construction: ``A``
    is power-of-two padded to ``half``, each entry becomes the symmetric
    pair ``(i, half + j)`` / ``(half + j, i)``.
    """
    half = 1 << max(1, (size - 1).bit_length())
    dilated: dict = {}
    for (i, j), value in entries.items():
        dilated[(i, half + j)] = value
        dilated[(half + j, i)] = value
    return 2 * half, dilated


def encode_sparse(matrix,
                  *,
                  prefer: str | None = None,
                  max_terms: int = 4096,
                  mu: int = 8,
                  value_bits: int = 8,
                  dim: int | None = None,
                  dilate: bool = False):
    """Encode a sparse real matrix on the cheaper generic construction.

    Parameters
    ----------
    matrix
        A dense 2-D array, a ``scipy.sparse`` matrix, or a
        ``(rows, cols, vals)`` tuple of parallel integer-indexed sequences
        (the input forms of ``SparseLCUEncoding``). Must be symmetric
        unless ``dilate=True``.
    dilate
        ``True`` encodes the Hermitian dilation ``[[0, A], [A^T, 0]]`` of
        a general square ``A`` (one extra system qubit; see the module
        docstring). ``False`` (default) raises on non-symmetric input,
        including a symmetric matrix supplied as one triangle.
    prefer
        ``"oracle"`` / ``"lcu"`` forces that path (raising if it is
        ineligible); ``None`` (default) dispatches on ``alpha``.
    max_terms
        Classical-table budget for the generic paths (module docstring);
        above it a path is ineligible, and with no path left this raises
        pointing at hand-written oracles.
    mu
        Alias-table keep precision for the LCU path (its
        ``discretization_bound``).
    value_bits
        Fixed-point angle bits for the oracle path (its quantization
        bound ``h * (pi/2) * 2^-value_bits``).
    dim
        Matrix dimension — required with triples, optional (checked)
        otherwise.

    Returns
    -------
    ``SparseOracleEncoding`` or ``SparseLCUEncoding``, with the pricing
    report attached as ``encoding.report``.
    """
    if prefer not in (None, "oracle", "lcu"):
        raise ValueError(f"prefer must be None, 'oracle', or 'lcu', "
                         f"got {prefer!r}")
    if int(max_terms) != max_terms or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    max_terms = int(max_terms)
    if int(mu) != mu or mu < 1:
        raise ValueError("mu must be a positive integer (mu >= 1)")
    mu = int(mu)
    if int(value_bits) != value_bits or value_bits < 1:
        raise ValueError("value_bits must be a positive integer")
    value_bits = int(value_bits)

    size, entries = _coerce_entries(matrix, dim)
    if not entries:
        raise ValueError("matrix has no nonzero entries: nothing to encode")
    asymmetry = _first_asymmetry(entries)
    if asymmetry is not None and not dilate:
        i, j, value, partner = asymmetry
        raise ValueError(
            f"matrix is not symmetric at ({i}, {j}): {value} vs {partner}. "
            "Both constructions encode a symmetric matrix: complete or "
            "symmetrize the input (a matrix stored as one triangle must be "
            "mirrored first), or pass dilate=True to encode the Hermitian "
            "dilation [[0, A], [A^T, 0]] on one extra system qubit")
    dilated = bool(dilate)
    if dilated:
        size, entries = _dilate_entries(entries, size)
    num_system = max(1, (size - 1).bit_length())
    nnz = len(entries)
    row_counts: dict = {}
    for (i, _) in entries:
        row_counts[i] = row_counts.get(i, 0) + 1
    d_row = max(row_counts.values())

    # ------------------------------------------------------------------
    # Classical pricing (no kernels; formulas match the constructions)
    # ------------------------------------------------------------------

    # LCU path: term one-norm + the exact register budget. The term list
    # is cheap (tuples); check the budget before materializing any SELECT
    # gate bodies, which are only needed for the eligible register width.
    terms = _build_terms(entries)
    alpha_lcu = sum(term[3] for term in terms)
    num_index = max(1, (len(terms) - 1).bit_length())
    lcu_reason = None
    if len(terms) > max_terms:
        lcu_reason = (f"the LCU path needs {len(terms)} terms (two alias "
                      f"bins per matrix entry), above max_terms={max_terms}")
        qubits_lcu = None
    else:
        _, select_work = _bodies_and_work(terms, num_system)
        # (2 * num_index + 2 * mu + 2) is AliasSamplingPrepare's num_garbage
        # for the select-variant QROM (num_ladder == num_index; the family
        # comparator uses one carry qubit), kept in lockstep with the
        # variant="select" pinned in SparseLCUEncoding.
        qubits_lcu = (num_system + num_index + (2 * num_index + 2 * mu + 2) +
                      select_work)

    # Oracle path: padded-sparsity alpha over the same slot assignment
    # qrom_oracles would build, and its exact register budget
    # (num_work = 2 * num_system + d).
    matchings, has_diagonal = _assign_slots(entries, size)
    d_slots = len(matchings) + (1 if has_diagonal else 0)
    m1 = max(1, (d_slots - 1).bit_length())
    h = max(abs(v) for v in entries.values())
    alpha_oracle = float((1 << m1) * h)
    qubits_oracle = (num_system + (m1 + 2 + num_system) +
                     (value_bits + 2 + (2 * num_system + d_slots)))
    oracle_reason = None
    if any(i == j and v < 0.0 for (i, j), v in entries.items()):
        oracle_reason = (
            "the matrix has negative diagonal entries, which the symmetric "
            "two-sided oracle construction cannot represent (their phase "
            "contribution is always +1); shift the diagonal or use the LCU "
            "path")
    elif d_slots * size > max_terms:
        oracle_reason = (f"the from-data oracles need {d_slots * size} QROM "
                         f"entries ({d_slots} slots x dimension {size}), "
                         f"above max_terms={max_terms}")

    pricing = {
        "oracle": (alpha_oracle, qubits_oracle, oracle_reason),
        "lcu": (alpha_lcu, qubits_lcu, lcu_reason),
    }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    if prefer is not None:
        reason = pricing[prefer][2]
        if reason is not None:
            raise ValueError(f"prefer={prefer!r} is not available: {reason}")
        path = prefer
    else:
        eligible = [p for p in ("lcu", "oracle") if pricing[p][2] is None]
        if not eligible:
            raise ValueError(
                "no generic sparse encoding fits within max_terms="
                f"{max_terms}: oracle path — {oracle_reason}; lcu path — "
                f"{lcu_reason}. Raise max_terms (the tables and circuits "
                "grow with it), or exploit the matrix structure with "
                "hand-written oracles for SparseOracleEncoding (see "
                "banded_oracles for the shape)")
        path = min(eligible, key=lambda p: (pricing[p][0], pricing[p][1]))

    report = {
        "path": path,
        "alpha": pricing[path][0],
        "alpha_alternatives": {
            p: pricing[p][0]
            for p in ("oracle", "lcu")
        },
        "qubits": {
            p: pricing[p][1]
            for p in ("oracle", "lcu")
        },
        "nnz": nnz,
        "max_row_nnz": d_row,
        "slots": d_slots,
        "dim": size,
        "dilated": dilated,
        "ineligible": {
            p: pricing[p][2]
            for p in ("oracle", "lcu")
        },
        "mu": mu,
        "value_bits": value_bits,
    }

    # ------------------------------------------------------------------
    # Build the chosen path only
    # ------------------------------------------------------------------

    keys = sorted(entries)
    triples = ([int(i) for i, _ in keys], [int(j) for _, j in keys],
               [float(entries[key]) for key in keys])
    if path == "oracle":
        oracles = qrom_oracles(triples, value_bits, dim=size)
        encoding = SparseOracleEncoding(oracles, num_system=num_system)
    else:
        encoding = SparseLCUEncoding(triples, mu=mu, dim=size)
    encoding.report = report
    return encoding
