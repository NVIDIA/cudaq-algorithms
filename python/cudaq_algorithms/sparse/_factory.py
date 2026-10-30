# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``encode_sparse``: the sparse-access oracle encoding built from matrix data.

``encode_sparse(matrix)`` validates the data, optionally takes the
Hermitian dilation, prices the construction classically and returns a
``SparseOracleEncoding`` over ``qrom_oracles`` with the figures attached as
``encoding.report``. The normalization is ``alpha = d_padded * h``: ``d``
the slots of the matching-based slot assignment (``_from_data``),
``d_padded = 2^ceil(log2 d)`` and ``h = max |H_ij|``, so the encoding pays
the largest element at the padded sparsity.

Non-symmetric input raises by default, naming the first asymmetry: a
symmetric matrix stored as one triangle, or one with rounding noise
above ``1e-12 * max|H|``, must be completed or symmetrized by the caller
rather than silently encoded as something else. With ``dilate=True`` the
input is Hermitian-dilated once, at the data level (``[[0, A], [A^T,
0]]``, one extra system qubit as the most significant bit). Every entry
of ``A`` lands off-diagonal in the dilation, which is also the way to
encode a matrix with negative diagonal entries, since the symmetric
two-sided oracle construction cannot represent those directly.

``max_entries`` bounds the classical table the from-data oracles compile
(``slots * dim`` QROM entries). Above it ``encode_sparse`` raises: a huge
dense matrix has no good generic route, and the fix is structure —
hand-written oracles for ``SparseOracleEncoding`` (see ``banded_oracles``
for the shape).

The report carries ``alpha``, ``qubits`` (system + ancilla), ``nnz``,
``max_row_nnz`` (the largest number of nonzeros in a row), ``slots`` (the
structure slots the matching assignment uses, which set ``alpha``;
possibly more than ``max_row_nnz``), ``dim``, ``dilated`` and
``value_bits``, and surfaces a one-line summary in ``repr()``; ``report``
is ``None`` on encodings built directly.
"""

from __future__ import annotations

import math

from ._entries import _coerce_entries, _first_asymmetry
from ._from_data import _assign_slots, qrom_oracles
from ._sparse_oracle import SparseOracleEncoding

__all__ = ["encode_sparse"]


def _dilate_entries(entries: dict, size: int) -> tuple[int, dict]:
    """The Hermitian dilation ``[[0, A], [A^T, 0]]`` at the data level:
    ``A`` is power-of-two padded to ``half`` and each entry becomes the
    symmetric pair ``(i, half + j)`` / ``(half + j, i)``."""
    half = 1 << max(1, (size - 1).bit_length())
    dilated: dict = {}
    for (i, j), value in entries.items():
        dilated[(i, half + j)] = value
        dilated[(half + j, i)] = value
    return 2 * half, dilated


def encode_sparse(matrix,
                  *,
                  value_bits: int = 8,
                  dim: int | None = None,
                  dilate: bool = False,
                  max_entries: int = 4096) -> SparseOracleEncoding:
    """Encode a sparse real matrix on the sparse-access oracle construction.

    Parameters
    ----------
    matrix
        A dense 2-D array, a ``scipy.sparse`` matrix, or a
        ``(rows, cols, vals)`` tuple of parallel integer-indexed sequences.
        Must be symmetric with a non-negative diagonal unless
        ``dilate=True``.
    value_bits
        Fixed-point angle bits of the value oracle (quantization bound
        ``h * (pi/2) * 2^-value_bits`` per element).
    dim
        Matrix dimension — required with triples, optional (checked)
        otherwise.
    dilate
        ``True`` encodes the Hermitian dilation ``[[0, A], [A^T, 0]]`` of
        a general square ``A`` (one extra system qubit; module docstring).
        ``False`` (default) raises on non-symmetric input, including a
        symmetric matrix supplied as one triangle, and on negative
        diagonal entries.
    max_entries
        Budget for the from-data oracles' classical table (``slots * dim``
        QROM entries); above it this raises pointing at hand-written
        oracles.

    Returns
    -------
    ``SparseOracleEncoding`` with the pricing report attached as
    ``encoding.report``.
    """
    if int(max_entries) != max_entries or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    max_entries = int(max_entries)
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
            "The construction encodes a symmetric matrix: complete or "
            "symmetrize the input (a matrix stored as one triangle must be "
            "mirrored first), or pass dilate=True to encode the Hermitian "
            "dilation [[0, A], [A^T, 0]] on one extra system qubit")
    dilated = bool(dilate)
    if dilated:
        size, entries = _dilate_entries(entries, size)
    negative = [i for (i, j), v in entries.items() if i == j and v < 0.0]
    if negative:
        raise ValueError(
            f"diagonal entry ({negative[0]}, {negative[0]}) is negative: the "
            "symmetric two-sided oracle construction cannot represent "
            "negative diagonal elements (their phase contribution is always "
            "+1); shift the diagonal, or pass dilate=True to encode the "
            "Hermitian dilation [[0, A], [A^T, 0]], where every entry sits "
            "off the diagonal")
    num_system = max(1, (size - 1).bit_length())
    nnz = len(entries)
    row_counts: dict = {}
    for (i, _) in entries:
        row_counts[i] = row_counts.get(i, 0) + 1
    d_row = max(row_counts.values())

    # Classical pricing over the same slot assignment qrom_oracles builds:
    # alpha = d_padded * h and the exact register budget
    # (num_work = 2 * num_system + d).
    matchings, has_diagonal = _assign_slots(entries, size)
    d_slots = len(matchings) + (1 if has_diagonal else 0)
    if d_slots * size > max_entries:
        raise ValueError(
            f"the from-data oracles need {d_slots * size} QROM entries "
            f"({d_slots} slots x dimension {size}), above max_entries="
            f"{max_entries}. Raise max_entries (the tables and circuits grow "
            "with it), or exploit the matrix structure with hand-written "
            "oracles for SparseOracleEncoding (see banded_oracles for the "
            "shape)")
    m1 = max(1, (d_slots - 1).bit_length())
    h = max(abs(v) for v in entries.values())
    alpha = float((1 << m1) * h)
    if not math.isfinite(alpha):
        raise ValueError(
            f"alpha = d_padded * h = {1 << m1} * {h} overflows to infinity; "
            "rescale the matrix (the encoding is of H / alpha, so a common "
            "factor costs nothing)")
    qubits = (num_system + (m1 + 2 + num_system) +
              (value_bits + 2 + (2 * num_system + d_slots)))

    keys = sorted(entries)
    triples = ([int(i) for i, _ in keys], [int(j) for _, j in keys],
               [float(entries[key]) for key in keys])
    oracles = qrom_oracles(triples, value_bits, dim=size)
    encoding = SparseOracleEncoding(oracles, num_system=num_system)
    encoding.report = {
        "alpha": alpha,
        "qubits": qubits,
        "nnz": nnz,
        "max_row_nnz": d_row,
        "slots": d_slots,
        "dim": size,
        "dilated": dilated,
        "value_bits": value_bits,
    }
    return encoding
