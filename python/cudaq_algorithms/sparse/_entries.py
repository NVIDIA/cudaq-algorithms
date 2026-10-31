# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matrix-data input handling shared by ``qrom_oracles`` and ``encode_sparse``.

Accepted forms: a dense 2-D array, a ``scipy.sparse`` matrix (anything
with ``tocoo()``), or ``(rows, cols, vals)`` triples with integer-typed
``rows``/``cols`` and an explicit ``dim``. Zero entries are dropped,
duplicate triples sum (COO semantics), entries must be real and finite,
and symmetry is judged with a matrix-relative tolerance.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["_coerce_entries", "_first_asymmetry", "_real_entry"]


def _real_entry(value, i: int, j: int) -> float:
    entry = complex(value)
    if not (math.isfinite(entry.real) and math.isfinite(entry.imag)):
        raise ValueError(f"matrix entry ({i}, {j}) must be finite, "
                         f"got {value!r}")
    if abs(entry.imag) > 1e-12:
        raise ValueError(
            f"matrix entry ({i}, {j}) is complex ({value!r}): complex "
            "matrices are not supported (deferred; encode real data)")
    return entry.real


def _looks_like_triples(matrix) -> bool:
    """True for a ``(rows, cols, vals)`` tuple: three one-dimensional
    sequences whose first two have an integer dtype. A tuple of three
    matrix rows (floats, or 2-D) is a dense matrix, not triples."""
    if not (isinstance(matrix, tuple) and len(matrix) == 3):
        return False
    try:
        rows, cols, vals = (np.asarray(part) for part in matrix)
    except (TypeError, ValueError):
        return False
    return (rows.ndim == cols.ndim == vals.ndim == 1
            and rows.dtype.kind in "iu" and cols.dtype.kind in "iu")


def _coerce_entries(matrix, dim: int | None) -> tuple[int, dict]:
    """Normalize any accepted input form to ``(dim, {(i, j): value})``.

    Zero entries are dropped; duplicate triples sum (COO semantics).
    Triples are recognized by shape and dtype (``_looks_like_triples``);
    anything else goes through ``np.asarray`` as a dense matrix.
    """
    if hasattr(matrix, "tocoo"):  # scipy.sparse, without a hard dependency
        coo = matrix.tocoo()
        if coo.shape[0] != coo.shape[1]:
            raise ValueError(f"matrix must be square, got shape {coo.shape}")
        size = int(coo.shape[0])
        rows, cols, vals = coo.row, coo.col, coo.data
    elif _looks_like_triples(matrix):
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


def _first_asymmetry(entries: dict) -> tuple | None:
    """The first ``(i, j, value, partner)`` violating symmetry, or None.

    Shared by ``qrom_oracles`` (which raises on it) and ``encode_sparse``
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
