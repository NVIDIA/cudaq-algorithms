# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-access block encodings.

Two generic block encodings of sparse real symmetric matrices
(:class:`SparseOracleEncoding`, :class:`SparseLCUEncoding`), three ways to feed
them (:func:`banded_oracles`, :func:`qrom_oracles`, :class:`OracleKernels`), and
a classical pricing dispatcher (:func:`encode_sparse`). These names are also
re-exported from the package root.
"""

from ._banded import banded_oracles
from ._factory import encode_sparse
from ._from_data import qrom_oracles
from ._sparse_lcu import SparseLCUEncoding
from ._sparse_oracle import OracleKernels, SparseOracleEncoding

__all__ = [
    "OracleKernels",
    "SparseLCUEncoding",
    "SparseOracleEncoding",
    "banded_oracles",
    "encode_sparse",
    "qrom_oracles",
]
