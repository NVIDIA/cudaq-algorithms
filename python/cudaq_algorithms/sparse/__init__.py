# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-access block encodings.

The sparse-access block encoding of real symmetric matrices
(:class:`SparseOracleEncoding`), three ways to feed it (:func:`banded_oracles`,
:func:`qrom_oracles`, :class:`OracleKernels`), and the data-level constructor
(:func:`encode_sparse`). These names are also re-exported from the package
root.
"""

from ._banded import banded_oracles, quantized_angle
from ._factory import encode_sparse
from ._from_data import qrom_oracles
from ._sparse_oracle import OracleKernels, SparseOracleEncoding

__all__ = [
    "OracleKernels",
    "SparseOracleEncoding",
    "banded_oracles",
    "encode_sparse",
    "qrom_oracles",
    "quantized_angle",
]
