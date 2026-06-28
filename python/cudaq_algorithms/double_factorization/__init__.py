# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Double factorization of two-electron integrals (X-DF and C-DF).

Explicit (X-DF) and compressed (C-DF) double factorization following Cohn,
Motta, and Parrish, PRX Quantum 2, 040352 (2021) (arXiv:2104.08957). Heavy
linear algebra runs on the NVIDIA math libraries (cuSOLVER / cuBLAS via CuPy)
when a GPU is available, with a NumPy/SciPy fallback selected automatically.
"""
from ._backend import cupy_gpu_available, resolve_backend
from ._factorization import (DoubleFactorization,
                             compressed_double_factorization,
                             double_factorization_one_norm,
                             explicit_double_factorization,
                             factorization_error,
                             modified_one_body_integrals, reconstruct_eri)

__all__ = [
    "DoubleFactorization",
    "explicit_double_factorization",
    "compressed_double_factorization",
    "reconstruct_eri",
    "factorization_error",
    "modified_one_body_integrals",
    "double_factorization_one_norm",
    "cupy_gpu_available",
    "resolve_backend",
]
