# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerant circuit primitives: unary iteration and QROM.

``unary_iteration_kernels`` mints the fused tree walk of Babbush et al.
(`arXiv:1805.03662`, Fig. 7) — apply a per-address body exactly when the
address register equals ``k`` — and ``QROM`` builds coherent
classical-table lookups on top of it, either as the plain
unary-iteration SELECT or as the SELECT-SWAP / QROAM construction of
Low, Kliuchnikov and Schaeffer (`arXiv:1812.00954`).

Everything here is strictly unitary: the papers' headline Toffoli counts
assume measurement-based ancilla uncomputation, which this library does
not use (primitives must stay statevector-testable and
inverse-composable); the module docstrings state the coherent costs the
minted kernels actually have, and the resource tests pin them against
the compiler.

Import the subpackage directly (``from cudaq_algorithms.primitives
import QROM``); nothing here is re-exported from the package root.
"""

from ._qrom import QROM
from ._unary_iteration import UnaryIterationKernels, unary_iteration_kernels

__all__ = [
    "QROM",
    "UnaryIterationKernels",
    "unary_iteration_kernels",
]
