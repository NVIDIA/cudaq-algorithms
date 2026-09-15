# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerant circuit primitives: unary iteration and QROM.

``unary_iteration_kernels`` mints unary iteration (Babbush et al.,
`arXiv:1805.03662`, Fig. 7) in its strictly unitary form — no
measurement-based uncomputation, i.e. the fused select(V) walk of
Childs et al. (`arXiv:1711.10980`, Appendix G.4) — applying a
per-address body exactly when the address register equals ``k``.
``QROM`` builds coherent
classical-table lookups on top of it, either as the plain
unary-iteration SELECT or as the SELECT-SWAP / QROAM construction of
Low, Kliuchnikov and Schaeffer (`arXiv:1812.00954`).

Everything here is strictly unitary: the papers' headline Toffoli counts
assume measurement-based ancilla uncomputation, which this library does
not use (primitives must stay statevector-testable and
inverse-composable); the module docstrings state the coherent costs the
minted kernels actually have, and the resource tests pin them against
the compiler.

Composition (CUDA-Q >= 0.16): minted kernels are building blocks — call
them from your own kernels, and wrap the whole composition in
``cudaq.control`` to obtain the controlled operation. This is the
intended consumption model for SELECT-style constructions (the
select_swap QROM is itself built this way), and it is pinned by tests
(``test_primitives_composition.py``). Through CUDA-Q 0.15,
control-variant generation rejected kernels that call kernels, so on
0.15 use the factory's ``controlled=True`` variants instead — they also
remain the cheaper option everywhere (the control is folded into the
walk rather than added to every gate).

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
