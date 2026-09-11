# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerant circuit primitives: unary iteration, QROM, arithmetic.

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

The reversible integer arithmetic device kernels (:mod:`._arithmetic`)
are the in-place little-endian adders and comparators the lookup-based
constructions compose with: the CDKM/Cuccaro ripple-carry family
(``add_register`` / ``subtract_register``, ``add_constant`` /
``subtract_constant``, ``cmp_ge_constant``, and the register-register
comparators ``cmp_ge_register`` / ``cmp_gt_register``) and the ancilla-free Draper
QFT family (``qft`` / ``iqft``, ``add_constant_qft`` /
``subtract_constant_qft``, the ``cmp_ge_constant_qft`` /
``cmp_ge_constant_qft_adj`` pair). Every inverse is hand-written and the
gate prices are compiler-pinned by the resource tests.

Import the subpackage directly (``from cudaq_algorithms.primitives
import QROM``); nothing here is re-exported from the package root.
"""

from ._arithmetic import (add_constant, add_constant_qft, add_register,
                          cmp_ge_constant, cmp_ge_constant_qft,
                          cmp_ge_constant_qft_adj, cmp_ge_register,
                          cmp_gt_register, iqft, phase_add_constant, qft,
                          subtract_constant, subtract_constant_qft,
                          subtract_register)
from ._qrom import QROM
from ._unary_iteration import UnaryIterationKernels, unary_iteration_kernels

__all__ = [
    "QROM",
    "UnaryIterationKernels",
    "add_constant",
    "add_constant_qft",
    "add_register",
    "cmp_ge_constant",
    "cmp_ge_constant_qft",
    "cmp_ge_constant_qft_adj",
    "cmp_ge_register",
    "cmp_gt_register",
    "iqft",
    "phase_add_constant",
    "qft",
    "subtract_constant",
    "subtract_constant_qft",
    "subtract_register",
    "unary_iteration_kernels",
]
