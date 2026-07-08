# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Encoding-independent device kernels shared by the primitives.

Reflections about the all-zero state and projector phases on the all-zero
(signal) subspace. These act only on register geometry — they are valid
for any zero-flagged block encoding, and are composable from user kernels.

Controlled variants take a combined register whose qubit 0 is the external
control (a CUDA-Q Python control set cannot mix a bare qubit with a
separate register).
"""

from __future__ import annotations

import cudaq

@cudaq.kernel
def reflect_about_zero(register: cudaq.qview):
    """I - 2|0...0><0...0| (phases the all-zero state by -1)."""
    n = register.size()
    if n == 0:
        return
    for i in range(n):
        x(register[i])
    if n == 1:
        z(register[0])
    else:
        z.ctrl(register.front(n - 1), register[n - 1])
    for i in range(n):
        x(register[i])

@cudaq.kernel
def controlled_reflect_about_zero(control_and_register: cudaq.qview):
    """Zero-state reflection on qubits 1.. controlled by qubit 0.

    Qubit 0 of ``control_and_register`` is the external control (see
    controlled_select for why the control shares a register).
    """
    total = control_and_register.size()
    n = total - 1
    for i in range(n):
        x(control_and_register[1 + i])
    if n == 0:
        z(control_and_register[0])
    else:
        z.ctrl(control_and_register.front(total - 1),
               control_and_register[total - 1])
    for i in range(n):
        x(control_and_register[1 + i])

@cudaq.kernel
def signal_phase(register: cudaq.qview, phase: float):
    """exp(i * phase * |0...0><0...0|) on the signal register."""
    n = register.size()
    if n == 0:
        return
    for i in range(n):
        x(register[i])
    if n == 1:
        r1(phase, register[0])
    else:
        r1.ctrl(phase, register.front(n - 1), register[n - 1])
    for i in range(n):
        x(register[i])

@cudaq.kernel
def controlled_signal_phase(control_and_register: cudaq.qview, phase: float):
    """Signal phase on qubits 1.. controlled by qubit 0."""
    total = control_and_register.size()
    n = total - 1
    for i in range(n):
        x(control_and_register[1 + i])
    if n == 0:
        r1(phase, control_and_register[0])
    else:
        r1.ctrl(phase, control_and_register.front(total - 1),
                control_and_register[total - 1])
    for i in range(n):
        x(control_and_register[1 + i])
