# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Encoding-independent kernels and operator helpers shared by the primitives.

Reflections about the all-zero state, projector phases on the all-zero
(signal) subspace, and the single-qubit projector spin operator. These act
only on register geometry — they are valid for any zero-flagged block
encoding, and the kernels are composable from user kernels.

Controlled variants take a combined register whose qubit 0 is the external
control (a CUDA-Q Python control set cannot mix a bare qubit with a
separate register).

Guards use positive ``if n > 0:`` blocks, never early ``return``: kernel
``return`` is silently ignored by the compiler
(https://github.com/NVIDIA/cuda-quantum/issues/4845).
"""

from __future__ import annotations

import cudaq
from cudaq import spin


def state_from(ket) -> cudaq.State:
    """Build a cudaq.State from array data at the current target's precision.

    fp32 simulators (e.g. the default `nvidia` target) reject complex128
    input ("[sim-state] invalid data precision"); cudaq.complex() reports
    the dtype the active target expects.
    """
    import numpy as np

    return cudaq.State.from_data(np.asarray(ket, dtype=cudaq.complex()))


def _validate_control_state(control_state: int) -> int:
    """Require control_state to be exactly 0 or 1 (no silent coercion)."""
    if control_state not in (0, 1):
        raise ValueError("control_state must be 0 or 1")
    return int(control_state)


def _validate_power(power: int) -> int:
    """Require a non-negative integral walk power (no silent truncation)."""
    steps = int(power)
    if steps != power or steps < 0:
        raise ValueError("power must be a non-negative integer")
    return steps


def _bit_projector(qubit: int, bit: int) -> cudaq.SpinOperator:
    """|bit><bit| on one qubit as a spin operator."""
    sign = 1.0 - 2.0 * float(bit)
    return 0.5 * (spin.i(qubit) + sign * spin.z(qubit))


@cudaq.kernel
def signal_phase(register: cudaq.qview, phase: float):
    """exp(i * phase * |0...0><0...0|) on the signal register."""
    n = register.size()
    if n > 0:
        for i in range(n):
            x(register[i])
        if n == 1:
            r1(phase, register[0])
        else:
            r1.ctrl(phase, register.front(n - 1), register[n - 1])
        for i in range(n):
            x(register[i])


@cudaq.kernel
def reflect_about_zero(register: cudaq.qview):
    """I - 2|0...0><0...0| (phases the all-zero state by -1).

    Exactly ``signal_phase(register, pi)``: ``r1(pi)`` is ``Z`` with no
    global phase, so the reflection is the phase = pi special case.
    """
    signal_phase(register, 3.141592653589793)


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


@cudaq.kernel
def controlled_reflect_about_zero(control_and_register: cudaq.qview):
    """Zero-state reflection on qubits 1.. controlled by qubit 0.

    Exactly ``controlled_signal_phase(..., pi)`` (see reflect_about_zero).
    """
    controlled_signal_phase(control_and_register, 3.141592653589793)
