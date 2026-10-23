# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register-only fermionic basis changes using existing CUDA-Q gates.

The paired Jordan-Wigner Pauli rotations implement each complete Givens
mixing, including parity on intervening physical modes. Parameter words
remain caller-owned; this backend neither allocates qubits nor performs lookup.
"""

import cudaq


# Gate tape: (opcode, first, second, parameter index, parameter sign).
# H=0, S=1, S-adjoint=2, CX=3, RZ=4, exact number phase R1=5.
def _gate_tape(schedule, adjoint=False):
    tape = []
    for operation in schedule.operations:
        first = schedule.structure.mode_map[operation.first]
        slot, sign = operation.parameter_index, operation.scale
        if operation.kind == "phase":
            tape.append((5, first, first, slot, sign))
            continue
        second = schedule.structure.mode_map[operation.second]
        if first > second:
            first, second, sign = second, first, -sign
        # G_pq(theta) = R_(Y Z... X)(-theta) R_(X Z... Y)(theta).
        # Each parity ladder includes all intervening physical modes.
        for y_endpoint, angle_sign in ((first, -sign), (second, sign)):
            tape.append((2, y_endpoint, y_endpoint, slot, 0.))
            tape.extend((0, p, p, slot, 0.) for p in (first, second))
            tape.extend((3, p, p + 1, slot, 0.) for p in range(first, second))
            tape.append((4, second, second, slot, angle_sign))
            tape.extend((3, p, p + 1, slot, 0.)
                        for p in range(second - 1, first - 1, -1))
            tape.extend((0, p, p, slot, 0.) for p in (first, second))
            tape.append((1, y_endpoint, y_endpoint, slot, 0.))
    if adjoint:
        tape = [(2 if code == 1 else 1 if code == 2 else code, first, second,
                 slot, -sign)
                for code, first, second, slot, sign in reversed(tape)]
    return tuple(tape)


def classical_kernel(schedule, adjoint=False):
    """Capture a numerical schedule in a kernel accepting an existing qview."""
    tape = _gate_tape(schedule, adjoint)
    codes = [row[0] for row in tape]
    first = [row[1] for row in tape]
    second = [row[2] for row in tape]
    angles = [row[4] * schedule.parameters[row[3]] for row in tape]

    @cudaq.kernel
    def apply_basis(register: cudaq.qview):
        for k in range(len(codes)):
            if codes[k] == 0:
                h(register[first[k]])
            elif codes[k] == 1:
                s(register[first[k]])
            elif codes[k] == 2:
                s.adj(register[first[k]])
            elif codes[k] == 3:
                x.ctrl(register[first[k]], register[second[k]])
            elif codes[k] == 4:
                rz(angles[k], register[first[k]])
            else:
                r1(angles[k], register[first[k]])

    return apply_basis


def programmable_kernel(schedule, parameter_registers, weights, adjoint=False):
    """Apply coherent parameter words through ordinary controlled rotations.

    ``parameter_registers`` contains one validated contiguous word per slot;
    ``weights`` gives radians per little-endian word bit. Each word bit
    controls a rotation of ``sign * weights[bit]`` on the target qubit. The
    host expands those rotations into rows of (opcode, control, target,
    angle), so the kernel is one flat loop that CUDA-Q can reverse, with no
    mutable weight accumulator or machine-width integer shifts. Every slot
    is retained even when its numerical schedule value is zero.
    """
    rows = []
    for code, first, second, slot, sign in _gate_tape(schedule, adjoint):
        if code < 4:
            # Clifford rows ignore their zero angle, as in classical_kernel.
            rows.append((code, first, second, 0.))
            continue
        # A rotation row names its target in both fields; each word bit
        # controls one weighted rotation of that target.
        word = parameter_registers[slot]
        rows.extend((code, word[bit], second, float(sign * weight))
                    for bit, weight in enumerate(weights))
    codes = [row[0] for row in rows]
    first = [row[1] for row in rows]
    second = [row[2] for row in rows]
    angles = [row[3] for row in rows]

    @cudaq.kernel
    def apply_word_basis(register: cudaq.qview):
        for k in range(len(codes)):
            if codes[k] == 0:
                h(register[first[k]])
            elif codes[k] == 1:
                s(register[first[k]])
            elif codes[k] == 2:
                s.adj(register[first[k]])
            elif codes[k] == 3:
                x.ctrl(register[first[k]], register[second[k]])
            elif codes[k] == 4:
                rz.ctrl(angles[k], register[first[k]], register[second[k]])
            else:
                r1.ctrl(angles[k], register[first[k]], register[second[k]])

    return apply_word_basis
