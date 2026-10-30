# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fermionic basis changes on an existing register, built from H, S, CNOT,
Rz and R1 (bit-controlled Rz and R1 for programmable plans).

The paired Jordan-Wigner Pauli rotations implement each complete Givens
mixing, including parity on intervening physical modes. Parameter words
remain caller-owned; this backend neither allocates qubits nor performs lookup.

Controlled variants take the enable at register index zero and shift every
other index by one. Each Pauli rotation is ``V Rz V†`` with a Clifford
scaffold ``V``, so the enable controls only the Rz and R1 rotations; with
the enable off the scaffold cancels exactly.
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


def _capture(rows, used_second, offset):
    """Split ``(code, first, second, angle)`` rows into captured lists.

    CUDA-Q 0.16 fuses value-identical captured lists, so a tape without
    CNOT rows would turn the unused ``x.ctrl(register[first[k]],
    register[second[k]])`` branch into a CNOT with control equal to target,
    and lowering under ``cudaq.control`` never terminates
    (cuda-quantum#5794). An unused second index is therefore ``first + 1``,
    so ``second[k] != first[k]`` on every row and the two lists never
    coincide.
    """
    codes = [row[0] for row in rows]
    first = [row[1] + offset for row in rows]
    second = [(row[2] if row[0] in used_second else row[1] + 1) + offset
              for row in rows]
    angles = [row[3] for row in rows]
    return codes, first, second, angles


def classical_kernel(schedule, adjoint=False, controlled=False):
    """Capture a numerical schedule in a kernel accepting an existing qview.

    With ``controlled=True`` the enable is register index 0, every other
    index shifts by one, and only the rotations take the enable.
    """
    rows = [
        (code, first, second, sign * schedule.parameters[slot])
        for code, first, second, slot, sign in _gate_tape(schedule, adjoint)
    ]
    codes, first, second, angles = _capture(rows, (3, ), int(controlled))

    if controlled:

        @cudaq.kernel
        def apply_controlled_basis(register: cudaq.qview):
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
                    rz.ctrl(angles[k], register[0], register[first[k]])
                else:
                    r1.ctrl(angles[k], register[0], register[first[k]])

        return apply_controlled_basis

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


def programmable_kernel(schedule,
                        parameter_registers,
                        weights,
                        adjoint=False,
                        controlled=False):
    """Apply coherent parameter words through ordinary controlled rotations.

    ``parameter_registers`` contains one validated word per slot, in
    little-endian bit order; ``weights`` gives radians per word bit. Each
    word bit controls a rotation of ``sign * weights[bit]`` on the target
    qubit. The host expands those rotations into rows of (opcode, control,
    target, angle), so the kernel is one flat loop that CUDA-Q can reverse,
    with no mutable weight accumulator or machine-width integer shifts.
    Every slot is retained even when its numerical schedule value is zero.
    With ``controlled=True`` the enable is register index 0, every other
    index shifts by one, and only the rotations take the enable.
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
    codes, first, second, angles = _capture(rows, (3, 4, 5), int(controlled))

    if controlled:

        @cudaq.kernel
        def apply_controlled_word_basis(register: cudaq.qview):
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
                    rz.ctrl(angles[k], register[0], register[first[k]],
                            register[second[k]])
                else:
                    r1.ctrl(angles[k], register[0], register[first[k]],
                            register[second[k]])

        return apply_controlled_word_basis

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
