# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-opcode semantic pins for the flat interpreter.

One test per (opcode, interpreter variant): mint a one-instruction tape,
run it on every basis state of the qubits the gate touches, and compare
the full statevector — including phases — against the opcode's documented
action. Running each opcode under EVERY variant whose signature supports
it also proves each variant's dispatch chain has an arm for it (a missing
arm would silently no-op the gate and fail the pin).

Also pins the mint-time tape validation: a tape carrying an opcode
outside the minted variant's dispatch set must be rejected loudly, never
silently skipped.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives._unary_iteration import (
    _BASE_OPS, _CONTROL_OPS, _OP_AND_TT, _OP_AND_WT, _OP_BODY_X,
    _OP_BODY_X_W, _OP_BODY_Y, _OP_BODY_Z, _OP_BODY_Z_W, _OP_CCX,
    _OP_CCX_ADDR_ADDR, _OP_CCX_CTRL, _OP_COPY_TW, _OP_CX_ADDR_ADDR,
    _OP_CX_ADDR_LADDER, _OP_CX_CTRL_LADDER, _OP_CX_LADDER_LADDER,
    _OP_CX_LADDER_TARGET, _OP_FREE_CX, _OP_FREE_X, _OP_X_ADDR, _OP_X_LADDER,
    _OP_Z_LADDER, _WORK_OPS, _mint_interpreter)

# Register widths in the harness: control 1, address 2, ladder 2,
# target 2, work 2 — enough for two distinct operand indices per
# register, so operand-order bugs are visible.
_WIDTHS = {"c": 1, "a": 2, "l": 2, "t": 2, "w": 2}

# Each spec: opcode, its (a, b, c) operand values, the gate kind, and
# the touched qubits as (register, index) refs in gate order — controls
# first, target last. Operand values are chosen to match the dispatch
# arms in _mint_interpreter exactly.
_SPECS = {
    "x_addr": (_OP_X_ADDR, (1, 0, 0), "x", [("a", 1)]),
    "x_ladder": (_OP_X_LADDER, (1, 0, 0), "x", [("l", 1)]),
    "cx_addr_ladder": (_OP_CX_ADDR_LADDER, (1, 0, 0), "cx", [("a", 1),
                                                             ("l", 0)]),
    "cx_ladder_ladder": (_OP_CX_LADDER_LADDER, (0, 1, 0), "cx", [("l", 0),
                                                                 ("l", 1)]),
    "ccx": (_OP_CCX, (0, 1, 1), "ccx", [("l", 0), ("a", 1), ("l", 1)]),
    "body_x": (_OP_BODY_X, (0, 1, 0), "cx", [("l", 0), ("t", 1)]),
    "body_y": (_OP_BODY_Y, (0, 1, 0), "cy", [("l", 0), ("t", 1)]),
    "body_z": (_OP_BODY_Z, (0, 1, 0), "cz", [("l", 0), ("t", 1)]),
    "cx_ctrl_ladder": (_OP_CX_CTRL_LADDER, (0, 1, 0), "cx", [("c", 0),
                                                             ("l", 1)]),
    "ccx_ctrl": (_OP_CCX_CTRL, (0, 1, 1), "ccx", [("c", 0), ("a", 1),
                                                  ("l", 1)]),
    "free_x": (_OP_FREE_X, (1, 0, 0), "x", [("t", 1)]),
    "free_cx": (_OP_FREE_CX, (0, 1, 0), "cx", [("t", 0), ("t", 1)]),
    "and_tt": (_OP_AND_TT, (0, 1, 1), "ccx", [("t", 0), ("t", 1), ("w", 1)]),
    "and_wt": (_OP_AND_WT, (0, 1, 1), "ccx", [("w", 0), ("t", 1), ("w", 1)]),
    "copy_tw": (_OP_COPY_TW, (0, 1, 0), "cx", [("t", 0), ("w", 1)]),
    "body_x_w": (_OP_BODY_X_W, (0, 1, 1), "ccx", [("l", 0), ("w", 1),
                                                  ("t", 1)]),
    "body_z_w": (_OP_BODY_Z_W, (0, 1, 0), "cz", [("l", 0), ("w", 1)]),
    "z_ladder": (_OP_Z_LADDER, (1, 0, 0), "z", [("l", 1)]),
    "cx_addr_addr": (_OP_CX_ADDR_ADDR, (0, 1, 0), "cx", [("a", 0), ("a", 1)]),
    "ccx_addr_addr": (_OP_CCX_ADDR_ADDR, (0, 1, 1), "ccx", [("a", 0),
                                                            ("a", 1),
                                                            ("l", 1)]),
    "cx_ladder_target": (_OP_CX_LADDER_TARGET, (0, 1, 0), "cx", [("l", 0),
                                                                 ("t", 1)]),
}

_ALL_VARIANTS = [(False, False), (True, False), (False, True), (True, True)]


def _variants_for(opcode):
    """Every (controlled, has_work) signature that supports the opcode."""
    if opcode in _CONTROL_OPS:
        return [(True, False), (True, True)]
    if opcode in _WORK_OPS:
        return [(False, True), (True, True)]
    return list(_ALL_VARIANTS)


def _spec_cases():
    return [
        pytest.param(opcode,
                     operands,
                     kind,
                     refs,
                     controlled,
                     has_work,
                     id=f"{name}-ctrl{int(controlled)}-work{int(has_work)}")
        for name, (opcode, operands, kind, refs) in sorted(_SPECS.items())
        for controlled, has_work in _variants_for(opcode)
    ]


def _mint_harness(op, controlled, has_work):
    """One-instruction interpreter plus a basis-state-prep entry kernel."""
    interp = _mint_interpreter([op], controlled, has_work)

    if controlled and has_work:

        @cudaq.kernel
        def run_cw(c_init: int, a_init: int, l_init: int, t_init: int,
                   w_init: int):
            control = cudaq.qvector(1)
            address = cudaq.qvector(2)
            ladder = cudaq.qvector(2)
            target = cudaq.qvector(2)
            work = cudaq.qvector(2)
            if c_init == 1:
                x(control[0])
            for i in range(2):
                if ((a_init >> i) & 1) == 1:
                    x(address[i])
                if ((l_init >> i) & 1) == 1:
                    x(ladder[i])
                if ((t_init >> i) & 1) == 1:
                    x(target[i])
                if ((w_init >> i) & 1) == 1:
                    x(work[i])
            interp(control, address, ladder, target, work)

        return run_cw

    if controlled:

        @cudaq.kernel
        def run_c(c_init: int, a_init: int, l_init: int, t_init: int):
            control = cudaq.qvector(1)
            address = cudaq.qvector(2)
            ladder = cudaq.qvector(2)
            target = cudaq.qvector(2)
            if c_init == 1:
                x(control[0])
            for i in range(2):
                if ((a_init >> i) & 1) == 1:
                    x(address[i])
                if ((l_init >> i) & 1) == 1:
                    x(ladder[i])
                if ((t_init >> i) & 1) == 1:
                    x(target[i])
            interp(control, address, ladder, target)

        return run_c

    if has_work:

        @cudaq.kernel
        def run_w(a_init: int, l_init: int, t_init: int, w_init: int):
            address = cudaq.qvector(2)
            ladder = cudaq.qvector(2)
            target = cudaq.qvector(2)
            work = cudaq.qvector(2)
            for i in range(2):
                if ((a_init >> i) & 1) == 1:
                    x(address[i])
                if ((l_init >> i) & 1) == 1:
                    x(ladder[i])
                if ((t_init >> i) & 1) == 1:
                    x(target[i])
                if ((w_init >> i) & 1) == 1:
                    x(work[i])
            interp(address, ladder, target, work)

        return run_w

    @cudaq.kernel
    def run_p(a_init: int, l_init: int, t_init: int):
        address = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        for i in range(2):
            if ((a_init >> i) & 1) == 1:
                x(address[i])
            if ((l_init >> i) & 1) == 1:
                x(ladder[i])
            if ((t_init >> i) & 1) == 1:
                x(target[i])
        interp(address, ladder, target)

    return run_p


def _layout(controlled, has_work):
    """(register -> bit offset, total qubits) for the harness layout."""
    offsets = {}
    off = 0
    if controlled:
        offsets["c"] = off
        off += 1
    for reg in ("a", "l", "t"):
        offsets[reg] = off
        off += _WIDTHS[reg]
    if has_work:
        offsets["w"] = off
        off += 2
    return offsets, off


def _apply(kind, refs, regs):
    """The opcode's documented action on basis-state register values."""
    out = dict(regs)

    def bit(ref):
        reg, i = ref
        return (out[reg] >> i) & 1

    def flip(ref):
        reg, i = ref
        out[reg] ^= 1 << i

    phase = 1.0 + 0.0j
    if kind == "x":
        flip(refs[0])
    elif kind == "cx":
        if bit(refs[0]):
            flip(refs[1])
    elif kind == "ccx":
        if bit(refs[0]) and bit(refs[1]):
            flip(refs[2])
    elif kind == "cy":
        if bit(refs[0]):
            phase = 1.0j if bit(refs[1]) == 0 else -1.0j
            flip(refs[1])
    elif kind == "cz":
        if bit(refs[0]) and bit(refs[1]):
            phase = -1.0
    elif kind == "z":
        if bit(refs[0]):
            phase = -1.0
    else:
        raise AssertionError(f"unknown gate kind {kind}")
    return out, phase


@pytest.mark.parametrize("opcode,operands,kind,refs,controlled,has_work",
                         _spec_cases())
def test_opcode_semantics(opcode, operands, kind, refs, controlled, has_work):
    a, b, c = operands
    harness = _mint_harness((opcode, a, b, c), controlled, has_work)
    offsets, num_qubits = _layout(controlled, has_work)

    touched = sorted(set(refs))
    for combo in range(1 << len(touched)):
        regs = {"c": 0, "a": 0, "l": 0, "t": 0, "w": 0}
        for j, (reg, i) in enumerate(touched):
            if (combo >> j) & 1:
                regs[reg] |= 1 << i

        args = []
        if controlled:
            args.append(regs["c"])
        args.extend([regs["a"], regs["l"], regs["t"]])
        if has_work:
            args.append(regs["w"])
        state = np.array(cudaq.get_state(harness, *args))

        out, phase = _apply(kind, refs, regs)
        expected = np.zeros(1 << num_qubits, dtype=np.complex128)
        index = sum(out[reg] << off for reg, off in offsets.items())
        expected[index] = phase
        np.testing.assert_allclose(
            state, expected, atol=1e-12,
            err_msg=(f"opcode {opcode} (ctrl={controlled}, work={has_work}) "
                     f"wrong on input {regs}"))


def test_mint_rejects_out_of_set_opcodes():
    # A work opcode on a no-work tape (and a control opcode on an
    # uncontrolled tape) must be a loud mint-time error: the dispatch
    # loop would silently skip it.
    with pytest.raises(ValueError, match="outside the interpreter"):
        _mint_interpreter([(_OP_AND_TT, 0, 1, 1)],
                          controlled=False,
                          has_work=False)
    with pytest.raises(ValueError, match="outside the interpreter"):
        _mint_interpreter([(_OP_CCX_CTRL, 0, 1, 1)],
                          controlled=False,
                          has_work=True)
    with pytest.raises(ValueError, match="outside the interpreter"):
        _mint_interpreter([(_OP_BODY_Z_W, 0, 1, 0)],
                          controlled=True,
                          has_work=False)


def test_opcode_sets_cover_the_specs():
    # The spec table above and the module's declared dispatch sets must
    # name exactly the same opcodes — a new opcode must land in both.
    spec_ops = {spec[0] for spec in _SPECS.values()}
    assert spec_ops == (_BASE_OPS | _CONTROL_OPS | _WORK_OPS)
