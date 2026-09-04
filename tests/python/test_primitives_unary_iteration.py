# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused unary-iteration walk.

Two layers of evidence. The emitted instruction tape is verified
*classically and exhaustively* (every address width up to 6, every
partial-tree size, controlled and uncontrolled) by tracking each ladder
line as a Boolean truth table over the address bits: at every body slot
the leaf line must be the exact address-``k`` indicator, and at the end
the ladder must be zero and the address wires restored — that is the
full circuit identity "walk == SELECT". The minted kernels are then
pinned on the statevector: per-address phase kicks (including partial
trees), the Y-body sign convention, the hand-written inverse against
non-commuting bodies, and the externally controlled variant at |0> and
|1>.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import unary_iteration_kernels
from cudaq_algorithms.primitives._unary_iteration import (
    _OP_BODY_X, _OP_CCX, _OP_CCX_ADDR_ADDR, _OP_CCX_CTRL, _OP_CX_ADDR_ADDR,
    _OP_CX_ADDR_LADDER, _OP_CX_CTRL_LADDER, _OP_CX_LADDER_LADDER, _OP_X_ADDR,
    _OP_X_LADDER, _TOFFOLI_OPCODES, _emit_walk, _walk_toffoli_count)


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


# ----------------------------------------------------------------------
# Classical truth-table verification of the emitted tape
# ----------------------------------------------------------------------


def _verify_tape(num_bits: int, num_items: int, controlled: bool):
    """Track every wire as a truth table over the (control +) address
    bits; body markers are injected through a body that emits a plain
    ("x", 0) so their tape positions are recoverable."""
    marker = [("x", 0)]
    ops = _emit_walk(num_bits, num_items, controlled, lambda k: marker)
    num_inputs = num_bits + (1 if controlled else 0)
    npoints = 1 << num_inputs
    full = (1 << npoints) - 1

    def input_wire(i):
        table = 0
        for point in range(npoints):
            if (point >> i) & 1:
                table |= 1 << point
        return table

    address = [input_wire(i) for i in range(num_bits)]
    control = input_wire(num_bits) if controlled else 0
    ladder = [0] * num_bits
    original = list(address)
    bodies_seen = 0
    for opcode, a, b, c in ops:
        if opcode == _OP_X_ADDR:
            address[a] ^= full
        elif opcode == _OP_X_LADDER:
            ladder[a] ^= full
        elif opcode == _OP_CX_ADDR_LADDER:
            ladder[b] ^= address[a]
        elif opcode == _OP_CX_LADDER_LADDER:
            ladder[b] ^= ladder[a]
        elif opcode == _OP_CCX:
            ladder[c] ^= ladder[a] & address[b]
        elif opcode == _OP_BODY_X:
            # The body marker: leaf line must be [addr == k].
            expected = 0
            for point in range(npoints):
                active = (point & ((1 << num_bits) - 1)) == bodies_seen
                if controlled and not ((point >> num_bits) & 1):
                    active = False
                if active:
                    expected |= 1 << point
            assert ladder[a] == expected, (
                f"body({bodies_seen}): leaf line is not the address "
                f"indicator (bits={num_bits}, items={num_items}, "
                f"controlled={controlled})")
            bodies_seen += 1
        elif opcode == _OP_CX_CTRL_LADDER:
            ladder[b] ^= control
        elif opcode == _OP_CCX_CTRL:
            ladder[c] ^= control & address[b]
        elif opcode == _OP_CX_ADDR_ADDR:
            address[b] ^= address[a]
        elif opcode == _OP_CCX_ADDR_ADDR:
            ladder[c] ^= address[a] & address[b]
        else:
            raise AssertionError(f"unexpected opcode {opcode} in walk tape")
    assert bodies_seen == num_items
    assert address == original, "address wires not restored"
    assert all(line == 0 for line in ladder), "ladder not returned to |0>"
    return sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)


@pytest.mark.parametrize("controlled", [False, True])
def test_tape_is_select_for_every_tree_shape(controlled):
    # Exhaustive: every width up to 6 and every partial-tree size.
    for num_bits in range(1, 7):
        for num_items in range(1, (1 << num_bits) + 1):
            _verify_tape(num_bits, num_items, controlled)


def test_tape_toffoli_counts_match_documented_formulas():
    # Fused-walk headline: 3N/2 - 5 uncontrolled (N >= 8; 2 at N = 4,
    # 0 at N = 2) and 3N/2 - 1 controlled, on full trees. The paper's
    # N - 1 (arXiv:1805.03662) assumes measurement-based uncomputation;
    # these are the strictly unitary counts.
    for num_bits in range(1, 8):
        capacity = 1 << num_bits
        counted = _verify_tape(num_bits, capacity, False)
        assert counted == _walk_toffoli_count(num_bits, capacity, False)
        if num_bits >= 3:
            assert counted == 3 * capacity // 2 - 5
        counted = _verify_tape(num_bits, capacity, True)
        assert counted == _walk_toffoli_count(num_bits, capacity, True)
        assert counted == 3 * capacity // 2 - 1
        # Partial trees: the analytic counter matches the tape exactly.
        for num_items in range(1, capacity + 1):
            assert (_verify_tape(num_bits, num_items,
                                 False) == _walk_toffoli_count(
                                     num_bits, num_items, False))


# ----------------------------------------------------------------------
# Statevector pinning of the minted kernels
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_items,marked", [(8, (1, 3, 4)), (5, (0, 4)),
                                              (1, (0, ))])
def test_unary_iteration_phase_kick_per_address(num_items, marked):
    # body(k) = Z on a |1> target for marked addresses: a uniform address
    # superposition picks up a -1 phase exactly on those addresses.
    walk = unary_iteration_kernels(3, num_items, lambda k: [("z", 0)]
                                   if k in marked else [])
    kernel = walk.kernel

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        target = cudaq.qvector(1)
        x(target[0])
        for b in range(3):
            h(address_reg[b])
        kernel(address_reg, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 7, dtype=np.complex128)
    for address in range(8):
        sign = -1.0 if address in marked else 1.0
        expected[address + (1 << 6)] = sign / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_unary_iteration_y_body_sign_convention():
    # body(2) = Y on target 0: |2>|0> -> i |2>|1>; other addresses idle.
    walk = unary_iteration_kernels(2, 4, lambda k: [("y", 0)]
                                   if k == 2 else [])
    kernel = walk.kernel

    def run_state(address: int) -> np.ndarray:

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(2)
            ladder = cudaq.qvector(2)
            target = cudaq.qvector(1)
            for b in range(2):
                if ((address >> b) & 1) == 1:
                    x(address_reg[b])
            kernel(address_reg, ladder, target)

        return np.array(cudaq.get_state(run))

    expected = np.zeros(1 << 5, dtype=np.complex128)
    expected[2 + (1 << 4)] = 1.0j
    np.testing.assert_allclose(run_state(2), expected, atol=1e-12)
    np.testing.assert_allclose(run_state(1), _basis(1, 5), atol=1e-12)


def test_unary_iteration_matches_dense_select_reference():
    # Independent dense reference: SELECT = sum_k |k><k| (x) U_k (+
    # identity on unvisited addresses), built from NumPy Pauli matrices.
    # Multi-gate bodies on two targets, partial tree.
    paulis = {
        "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    bodies = {
        0: [("x", 0), ("y", 1)],
        1: [("z", 0)],
        2: [("y", 0), ("x", 1), ("z", 1)],
        4: [("x", 1)],
    }
    num_items = 5
    walk = unary_iteration_kernels(3, num_items, lambda k: bodies.get(k, []))
    kernel = walk.kernel
    identity = np.eye(4, dtype=np.complex128)

    def body_matrix(k: int) -> np.ndarray:
        matrix = identity
        for gate, t in bodies.get(k, []):
            op = [np.eye(2, dtype=np.complex128)] * 2
            op[t] = paulis[gate]
            # qubit 0 = LSB: rightmost Kronecker factor.
            matrix = np.kron(op[1], op[0]) @ matrix
        return matrix

    # Column extraction: every (address, target) basis input against the
    # matching dense-reference column.
    for address in range(8):
        for column in range(4):

            @cudaq.kernel
            def run():
                address_reg = cudaq.qvector(3)
                ladder = cudaq.qvector(3)
                target = cudaq.qvector(2)
                for b in range(3):
                    if ((address >> b) & 1) == 1:
                        x(address_reg[b])
                for b in range(2):
                    if ((column >> b) & 1) == 1:
                        x(target[b])
                kernel(address_reg, ladder, target)

            state = np.array(cudaq.get_state(run))
            expected = np.zeros(1 << 8, dtype=np.complex128)
            reference = (body_matrix(address)
                         if address < num_items else identity)[:, column]
            for t in range(4):
                expected[address + (t << 6)] = reference[t]
            np.testing.assert_allclose(state, expected, atol=1e-12)


def test_unary_iteration_adjoint_composes_to_identity():
    # Non-commuting multi-gate bodies (X, Z, Y mixed) so the walk is NOT
    # an involution; only the hand-written reversed-instruction inverse
    # restores the state.
    def body(k):
        if k % 2 == 1:
            return [("x", 0), ("z", 0), ("y", 1)]
        return [("x", 1)]

    walk = unary_iteration_kernels(3, 6, body)
    kernel = walk.kernel
    kernel_adj = walk.kernel_adj

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        target = cudaq.qvector(2)
        x(target[0])
        for b in range(3):
            h(address_reg[b])
        kernel(address_reg, ladder, target)
        kernel_adj(address_reg, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 8, dtype=np.complex128)
    for address in range(8):
        expected[address + (1 << 6)] = 1.0 / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("control_value", [0, 1])
def test_unary_iteration_controlled_variant(control_value):
    marked = (1, 2)
    walk = unary_iteration_kernels(2,
                                   4,
                                   lambda k: [("z", 0)] if k in marked else [],
                                   controlled=True)
    kernel = walk.kernel
    assert walk.controlled

    @cudaq.kernel
    def run():
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(1)
        if control_value == 1:
            x(control[0])
        x(target[0])
        for b in range(2):
            h(address_reg[b])
        kernel(control, address_reg, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 6, dtype=np.complex128)
    for address in range(4):
        kicked = control_value == 1 and address in marked
        sign = -1.0 if kicked else 1.0
        # Layout: control at 0, address at [1, 3), ladder [3, 5), target 5.
        expected[control_value + (address << 1) + (1 << 5)] = \
            sign / 2.0
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_unary_iteration_validation_raises():
    body = lambda k: []
    with pytest.raises(ValueError, match="num_address_bits must be a"):
        unary_iteration_kernels(0, 1, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        unary_iteration_kernels(2, 0, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        unary_iteration_kernels(2, 5, body)
    with pytest.raises(ValueError, match="unsupported gate 'q'"):
        unary_iteration_kernels(2, 4, lambda k: [("q", 0)])
    with pytest.raises(ValueError, match="invalid target qubit index"):
        unary_iteration_kernels(2, 4, lambda k: [("x", -1)])


def test_describe_decodes_the_tape():
    # describe() is the human-readable form of the minted circuit: its
    # Toffoli-tagged lines must agree with toffoli_count, and the body
    # gates must appear once per address, in address order.
    walk = unary_iteration_kernels(2, 4, lambda k: [("x", 0)])
    text = walk.describe()
    lines = text.splitlines()
    assert lines[0].startswith("unary-iteration walk: 4 addresses")
    assert sum("# Toffoli" in line for line in lines) == walk.toffoli_count
    body_lines = [line for line in lines if line.startswith("x(target[0])")]
    assert len(body_lines) == 4
