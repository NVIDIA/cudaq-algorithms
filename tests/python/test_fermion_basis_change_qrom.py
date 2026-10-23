# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public factory integration with caller-owned QROM and coherent controls."""

import cudaq
import numpy as np
import pytest

from cudaq_algorithms import fermion_basis_change, state_from
from cudaq_algorithms.basis_change import GivensStructure
from cudaq_algorithms.primitives import QROM


def _fock_two_modes(unitary):
    result = np.zeros((4, 4), dtype=complex)
    result[0, 0] = 1
    result[1:3, 1:3] = unitary
    result[3, 3] = np.linalg.det(unitary)
    return result


def _fock_matrix(orbital):
    """Independent determinant minors, with qubit zero least significant."""
    occupied = [[p for p in range(len(orbital)) if bits & (1 << p)]
                for bits in range(1 << len(orbital))]
    result = np.zeros((len(occupied), len(occupied)), dtype=complex)
    for row, rows in enumerate(occupied):
        for col, cols in enumerate(occupied):
            if len(rows) == len(cols):
                result[row, col] = np.linalg.det(orbital[np.ix_(rows, cols)])
    return result


def _creation(qubit, size):
    result = np.zeros((1 << size, 1 << size), dtype=complex)
    for bits in range(1 << size):
        if not bits & (1 << qubit):
            parity = (bits & ((1 << qubit) - 1)).bit_count()
            result[bits | (1 << qubit), bits] = (-1)**parity
    return result


def _orbital(structure, codes, bits):
    """Orbital matrix of decoded words from the documented schedule.

    Full m=2 applies the diagonal phases, then P_1(phi) G(theta) P_1(-phi).
    The partial chains used here apply G_01, then G_12, then output phases.
    """
    angles = 2 * np.pi * np.asarray(codes) / (1 << bits)
    m = structure.num_modes

    def rotation(upper, theta):
        result = np.eye(m, dtype=complex)
        c, s = np.cos(theta), np.sin(theta)
        result[upper:upper + 2, upper:upper + 2] = [[c, -s], [s, c]]
        return result

    if structure.kind == "full":
        a, b, theta, phi = angles
        twist = np.diag([1, np.exp(1j * phi)])
        phases = np.diag(np.exp(1j * np.array([a, b])))
        return twist @ rotation(0, theta) @ twist.conj() @ phases
    result = np.eye(m, dtype=complex)
    for upper, theta in enumerate(angles[:m - 1]):
        result = rotation(upper, theta) @ result
    return np.diag(np.exp(1j * angles[m - 1:])) @ result


def _selected_operator(structure, codes, bits, action, reference=None):
    """Dense operator for one table row on the physical system span."""
    orbital = _orbital(structure, codes, bits)
    size = structure.num_system_qubits
    if action == "reflection":
        # Q = I - 2 b^dagger b with b^dagger = sum_p v_p a^dagger_(mode_map[p]).
        created = sum(orbital[p, reference] * _creation(qubit, size)
                      for p, qubit in enumerate(structure.mode_map))
        return np.eye(1 << size) - 2 * created @ created.conj().T
    physical = np.eye(size, dtype=complex)
    physical[np.ix_(structure.mode_map, structure.mode_map)] = orbital
    operator = _fock_matrix(physical)
    return operator.conj().T if action == "adjoint" else operator


def _reversed_reflection(structure, codes, bits, reference):
    """W^dagger Z_r W: the reflection with its conjugation order reversed."""
    forward = _selected_operator(structure, codes, bits, "forward")
    qubit = structure.mode_map[reference]
    parity = np.diag(
        [-1. if index >> qubit & 1 else 1. for index in range(len(forward))])
    return forward.conj().T @ parity @ forward


def _packed_row(codes, layout, base):
    """Put bit i of slot s at QROM output position layout[s][i] - base."""
    return sum(((code >> bit) & 1) << (qubit - base)
               for code, word in zip(codes, layout)
               for bit, qubit in enumerate(word))


def _branches(count, width, seed):
    """Random system amplitudes for each address, normalized jointly."""
    rng = np.random.default_rng(seed)
    branches = (rng.normal(size=(count, 1 << width)) +
                1j * rng.normal(size=(count, 1 << width)))
    return branches / np.linalg.norm(branches)


def _joint_state(branches, address, size):
    """Place branch j at address j with every other qubit clear."""
    state = np.zeros(1 << size, dtype=complex)
    for value, branch in enumerate(branches):
        state[np.arange(len(branch)) + (value << address)] = branch
    return state


def _lookup_apply_unlookup(lookup, operation, initial, *, words, address,
                           scratch):
    """Run lookup, ``operation`` on ``q[0:address]``, and inverse lookup."""
    total = int(np.log2(len(initial)))

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        lookup(q[address:scratch], q[scratch:total], q[words:address])
        operation(q[0:address])
        lookup(q[address:scratch], q[scratch:total], q[words:address])

    return np.asarray(
        cudaq.get_state(run, state_from(np.ascontiguousarray(initial))))


@pytest.mark.parametrize("kind", ["full", "partial"])
@pytest.mark.parametrize("action",
                         ["forward", "inverse", "reflection", "roundtrip"])
@pytest.mark.parametrize("controlled", [False, True])
def test_lookup_consume_unlookup_preserves_joint_state(kind, action,
                                                       controlled):
    # Exactly represented on the two-bit grid. The first plan has all zero
    # angles, so this also checks that runtime words override those slots.
    unitaries = [np.eye(2), np.array([[0., -1.], [1j, 0.]])]
    plans = [
        fermion_basis_change(u if kind == "full" else u[:, 0],
                             programmable=True,
                             angle_bits=2) for u in unitaries
    ]
    plan = plans[0]
    assert plans[1].structure == plan.structure
    assert all(p.quantization_error_bound < 1e-14 for p in plans)
    rows = [
        sum(code << (2 * slot) for slot, code in enumerate(p.encoded_words))
        for p in plans
    ]
    table = QROM(rows,
                 address_bits=1,
                 output_bits=2 * plan.structure.num_parameters,
                 variant="select")
    lookup = table.kernel
    forward, inverse = plan.kernel(), plan.adjoint_kernel()

    @cudaq.kernel
    def reflection(register: cudaq.qview):
        inverse(register)
        z(register[0])
        forward(register)

    @cudaq.kernel
    def roundtrip(register: cudaq.qview):
        forward(register)
        inverse(register)

    selected = {
        "forward": forward,
        "inverse": inverse,
        "reflection": reflection,
        "roundtrip": roundtrip
    }[action]
    span = plan.num_qubits

    @cudaq.kernel
    def controlled_composite(register: cudaq.qview):
        cudaq.control(selected, register[0], register[1:span + 1])

    if controlled:
        operation = (plan.controlled_kernel(adjoint=action == "inverse")
                     if action in ("forward",
                                   "inverse") else controlled_composite)
    else:
        operation = selected
    offset = int(controlled)
    address = offset + span
    scratch = address + 1
    total = scratch + table.num_ladder

    @cudaq.kernel
    def run():
        q = cudaq.qvector(total)
        # Coherence across all occupations detects vacuum/determinant phases.
        h(q[offset])
        h(q[offset + 1])
        h(q[address])
        if controlled:
            h(q[0])
        lookup(q[address:scratch], q[scratch:total], q[offset + 2:address])
        operation(q[0:address])
        lookup(q[address:scratch], q[scratch:total], q[offset + 2:address])

    actual = np.asarray(cudaq.get_state(run))
    expected = np.zeros(1 << total, dtype=complex)
    initial = np.ones(4) / 2
    for row, unitary in enumerate(unitaries):
        matrix = _fock_two_modes(unitary)
        if action == "inverse":
            matrix = matrix.conj().T
        elif action == "reflection":
            matrix = matrix @ np.diag([1., -1., 1., -1.]) @ matrix.conj().T
        elif action == "roundtrip":
            matrix = np.eye(4)
        for enable in range(2 if controlled else 1):
            branch = initial if controlled and enable == 0 else matrix @ initial
            indices = (np.arange(4) << offset) + (row << address) + enable
            expected[indices] = branch / np.sqrt(2 * (2 if controlled else 1))
    # Full joint comparison also asserts all angle and scratch qubits clear.
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("action", ["forward", "adjoint", "reflection"])
@pytest.mark.parametrize(
    "case", ["full", "partial", "partial_mixing", "three_bit_mixing"])
def test_distinct_rows_select_their_transformations(case, action):
    # Rows are distinct and not palindromic. Slot-order mistakes change
    # every case, bit-order mistakes every multi-bit case, and address-order
    # mistakes every case with two address bits, except the one-bit partial
    # reflection: it is Z on physical qubit 0 at every address, so it checks
    # only Z_r placement, padded row included.
    if case == "full":
        structure, bits, address_bits = GivensStructure.full(2), 2, 2
        reference = 1  # Reflect about a transformed mode other than zero.
        codes = [(0, 2, 1, 1), (1, 3, 2, 0), (0, 0, 1, 2), (2, 2, 0, 2)]
    elif case == "three_bit_mixing":
        # Three-bit words mix both modes. Address 3 reads the zero row.
        structure = GivensStructure.partial(2, reference_mode=1)
        reference = structure.reference_mode
        bits, address_bits = 3, 2
        codes = [(1, 3, 6), (5, 2, 7), (3, 6, 1)]
    else:
        structure = GivensStructure.partial(3,
                                            mode_map=(2, 0, 1),
                                            reference_mode=1)
        reference = structure.reference_mode
        if case == "partial":
            # One-bit words encode 0 or pi: signs without mixing. Address 3
            # reads the zero row.
            bits, address_bits = 1, 2
            codes = [(1, 0, 1, 1, 0), (0, 0, 1, 1, 1), (0, 1, 0, 1, 1)]
        else:
            # Two-bit words mix modes across mapped physical qubit 1.
            bits, address_bits = 2, 1
            codes = [(1, 3, 0, 2, 2), (2, 1, 2, 0, 2)]
    plan = fermion_basis_change(structure=structure,
                                programmable=True,
                                angle_bits=bits)
    words = structure.num_system_qubits
    # Default words follow the system span in parameter order.
    assert plan.parameter_registers == tuple(
        tuple(range(words + bits * slot, words + bits * (slot + 1)))
        for slot in range(structure.num_parameters))
    rows = [
        sum(code << (bits * slot) for slot, code in enumerate(row))
        for row in codes
    ]
    table = QROM(rows,
                 address_bits=address_bits,
                 output_bits=plan.num_qubits - words,
                 variant="select")
    forward, inverse = plan.kernel(), plan.adjoint_kernel()
    # Z acts on the physical qubit of the reflected logical mode.
    target = structure.mode_map[reference]

    @cudaq.kernel
    def reflection(register: cudaq.qview):
        inverse(register)
        z(register[target])
        forward(register)

    operation = {
        "forward": forward,
        "adjoint": inverse,
        "reflection": reflection
    }[action]
    address = plan.num_qubits
    scratch = address + address_bits
    size = scratch + table.num_ladder
    padded = codes + [(0, ) * structure.num_parameters] * (
        (1 << address_bits) - len(codes))
    branches = _branches(1 << address_bits, words, 2026)
    initial = _joint_state(branches, address, size)
    expected = _joint_state([
        _selected_operator(structure, row, bits, action, reference) @ branch
        for row, branch in zip(padded, branches)
    ], address, size)
    if action == "reflection" and case.endswith("mixing"):
        # Some row tells W Z_r W^dagger from the reversed W^dagger Z_r W, so
        # these cases also fix the conjugation order.
        order_sensitive = [
            not np.allclose(
                _selected_operator(structure, row, bits, action, reference),
                _reversed_reflection(structure, row, bits, reference))
            for row in codes
        ]
        assert any(order_sensitive)
    actual = _lookup_apply_unlookup(table.kernel,
                                    operation,
                                    initial,
                                    words=words,
                                    address=address,
                                    scratch=scratch)
    # The complete state also requires restored words and lookup scratch.
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("action", ["forward", "adjoint"])
def test_explicit_layout_rows_place_each_bit_at_its_register(action):
    structure = GivensStructure.full(2)
    # Reordered slots, with gaps at qubits 2 and 7 inside the output view.
    layout = ((10, 11), (3, 4), (8, 9), (5, 6))
    plan = fermion_basis_change(structure=structure,
                                programmable=True,
                                angle_bits=2,
                                parameter_registers=layout)
    assert plan.parameter_registers == layout
    assert plan.num_qubits == 12
    codes = [(1, 2, 2, 0), (3, 2, 1, 3)]
    base = structure.num_system_qubits  # first qubit of the output view
    rows = [_packed_row(row, layout, base) for row in codes]
    assert rows == [0b0110000100, 0b1101011100]  # gap positions 0, 5 clear
    table = QROM(rows,
                 address_bits=1,
                 output_bits=plan.num_qubits - base,
                 variant="select")
    operation = plan.kernel() if action == "forward" else plan.adjoint_kernel()
    address = plan.num_qubits
    scratch = address + 1
    size = scratch + table.num_ladder
    branches = _branches(2, structure.num_system_qubits, 711)
    initial = _joint_state(branches, address, size)
    expected = _joint_state([
        _selected_operator(structure, row, 2, action) @ branch
        for row, branch in zip(codes, branches)
    ], address, size)
    actual = _lookup_apply_unlookup(table.kernel,
                                    operation,
                                    initial,
                                    words=base,
                                    address=address,
                                    scratch=scratch)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("guard_invalid", [False, True])
def test_zero_padded_lookup_needs_validity_control_for_reflection(
        guard_invalid):
    structure = GivensStructure.partial(2, reference_mode=1)
    plan = fermion_basis_change(structure=structure,
                                programmable=True,
                                angle_bits=2)
    # Three distinct valid rows; address 3 reads as the all-zero row.
    codes = [(3, 0, 2), (0, 3, 3), (1, 0, 0)]
    rows = [
        sum(code << (2 * slot) for slot, code in enumerate(row))
        for row in codes
    ]
    table = QROM(rows, address_bits=2, output_bits=6, variant="select")
    assert plan.num_qubits == 8 and table.num_ladder == 2
    lookup = table.kernel
    forward, inverse = plan.kernel(), plan.adjoint_kernel()
    target = structure.mode_map[structure.reference_mode]

    @cudaq.kernel
    def run(state: cudaq.State):
        # System 0:2, angles 2:8, address 8:10, lookup scratch 10:12, valid 12.
        q = cudaq.qvector(state)
        lookup(q[8:10], q[10:12], q[2:8])
        inverse(q[0:8])
        if guard_invalid:
            x(q[12])
            x.ctrl(q[8:10], q[12])  # valid iff address != 3
            z.ctrl(q[12], q[target])
            x.ctrl(q[8:10], q[12])
            x(q[12])
        else:
            z(q[target])
        forward(q[0:8])
        lookup(q[8:10], q[10:12], q[2:8])

    operators = [
        _selected_operator(structure, row, 2, "reflection",
                           structure.reference_mode)
        for row in codes + [(0, 0, 0)]
    ]
    # Unguarded, the zero row still reflects the reference occupation.
    np.testing.assert_allclose(operators[3], np.diag([1., 1., -1., -1.]))
    if guard_invalid:
        operators[3] = np.eye(4)
    branches = _branches(4, 2, 61069)
    initial = _joint_state(branches, 8, 13)
    expected = _joint_state(
        [operator @ branch for operator, branch in zip(operators, branches)],
        8, 13)
    actual = np.asarray(
        cudaq.get_state(run, state_from(np.ascontiguousarray(initial))))
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)


def test_public_quantization_bound_covers_full_fock_operator_error():
    rng = np.random.default_rng(61069)
    unitary, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    plan = fermion_basis_change(unitary, programmable=True, angle_bits=3)
    represented = _fock_two_modes(
        plan.schedule.orbital_matrix(plan.represented_parameters))
    measured = np.linalg.norm(represented - _fock_two_modes(unitary), ord=2)
    assert measured <= plan.quantization_error_bound + 1e-12
    assert 0 < plan.quantization_error_bound < 2
