# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed, QROM-programmed and selected-reflection fermionic basis changes.

Run: python3 fermion_basis_change.py. Uses fp64 qpp-cpu by default.
The full two-mode fixture uses sixteen qubits; partial programs use three
angle words instead of four. All references include every particle sector.
Reuses the example from PR #78, revision 6f542f156e5d719212cc8fcb364899d3c25bcb8a.
"""

# [Begin Documentation]
import os

import cudaq
import numpy as np

from cudaq_algorithms import fermion_basis_change, state_from
from cudaq_algorithms.basis_change import (
    GivensStructure, estimate_fermion_basis_change_resources)
from cudaq_algorithms.primitives import QROM

ATOL = 1e-10


def check(label, actual, expected, tolerance=ATOL):
    error = float(np.max(np.abs(actual - expected)))
    print(f"{label}: max error={error:.3e}, tolerance={tolerance:.1e} "
          f"{'PASS' if error <= tolerance else 'FAIL'}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def fock_two_modes(orbital):
    """Vacuum, mode 0, mode 1, both; vacuum phase is fixed to one."""
    result = np.zeros((4, 4), dtype=complex)
    result[0, 0] = 1
    result[1:3, 1:3] = orbital
    result[3, 3] = np.linalg.det(orbital)
    return result


def reconstruct(plan):
    """Independent NumPy product of the public chronological metadata."""
    angles = [plan.angle_encoding.decode(code) for code in plan.encoded_words]
    result = np.eye(plan.structure.num_modes, dtype=complex)
    for op in plan.schedule.operations:
        theta = op.scale * angles[op.parameter_index]
        gate = np.eye(len(result), dtype=complex)
        if op.kind == "phase":
            gate[op.first, op.first] = np.exp(1j * theta)
        else:
            c, s = np.cos(theta), np.sin(theta)
            gate[np.ix_([op.first, op.second],
                        [op.first, op.second])] = [[c, -s], [s, c]]
        result = gate @ result
    check("decoded metadata vs schedule", result,
          plan.schedule.orbital_matrix(plan.represented_parameters))
    return result


def ideal_reflection(column):
    """Compare second quantization with independent JW creation matrices."""
    # a0^dagger and a1^dagger, qubit zero least significant.
    raise_ = np.array([[0., 0.], [1., 0.]])
    creators = [
        np.kron(np.eye(2), raise_),
        np.kron(raise_, np.diag([1., -1.]))
    ]
    b_dagger = sum(v * a for v, a in zip(column, creators))
    jw = np.eye(4) - 2 * b_dagger @ b_dagger.conj().T
    lifted = fock_two_modes(np.eye(2) - 2 * np.outer(column, column.conj()))
    check("orbital reflection vs JW I-2b†b", lifted, jw)
    return lifted


def selected_reflections(columns, bits, label):
    """Lookup, W†, Z_reference, W, unlookup, preserving address phase."""
    structure = GivensStructure.partial(2, reference_mode=0)
    plans = [
        fermion_basis_change(v,
                             structure=structure,
                             programmable=True,
                             angle_bits=bits) for v in columns
    ]
    program = plans[0]
    assert structure.num_parameters == 3  # retain zero-valued slots
    assert all(p.structure == program.structure for p in plans)
    rows = [
        sum(code << (bits * slot) for slot, code in enumerate(p.encoded_words))
        for p in plans
    ]
    table = QROM(rows,
                 address_bits=1,
                 output_bits=bits * structure.num_parameters,
                 variant="select")
    lookup = table.kernel  # property, unlike the plan's kernel factories
    forward, inverse = program.kernel(), program.adjoint_kernel()
    span = program.num_qubits
    address = span
    scratch = address + table.num_address
    enable = scratch + table.num_ladder
    represented, ideals = [], []
    z_reference = np.diag([1., -1., 1., -1.])
    for p, v in zip(plans, columns):
        w = fock_two_modes(reconstruct(p))
        represented.append(w @ z_reference @ w.conj().T)
        ideals.append(ideal_reflection(v))

    rng = np.random.default_rng(7)
    ket = rng.normal(size=4) + 1j * rng.normal(size=4)
    ket /= np.linalg.norm(ket)  # vacuum and both particles carry phase too
    address_amplitudes = np.array([1., 1j]) / np.sqrt(2)
    for external, controls, description in ((False, [1.], "uncontrolled"),
                                            (True, [1., 0.], "enable=0"),
                                            (True, [0., 1.], "enable=1"),
                                            (True,
                                             [1 / np.sqrt(2), 1j / np.sqrt(2)
                                              ], "coherent enable")):

        total = enable + int(external)

        @cudaq.kernel
        def reflect(state: cudaq.State):
            q = cudaq.qvector(state)
            lookup(q[address:scratch], q[scratch:enable], q[2:span])
            inverse(q[0:span])
            if external:
                z.ctrl(q[enable], q[0])
            else:
                z(q[0])
            forward(q[0:span])
            cudaq.adjoint(lookup, q[address:scratch], q[scratch:enable],
                          q[2:span])

        initial = np.zeros(1 << total, dtype=complex)
        expected = np.zeros_like(initial)
        for control, amplitude in enumerate(controls):
            for j, address_amplitude in enumerate(address_amplitudes):
                indices = np.arange(4) + (j << address) + (control << enable)
                weight = amplitude * address_amplitude
                initial[indices] = weight * ket
                operator = (np.eye(4)
                            if external and control == 0 else represented[j])
                expected[indices] = weight * (operator @ ket)
        actual = np.asarray(cudaq.get_state(reflect, state_from(initial)))
        check(f"{label}, b={bits}, {description}, full state/workspace",
              actual, expected)

    resources = estimate_fermion_basis_change_resources(program)
    print(
        f"{label}: {structure.num_system_qubits} system + "
        f"{resources.num_program_qubits} angle + {table.num_address} address + "
        f"{table.num_ladder} ladder = {enable} qubits; "
        f"external enable adds one ({total}); slots={structure.num_parameters}, "
        f"radian step={program.angle_encoding.step:.6g}")
    print(
        f"  consumer: {resources.num_bit_controlled_rotations} bit-controlled "
        "rotations; excludes lookup/unlookup and external control")
    for j, p in enumerate(plans):
        error = float(np.linalg.norm(represented[j] - ideals[j], 2))
        action_error = float(np.linalg.norm(
            (represented[j] - ideals[j]) @ ket))
        bound = p.quantization_error.reflection_error_bound
        print(f"  row {j}: words={p.encoded_words}; ideal/represented "
              f"operator error={error:.6g}, action error={action_error:.6g}, "
              f"reflection bound={bound:.6g}")
        assert error <= bound + ATOL
        assert action_error <= error + ATOL


def full_transformations():
    print("Fixed/externally controlled transformations use classical angles; "
          "QROM-selected transformations consume caller-owned quantum words.")
    # A 2x2 mode matrix acts on TWO occupation qubits. Its full Fock action
    # preserves vacuum, acts as U on one particle, and as det(U) on two particles.
    U = np.array([[1., -1.], [1j, 1j]]) / np.sqrt(2)
    fixed = fermion_basis_change(U)
    apply_fixed = fixed.kernel()

    # H on both qubits gives amplitude 1/2 on every Fock state.
    @cudaq.kernel
    def fixed_demo():
        system = cudaq.qvector(2)
        h(system[0])
        h(system[1])
        apply_fixed(system)

    fock = np.zeros((4, 4), dtype=complex)
    fock[0, 0] = 1
    fock[1:3, 1:3] = U
    fock[3, 3] = np.linalg.det(U)
    check("fixed full Fock action",
          np.asarray(cudaq.get_state(fixed_demo)),
          fock @ (np.ones(4) / 2),
          tolerance=1e-12)

    # An external enable can coherently control the same fixed transformation.
    # The returned wrapper expects [enable, system...]. Its angles stay classical.
    controlled_fixed = fixed.controlled_kernel()

    @cudaq.kernel
    def controlled_demo():
        q = cudaq.qvector(3)
        h(q[0])
        x(q[1])
        controlled_fixed(q)

    expected_controlled = np.zeros(8, dtype=complex)
    expected_controlled[2] = 1 / np.sqrt(2)  # enable=0, mode 0 unchanged
    expected_controlled[3] = U[0, 0] / np.sqrt(2)
    expected_controlled[5] = U[1, 0] / np.sqrt(2)
    check("controlled fixed Fock action",
          np.asarray(cudaq.get_state(controlled_demo)),
          expected_controlled,
          tolerance=1e-12)

    # Now choose I or U using a quantum address. Both matrices share one Givens
    # structure, so one programmable circuit can consume either word string.
    # QROM is one way to load those words.
    bits = 3
    plans = [
        fermion_basis_change(matrix, programmable=True, angle_bits=bits)
        for matrix in (np.eye(2), U)
    ]
    program = plans[0]
    assert all(p.structure == program.structure for p in plans)
    assert plans[0].encoded_words == (0, 0, 0, 0)
    assert plans[1].encoded_words == (0, 2, 1, 2)

    # Four slots: diagonal phases, mixing angle, relative phase. Keep zero slots:
    # the loaded quantum words override every slot, including host zero angles.
    # With the default layout and the output view q[2:span], slot s starts at
    # output bit bits*s; other layouts use parameter_registers (see the guide).
    rows = [
        sum(code << (bits * slot) for slot, code in enumerate(p.encoded_words))
        for p in plans
    ]
    table = QROM(rows,
                 address_bits=1,
                 output_bits=bits * program.structure.num_parameters,
                 variant="select")
    lookup = table.kernel
    consume = program.kernel()
    span = program.num_qubits  # 2 system + 4 slots * 3 bits = 14
    address = span
    scratch = address + table.num_address
    total = scratch + table.num_ladder  # 16, including address and QROM scratch

    @cudaq.kernel
    def selected_preparation():
        q = cudaq.qvector(total)
        x(q[0])
        h(q[address])
        lookup(q[address:scratch], q[scratch:total], q[2:span])
        consume(q[0:span])
        lookup(q[address:scratch], q[scratch:total], q[2:span])

    # |0>_addr |mode 0>/sqrt(2) + |1>_addr (|mode 0>+i|mode 1>)/2.
    # Compare the complete state, so nonzero angle/scratch outputs also fail.
    expected = np.zeros(1 << total, dtype=complex)
    expected[1] = 1 / np.sqrt(2)
    expected[(1 << address) + 1] = 0.5
    expected[(1 << address) + 2] = 0.5j
    state = np.asarray(cudaq.get_state(selected_preparation))
    check("full QROM preparation/workspace", state, expected, tolerance=1e-11)
    print("Selected preparation:",
          [(i, complex(a)) for i, a in enumerate(state) if abs(a) > 1e-10])
    print("Register widths:", program.structure.num_system_qubits, "system +",
          table.num_output, "angles +", table.num_address, "address +",
          table.num_ladder, "scratch =", total)
    print("Quantization bounds:", [p.quantization_error_bound for p in plans])
    print("Fixed, controlled, and selected checks passed.")


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    full_transformations()
    selected_reflections([np.array([1., 0.]),
                          np.array([1., 1j]) / np.sqrt(2)], 3, "on-grid")
    off_grid = np.array([np.cos(.47), np.exp(.61j) * np.sin(.47)])
    for bits in (3, 4):
        selected_reflections([np.array([1., 0.]), off_grid], bits, "off-grid")
    print("All fixed, programmed and selected-reflection checks PASS.")


if __name__ == "__main__":
    main()
