#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Fixed, QROM-programmed and selected-reflection fermionic basis changes.

Fermionic simulation often becomes simpler after mixing the orbital modes into
a different basis. A Givens rotation performs one such mixing between a pair
of modes while preserving particle number. This example first applies a fixed
basis change, then shows how a quantum address can load angle words from QROM
so one programmable circuit performs the transformation selected by that
address.

You should learn the difference between enabling a fixed operation and
supplying its parameters as quantum data. A selected-orbital reflection then
transforms into the chosen orbital basis, marks occupation with a sign, and
transforms back, providing a useful building block for chemistry encodings.
Complete-state NumPy checks follow the address phases, particle sectors, and
cleared angle/scratch registers. Comparing two angle precisions shows how a
represented transformation can match its circuit closely while still
approximating the desired transformation.

Run: python3 fermion_basis_change.py. Uses fp64 qpp-cpu by default.
The full two-mode fixture uses sixteen qubits; partial programs use three
angle words instead of four. All references include every particle sector.
Add --verbose to show individual validation checks; failures are always reported.
"""

import argparse
import os

import cudaq
import numpy as np

from cudaq_algorithms import fermion_basis_change, state_from
from cudaq_algorithms.basis_change import (
    GivensStructure, estimate_fermion_basis_change_resources)
from cudaq_algorithms.primitives import QROM

ATOL = 1e-10


def bound_check(label, error, tolerance, *, verbose=False):
    passed = np.isfinite(error) and error <= tolerance
    if verbose or not passed:
        print(f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
              f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(label)


def check(label, actual, expected, *, verbose=False):
    bound_check(label,
                float(np.max(np.abs(actual - expected))),
                ATOL,
                verbose=verbose)


def full_transformations(*, verbose=False):
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
          verbose=verbose)

    # An external enable can coherently control the same fixed transformation.
    # The returned wrapper expects [enable, system...]. Its angles stay classical.
    controlled_fixed = fixed.controlled_kernel()

    @cudaq.kernel
    def controlled_demo():
        q = cudaq.qvector(3)
        h(q[0])
        h(q[1])
        h(q[2])
        controlled_fixed(q)

    # The enable is qubit 0: even indices (enable=0) keep the uniform two-mode
    # state, odd indices (enable=1) carry the full Fock action.
    uniform = np.ones(4) / (2 * np.sqrt(2))
    expected_controlled = np.zeros(8, dtype=complex)
    expected_controlled[0::2] = uniform
    expected_controlled[1::2] = fock @ uniform
    check("controlled fixed Fock action",
          np.asarray(cudaq.get_state(controlled_demo)),
          expected_controlled,
          verbose=verbose)

    # Now choose I or U using a quantum address. Both matrices share one Givens
    # structure, so one programmable circuit can consume either word string.
    # QROM is one way to load those words.
    bits = 3
    plans = [
        fermion_basis_change(matrix, programmable=True, angle_bits=bits)
        for matrix in (np.eye(2), U)
    ]
    program = plans[0]
    if any(p.structure != program.structure for p in plans):
        raise AssertionError("table rows must share one Givens structure")
    if [p.encoded_words for p in plans] != [(0, 0, 0, 0), (0, 2, 1, 2)]:
        raise AssertionError("unexpected angle words for I and U")

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
        h(q[0])
        h(q[1])
        h(q[address])
        lookup(q[address:scratch], q[scratch:total], q[2:span])
        consume(q[0:span])
        lookup(q[address:scratch], q[scratch:total], q[2:span])

    # Address 0 keeps the uniform two-mode state; address 1 applies the full
    # Fock action of U, so all four loaded words and det(U) are observed.
    # Compare the complete state, so nonzero angle/scratch outputs also fail.
    expected = np.zeros(1 << total, dtype=complex)
    expected[0:4] = uniform
    expected[(1 << address):(1 << address) + 4] = fock @ uniform
    state = np.asarray(cudaq.get_state(selected_preparation))
    check("full QROM preparation/workspace", state, expected, verbose=verbose)
    print("Selected preparation:",
          [(i, complex(np.round(a, 3)))
           for i, a in enumerate(state) if abs(a) > ATOL])
    print("Register widths:", program.structure.num_system_qubits, "system +",
          table.num_output, "angles +", table.num_address, "address +",
          table.num_ladder, "scratch =", total)
    print("Quantization bounds:", [p.quantization_error_bound for p in plans])
    print("Fixed, controlled, and selected checks passed.")


def fock_two_modes(orbital):
    """Vacuum, mode 0, mode 1, both; vacuum phase is fixed to one."""
    result = np.zeros((4, 4), dtype=complex)
    result[0, 0] = 1
    result[1:3, 1:3] = orbital
    result[3, 3] = np.linalg.det(orbital)
    return result


def reconstruct(plan, *, verbose=False):
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
    check("decoded metadata vs schedule",
          result,
          plan.schedule.orbital_matrix(plan.represented_parameters),
          verbose=verbose)
    return result


def ideal_reflection(column, *, verbose=False):
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
    check("orbital reflection vs JW I-2b†b", lifted, jw, verbose=verbose)
    return lifted


def selected_reflections(columns, bits, label, *, verbose=False):
    """Lookup, W†, Z_reference, W, inverse lookup; keeps the address phase."""
    structure = GivensStructure.partial(2, reference_mode=0)
    plans = [
        fermion_basis_change(v,
                             structure=structure,
                             programmable=True,
                             angle_bits=bits) for v in columns
    ]
    program = plans[0]
    if structure.num_parameters != 3:  # zero-valued slots are kept
        raise AssertionError("a partial two-mode plan has three slots")
    if any(p.structure != program.structure for p in plans):
        raise AssertionError("table rows must share one Givens structure")
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
        w = fock_two_modes(reconstruct(p, verbose=verbose))
        represented.append(w @ z_reference @ w.conj().T)
        ideals.append(ideal_reflection(v, verbose=verbose))

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
            # Inverse lookup. The XOR lookup is its own inverse, so the same
            # call clears the words and the ladder.
            lookup(q[address:scratch], q[scratch:enable], q[2:span])

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
              actual,
              expected,
              verbose=verbose)

    resources = estimate_fermion_basis_change_resources(program)
    print(
        f"{label}: {structure.num_system_qubits} system + "
        f"{resources.num_program_qubits} angle + {table.num_address} address + "
        f"{table.num_ladder} ladder = {enable} qubits; "
        f"external enable adds one ({total}); slots={structure.num_parameters}, "
        f"radian step={program.angle_encoding.step:.6g}")
    print(
        f"  consumer: {resources.num_bit_controlled_rotations} bit-controlled "
        "rotations; excludes lookup, inverse lookup and external control")
    for j, p in enumerate(plans):
        error = float(np.linalg.norm(represented[j] - ideals[j], 2))
        action_error = float(np.linalg.norm(
            (represented[j] - ideals[j]) @ ket))
        bound = p.quantization_error.reflection_error_bound
        print(f"  row {j}: words={p.encoded_words}; ideal/represented "
              f"operator error={error:.6g}, action error={action_error:.6g}, "
              f"reflection bound={bound:.6g}")
        bound_check(f"{label}, b={bits}, row {j}, operator error within bound",
                    error,
                    bound + ATOL,
                    verbose=verbose)
        bound_check(
            f"{label}, b={bits}, row {j}, action error within operator",
            action_error,
            error + ATOL,
            verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each validation check (failures are always reported)")
    verbose = parser.parse_args().verbose
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    full_transformations(verbose=verbose)
    selected_reflections([np.array([1., 0.]),
                          np.array([1., 1j]) / np.sqrt(2)],
                         3,
                         "on-grid",
                         verbose=verbose)
    off_grid = np.array([np.cos(.47), np.exp(.61j) * np.sin(.47)])
    for bits in (3, 4):
        selected_reflections([np.array([1., 0.]), off_grid],
                             bits,
                             "off-grid",
                             verbose=verbose)
    print("All fixed, programmed and selected-reflection checks PASS.")


if __name__ == "__main__":
    main()
