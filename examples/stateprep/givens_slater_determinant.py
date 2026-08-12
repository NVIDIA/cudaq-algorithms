#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Givens-rotation Slater determinant preparation.

Prepares a real 4-orbital / 2-electron determinant and a complex
5-orbital / 3-electron determinant with the composable ``stateprep``
kernels, and checks both against the dense reference (all minors of the
orbital-coefficient matrix).

Run with:  python3 givens_slater_determinant.py
"""

import os

import numpy as np

import cudaq

from cudaq_algorithms import stateprep


@cudaq.kernel
def prepare_real(num_spin_orbitals: int, orbital_indices: list[int],
                 angles: list[float], num_electrons: int):
    qubits = cudaq.qvector(num_spin_orbitals)
    stateprep.slater_determinant(qubits, orbital_indices, angles,
                                 num_electrons)


@cudaq.kernel
def prepare_complex(num_spin_orbitals: int, orbital_indices: list[int],
                    angles: list[float], phases: list[float],
                    final_phases: list[float], num_electrons: int):
    qubits = cudaq.qvector(num_spin_orbitals)
    stateprep.complex_slater_determinant(qubits, orbital_indices, angles,
                                         phases, final_phases, num_electrons)


def reference_slater_state(orbital_coefficients):
    orbital_coefficients = np.asarray(orbital_coefficients, dtype=complex)
    num_spin_orbitals, num_electrons = orbital_coefficients.shape
    state = np.zeros(2**num_spin_orbitals, dtype=complex)
    for basis_index in range(2**num_spin_orbitals):
        occupied = [
            orbital for orbital in range(num_spin_orbitals)
            if (basis_index >> orbital) & 1
        ]
        if len(occupied) != num_electrons:
            continue
        state[basis_index] = np.linalg.det(orbital_coefficients[np.ix_(
            occupied, range(num_electrons))])
    return state


def phase_aligned_l2(actual, expected):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    pivot = int(np.argmax(np.abs(expected)))
    phase = 1.0
    if abs(expected[pivot]) > 1.0e-14:
        phase = actual[pivot] / expected[pivot]
        phase /= abs(phase)
    return np.linalg.norm(actual - phase * expected)


def run_case(label, orbital_coefficients):
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
    resources = stateprep.estimate_givens_resources(schedule)

    indices = stateprep.get_givens_rotation_indices(schedule)
    angles = stateprep.get_givens_rotation_angles(schedule)
    if schedule.is_complex:
        state = cudaq.get_state(prepare_complex, schedule.num_spin_orbitals,
                                indices, angles,
                                stateprep.get_givens_rotation_phases(schedule),
                                list(schedule.final_phases),
                                schedule.num_electrons)
    else:
        state = cudaq.get_state(prepare_real, schedule.num_spin_orbitals,
                                indices, angles, schedule.num_electrons)

    error = phase_aligned_l2(np.asarray(state),
                             reference_slater_state(orbital_coefficients))
    print(label)
    print(f"  spin orbitals:          {schedule.num_spin_orbitals}")
    print(f"  electrons:              {schedule.num_electrons}")
    print(f"  complex:                {schedule.is_complex}")
    print(f"  Givens rotations:       {resources.num_givens_rotations}")
    print(f"  exp_pauli calls:        {resources.num_exp_pauli_calls}")
    print(f"  phase rotations:        {resources.num_phase_rotations}")
    print(f"  phase-aligned L2 error: {error:.3e}")
    if error > 1.0e-6:
        raise SystemExit(f"{label}: prepared state does not match the "
                         "Slater determinant")


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    rng = np.random.default_rng(11)
    real_orbitals, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    run_case("real orbital-coefficient matrix (4 spin orbitals, 2 electrons)",
             real_orbitals)

    rng = np.random.default_rng(13)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    complex_orbitals, _ = np.linalg.qr(raw)
    run_case(
        "complex orbital-coefficient matrix (5 spin orbitals, 3 electrons)",
        complex_orbitals)

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
