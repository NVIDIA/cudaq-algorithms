# /****************************************************************-*- Python -*-***
#  * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       *
#  * All rights reserved.                                                      *
#  *                                                                           *
#  * This source code and the accompanying materials are made available under  *
#  * the terms of the Apache License 2.0 which accompanies this distribution.  *
#  ****************************************************************************/

import numpy as np

import cudaq
import cudaq_algorithms as algorithms


def reference_slater_state(occupied_orbitals):
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=complex)
    num_orbitals, num_electrons = occupied_orbitals.shape
    state = np.zeros(2**num_orbitals, dtype=complex)

    for basis_index in range(2**num_orbitals):
        occupied = [
            orbital for orbital in range(num_orbitals)
            if (basis_index >> orbital) & 1
        ]
        if len(occupied) != num_electrons:
            continue
        state[basis_index] = np.linalg.det(
            occupied_orbitals[np.ix_(occupied, range(num_electrons))])

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


def run_real_example():
    rng = np.random.default_rng(11)
    occupied_orbitals, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    plan = algorithms.stateprep.make_slater_determinant_plan(
        occupied_orbitals)
    resources = algorithms.stateprep.estimate_givens_stateprep_resources(plan)
    num_orbitals = plan.num_orbitals
    num_electrons = plan.num_electrons

    @cudaq.kernel
    def kernel(orbital_indices: list[int], angles: list[float]):
        q = cudaq.qvector(num_orbitals)
        algorithms.stateprep.prepare_slater_determinant(
            q, orbital_indices, angles, num_electrons)

    state = np.asarray(cudaq.get_state(kernel, plan.orbital_indices,
                                       plan.angles))
    reference = reference_slater_state(occupied_orbitals)

    print("real occupied-orbital matrix")
    print(f"  orbitals: {plan.num_orbitals}")
    print(f"  electrons: {plan.num_electrons}")
    print(f"  Givens rotations: {resources.num_givens_rotations}")
    print(f"  exp_pauli calls: {resources.num_exp_pauli_calls}")
    error = phase_aligned_l2(state, reference)
    print(f"  phase-aligned L2 error: {error:.3e}")
    assert error < 1.0e-6, "prepared state does not match the Slater determinant"


def run_complex_example():
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    occupied_orbitals, _ = np.linalg.qr(raw)
    plan = algorithms.stateprep.make_slater_determinant_plan(
        occupied_orbitals)
    resources = algorithms.stateprep.estimate_givens_stateprep_resources(plan)
    num_orbitals = plan.num_orbitals
    num_electrons = plan.num_electrons

    @cudaq.kernel
    def kernel(orbital_indices: list[int], angles: list[float],
               phases: list[float], final_phases: list[float]):
        q = cudaq.qvector(num_orbitals)
        algorithms.stateprep.prepare_complex_slater_determinant(
            q, orbital_indices, angles, phases, final_phases,
            num_electrons)

    state = np.asarray(
        cudaq.get_state(kernel, plan.orbital_indices, plan.angles,
                        plan.phases, plan.final_phases))
    reference = reference_slater_state(occupied_orbitals)

    print("complex occupied-orbital matrix")
    print(f"  orbitals: {plan.num_orbitals}")
    print(f"  electrons: {plan.num_electrons}")
    print(f"  Givens rotations: {resources.num_givens_rotations}")
    print(f"  exp_pauli calls: {resources.num_exp_pauli_calls}")
    print(f"  phase rotations: {resources.num_phase_rotations}")
    error = phase_aligned_l2(state, reference)
    print(f"  phase-aligned L2 error: {error:.3e}")
    assert error < 1.0e-6, "prepared state does not match the Slater determinant"


if __name__ == "__main__":
    cudaq.set_target("qpp-cpu")
    run_real_example()
    run_complex_example()
