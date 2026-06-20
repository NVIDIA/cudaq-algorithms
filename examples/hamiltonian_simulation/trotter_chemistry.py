# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Suzuki-Trotter Hamiltonian simulation for a small chemistry-style Hamiltonian.

The Hamiltonian is hard-coded as Pauli terms to keep this example focused on
algorithm primitives rather than an electronic-structure package bridge.
"""

import numpy as np

import cudaq
from cudaq import spin
from cudaq_algorithms import hamiltonian_simulation


def apply_pauli_to_vector(word, vector):
    word = str(word)
    result = np.zeros_like(vector, dtype=np.complex128)
    for basis, amplitude in enumerate(vector):
        target = basis
        phase = 1.0 + 0.0j
        for qubit, op in enumerate(word):
            bit = (basis >> qubit) & 1
            if op == "I":
                continue
            if op == "X":
                target ^= 1 << qubit
            elif op == "Y":
                target ^= 1 << qubit
                phase *= -1.0j if bit else 1.0j
            elif op == "Z":
                phase *= -1.0 if bit else 1.0
            else:
                raise ValueError(op)
        result[target] += phase * amplitude
    return result


def pauli_matrix(word):
    dim = 2**len(word)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        vector = np.zeros(dim, dtype=np.complex128)
        vector[basis] = 1.0
        matrix[:, basis] = apply_pauli_to_vector(word, vector)
    return matrix


def exact_evolve(coefficients, words, identity, time, ket):
    matrix = identity * np.eye(ket.size, dtype=np.complex128)
    for coefficient, word in zip(coefficients, words):
        matrix += coefficient * pauli_matrix(str(word))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ (np.exp(-1.0j * time * eigenvalues) *
                           (eigenvectors.conj().T @ ket))


def phase_align_error(actual, expected):
    overlap = np.vdot(expected, actual)
    if abs(overlap) > 0.0:
        actual = actual * np.exp(-1.0j * np.angle(overlap))
    return np.linalg.norm(actual - expected)


def main():
    # A four-qubit molecular-style Pauli Hamiltonian. In a production chemistry
    # workflow, these terms would usually come from a fermion-to-qubit mapping.
    hamiltonian = (
        -0.81054798 * spin.i(0) + 0.17218393 * spin.z(0) -
        0.22575349 * spin.z(1) + 0.17218393 * spin.z(2) -
        0.22575349 * spin.z(3) + 0.12091263 * spin.z(0) * spin.z(1) +
        0.16892754 * spin.z(0) * spin.z(2) +
        0.16614543 * spin.z(0) * spin.z(3) +
        0.04523280 * spin.y(0) * spin.y(1) * spin.y(2) * spin.y(3) +
        0.04523280 * spin.x(0) * spin.x(1) * spin.y(2) * spin.y(3) +
        0.04523280 * spin.y(0) * spin.y(1) * spin.x(2) * spin.x(3) +
        0.04523280 * spin.x(0) * spin.x(1) * spin.x(2) * spin.x(3) +
        0.16614543 * spin.z(1) * spin.z(2) +
        0.17464343 * spin.z(1) * spin.z(3) +
        0.12091263 * spin.z(2) * spin.z(3))

    time = 0.6
    steps = 4
    order = 2
    plan = hamiltonian_simulation.make_trotter_plan(
        hamiltonian,
        time=time,
        steps=steps,
        order=order,
        ordering=hamiltonian_simulation.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING,
    )
    resources = hamiltonian_simulation.estimate_trotter_resources(plan)

    @cudaq.kernel
    def evolve(coefficients: list[float], words: list[cudaq.pauli_word],
               t: float, n_steps: int, formula_order: int):
        q = cudaq.qvector(4)
        # Prepare a small superposition so non-commuting terms have visible
        # effect in the output amplitudes.
        ry(0.31, q[0])
        rx(-0.27, q[1])
        ry(0.19, q[2])
        rx(0.23, q[3])
        hamiltonian_simulation.apply_trotter(coefficients, words, t, n_steps,
                                             formula_order, q)

    ket0 = np.asarray(
        cudaq.get_state(evolve, plan.coefficients, plan.words, 0.0, steps,
                        order),
        dtype=np.complex128)
    trotter_state = np.asarray(
        cudaq.get_state(evolve, plan.coefficients, plan.words, time, steps,
                        order),
        dtype=np.complex128)
    exact_state = exact_evolve(plan.coefficients, plan.words,
                               plan.identity_coefficient, time, ket0)

    print("Suzuki-Trotter chemistry-style example")
    print(f"num_qubits: {plan.num_qubits}")
    print(f"num_terms: {resources.num_terms}")
    print(f"identity_coefficient: {plan.identity_coefficient:.8f}")
    print(f"order: {plan.order}")
    print(f"steps: {plan.steps}")
    print(f"pauli_rotations: {resources.pauli_rotations}")
    print(f"estimated_cx_count: {resources.estimated_cx_count}")
    print(f"l2_error_vs_exact: {phase_align_error(trotter_state, exact_state):.6e}")
    print("first four amplitudes:")
    for idx, amplitude in enumerate(trotter_state[:4]):
        print(f"  |{idx:04b}> {amplitude.real:+.8f}{amplitude.imag:+.8f}j")


if __name__ == "__main__":
    main()
