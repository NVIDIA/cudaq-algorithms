#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#                                                                              #
# [Begin Documentation]
"""Hartree-Fock + fixed-parameter UCCSD state preparation.

Prepares a 4-qubit / 2-electron UCCSD ansatz state at fixed amplitudes
(H2-style; in a real workflow the parameters would come from a classical
pre-optimization) and validates it against dense matrix exponentials of
the operator pool. Also demonstrates the open-shell Hartree-Fock
occupation, whose interleaved alpha/beta layout matches the UCCSD
excitation enumeration.

Run with:  python3 hartree_fock_ucc.py
"""

import os

import numpy as np
from scipy.linalg import expm

import cudaq

from cudaq_algorithms import stateprep

NUM_QUBITS = 4
NUM_ELECTRONS = 2
PARAMETERS = [0.1129, -0.0421, 0.2839]  # fixed UCCSD amplitudes

PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


def dense_reference(pool, parameters, ket, num_qubits):
    """prod_g exp(i theta_g G_g) |ket> from the pool's Pauli terms."""
    state = np.array(ket, dtype=np.complex128)
    for theta, op in zip(parameters, pool):
        generator = np.zeros((1 << num_qubits, 1 << num_qubits),
                             dtype=np.complex128)
        for term in cudaq.SpinOperator(op):
            word = str(term.get_pauli_word(num_qubits))
            matrix = np.array([[1.0]], dtype=np.complex128)
            # Qubit 0 is the least-significant statevector index.
            for label in reversed(word):
                matrix = np.kron(matrix, PAULI_MATRICES[label])
            generator += term.evaluate_coefficient().real * matrix
        state = expm(1.0j * theta * generator) @ state
    return state


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    pool = stateprep.make_uccsd_operator_pool(NUM_QUBITS, NUM_ELECTRONS)
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(
        pool, NUM_QUBITS)
    resources = stateprep.estimate_fixed_parameter_ucc_resources(
        NUM_QUBITS, words)

    prep = stateprep.hartree_fock_ucc_kernel(NUM_QUBITS,
                                             PARAMETERS,
                                             words,
                                             coeffs,
                                             num_electrons=NUM_ELECTRONS)

    @cudaq.kernel
    def entry():
        qubits = cudaq.qvector(NUM_QUBITS)
        prep(qubits)

    state = np.array(cudaq.get_state(entry))

    hf_ket = np.zeros(1 << NUM_QUBITS, dtype=np.complex128)
    hf_ket[0b0011] = 1.0
    expected = dense_reference(pool, PARAMETERS, hf_ket, NUM_QUBITS)
    error = float(np.linalg.norm(state - expected))

    print("Fixed-parameter UCCSD state preparation")
    print("=" * 64)
    print(f"qubits:                {NUM_QUBITS}")
    print(f"electrons:             {NUM_ELECTRONS}")
    print(f"excitation groups:     {resources.num_excitations}")
    print(f"pauli rotations:       {resources.num_pauli_rotations}")
    print(f"max rotations/group:   "
          f"{resources.max_pauli_rotations_per_excitation}")
    print(f"L2 error vs dense:     {error:.3e}")
    print("dominant amplitudes:")
    for index in np.argsort(np.abs(state))[::-1][:4]:
        print(f"  |{index:04b}> {state[index].real:+.6f}"
              f"{state[index].imag:+.6f}j "
              f"(P = {abs(state[index])**2:.4f})")

    print()
    print("Open-shell Hartree-Fock (8 qubits, 4 electrons, spin 2):")
    occupation = stateprep.make_hartree_fock_occupation(8, 4, 2)
    print(f"  occupied spin orbitals: {occupation}  (interleaved "
          "alpha/beta, NOT the contiguous {0, 1, 2, 3})")

    if error > 1e-6:
        raise SystemExit("UCC state does not match the dense reference")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
