#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Suzuki-Trotter simulation of a chemistry-style Hamiltonian.

The Hamiltonian is hard-coded as Pauli terms to keep the example focused
on the algorithm primitives. Two ways to run the same evolution are shown:

1. ``sim_utils.evolve(plan, ket)`` — one call, identity phase included
   (simulation helper).
2. A user kernel composing ``trotter.apply_trotter`` with state
   preparation — the hardware-shaped composition path.

Run with:  python3 trotter_chemistry.py
"""

import os

import numpy as np


import cudaq

from cudaq_algorithms import sim_utils, trotter

# A four-qubit molecular-style Pauli Hamiltonian. In a production chemistry
# workflow these terms would come from a fermion-to-qubit mapping.
HAMILTONIAN = {
    "IIII": -0.81054798,
    "ZIII": 0.17218393,
    "IZII": -0.22575349,
    "IIZI": 0.17218393,
    "IIIZ": -0.22575349,
    "ZZII": 0.12091263,
    "ZIZI": 0.16892754,
    "ZIIZ": 0.16614543,
    "YYYY": 0.04523280,
    "XXYY": 0.04523280,
    "YYXX": 0.04523280,
    "XXXX": 0.04523280,
    "IZZI": 0.16614543,
    "IZIZ": 0.17464343,
    "IIZZ": 0.12091263,
}
TIME = 0.6
STEPS = 4
ORDER = 2


@cudaq.kernel
def prepare_state(q: cudaq.qview):
    """Small product-state superposition so non-commuting terms have
    visible effect in the output amplitudes."""
    ry(0.31, q[0])
    rx(-0.27, q[1])
    ry(0.19, q[2])
    rx(0.23, q[3])


def pauli_matrix(word):
    dim = 2**len(word)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        target_col = np.zeros(dim, dtype=np.complex128)
        target_col[basis] = 1.0
        result = np.zeros(dim, dtype=np.complex128)
        for b, amplitude in enumerate(target_col):
            if amplitude == 0.0:
                continue
            target, phase = b, 1.0 + 0.0j
            for qubit, op in enumerate(word):
                bit = (b >> qubit) & 1
                if op == "X":
                    target ^= 1 << qubit
                elif op == "Y":
                    target ^= 1 << qubit
                    phase *= -1.0j if bit else 1.0j
                elif op == "Z":
                    phase *= -1.0 if bit else 1.0
            result[target] += phase * amplitude
        matrix[:, basis] = result
    return matrix


def exact_evolve(evolution, time, ket):
    matrix = evolution.identity_coefficient * np.eye(ket.size,
                                                     dtype=np.complex128)
    for coefficient, word in zip(evolution.coefficients, evolution.words):
        matrix += coefficient * pauli_matrix(str(word))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ (np.exp(-1.0j * time * eigenvalues) *
                           (eigenvectors.conj().T @ ket))


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    evolution = trotter.Trotter(
        HAMILTONIAN,
        ordering=trotter.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING)
    resources = evolution.resources(steps=STEPS, order=ORDER)

    @cudaq.kernel
    def prepare_only():
        q = cudaq.qvector(4)
        prepare_state(q)

    ket0 = np.asarray(cudaq.get_state(prepare_only), dtype=np.complex128)

    # Path 1: the one-call simulation helper (identity phase included).
    evolved = sim_utils.evolve(evolution, ket0, time=TIME, steps=STEPS,
                               order=ORDER)
    exact = exact_evolve(evolution, TIME, ket0)
    direct_error = float(np.linalg.norm(evolved - exact))

    # Path 2: the escape hatch — compose apply_trotter in a user kernel.
    coefficients = evolution.coefficients
    words = [str(w) for w in evolution.words]

    @cudaq.kernel
    def evolve_kernel(coeffs: list[float], paulis: list[cudaq.pauli_word],
                      t: float, n_steps: int, formula_order: int):
        q = cudaq.qvector(4)
        prepare_state(q)
        trotter.apply_trotter(coeffs, paulis, t, n_steps, formula_order, q)

    kernel_state = np.asarray(cudaq.get_state(evolve_kernel, coefficients,
                                              words, TIME, STEPS, ORDER),
                              dtype=np.complex128)
    # The kernel path omits the identity phase; reintroduce it for comparison.
    kernel_state = kernel_state * np.exp(
        -1.0j * evolution.identity_coefficient * TIME)
    paths_agree = float(np.linalg.norm(kernel_state - evolved))

    print("Suzuki-Trotter chemistry-style example")
    print("=" * 62)
    print(f"num_qubits:           {evolution.num_qubits}")
    print(f"num_terms:            {resources.num_terms}")
    print(f"identity_coefficient: {evolution.identity_coefficient:+.8f}")
    print(f"order:                {resources.order}")
    print(f"steps:                {resources.steps}")
    print(f"pauli_rotations:      {resources.pauli_rotations}")
    print(f"estimated_cx_count:   {resources.estimated_cx_count}")
    print(f"l2_error_vs_exact:    {direct_error:.6e}   (identity phase "
          f"included -> no phase alignment needed)")
    print(f"kernel_vs_evolve:     {paths_agree:.6e}")
    print("first four amplitudes:")
    for idx, amplitude in enumerate(evolved[:4]):
        print(f"  |{idx:04b}> {amplitude.real:+.8f}{amplitude.imag:+.8f}j")

    if direct_error > 5e-3 or paths_agree > 1e-12:
        raise SystemExit("Trotter example exceeded tolerance")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
