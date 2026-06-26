#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Measure a QSVT good-subspace probability without statevector extraction.

This example applies a degree-1 QSVT sequence to a PauliLCU block encoding and
estimates the probability that the signal register is measured in |0...0>. For
the phase sequence [0, 0], that probability is ||H|psi> / alpha||^2. This is a
hardware-shaped output pattern: the primary result is an observable expectation,
not cudaq.get_state().
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cudaq
from cudaq import spin
import cudaq_algorithms as algorithms
import numpy as np


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    word: str


TERMS = [
    PauliTerm(0.70, "ZIII"),
    PauliTerm(-0.43, "IZII"),
    PauliTerm(0.31, "IIZI"),
    PauliTerm(-0.22, "IIIZ"),
    PauliTerm(0.19, "XXII"),
    PauliTerm(-0.17, "IYYI"),
    PauliTerm(0.13, "IZZX"),
    PauliTerm(0.11, "XYYX"),
]


def spin_word(word: str):
    operator = None
    for qubit, label in enumerate(word):
        if label == "I":
            continue
        factor = {
            "X": spin.x,
            "Y": spin.y,
            "Z": spin.z,
        }[label](qubit)
        operator = factor if operator is None else operator * factor
    return 1.0 if operator is None else operator


def spin_hamiltonian(terms: list[PauliTerm]):
    hamiltonian = 0.0
    for term in terms:
        hamiltonian = hamiltonian + term.coefficient * spin_word(term.word)
    return hamiltonian


def pauli_sum_matrix(terms: list[PauliTerm], num_qubits: int) -> np.ndarray:
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    for term in terms:
        for column in range(dimension):
            row = column
            phase = 1.0 + 0.0j
            for qubit, label in enumerate(term.word):
                bit = (column >> qubit) & 1
                if label == "I":
                    continue
                if label == "X":
                    row ^= (1 << qubit)
                elif label == "Y":
                    row ^= (1 << qubit)
                    phase *= 1.0j if bit == 0 else -1.0j
                elif label == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
                else:
                    raise ValueError(f"Unsupported Pauli operator: {label}")
            matrix[row, column] += term.coefficient * phase

    return matrix


def basis_ket(num_qubits: int, occupied_qubits: tuple[int, ...]) -> np.ndarray:
    index = 0
    for qubit in occupied_qubits:
        index |= 1 << qubit
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


def observe_expectation(kernel, observable, shots_count: int) -> float:
    if shots_count > 0:
        return float(
            cudaq.observe(shots_count, kernel, observable).expectation())
    return float(cudaq.observe(kernel, observable).expectation())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="qpp-cpu")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    cudaq.set_target(args.target)

    num_qubits = len(TERMS[0].word)
    hamiltonian = spin_hamiltonian(TERMS)
    hamiltonian_matrix = pauli_sum_matrix(TERMS, num_qubits)
    initial_ket = basis_ket(num_qubits, occupied_qubits=(0, 1))

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=num_qubits)
    sequence = algorithms.qsvt.phase_sequence([0.0, 0.0])
    phases, walk_directions, angles, term_controls, term_ops, term_lengths, \
        term_signs = algorithms.qsvt.pauli_lcu_kernel_args(
            sequence, encoding.kernel_data())
    num_signal = encoding.num_ancilla

    @cudaq.kernel
    def qsvt_kernel():
        signal = cudaq.qvector(num_signal)
        system = cudaq.qvector(num_qubits)
        x(system[0])
        x(system[1])
        algorithms.qsvt.apply_phase_sequence(signal, system, phases,
                                             walk_directions, angles,
                                             term_controls, term_ops,
                                             term_lengths, term_signs)

    projector = algorithms.qubitization.build_ancilla_zero_projector(
        num_signal)
    observed_probability = observe_expectation(qsvt_kernel, projector,
                                               args.shots)
    expected_state = (
        hamiltonian_matrix @ initial_ket) / encoding.normalization
    expected_probability = float(np.vdot(expected_state, expected_state).real)
    absolute_error = abs(observed_probability - expected_probability)

    print("QSVT PauliLCU observable example")
    print("=" * 48)
    print(f"CUDA-Q target:              {args.target}")
    print(f"Shots:                      {args.shots}")
    print(f"Number of system qubits:    {num_qubits}")
    print(f"Number of signal qubits:    {num_signal}")
    print(f"LCU normalization alpha:    {encoding.normalization:.12f}")
    print(f"Observed good probability:  {observed_probability:.12f}")
    print(f"Expected good probability:  {expected_probability:.12f}")
    print(f"Absolute error:             {absolute_error:.6e}")

    if args.shots == 0 and absolute_error > args.tolerance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
