# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared dense-reference helpers for the Python test suite."""

import numpy as np


def dense_matrix(terms, num_qubits):
    """Dense Pauli-sum matrix in CUDA-Q's little-endian qubit order."""
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for coeff, word in terms:
        for column in range(dimension):
            row = column
            phase = complex(coeff)
            for qubit, label in enumerate(word):
                bit = (column >> qubit) & 1
                if label == "X":
                    row ^= 1 << qubit
                elif label == "Y":
                    row ^= 1 << qubit
                    phase *= 1.0j if bit == 0 else -1.0j
                elif label == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
            matrix[row, column] += phase
    return matrix


def random_ket(num_qubits, seed):
    rng = np.random.default_rng(seed)
    ket = rng.normal(
        size=1 << num_qubits) + 1.0j * rng.normal(size=1 << num_qubits)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)
