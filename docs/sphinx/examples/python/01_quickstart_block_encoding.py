#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 1 — Quickstart: block-encode a Hamiltonian and walk it.

The five-minute tour of the core idea. A Hamiltonian H is not a circuit
(it is not unitary), so you *block-encode* it: PauliLCU hides H / alpha
inside a bigger unitary on a few ancilla qubits. The qubitization Walk
then turns powers of that unitary into Chebyshev polynomials of H --
measured as expectation values, the raw data spectral algorithms consume.

Everything downstream (examples 2-8) is built on these three objects:
PauliLCU (the block encoding), Walk (qubitization), and the observable
`moment` path. Here we build them and check the moments against a dense
matrix, in the house style: every claim verified against an independent
reference.

Run: python3 01_quickstart_block_encoding.py
"""

from __future__ import annotations

import os

import cudaq
import numpy as np

from cudaq_algorithms import PauliLCU, Walk

# The convention: read docs/sphinx/conventions.rst. The walk returns +<T_k(H/alpha)>.
_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def dense_hamiltonian(terms: dict) -> np.ndarray:
    """Dense matrix of a Pauli sum (little-endian: qubit 0 is the low bit,
    so word[0] is the rightmost tensor factor -- matching CUDA-Q)."""
    matrix = 0
    for word, coefficient in terms.items():
        factor = np.array([[1]], dtype=complex)
        for label in word:
            factor = np.kron(_PAULIS[label], factor)
        matrix = matrix + coefficient * factor
    return matrix


def chebyshev(x: np.ndarray, k: int) -> np.ndarray:
    """T_k(x) for a matrix argument, by the Chebyshev recurrence."""
    if k == 0:
        return np.eye(x.shape[0], dtype=complex)
    previous, current = np.eye(x.shape[0], dtype=complex), x
    for _ in range(2, k + 1):
        previous, current = current, 2 * x @ current - previous
    return current


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    # 1. Block-encode a Hamiltonian given as {pauli_word: coefficient}.
    terms = {"ZZ": 0.5, "XI": 0.3, "IX": 0.3}
    encoding = PauliLCU(terms)
    print(f"system qubits : {encoding.num_system}")
    print(f"ancillas      : {encoding.num_ancilla}")
    print(f"alpha (1-norm): {encoding.alpha:.6f}   # W encodes H / alpha")

    # 2. Build the qubitization walk over that encoding.
    walk = Walk(encoding)

    # 3. Measure Chebyshev moments <T_k(H/alpha)> from a chosen state.
    rng = np.random.default_rng(0)
    state = rng.normal(size=1 << encoding.num_system).astype(np.complex128)
    state /= np.linalg.norm(state)
    moments = walk.moments(state, 4)

    # 4. Verify against the dense matrix -- the numbers must agree.
    x = dense_hamiltonian(terms) / encoding.alpha
    print("\n k   walk.moment    dense <T_k(H/alpha)>")
    for k, measured in enumerate(moments):
        reference = float(np.real(state.conj() @ chebyshev(x, k) @ state))
        print(f" {k}   {measured:+.8f}    {reference:+.8f}")
        assert abs(measured - reference) < 1e-10

    print("\nOK — the walk reproduces Chebyshev polynomials of H to machine "
          "precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
