# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A user-written block encoding, head to head with the built-in PauliLCU.

The point of this example pair: ``DoubleFactorizedEncoding`` is *not* part
of the package — it lives next door in ``df_encoding.py``, implemented
against the public ``BlockEncoding`` protocol only. Because the protocol is
structural, the same ``Walk`` consumer drives the user-written encoding and
the built-in ``PauliLCU`` identically.

Builds a random two-orbital electronic-structure system (symmetric one-body
matrix, positive-semidefinite chemist ERI), block-encodes it two ways —

  * ``DoubleFactorizedEncoding`` (from ``df_encoding.py`` beside this
    script): frames of ancilla-controlled Z words conjugated by
    uncontrolled Givens networks (von Burg construction),
  * ``PauliLCU`` of the same Hamiltonian's flat Pauli expansion —

and compares the normalization ``alpha``, term counts, and circuit
structure. Both are ``BlockEncoding``s, so the same ``Walk`` consumer
measures Chebyshev moments through either. Truncating the factorization
shrinks the DF encoding further; the flat expansion has no such knob.

Runs on the CPU statevector simulator; no compiled extension needed.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

import cudaq

from cudaq_algorithms import PauliLCU, Walk
from cudaq_algorithms import double_factorization as df

# The encoding under demonstration is a sibling example file, not a package
# module (CUDA-Q kernels need real .py files, so a plain import is fine).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from df_encoding import DoubleFactorizedEncoding

cudaq.set_target("qpp-cpu")

# ----------------------------------------------------------------------
# A small electronic-structure system (dense JW reference in NumPy)
# ----------------------------------------------------------------------

rng = np.random.default_rng(7)
N_ORBITALS = 2
one_body = rng.normal(size=(N_ORBITALS, N_ORBITALS))
one_body = 0.5 * (one_body + one_body.T)
eri = np.zeros((N_ORBITALS, ) * 4)
for _ in range(2 * N_ORBITALS):
    s = rng.normal(size=(N_ORBITALS, N_ORBITALS))
    s = 0.5 * (s + s.T)
    eri += float(rng.uniform(0.1, 1.0)) * np.einsum('pq,rs->pqrs', s, s)

NUM_QUBITS = 2 * N_ORBITALS
DIM = 1 << NUM_QUBITS

_I2, _Z2 = np.eye(2), np.diag([1.0, -1.0])
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])


def annihilator(mode: int) -> np.ndarray:
    ops = ([_Z2] * mode + [_LOWER] + [_I2] * (NUM_QUBITS - mode - 1))[::-1]
    out = np.array([[1.0]])
    for op in ops:
        out = np.kron(out, op)
    return out


lower = [annihilator(j) for j in range(NUM_QUBITS)]
raise_ = [a.conj().T for a in lower]


def excite(p, q):
    return (raise_[2 * p] @ lower[2 * q] +
            raise_[2 * p + 1] @ lower[2 * q + 1])


hamiltonian = np.zeros((DIM, DIM), dtype=complex)
for p, q in itertools.product(range(N_ORBITALS), repeat=2):
    hamiltonian += one_body[p, q] * excite(p, q)
    for r, s in itertools.product(range(N_ORBITALS), repeat=2):
        hamiltonian += 0.5 * eri[p, q, r, s] * (excite(p, q) @ excite(r, s) -
                                                (q == r) * excite(p, s))

# Flat Pauli expansion of the same Hamiltonian, for the PauliLCU baseline.
_PAULI = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": _Z2
}
pauli_terms = {}
for word in itertools.product("IXYZ", repeat=NUM_QUBITS):
    matrix = np.array([[1.0]])
    for ch in reversed(word):  # qubit 0 least significant
        matrix = np.kron(matrix, _PAULI[ch])
    coefficient = float(np.real(np.trace(matrix @ hamiltonian))) / DIM
    if abs(coefficient) > 1e-12:
        pauli_terms["".join(word)] = coefficient

# ----------------------------------------------------------------------
# The two encodings, and truncated variants
# ----------------------------------------------------------------------

flat = PauliLCU(pauli_terms)
factorized = DoubleFactorizedEncoding(one_body, eri)

print(f"System: {N_ORBITALS} spatial orbitals -> {NUM_QUBITS} qubits")
print(f"  PauliLCU:                 alpha = {flat.alpha:.4f}, "
      f"{flat.num_terms} Pauli terms")
print(f"  DoubleFactorizedEncoding: alpha = {factorized.alpha:.4f}, "
      f"{factorized.num_terms} Z-word terms in {factorized.num_frames} "
      f"frames, {factorized.num_givens_rotations} Givens rotations")

print("\nTruncating the factorization (the knob PauliLCU does not have):")
for leaves in range(1, factorized.factorization.num_leaves + 1):
    truncated = df.explicit_double_factorization(eri, max_num_leaves=leaves)
    enc = DoubleFactorizedEncoding(one_body, truncated)
    error = df.factorization_error(eri, truncated)
    print(f"  {leaves} leaves: alpha = {enc.alpha:.4f}, "
          f"{enc.num_terms} terms, {enc.num_frames} frames, "
          f"tensor error {error:.2e}")

# ----------------------------------------------------------------------
# Same consumer, either encoding: Chebyshev moments via Walk
# ----------------------------------------------------------------------

ket = rng.normal(size=DIM) + 1.0j * rng.normal(size=DIM)
ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

print("\nEven Chebyshev moments <T_k(H/alpha)> through Walk (same consumer,"
      "\ndifferent alpha, so values differ; both descend from the same H):")
for order in (0, 2, 4):
    scaled_flat = hamiltonian / flat.alpha
    scaled_df = hamiltonian / factorized.alpha

    def chebyshev(matrix, k):
        t_prev, t_cur = np.eye(DIM, dtype=complex), matrix.copy()
        if k == 0:
            return t_prev
        for _ in range(k - 1):
            t_prev, t_cur = t_cur, 2.0 * matrix @ t_cur - t_prev
        return t_cur

    measured_flat = Walk(flat).moment(ket, order)
    measured_df = Walk(factorized).moment(ket, order)
    exact_flat = float(
        np.real(ket.conj() @ chebyshev(scaled_flat, order) @ ket))
    exact_df = float(np.real(ket.conj() @ chebyshev(scaled_df, order) @ ket))
    print(f"  T_{order}:  PauliLCU {measured_flat:+.8f} (exact "
          f"{exact_flat:+.8f})   DF {measured_df:+.8f} (exact "
          f"{exact_df:+.8f})")
