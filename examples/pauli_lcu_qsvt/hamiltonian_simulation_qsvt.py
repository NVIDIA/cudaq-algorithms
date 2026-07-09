#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Hamiltonian simulation with the PauliLCU block encoding and QSVT.

Evolves a 4-qubit Pauli Hamiltonian with QSPPACK-generated phases and checks
the result against exact diagonalization.

Requires qsppack and scipy.  Run with:
    python3 hamiltonian_simulation_qsvt.py
"""

import os

import numpy as np


import cudaq

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import (PauliLCU, PhaseSequence, QSVT,
                              recover_real_time_evolution)

HAMILTONIAN = {
    "ZIII": 0.70,
    "IZII": -0.43,
    "IIZI": 0.31,
    "IIIZ": -0.22,
    "XXII": 0.19,
    "IYYI": -0.17,
    "IZZX": 0.13,
    "XYYX": 0.11,
}
TIME = 0.8
DEGREE = 16


def qsppack_phases(tau, degree):
    """cos/sin QSP phases for exp(-i tau x) via Jacobi-Anger + QSPPACK."""
    try:
        import qsppack
        from scipy import special
    except ImportError as exc:
        raise SystemExit(
            "This example needs qsppack and scipy: pip install qsppack scipy"
        ) from exc

    cos_coefficients = np.array(
        [0.5 * special.jv(0, tau)] +
        [((-1)**k) * special.jv(2 * k, tau)
         for k in range(1, degree // 2 + 1)])
    sin_coefficients = np.array(
        [((-1)**k) * special.jv(2 * k + 1, tau) for k in range(degree // 2)])
    options = {
        "criteria": 1e-12,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
    }
    cos_phases, _ = qsppack.solve(cos_coefficients, 0, {
        **options, "targetPre": True
    })
    sin_phases, _ = qsppack.solve(sin_coefficients, 1, {
        **options, "targetPre": False
    })
    return [float(p) for p in cos_phases], [float(p) for p in sin_phases]


# Kept inline so the example runs standalone; the canonical copy of
# this dense reference lives in tests/python/dense_references.py.
def dense_matrix(terms, num_qubits):
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for word, coeff in terms.items():
        for column in range(dimension):
            row, phase = column, complex(coeff)
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


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    # -- quantum workflow -------------------------------------------------
    encoding = PauliLCU(HAMILTONIAN)
    transformer = QSVT(encoding)
    tau = encoding.alpha * TIME
    cos_phases, sin_phases = qsppack_phases(tau, DEGREE)

    rng = np.random.default_rng(13)
    psi = rng.normal(size=1 << encoding.num_system).astype(np.complex128)
    psi /= np.linalg.norm(psi)

    cos_state = sim.transform(
        transformer, psi, PhaseSequence(cos_phases, convention="qsp"))
    sin_state = sim.transform(
        transformer, psi, PhaseSequence(sin_phases, convention="qsp"))
    evolved = recover_real_time_evolution(cos_state, sin_state,
                                          cos_phases, sin_phases)
    # ----------------------------------------------------------------------

    matrix = dense_matrix(HAMILTONIAN, encoding.num_system)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact = eigenvectors @ (np.exp(-1.0j * TIME * eigenvalues) *
                            (eigenvectors.conj().T @ psi))

    l2_error = float(np.linalg.norm(evolved - exact))
    fidelity = float(abs(np.vdot(exact, evolved))**2)

    print("QSVT Hamiltonian simulation (pure-Python prototype)")
    print("=" * 56)
    print(f"encoding:        {encoding}")
    print(f"evolution time:  {TIME}")
    print(f"tau = alpha*t:   {tau:.6f}")
    print(f"QSPPACK degree:  {DEGREE}")
    print(f"L2 state error:  {l2_error:.3e}")
    print(f"fidelity:        {fidelity:.12f}")

    if l2_error > 1e-10:
        raise SystemExit("evolution error exceeded 1e-10")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
