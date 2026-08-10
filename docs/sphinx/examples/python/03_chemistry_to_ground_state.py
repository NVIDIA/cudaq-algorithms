#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 3 — From a molecule to its ground-state energy.

The full classical -> quantum -> classical loop, end to end:

    PySCF mean field
      -> chemistry.from_pyscf         (chemist (pq|rs) MO integrals)
      -> chemistry.qubit_hamiltonian  (Jordan-Wigner -> SpinOperator)
      -> PauliLCU                     (block-encode H / alpha)
      -> Walk.moments                 (Chebyshev moments <T_k(H/alpha)>)
      -> classical Krylov solve       (quantum exact Lanczos)
      -> ground-state energy, checked against FCI.

This is the example to read first if you care about chemistry: it shows
how integrals become a qubit Hamiltonian, how the qubitization walk turns
into measured moments, and how those moments feed a real ground-state
algorithm. Every number is checked against an independent reference (FCI).

Reference: Kirby, Motta, Mezzacapo, "Exact and efficient Lanczos method
on a quantum computer", Quantum 7, 1018 (2023), arXiv:2208.00567.

Prerequisite: PySCF (`pip install pyscf`). Run: python3 03_chemistry_to_ground_state.py
"""

from __future__ import annotations

import os

import cudaq
import numpy as np

from cudaq_algorithms import PauliLCU, Walk, chemistry


def krylov_matrices(moments: np.ndarray, dimension: int):
    """Overlap S and scaled-Hamiltonian H~ from Chebyshev moments.

    The Krylov basis is |phi_i> = T_i(H/alpha)|ref>. With
    mu_k = <ref|T_k(H/alpha)|ref> and the Chebyshev product identities
    T_i T_j = (T_{i+j} + T_{|i-j|}) / 2 and x T_j = (T_{j+1} + T_{|j-1|}) / 2,
    both matrices are just combinations of the measured moments.
    """
    mu = moments
    overlap = np.empty((dimension, dimension))
    scaled = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            overlap[i, j] = 0.5 * (mu[i + j] + mu[abs(i - j)])
            scaled[i,
                   j] = 0.25 * (mu[i + j + 1] + mu[abs(i - j - 1)] +
                                mu[i + abs(j - 1)] + mu[abs(i - abs(j - 1))])
    return overlap, scaled


def ground_eigenvalue(scaled, overlap, cutoff=1e-8):
    """Solve the projected generalized eigenproblem, dropping null directions."""
    values, vectors = np.linalg.eigh(overlap)
    keep = values > cutoff
    transform = vectors[:, keep] / np.sqrt(values[keep])
    projected = transform.T @ scaled @ transform
    return float(np.linalg.eigvalsh(projected).min())


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    try:
        from pyscf import fci, gto, scf
    except ImportError:
        print("This example needs PySCF:  pip install pyscf")
        return 0

    # 1. Converged restricted mean field for H2 / STO-3G.
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))]
    mol = gto.M(atom=geometry, basis="sto-3g", symmetry=False)
    mean_field = scf.RHF(mol).run(verbose=0)

    # 2. Extract chemist-notation MO integrals + nuclear repulsion.
    one_body, eri, nuclear_repulsion = chemistry.from_pyscf(mean_field)
    print(f"molecule            : H2 / STO-3G  ({one_body.shape[0]} spatial "
          f"orbitals, {2 * one_body.shape[0]} qubits)")

    # 3. Jordan-Wigner qubit Hamiltonian (nuclear repulsion kept classical).
    hamiltonian = chemistry.qubit_hamiltonian(one_body, eri, scalar_offset=0.0)

    # 4. Block-encode it and build the qubitization walk.
    encoding = PauliLCU(hamiltonian)
    walk = Walk(encoding)
    print(f"block encoding      : {hamiltonian.term_count} Pauli terms, "
          f"alpha = {encoding.alpha:.6f}, {encoding.num_ancilla} ancillas")

    # 5. Measure Chebyshev moments from the Hartree-Fock reference.
    #    (HF determinant: the lowest `num_electrons` qubits occupied.)
    reference = np.zeros(1 << encoding.num_system, dtype=np.complex128)
    reference[(1 << mol.nelectron) - 1] = 1.0
    krylov_dimension = 4
    moments = np.asarray(walk.moments(reference, 2 * krylov_dimension))
    print(f"Chebyshev moments   : {np.array2string(moments, precision=4)}")

    # 6. Classical Krylov solve -> QEL energy (add nuclear repulsion back).
    overlap, scaled = krylov_matrices(moments, krylov_dimension)
    qel_energy = ground_eigenvalue(scaled, overlap) * encoding.alpha \
        + nuclear_repulsion

    # 7. Check against FCI.
    fci_energy = float(fci.FCI(mean_field).kernel()[0])
    print(f"\nQEL ground energy   : {qel_energy:.10f} Ha")
    print(f"FCI reference       : {fci_energy:.10f} Ha")
    print(f"|error|             : {abs(qel_energy - fci_energy):.2e} Ha")
    assert abs(qel_energy - fci_energy) < 1e-8
    print("\nOK — the qubitization moments reproduce the exact ground state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
