#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pedagogical Quantum Exact Lanczos walkthrough for H2.

This example is intentionally written as a teaching script. It follows the data
flow of a QEL calculation without introducing a public ``quantum_exact_lanczos``
API:

1. Load a precomputed qubit Hamiltonian.
2. Build a Pauli LCU block encoding.
3. Use qubitization to collect Chebyshev moments.
4. Build the Krylov Hamiltonian and overlap matrices.
5. Solve the filtered generalized eigenproblem classically.
6. Convert the scaled QEL eigenvalue back to an energy.

The overlap filtering and dense diagonalization reference are local to this
example. They are included to make the numerical workflow visible.

Reference: Kirby, Motta, and Mezzacapo, "Exact and efficient
Lanczos method on a quantum computer," arXiv:2208.00567.
"""

from __future__ import annotations

import argparse

import cudaq
import numpy as np

import quantum_exact_lanczos_molecules as qel


def section(title: str):
    """Print a visible section divider for the walkthrough output."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_terms(data: qel.QubitHamiltonianData, max_terms: int):
    """Print the Pauli terms users should recognize in the input data."""
    shown_terms = data.terms[:max_terms]
    for index, term in enumerate(shown_terms):
        print(f"  {index:2d}: {term.coefficient:+.10f} * {term.word}")
    if len(data.terms) > max_terms:
        print(f"  ... {len(data.terms) - max_terms} more terms")


def print_matrix(name: str, matrix: np.ndarray):
    """Print a named dense matrix with compact numerical formatting."""
    print(f"{name} shape: {matrix.shape}")
    print(np.array2string(matrix, precision=8, suppress_small=True))


def main() -> int:
    """Run the H2 walkthrough from Hamiltonian data to final energy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="qpp-cpu")
    parser.add_argument("--krylov-dimension", type=int)
    parser.add_argument("--overlap-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1.0e-2)
    parser.add_argument("--max-terms", type=int, default=14)
    args = parser.parse_args()

    cudaq.set_target(args.target)
    data = qel.load_qubit_hamiltonian(qel.DATA_FILES["h2"])
    krylov_dimension = (args.krylov_dimension if args.krylov_dimension
                        is not None else data.recommended_krylov_dimension)

    section("1. Load a precomputed qubit Hamiltonian")
    print(f"Molecule: {data.name}")
    print(f"Mapping: {data.mapping}")
    print(f"System qubits: {data.num_qubits}")
    print(f"Electrons: {data.num_electrons}")
    print(f"Hartree-Fock occupied qubits: {data.occupied_qubits}")
    print(f"Scalar constant: {data.constant:+.12f}")
    print(f"Non-identity Pauli terms: {len(data.terms)}")
    print_terms(data, args.max_terms)

    section("2. Build a Pauli LCU block encoding")
    hamiltonian = qel.spin_hamiltonian(data.terms)
    encoding = qel.algorithms.PauliLCU(hamiltonian, data.num_qubits)
    print("The block encoding represents H_nonidentity / alpha.")
    print(f"LCU normalization alpha: {encoding.normalization:.12f}")
    print(f"System qubits: {encoding.num_system}")
    print(f"Ancilla qubits: {encoding.num_ancilla}")
    print(
        "The scalar constant is not included in the block encoding; it is added"
    )
    print("back after solving the scaled Krylov problem.")

    section("3. Collect Chebyshev moments with qubitization")
    num_moments = qel.algorithms.krylov.required_chebyshev_moments(
        krylov_dimension)
    print(f"Krylov dimension: {krylov_dimension}")
    print(f"Required moments: {num_moments}")
    print(
        "Moment k estimates <psi|T_k(H/alpha)|psi> for the prepared HF state.")
    moments = qel.collect_chebyshev_moments(encoding, data.occupied_qubits,
                                            krylov_dimension, args.shots)
    print("Chebyshev moments:")
    print(np.array2string(moments, precision=8, suppress_small=True))

    section("4. Build the Krylov matrices")
    matrices = qel.algorithms.krylov.build_chebyshev_matrices(
        moments.tolist(), krylov_dimension)
    hamiltonian_matrix = np.asarray(matrices.hamiltonian_matrix(),
                                    dtype=np.float64)
    overlap_matrix = np.asarray(matrices.overlap_matrix(), dtype=np.float64)
    print_matrix("Krylov Hamiltonian matrix", hamiltonian_matrix)
    print_matrix("Krylov overlap matrix", overlap_matrix)

    section("5. Filter the overlap matrix and solve the eigenproblem")
    conditioned = qel.solve_conditioned_generalized_eigenproblem(
        hamiltonian_matrix, overlap_matrix, args.overlap_cutoff)
    print("Overlap eigenvalues before filtering:")
    print(
        np.array2string(conditioned.overlap_eigenvalues,
                        precision=8,
                        suppress_small=True))
    print(f"Kept Krylov rank: {conditioned.kept_rank}/{krylov_dimension}")
    print(
        f"Condition estimate after filtering: {conditioned.condition_estimate:.6e}"
    )
    print("Scaled QEL eigenvalues:")
    print(
        np.array2string(conditioned.eigenvalues,
                        precision=8,
                        suppress_small=True))

    section("6. Convert the scaled eigenvalue back to an energy")
    scaled_ground = float(conditioned.eigenvalues.min())
    qel_energy = scaled_ground * encoding.normalization + data.constant
    exact_energy = qel.exact_ground_energy(data)
    error = abs(qel_energy - exact_energy)

    print(f"Scaled ground eigenvalue: {scaled_ground:.12f}")
    print(f"Energy = scaled_eigenvalue * alpha + constant")
    print(f"QEL energy: {qel_energy:.12f}")
    print(f"Dense exact energy: {exact_energy:.12f}")
    print(f"Absolute error: {error:.6e}")

    if error > args.tolerance:
        raise RuntimeError(
            "QEL energy differs from dense exact diagonalization "
            f"by more than {args.tolerance}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
