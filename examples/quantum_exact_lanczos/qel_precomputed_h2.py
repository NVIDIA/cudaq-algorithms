#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Quantum Exact Lanczos style workflow from a precomputed qubit Hamiltonian.

This example starts from a small H2 Jordan-Wigner Hamiltonian that was generated
outside this repository and stored as Pauli coefficients. The example is meant
to show how application code can compose the library primitives:

* PauliLCU builds a block encoding of the non-identity Pauli sum.
* qubitization observables estimate Chebyshev moments of the normalized
  Hamiltonian.
* krylov.build_chebyshev_matrices constructs the generalized eigenproblem.

The overlap-matrix conditioning below is intentionally local to the example. It
is a common numerical safeguard for small Krylov demonstrations, but it is not a
supported public Quantum Exact Lanczos API.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cudaq
from cudaq import spin
import cudaq_algorithms as algorithms
import numpy as np

DEFAULT_DATA_FILE = Path(
    __file__).resolve().parent / "data" / "h2_sto3g_jw.json"


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    word: str


@dataclass(frozen=True)
class QubitHamiltonianData:
    name: str
    source: str
    mapping: str
    num_qubits: int
    num_electrons: int
    occupied_qubits: tuple[int, ...]
    constant: float
    terms: tuple[PauliTerm, ...]


@dataclass(frozen=True)
class ConditionedEigenproblemResult:
    eigenvalues: np.ndarray
    overlap_eigenvalues: np.ndarray
    kept_rank: int
    condition_estimate: float


def load_qubit_hamiltonian(path: Path) -> QubitHamiltonianData:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    num_qubits = int(payload["num_qubits"])
    terms = tuple(
        PauliTerm(float(item["coefficient"]), str(item["pauli"]))
        for item in payload["terms"])

    for term in terms:
        if len(term.word) != num_qubits:
            raise ValueError("All Pauli words must match num_qubits.")

    return QubitHamiltonianData(
        name=str(payload["name"]),
        source=str(payload.get("source", "unknown")),
        mapping=str(payload.get("mapping", "unknown")),
        num_qubits=num_qubits,
        num_electrons=int(payload["num_electrons"]),
        occupied_qubits=tuple(int(q) for q in payload["occupied_qubits"]),
        constant=float(payload.get("constant", 0.0)),
        terms=terms,
    )


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


def spin_hamiltonian(terms: tuple[PauliTerm, ...]):
    hamiltonian = 0.0
    for term in terms:
        hamiltonian = hamiltonian + term.coefficient * spin_word(term.word)
    return hamiltonian


def pauli_sum_matrix(terms: tuple[PauliTerm, ...],
                     num_qubits: int) -> np.ndarray:
    """Build a dense matrix with the same little-endian qubit order as CUDA-Q."""

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


def exact_ground_energy(data: QubitHamiltonianData) -> float:
    shifted = pauli_sum_matrix(data.terms, data.num_qubits)
    shifted = shifted + data.constant * np.eye(shifted.shape[0],
                                               dtype=np.complex128)
    return float(np.linalg.eigvalsh(shifted).min())


def kernel_data(encoding: algorithms.PauliLCU):
    return (
        [float(value) for value in encoding.get_angles()],
        [int(value) for value in encoding.get_term_controls()],
        [int(value) for value in encoding.get_term_ops()],
        [int(value) for value in encoding.get_term_lengths()],
        [int(value) for value in encoding.get_term_signs()],
    )


def observe_expectation(kernel, observable, shots_count: int) -> float:
    if shots_count > 0:
        return float(
            cudaq.observe(shots_count, kernel, observable).expectation())
    return float(cudaq.observe(kernel, observable).expectation())


def measure_moment(encoding: algorithms.PauliLCU, occupied_qubits: tuple[int,
                                                                         ...],
                   moment_order: int, shots_count: int) -> float:
    num_ancilla = encoding.num_ancilla
    num_system = encoding.num_system
    power = moment_order // 2
    is_even = moment_order % 2 == 0
    angles, term_controls, term_ops, term_lengths, term_signs = kernel_data(
        encoding)

    if is_even:
        observable = algorithms.qubitization.build_qubitization_reflection_observable(
            num_ancilla)
    else:
        observable = algorithms.qubitization.build_lcu_select_observable(
            encoding)

    if occupied_qubits != (0, 1):
        raise ValueError("This example currently prepares the H2 Hartree-Fock "
                         "determinant with occupied qubits (0, 1).")

    @cudaq.kernel
    def moment_kernel():
        ancilla = cudaq.qvector(num_ancilla)
        system = cudaq.qvector(num_system)

        # Example-local Hartree-Fock determinant for the precomputed H2 data.
        x(system[0])
        x(system[1])

        algorithms.block_encoding.prepare(ancilla, angles)
        for _ in range(power):
            algorithms.qubitization.apply_walk(ancilla, system, angles,
                                               term_controls, term_ops,
                                               term_lengths, term_signs)
        if is_even:
            algorithms.block_encoding.unprepare(ancilla, angles)

    return observe_expectation(moment_kernel, observable, shots_count)


def collect_chebyshev_moments(encoding: algorithms.PauliLCU,
                              occupied_qubits: tuple[int, ...], dimension: int,
                              shots_count: int) -> np.ndarray:
    num_moments = algorithms.krylov.required_chebyshev_moments(dimension)
    return np.asarray([
        measure_moment(encoding, occupied_qubits, order, shots_count)
        for order in range(num_moments)
    ],
                      dtype=np.float64)


def solve_conditioned_generalized_eigenproblem(
        hamiltonian_matrix: np.ndarray, overlap_matrix: np.ndarray,
        overlap_cutoff: float) -> ConditionedEigenproblemResult:
    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap_matrix)
    keep = overlap_eigenvalues > overlap_cutoff
    if not np.any(keep):
        raise RuntimeError("Overlap matrix is numerically singular.")

    transform = (overlap_eigenvectors[:, keep] @ np.diag(
        1.0 / np.sqrt(overlap_eigenvalues[keep])))
    conditioned_hamiltonian = transform.conj(
    ).T @ hamiltonian_matrix @ transform
    eigenvalues = np.linalg.eigvalsh(conditioned_hamiltonian)
    condition_estimate = float(overlap_eigenvalues[keep].max() /
                               overlap_eigenvalues[keep].min())

    return ConditionedEigenproblemResult(
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        overlap_eigenvalues=np.asarray(overlap_eigenvalues, dtype=np.float64),
        kept_rank=int(np.count_nonzero(keep)),
        condition_estimate=condition_estimate,
    )


def run_qel_workflow(data: QubitHamiltonianData, krylov_dimension: int,
                     overlap_cutoff: float, shots_count: int):
    encoding = algorithms.PauliLCU(spin_hamiltonian(data.terms),
                                   data.num_qubits)
    moments = collect_chebyshev_moments(encoding, data.occupied_qubits,
                                        krylov_dimension, shots_count)
    matrices = algorithms.krylov.build_chebyshev_matrices(
        moments.tolist(), krylov_dimension)
    hamiltonian_matrix = np.asarray(matrices.hamiltonian_matrix(),
                                    dtype=np.float64)
    overlap_matrix = np.asarray(matrices.overlap_matrix(), dtype=np.float64)
    conditioned = solve_conditioned_generalized_eigenproblem(
        hamiltonian_matrix, overlap_matrix, overlap_cutoff)
    qel_energy = float(conditioned.eigenvalues.min() * encoding.normalization +
                       data.constant)
    exact_energy = exact_ground_energy(data)

    return {
        "encoding": encoding,
        "moments": moments,
        "hamiltonian_matrix": hamiltonian_matrix,
        "overlap_matrix": overlap_matrix,
        "conditioned": conditioned,
        "qel_energy": qel_energy,
        "exact_energy": exact_energy,
        "energy_error": abs(qel_energy - exact_energy),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--target", default="qpp-cpu")
    parser.add_argument("--krylov-dimension", type=int, default=4)
    parser.add_argument("--overlap-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=5.0e-2)
    args = parser.parse_args()

    cudaq.set_target(args.target)
    data = load_qubit_hamiltonian(args.data)
    result = run_qel_workflow(data, args.krylov_dimension, args.overlap_cutoff,
                              args.shots)

    encoding = result["encoding"]
    conditioned = result["conditioned"]

    print(f"Molecule: {data.name}")
    print(f"Mapping: {data.mapping}")
    print(f"Terms: {len(data.terms)} non-identity Pauli terms")
    print(f"LCU normalization alpha: {encoding.normalization:.12f}")
    print(f"Krylov dimension: {args.krylov_dimension}")
    print(f"Overlap kept rank: {conditioned.kept_rank}")
    print(f"Overlap condition estimate: {conditioned.condition_estimate:.6e}")
    print("Chebyshev moments:", np.array2string(result["moments"],
                                                precision=8))
    print("Overlap eigenvalues:",
          np.array2string(conditioned.overlap_eigenvalues, precision=8))
    print(f"QEL energy: {result['qel_energy']:.12f}")
    print(f"Exact energy: {result['exact_energy']:.12f}")
    print(f"Absolute error: {result['energy_error']:.6e}")

    if result["energy_error"] > args.tolerance:
        raise RuntimeError("QEL energy differs from exact diagonalization by "
                           f"more than {args.tolerance}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
