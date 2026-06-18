#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Quantum Exact Lanczos style workflow from a precomputed qubit Hamiltonian.

This example starts from precomputed Jordan-Wigner Hamiltonians that were
generated outside this repository and stored as Pauli coefficients. The example
is meant to show how application code can compose the library primitives:

* PauliLCU builds a block encoding of the non-identity Pauli sum.
* qubitization observables estimate Chebyshev moments of the normalized
  Hamiltonian.
* krylov.build_chebyshev_matrices constructs the generalized eigenproblem.

The overlap-matrix conditioning below is intentionally local to the example. It
is a common numerical safeguard for small Krylov demonstrations, but it is not a
supported public Quantum Exact Lanczos API.

Reference: Kirby, Motta, and Mezzacapo, "Exact and efficient
Lanczos method on a quantum computer," arXiv:2208.00567.
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
DATA_FILES = {
    "h2": DEFAULT_DATA_FILE,
    "lih": DEFAULT_DATA_FILE.parent / "lih_sto3g_jw.json",
    "n2": DEFAULT_DATA_FILE.parent / "n2_active_space_jw.json",
    "benzene": DEFAULT_DATA_FILE.parent / "benzene_active_space_jw.json",
}
DEFAULT_EXACT_MAX_QUBITS = 8


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
    reference_energy: float | None
    reference_energy_kind: str
    recommended_krylov_dimension: int
    terms: tuple[PauliTerm, ...]


@dataclass(frozen=True)
class ConditionedEigenproblemResult:
    eigenvalues: np.ndarray
    overlap_eigenvalues: np.ndarray
    kept_rank: int
    condition_estimate: float


def load_qubit_hamiltonian(path: Path) -> QubitHamiltonianData:
    """Load a precomputed qubit Hamiltonian fixture into typed example data."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    num_qubits = int(payload["num_qubits"])
    terms = tuple(
        PauliTerm(float(item["coefficient"]), str(item["pauli"]))
        for item in payload["terms"])

    for term in terms:
        if len(term.word) != num_qubits:
            raise ValueError("All Pauli words must match num_qubits.")

    reference_energy = payload.get("reference_energy")
    reference_energy_kind = str(
        payload.get("reference_energy_kind", "reference"))
    if reference_energy is None and "fci_energy" in payload:
        reference_energy = payload["fci_energy"]
        reference_energy_kind = "FCI"
    if reference_energy is None and "casci_energy" in payload:
        reference_energy = payload["casci_energy"]
        reference_energy_kind = "CASCI"

    return QubitHamiltonianData(
        name=str(payload["name"]),
        source=str(payload.get("source", "unknown")),
        mapping=str(payload.get("mapping", "unknown")),
        num_qubits=num_qubits,
        num_electrons=int(payload["num_electrons"]),
        occupied_qubits=tuple(int(q) for q in payload["occupied_qubits"]),
        constant=float(payload.get("constant", 0.0)),
        reference_energy=(None if reference_energy is None else
                          float(reference_energy)),
        reference_energy_kind=reference_energy_kind,
        recommended_krylov_dimension=int(
            payload.get("recommended_krylov_dimension", 5)),
        terms=terms,
    )


def spin_word(word: str):
    """Convert an I/X/Y/Z Pauli word into an unscaled spin operator."""
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
    """Build the non-identity spin Hamiltonian used by PauliLCU."""
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
    """Compute the small-system dense exact ground-state energy."""
    shifted = pauli_sum_matrix(data.terms, data.num_qubits)
    shifted = shifted + data.constant * np.eye(shifted.shape[0],
                                               dtype=np.complex128)
    return float(np.linalg.eigvalsh(shifted).min())


def comparison_energy(data: QubitHamiltonianData,
                      exact_max_qubits: int) -> tuple[float | None, str]:
    """Choose dense exact diagonalization or stored reference data."""
    if data.num_qubits <= exact_max_qubits:
        return exact_ground_energy(data), "dense exact diagonalization"
    if data.reference_energy is not None:
        return data.reference_energy, data.reference_energy_kind
    return None, "none"


def kernel_data(encoding: algorithms.PauliLCU):
    """Extract simple arrays that can be captured by CUDA-Q kernels."""
    return (
        [float(value) for value in encoding.get_angles()],
        [int(value) for value in encoding.get_term_controls()],
        [int(value) for value in encoding.get_term_ops()],
        [int(value) for value in encoding.get_term_lengths()],
        [int(value) for value in encoding.get_term_signs()],
    )


def observe_expectation(kernel, observable, shots_count: int) -> float:
    """Evaluate an observable exactly or with finite shots."""
    if shots_count > 0:
        return float(
            cudaq.observe(shots_count, kernel, observable).expectation())
    return float(cudaq.observe(kernel, observable).expectation())


def measure_moment(encoding: algorithms.PauliLCU, occupied_qubits: tuple[int,
                                                                         ...],
                   moment_order: int, shots_count: int) -> float:
    """Measure one Chebyshev moment using the QEL even/odd convention.

    Even moments measure a reflected ancilla observable after unpreparing
    PREPARE. Odd moments measure the LCU SELECT observable directly.
    """
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

    if occupied_qubits == (0, 1):

        @cudaq.kernel
        def moment_kernel():
            """Prepare H2 and apply the walk circuit for one moment."""
            ancilla = cudaq.qvector(num_ancilla)
            system = cudaq.qvector(num_system)

            # Example-specific Hartree-Fock state preparation for H2.
            x(system[0])
            x(system[1])

            algorithms.block_encoding.prepare(ancilla, angles)
            for _ in range(power):
                algorithms.qubitization.apply_walk(ancilla, system, angles,
                                                   term_controls, term_ops,
                                                   term_lengths, term_signs)
            if is_even:
                algorithms.block_encoding.unprepare(ancilla, angles)

    elif occupied_qubits == (0, 1, 2, 3):

        @cudaq.kernel
        def moment_kernel():
            """Prepare the four-electron fixture and apply one moment circuit."""
            ancilla = cudaq.qvector(num_ancilla)
            system = cudaq.qvector(num_system)

            # Example-specific Hartree-Fock state preparation for the
            # included four-electron LiH, N2, and benzene active-space data.
            x(system[0])
            x(system[1])
            x(system[2])
            x(system[3])

            algorithms.block_encoding.prepare(ancilla, angles)
            for _ in range(power):
                algorithms.qubitization.apply_walk(ancilla, system, angles,
                                                   term_controls, term_ops,
                                                   term_lengths, term_signs)
            if is_even:
                algorithms.block_encoding.unprepare(ancilla, angles)

    else:
        raise ValueError("This example currently prepares Hartree-Fock states "
                         "with occupied qubits (0, 1) or (0, 1, 2, 3).")

    return observe_expectation(moment_kernel, observable, shots_count)


def collect_chebyshev_moments(encoding: algorithms.PauliLCU,
                              occupied_qubits: tuple[int, ...], dimension: int,
                              shots_count: int) -> np.ndarray:
    """Collect the moments needed to build a Chebyshev Krylov basis."""
    num_moments = algorithms.krylov.required_chebyshev_moments(dimension)
    return np.asarray([
        measure_moment(encoding, occupied_qubits, order, shots_count)
        for order in range(num_moments)
    ],
                      dtype=np.float64)


def solve_conditioned_generalized_eigenproblem(
        hamiltonian_matrix: np.ndarray, overlap_matrix: np.ndarray,
        overlap_cutoff: float) -> ConditionedEigenproblemResult:
    """Filter near-null overlap directions and solve the projected problem."""
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
                     overlap_cutoff: float, shots_count: int,
                     exact_max_qubits: int):
    """Run the example QEL workflow and return intermediate results."""
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
    reference, reference_label = comparison_energy(data, exact_max_qubits)
    energy_error = None if reference is None else abs(qel_energy - reference)

    return {
        "encoding": encoding,
        "moments": moments,
        "hamiltonian_matrix": hamiltonian_matrix,
        "overlap_matrix": overlap_matrix,
        "conditioned": conditioned,
        "qel_energy": qel_energy,
        "comparison_energy": reference,
        "comparison_label": reference_label,
        "energy_error": energy_error,
    }


def main() -> int:
    """Parse CLI options and run the selected precomputed-molecule example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule",
                        choices=sorted(DATA_FILES),
                        default="h2",
                        help="Named precomputed molecule fixture to run.")
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to a custom precomputed qubit Hamiltonian JSON file.")
    parser.add_argument("--target", default="qpp-cpu")
    parser.add_argument("--krylov-dimension", type=int)
    parser.add_argument("--overlap-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=5.0e-2)
    parser.add_argument(
        "--exact-max-qubits",
        type=int,
        default=DEFAULT_EXACT_MAX_QUBITS,
        help="Largest system size for dense exact diagonalization.")
    parser.add_argument("--describe-only",
                        action="store_true",
                        help="Print fixture metadata without running QEL.")
    args = parser.parse_args()

    data_path = args.data if args.data is not None else DATA_FILES[
        args.molecule]
    data = load_qubit_hamiltonian(data_path)
    krylov_dimension = (args.krylov_dimension if args.krylov_dimension
                        is not None else data.recommended_krylov_dimension)

    print(f"Molecule: {data.name}")
    print(f"Mapping: {data.mapping}")
    print(f"Qubits: {data.num_qubits}")
    print(f"Electrons: {data.num_electrons}")
    print(f"Terms: {len(data.terms)} non-identity Pauli terms")
    print(f"Constant term: {data.constant:.12f}")
    print(f"Recommended Krylov dimension: {data.recommended_krylov_dimension}")
    if data.reference_energy is not None:
        print(f"Stored {data.reference_energy_kind} reference energy: "
              f"{data.reference_energy:.12f}")

    if args.describe_only:
        return 0

    cudaq.set_target(args.target)
    result = run_qel_workflow(data, krylov_dimension, args.overlap_cutoff,
                              args.shots, args.exact_max_qubits)

    encoding = result["encoding"]
    conditioned = result["conditioned"]

    print(f"LCU normalization alpha: {encoding.normalization:.12f}")
    print(f"Krylov dimension: {krylov_dimension}")
    print(f"Overlap kept rank: {conditioned.kept_rank}")
    print(f"Overlap condition estimate: {conditioned.condition_estimate:.6e}")
    print("Chebyshev moments:", np.array2string(result["moments"],
                                                precision=8))
    print("Overlap eigenvalues:",
          np.array2string(conditioned.overlap_eigenvalues, precision=8))
    print(f"QEL energy: {result['qel_energy']:.12f}")

    comparison = result["comparison_energy"]
    if comparison is not None:
        print(f"Comparison energy ({result['comparison_label']}): "
              f"{comparison:.12f}")
        print(f"Absolute error: {result['energy_error']:.6e}")
        if result["energy_error"] > args.tolerance:
            raise RuntimeError(
                "QEL energy differs from the comparison reference by "
                f"more than {args.tolerance}.")
    else:
        print("Comparison energy: unavailable")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
