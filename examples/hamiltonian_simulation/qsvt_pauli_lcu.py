#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Hamiltonian simulation with PauliLCU, qubitization, and QSVT.

This example evolves a 4-qubit Pauli Hamiltonian in two equivalent CUDA-Q ways:

1. Use the QSVT convenience primitive, qsvt.apply_phase_sequence().
2. Write the same phase sequence explicitly with qsvt.apply_signal_phase(),
   block_encoding.apply(), and qubitization.reflect_about_zero().

Both quantum paths use QSPPACK-generated phases and are compared with exact
NumPy diagonalization of the same Hamiltonian. The Hamiltonian and initial state
are real so the cosine and sine QSVT parity components can be recovered directly
from the simulated good-subspace statevectors.

The call to cudaq.get_state() is used only because this is a simulation example;
algorithm code intended for hardware should return observables or samples
instead of statevectors.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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


def random_real_ket(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << num_qubits).astype(np.complex128)
    return state / np.linalg.norm(state)


def exact_time_evolution(hamiltonian: np.ndarray, ket: np.ndarray,
                         time: float) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    amplitudes = eigenvectors.conj().T @ ket
    evolved = eigenvectors @ (np.exp(-1.0j * time * eigenvalues) * amplitudes)
    return evolved, eigenvalues


def qsppack_hamiltonian_simulation_phases(
        tau: float, degree: int,
        show_log: bool) -> tuple[list[float], list[float], float, float]:
    try:
        import qsppack
        from scipy import special
    except ImportError as exc:
        raise ImportError(
            "This example requires QSPPACK and SciPy for phase generation. "
            "Install qsppack and scipy, then rerun the example.") from exc

    if degree <= 0 or degree % 2 != 0:
        raise ValueError("degree must be a positive even integer")

    cos_coefficients = np.array([0.5 * special.jv(0, tau)] +
                                [((-1)**k) * special.jv(2 * k, tau)
                                 for k in range(1, degree // 2 + 1)],
                                dtype=np.float64)
    sin_coefficients = np.array([((-1)**k) * special.jv(2 * k + 1, tau)
                                 for k in range(degree // 2)],
                                dtype=np.float64)

    options = {
        "criteria": 1e-12,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
    }
    output_context = (contextlib.nullcontext() if show_log else
                      contextlib.redirect_stdout(io.StringIO()))
    with output_context:
        cos_phases, cos_info = qsppack.solve(cos_coefficients, 0, {
            **options, "targetPre": True
        })
        sin_phases, sin_info = qsppack.solve(sin_coefficients, 1, {
            **options, "targetPre": False
        })

    return ([float(phase)
             for phase in cos_phases], [float(phase) for phase in sin_phases],
            float(cos_info["value"]), float(sin_info["value"]))


def qsp_to_projector_phases(phases: list[float]) -> list[float]:
    return [2.0 * phase for phase in phases]


def kernel_data(encoding: algorithms.PauliLCU):
    return (
        [float(value) for value in encoding.get_angles()],
        [int(value) for value in encoding.get_term_controls()],
        [int(value) for value in encoding.get_term_ops()],
        [int(value) for value in encoding.get_term_lengths()],
        [int(value) for value in encoding.get_term_signs()],
    )


def good_subspace(full_state: cudaq.State, num_system: int,
                  num_signal: int) -> np.ndarray:
    state_vector = np.asarray(full_state, dtype=np.complex128)
    system_dimension = 1 << num_system
    expected_dimension = 1 << (num_system + num_signal)
    if state_vector.shape != (expected_dimension, ):
        raise RuntimeError("Unexpected statevector dimension.")

    # The system register is allocated before the signal register. With the
    # Python CUDA-Q statevector conversion, signal=|0...0> is the first system
    # block.
    return state_vector[:system_dimension].copy()


def run_with_qsvt_primitive(
    initial_state: cudaq.State, num_system: int, num_signal: int,
    sequence: algorithms.qsvt.PhaseSequence, data: tuple[list[float],
                                                         list[int], list[int],
                                                         list[int], list[int]]
) -> np.ndarray:
    angles, term_controls, term_ops, term_lengths, term_signs = data
    phases = sequence.phase_data
    walk_directions = sequence.walk_direction_data

    @cudaq.kernel
    def qsvt_kernel(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_signal)
        algorithms.qsvt.apply_phase_sequence(signal, system, phases,
                                             walk_directions, angles,
                                             term_controls, term_ops,
                                             term_lengths, term_signs)

    return good_subspace(cudaq.get_state(qsvt_kernel, initial_state),
                         num_system, num_signal)


def run_with_custom_qubitization_loop(
    initial_state: cudaq.State, num_system: int, num_signal: int,
    sequence: algorithms.qsvt.PhaseSequence, data: tuple[list[float],
                                                         list[int], list[int],
                                                         list[int], list[int]]
) -> np.ndarray:
    """Spell out the same walk convention used by qsvt.apply_phase_sequence()."""

    angles, term_controls, term_ops, term_lengths, term_signs = data
    phases = sequence.phase_data

    @cudaq.kernel
    def qsvt_kernel(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_signal)

        algorithms.qsvt.apply_signal_phase(signal, phases[0])
        for index in range(1, len(phases)):
            algorithms.block_encoding.apply(signal, system, angles,
                                            term_controls, term_ops,
                                            term_lengths, term_signs)
            algorithms.qubitization.reflect_about_zero(signal)
            algorithms.qsvt.apply_signal_phase(signal, phases[index])

    return good_subspace(cudaq.get_state(qsvt_kernel, initial_state),
                         num_system, num_signal)


def recover_time_evolved_state(cos_state: np.ndarray, sin_state: np.ndarray,
                               cos_qsp_phases: list[float],
                               sin_qsp_phases: list[float]) -> np.ndarray:
    # QSPPACK phases and qsvt.apply_signal_phase() differ by a simple global
    # phase convention. After correcting that convention, the real part of the
    # cosine sequence and the imaginary part of the sine sequence encode the two
    # parity components of exp(-i H t). The factor of two undoes the 0.5 target
    # scaling used during phase generation.
    cos_state = cos_state * np.exp(-1.0j * np.sum(cos_qsp_phases))
    sin_state = sin_state * np.exp(-1.0j * np.sum(sin_qsp_phases))
    return 2.0 * (cos_state.real + 1.0j * sin_state.imag)


def comparison_metrics(reference: np.ndarray,
                       candidate: np.ndarray) -> tuple[float, float, float]:
    l2_error = np.linalg.norm(candidate - reference)
    max_error = np.max(np.abs(candidate - reference))
    fidelity = abs(np.vdot(reference, candidate))**2
    return float(l2_error), float(max_error), float(fidelity)


def print_metrics(label: str, metrics: tuple[float, float, float]) -> None:
    l2_error, max_error, fidelity = metrics
    print(label)
    print(f"  L2 state error:      {l2_error:.6e}")
    print(f"  Max amplitude error: {max_error:.6e}")
    print(f"  Fidelity:            {fidelity:.12f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="qpp-cpu")
    parser.add_argument("--time", type=float, default=0.8)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--show-qsppack-log", action="store_true")
    args = parser.parse_args()

    cudaq.set_target(args.target)

    num_qubits = len(TERMS[0].word)
    hamiltonian = spin_hamiltonian(TERMS)
    matrix = pauli_sum_matrix(TERMS, num_qubits)
    initial_ket = random_real_ket(num_qubits, args.seed)
    exact_state, eigenvalues = exact_time_evolution(matrix, initial_ket,
                                                    args.time)

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=num_qubits)
    alpha = float(encoding.normalization)
    tau = alpha * args.time
    cos_qsp_phases, sin_qsp_phases, cos_objective, sin_objective = (
        qsppack_hamiltonian_simulation_phases(tau, args.degree,
                                              args.show_qsppack_log))

    cos_sequence = algorithms.qsvt.phase_sequence(
        qsp_to_projector_phases(cos_qsp_phases))
    sin_sequence = algorithms.qsvt.phase_sequence(
        qsp_to_projector_phases(sin_qsp_phases))

    data = kernel_data(encoding)
    initial_state = cudaq.State.from_data(initial_ket)

    primitive_cos = run_with_qsvt_primitive(initial_state, num_qubits,
                                            encoding.num_ancilla, cos_sequence,
                                            data)
    primitive_sin = run_with_qsvt_primitive(initial_state, num_qubits,
                                            encoding.num_ancilla, sin_sequence,
                                            data)
    primitive_state = recover_time_evolved_state(primitive_cos, primitive_sin,
                                                 cos_qsp_phases,
                                                 sin_qsp_phases)

    custom_cos = run_with_custom_qubitization_loop(initial_state, num_qubits,
                                                   encoding.num_ancilla,
                                                   cos_sequence, data)
    custom_sin = run_with_custom_qubitization_loop(initial_state, num_qubits,
                                                   encoding.num_ancilla,
                                                   sin_sequence, data)
    custom_state = recover_time_evolved_state(custom_cos, custom_sin,
                                              cos_qsp_phases, sin_qsp_phases)

    primitive_metrics = comparison_metrics(exact_state, primitive_state)
    custom_metrics = comparison_metrics(exact_state, custom_state)
    primitive_custom_metrics = comparison_metrics(primitive_state,
                                                  custom_state)

    print("QSVT Hamiltonian simulation with PauliLCU")
    print("=" * 56)
    print(f"CUDA-Q target:           {args.target}")
    print(f"Number of system qubits: {num_qubits}")
    print(f"Number of Pauli terms:   {len(TERMS)}")
    print(f"Number of signal qubits: {encoding.num_ancilla}")
    print(f"LCU normalization alpha: {alpha:.12f}")
    print(f"Evolution time:          {args.time:.12f}")
    print(f"Scaled time alpha * t:   {tau:.12f}")
    print(f"QSPPACK degree:          {args.degree}")
    print(f"QSPPACK objectives:      cos={cos_objective:.3e}, "
          f"sin={sin_objective:.3e}")
    print("Scaled eigenvalue range: "
          f"[{(eigenvalues / alpha).min():.12f}, "
          f"{(eigenvalues / alpha).max():.12f}]")
    print()
    print_metrics("QSVT primitive vs exact diagonalization", primitive_metrics)
    print()
    print_metrics("Custom qubitization loop vs exact diagonalization",
                  custom_metrics)
    print()
    print_metrics("QSVT primitive vs custom qubitization loop",
                  primitive_custom_metrics)

    if primitive_metrics[0] > args.tolerance:
        return 1
    if custom_metrics[0] > args.tolerance:
        return 1
    if primitive_custom_metrics[0] > args.tolerance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
