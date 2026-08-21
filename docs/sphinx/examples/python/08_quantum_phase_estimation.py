#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 8 — Quantum phase estimation with three energy oracles.

QPE turns the eigenphases of a unitary into measured bit strings. This
example estimates ground-state energies of the H2/STO-3G Hamiltonian with
one QPE shell around three different oracle constructions:

    1. exact dense-matrix time evolution  exp(-iHt)   (simulator reference)
    2. first-order product-formula evolution          (trotter.apply_trotter)
    3. an FTQC-oriented qubitization WALK              (PauliLCU)

All three use the same Hartree-Fock input, the same six-qubit counting
register, and the same hand-written inverse QFT. What changes is how the
nonunitary Hamiltonian H becomes a unitary whose phase reveals energy:
the time-evolution oracles decode linearly, E = -wrap(2*pi*y/M)/t, while
the walk decodes through the qubitization convention E = -alpha*cos(theta)
after folding the walk's conjugate mirror phases.

QPE does not choose the ground state: on a superposed input it samples
eigenvalues with the input state's spectral weights. H2 works well because
Hartree-Fock has dominant ground-state overlap. Each modal estimate is
checked to land within one phase cell of the exact ground energy.

The walk QPE core follows the centered exponent schedule of Berry et al.,
"Rapid initial-state preparation for the quantum simulation of strongly
correlated molecules", PRX Quantum 6, 020327 (2025).

Prerequisite: PySCF (`pip install pyscf`). Run: python3 08_quantum_phase_estimation.py
"""

from __future__ import annotations

import math
import os

import cudaq
import numpy as np
import scipy

from cudaq_algorithms import PauliLCU, chemistry, stateprep, trotter

# ----------------------------------------------------------------------
# Classical decoding helpers: counts -> probabilities -> modal bin -> energy
# ----------------------------------------------------------------------


def logical_integer(cudaq_bitstring: str) -> int:
    """Convert CUDA-Q's q0-first display string to logical integer ``y``."""
    return int(str(cudaq_bitstring)[::-1], 2)


def wrap_to_pi(angle: float) -> float:
    """Map an angle to the half-open interval ``[-pi, pi)``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def sampled_probabilities(counts, counting_qubits: int,
                          shots: int) -> np.ndarray:
    """Return logical-QPE-bin probabilities from CUDA-Q sample counts."""
    if counting_qubits < 1:
        raise ValueError("counting_qubits must be positive")
    if shots < 1:
        raise ValueError("shots must be positive")

    probabilities = np.zeros(1 << counting_qubits, dtype=np.float64)
    for bits, count in counts.items():
        probabilities[logical_integer(str(bits))] += int(count) / shots
    return probabilities


def decode_time_qpe(probabilities: np.ndarray,
                    evolution_time: float) -> tuple[int, float]:
    """Decode the modal bin for ``U = exp(-i H t)`` into an energy."""
    if evolution_time <= 0.0:
        raise ValueError("evolution_time must be positive")
    num_bins = len(probabilities)
    mode = int(np.argmax(probabilities))
    phase = wrap_to_pi(2.0 * math.pi * mode / num_bins)
    return mode, -phase / evolution_time


def decode_walk_qpe(probabilities: np.ndarray,
                    alpha: float) -> tuple[int, float]:
    """Fold conjugate WALK phases and decode ``E = -alpha*cos(theta)``."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    num_bins = len(probabilities)
    if num_bins % 2:
        raise ValueError("the phase-register dimension must be even")

    # Each energy appears at mirror phases +theta and -theta; add bin y
    # to bin M - y before selecting the mode.
    folded = probabilities[:num_bins // 2 + 1].copy()
    folded[1:-1] += probabilities[-1:num_bins // 2:-1]
    mode = int(np.argmax(folded))
    phase = 2.0 * math.pi * mode / num_bins
    return mode, -alpha * math.cos(phase)


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    try:
        import pyscf
        from pyscf import fci as pyscf_fci
    except ImportError:
        print("This example needs PySCF:  pip install pyscf")
        return 0

    # 1. H2 at 0.7474 Angstrom in STO-3G: two electrons in a singlet.
    #    With two spatial orbitals the spin-orbital Hamiltonian acts on
    #    four qubits -- deliberately small, so QPE can be compared with
    #    exact NumPy diagonalization.
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))]
    mol = pyscf.gto.M(atom=geometry, basis="sto-3g", spin=0, charge=0)
    mean_field = pyscf.scf.RHF(mol).run(verbose=0)
    hf_energy = float(mean_field.e_tot)
    fci_energy = float(pyscf_fci.FCI(mean_field).kernel()[0])

    one_body, eri, core = chemistry.from_pyscf(mean_field)
    spin_hamiltonian = chemistry.qubit_hamiltonian(one_body,
                                                   eri,
                                                   scalar_offset=core)
    electron_count = int(mol.nelectron)
    system_qubit_count = 2 * one_body.shape[0]
    print(f"electrons: {electron_count}")
    print(f"system qubits: {system_qubit_count}")
    print(f"RHF reference: {hf_energy:+.12f} Ha")
    print(f"FCI reference: {fci_energy:+.12f} Ha")

    # 2. Let the library organize the Pauli sum. PauliLCU retains the
    #    term pairs and computes the LCU normalization alpha = sum |a_l|
    #    (a spectral bound: ||H|| <= alpha). Trotter reuses the same
    #    pairs, orders the nonidentity terms, and reports the identity
    #    coefficient separately -- an identity contribution is a global
    #    phase for free evolution but a measurable relative phase under
    #    QPE control.
    encoding = PauliLCU(spin_hamiltonian)
    product_formula = trotter.Trotter(list(encoding.terms))

    trotter_coefficients = product_formula.coefficients
    trotter_words = [cudaq.pauli_word(word) for word in product_formula.words]
    identity_coefficient = product_formula.identity_coefficient

    hamiltonian_matrix = np.asarray(spin_hamiltonian.to_matrix(),
                                    dtype=np.complex128)
    exact_energies = np.linalg.eigvalsh(hamiltonian_matrix)
    exact_ground_energy = float(exact_energies[0])

    print(f"Pauli terms (including identity): {encoding.num_terms}")
    print(f"LCU alpha: {encoding.alpha:.12f} Ha")
    print(f"signal qubits: {encoding.num_ancilla}")
    print(f"NumPy ground energy: {exact_ground_energy:+.12f} Ha")

    # The classical references must agree before any circuit runs: the
    # variational RHF energy sits above FCI, and FCI equals the exact
    # ground energy of the qubit Hamiltonian we are about to encode.
    assert hf_energy > fci_energy
    assert abs(fci_energy - exact_ground_energy) < 1e-9

    # 3. Phase register and evolution time. M = 2^6 = 64 phase bins.
    #    t = 0.9*pi/alpha keeps |E_j t| < pi for the whole spectrum, so
    #    the linear phase-to-energy decoder cannot alias across the
    #    [-pi, pi) boundary. Eight first-order Trotter steps suffice:
    #    counting-register resolution, not Trotter error, dominates the
    #    six-bit modal estimate.
    counting_qubits = 6
    dimension = 1 << counting_qubits
    shots = 20_000

    evolution_time = 0.9 * math.pi / encoding.alpha
    n_trotter_steps = 8
    trotter_order = 1
    time_half_cell = math.pi / (evolution_time * dimension)

    print(f"phase bins: {dimension}")
    print(f"shots per method: {shots}")
    print(f"evolution time: {evolution_time:.12f} Ha^-1")
    print(f"Trotter steps/order: {n_trotter_steps}/{trotter_order}")
    print(f"time-QPE half-cell: {1000 * time_half_cell:.3f} mHa")

    # 4. Shared state preparation and inverse QFT. Every method uses the
    #    same Hartree-Fock preparation and the same hand-written IQFT
    #    (reverse the register, then Hadamards with negative controlled
    #    phases).

    @cudaq.kernel
    def prepare_hartree_fock(system: cudaq.qview):
        # H2/STO-3G has two electrons.
        stateprep.hartree_fock(system, 2)

    @cudaq.kernel
    def iqft(qubits: cudaq.qview):
        size = qubits.size()

        for index in range(size // 2):
            swap(qubits[index], qubits[size - index - 1])

        for index in range(size - 1):
            h(qubits[index])
            next_index = index + 1
            for target in range(index, -1, -1):
                r1.ctrl(-np.pi / 2**(next_index - target), qubits[next_index],
                        qubits[target])

        h(qubits[size - 1])

    # 5a. Method 1 -- exact-matrix QPE. SciPy builds the dense evolution
    #     operator U_t = expm(-iHt) directly: QPE with no Hamiltonian-
    #     simulation approximation. A useful regression reference, not a
    #     scalable algorithm (the dense 2^n x 2^n unitary is exponential).
    #     The reversed target order reconciles CUDA-Q's registered-
    #     operation target significance with the matrix ordering returned
    #     by SpinOperator.to_matrix() for this four-qubit example.
    exact_evolution_matrix = scipy.linalg.expm(-1j * hamiltonian_matrix *
                                               evolution_time)
    cudaq.register_operation("exact_matrix", exact_evolution_matrix)

    @cudaq.kernel
    def exact_time_step(system: cudaq.qview):
        exact_matrix(system[3], system[2], system[1], system[0])

    @cudaq.kernel
    def qpe_exact(n_counting_q: int, n_system_q: int):
        counting = cudaq.qvector(n_counting_q)
        system = cudaq.qvector(n_system_q)

        prepare_hartree_fock(system)
        h(counting)

        for bit in range(n_counting_q):
            for _ in range(1 << bit):
                cudaq.control(exact_time_step, [counting[bit]], system)

        iqft(counting)
        mz(counting)

    # 5b. Method 2 -- product-formula QPE. trotter.apply_trotter applies
    #     the ordered first-order Pauli product for the nonidentity part
    #     H' under coherent control. The library intentionally omits the
    #     identity term c_I (a global phase for free evolution); under
    #     QPE control it becomes observable, so each controlled base
    #     oracle also applies r1(-t*c_I) on the control's |1> branch.
    @cudaq.kernel
    def qpe_trotter(n_counting_q: int, n_system_q: int,
                    coefficients: list[float], words: list[cudaq.pauli_word],
                    t: float, steps: int, order: int, identity: float):
        counting = cudaq.qvector(n_counting_q)
        system = cudaq.qvector(n_system_q)

        prepare_hartree_fock(system)
        h(counting)

        for bit in range(n_counting_q):
            # Repetition realizes U(t)**(2**bit) using one fixed base oracle.
            for _ in range(1 << bit):
                cudaq.control(trotter.apply_trotter, counting[bit],
                              coefficients, words, t, steps, order, system)
                # Restore the identity term's controlled relative phase --
                # subtle but required: apply_trotter evolves H' only.
                r1(-t * identity, counting[bit])

        iqft(counting)
        mz(counting)

    # 5c. Method 3 -- PauliLCU / qubitization WALK QPE. The walk W maps
    #     energies to phases through cos(theta_j) = -E_j / alpha, so each
    #     energy appears at mirror phases +/- theta_j (folded in the
    #     decoder). The centered exponent schedule
    #         n - M/2 = -1 + b_0 + sum_k 2^(k-1) (2 b_k - 1)
    #     replaces powers W^n by W^(n - M/2) [PRX Quantum 6, 020327
    #     (2025)]: an unconditional W-dagger supplies the -1, bit zero
    #     supplies controlled W, and every upper bit coherently chooses
    #     W^(-2^(k-1)) for 0 or W^(+2^(k-1)) for 1.
    #
    #     The controlled-walk kernels expect one contiguous view ordered
    #     [external control, signal...]. The combined counting-and-signal
    #     allocation therefore makes the final counting qubit the fixed
    #     control port; SWAPs move each active counting-bit state into
    #     that port and back without copying or measuring it.
    prepare_signal = encoding.prepare_kernel()
    walk_dagger = encoding.adjoint_walk_step_kernel()
    controlled_walk = encoding.controlled_walk_step_kernel()
    controlled_walk_dagger = encoding.controlled_adjoint_walk_step_kernel()

    @cudaq.kernel
    def qpe_walk(n_counting_q: int, n_system_q: int, n_signal_q: int):
        counting_and_signal = cudaq.qvector(n_counting_q + n_signal_q)
        counting = counting_and_signal.front(n_counting_q)
        signal = counting_and_signal.back(n_signal_q)
        control_and_signal = counting_and_signal.back(n_signal_q + 1)
        system = cudaq.qvector(n_system_q)

        prepare_hartree_fock(system)
        h(counting)
        prepare_signal(signal)

        # Constant -1 contribution to n - M/2.
        walk_dagger(signal, system)

        # Least-significant bit contributes +b0.
        swap(counting[0], counting[n_counting_q - 1])
        controlled_walk(control_and_signal, system)
        swap(counting[0], counting[n_counting_q - 1])

        for bit in range(1, n_counting_q):
            # Move b_k into the fixed external-control port.
            if bit != n_counting_q - 1:
                swap(counting[bit], counting[n_counting_q - 1])

            for _ in range(1 << (bit - 1)):
                # b_k=0 applies W dagger; b_k=1 applies W.
                x(control_and_signal[0])
                controlled_walk_dagger(control_and_signal, system)
                x(control_and_signal[0])
                controlled_walk(control_and_signal, system)

            if bit != n_counting_q - 1:
                swap(counting[bit], counting[n_counting_q - 1])

        iqft(counting)
        mz(counting)

    # 6. Execute all three circuits: the same phase register, the same
    #    Hamiltonian, the same Hartree-Fock state, and 20,000 shots each.
    #    Fixed seeds make the outputs reproducible. Only the counting
    #    register is measured; the system (and the signal register in
    #    walk QPE) stays coherent until that measurement.
    cudaq.set_random_seed(103)
    exact_counts = cudaq.sample(qpe_exact,
                                counting_qubits,
                                system_qubit_count,
                                shots_count=shots)
    exact_sampled = sampled_probabilities(exact_counts, counting_qubits, shots)

    cudaq.set_random_seed(102)
    trotter_counts = cudaq.sample(qpe_trotter,
                                  counting_qubits,
                                  system_qubit_count,
                                  trotter_coefficients,
                                  trotter_words,
                                  evolution_time,
                                  n_trotter_steps,
                                  trotter_order,
                                  identity_coefficient,
                                  shots_count=shots)
    trotter_sampled = sampled_probabilities(trotter_counts, counting_qubits,
                                            shots)

    cudaq.set_random_seed(107)
    walk_counts = cudaq.sample(qpe_walk,
                               counting_qubits,
                               system_qubit_count,
                               encoding.num_ancilla,
                               shots_count=shots)
    walk_sampled = sampled_probabilities(walk_counts, counting_qubits, shots)

    # 7. Decode modal phase bins into energies. No continuous fit: each
    #    result is the center of one discrete phase cell.
    exact_mode, exact_modal_energy = decode_time_qpe(exact_sampled,
                                                     evolution_time)
    trotter_mode, trotter_modal_energy = decode_time_qpe(
        trotter_sampled, evolution_time)
    walk_mode, walk_modal_energy = decode_walk_qpe(walk_sampled,
                                                   encoding.alpha)

    print(f"\nexact-matrix modal bin: {exact_mode}  ->"
          f" {exact_modal_energy:+.12f} Ha")
    print(f"Trotter modal bin: {trotter_mode}  ->"
          f" {trotter_modal_energy:+.12f} Ha")
    print(f"folded WALK modal bin: {walk_mode}  ->"
          f" {walk_modal_energy:+.12f} Ha")

    # 8. Self-verification: every modal estimate must land within one
    #    phase cell of the exact ground energy. For the time oracles the
    #    cell half-width is pi/(t*M); for the walk the cell around the
    #    modal bin is asymmetric because E = -alpha*cos(theta) is
    #    nonlinear, so both edges come from the neighboring half-bins.
    for name, energy in (("exact-matrix", exact_modal_energy),
                         ("Trotter", trotter_modal_energy)):
        error = abs(energy - exact_ground_energy)
        print(f"{name} |E - E0| = {1000 * error:.3f} mHa "
              f"(half-cell {1000 * time_half_cell:.3f} mHa)")
        assert error <= time_half_cell + 1e-9

    walk_lower = -encoding.alpha * math.cos(
        2.0 * math.pi * max(0.0, walk_mode - 0.5) / dimension)
    walk_upper = -encoding.alpha * math.cos(
        2.0 * math.pi * min(dimension / 2, walk_mode + 0.5) / dimension)
    print(
        f"WALK modal cell: [{walk_lower:+.9f}, {walk_upper:+.9f}] Ha, "
        f"|E - E0| = {1000 * abs(walk_modal_energy - exact_ground_energy):.3f} mHa"
    )
    assert walk_lower - 1e-9 <= exact_ground_energy <= walk_upper + 1e-9

    print("\nOK — all three QPE oracles land within one phase cell of the "
          "exact ground energy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
