#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Walkthrough of the PauliLCU block-encoding API.

Run with:  python3 pauli_lcu_demo.py
"""

import os
import time

import numpy as np

import cudaq
from cudaq import spin

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT, Walk, state_from
from cudaq_algorithms.pauli_lcu import prepare, select, unprepare
from cudaq_algorithms.primitives import AliasSamplingPrepare


def banner(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def check(label, actual, expected, tolerance=1e-10):
    error = float(np.max(np.abs(actual - expected)))
    passed = np.isfinite(error) and error <= tolerance
    print(f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(label)


def dense_pauli(word):
    """Explicit little-endian Pauli matrix, preserving idle qubits."""
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0., 1.], [1., 0.]]),
        "Y": np.array([[0., -1j], [1j, 0.]]),
        "Z": np.diag([1., -1.])
    }
    result = np.ones((1, 1), dtype=complex)
    for letter in word:
        result = np.kron(matrices[letter], result)
    return result


def represented_hamiltonian(encoding):
    """Use public alias tables, never the PauliLCU's private LCU instance."""
    terms = encoding.terms
    if encoding.backend == "exact":
        return sum(coefficient * dense_pauli(word)
                   for coefficient, word in terms)
    sampler = AliasSamplingPrepare([abs(c) for c, _ in terms],
                                   mu=encoding.mu,
                                   variant="select")
    counts = np.zeros(sampler.num_bins, dtype=int)
    for original in range(sampler.num_bins):
        for reference in range(1 << sampler.mu):
            selected = (original if reference < sampler.keep[original] else
                        sampler.alias[original])
            counts[selected] += 1
    probabilities = counts / (sampler.num_bins * (1 << sampler.mu))
    check("alias branch counts versus public probabilities", probabilities,
          sampler.table_probabilities)
    if len(terms) < sampler.num_bins:
        check("zero-probability padding", probabilities[len(terms):],
              np.zeros(sampler.num_bins - len(terms)))
    check("per-bin alias probability bound",
          float(np.max(np.abs(probabilities - sampler.probabilities))),
          0.,
          tolerance=sampler.discretization_bound + 1e-12)
    return sum(encoding.alpha * probabilities[k] * np.sign(coefficient) *
               dense_pauli(word)
               for k, (coefficient, word) in enumerate(terms))


def scalar_response(phases, x):
    # Independent all-forward signal model; the step already includes -H/alpha.
    s = np.sqrt(max(0.0, 1 - x * x))
    step = np.array([[-x, -s], [s, -x]], dtype=complex)
    matrix = np.diag([np.exp(1j * phases[0]), 1.0])
    for phase in phases[1:]:
        matrix = np.diag([np.exp(1j * phase), 1.0]) @ step @ matrix
    return matrix[0, 0]


def resource_row(encoding, operator_error):
    apply = encoding.apply_kernel()
    n_system, n_ancilla = encoding.num_system, encoding.num_ancilla

    @cudaq.kernel
    def harness():
        system = cudaq.qvector(n_system)
        ancilla = cudaq.qvector(n_ancilla)
        apply(ancilla, system)

    resources = cudaq.estimate_resources(harness)
    rotation_gates = ("rx", "ry", "rz", "r1")
    rotations = sum(
        resources.count_controls(gate, 0) for gate in rotation_gates)
    controlled_rotations = sum(
        resources.count_controls(gate, controls) for gate in rotation_gates
        for controls in range(1, n_ancilla + 1))
    higher_x = sum(
        resources.count_controls("x", controls)
        for controls in range(3, n_ancilla + 2))
    return (f"{encoding.backend:6s} {str(encoding.mu):>4s} "
            f"{n_system:3d} {n_ancilla:3d} {n_system+n_ancilla:5d} "
            f"{encoding.alpha:6.3f} {operator_error:9.3e} "
            f"{rotations:4d} {controlled_rotations:5d} "
            f"{resources.count_controls('x', 2):4d} {higher_x:4d} "
            f"{resources.count_controls('swap', 1):7d}")


def compare_backends(exact):
    banner("8. Same Hamiltonian, exact or alias PREPARE/SELECT")
    started = time.perf_counter()
    rng = np.random.default_rng(7)
    ket = rng.normal(size=4) + 1j * rng.normal(size=4)
    ket /= np.linalg.norm(ket)
    ideal = represented_hamiltonian(exact)
    encodings = [exact] + [
        PauliLCU(dict((word, coefficient)
                      for coefficient, word in exact.terms),
                 backend="alias",
                 mu=mu) for mu in (2, 4)
    ]
    print("Only mu=2 and mu=4 are executed; mu=3 is intentionally skipped.")
    resource_rows = []
    phases = [0.3, -0.4, 0.25]
    for encoding in encodings:
        if encoding.terms != exact.terms or encoding.num_system != exact.num_system:
            raise AssertionError(
                "backends did not retain the same intended terms")
        check("one-norm agrees with exact backend", encoding.alpha,
              exact.alpha)
        represented = represented_hamiltonian(encoding)
        discretization_error = float(np.linalg.norm(represented - ideal, 2))
        if encoding.backend == "alias":
            # Each Pauli has norm one; sum the per-bin weight bounds.
            bins = 1 << max(1, (encoding.num_terms - 1).bit_length())
            bound = encoding.alpha * bins * (1.5 / (bins * (1 << encoding.mu)))
            check("operator discretization bound",
                  discretization_error,
                  0.,
                  tolerance=bound + 1e-12)
        resource_rows.append(resource_row(encoding, discretization_error))
        name = f"{encoding.backend}, mu={encoding.mu}"
        normalized = represented / encoding.alpha
        check(f"{name} action", sim.action(encoding, ket), normalized @ ket)
        for power, polynomial in ((1, -normalized),
                                  (2,
                                   2 * normalized @ normalized - np.eye(4))):
            output = cudaq.get_state(
                Walk(encoding).kernel(power=power), state_from(ket))
            check(f"{name} Walk {power}", sim.good_subspace(encoding, output),
                  polynomial @ ket)
        values, vectors = np.linalg.eigh(normalized)
        responses = np.array(
            [scalar_response(phases, value) for value in values])
        expected = vectors @ (responses * (vectors.conj().T @ ket))
        check(f"{name} QSVT",
              sim.transform(QSVT(encoding), ket, PhaseSequence(phases)),
              expected)

    print("\nResources count one encoding, without input-state preparation.")
    print(
        "backend   mu sys anc total  alpha  ||H_mu-H|| rot c-rot  CCX MCX Fredkin"
    )
    for row in resource_rows:
        print(row)

    banner("9. Odd/even moments on a separate small fixture")
    small_ket = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
    for encoding in (PauliLCU({
            "Z": 0.6,
            "X": -0.4
    }), PauliLCU({
            "Z": 0.6,
            "X": -0.4
    }, backend="alias", mu=2)):
        normalized = represented_hamiltonian(encoding) / encoding.alpha
        previous, current = np.eye(2), normalized
        for order in range(4):
            if order == 0:
                polynomial = previous
            elif order == 1:
                polynomial = current
            else:
                polynomial = 2 * normalized @ current - previous
                previous, current = current, polynomial
            expected = np.vdot(small_ket, polynomial @ small_ket).real
            check(f"{encoding.backend} measured T_{order}(+H/alpha)",
                  Walk(encoding).moment(small_ket, order), expected)
    print("Moments use +H/alpha; walk good blocks use T_k(-H/alpha).")
    print(
        "The full ancilla register includes PREPARE garbage and SELECT work.")
    print("Exact and alias share kernel factories; kernel_args is exact-only.")
    print(
        "Gate families are reported separately, without assigning a T count.")
    print(f"Maximum executed qubits: "
          f"{max(e.num_system+e.num_ancilla for e in encodings)}; "
          f"comparison elapsed: {time.perf_counter()-started:.2f}s")


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    banner("1. Construct from a plain dict (word -> coefficient)")
    enc = PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
    print(enc)
    print(f"terms:         {enc.terms}")
    print(f"alpha:         {enc.alpha}")
    print(f"constant term: {enc.constant_term}")

    banner("2. Or from a cudaq.SpinOperator — same encoding")
    h = (0.70 * spin.z(0) - 0.43 * spin.z(1) + 0.19 * spin.x(0) * spin.x(1) +
         0.11 * spin.y(0) * spin.z(1))
    print(PauliLCU(h, num_qubits=2))

    banner("3. One call: (H/alpha) |psi> via encode + postselect")
    rng = np.random.default_rng(7)
    psi = rng.normal(size=4).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    good = sim.action(enc, psi)
    print(f"|| action(psi) ||           = {np.linalg.norm(good):.6f}")
    print(f"success probability         = {np.linalg.norm(good)**2:.6f}")

    banner("4. Kernel factories for sampling/observing workflows")
    kernel = enc.encode_kernel()
    state = cudaq.get_state(kernel, state_from(psi))
    print(f"full statevector dimension  = {len(np.asarray(state))}")
    print(f"good-subspace dimension     = "
          f"{len(sim.good_subspace(enc, state))}")

    banner("5. Qubitization walks (Chebyshev moments)")
    moment_enc = PauliLCU({"I": 0.2, "X": 0.5, "Z": 0.3})
    ket = np.array([np.cos(0.35), np.sin(0.35)], dtype=np.complex128)
    for k in (1, 2, 3):
        walked = cudaq.get_state(moment_enc.walk_kernel(power=k),
                                 state_from(ket))
        p0 = float(np.sum(np.abs(sim.good_subspace(moment_enc, walked))**2))
        print(f"<T_{2*k}(H/alpha)> from the circuit = {2 * p0 - 1:+.10f}")

    banner("6. Compose the module-level kernels inside your own kernel")
    angles, controls, ops, lengths, signs = enc.kernel_args
    n_anc = enc.num_ancilla

    @cudaq.kernel
    def custom(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        prepare(ancilla, angles)
        select(ancilla, system, controls, ops, lengths, signs)
        unprepare(ancilla, angles)

    manual = sim.good_subspace(enc, cudaq.get_state(custom, state_from(psi)))
    print(f"manual composition matches action(): "
          f"{np.allclose(manual, good, atol=1e-12)}")

    banner("7. Single-term encodings (normalized to one ancilla)")
    negative_single = PauliLCU({"XZ": -0.5})
    positive_single = PauliLCU({"XZ": +0.5})
    print(negative_single)
    opposite = np.allclose(sim.action(negative_single, psi),
                           -sim.action(positive_single, psi),
                           atol=1e-12)
    print(f"-0.5*XZ encodes the opposite state of +0.5*XZ: {opposite}")
    print(f"action norm = "
          f"{np.linalg.norm(sim.action(negative_single, psi)):.6f} "
          f"(single unitary Pauli word: exactly 1)")

    compare_backends(enc)
    print()


if __name__ == "__main__":
    main()
