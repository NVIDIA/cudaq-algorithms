#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Walkthrough of the PauliLCU block-encoding API.

A Hamiltonian is a weighted sum of simple operations. To use it in quantum
algorithms, we need a larger circuit that contains a scaled copy of that sum
in the part where its extra qubits are zero. This is a block encoding. This
example follows one small Hamiltonian from its coefficients to an executable
block encoding, then uses that encoding in a quantum walk and a polynomial
transformation.

The comparison asks what changes when the same Hamiltonian uses exact state
preparation or finite-precision alias sampling. You should learn how
preparation weights, the selected operations, and the normalization determine
the encoded matrix; how extra qubits contribute to cost; and how to compare
accuracy across implementations. The NumPy checks separate errors caused by
rounded coefficients from errors in executing the circuit, so the printed
resource savings or costs can be interpreted alongside the approximation.

Run with:  python3 pauli_lcu_demo.py
Add --verbose to show individual numerical checks.
"""

import argparse
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


def check(label, actual, expected, tolerance=1e-10, *, verbose=False):
    error = float(np.max(np.abs(actual - expected)))
    passed = np.isfinite(error) and error <= tolerance
    if verbose or not passed:
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


def represented_hamiltonian(encoding, *, verbose=False):
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
    check("alias branch counts versus public probabilities",
          probabilities,
          sampler.table_probabilities,
          verbose=verbose)
    if len(terms) < sampler.num_bins:
        check("zero-probability padding",
              probabilities[len(terms):],
              np.zeros(sampler.num_bins - len(terms)),
              verbose=verbose)
    check("per-bin alias probability bound",
          float(np.max(np.abs(probabilities - sampler.probabilities))),
          0.,
          tolerance=sampler.discretization_bound + 1e-12,
          verbose=verbose)
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


def resource_row(encoding, *, verbose=False):
    """Every gate family of one encoding (apply = PREPARE, SELECT, PREPARE†)."""
    apply = encoding.apply_kernel()
    n_system, n_ancilla = encoding.num_system, encoding.num_ancilla

    @cudaq.kernel
    def harness():
        system = cudaq.qvector(n_system)
        ancilla = cudaq.qvector(n_ancilla)
        apply(ancilla, system)

    resources = cudaq.estimate_resources(harness)
    rotation_gates = ("rx", "ry", "rz", "r1")
    paulis = ("x", "y", "z")
    arities = range(n_ancilla + 2)
    rotations = sum(
        resources.count_controls(gate, 0) for gate in rotation_gates)
    controlled_rotations = sum(
        resources.count_controls(gate, controls) for gate in rotation_gates
        for controls in arities if controls >= 1)
    single_qubit = sum(
        resources.count_controls(gate, 0) for gate in paulis + ("h", "s", "t"))
    one_control = sum(resources.count_controls(gate, 1) for gate in paulis)
    two_control = sum(resources.count_controls(gate, 2) for gate in paulis)
    higher = sum(
        resources.count_controls(gate, controls) for gate in paulis
        for controls in arities if controls >= 3)
    fredkins = resources.count_controls("swap", 1)
    counted = (rotations + controlled_rotations + single_qubit + one_control +
               two_control + higher + fredkins)
    check(f"{encoding.backend} mu={encoding.mu} gate inventory is complete",
          counted,
          resources.count(),
          0,
          verbose=verbose)
    precision = "-" if encoding.mu is None else str(encoding.mu)
    return (f"{encoding.backend:6s} {precision:>4s} "
            f"{n_system:3d} {n_ancilla:3d} {n_system+n_ancilla:5d} "
            f"{rotations:4d} {controlled_rotations:5d} {single_qubit:4d} "
            f"{one_control:4d} {two_control:4d} {higher:4d} {fredkins:7d}")


def compare_backends(exact, *, verbose=False):
    banner("8. Exact versus alias: accuracy and resources")
    started = time.perf_counter()
    print(
        f"{exact.num_system} system qubits; normalization alpha = {exact.alpha:.3g}"
    )
    print("Alias sampling rounds the term weights; both backends use the same "
          "action, Walk and QSVT interfaces.")
    rng = np.random.default_rng(7)
    ket = rng.normal(size=4) + 1j * rng.normal(size=4)
    ket /= np.linalg.norm(ket)
    ideal = represented_hamiltonian(exact, verbose=verbose)
    encodings = [exact] + [
        PauliLCU(dict((word, coefficient)
                      for coefficient, word in exact.terms),
                 backend="alias",
                 mu=mu) for mu in (2, 3, 4)
    ]
    # Every encoding is priced and its represented operator compared; the
    # walk and QSVT circuits run at mu=2 and mu=3, and the 19-qubit mu=4
    # encoding runs once (its walks and QSVT cost minutes on a CPU).
    circuit_budget = 3
    resource_rows = []
    accuracy_rows = []
    operator_errors = {}
    represented_operators = {}
    phases = [0.3, -0.4, 0.25]
    for encoding in encodings:
        if encoding.terms != exact.terms or encoding.num_system != exact.num_system:
            raise AssertionError(
                "backends did not retain the same intended terms")
        check("one-norm agrees with exact backend",
              encoding.alpha,
              exact.alpha,
              verbose=verbose)
        represented = represented_hamiltonian(encoding, verbose=verbose)
        discretization_error = float(np.linalg.norm(represented - ideal, 2))
        operator_errors[encoding.mu] = discretization_error
        represented_operators[encoding.mu] = represented
        if encoding.backend == "alias":
            # Each Pauli has norm one: sum the per-bin probability bounds
            # over the 2^m bins, which gives alpha * 1.5 / 2^mu.
            bins = 1 << max(1, (encoding.num_terms - 1).bit_length())
            bound = encoding.alpha * bins * (1.5 / (bins * (1 << encoding.mu)))
            check("operator discretization bound",
                  discretization_error,
                  0.,
                  tolerance=bound + 1e-12,
                  verbose=verbose)
        resource_rows.append(resource_row(encoding, verbose=verbose))
        name = f"{encoding.backend}, mu={encoding.mu}"
        normalized = represented / encoding.alpha
        check(f"{name} action",
              sim.action(encoding, ket),
              normalized @ ket,
              verbose=verbose)
        consumers = (encoding.backend == "exact"
                     or encoding.mu <= circuit_budget)
        precision = "-" if encoding.mu is None else str(encoding.mu)
        demonstrations = "Action, Walk, QSVT" if consumers else "Action only"
        accuracy_rows.append(
            f"{encoding.backend:7s} {precision:>2s} "
            f"{discretization_error:18.5f}    {demonstrations}")
        if not consumers:
            continue
        for power, polynomial in ((1, -normalized),
                                  (2,
                                   2 * normalized @ normalized - np.eye(4))):
            output = cudaq.get_state(
                Walk(encoding).kernel(power=power), state_from(ket))
            check(f"{name} Walk {power}",
                  sim.good_subspace(encoding, output),
                  polynomial @ ket,
                  verbose=verbose)
        values, vectors = np.linalg.eigh(normalized)
        responses = np.array(
            [scalar_response(phases, value) for value in values])
        expected = vectors @ (responses * (vectors.conj().T @ ket))
        check(f"{name} QSVT",
              sim.transform(QSVT(encoding), ket, PhaseSequence(phases)),
              expected,
              verbose=verbose)

    print("\nbackend mu  Hamiltonian error    Executed demonstrations")
    for row in accuracy_rows:
        print(row)
    print("Hamiltonian error is the absolute operator norm ||H_encoded-H||_2.")
    if np.allclose(represented_operators[2],
                   represented_operators[3],
                   rtol=0,
                   atol=1e-12):
        print("mu=3 gives the same rounded Hamiltonian as mu=2.")
    if 0 < operator_errors[4] < operator_errors[2]:
        improvement = operator_errors[2] / operator_errors[4]
        print(f"On this Hamiltonian, mu=4 reduces the error by about "
              f"{improvement:.1f}x versus mu=2.")
    print("Each extra bit halves the error bound; the realized rounding error "
          "can plateau.")
    largest = encodings[-1]
    print(
        f"mu=4 uses {largest.num_system + largest.num_ancilla} total qubits; "
        "its Walk/QSVT runs are skipped to keep CPU runtime manageable.")
    print("All executed circuits agree with independent classical references "
          "within 1e-10.")
    print("\nResources count one encoding, without input-state preparation.")
    print("backend   mu sys anc total  rot c-rot   1q  c-P cc-P mc-P Fredkin")
    for row in resource_rows:
        print(row)
    print(
        "Ancillas include alias-preparation garbage and SELECT work. "
        "Gate families are counted separately, without a synthesized T cost.")

    banner("9. Odd/even moments on a separate small fixture")
    print("One-qubit Hamiltonian: H = 0.6 Z - 0.4 X; alias precision mu=2.")
    moment_values = []
    small_ket = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
    for encoding in (PauliLCU({
            "Z": 0.6,
            "X": -0.4
    }), PauliLCU({
            "Z": 0.6,
            "X": -0.4
    }, backend="alias", mu=2)):
        normalized = represented_hamiltonian(encoding,
                                             verbose=verbose) / encoding.alpha
        measured = []
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
            actual = Walk(encoding).moment(small_ket, order)
            check(f"{encoding.backend} measured T_{order}(+H/alpha)",
                  actual,
                  expected,
                  verbose=verbose)
            measured.append(actual)
        moment_values.append(measured)
    print("moment       exact    alias(mu=2)")
    for order in range(4):
        print(f"T_{order}    {moment_values[0][order]:+10.6f} "
              f"{moment_values[1][order]:+14.6f}")
    print("Moments use +H/alpha; walk good blocks use T_k(-H/alpha).")
    print("Both moment columns agree with their represented Hamiltonians "
          "within 1e-10.")
    print("Exact and alias share kernel factories; kernel_args is exact-only.")
    if verbose:
        print(f"Comparison elapsed: {time.perf_counter()-started:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="show each numerical and gate-accounting check")
    arguments = parser.parse_args()
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

    compare_backends(enc, verbose=arguments.verbose)
    print()


if __name__ == "__main__":
    main()
