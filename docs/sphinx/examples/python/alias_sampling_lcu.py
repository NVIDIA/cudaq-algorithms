#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Coherent alias sampling: weighted indices, garbage, and a generic LCU.

Quantum models are often written as weighted sums of simple operations, and
later algorithms need a circuit representation of that sum. This example
builds one from an X gate that flips a data qubit and a signed controlled-Z
operation that changes selected amplitudes' signs. An extra index register
acts as a coherent selector, with alias sampling supplying the relative
weights. Preparing the selector, applying its chosen operations, and reversing
the preparation places the weighted sum in the part of the larger circuit
where all its extra qubits are zero.

You should learn why the sampling weights stay positive while a negative
coefficient is supplied by a phase in the selected operation, and why the
sampler's temporary information can remain entangled with the index until
preparation is reversed. The example also distinguishes scratch that must
return to zero from amplitude that legitimately remains outside the encoded
block after selection. Independent NumPy checks compare the requested weights,
their finite-precision replacements, and the resulting matrix before the same
encoding is used by Walk and QSVT.

Run with: python3 alias_sampling_lcu.py
The index has a weighted marginal, while the other registers remain entangled.
Use PREPARE/SELECT/UNPREPARE to obtain the operator's all-zero-ancilla block.
"""

from fractions import Fraction
import os
import time

import numpy as np

import cudaq
from cudaq_algorithms import LCUBlockEncoding, PhaseSequence, QSVT, Walk, state_from
from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms.primitives import AliasSamplingPrepare

ATOL = 1e-10


def bound_check(label, error, tolerance):
    passed = np.isfinite(error) and error <= tolerance
    print(f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(label)


def check(label, actual, expected):
    bound_check(label, float(np.max(np.abs(actual - expected))), ATOL)


def enumerate_tables(sampler):
    """Count classical bin/reference branches, preserving their garbage."""
    m, mu, bins = sampler.num_index, sampler.mu, sampler.num_bins
    unit = 1 << mu
    counts = [0] * bins
    expected = np.zeros(1 << (m + sampler.num_garbage), dtype=complex)
    amplitude = 1 / np.sqrt(bins * unit)
    for original in range(bins):
        for reference in range(unit):
            flag = int(reference >= sampler.keep[original])
            selected = sampler.alias[original] if flag else original
            alias_after_swap = original if flag else sampler.alias[original]
            garbage = (alias_after_swap + (sampler.keep[original] << m) +
                       (reference << (m + mu)) + (flag << (m + 2 * mu)))
            expected[selected + (garbage << m)] += amplitude
            counts[selected] += 1
    fractions = [Fraction(count, bins * unit) for count in counts]
    return np.array([float(value) for value in fractions]), fractions, expected


def preparation_circuit(sampler, *, roundtrip=False, adjoint=False):
    operation = sampler.adjoint_kernel() if adjoint else sampler.kernel()
    unprepare = sampler.adjoint_kernel()
    m, garbage_width = sampler.num_index, sampler.num_garbage

    @cudaq.kernel
    def run():
        index = cudaq.qvector(m)
        garbage = cudaq.qvector(garbage_width)
        operation(index, garbage)
        if roundtrip:
            unprepare(index, garbage)

    return run


def scalar_response(phases, x):
    """Independent two-dimensional model for all-forward projector phases."""
    s = np.sqrt(max(0.0, 1 - x * x))
    step = np.array([[-x, -s], [s, -x]], dtype=complex)
    matrix = np.diag([np.exp(1j * phases[0]), 1.0])
    for phase in phases[1:]:
        matrix = np.diag([np.exp(1j * phase), 1.0]) @ step @ matrix
    return matrix[0, 0]


def main():
    start = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    sampler = AliasSamplingPrepare([1.0, 0.0, np.sqrt(2)], mu=2)
    print(sampler)
    print("Garbage: [alias | keep | reference | flag | QROM ladder | carry].")
    state = np.asarray(cudaq.get_state(preparation_circuit(sampler)))
    table_probabilities, fractions, expected_state = enumerate_tables(sampler)
    check("complete coherent preparation, including clean ladder/carry", state,
          expected_state)
    # The index is the low register; sum over the high garbage register.
    matrix = state.reshape(1 << sampler.num_garbage, sampler.num_bins)
    marginal = np.sum(np.abs(matrix)**2, axis=0)
    check("index marginal versus integer branch counts", marginal,
          table_probabilities)
    check("public table probabilities", sampler.table_probabilities,
          table_probabilities)
    check("zero-weight and padded bins", marginal[[1, 3]], np.zeros(2))
    expected_bound = 1.5 / (sampler.num_bins * (1 << sampler.mu))
    check("public per-bin rounding bound", sampler.discretization_bound,
          expected_bound)
    bound_check(
        "ideal-versus-table probability error",
        float(np.max(np.abs(table_probabilities - sampler.probabilities))),
        expected_bound + ATOL)
    print("\nbin  keep  alias  exact table fraction  marginal   ideal")
    for k, fraction in enumerate(fractions):
        print(f"{k:3d} {sampler.keep[k]:5d} {sampler.alias[k]:6d} "
              f"{str(fraction):>20s} {marginal[k]:9.6f} "
              f"{sampler.probabilities[k]:9.6f}")
    reduced = matrix.T @ matrix.conj()
    print(
        f"Index purity: {np.trace(reduced @ reduced).real:.6f}; "
        f"probability of all-zero garbage: {np.sum(np.abs(matrix[0])**2):.6f}")
    print("A correct marginal does not imply a clean amplitude state.")

    roundtrip = np.asarray(
        cudaq.get_state(preparation_circuit(sampler, roundtrip=True)))
    zero = np.zeros_like(roundtrip)
    zero[0] = 1
    check("PREPARE/adjoint restores every qubit", roundtrip, zero)
    for adjoint in (False, True):
        resources = cudaq.estimate_resources(
            preparation_circuit(sampler, adjoint=adjoint))
        expected_ccx = sampler.qrom.toffoli_count + 2 * sampler.mu - 1
        if resources.count_controls("x", 2) != expected_ccx:
            raise AssertionError("PREPARE Toffoli price differs from compiler")
        if resources.count_controls("swap", 1) != sampler.num_index:
            raise AssertionError("PREPARE controlled-swap count differs")
        print(f"{'UNPREPARE' if adjoint else 'PREPARE'}: "
              f"{expected_ccx} CCX + {sampler.num_index} Fredkins; "
              f"{sampler.num_index + sampler.num_garbage} qubits")

    print(
        "\nClassical precision sweep; these extra circuits are not executed:")
    for mu in (1, 2, 4, 6):
        candidate = AliasSamplingPrepare([1.0, 0.0, np.sqrt(2)], mu=mu)
        error = float(
            np.max(
                np.abs(candidate.table_probabilities -
                       candidate.probabilities)))
        bound_check(f"mu={mu} probability bound", error,
                    candidate.discretization_bound + ATOL)
        print(f"  index={candidate.num_index} garbage={candidate.num_garbage} "
              f"QROM={candidate.qrom.variant}; no monotonic-error assumption")

    print("\nLCU: H = 0.6 X_0 - 0.4 CZ_01, a non-Pauli SELECT body.")
    # Positive weights; the negative sign belongs to the second SELECT body.
    bodies = [[("x", 0)],
              [("and_tt", 0, 1, 0), ("z_w", 0), ("and_tt", 0, 1, 0),
               ("sign", )]]
    encoding = LCUBlockEncoding(num_system=2,
                                weights=[0.6, 0.4],
                                bodies=bodies,
                                mu=2)
    x0 = np.kron(np.eye(2), np.array([[0., 1.], [1., 0.]]))
    negative_cz = -np.diag([1., 1., 1., -1.])
    terms = (x0, negative_cz)
    for term in terms:
        check("SELECT term is Hermitian", term, term.conj().T)
        check("SELECT term is an involution", term @ term, np.eye(4))
    ideal = 0.6 * x0 + 0.4 * negative_cz
    realized = sum(
        weight * term
        for weight, term in zip(encoding.discretized_weights, terms))
    columns = [
        sim.action(encoding, np.ascontiguousarray(column))
        for column in np.eye(4, dtype=complex).T
    ]
    check("complete LCU block versus represented operator",
          np.column_stack(columns), realized / encoding.alpha)
    check("exact one-norm is preserved", encoding.alpha, 1.0)
    weight_bound = float(
        np.sum(np.abs(encoding.discretized_weights - np.array([0.6, 0.4]))))
    bound_check("operator discretization versus weight error",
                float(np.linalg.norm(realized - ideal, 2)),
                weight_bound + ATOL)
    bound_check(
        "weight error versus per-bin bound", weight_bound, encoding.alpha *
        encoding.preparation.num_bins * encoding.discretization_bound + ATOL)

    rng = np.random.default_rng(7)
    ket = rng.normal(size=4) + 1j * rng.normal(size=4)
    ket /= np.linalg.norm(ket)
    state = np.asarray(
        cudaq.get_state(encoding.encode_kernel(), state_from(ket)))
    # SELECT work occupies the highest bits and is always restored.
    work_start = encoding.num_system + encoding.num_index + encoding.num_garbage
    check("SELECT work is clean", state[1 << work_start:],
          np.zeros_like(state[1 << work_start:]))
    print(f"Probability outside the good ancilla block: "
          f"{1 - np.sum(np.abs(state[:4])**2):.6f}")
    print(
        "UNPREPARE after SELECT generally leaves amplitude outside that block."
    )

    scaled = realized / encoding.alpha
    for power, polynomial in ((1, -scaled), (2,
                                             2 * scaled @ scaled - np.eye(4))):
        actual = sim.good_subspace(
            encoding,
            cudaq.get_state(
                Walk(encoding).kernel(power=power), state_from(ket)))
        check(f"Walk power={power}, T_k(-H_mu/alpha)", actual,
              polynomial @ ket)
    phases = [0.3, -0.4, 0.25]
    values, vectors = np.linalg.eigh(scaled)
    response = np.array([scalar_response(phases, value) for value in values])
    expected = vectors @ (response * (vectors.conj().T @ ket))
    check("QSVT versus independent scalar model",
          sim.transform(QSVT(encoding), ket, PhaseSequence(phases)), expected)
    print("This non-Pauli body has no select_observable for odd Walk.moment.")
    print(
        "Use statevector walk blocks here; even reflection observables can grow."
    )
    print(
        f"LCU: {encoding.num_system} system + {encoding.num_index} index + "
        f"{encoding.num_garbage} garbage + {encoding.num_select_work} work qubits."
    )
    print(
        "The standalone sampler can select a QROM strategy; LCU pins SELECT.")
    print(
        "Logical gate counts exclude fault-tolerant synthesis; Fredkins are separate."
    )
    maximum = max(sampler.num_index + sampler.num_garbage,
                  encoding.num_system + encoding.num_ancilla)
    print(f"Maximum executed qubits: {maximum}; "
          f"elapsed: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
