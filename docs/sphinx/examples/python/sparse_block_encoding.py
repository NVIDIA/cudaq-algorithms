#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Matrix data or arithmetic oracles: the same sparse BlockEncoding protocol.

A sparse matrix has only a few nonzero entries in each row. This example asks
how to give a quantum algorithm access to those entries without first
expressing the entire matrix as a sum of Pauli operations. It constructs the
same small matrix from stored matrix data and from arithmetic rules describing
its bands, showing two ways to supply the locations and values needed for a
sparse block encoding.

You should learn what the matrix-access oracles must return, how a banded
pattern handles boundaries, and why the two access methods can have different
normalization and workspace costs. Both encodings feed the same Walk and QSVT
interfaces. The checks reconstruct the matrix represented by finite-precision
value angles and compare the circuit with that matrix, so differences in data
access and numerical precision remain visible when assessing the results.

The default output shows the requested matrix, its four-bit representation,
and matrices reconstructed from both circuits. The same complex input drives
both matrix-vector comparisons. A four-point arithmetic case makes sparsity
and clipped boundaries visible. Every circuit is compared with independently
quantized NumPy matrices; simulator state access supplies the displayed data.
Run: python3 sparse_block_encoding.py
Add --verbose for individual checks and constructor/register diagnostics.
All checks run in either mode; failures are always reported.
"""

import argparse
import os
import time

import cudaq
import numpy as np
from scipy import sparse

from cudaq_algorithms import PhaseSequence, QSVT, Walk, state_from
from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms.sparse import (SparseOracleEncoding, banded_oracles,
                                     encode_sparse)

ATOL = 1e-10


def check(label, actual, expected, tolerance=ATOL, *, verbose=False):
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    passed = np.isfinite(error) and error <= tolerance
    if verbose or not passed:
        print(f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
              f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(label)


def quantized_matrix(matrix, h, bits):
    """Independent nearest-even arcsine/round/clamp/sine-squared reference."""
    result = np.zeros_like(matrix)
    for row, column in zip(*np.nonzero(matrix)):
        value = matrix[row, column]
        theta = np.arcsin(np.sqrt(abs(value) / h))
        code = min((1 << bits) - 1, round((1 << bits) * theta / (np.pi / 2)))
        result[row, column] = np.sign(value) * h * np.sin(np.pi * code /
                                                          (2 * (1 << bits)))**2
    return result


def scalar_response(phases, x):
    """Independent all-forward two-dimensional signal model."""
    step = np.array([[-x, -np.sqrt(max(0., 1 - x * x))],
                     [np.sqrt(max(0., 1 - x * x)), -x]],
                    dtype=complex)
    result = np.diag([np.exp(1j * phases[0]), 1.])
    for phase in phases[1:]:
        result = np.diag([np.exp(1j * phase), 1.]) @ step @ result
    return result[0, 0]


def show_matrix(label, matrix):
    print(label)
    displayed = np.round(np.real_if_close(matrix), 6)
    # Show structural zeros without a minus sign.
    displayed[displayed == 0] = 0.
    print(
        np.array2string(displayed,
                        precision=6,
                        suppress_small=True,
                        floatmode="fixed"))


def show_vectors(columns):
    """Side-by-side amplitudes, with roundoff hidden only for display."""
    print(" index" + "".join(f"{label:>24}" for label in columns))
    for index, values in enumerate(zip(*columns.values())):
        cells = []
        for value in values:
            real = round(float(np.real(value)), 6) + 0.
            imag = round(float(np.imag(value)), 6) + 0.
            text = (f"{real:.6f}" if imag == 0 else f"{real:.6f}{imag:+.6f}j")
            cells.append(f"{text:>24}")
        print(f"{index:6d}" + "".join(cells))


def check_encoding(name, encoding, ideal, ket, *, verbose=False):
    print(f"Running {name} circuits...", flush=True)
    if verbose:
        print(encoding)
        print(
            f"system={encoding.num_system}, block ancillas={encoding.num_block_ancilla}, "
            f"scratch={encoding.num_scratch}, all ancillas={encoding.num_ancilla}, "
            f"slots={encoding.d}/{encoding.d_padded}, h={encoding.h}, "
            f"alpha={encoding.alpha}, angle bits={encoding.value_bits}")
    realized = quantized_matrix(ideal, encoding.h, encoding.value_bits)
    dimension = len(ideal)
    columns = []
    for index in range(dimension):
        basis = np.zeros(dimension, dtype=complex)
        basis[index] = 1
        columns.append(sim.action(encoding, basis))
    block = np.column_stack(columns)
    check(f"{name} alpha*good-block",
          encoding.alpha * block,
          realized,
          verbose=verbose)
    entry_bound = encoding.h * np.pi / (2 * (1 << encoding.value_bits))
    row_nnz = int(np.count_nonzero(ideal, axis=1).max())
    check(f"{name} ideal-versus-quantized operator",
          np.linalg.norm(realized - ideal, 2),
          0.,
          row_nnz * entry_bound + ATOL,
          verbose=verbose)
    if verbose:
        print(f"Matrix-rounding error bound={row_nnz * entry_bound:.3e}")
        print("The largest angle code is clamped; even the maximal diagonal "
              "is rounded at finite precision.")
    state = np.asarray(
        cudaq.get_state(encoding.encode_kernel(), state_from(ket)))
    action = encoding.alpha * sim.good_subspace(encoding, state)
    check(f"{name} recovered matrix-vector action",
          action,
          realized @ ket,
          verbose=verbose)
    # Scratch is the high register and is zero outside oracle load windows.
    reshaped = state.reshape(
        1 << encoding.num_scratch, 1 <<
        (encoding.num_system + encoding.num_block_ancilla))
    check(f"{name} complete deterministic scratch restoration",
          reshaped[1:],
          np.zeros_like(reshaped[1:]),
          verbose=verbose)
    scaled = realized / encoding.alpha
    walk_errors = []
    for power, polynomial in ((1, -scaled),
                              (2, 2 * scaled @ scaled - np.eye(dimension))):
        output = cudaq.get_state(
            Walk(encoding).kernel(power=power), state_from(ket))
        walked = sim.good_subspace(encoding, output)
        walk_errors.append(float(np.max(np.abs(walked - polynomial @ ket))))
        check(f"{name} Walk power {power}",
              walked,
              polynomial @ ket,
              verbose=verbose)
    phases = [0.3, -0.4, 0.25]
    values, vectors = np.linalg.eigh(scaled)
    response = np.array([scalar_response(phases, value) for value in values])
    expected = vectors @ (response * (vectors.conj().T @ ket))
    transformed = sim.transform(QSVT(encoding), ket, PhaseSequence(phases))
    check(f"{name} QSVT", transformed, expected, verbose=verbose)
    return dict(block=block,
                matrix=encoding.alpha * block,
                action=action,
                represented=realized,
                rounding_error=float(np.linalg.norm(realized - ideal, 2)),
                matrix_error=float(
                    np.max(np.abs(encoding.alpha * block - realized))),
                action_error=float(np.max(np.abs(action - realized @ ket))),
                walk_error=max(walk_errors),
                qsvt_error=float(np.max(np.abs(transformed - expected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show individual checks and constructor/register diagnostics")
    verbose = parser.parse_args().verbose
    started = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    print(
        "Sparse access: turn a banded matrix into quantum matrix-vector multiplication."
    )
    four = (np.diag(np.full(4, 1.5)) + np.diag(np.full(3, -.25), 1) +
            np.diag(np.full(3, -.25), -1))
    show_matrix("A sparse 4x4 example (10 nonzeros; at most three per row):",
                four)
    print("Each row couples a site to itself and its immediate neighbours. "
          "The endpoints stop at the boundary.")
    print(
        "\n1. Compare stored entries with arithmetic rules on a small 2x2 instance."
    )
    ideal = np.array([[1.5, -0.25], [-0.25, 1.5]])
    show_matrix("Requested matrix A:", ideal)
    print(
        "QROM loads stored locations/values; arithmetic computes the three bands."
    )
    print(
        "The selected circuit block contains A~/alpha, where A~ uses four-bit angles. "
        "Multiply that block by alpha to recover the represented matrix.")
    data = encode_sparse(sparse.csr_matrix(ideal), value_bits=4)
    required = ("dim", "nnz", "max_row_nnz", "slots", "alpha", "value_bits",
                "dilated", "qubits")
    report = {field: data.report[field] for field in required}
    if verbose:
        print("SciPy data constructor report:")
        for field, value in report.items():
            print(f"  {field}: {value}")
    check("report total qubits",
          data.report["qubits"],
          data.num_system + data.num_ancilla,
          0,
          verbose=verbose)
    rows, columns = np.nonzero(ideal)
    triples = (rows.tolist(), columns.tolist(), ideal[rows, columns].tolist())
    # Equivalent constructors are shown through metadata; circuit execution
    # below covers the SciPy route, which shares their validated factory.
    for label, source, keywords in (("NumPy", ideal, {}), ("integer triples",
                                                           triples, {
                                                               "dim": 2
                                                           })):
        equivalent = encode_sparse(source, value_bits=4, **keywords)
        if equivalent.report != data.report:
            raise AssertionError(f"{label} constructor metadata differs")
        if verbose:
            print(f"{label} constructor: identical report (metadata only).")
    if verbose:
        print("Assigned matching slots need not equal the maximum row nnz.")
    ket = np.array([1., 1j]) / np.sqrt(2)
    stored = check_encoding("QROM 2x2", data, ideal, ket, verbose=verbose)

    arithmetic = SparseOracleEncoding(banded_oracles([-1, 0, 1],
                                                     [-.25, 1.5, -.25],
                                                     num_system=1,
                                                     value_bits=4),
                                      num_system=1)
    computed = check_encoding("arithmetic 2x2",
                              arithmetic,
                              ideal,
                              ket,
                              verbose=verbose)
    check("both access methods recover the same matrix",
          stored["matrix"],
          computed["matrix"],
          2 * ATOL,
          verbose=verbose)
    check("both access methods agree on the shared input",
          stored["action"],
          computed["action"],
          2 * ATOL,
          verbose=verbose)
    show_matrix("Expected four-bit matrix A~ (NumPy):", stored["represented"])
    show_matrix(
        f"Recovered from QROM: alpha * circuit block (alpha={data.alpha:g}):",
        stored["matrix"])
    show_matrix(
        f"Recovered from arithmetic: alpha * circuit block (alpha={arithmetic.alpha:g}):",
        computed["matrix"])
    print(
        f"Matrix-rounding error ||A~-A||_2 = {stored['rounding_error']:.6f}. "
        "Circuit agreement with A~ is checked separately below.")
    print(
        "\nBoth circuits use b = [1, i]/sqrt(2). Compare their matrix-vector outputs:"
    )
    show_vectors({
        "NumPy A@b": ideal @ ket,
        "NumPy A~@b": stored["represented"] @ ket,
        "QROM alpha*y": stored["action"],
        "Arithmetic alpha*y": computed["action"]
    })
    print(
        "Here y is the unnormalized component in the selected ancilla sector. "
        "Its scale differs between encodings; alpha*y recovers A~@b.")

    print("\n2. Apply the 4x4 banded matrix to an endpoint input.")
    banded = SparseOracleEncoding(banded_oracles([-1, 0, 1], [-.25, 1.5, -.25],
                                                 num_system=2,
                                                 value_bits=4),
                                  num_system=2)
    endpoint = np.array([1., 0., 0., 0.], dtype=complex)
    larger = check_encoding("arithmetic 4x4",
                            banded,
                            four,
                            endpoint,
                            verbose=verbose)
    check("clipped corners do not form a ring",
          larger["block"][[0, 3], [3, 0]],
          np.zeros(2),
          verbose=verbose)
    show_matrix("Matrix recovered from the arithmetic circuit:",
                larger["matrix"])
    print(
        "For b = [1, 0, 0, 0], only the first two output entries are nonzero:")
    show_vectors({
        "NumPy A@b": four @ endpoint,
        "NumPy A~@b": larger["represented"] @ endpoint,
        "Circuit alpha*y": larger["action"]
    })
    print("The far endpoint stays zero: the bands do not wrap into a ring.")
    print(
        f"Four-bit matrix-rounding error ||A~-A||_2 = {larger['rounding_error']:.6f}."
    )

    # This larger geometry is priced classically, without simulating it.
    dilated = encode_sparse(np.array([[0., -.5], [.25, 0.]]),
                            value_bits=4,
                            dilate=True)
    if verbose:
        print("\nGeneral real matrix: Hermitian dilation metadata only:",
              dilated.report)
        print(
            "Direct encoding supports real symmetric matrices, nonnegative "
            "diagonals and signed off-diagonals. Dilation handles general real data."
        )
        print(
            "Factory non-power-of-two padding is zero, unlike example 07's "
            "dense identity padding. Reciprocal examples need a nonzero spectrum."
        )
        print(
            "Sparse encodings lack odd-moment SELECT observables; expanded even "
            "reflection observables are costly. Validate statevector walk blocks here."
        )
    cases = (("QROM 2x2", data, stored), ("Arithmetic 2x2", arithmetic,
                                          computed), ("Arithmetic 4x4", banded,
                                                      larger))
    print("\n3. Compare circuit accuracy and cost.")
    print(
        "Maximum errors against the independent NumPy references (tolerance 1e-10):"
    )
    print(
        f"{'Method':<18} {'Matrix':>11} {'Matvec':>11} {'Walk (1,2)':>12} {'QSVT':>11}"
    )
    for name, _, result in cases:
        print(f"{name:<18} {result['matrix_error']:11.2e} "
              f"{result['action_error']:11.2e} {result['walk_error']:12.2e} "
              f"{result['qsvt_error']:11.2e}")
    print("\nThe same Walk and QSVT APIs consumed each encoding.")
    print(
        f"{'Method':<18} {'System':>7} {'Ancillas':>9} {'Total':>7} {'Padded slots':>13} {'alpha':>7}"
    )
    for name, encoding, _ in cases:
        print(f"{name:<18} {encoding.num_system:7d} {encoding.num_ancilla:9d} "
              f"{encoding.num_system+encoding.num_ancilla:7d} "
              f"{encoding.d_padded:13d} {encoding.alpha:7g}")
    print(
        "Arithmetic uses fewer qubits for this 2x2 fixture but doubles alpha. "
        "A larger alpha moves normalized eigenvalues closer to zero, "
        "which makes inverse polynomials more expensive.")
    print(
        "All checks passed, including workspace restoration and clipped boundaries."
    )
    print(
        "Displayed matrices/vectors are read from the simulator statevector.")
    maximum = max(e.num_system + e.num_ancilla
                  for e in (data, arithmetic, banded))
    print(f"Maximum executed qubits: {maximum}; elapsed: "
          f"{time.perf_counter()-started:.2f}s")


if __name__ == "__main__":
    main()
