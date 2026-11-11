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

Two-by-two SPD data keeps table-backed workspace small. The same coefficients
also define clipped, nonperiodic banded oracles, with a different slot layout
and alpha. A four-point arithmetic case checks that boundaries do not wrap.
Every circuit is compared with independently quantized NumPy matrices.
Run: python3 sparse_block_encoding.py
Add --verbose to show individual validation checks; failures are always reported.
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


def check_encoding(name, encoding, ideal, rng, *, verbose=False):
    print(f"\n{name}: {encoding}")
    print(
        f"system={encoding.num_system}, block ancillas={encoding.num_block_ancilla}, "
        f"scratch={encoding.num_scratch}, all ancillas={encoding.num_ancilla}, "
        f"total qubits={encoding.num_system+encoding.num_ancilla}, "
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
    print(
        f"Quantization operator error={np.linalg.norm(realized - ideal, 2):.3e}; "
        f"bound={row_nnz * entry_bound:.3e}")
    print("The largest angle code is clamped; even the maximal diagonal "
          "is rounded at finite precision.")

    ket = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    ket /= np.linalg.norm(ket)
    state = np.asarray(
        cudaq.get_state(encoding.encode_kernel(), state_from(ket)))
    check(f"{name} complex action",
          sim.good_subspace(encoding, state),
          realized @ ket / encoding.alpha,
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
    for power, polynomial in ((1, -scaled),
                              (2, 2 * scaled @ scaled - np.eye(dimension))):
        output = cudaq.get_state(
            Walk(encoding).kernel(power=power), state_from(ket))
        check(f"{name} Walk power {power}",
              sim.good_subspace(encoding, output),
              polynomial @ ket,
              verbose=verbose)
    phases = [0.3, -0.4, 0.25]
    values, vectors = np.linalg.eigh(scaled)
    response = np.array([scalar_response(phases, value) for value in values])
    expected = vectors @ (response * (vectors.conj().T @ ket))
    check(f"{name} QSVT",
          sim.transform(QSVT(encoding), ket, PhaseSequence(phases)),
          expected,
          verbose=verbose)
    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each validation check (failures are always reported)")
    verbose = parser.parse_args().verbose
    started = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    rng = np.random.default_rng(7)
    ideal = np.array([[1.5, -0.25], [-0.25, 1.5]])
    data = encode_sparse(sparse.csr_matrix(ideal), value_bits=4)
    print("SciPy data constructor report:", data.report)
    required = ("dim", "nnz", "max_row_nnz", "slots", "alpha", "value_bits",
                "dilated", "qubits")
    for field in required:
        print(f"  {field}: {data.report[field]}")
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
        print(f"{label} constructor: identical report (metadata only).")
    print("Assigned matching slots need not equal the maximum row nnz.")
    check_encoding("QROM-backed 2x2", data, ideal, rng, verbose=verbose)

    arithmetic = SparseOracleEncoding(banded_oracles([-1, 0, 1],
                                                     [-.25, 1.5, -.25],
                                                     num_system=1,
                                                     value_bits=4),
                                      num_system=1)
    check_encoding("arithmetic 2x2", arithmetic, ideal, rng, verbose=verbose)
    print(f"Same ideal matrix: data alpha={data.alpha}, "
          f"arithmetic alpha={arithmetic.alpha}; raw blocks differ.")

    four = (np.diag(np.full(4, 1.5)) + np.diag(np.full(3, -.25), 1) +
            np.diag(np.full(3, -.25), -1))
    banded = SparseOracleEncoding(banded_oracles([-1, 0, 1], [-.25, 1.5, -.25],
                                                 num_system=2,
                                                 value_bits=4),
                                  num_system=2)
    block = check_encoding("arithmetic 4x4",
                           banded,
                           four,
                           rng,
                           verbose=verbose)
    check("clipped corners do not form a ring",
          block[[0, 3], [3, 0]],
          np.zeros(2),
          verbose=verbose)

    # This larger geometry is priced classically, without simulating it.
    dilated = encode_sparse(np.array([[0., -.5], [.25, 0.]]),
                            value_bits=4,
                            dilate=True)
    print("\nGeneral real matrix: Hermitian dilation metadata only:",
          dilated.report)
    print(
        "Direct encoding supports real symmetric matrices, nonnegative "
        "diagonals and signed off-diagonals. Dilation handles general real data."
    )
    print(
        "Factory non-power-of-two padding is zero, unlike example 07's "
        "dense identity padding. Reciprocal examples need a nonzero spectrum.")
    print(
        "Sparse encodings lack odd-moment SELECT observables; expanded even "
        "reflection observables are costly. Validate statevector walk blocks here."
    )
    maximum = max(e.num_system + e.num_ancilla
                  for e in (data, arithmetic, banded))
    print(f"Maximum executed qubits: {maximum}; elapsed: "
          f"{time.perf_counter()-started:.2f}s")


if __name__ == "__main__":
    main()
