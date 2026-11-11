#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""One backward-Euler heat step in a rod, using a sparse QSVT inverse.

Start with a temperature pattern along a rod and ask what it becomes after one
step of heat diffusion with fixed endpoint temperatures. A finite-difference
model turns this physical question into a banded linear system. This example
builds that system, solves it with NumPy, and uses a sparse quantum block
encoding and an inverse polynomial to compute the same discrete update,
connecting the earlier matrix tools to a recognizable application.

You should learn how the physical assumptions determine the matrix and its
boundary entries, how a sparse inverse produces the updated temperature
vector, and how to restore the output's physical scale. The per-position table
compares the ideal NumPy solution, a NumPy solution with the circuit's rounded
matrix coefficients, and the quantum result. Those comparisons show how matrix
precision and inverse approximation contribute separately to error; they
validate the chosen discrete problem, while accuracy relative to the
continuous heat equation requires further space and time refinement.

Eliminating fixed zero endpoints gives A=tridiag(-r,1+2r,-r), where
r=D*dt/dx**2. N interior positions have dx=1/(N+1). Temperatures are above
ambient; D and dt are not assigned physical units here. NumPy solves the
same discrete system independently of the sparse quantum circuit.

The QSP real-response extraction uses full-state simulator postprocessing;
the raw good-block norm is not the success probability of an isolated
inverse filter. It does not demonstrate a coherent readout of every node.
Run: python3 10_sparse_heat_equation.py
The default (four angle bits, 1e-2 inverse target, 16 qubits) is sized for
the CPU simulator; --value-bits 6 --inverse-epsilon 1e-3 is the 18-qubit
precision configuration, practical on a GPU. Optional: --plot heat_step.png
Add --verbose to show individual validation checks; failures are always reported.
"""

import argparse
import contextlib
import io
import os
from pathlib import Path
import time

import cudaq
import numpy as np
from numpy.polynomial import chebyshev
from scipy.stats import binom

from cudaq_algorithms import PhaseSequence, QSVT
from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms.sparse import SparseOracleEncoding, banded_oracles

BLOCK_ATOL = 1e-10
CIRCUIT_ATOL = 1e-8
PHASE_ATOL = 2e-8
MAX_QUBITS = 20
MAX_DEGREE = 511


def check(label, error, tolerance, *, verbose=False):
    passed = np.isfinite(error) and error <= tolerance
    if verbose or not passed:
        print(
            f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True)
    if not passed:
        raise AssertionError(label)


def quantized_matrix(matrix, h, bits):
    """Independent public angle-format reference, including the top clamp."""
    result = np.zeros_like(matrix)
    for row, column in zip(*np.nonzero(matrix)):
        value = matrix[row, column]
        theta = np.arcsin(np.sqrt(abs(value) / h))
        code = min((1 << bits) - 1, round((1 << bits) * theta / (np.pi / 2)))
        result[row, column] = np.sign(value) * h * np.sin(np.pi * code /
                                                          (2 * (1 << bits)))**2
    return result


def inverse_polynomial(x_min, epsilon, full_cks=False, *, verbose=False):
    """CKS coefficients, adapted from example 07, with stable binomial tails.

    P[Binomial(2B,1/2)>B+j] replaces integer sums divided by 2**(2B),
    avoiding overflow. c=0.9/sqrt(B) gives analytical global headroom since
    |[1-(1-x*x)**B]/x| <= min(B*|x|,1/|x|) <= sqrt(B).
    """
    half_degree = max(3, int(np.ceil(np.log(1 / epsilon) / x_min**2)))
    degree = 2 * half_degree - 1
    print(
        f"CKS B={half_degree}, degree={degree}, requested relative "
        f"inverse target={epsilon:.3e}",
        flush=True)
    if degree > MAX_DEGREE:
        raise ValueError(
            f"degree {degree} exceeds example guard {MAX_DEGREE}; "
            "choose a less expensive explicit configuration")
    scale = .9 / np.sqrt(half_degree)
    indices = np.arange(half_degree)
    odd = scale * 4 * (-1.)**indices * binom.sf(half_degree + indices,
                                                2 * half_degree, .5)
    # CKS's degree bound is conservative. Its high-order Chebyshev
    # coefficients are tiny binomial tails. Since |T_k(x)| <= 1, the sum
    # of discarded absolute coefficients rigorously bounds the change
    # everywhere on [-1,1]. Keep this below 1e-10 and account for it
    # separately; the requested matrix precision and inverse target stay
    # unchanged. --full-cks executes every degree of the original recipe.
    tails = np.r_[np.cumsum(np.abs(odd[::-1]))[::-1], 0.]
    retained = half_degree
    if not full_cks:
        tail_target = min(1e-10, .01 * epsilon * scale)
        retained = next(k for k in range(1, half_degree + 1)
                        if tails[k] <= tail_target)
        if np.exp(-half_degree * x_min *
                  x_min) + tails[retained] / scale > epsilon:
            retained = half_degree
    tail_bound = float(tails[retained])
    odd = odd[:retained]
    degree = 2 * retained - 1
    print(
        f"Executed degree={degree}; discarded Chebyshev tail bound="
        f"{tail_bound:.3e} on all [-1,1]",
        flush=True)
    check("combined CKS and tail relative bound",
          np.exp(-half_degree * x_min * x_min) + tail_bound / scale,
          epsilon,
          verbose=verbose)
    coefficients = np.zeros(degree + 1)
    coefficients[1::2] = odd
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(
            "inverse coefficient construction produced nonfinite values")
    return half_degree, scale, odd, coefficients, tail_bound


def qsp_response(phases, values):
    """Complex raw device response from an independent 2x2 signal model."""
    responses = []
    for x in values:
        s = np.sqrt(max(0., 1 - x * x))
        step = np.array([[-x, -s], [s, -x]], dtype=complex)
        result = np.diag([np.exp(1j * phases[0]), np.exp(-1j * phases[0])])
        for phase in phases[1:]:
            result = np.diag([np.exp(1j * phase),
                              np.exp(-1j * phase)]) @ step @ result
        responses.append(np.exp(1j * np.sum(phases)) * result[0, 0])
    return np.array(responses)


def save_plot(path, positions, initial, numpy_solution, qsvt_solution):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.r_[0., positions, 1.]
    for values, label, marker in ((initial, "Initial", "o"),
                                  (numpy_solution, "NumPy next step", "s"),
                                  (qsvt_solution, "QSVT reconstruction", "x")):
        axis.plot(x, np.r_[0., values, 0.], marker=marker, label=label)
    axis.set(xlabel="Position x/L",
             ylabel="Temperature above ambient",
             title="One implicit heat-diffusion step")
    axis.legend()
    axis.grid(alpha=.25)
    figure.tight_layout()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"Plot saved: {path}")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--r", type=float, default=.25)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--inverse-epsilon", type=float, default=1e-2)
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--full-cks",
        action="store_true",
        help="execute the uncompressed CKS polynomial (much slower on CPU)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each validation check (failures are always reported)")
    args = parser.parse_args()
    if args.points < 2 or args.points & (args.points - 1):
        parser.error("--points must be a power of two, at least 2")
    if not np.isfinite(args.r) or args.r <= 0:
        parser.error("--r must be finite and positive")
    if not 1 <= args.value_bits <= 12:
        parser.error("--value-bits must lie in [1,12]")
    if not np.isfinite(
            args.inverse_epsilon) or not 0 < args.inverse_epsilon < .1:
        parser.error("--inverse-epsilon must lie strictly between 0 and 0.1")
    system_bits = args.points.bit_length() - 1
    # Three bands pad to four slots. Guard before constructing matrices,
    # kernels, coefficient arrays, or statevectors.
    if 2 * system_bits + args.value_bits + 8 > MAX_QUBITS:
        parser.error(
            f"requested geometry exceeds the {MAX_QUBITS}-qubit example guard")
    return args


def main():
    args = arguments()
    verbose = args.verbose
    started = time.perf_counter()
    target = os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")
    if target == "nvidia":
        cudaq.set_target(target, option="fp64")
    else:
        cudaq.set_target(target)
    n, r = args.points, args.r
    dx = 1 / (n + 1)
    positions = np.arange(1, n + 1) * dx
    initial = np.zeros(n)
    initial[1] = 1.
    beta = float(np.linalg.norm(initial))
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(
            "the initial temperature must have finite nonzero norm")
    ideal = (np.diag(np.full(n, 1 + 2 * r)) + np.diag(np.full(n - 1, -r), 1) +
             np.diag(np.full(n - 1, -r), -1))
    numpy_solution = np.linalg.solve(ideal, initial)
    spectrum = 1 + 2 * r - 2 * r * np.cos(
        np.arange(1, n + 1) * np.pi / (n + 1))
    check("analytic SPD spectrum",
          np.max(np.abs(np.linalg.eigvalsh(ideal) - spectrum)),
          BLOCK_ATOL,
          verbose=verbose)
    check("ideal NumPy residual",
          np.linalg.norm(ideal @ numpy_solution - initial),
          BLOCK_ATOL,
          verbose=verbose)
    if n == 4 and r == .25:
        analytic = np.array([140., 840., 144., 24.]) / 1189.
        check("default rational NumPy solution",
              np.max(np.abs(numpy_solution - analytic)),
              BLOCK_ATOL,
              verbose=verbose)
    print(
        f"N={n}, L=1, dx={dx:.6f}, r=D*dt/dx**2={r}; "
        "fixed zero endpoints are eliminated",
        flush=True)

    system_bits = n.bit_length() - 1
    oracles = banded_oracles([-1, 0, 1], [-r, 1 + 2 * r, -r],
                             num_system=system_bits,
                             value_bits=args.value_bits)
    encoding = SparseOracleEncoding(oracles, num_system=system_bits)
    check("three bands pad to four slots",
          abs(encoding.d_padded - 4),
          0.,
          verbose=verbose)
    check("sparse normalization",
          abs(encoding.alpha - 4 * (1 + 2 * r)),
          BLOCK_ATOL,
          verbose=verbose)
    print(
        f"system={encoding.num_system}, block ancillas={encoding.num_block_ancilla}, "
        f"scratch={encoding.num_scratch}, all ancillas={encoding.num_ancilla}, "
        f"total={encoding.num_system+encoding.num_ancilla}, "
        f"angle bits={encoding.value_bits}, slots={encoding.d}/{encoding.d_padded}, "
        f"h={encoding.h}, alpha={encoding.alpha}",
        flush=True)
    realized = quantized_matrix(ideal, encoding.h, args.value_bits)
    values, vectors = np.linalg.eigh(realized)
    normalized = values / encoding.alpha
    if not np.all((normalized > 0) & (normalized <= 1)):
        raise ValueError(
            "the quantized matrix must have spectrum in (0,alpha]")
    quantized_solution = np.linalg.solve(realized, initial)
    half_degree, scale, odd, coefficients, tail_bound = inverse_polynomial(
        float(normalized.min()),
        args.inverse_epsilon,
        args.full_cks,
        verbose=verbose)
    domain = np.linspace(normalized.min(), normalized.max(), 257)
    polynomial = chebyshev.chebval(domain, coefficients)
    check("CKS relative inverse approximation on interval",
          np.max(np.abs(1 - domain * polynomial / scale)),
          args.inverse_epsilon + 1e-12,
          verbose=verbose)
    grid = np.linspace(-1., 1., 4097)
    check("global polynomial headroom",
          np.max(np.abs(chebyshev.chebval(grid, coefficients))),
          .9 + 1e-12,
          verbose=verbose)

    try:
        import qsppack
    except ImportError as error:
        raise SystemExit(
            "This example needs QSPPACK: pip install qsppack") from error
    options = {
        "criteria": 1e-10,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
        "targetPre": True
    }
    with contextlib.redirect_stdout(io.StringIO()):
        phases, _ = qsppack.solve(odd.copy(), 1, options)
    phases = [float(phase) for phase in phases]
    if len(phases) != len(coefficients):
        raise AssertionError("QSPPACK returned the wrong phase count")
    response = qsp_response(phases, normalized)
    # The walk signal is -A/alpha. Odd CKS parity therefore contributes a
    # minus sign, and doubled projector phases add exp(i*sum(raw phases)).
    # Both corrections are analytical; the scalar model verifies them.
    corrected = -(response * np.exp(-1j * np.sum(phases))).real
    intended = chebyshev.chebval(normalized, coefficients)
    phase_error = float(np.max(np.abs(corrected - intended)))
    check("phase synthesis on actual spectrum",
          phase_error,
          PHASE_ATOL,
          verbose=verbose)
    probes = qsp_response(phases, domain)
    check("phase synthesis on interval probes",
          np.max(
              np.abs(-(probes * np.exp(-1j * np.sum(phases))).real -
                     polynomial)),
          PHASE_ATOL,
          verbose=verbose)
    print(
        f"phases={len(phases)}, analytical c={scale:.9f}, beta={beta}; "
        f"physical recovery beta/(c*alpha)={beta/(scale*encoding.alpha):.9f}",
        flush=True)

    columns = []
    for index in range(n):
        basis = np.zeros(n, dtype=complex)
        basis[index] = 1.
        columns.append(sim.action(encoding, basis))
    block = np.column_stack(columns)
    check("circuit block versus independently quantized matrix",
          np.max(np.abs(encoding.alpha * block - realized)),
          BLOCK_ATOL,
          verbose=verbose)
    if n > 2:
        check("no end-to-end corner coupling",
              np.max(np.abs(block[[0, n - 1], [n - 1, 0]])),
              BLOCK_ATOL,
              verbose=verbose)
    print("Executing the sparse QSVT circuit...", flush=True)
    ket = np.asarray(initial / beta, dtype=complex)
    raw = sim.run_on_large_stack(sim.transform, QSVT(encoding), ket,
                                 PhaseSequence(phases, convention="qsp"))
    expected_raw = vectors @ (response * (vectors.conj().T @ ket))
    check("raw complex QSVT block versus scalar model",
          np.max(np.abs(raw - expected_raw)),
          CIRCUIT_ATOL,
          verbose=verbose)
    reconstructed = -(raw * np.exp(-1j * np.sum(phases))).real * beta / (
        scale * encoding.alpha)
    quantization_error = float(np.linalg.norm(realized - ideal, 2))
    inverse_norm = 1 / values.min()
    coefficient_bound = inverse_norm * quantization_error * np.linalg.norm(
        numpy_solution)
    polynomial_bound = args.inverse_epsilon * np.linalg.norm(
        quantized_solution)
    truncation_bound = beta * tail_bound / (scale * encoding.alpha)
    phase_bound = beta * phase_error / (scale * encoding.alpha)
    roundoff_bound = beta * np.sqrt(n) * CIRCUIT_ATOL / (scale *
                                                         encoding.alpha)
    solve_bound = polynomial_bound + truncation_bound + phase_bound + roundoff_bound
    total_bound = coefficient_bound + solve_bound
    check("quantized versus ideal solve",
          np.linalg.norm(quantized_solution - numpy_solution),
          coefficient_bound + 1e-12,
          verbose=verbose)
    check("QSVT versus quantized NumPy solve",
          np.linalg.norm(reconstructed - quantized_solution),
          solve_bound,
          verbose=verbose)
    check("QSVT versus ideal NumPy solve",
          np.linalg.norm(reconstructed - numpy_solution),
          total_bound,
          verbose=verbose)

    relative_error = np.linalg.norm(
        reconstructed - numpy_solution) / np.linalg.norm(numpy_solution)
    fidelity = abs(np.vdot(reconstructed, numpy_solution))**2 / (
        np.linalg.norm(reconstructed)**2 * np.linalg.norm(numpy_solution)**2)
    residual = np.linalg.norm(ideal @ reconstructed - initial) / beta
    residual_bound = np.linalg.norm(ideal, 2) * total_bound / beta
    check("relative physical residual",
          residual,
          residual_bound,
          verbose=verbose)
    print(
        f"matrix quantization norm={quantization_error:.6e}; "
        f"quantized-solve relative error="
        f"{np.linalg.norm(quantized_solution-numpy_solution)/np.linalg.norm(numpy_solution):.6e}"
    )
    print(f"error bounds: coefficients={coefficient_bound:.3e}, "
          f"polynomial={polynomial_bound:.3e}, tail={truncation_bound:.3e}, "
          f"phases={phase_bound:.3e}, "
          f"roundoff={roundoff_bound:.3e}, total={total_bound:.3e}")
    print(
        "\nPosition   Initial      NumPy       Quantized NumPy    QSVT        Abs error"
    )
    for x, before, exact, rounded, output in zip(positions, initial,
                                                 numpy_solution,
                                                 quantized_solution,
                                                 reconstructed):
        print(f"{x:8.4f} {before:9.5f} {exact:13.9f} {rounded:15.9f} "
              f"{output:13.9f} {abs(output-exact):11.3e}")
    print(f"full-vector relative error={relative_error:.6e}; "
          f"direction fidelity={fidelity:.9f}; residual={residual:.6e}")
    print(
        "Raw good-block norm is not isolated inverse-filter success probability."
    )
    print(
        "Full-state and real-response extraction are simulator postprocessing."
    )
    print("These checks validate the discrete solve; PDE space/time accuracy "
          "needs a separate convergence study.")
    if args.plot:
        save_plot(args.plot, positions, initial, numpy_solution, reconstructed)
    print(
        f"Maximum executed qubits: {encoding.num_system+encoding.num_ancilla}; "
        f"degree={len(phases)-1} (full CKS {2*half_degree-1}); "
        f"elapsed: {time.perf_counter()-started:.2f}s")


if __name__ == "__main__":
    main()
