#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 7 — Invert a 5x5 matrix with QSVT (the quantum linear solver).

The task is to solve a linear system: given a matrix A and an input vector b,
find the vector x satisfying A x = b. QSVT approaches this by applying a
polynomial that approximates the reciprocal of each eigenvalue of the encoded
matrix. This example carries that idea through a small dense solve and a
sparse solve, with independent NumPy solutions providing concrete targets for
the circuit output.

You should learn why eigenvalues near zero make inversion harder, why the
quantum circuit implements a scaled approximation to the inverse, and how
known input and encoding scales recover the solution's magnitude. The
comparisons separate matrix rounding, polynomial approximation, and circuit
error, and check both the output vector and its residual in A x = b. Reading
all coordinates from a simulator is part of this demonstration; the example
explains what would still be needed to extract useful information from a
quantum state on hardware.

Time evolution (example 2) applied a QSVT polynomial approximating
exp(-ixt). Swap the polynomial and the same machinery solves A x = b:
apply a polynomial approximating 1/x to the block-encoded matrix and the
good-subspace state is proportional to A^-1 b. This is the QSVT reading of
HHL, and it reuses PauliLCU + QSVT unchanged -- only the phase sequence is
new.

Three ideas carry the example:

  * Embedding. A is 5x5; a register holds 2^k amplitudes. Pad A to the
    next power of two (8x8, three qubits) with an identity block. The pad
    is block-diagonal, so with b supported on the first five coordinates
    the solution never leaks into the padding.
  * The polynomial. 1/x is singular at 0, so it can only be approximated
    away from zero, on the spectrum's domain [x_min, x_max]. We use the
    Childs-Kothari-Somma closed form: (1 - (1-x^2)^b)/x is an odd
    polynomial (degree 2b-1) with an exact Chebyshev expansion, and it
    tracks 1/x wherever |x| is bounded away from 0. Degree grows like
    (alpha/lambda_min)^2 log(1/eps): encoding normalization matters as well
    as the physical condition number.
  * Normalization. A QSVT polynomial must satisfy |p| <= 1 on [-1, 1],
    but 1/x is unbounded. So we implement a *scaled* inverse p(x) ~ c/x;
    the circuit returns c*alpha*A^-1 b, and the classical constant is
    recovered from the known input, polynomial, and encoding scales.

A second, 2x2 sparse case uses table-backed encode_sparse at three angle
bits, independently reconstructs the quantized operator, and generates its
own inverse phases. It separates circuit, quantization, and polynomial
errors. Real-response extraction uses full-state simulator postprocessing;
it is not a coherent procedure for reading out every solution coordinate.

Verified against numpy.linalg.solve: direction fidelity ~1 and a small
residual ||A x_hat - b||. Run: python3 07_matrix_inversion_qsvt.py
Add --verbose to show individual validation checks; failures are always reported.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

import cudaq
import numpy as np
from numpy.polynomial import chebyshev
from scipy.stats import binom

from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT
from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms.sparse import encode_sparse

_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def pauli_terms(matrix: np.ndarray, num_qubits: int) -> dict:
    """Decompose a Hermitian matrix into a {pauli_word: coefficient} sum.

    Little-endian, matching CUDA-Q and example 1: word[0] is the rightmost
    tensor factor. Coefficients are c_P = tr(P^dagger M) / 2^n; for a
    Hermitian M they come out real.
    """
    terms: dict = {}
    for index in range(4**num_qubits):
        labels, digits = [], index
        for _ in range(num_qubits):
            labels.append("IXYZ"[digits % 4])
            digits //= 4
        word = "".join(labels)
        factor = np.array([[1]], dtype=complex)
        for label in word:
            factor = np.kron(_PAULIS[label], factor)
        coefficient = np.trace(factor.conj().T @ matrix) / (2**num_qubits)
        if abs(coefficient) > 1e-10:
            terms[word] = coefficient.real
    return terms


def inverse_chebyshev_coeffs(b: int) -> np.ndarray:
    """Chebyshev coefficients of (1 - (1-x^2)^b)/x, the CKS 1/x proxy.

    Returns the coefficients of T_1, T_3, ..., T_{2b-1} (odd parity) --
    exactly the form qsppack.solve consumes. The identity is

    (1-(1-x^2)^b)/x = 4 sum_j (-1)^j [2^-2b sum_{i=j+1}^b C(2b, b+i)] T_{2j+1}.
    """
    # Stable binomial survival probabilities avoid overflowing 2.0**(2*b).
    j = np.arange(b)
    return 4 * (-1.)**j * binom.sf(b + j, 2 * b, .5)


def evaluate_odd(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate sum_j coeffs[j] T_{2j+1}(x)."""
    full = np.zeros(2 * len(coeffs))
    full[1::2] = coeffs
    return chebyshev.chebval(x, full)


def inverse_phase_sequence(x_min: float, epsilon: float):
    """Phase factors for a scaled 1/x on [x_min, 1], and the scale c.

    Picks the CKS degree from the error bound |(1-x^2)^b| <= exp(-b x_min^2),
    scales the polynomial to sit safely inside |p| <= 1 (so the QSP solve
    stays well-conditioned), and returns qsp-convention phases.
    """
    import qsppack

    b = max(int(np.ceil(np.log(1 / epsilon) / x_min**2)), 3)
    print(f"Planning CKS degree {2*b-1} from x_min={x_min:.6f}", flush=True)
    if 2 * b - 1 > 511:
        raise ValueError(
            "requested inverse exceeds this example's degree-511 guard")
    coeffs = inverse_chebyshev_coeffs(b)
    # |f_B(x)| <= min(B|x|, 1/|x|) <= sqrt(B) gives global headroom.
    scale = 0.9 / np.sqrt(b)
    coeffs = coeffs * scale

    options = {
        "criteria": 1e-10,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
        "targetPre": True,
    }
    with contextlib.redirect_stdout(io.StringIO()):  # hush the QSP solver
        phases, _ = qsppack.solve(coeffs.copy(), 1, options)
    return [float(p) for p in phases], scale, b


def check(label, error, tolerance=1e-10, *, verbose=False):
    passed = np.isfinite(error) and error <= tolerance
    if verbose or not passed:
        print(f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
              f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(label)


def qsp_response(phases, values):
    """Independent raw device response; the walk signal is minus H/alpha."""
    responses = []
    for value in values:
        root = np.sqrt(max(0., 1 - value * value))
        step = np.array([[-value, -root], [root, -value]], dtype=complex)
        result = np.diag([np.exp(1j * phases[0]), np.exp(-1j * phases[0])])
        for phase in phases[1:]:
            result = np.diag([np.exp(1j * phase),
                              np.exp(-1j * phase)]) @ step @ result
        responses.append(np.exp(1j * np.sum(phases)) * result[0, 0])
    return np.array(responses)


def sparse_inverse(*, verbose=False):
    """Fresh phases for a power-of-two sparse SPD system, with no fitted scale."""
    started = time.perf_counter()
    ideal = np.array([[1.5, -.25], [-.25, 1.5]])
    rhs = np.array([1., 0.])  # Not an eigenvector.
    beta = float(np.linalg.norm(rhs))
    encoding = encode_sparse(ideal, value_bits=3)
    print("\nSparse inverse: QROM-backed 2x2 SPD matrix; report:",
          encoding.report)
    print(
        f"system={encoding.num_system}, block ancillas={encoding.num_block_ancilla}, "
        f"scratch={encoding.num_scratch}, all ancillas={encoding.num_ancilla}, "
        f"total={encoding.num_system+encoding.num_ancilla}")
    # The same independent angle rule as the public storage contract. Do not
    # obtain the reference by multiplying a measured block by alpha.
    realized = np.zeros_like(ideal)
    for row, column in zip(*np.nonzero(ideal)):
        value = ideal[row, column]
        theta = np.arcsin(np.sqrt(abs(value) / encoding.h))
        code = min((1 << encoding.value_bits) - 1,
                   round((1 << encoding.value_bits) * theta / (np.pi / 2)))
        realized[row, column] = np.sign(value) * encoding.h * np.sin(
            np.pi * code / (2 * (1 << encoding.value_bits)))**2
    columns = []
    for index in range(2):
        basis = np.zeros(2, dtype=complex)
        basis[index] = 1.
        columns.append(sim.action(encoding, basis))
    check("sparse encoding versus quantized matrix",
          np.max(np.abs(encoding.alpha * np.column_stack(columns) - realized)),
          verbose=verbose)
    exact = np.linalg.solve(ideal, rhs)
    rounded = np.linalg.solve(realized, rhs)
    values, vectors = np.linalg.eigh(realized)
    normalized = values / encoding.alpha
    epsilon = 1e-3
    phases, scale, half_degree = inverse_phase_sequence(
        float(normalized.min()), epsilon)
    polynomial = scale * evaluate_odd(inverse_chebyshev_coeffs(half_degree),
                                      normalized)
    check("sparse relative inverse polynomial",
          np.max(np.abs(1 - normalized * polynomial / scale)),
          epsilon + 1e-12,
          verbose=verbose)
    response = qsp_response(phases, normalized)
    # Odd parity under x -> -x supplies minus; doubled projector phases
    # supply exp(i*sum(phi)). Verify these corrections before any solve.
    corrected = -(response * np.exp(-1j * np.sum(phases))).real
    phase_error = float(np.max(np.abs(corrected - polynomial)))
    check("sparse phase synthesis versus inverse polynomial",
          phase_error,
          2e-8,
          verbose=verbose)
    ket = np.asarray(rhs / beta, dtype=complex)
    raw = sim.run_on_large_stack(sim.transform, QSVT(encoding), ket,
                                 PhaseSequence(phases, convention="qsp"))
    expected = vectors @ (response * (vectors.conj().T @ ket))
    circuit_atol = 1e-8
    check("sparse raw complex QSVT versus scalar model",
          np.max(np.abs(raw - expected)),
          circuit_atol,
          verbose=verbose)
    solution = -(raw * np.exp(-1j * np.sum(phases))).real * beta / (
        scale * encoding.alpha)
    matrix_error = float(np.linalg.norm(realized - ideal, 2))
    quantization_bound = matrix_error * np.linalg.norm(exact) / values.min()
    polynomial_bound = epsilon * np.linalg.norm(rounded)
    phase_bound = beta * phase_error / (scale * encoding.alpha)
    roundoff_bound = beta * np.sqrt(2) * circuit_atol / (scale *
                                                         encoding.alpha)
    inverse_bound = polynomial_bound + phase_bound + roundoff_bound
    overall_bound = quantization_bound + inverse_bound
    check("sparse quantized versus ideal NumPy solve",
          np.linalg.norm(rounded - exact),
          quantization_bound + 1e-12,
          verbose=verbose)
    check("sparse reconstructed versus quantized solve",
          np.linalg.norm(solution - rounded),
          inverse_bound,
          verbose=verbose)
    check("sparse reconstructed versus ideal solve",
          np.linalg.norm(solution - exact),
          overall_bound,
          verbose=verbose)
    residual = np.linalg.norm(ideal @ solution - rhs) / beta
    check("sparse physical residual",
          residual,
          np.linalg.norm(ideal, 2) * overall_bound / beta,
          verbose=verbose)
    print("  coordinate   ideal NumPy    quantized NumPy    QSVT")
    for index in range(2):
        print(f"  {index:10d} {exact[index]:13.9f} {rounded[index]:17.9f} "
              f"{solution[index]:13.9f}")
    print(
        f"matrix quantization error={matrix_error:.3e}; "
        f"relative solution error={np.linalg.norm(solution-exact)/np.linalg.norm(exact):.3e}"
    )
    print(f"degree={2*half_degree-1}; c={scale:.9f}; beta={beta}; "
          f"known physical recovery={beta/(scale*encoding.alpha):.9f}")
    print(f"error bounds: quantization={quantization_bound:.3e}, "
          f"polynomial={polynomial_bound:.3e}, phase={phase_bound:.3e}, "
          f"roundoff={roundoff_bound:.3e}")
    print(
        "Real-response extraction is simulator postprocessing; the raw good-block "
        "norm is not isolated inverse-filter success probability.")
    print(
        "No zero padding is introduced; zero-padded reciprocal problems need "
        "explicit treatment of zero eigenvalues and RHS support.")
    print(
        f"Sparse case elapsed: {time.perf_counter()-started:.2f}s; "
        f"maximum executed qubits: {encoding.num_system+encoding.num_ancilla}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each validation check (failures are always reported)")
    verbose = parser.parse_args().verbose
    started = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    # 1. A 5x5 symmetric positive-definite matrix with a controlled spectrum
    #    in [1, 2] (condition number 2). The smallest eigenvalue is 1, so
    #    padding to 8x8 with an identity block adds no new small eigenvalue.
    rng = np.random.default_rng(3)
    basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    spectrum = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    a5 = (basis * spectrum) @ basis.T
    a5 = 0.5 * (a5 + a5.T)
    a8 = np.eye(8, dtype=complex)
    a8[:5, :5] = a5

    # 2. Block-encode A (Hermitian) as a Pauli LCU. QSVT applies polynomials
    #    to x = eigenvalue / alpha, so the working domain is [x_min, x_max].
    terms = pauli_terms(a8, num_qubits=3)
    encoding = PauliLCU(terms)
    transformer = QSVT(encoding)

    eigenvalues = np.linalg.eigvalsh(a8)
    x = eigenvalues / encoding.alpha
    x_min, x_max = float(x.min()), float(x.max())
    print(f"matrix                : 5x5 SPD, padded to 8x8 (3 system qubits)")
    print(f"Pauli LCU terms       : {len(terms)}  "
          f"(system {encoding.num_system} + ancilla {encoding.num_ancilla})")
    print(f"alpha (1-norm)        : {encoding.alpha:.6f}")
    print(
        f"condition number      : {eigenvalues.max() / eigenvalues.min():.2f}")
    print(f"spectral domain x     : [{x_min:.4f}, {x_max:.4f}]")

    # 3. Build phases for a scaled 1/x over the spectral domain, and check
    #    the polynomial tracks scale/x there before trusting the circuit.
    phases, scale, degree_half = inverse_phase_sequence(x_min, epsilon=1e-2)
    domain = np.linspace(x_min, x_max, 200)
    polynomial = scale * evaluate_odd(inverse_chebyshev_coeffs(degree_half),
                                      domain)
    poly_error = float(np.max(np.abs(polynomial - scale / domain)))
    print(f"CKS polynomial degree : {2 * degree_half - 1}  "
          f"(scaled by c = {scale:.5f})")
    print(f"|p(x) - c/x| on domain: {poly_error:.2e}")
    print(f"phase factors         : {len(phases)}")

    # 4. Solve A x = b. Prepare b on the system register, run the QSVT
    #    sequence, keep the all-zero-ancilla block. The qsp convention makes
    #    the good-subspace amplitude complex (p(x) + i * complementary(x));
    #    with a real b and real eigenvectors, the real part -- after removing
    #    the qsp global phase exp(i * sum phi) and odd-parity minus sign --
    #    is p(A/alpha) b. Full-state real extraction is simulator postprocessing.
    b = np.zeros(8, dtype=complex)
    b[:5] = rng.normal(size=5)
    b /= np.linalg.norm(b)

    raw = sim.transform(transformer, b, PhaseSequence(phases,
                                                      convention="qsp"))
    response = qsp_response(phases, x)
    corrected_response = -(response * np.exp(-1j * np.sum(phases))).real
    intended = scale * evaluate_odd(inverse_chebyshev_coeffs(degree_half), x)
    check("dense phase convention versus polynomial",
          np.max(np.abs(corrected_response - intended)),
          2e-8,
          verbose=verbose)
    values, vectors = np.linalg.eigh(a8)
    response = qsp_response(phases, values / encoding.alpha)
    expected_raw = vectors @ (response * (vectors.conj().T @ b))
    check("dense raw QSVT versus scalar model",
          np.max(np.abs(raw - expected_raw)),
          1e-8,
          verbose=verbose)
    good = -(raw * np.exp(-1j * np.sum(phases))).real

    # 5. b has unit norm. The known polynomial and encoding scales recover
    #    A^-1 b directly; no scalar is fitted to NumPy or to the residual.
    constant = scale * encoding.alpha
    x_hat = good / constant

    exact = np.linalg.solve(a8, b).real
    fidelity = float(
        abs(np.vdot(good / np.linalg.norm(good),
                    exact / np.linalg.norm(exact))))
    residual = float(np.linalg.norm(a8 @ x_hat - b))
    solution_error = float(
        np.linalg.norm(x_hat - exact) / np.linalg.norm(exact))

    print(f"\nknown response scale  : c*alpha = {constant:+.5f}")
    print(f"direction fidelity    : {fidelity:.6f}")
    print(f"residual ||A x - b||  : {residual:.3e}")
    print(f"solution rel. error   : {solution_error:.3e}")

    assert fidelity > 0.999
    assert residual < 1e-2
    print("\nOK — QSVT applied 1/x to the block-encoded matrix; the "
          "good-subspace state solves A x = b.")
    sparse_inverse(verbose=verbose)
    print(f"Total example elapsed: {time.perf_counter()-started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
