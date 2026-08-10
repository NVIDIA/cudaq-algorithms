#!/usr/bin/env python3
# [Begin Documentation]
"""Example 7 — Invert a 5x5 matrix with QSVT (the quantum linear solver).

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
    kappa^2 log(1/eps), so a well-conditioned A keeps it small.
  * Normalization. A QSVT polynomial must satisfy |p| <= 1 on [-1, 1],
    but 1/x is unbounded. So we implement a *scaled* inverse p(x) ~ c/x;
    the circuit returns c*alpha*A^-1 b, and the classical constant is
    recovered from A and b afterward (standard HHL post-processing).

Verified against numpy.linalg.solve: direction fidelity ~1 and a small
residual ||A x_hat - b||. Run: python3 07_matrix_inversion_qsvt.py
"""
from __future__ import annotations

import contextlib
import io
import os

import cudaq
import numpy as np
from numpy.polynomial import chebyshev
from scipy.special import comb

from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT
from cudaq_algorithms import sim_utils as sim

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
    coeffs = np.zeros(b)
    for j in range(b):
        tail = sum(comb(2 * b, b + i, exact=True) for i in range(j + 1, b + 1))
        coeffs[j] = 4 * ((-1)**j) * tail / (2.0**(2 * b))
    return coeffs


def evaluate_odd(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate sum_j coeffs[j] T_{2j+1}(x)."""
    value = np.zeros_like(x, dtype=float)
    for j, coefficient in enumerate(coeffs):
        basis = np.zeros(2 * j + 2)
        basis[2 * j + 1] = 1.0
        value += coefficient * chebyshev.chebval(x, basis)
    return value


def inverse_phase_sequence(x_min: float, epsilon: float):
    """Phase factors for a scaled 1/x on [x_min, 1], and the scale c.

    Picks the CKS degree from the error bound |(1-x^2)^b| <= exp(-b x_min^2),
    scales the polynomial to sit safely inside |p| <= 1 (so the QSP solve
    stays well-conditioned), and returns qsp-convention phases.
    """
    import qsppack

    b = max(int(np.ceil(np.log(1 / epsilon) / x_min**2)), 3)
    coeffs = inverse_chebyshev_coeffs(b)
    # 1/x-proxy peaks at ~sqrt(b); rescale so |p| tops out near 0.9. Leaving
    # headroom below 1 is what keeps the phase-factor solve well-conditioned.
    grid = np.linspace(-1, 1, 4000)
    scale = 0.9 / np.max(np.abs(evaluate_odd(coeffs, grid)))
    coeffs = coeffs * scale

    options = {
        "criteria": 1e-8,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
        "targetPre": True,
    }
    with contextlib.redirect_stdout(io.StringIO()):  # hush the QSP solver
        phases, _ = qsppack.solve(coeffs.copy(), 1, options)
    return [float(p) for p in phases], scale, b


def main() -> int:
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
    #    the qsp global phase exp(i * sum phi) -- is p(A/alpha) b.
    b = np.zeros(8, dtype=complex)
    b[:5] = rng.normal(size=5)
    b /= np.linalg.norm(b)

    raw = sim.transform(transformer, b, PhaseSequence(phases,
                                                      convention="qsp"))
    good = (raw * np.exp(-1j * np.sum(phases))).real

    # 5. The state is proportional to A^-1 b. Recover the real constant from
    #    A and b (both known), then check the residual -- HHL post-processing.
    #    c = <b, A good> / <b, b> = scale * alpha up to the convention sign.
    constant = float(np.real(np.vdot(b, a8 @ good)) / np.real(np.vdot(b, b)))
    x_hat = good / constant

    exact = np.linalg.solve(a8, b).real
    fidelity = float(
        abs(np.vdot(good / np.linalg.norm(good),
                    exact / np.linalg.norm(exact))))
    residual = float(np.linalg.norm(a8 @ x_hat - b))
    solution_error = float(
        np.linalg.norm(x_hat - exact) / np.linalg.norm(exact))

    print(f"\nrecovered constant c  : {constant:+.5f}  "
          f"(|c| vs scale*alpha = {scale * encoding.alpha:.5f})")
    print(f"direction fidelity    : {fidelity:.6f}")
    print(f"residual ||A x - b||  : {residual:.3e}")
    print(f"solution rel. error   : {solution_error:.3e}")

    assert fidelity > 0.999
    assert residual < 1e-2
    print("\nOK — QSVT applied 1/x to the block-encoded matrix; the "
          "good-subspace state solves A x = b.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
