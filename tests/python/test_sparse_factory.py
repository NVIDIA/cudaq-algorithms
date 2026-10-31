# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""encode_sparse and the QROM-backed from-data oracles.

What this file pins:

- ``encode_sparse`` builds ``SparseOracleEncoding`` over ``qrom_oracles``
  and reports ``alpha = d_padded * h`` with the matching-based slot count
  (which can exceed the row sparsity), the register budget and the input
  shape;
- skewed and uniform matrices encode within the derived quantization
  bound; signed off-diagonals and a partial diagonal through
  ``qrom_oracles`` directly;
- non-symmetric input and negative diagonals raise, naming the way out;
  ``dilate=True`` encodes ``[[0, A], [A^T, 0]]`` once at the data level
  and accepts both;
- above ``max_entries`` the factory raises pointing at hand-written
  oracles; argument validation; ``report`` is ``None`` on encodings built
  directly.

Circuit-level checks follow the two-level validation culture at dim <= 8:
the block against the *exactly encoded* matrix (fixed-point angles) at
1e-10, and that matrix against the ideal one within the derived
quantization bound.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms.sparse import (SparseOracleEncoding, encode_sparse,
                                     qrom_oracles)


def quantized_angle(value: float, h: float, value_bits: int) -> int:
    """Test-local copy of the oracle contract's fixed-point angle,
    ``round(arcsin(sqrt(|v|/h)) / (pi/2) * 2^bits)`` clamped to
    ``2^bits - 1``, so the references do not import the quantizer under
    test."""
    theta = math.asin(math.sqrt(min(1.0, abs(value) / h)))
    return min((1 << value_bits) - 1,
               int(round(theta / (0.5 * math.pi) * (1 << value_bits))))


# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------


def oracle_quantized_dense(matrix: np.ndarray, h: float,
                           value_bits: int) -> np.ndarray:
    """The exactly encoded matrix: each element's magnitude replaced by
    ``h sin^2(theta_q)`` (both T factors load the same fixed-point angle),
    signs exact."""
    quantized = np.zeros_like(matrix)
    for i, j in zip(*np.nonzero(matrix)):
        angle = quantized_angle(matrix[i, j], h, value_bits)
        magnitude = h * math.sin(angle / (1 << value_bits) * 0.5 * math.pi)**2
        quantized[i, j] = math.copysign(magnitude, matrix[i, j])
    return quantized


def oracle_quantization_atol(h: float, value_bits: int) -> float:
    """Per-element bound |H_quantized - H|."""
    return h * 0.5 * math.pi * 2.0**(-value_bits)


def random_ket(dim: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


def encoded_block(encoding, kernel, ket: np.ndarray) -> np.ndarray:
    """<0|_anc U |0_anc, ket>: the first 2**num_system amplitudes."""
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << encoding.num_system]


def encoded_matrix(encoding) -> np.ndarray:
    """Extract the encoded block column-by-column via cudaq.get_state."""
    dim = 1 << encoding.num_system
    kernel = encoding.encode_kernel()
    block = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        column = np.zeros(dim, dtype=np.complex128)
        column[j] = 1.0
        block[:, j] = encoded_block(encoding, kernel, column)
    return block


def skewed_matrix() -> np.ndarray:
    """One dominant pair and three tiny ones (all 1-sparse rows): a single
    matching slot, alpha = d_padded * h = 2 * 1.0."""
    matrix = np.zeros((8, 8))
    for i, j, value in [(0, 1, 1.0), (2, 3, 0.05), (4, 5, 0.05), (6, 7, 0.05)]:
        matrix[i, j] = matrix[j, i] = value
    return matrix


def tridiagonal_matrix(dim: int) -> np.ndarray:
    """Uniform tridiagonal (diagonal and off-diagonals all 1.0): two
    matchings plus the diagonal slot, padded to four, alpha = 4."""
    return (np.eye(dim) + np.eye(dim, k=1) + np.eye(dim, k=-1))


def total_qubits(encoding) -> int:
    return encoding.num_system + encoding.num_ancilla


# ----------------------------------------------------------------------
# 1. encode_sparse: the report and the encoded block
# ----------------------------------------------------------------------


def test_uniform_banded_matrix_reports_slots_and_alpha():
    matrix = tridiagonal_matrix(4)
    encoding = encode_sparse(matrix, value_bits=4)
    assert isinstance(encoding, SparseOracleEncoding)

    report = encoding.report
    # 2 greedy matchings + the diagonal slot, padded to 4 slots, h = 1.
    assert encoding.d == 3 and encoding.d_padded == 4
    assert report["alpha"] == pytest.approx(4.0)
    assert report["alpha"] == pytest.approx(encoding.alpha, abs=0.0)
    assert report["nnz"] == 10
    assert report["max_row_nnz"] == 3  # max nonzeros per row
    assert report["slots"] == 3
    assert report["dim"] == 4
    assert report["dilated"] is False
    assert report["value_bits"] == 4
    assert report["qubits"] == total_qubits(encoding)
    text = repr(encoding)
    assert "encode_sparse" in text and "nnz=10" in text and "slots=3" in text

    ket = random_ket(4, seed=23)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    quantized = oracle_quantized_dense(matrix, 1.0, 4)
    np.testing.assert_allclose(encoding.alpha * block,
                               quantized @ ket,
                               atol=1e-10)
    ideal_atol = oracle_quantization_atol(1.0, 4) * math.sqrt(4)
    np.testing.assert_allclose(encoding.alpha * block,
                               matrix @ ket,
                               atol=ideal_atol)


def test_skewed_weights_pay_the_largest_element_at_the_padded_sparsity():
    # One matching holds all four pairs: d = 1, padded to 2, h = 1, so
    # alpha = 2 although the one-norm of the entries is 1.15 -- the price
    # of the oracle construction on skewed data, stated by the report.
    matrix = skewed_matrix()
    encoding = encode_sparse(matrix, value_bits=6)
    report = encoding.report
    assert report["alpha"] == pytest.approx(2.0)
    assert report["nnz"] == 8
    assert report["max_row_nnz"] == 1 and report["slots"] == 1
    assert encoding.d == 1 and encoding.d_padded == 2

    ket = random_ket(8)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    quantized = oracle_quantized_dense(matrix, 1.0, 6)
    np.testing.assert_allclose(encoding.alpha * block,
                               quantized @ ket,
                               atol=1e-10)
    np.testing.assert_allclose(encoding.alpha * block,
                               matrix @ ket,
                               atol=oracle_quantization_atol(1.0, 6) *
                               math.sqrt(8))


# ----------------------------------------------------------------------
# 2. The from-data (QROM-backed) adapter, standalone
# ----------------------------------------------------------------------


def test_qrom_oracles_standalone_matches_dense():
    # Non-banded pattern with signed off-diagonals and a partial positive
    # diagonal; the two edges form ONE matching ({0<->3, 1<->2}), plus the
    # diagonal slot: d = 2, d_padded = 2, h = 0.9, alpha = 1.8.
    matrix = np.zeros((4, 4))
    matrix[0, 3] = matrix[3, 0] = -0.7
    matrix[1, 2] = matrix[2, 1] = 0.5
    matrix[0, 0] = 0.3
    matrix[2, 2] = 0.9

    oracles = qrom_oracles(matrix, 6)
    assert oracles.d == 2
    assert oracles.h == pytest.approx(0.9)
    assert oracles.slot_flip == [0, 1]  # involutions: all self-paired

    encoding = SparseOracleEncoding(oracles, num_system=2)
    assert encoding.alpha == pytest.approx(1.8)

    ket = random_ket(4, seed=17)
    block = encoding.alpha * encoded_block(encoding, encoding.encode_kernel(),
                                           ket)
    quantized = oracle_quantized_dense(matrix, 0.9, 6)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)
    np.testing.assert_allclose(block,
                               matrix @ ket,
                               atol=oracle_quantization_atol(0.9, 6) * 2.0)


def test_qrom_oracles_validation_raises():
    asymmetric = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="not symmetric"):
        qrom_oracles(asymmetric, 4)
    with pytest.raises(ValueError, match="negative"):
        qrom_oracles(np.diag([-1.0, 0.5]), 4)
    with pytest.raises(ValueError, match="no nonzero"):
        qrom_oracles(np.zeros((2, 2)), 4)
    with pytest.raises(ValueError, match="value_bits"):
        qrom_oracles(np.eye(2), 0)
    with pytest.raises(ValueError, match="h="):
        qrom_oracles(np.eye(2), 4, h=0.5)


# ----------------------------------------------------------------------
# 3. Negative diagonals and non-symmetric input: raise, or dilate
# ----------------------------------------------------------------------


def test_negative_diagonal_raises_and_dilates_on_request():
    matrix = np.diag([-1.0, 0.5, 0.25, 0.75])
    with pytest.raises(ValueError, match="negative"):
        encode_sparse(matrix, value_bits=4)
    # The dilation puts every entry off the diagonal: [[0, A], [A^T, 0]]
    # with A = diag(-1, 0.5, 0.25, 0.75) is encodable, alpha = 2 * 1.
    encoding = encode_sparse(matrix, value_bits=6, dilate=True)
    assert encoding.report["dilated"] is True
    assert encoding.report["dim"] == 8 and encoding.num_system == 3
    assert encoding.report["alpha"] == pytest.approx(2.0)
    dilation = np.zeros((8, 8))
    dilation[0:4, 4:8] = matrix
    dilation[4:8, 0:4] = matrix.T
    ket = random_ket(8, seed=29)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    quantized = oracle_quantized_dense(dilation, 1.0, 6)
    np.testing.assert_allclose(encoding.alpha * block,
                               quantized @ ket,
                               atol=1e-10)


def test_nonsymmetric_input_dilates_once_at_the_data_level():
    # Dilated entries (exact binary values): (0,2)=0.5, (0,3)=1.0,
    # (1,3)=-0.5 and transposes: one matching, h = 1, alpha = 2.
    a = np.array([[0.5, 1.0], [0.0, -0.5]])
    encoding = encode_sparse(a, value_bits=6, dilate=True)
    report = encoding.report
    assert report["dilated"] is True
    assert report["dim"] == 4 and encoding.num_system == 2
    assert report["nnz"] == 6
    assert report["alpha"] == pytest.approx(2.0)
    assert report["qubits"] == total_qubits(encoding)

    dilation = np.zeros((4, 4))
    dilation[0:2, 2:4] = a
    dilation[2:4, 0:2] = a.T
    ket = random_ket(4, seed=31)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    quantized = oracle_quantized_dense(dilation, 1.0, 6)
    np.testing.assert_allclose(encoding.alpha * block,
                               quantized @ ket,
                               atol=1e-10)
    np.testing.assert_allclose(encoding.alpha * block,
                               dilation @ ket,
                               atol=oracle_quantization_atol(1.0, 6) * 2.0)


def test_nonsymmetric_input_raises_unless_dilate():
    # A genuinely asymmetric matrix and a symmetric matrix stored as one
    # triangle both raise by default, naming the asymmetry and the way
    # out; dilate=True opts in.
    with pytest.raises(ValueError, match="dilate=True"):
        encode_sparse(np.array([[0.5, 1.0], [0.0, -0.5]]))
    upper = ([0, 1, 2], [1, 2, 2], [0.5, -0.3, 0.8])
    with pytest.raises(ValueError, match=r"not symmetric at \(0, 1\)"):
        encode_sparse(upper, dim=3)
    noisy = np.array([[1.0, 0.5], [0.5 * (1 + 1e-10), 1.0]])
    with pytest.raises(ValueError, match="symmetrize"):
        encode_sparse(noisy)
    encoding = encode_sparse(upper, dim=3, value_bits=3, dilate=True)
    assert encoding.report["dilated"] is True and encoding.report["dim"] == 8


def test_uniformly_tiny_asymmetric_matrix_dilates():
    # A genuinely asymmetric matrix whose entries all sit below a fixed
    # 1e-12 floor must still be recognized as asymmetric (the symmetry
    # test is matrix-relative) and dilated, not silently symmetrized or
    # crashed on a missing mirror entry.
    for scale in (1e-13, 1.0, 1e6):
        a = np.array([[0.0, scale], [0.0, 0.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            encode_sparse(a, value_bits=4)
        encoding = encode_sparse(a, value_bits=4, dilate=True)
        assert encoding.report["dilated"] is True, scale
        assert encoding.report["dim"] == 4
        assert encoding.alpha == pytest.approx(2.0 * scale)


# ----------------------------------------------------------------------
# 4. Guards and argument validation
# ----------------------------------------------------------------------


def test_guard_over_budget_raises_actionably():
    rng = np.random.default_rng(7)
    dense = rng.normal(size=(8, 8))
    dense = dense + dense.T + 8.0 * np.eye(8)  # symmetric, positive diagonal
    # 8 slots x 8 rows of QROM entries: above a budget of 16.
    with pytest.raises(ValueError, match="hand-written oracles"):
        encode_sparse(dense, max_entries=16)
    # The budget is checked before any kernel is minted: a 64 x 64 dense
    # matrix is refused on the count alone.
    with pytest.raises(ValueError, match="max_entries=1"):
        encode_sparse(np.ones((64, 64)), max_entries=1)


def test_argument_validation_raises():
    with pytest.raises(ValueError, match="max_entries"):
        encode_sparse(np.eye(2), max_entries=0)
    with pytest.raises(ValueError, match="no nonzero"):
        encode_sparse(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="value_bits"):
        encode_sparse(np.eye(2), value_bits=0)
    with pytest.raises(ValueError, match="dim="):
        encode_sparse(np.eye(2), dim=3)
    with pytest.raises(ValueError, match="dim is required"):
        encode_sparse(([0], [0], [1.0]))
    # Finiteness is checked on both parts of an entry before the real
    # tolerance, and alpha = d_padded * h must be finite.
    with pytest.raises(ValueError, match="finite"):
        encode_sparse(np.array([[complex(1.0, float("nan"))]]))
    with pytest.raises(ValueError, match="finite"):
        qrom_oracles(np.array([[complex(1.0, float("inf"))]]), 4)
    with pytest.raises(ValueError, match="complex"):
        encode_sparse(np.array([[1.0 + 1.0j]]))
    with pytest.raises(ValueError, match="overflows"):
        encode_sparse(1e308 * np.eye(2))


def test_report_distinguishes_row_sparsity_from_slots():
    # The triangle K3 has two nonzeros per row but its edges need three
    # matchings, so the encoding pays alpha = d_padded * h = 4 on three
    # slots; the report names both numbers instead of one ambiguous "d".
    triangle = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    encoding = encode_sparse(triangle, value_bits=2)
    assert encoding.report["max_row_nnz"] == 2
    assert encoding.report["slots"] == 3
    assert encoding.d == 3
    assert encoding.alpha == pytest.approx(4.0)
    assert "max_row_nnz=2, slots=3" in repr(encoding)


def test_report_is_none_on_direct_constructions():
    oracle = SparseOracleEncoding(qrom_oracles(np.eye(2), 2), num_system=1)
    assert oracle.report is None
    assert "encode_sparse" not in repr(oracle)
