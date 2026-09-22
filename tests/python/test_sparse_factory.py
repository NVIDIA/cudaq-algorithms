# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""encode_sparse: pricing, dispatch, and the QROM-backed from-data oracles.

The dispatch decisions this file pins:

- skewed weights (one big pair, many tiny) -> LCU (one-norm 1.15 beats
  the padded-sparsity 2.0);
- uniform banded with tiny d -> oracle (padded sparsity 4 beats the
  one-norm 7);
- negative diagonal -> oracle ineligible (recorded), LCU dispatched;
- non-symmetric -> dilated once at the data level, both paths priced on
  the dilation; the chosen tie-break case resolves by qubit count;
- ``prefer=`` overrides pricing in both directions and raises with the
  recorded reason when the preferred path is ineligible;
- above ``max_terms`` a path drops out; with no path left the factory
  raises pointing at hand-written oracles.

Circuit-level checks follow the two-level validation culture at
dim <= 8: the block against the *exactly encoded* matrix (alias tables /
fixed-point angles) at 1e-10, and that matrix against the ideal one
within the derived discretization/quantization bound.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms.sparse import (SparseLCUEncoding, SparseOracleEncoding,
                                     encode_sparse, qrom_oracles)
from cudaq_algorithms.sparse._banded import quantized_angle

# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------


def term_matrix(kind, i, j, dim) -> np.ndarray:
    """The dense LCU term unitaries, straight from their definitions."""
    if kind == "identity":
        return np.eye(dim)
    if kind == "reflection":  # Z_i = I - 2|i><i|
        matrix = np.eye(dim)
        matrix[i, i] = -1.0
        return matrix
    if kind == "transposition":  # T = move + (I - P)
        matrix = np.eye(dim)
        matrix[i, i] = matrix[j, j] = 0.0
        matrix[i, j] = matrix[j, i] = 1.0
        return matrix
    if kind == "reflected_transposition":  # T' = move - (I - P)
        matrix = -np.eye(dim)
        matrix[i, i] = matrix[j, j] = 0.0
        matrix[i, j] = matrix[j, i] = 1.0
        return matrix
    raise ValueError(f"unknown term kind {kind!r}")


def lcu_discretized_dense(encoding) -> np.ndarray:
    """The exactly encoded matrix of the LCU path: alias-table weights
    (integer-exact) times the dense term unitaries, padding bins as
    identity branches."""
    dim = 1 << encoding.num_system
    probabilities = encoding.preparation.table_probabilities
    total = np.zeros((dim, dim))
    for k, (kind, i, j, _, sign) in enumerate(encoding.terms):
        total += (encoding.alpha * probabilities[k] * sign *
                  term_matrix(kind, i, j, dim))
    for k in range(len(encoding.terms), len(probabilities)):
        total += encoding.alpha * probabilities[k] * np.eye(dim)
    return total


def lcu_discretization_atol(encoding) -> float:
    """Derived elementwise bound |H_discretized - H| for the LCU path."""
    return (encoding.alpha * encoding.preparation.num_bins *
            encoding.discretization_bound)


def oracle_quantized_dense(matrix: np.ndarray, h: float,
                           value_bits: int) -> np.ndarray:
    """The exactly encoded matrix of the oracle path: each element's
    magnitude replaced by ``h sin^2(theta_q)`` (both T factors load the
    same fixed-point angle), signs exact."""
    quantized = np.zeros_like(matrix)
    for i, j in zip(*np.nonzero(matrix)):
        angle = quantized_angle(matrix[i, j], h, value_bits)
        magnitude = h * math.sin(angle / (1 << value_bits) * 0.5 * math.pi)**2
        quantized[i, j] = math.copysign(magnitude, matrix[i, j])
    return quantized


def oracle_quantization_atol(h: float, value_bits: int) -> float:
    """Per-element bound |H_quantized - H| for the oracle path."""
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
    """One dominant pair and three tiny ones (all 1-sparse rows):
    one-norm 1.15 vs padded-sparsity alpha 2.0."""
    matrix = np.zeros((8, 8))
    for i, j, value in [(0, 1, 1.0), (2, 3, 0.05), (4, 5, 0.05), (6, 7, 0.05)]:
        matrix[i, j] = matrix[j, i] = value
    return matrix


def tridiagonal_matrix(dim: int) -> np.ndarray:
    """Uniform tridiagonal (diagonal and off-diagonals all 1.0):
    padded-sparsity alpha 4 vs one-norm dim + (dim - 1)."""
    return (np.eye(dim) + np.eye(dim, k=1) + np.eye(dim, k=-1))


def total_qubits(encoding) -> int:
    return encoding.num_system + encoding.num_ancilla


# ----------------------------------------------------------------------
# 1. Dispatch by alpha, blocks verified against dense references
# ----------------------------------------------------------------------


def test_dispatch_skewed_weights_picks_lcu():
    matrix = skewed_matrix()
    encoding = encode_sparse(matrix, mu=2)
    assert isinstance(encoding, SparseLCUEncoding)

    report = encoding.report
    assert report["path"] == "lcu"
    assert report["alpha"] == pytest.approx(1.15, abs=1e-12)
    assert report["alpha"] == pytest.approx(encoding.alpha, abs=0.0)
    assert report["alpha_alternatives"]["oracle"] == pytest.approx(2.0)
    assert report["nnz"] == 8
    assert report["d"] == 1  # every row is 1-sparse
    assert report["dilated"] is False
    assert report["ineligible"] == {"oracle": None, "lcu": None}
    assert report["qubits"]["lcu"] == total_qubits(encoding)

    ket = random_ket(8)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    quantized = lcu_discretized_dense(encoding)
    np.testing.assert_allclose(encoding.alpha * block,
                               quantized @ ket,
                               atol=1e-10)
    assert np.abs(quantized -
                  matrix).max() <= lcu_discretization_atol(encoding)


def test_dispatch_uniform_banded_picks_oracle():
    matrix = tridiagonal_matrix(4)
    encoding = encode_sparse(matrix, value_bits=4)
    assert isinstance(encoding, SparseOracleEncoding)

    report = encoding.report
    assert report["path"] == "oracle"
    # 2 greedy matchings + the diagonal slot, padded to 4 slots, h = 1.
    assert encoding.d == 3 and encoding.d_padded == 4
    assert report["alpha"] == pytest.approx(4.0)
    assert report["alpha"] == pytest.approx(encoding.alpha, abs=0.0)
    assert report["alpha_alternatives"]["lcu"] == pytest.approx(7.0)
    assert report["nnz"] == 10
    assert report["d"] == 3  # max nonzeros per row
    assert report["ineligible"] == {"oracle": None, "lcu": None}
    assert report["qubits"]["oracle"] == total_qubits(encoding)

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


# ----------------------------------------------------------------------
# 2. The from-data (QROM-backed) V1 adapter, standalone
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

    block = encoding.alpha * encoded_matrix(encoding)
    quantized = oracle_quantized_dense(matrix, 0.9, 6)
    np.testing.assert_allclose(block, quantized, atol=1e-10)
    np.testing.assert_allclose(block,
                               matrix,
                               atol=oracle_quantization_atol(0.9, 6))


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
# 3. Negative diagonal: oracle ineligible, LCU dispatched
# ----------------------------------------------------------------------


def test_negative_diagonal_reports_oracle_ineligible():
    matrix = np.diag([-1.0, 0.5, 0.25, 0.75])
    encoding = encode_sparse(matrix, mu=2)
    assert isinstance(encoding, SparseLCUEncoding)
    assert encoding.report["path"] == "lcu"
    assert "negative diagonal" in encoding.report["ineligible"]["oracle"]
    assert encoding.report["ineligible"]["lcu"] is None
    assert encoding.alpha == pytest.approx(2.5)

    with pytest.raises(ValueError, match="negative diagonal"):
        encode_sparse(matrix, mu=2, prefer="oracle")


# ----------------------------------------------------------------------
# 4. Non-symmetric input: one data-level dilation for both paths
# ----------------------------------------------------------------------


def test_nonsymmetric_input_dilates_both_paths():
    # Dilated entries (exact binary values): (0,2)=0.5, (0,3)=1.0,
    # (1,3)=-0.5 and transposes. Both paths price the SAME dilation:
    # oracle alpha = d_padded * h = 2 * 1.0, lcu alpha = 0.5 + 1.0 + 0.5
    # = 2.0 — an exact tie, resolved by qubit count in favor of LCU.
    a = np.array([[0.5, 1.0], [0.0, -0.5]])
    encoding = encode_sparse(a, mu=2)
    report = encoding.report

    assert report["dilated"] is True
    assert report["dim"] == 4 and encoding.num_system == 2
    assert report["nnz"] == 6
    assert report["alpha_alternatives"]["oracle"] == pytest.approx(2.0)
    assert report["alpha_alternatives"]["lcu"] == pytest.approx(2.0)
    assert report["ineligible"] == {"oracle": None, "lcu": None}
    # Exact alpha tie -> fewer qubits wins.
    assert report["qubits"]["lcu"] < report["qubits"]["oracle"]
    assert report["path"] == "lcu"
    assert isinstance(encoding, SparseLCUEncoding)

    # The encoded operator is the dilation [[0, A], [A^T, 0]].
    dilation = np.zeros((4, 4))
    dilation[0:2, 2:4] = a
    dilation[2:4, 0:2] = a.T
    ket = random_ket(4, seed=31)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    quantized = lcu_discretized_dense(encoding)
    np.testing.assert_allclose(encoding.alpha * block,
                               quantized @ ket,
                               atol=1e-10)
    assert np.abs(quantized -
                  dilation).max() <= lcu_discretization_atol(encoding)


# ----------------------------------------------------------------------
# 5. prefer= overrides and the repr surface
# ----------------------------------------------------------------------


def test_prefer_overrides_pricing_both_ways():
    # Force the oracle path where LCU is cheaper...
    forced_oracle = encode_sparse(skewed_matrix(),
                                  value_bits=4,
                                  prefer="oracle")
    assert isinstance(forced_oracle, SparseOracleEncoding)
    assert forced_oracle.report["path"] == "oracle"
    assert forced_oracle.report["alpha"] == pytest.approx(2.0)
    assert forced_oracle.report["alpha"] == pytest.approx(forced_oracle.alpha,
                                                          abs=0.0)
    assert forced_oracle.report["alpha_alternatives"]["lcu"] == \
        pytest.approx(1.15, abs=1e-12)

    # ... and the LCU path where the oracle is cheaper.
    forced_lcu = encode_sparse(tridiagonal_matrix(4), mu=2, prefer="lcu")
    assert isinstance(forced_lcu, SparseLCUEncoding)
    assert forced_lcu.report["path"] == "lcu"
    assert forced_lcu.report["alpha"] == pytest.approx(7.0)

    for encoding in (forced_oracle, forced_lcu):
        text = repr(encoding)
        assert "encode_sparse" in text
        assert f"path={encoding.report['path']!r}" in text
        assert f"nnz={encoding.report['nnz']}" in text


# ----------------------------------------------------------------------
# 6. Guards
# ----------------------------------------------------------------------


def test_guard_no_path_raises_actionably():
    rng = np.random.default_rng(7)
    dense = rng.normal(size=(8, 8))
    dense = dense + dense.T + 8.0 * np.eye(8)  # symmetric, positive diagonal
    # 72 LCU terms and 8 slots x 8 rows of QROM entries: neither fits.
    with pytest.raises(ValueError, match="hand-written oracles"):
        encode_sparse(dense, max_terms=16)


def test_guard_drops_one_path_and_reports_it():
    # Skewed pairs plus a full tiny diagonal: 24 LCU terms, but only
    # 2 slots x 8 rows = 16 QROM entries. max_terms=16 disqualifies the
    # LCU path alone, so the oracle path wins despite its worse alpha.
    matrix = skewed_matrix() + 0.01 * np.eye(8)
    encoding = encode_sparse(matrix, value_bits=4, max_terms=16)
    assert isinstance(encoding, SparseOracleEncoding)
    assert encoding.report["path"] == "oracle"
    assert "max_terms=16" in encoding.report["ineligible"]["lcu"]
    assert encoding.report["ineligible"]["oracle"] is None
    # The over-budget LCU path never has its register width priced (its
    # gate bodies are not built), so its qubit count is reported as None.
    assert encoding.report["qubits"]["lcu"] is None

    with pytest.raises(ValueError, match="max_terms=16"):
        encode_sparse(matrix, max_terms=16, prefer="lcu")


def test_over_budget_lcu_skips_building_gate_bodies():
    # An over-budget LCU term count must be rejected on the count alone,
    # without materializing every SELECT body (which is O(nnz) work and
    # memory the budget is meant to avoid).
    import cudaq_algorithms.sparse._sparse_lcu as lcu_mod

    built = 0
    original = lcu_mod._term_body

    def counting(*args, **kwargs):
        nonlocal built
        built += 1
        return original(*args, **kwargs)

    lcu_mod._term_body = counting
    try:
        with pytest.raises(ValueError, match="no generic sparse encoding"):
            encode_sparse(np.ones((64, 64)), max_terms=1)
    finally:
        lcu_mod._term_body = original
    assert built == 0


def test_uniformly_tiny_asymmetric_matrix_dilates():
    # A genuinely asymmetric matrix whose entries all sit below a fixed
    # 1e-12 floor must still be recognized as asymmetric (the symmetry
    # test is matrix-relative) and dilated, not silently symmetrized or
    # crashed on a missing mirror entry.
    scales = [1e-13, 1.0, 1e6]
    for scale in scales:
        a = np.array([[0.0, scale], [0.0, 0.0]])
        for prefer in ("lcu", "oracle"):
            encoding = encode_sparse(a, prefer=prefer, value_bits=4)
            assert encoding.report["dilated"] is True, (scale, prefer)
            assert encoding.report["dim"] == 4


def test_argument_validation_raises():
    with pytest.raises(ValueError, match="prefer"):
        encode_sparse(np.eye(2), prefer="both")
    with pytest.raises(ValueError, match="max_terms"):
        encode_sparse(np.eye(2), max_terms=0)
    with pytest.raises(ValueError, match="no nonzero"):
        encode_sparse(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="mu"):
        encode_sparse(np.eye(2), mu=0)
    with pytest.raises(ValueError, match="value_bits"):
        encode_sparse(np.eye(2), value_bits=0)
