# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SparseLCUEncoding: correctness against independent dense references.

Two-level validation per the library's validation culture:

- the circuit is compared against the *exactly encoded* matrix — the term
  unitaries built densely in NumPy, weighted by the alias tables'
  ``table_probabilities`` (integer-exact) — at simulator precision
  (1e-10), which catches circuit/convention bugs undiluted by
  discretization;
- the exactly encoded matrix is compared against the ideal one within the
  derived bound ``alpha * num_bins * discretization_bound`` (each of the
  ``num_bins`` alias bins carries at most ``discretization_bound`` of
  probability rounding, each term unitary has entries of modulus <= 1);
- the term enumeration itself (the factor-of-2 split of each off-diagonal
  pair into T/T' and each diagonal into I/Z_i, and the sign placement) is
  pinned *classically*: the ideal-weight term sum must reproduce H
  exactly, with no circuit in the loop.

``Walk.moment`` is not exercised for the same reason as the V1 tests: the
reflection observable expands to 2^num_ancilla Pauli terms. Walk circuits
are pinned densely via ``walk_kernel`` powers instead.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import BlockEncoding, PhaseSequence, QSVT, Walk
from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms.sparse import (OracleKernels, SparseLCUEncoding,
                                     SparseOracleEncoding)
from cudaq_algorithms.sparse._banded import quantized_angle

# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------


def term_matrix(kind, i, j, dim) -> np.ndarray:
    """The dense term unitaries, straight from their definitions."""
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


def ideal_term_sum(encoding) -> np.ndarray:
    """sum_k w_k s_k U_k with the *ideal* weights — must equal H exactly
    (the classical factor-of-2 / sign bookkeeping pin)."""
    dim = 1 << encoding.num_system
    total = np.zeros((dim, dim))
    for kind, i, j, weight, sign in encoding.terms:
        total += weight * sign * term_matrix(kind, i, j, dim)
    return total


def discretized_dense(encoding) -> np.ndarray:
    """The exactly encoded matrix: alias-table weights, dense unitaries.

    Padding bins (index >= len(terms)) act as the identity in SELECT; the
    integer Vose residual redistribution can in principle leave them a
    nonzero table probability, so they are included.
    """
    dim = 1 << encoding.num_system
    probabilities = encoding.preparation.table_probabilities
    total = np.zeros((dim, dim))
    for k, (kind, i, j, _, sign) in enumerate(encoding.terms):
        total += (encoding.alpha * probabilities[k] * sign *
                  term_matrix(kind, i, j, dim))
    for k in range(len(encoding.terms), len(probabilities)):
        total += encoding.alpha * probabilities[k] * np.eye(dim)
    return total


def discretization_atol(encoding) -> float:
    """Derived elementwise bound |H_discretized - H|."""
    return (encoding.alpha * encoding.preparation.num_bins *
            encoding.discretization_bound)


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


def dense_from_entries(entries: dict, dim: int) -> np.ndarray:
    matrix = np.zeros((dim, dim))
    for (i, j), value in entries.items():
        matrix[i, j] = value
    return matrix


# The small mixed-sign fixture shared by the consumer tests: one
# off-diagonal pair plus a *negative diagonal* (the case V1's symmetric
# assembler provably cannot represent).
SMALL_ENTRIES = {(0, 3): 0.7, (3, 0): 0.7, (1, 1): -0.45}


def small_encoding(mu: int = 1) -> SparseLCUEncoding:
    return SparseLCUEncoding(dense_from_entries(SMALL_ENTRIES, 4), mu=mu)


# ----------------------------------------------------------------------
# 1. Random sparse Hermitian matrices (mixed signs, negative diagonals)
# ----------------------------------------------------------------------

DIM8_ENTRIES = {
    (0, 5): -0.8,
    (5, 0): -0.8,
    (2, 3): 0.6,
    (3, 2): 0.6,
    (1, 1): -0.9,
    (6, 6): 0.35,
}

DIM16_ENTRIES = {
    (0, 9): 1.1,
    (9, 0): 1.1,
    (3, 12): -0.4,
    (12, 3): -0.4,
    (5, 5): -0.7,
}


def test_dim4_block_column_by_column():
    # Full column-by-column extraction (pins column/endianness
    # conventions) at the cheap register size; the larger matrices below
    # use dense random-ket action, which exercises every column in
    # superposition at a fraction of the simulation cost (the V1 tests'
    # pattern for wide registers).
    matrix = dense_from_entries(SMALL_ENTRIES, 4)
    encoding = SparseLCUEncoding(matrix, mu=2)
    np.testing.assert_allclose(ideal_term_sum(encoding), matrix, atol=1e-12)
    block = encoded_matrix(encoding) * encoding.alpha
    quantized = discretized_dense(encoding)
    np.testing.assert_allclose(block, quantized, atol=1e-10)
    np.testing.assert_allclose(block,
                               matrix,
                               atol=discretization_atol(encoding))


@pytest.mark.parametrize("entries,dim,num_system", [(DIM8_ENTRIES, 8, 3),
                                                    (DIM16_ENTRIES, 16, 4)])
def test_sparse_hermitian_block_dense_action(entries, dim, num_system):
    matrix = dense_from_entries(entries, dim)
    encoding = SparseLCUEncoding(matrix, mu=2)
    assert encoding.num_system == num_system
    # Classical pin: ideal weights reproduce H exactly (the factor-of-2
    # and sign bookkeeping, negative diagonals included).
    np.testing.assert_allclose(ideal_term_sum(encoding), matrix, atol=1e-12)
    ket = random_ket(dim, seed=7)
    block = encoded_block(encoding, encoding.encode_kernel(),
                          ket) * encoding.alpha
    quantized = discretized_dense(encoding)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)
    assert np.abs(quantized - matrix).max() <= discretization_atol(encoding)


# ----------------------------------------------------------------------
# 2. Alpha bookkeeping (the term one-norm, discretized == ideal)
# ----------------------------------------------------------------------


def test_alpha_is_the_term_one_norm():
    entries = {
        (0, 2): -0.6,
        (2, 0): -0.6,
        (1, 3): 0.4,
        (3, 1): 0.4,
        (0, 0): -0.5,
        (3, 3): 0.25,
    }
    matrix = dense_from_entries(entries, 4)
    encoding = SparseLCUEncoding(matrix, mu=4)

    # The defined one-norm: sum_{i<j} |H_ij| + sum_i |H_ii|.
    one_norm = 0.6 + 0.4 + 0.5 + 0.25
    assert encoding.alpha == pytest.approx(one_norm, abs=1e-12)
    # The discretized weights sum to alpha *exactly* (the alias tables
    # realize probabilities summing to one), so the discretized one-norm
    # is the ideal one identically.
    assert np.sum(encoding.discretized_weights) == pytest.approx(one_norm,
                                                                 abs=1e-12)
    # The T/T' split halves the naive elementwise bookkeeping of
    # 2|H_ij| per off-diagonal pair.
    naive = 2 * (0.6 + 0.4) + 0.5 + 0.25
    assert encoding.alpha < naive
    # Sanity: any block encoding must have alpha >= the spectral norm.
    assert encoding.alpha >= np.abs(np.linalg.eigvalsh(matrix)).max()
    # Two terms per nonzero upper-triangle entry, each at half weight.
    assert len(encoding.terms) == 8
    assert [t[3] for t in encoding.terms
            ] == pytest.approx([0.25, 0.25, 0.3, 0.3, 0.2, 0.2, 0.125, 0.125])
    np.testing.assert_allclose(ideal_term_sum(encoding), matrix, atol=1e-12)


# ----------------------------------------------------------------------
# 3. Headline: V2's one-norm alpha beats V1's d * max|H|
# ----------------------------------------------------------------------
#
# H is 1-sparse per row on 3 system qubits — pairs (0,1), (2,3), (4,5),
# (6,7) via the involution x -> x XOR 1 — with ONE big element (1.0 on
# the (0,1) pair) and many tiny ones (0.05 on the rest). V1 must pay
# alpha = d_padded * max|H| = 2.0; V2 pays the one-norm
# 1.0 + 3 * 0.05 = 1.15.

_HL_VALUE_BITS = 6
_HL_BIG = 1.0
_HL_TINY = 0.05
_HL_ANGLE_BIG = quantized_angle(_HL_BIG, 1.0, _HL_VALUE_BITS)
_HL_ANGLE_TINY = quantized_angle(_HL_TINY, 1.0, _HL_VALUE_BITS)
_HL_ANGLE_DIFF = _HL_ANGLE_BIG ^ _HL_ANGLE_TINY


@cudaq.kernel
def _hl_o_loc(slot: cudaq.qview, system: cudaq.qview, work: cudaq.qview):
    # c(0, x) = x XOR 1; the padding slot 1 idles.
    x(slot[0])
    x.ctrl(slot, system[0])
    x(slot[0])


@cudaq.kernel
def _hl_o_val(slot: cudaq.qview, system: cudaq.qview,
              value_and_sign: cudaq.qview, work: cudaq.qview):
    # XOR-load the row-dependent angle: rows 0 and 1 (system bits 1 and 2
    # both zero) carry the big element, every other row the tiny one. All
    # values are positive, so the sign/upper bits stay untouched.
    x(slot[0])
    for k in range(_HL_VALUE_BITS):
        if ((_HL_ANGLE_TINY >> k) & 1) == 1:
            x.ctrl(slot, value_and_sign[k])
    x(system[1])
    x(system[2])
    for k in range(_HL_VALUE_BITS):
        if ((_HL_ANGLE_DIFF >> k) & 1) == 1:
            x.ctrl(slot[0], system[1], system[2], value_and_sign[k])
    x(system[1])
    x(system[2])
    x(slot[0])


def _headline_dense(quantized: bool) -> np.ndarray:
    matrix = np.zeros((8, 8))
    for row in range(8):
        value = _HL_BIG if row < 2 else _HL_TINY
        if quantized:
            angle = _HL_ANGLE_BIG if row < 2 else _HL_ANGLE_TINY
            value = math.sin(angle / (1 << _HL_VALUE_BITS) * 0.5 * math.pi)**2
        matrix[row, row ^ 1] = value
    return matrix


def test_headline_v2_one_norm_beats_v1_sparsity_alpha():
    ideal = _headline_dense(False)
    ket = random_ket(8, seed=19)

    # V1: hand oracles (both XOR-load kernels are self-inverse).
    oracles = OracleKernels(o_loc=_hl_o_loc,
                            o_loc_adj=_hl_o_loc,
                            o_val=_hl_o_val,
                            o_val_adj=_hl_o_val,
                            d=1,
                            h=1.0,
                            value_bits=_HL_VALUE_BITS,
                            num_work=0,
                            slot_flip=[0, 1])
    v1 = SparseOracleEncoding(oracles, num_system=3)
    assert v1.alpha == pytest.approx(2.0)  # d_padded * h pays the max

    # V2: the term one-norm sees one big and three tiny pairs.
    v2 = SparseLCUEncoding(ideal, mu=3)
    assert v2.alpha == pytest.approx(_HL_BIG + 3 * _HL_TINY, abs=1e-12)
    assert v2.alpha < v1.alpha

    # Both encode H correctly, each within its own derived tolerance.
    v1_block = encoded_block(v1, v1.encode_kernel(), ket) * v1.alpha
    np.testing.assert_allclose(v1_block,
                               _headline_dense(True) @ ket,
                               atol=1e-10)
    v1_atol = 1.0 * 0.5 * math.pi * 2.0**(-_HL_VALUE_BITS)  # h (pi/2) 2^-vb
    np.testing.assert_allclose(v1_block,
                               ideal @ ket,
                               atol=v1_atol * math.sqrt(8))

    v2_block = encoded_block(v2, v2.encode_kernel(), ket) * v2.alpha
    v2_quantized = discretized_dense(v2)
    np.testing.assert_allclose(v2_block, v2_quantized @ ket, atol=1e-10)
    assert np.abs(v2_quantized - ideal).max() <= discretization_atol(v2)


# ----------------------------------------------------------------------
# 4. Hermitian dilation of non-symmetric data
# ----------------------------------------------------------------------


def test_from_general_dilation():
    # A is non-symmetric with signed values including a negative
    # *diagonal* entry — every element lands off-diagonal in the dilation.
    a_entries = {(0, 1): 0.8, (1, 2): -0.5, (2, 0): 0.3, (1, 1): -0.4}
    a_matrix = dense_from_entries(a_entries, 4)
    with pytest.raises(ValueError, match="from_general"):
        SparseLCUEncoding(a_matrix, mu=2)
    encoding = SparseLCUEncoding.from_general(a_matrix, mu=2)
    assert encoding.num_system == 3
    assert encoding.alpha == pytest.approx(0.8 + 0.5 + 0.3 + 0.4, abs=1e-12)

    zero = np.zeros((4, 4))
    dilated = np.block([[zero, a_matrix], [a_matrix.T, zero]])
    np.testing.assert_allclose(ideal_term_sum(encoding), dilated, atol=1e-12)
    ket = random_ket(8, seed=23)
    block = encoded_block(encoding, encoding.encode_kernel(),
                          ket) * encoding.alpha
    quantized = discretized_dense(encoding)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)
    assert np.abs(quantized - dilated).max() <= discretization_atol(encoding)


def test_scipy_sparse_input_matches_dense():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    matrix = dense_from_entries(SMALL_ENTRIES, 4)
    from_dense = SparseLCUEncoding(matrix, mu=2)
    from_sparse = SparseLCUEncoding(scipy_sparse.coo_matrix(matrix), mu=2)
    assert from_sparse.alpha == pytest.approx(from_dense.alpha, abs=1e-15)
    assert from_sparse.terms == from_dense.terms


def test_triples_input_matches_dense():
    rows, cols, vals = zip(*[(i, j, v) for (i, j), v in SMALL_ENTRIES.items()])
    from_triples = SparseLCUEncoding((list(rows), list(cols), list(vals)),
                                     dim=4,
                                     mu=2)
    from_dense = SparseLCUEncoding(dense_from_entries(SMALL_ENTRIES, 4), mu=2)
    assert from_triples.terms == from_dense.terms


# ----------------------------------------------------------------------
# 5. Protocol conformance and the Walk / QSVT consumers
# ----------------------------------------------------------------------


def test_satisfies_block_encoding_protocol():
    encoding = small_encoding()
    assert isinstance(encoding, BlockEncoding)
    assert BlockEncoding not in type(encoding).__mro__
    assert encoding.num_ancilla >= 1
    assert (encoding.num_ancilla == encoding.num_index + encoding.num_garbage +
            encoding.num_select_work)


def test_apply_is_involution():
    # Every term unitary is Hermitian and involutory, so SELECT^2 = I and
    # U_A = P-dagger S P is exactly self-adjoint and involutory — the
    # property QSVT relies on when it reuses apply_kernel for adjoint
    # directions. Applying twice must restore the full register.
    encoding = small_encoding()
    apply_u = encoding.apply_kernel()
    n_anc = encoding.num_ancilla

    @cudaq.kernel
    def twice(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        apply_u(ancilla, system)
        apply_u(ancilla, system)

    ket = random_ket(4, seed=5)
    out = np.array(cudaq.get_state(twice, state_from(ket)))
    expected = np.zeros_like(out)
    expected[:4] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_walk_kernel_applies_chebyshev_of_minus_h():
    encoding = small_encoding()
    scaled = discretized_dense(encoding) / encoding.alpha
    ket = random_ket(4, seed=9)
    block1 = encoded_block(encoding, encoding.walk_kernel(1), ket)
    np.testing.assert_allclose(block1, -scaled @ ket, atol=1e-10)
    # Power 2 pins the involution-backed Chebyshev property T_2.
    block2 = encoded_block(encoding, encoding.walk_kernel(2), ket)
    t2 = 2.0 * scaled @ scaled - np.eye(4)
    np.testing.assert_allclose(block2, t2 @ ket, atol=1e-10)


def test_walk_roundtrip_through_consumer_is_identity():
    encoding = small_encoding()
    walk = Walk(encoding)
    ket = random_ket(4, seed=13)
    out = np.array(
        cudaq.get_state(walk.roundtrip_kernel(power=2), state_from(ket)))
    expected = np.zeros_like(out)
    expected[:4] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_walk_controlled_roundtrip_through_consumer():
    encoding = small_encoding()
    ket = random_ket(4, seed=17)
    kernel = Walk(encoding).controlled_roundtrip_kernel(power=1)
    state = np.array(cudaq.get_state(kernel, state_from(ket)))
    # Control flipped on, roundtrip is the identity: back to
    # |control=1, ancilla=0...0> x |psi>.
    np.testing.assert_allclose(state[4:8], ket, atol=1e-10)


def _controlled_circuit(encoding, control_value: int):
    controlled = encoding.controlled_apply_kernel()
    n_anc = encoding.num_ancilla
    flip = control_value

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        if flip == 1:
            x(control_and_ancilla[0])
        controlled(control_and_ancilla, system)
        if flip == 1:
            x(control_and_ancilla[0])

    return circuit


def test_controlled_apply_control_conventions():
    encoding = small_encoding()
    quantized = discretized_dense(encoding)
    ket = random_ket(4, seed=21)

    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    identity_expected = np.zeros_like(at_zero)
    identity_expected[:4] = ket
    np.testing.assert_allclose(at_zero, identity_expected, atol=1e-10)

    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:4], (quantized @ ket) / encoding.alpha,
                               atol=1e-10)


def test_qsvt_sequence_matches_signal_model():
    from test_qsvt import reference_response

    encoding = small_encoding()
    h_scaled = discretized_dense(encoding) / encoding.alpha
    ket = random_ket(4, seed=23)

    sequence = PhaseSequence([0.23, -0.41, 0.11])
    kernel = QSVT(encoding).kernel(sequence)
    block = encoded_block(encoding, kernel, ket)

    eigenvalues, vectors = np.linalg.eigh(h_scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


def test_odd_moments_are_unavailable():
    encoding = small_encoding()
    with pytest.raises(NotImplementedError, match="select_observable"):
        encoding.select_observable()
    with pytest.raises(NotImplementedError):
        Walk(encoding).moment(random_ket(4), 1)


# ----------------------------------------------------------------------
# 6. Factory-boundary validation raises loudly
# ----------------------------------------------------------------------


def test_encoding_validation_raises():
    good = dense_from_entries(SMALL_ENTRIES, 4)
    with pytest.raises(ValueError, match="square"):
        SparseLCUEncoding(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="nothing to encode"):
        SparseLCUEncoding(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="not symmetric"):
        SparseLCUEncoding(np.array([[0.0, 1.0], [0.5, 0.0]]))
    with pytest.raises(ValueError, match="complex"):
        SparseLCUEncoding(np.array([[0.0, 1.0j], [-1.0j, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        SparseLCUEncoding(np.array([[np.inf, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="dim is required"):
        SparseLCUEncoding(([0], [1], [0.5]))
    with pytest.raises(ValueError, match="same length"):
        SparseLCUEncoding(([0, 1], [1], [0.5]), dim=2)
    with pytest.raises(ValueError, match="out of range"):
        SparseLCUEncoding(([0, 7], [7, 0], [0.5, 0.5]), dim=4)
    with pytest.raises(ValueError, match="must be integers"):
        SparseLCUEncoding(([0.5], [1], [0.5]), dim=2)
    with pytest.raises(ValueError, match="does not match"):
        SparseLCUEncoding(good, dim=8)
    with pytest.raises(ValueError, match="mu"):
        SparseLCUEncoding(good, mu=0)
    with pytest.raises(ValueError, match="nothing to encode"):
        SparseLCUEncoding.from_general(np.zeros((2, 2)))
    # Duplicate triples sum (COO semantics): cancelling duplicates leave
    # nothing to encode.
    with pytest.raises(ValueError, match="nothing to encode"):
        SparseLCUEncoding(([0, 0], [1, 1], [0.5, -0.5]), dim=2)


def test_wide_ancilla_even_moment_refuses_fast():
    # The even-moment reflection observable expands to 2^num_ancilla Pauli
    # terms; at this encoding's widths that would hang, not error. The
    # shared consumer must refuse immediately instead.
    from cudaq_algorithms import qubitization

    rng = np.random.default_rng(7)
    dense = np.zeros((16, 16))
    for i, j in ((0, 3), (2, 9), (5, 5), (11, 14)):
        dense[i, j] = dense[j, i] = rng.uniform(-1.0, 1.0)
    encoding = SparseLCUEncoding(dense)
    assert encoding.num_ancilla > 20  # the pathological regime is real
    with pytest.raises(ValueError, match="walk_kernel"):
        qubitization.reflection_observable(encoding)


def test_encode_kernel_state_prep_injection():
    # The state_prep variant (self-allocating kernel running a prep
    # callback) must act identically to the state-argument variant on the
    # same input state.
    entries = {(0, 1): 0.7, (1, 0): 0.7, (0, 0): -0.5, (2, 2): 0.3}
    matrix = dense_from_entries(entries, 4)
    encoding = SparseLCUEncoding(matrix, mu=3)

    @cudaq.kernel
    def prep(qubits: cudaq.qview):
        ry(0.7, qubits[0])
        ry(1.1, qubits[1])

    q0 = np.array([np.cos(0.35), np.sin(0.35)])
    q1 = np.array([np.cos(0.55), np.sin(0.55)])
    ket = np.kron(q1, q0)  # little-endian: qubits[0] is the LSB

    out = np.array(cudaq.get_state(encoding.encode_kernel(state_prep=prep)))
    block = out[:1 << encoding.num_system] * encoding.alpha
    quantized = discretized_dense(encoding)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)
