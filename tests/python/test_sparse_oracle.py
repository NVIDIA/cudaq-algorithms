# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SparseOracleEncoding: correctness against independent dense references.

Two-level validation per the library's validation culture:

- the circuit is compared against the *exactly encoded* (fixed-point
  quantized) dense matrix at simulator precision (1e-10) — this catches
  circuit/convention bugs undiluted by quantization;
- the quantized matrix is compared against the ideal one within the
  value_bits-derived bound ``h * (pi/2) * 2^-value_bits`` per element
  (|d sin^2(theta)/d theta| <= 1 and the fixed-point angle error is at
  most (pi/2) * 2^-value_bits including the top-of-range clamp).

Walk even moments through ``Walk.moment`` are exercised nowhere: the
reflection observable expands to 2^num_ancilla Pauli terms, which is
LCU-sized bookkeeping, not sparse-oracle-sized (this encoding carries a
dual system register in its ancilla). The walk circuits themselves are
pinned densely via ``walk_kernel`` powers instead.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import BlockEncoding, PhaseSequence, QSVT, Walk
from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms.sparse import (OracleKernels, SparseOracleEncoding,
                                     banded_oracles)
from cudaq_algorithms.sparse._banded import quantized_angle

# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------


def banded_dense(offsets, values, num_system) -> np.ndarray:
    """The ideal banded matrix H[j + o, j] = v_o (no wrap-around)."""
    dim = 1 << num_system
    matrix = np.zeros((dim, dim))
    for o, v in zip(offsets, values):
        for j in range(dim):
            if 0 <= j + o < dim:
                matrix[j + o, j] = v
    return matrix


def banded_quantized_dense(offsets, values, num_system, value_bits,
                           h) -> np.ndarray:
    """The exactly encoded matrix: fixed-point magnitudes, exact signs."""
    quantized = []
    for v in values:
        a = quantized_angle(v, h, value_bits)
        magnitude = h * math.sin(a / (1 << value_bits) * 0.5 * math.pi)**2
        quantized.append(math.copysign(magnitude, v) if v != 0.0 else 0.0)
    return banded_dense(offsets, quantized, num_system)


def quantization_atol(h: float, value_bits: int) -> float:
    """Per-element bound |H_quantized - H| (see module docstring)."""
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


LAPLACIAN_OFFSETS = [-1, 0, 1]
LAPLACIAN_VALUES = [-1.0, 2.0, -1.0]


def laplacian_encoding(num_system: int,
                       value_bits: int) -> SparseOracleEncoding:
    oracles = banded_oracles(LAPLACIAN_OFFSETS,
                             LAPLACIAN_VALUES,
                             num_system=num_system,
                             value_bits=value_bits)
    return SparseOracleEncoding(oracles, num_system=num_system)


# ----------------------------------------------------------------------
# 1. Tridiagonal Laplacian, column-by-column and dense-action
# ----------------------------------------------------------------------


@pytest.mark.parametrize("num_system,value_bits", [(2, 6), (3, 6)])
def test_laplacian_block_column_by_column(num_system, value_bits):
    encoding = laplacian_encoding(num_system, value_bits)
    block = encoded_matrix(encoding) * encoding.alpha
    quantized = banded_quantized_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES,
                                       num_system, value_bits, encoding.h)
    ideal = banded_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES, num_system)
    np.testing.assert_allclose(block, quantized, atol=1e-10)
    np.testing.assert_allclose(block,
                               ideal,
                               atol=quantization_atol(encoding.h, value_bits))


@pytest.mark.parametrize("num_system,value_bits", [(4, 4), (5, 3)])
def test_laplacian_block_dense_action(num_system, value_bits):
    # Larger registers: one application to a dense random ket exercises
    # every column in superposition at a fraction of the simulation cost.
    encoding = laplacian_encoding(num_system, value_bits)
    ket = random_ket(1 << num_system, seed=7)
    block = encoded_block(encoding, encoding.encode_kernel(),
                          ket) * encoding.alpha
    quantized = banded_quantized_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES,
                                       num_system, value_bits, encoding.h)
    ideal = banded_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES, num_system)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)
    # The action error compounds at most sqrt(dim) column errors.
    bound = quantization_atol(encoding.h, value_bits)
    np.testing.assert_allclose(block,
                               ideal @ ket,
                               atol=bound * math.sqrt(1 << num_system))


def test_high_precision_quantization():
    # One slow, tightened-tolerance run: value_bits=12 shrinks the
    # derived per-element bound to ~2.7e-4 * h.
    offsets, values = [1, -1], [0.7, 0.7]
    oracles = banded_oracles(offsets, values, num_system=2, value_bits=12)
    encoding = SparseOracleEncoding(oracles, num_system=2)
    ket = random_ket(4, seed=3)
    block = encoded_block(encoding, encoding.encode_kernel(),
                          ket) * encoding.alpha
    quantized = banded_quantized_dense(offsets, values, 2, 12, encoding.h)
    ideal = banded_dense(offsets, values, 2)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)
    assert quantization_atol(encoding.h, 12) < 3e-4
    np.testing.assert_allclose(block,
                               ideal @ ket,
                               atol=2.0 * quantization_atol(encoding.h, 12))


# ----------------------------------------------------------------------
# 2. Alpha bookkeeping (honest power-of-two padding)
# ----------------------------------------------------------------------


def test_alpha_reports_padded_sparsity():
    encoding = laplacian_encoding(3, 6)
    assert encoding.d == 3
    assert encoding.d_padded == 4
    assert encoding.h == pytest.approx(2.0)  # defaults to max |band value|
    assert encoding.alpha == pytest.approx(4 * encoding.h)
    assert (encoding.num_ancilla == encoding.num_block_ancilla +
            encoding.num_scratch)

    explicit_h = SparseOracleEncoding(banded_oracles(LAPLACIAN_OFFSETS,
                                                     LAPLACIAN_VALUES,
                                                     num_system=3,
                                                     value_bits=6,
                                                     h=2.5),
                                      num_system=3)
    assert explicit_h.alpha == pytest.approx(4 * 2.5)


# ----------------------------------------------------------------------
# 3. 1-sparse permutation via hand-written oracles
# ----------------------------------------------------------------------
#
# H is the 4x4 bit-reversal permutation (an involution, hence symmetric):
# o_loc flips both system qubits for slot 0 and idles the padding slot.
# All values are +1 = h, so the sign and upper bits stay untouched (the
# sign phases are inert at sign |0>).

_PERM_VALUE_BITS = 6


@cudaq.kernel
def _perm_o_loc(slot: cudaq.qview, system: cudaq.qview, work: cudaq.qview):
    x(slot[0])
    for k in range(system.size()):
        x.ctrl(slot, system[k])
    x(slot[0])


@cudaq.kernel
def _perm_o_val(slot: cudaq.qview, system: cudaq.qview,
                value_and_sign: cudaq.qview, work: cudaq.qview):
    # |H| = h everywhere on the permutation support: the clamped
    # fixed-point angle is all ones.
    x(slot[0])
    for k in range(_PERM_VALUE_BITS):
        x.ctrl(slot, value_and_sign[k])
    x(slot[0])


def permutation_encoding() -> SparseOracleEncoding:
    oracles = OracleKernels(o_loc=_perm_o_loc,
                            o_loc_adj=_perm_o_loc,
                            o_val=_perm_o_val,
                            o_val_adj=_perm_o_val,
                            d=1,
                            h=1.0,
                            value_bits=_PERM_VALUE_BITS,
                            num_work=0,
                            slot_flip=[0, 1])
    return SparseOracleEncoding(oracles, num_system=2)


def _permutation_dense(quantized: bool) -> np.ndarray:
    matrix = np.zeros((4, 4))
    a = quantized_angle(1.0, 1.0, _PERM_VALUE_BITS)
    value = math.sin(a / (1 << _PERM_VALUE_BITS) * 0.5 * math.pi)**2 \
        if quantized else 1.0
    for j in range(4):
        matrix[j ^ 3, j] = value
    return matrix


def test_one_sparse_permutation_block():
    encoding = permutation_encoding()
    assert encoding.alpha == pytest.approx(2.0)  # d=1 pads to 2, h=1
    block = encoded_matrix(encoding) * encoding.alpha
    np.testing.assert_allclose(block, _permutation_dense(True), atol=1e-10)
    np.testing.assert_allclose(block,
                               _permutation_dense(False),
                               atol=quantization_atol(1.0, _PERM_VALUE_BITS))


# ----------------------------------------------------------------------
# 4. Hermitian dilation of a non-Hermitian A
# ----------------------------------------------------------------------


def test_dilation_of_general_banded_matrix():
    # A is banded but non-symmetric, with random signed values including a
    # *negative diagonal* — the case the direct symmetric encoding provably
    # cannot represent; the dilation moves it off-diagonal where signs are
    # faithful.
    rng = np.random.default_rng(28)
    offsets = [0, 1]
    values = [-float(rng.uniform(0.3, 1.0)), float(rng.uniform(-1.0, 1.0))]
    oracles = banded_oracles(offsets,
                             values,
                             num_system=2,
                             value_bits=6,
                             hermitian=False)
    assert oracles.slot_flip is None
    encoding = SparseOracleEncoding.from_general_oracles(oracles, num_system=2)
    assert encoding.num_system == 3
    assert encoding.alpha == pytest.approx(2 * max(abs(v) for v in values))

    a_quantized = banded_quantized_dense(offsets, values, 2, 6, oracles.h)
    a_ideal = banded_dense(offsets, values, 2)
    zero = np.zeros((4, 4))
    dilated_quantized = np.block([[zero, a_quantized], [a_quantized.T, zero]])
    dilated_ideal = np.block([[zero, a_ideal], [a_ideal.T, zero]])

    block = encoded_matrix(encoding) * encoding.alpha
    np.testing.assert_allclose(block, dilated_quantized, atol=1e-10)
    np.testing.assert_allclose(block,
                               dilated_ideal,
                               atol=quantization_atol(oracles.h, 6))


# ----------------------------------------------------------------------
# 5. Protocol conformance and the Walk / QSVT consumers
# ----------------------------------------------------------------------


def test_satisfies_block_encoding_protocol():
    encoding = permutation_encoding()
    assert isinstance(encoding, BlockEncoding)
    assert BlockEncoding not in type(encoding).__mro__
    assert encoding.num_ancilla >= 1


def test_apply_is_involution():
    # U_A = T-dagger S T is exactly self-adjoint and involutory; applying
    # it twice must restore the full register (not only the good block).
    encoding = permutation_encoding()
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


def test_apply_is_involution_on_a_dirty_ancilla():
    # The oracle construction's U_A = T-dagger S T is involutory *globally*
    # (a matrix identity U_A^2 = I on the whole register), not only on the
    # |0...0>-ancilla block. Load an arbitrary state over system AND ancilla
    # -- scratch qubits deliberately non-zero -- and check U_A^2 restores it.
    # (SparseLCUEncoding's involution holds only on the |0>-ancilla domain,
    # so the analogous LCU test would fail here -- this pins the difference.)
    encoding = permutation_encoding()
    apply_u = encoding.apply_kernel()
    n_sys = encoding.num_system
    n_tot = encoding.num_system + encoding.num_ancilla

    @cudaq.kernel
    def twice_full(state: cudaq.State):
        reg = cudaq.qvector(state)
        apply_u(reg[n_sys:n_tot], reg[0:n_sys])
        apply_u(reg[n_sys:n_tot], reg[0:n_sys])

    ket = random_ket(1 << n_tot, seed=23)
    out = np.array(cudaq.get_state(twice_full, state_from(ket)))
    np.testing.assert_allclose(out, ket, atol=1e-10)


def test_walk_kernel_applies_chebyshev_of_minus_h():
    encoding = laplacian_encoding(2, 6)
    scaled = banded_quantized_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES, 2, 6,
                                    encoding.h) / encoding.alpha
    ket = random_ket(4, seed=9)
    block1 = encoded_block(encoding, encoding.walk_kernel(1), ket)
    np.testing.assert_allclose(block1, -scaled @ ket, atol=1e-10)
    # Power 2 pins the involution-backed Chebyshev property T_2.
    block2 = encoded_block(encoding, encoding.walk_kernel(2), ket)
    t2 = 2.0 * scaled @ scaled - np.eye(4)
    np.testing.assert_allclose(block2, t2 @ ket, atol=1e-10)


def test_walk_roundtrip_through_consumer_is_identity():
    encoding = permutation_encoding()
    walk = Walk(encoding)
    ket = random_ket(4, seed=13)
    out = np.array(
        cudaq.get_state(walk.roundtrip_kernel(power=2), state_from(ket)))
    expected = np.zeros_like(out)
    expected[:4] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_walk_controlled_roundtrip_through_consumer():
    encoding = permutation_encoding()
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
    encoding = laplacian_encoding(2, 6)
    quantized = banded_quantized_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES, 2,
                                       6, encoding.h)
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

    encoding = permutation_encoding()
    h_scaled = _permutation_dense(True) / encoding.alpha
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
    encoding = permutation_encoding()
    with pytest.raises(NotImplementedError, match="select_observable"):
        encoding.select_observable()
    with pytest.raises(NotImplementedError):
        Walk(encoding).moment(random_ket(4), 1)


# ----------------------------------------------------------------------
# 6. Factory-boundary validation raises loudly
# ----------------------------------------------------------------------


def _valid_oracles(**overrides) -> OracleKernels:
    fields = dict(o_loc=_perm_o_loc,
                  o_loc_adj=_perm_o_loc,
                  o_val=_perm_o_val,
                  o_val_adj=_perm_o_val,
                  d=1,
                  h=1.0,
                  value_bits=_PERM_VALUE_BITS,
                  num_work=0,
                  slot_flip=[0, 1])
    fields.update(overrides)
    return OracleKernels(**fields)


def test_encoding_validation_raises():
    with pytest.raises(ValueError, match="d >= 1"):
        SparseOracleEncoding(_valid_oracles(d=0), num_system=2)
    with pytest.raises(ValueError, match="h > 0"):
        SparseOracleEncoding(_valid_oracles(h=0.0), num_system=2)
    with pytest.raises(ValueError, match="value_bits >= 1"):
        SparseOracleEncoding(_valid_oracles(value_bits=0), num_system=2)
    with pytest.raises(ValueError, match="num_system must be a positive"):
        SparseOracleEncoding(_valid_oracles(), num_system=0)
    with pytest.raises(ValueError, match="num_work"):
        SparseOracleEncoding(_valid_oracles(num_work=-1), num_system=2)
    with pytest.raises(ValueError, match="o_val_adj is required"):
        SparseOracleEncoding(_valid_oracles(o_val_adj=None), num_system=2)
    with pytest.raises(ValueError, match="from_general_oracles"):
        SparseOracleEncoding(_valid_oracles(slot_flip=None), num_system=2)
    with pytest.raises(ValueError, match="padded slot range"):
        SparseOracleEncoding(_valid_oracles(slot_flip=[0]), num_system=2)
    with pytest.raises(ValueError, match="involution"):
        SparseOracleEncoding(_valid_oracles(slot_flip=[1, 1]), num_system=2)
    with pytest.raises(ValueError, match="self-paired"):
        SparseOracleEncoding(_valid_oracles(slot_flip=[1, 0]), num_system=2)
    with pytest.raises(ValueError, match="bit-0 adjacent"):
        SparseOracleEncoding(_valid_oracles(d=3, slot_flip=[2, 1, 0, 3]),
                             num_system=2)


def test_banded_validation_raises():
    with pytest.raises(ValueError, match="same length"):
        banded_oracles([0, 1], [1.0], num_system=2, value_bits=4)
    with pytest.raises(ValueError, match="distinct"):
        banded_oracles([1, 1], [1.0, 1.0], num_system=2, value_bits=4)
    with pytest.raises(ValueError, match="band offsets"):
        banded_oracles([4], [1.0], num_system=2, value_bits=4, hermitian=False)
    with pytest.raises(ValueError, match="nothing to encode"):
        banded_oracles([0], [0.0], num_system=2, value_bits=4)
    with pytest.raises(ValueError, match="h >= max"):
        banded_oracles([0], [2.0], num_system=2, value_bits=4, h=1.0)
    with pytest.raises(ValueError, match="closed under negation"):
        banded_oracles([0, 1], [1.0, 0.5], num_system=2, value_bits=4)
    with pytest.raises(ValueError, match="symmetric values"):
        banded_oracles([-1, 1], [0.5, 0.7], num_system=2, value_bits=4)
    with pytest.raises(ValueError, match="negative diagonal"):
        banded_oracles([-1, 0, 1], [0.5, -1.0, 0.5],
                       num_system=2,
                       value_bits=4)
    with pytest.raises(ValueError, match="value_bits must be a positive"):
        banded_oracles([0], [1.0], num_system=2, value_bits=0)


def test_walk_kernel_state_prep_injection():
    # walk_kernel(1, state_prep=...) must equal -H/alpha applied to the
    # prep's state, matching the state-argument variant's convention.
    encoding = laplacian_encoding(2, 6)
    scaled = banded_quantized_dense(LAPLACIAN_OFFSETS, LAPLACIAN_VALUES, 2, 6,
                                    encoding.h) / encoding.alpha

    @cudaq.kernel
    def prep(qubits: cudaq.qview):
        ry(0.9, qubits[0])
        ry(0.3, qubits[1])

    q0 = np.array([np.cos(0.45), np.sin(0.45)])
    q1 = np.array([np.cos(0.15), np.sin(0.15)])
    ket = np.kron(q1, q0)  # little-endian: qubits[0] is the LSB

    out = np.array(cudaq.get_state(encoding.walk_kernel(1, state_prep=prep)))
    block = out[:1 << encoding.num_system]
    np.testing.assert_allclose(block, -scaled @ ket, atol=1e-10)
