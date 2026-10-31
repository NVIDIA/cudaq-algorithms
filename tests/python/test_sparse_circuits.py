# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Circuit-level coverage of the sparse-access encoding, through CUDA-Q.

Every check here executes a kernel and reads the statevector; none relies
on construction metadata alone:

- QROM-backed oracles from data at every shape: 1x1, 2x2, diagonal-only,
  non-power-of-two 3x3 (padded rows and columns zero);
- the oracles themselves on basis states: locations and their inverses,
  value / sign / upper loads, filler and padded slots, clipped band
  boundaries, work qubits returned clean;
- the consumers on a QROM-backed encoding: controlled application, walk
  powers, QSVT, against independent dense references;
- reverse-slot branches: two nontrivial pairs, and the one-qubit slot
  register pair;
- the Hermitian dilation: padded slots, signed non-symmetric input,
  controlled application, walk and QSVT (the factory's data-level dilation
  is pinned in ``test_sparse_factory.py``);
- control coherence with a control in superposition;
- the input-width check of ``state_from``.

Register sizes stay at <= 3 system qubits (2 for QROM-backed cases, whose
work register adds 2n + d qubits) and 4 angle bits so the file runs in a
few minutes on qpp-cpu.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import QSVT, PhaseSequence, Walk
from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms.sparse import (SparseOracleEncoding, banded_oracles,
                                     encode_sparse, qrom_oracles)

VB = 4


def quantized_angle(value: float, h: float, value_bits: int) -> int:
    theta = math.asin(math.sqrt(min(1.0, abs(value) / h)))
    return min((1 << value_bits) - 1,
               int(round(theta / (0.5 * math.pi) * (1 << value_bits))))


def quantized_dense(matrix: np.ndarray, h: float,
                    value_bits: int) -> np.ndarray:
    """The exactly encoded matrix: fixed-point magnitudes, exact signs."""
    out = np.zeros_like(matrix, dtype=float)
    for i, j in zip(*np.nonzero(matrix)):
        a = quantized_angle(matrix[i, j], h, value_bits)
        magnitude = h * math.sin(a / (1 << value_bits) * 0.5 * math.pi)**2
        out[i, j] = math.copysign(magnitude, matrix[i, j])
    return out


def padded(matrix: np.ndarray, num_system: int) -> np.ndarray:
    dim = 1 << num_system
    out = np.zeros((dim, dim))
    out[:matrix.shape[0], :matrix.shape[1]] = matrix
    return out


def random_ket(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


def block_of(encoding, kernel, ket: np.ndarray) -> np.ndarray:
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << encoding.num_system]


def encoded_matrix(encoding) -> np.ndarray:
    dim = 1 << encoding.num_system
    kernel = encoding.encode_kernel()
    block = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        column = np.zeros(dim, dtype=np.complex128)
        column[j] = 1.0
        block[:, j] = block_of(encoding, kernel, column)
    return block


def basis_registers(state, widths):
    """The register contents of a computational basis state (one amplitude
    of modulus 1), split little-endian per register in allocation order."""
    amplitudes = np.asarray(state)
    index = int(np.argmax(np.abs(amplitudes)))
    assert abs(abs(amplitudes[index]) - 1.0) < 1e-9, "not a basis state"
    values = []
    for width in widths:
        values.append(index & ((1 << width) - 1))
        index >>= width
    return values


# ----------------------------------------------------------------------
# 1. QROM-backed oracles at every shape
# ----------------------------------------------------------------------

SHAPES = {
    "1x1": np.array([[0.7]]),
    "2x2": np.array([[0.2, -0.6], [-0.6, 0.0]]),
    "diag4": np.diag([0.3, 0.0, 0.9, 0.5]),
    "3x3": np.array([[0.0, 0.4, 0.0], [0.4, 0.5, -0.3], [0.0, -0.3, 0.0]]),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_qrom_oracles_encode_every_shape(name):
    matrix = SHAPES[name]
    size = matrix.shape[0]
    num_system = max(1, (size - 1).bit_length())
    oracles = qrom_oracles(matrix, VB)
    encoding = SparseOracleEncoding(oracles, num_system=num_system)
    h = np.abs(matrix).max()
    expected = padded(quantized_dense(matrix, h, VB),
                      num_system) / encoding.alpha
    dim = 1 << num_system
    block = encoded_matrix(encoding)
    np.testing.assert_allclose(block, expected, atol=1e-10)
    # Padded rows and columns are zero.
    assert np.all(np.abs(block[size:dim, :]) < 1e-10)
    assert np.all(np.abs(block[:, size:dim]) < 1e-10)
    via_factory = encode_sparse(matrix, value_bits=VB)
    assert via_factory.alpha == pytest.approx(encoding.alpha)
    assert via_factory.report["dim"] == size


# ----------------------------------------------------------------------
# 2. The oracles on basis states
# ----------------------------------------------------------------------

# Greedy matchings in sorted edge order: slot 0 = {0<->1} (rows 2 and 3
# are fillers), slot 1 = {0<->3, 1<->2}, slot 2 = the diagonal (H[0, 0]
# only; rows 1-3 are fillers): d = 3 slots, padded to 4.
PROBE = np.array([[0.5, 0.25, 0.0, -0.75], [0.25, 0.0, 0.5, 0.0],
                  [0.0, 0.5, 0.0, 0.0], [-0.75, 0.0, 0.0, 0.0]])


def _probe_oracles():
    return qrom_oracles(PROBE, VB)


def _oracle_probe(oracles, num_system, slot, row, which):
    """Prepare |slot>|row>, apply ``which`` oracle(s), return the registers
    (slot, system, value_and_sign, work)."""
    m = max(1, (oracles.d - 1).bit_length())
    vb = oracles.value_bits
    num_work = max(1, oracles.num_work)
    o_loc, o_loc_adj, o_val = oracles.o_loc, oracles.o_loc_adj, oracles.o_val

    @cudaq.kernel
    def probe():
        s = cudaq.qvector(m)
        x_reg = cudaq.qvector(num_system)
        value = cudaq.qvector(vb + 2)
        work = cudaq.qvector(num_work)
        for b in range(m):
            if ((slot >> b) & 1) == 1:
                x(s[b])
        for k in range(num_system):
            if ((row >> k) & 1) == 1:
                x(x_reg[k])
        if which == 0:
            o_loc(s, x_reg, work)
        elif which == 1:
            o_loc(s, x_reg, work)
            o_loc_adj(s, x_reg, work)
        else:
            o_val(s, x_reg, value, work)

    state = cudaq.get_state(probe)
    return basis_registers(state, [m, num_system, vb + 2, num_work])


def test_qrom_location_oracle_maps_rows_and_inverts():
    oracles = _probe_oracles()
    assert oracles.d == 3
    columns = {
        0: {
            0: 1,
            1: 0,
            2: 2,
            3: 3
        },
        1: {
            0: 3,
            3: 0,
            1: 2,
            2: 1
        },
        2: {
            0: 0,
            1: 1,
            2: 2,
            3: 3
        },
        3: {
            0: 0,
            1: 1,
            2: 2,
            3: 3
        }
    }
    for slot in range(4):  # padded slot 3 is the identity
        for row in range(4):
            s, x_out, value, work = _oracle_probe(oracles, 2, slot, row, 0)
            assert (s, x_out, value, work) == (slot, columns[slot][row], 0,
                                               0), (slot, row)
            s, x_out, value, work = _oracle_probe(oracles, 2, slot, row, 1)
            assert (s, x_out, value, work) == (slot, row, 0, 0), (slot, row)


def test_qrom_value_oracle_loads_assigned_entries_only():
    oracles = _probe_oracles()
    h = 0.75
    vb = oracles.value_bits
    # (slot, row) -> (column) for the assigned entries; fillers load zero.
    assigned = {
        (0, 0): 1,
        (0, 1): 0,
        (1, 0): 3,
        (1, 3): 0,
        (1, 1): 2,
        (1, 2): 1,
        (2, 0): 0
    }
    for slot in range(4):
        for row in range(4):
            s, x_out, value, work = _oracle_probe(oracles, 2, slot, row, 2)
            assert (s, x_out, work) == (slot, row, 0)
            if (slot, row) in assigned:
                column = assigned[(slot, row)]
                entry = PROBE[row, column]
                word = quantized_angle(entry, h, vb)
                if entry < 0.0:
                    word |= 1 << vb
                if row < column:
                    word |= 1 << (vb + 1)
                assert value == word, (slot, row)
            else:
                assert value == 0, (slot, row)


def test_banded_oracles_clip_at_the_boundary_and_skip_padded_slots():
    # Bands +1 / -1 on 3 qubits: slot 0 (offset +1) is valid for rows
    # 1..7, slot 1 (offset -1) for rows 0..6; the padded slots (d = 2 is
    # already a power of two here, so take d = 3 with the diagonal) load
    # nothing and leave the system alone.
    oracles = banded_oracles([1, -1, 0], [0.5, 0.5, 0.25],
                             num_system=3,
                             value_bits=VB)
    assert oracles.d == 3
    vb = oracles.value_bits
    angle_half = quantized_angle(0.5, 0.5, vb)
    angle_quarter = quantized_angle(0.25, 0.5, vb)
    for row in range(8):
        # slot 0: offset +1, H[row, row - 1] valid when row >= 1; upper = 0
        s, x_out, value, work = _oracle_probe(oracles, 3, 0, row, 2)
        assert (s, x_out, work) == (0, row, 0)
        assert value == (angle_half if row >= 1 else 0), row
        # slot 1: offset -1, H[row, row + 1] valid when row <= 6; upper = 1
        s, x_out, value, work = _oracle_probe(oracles, 3, 1, row, 2)
        assert value == ((angle_half | (1 << (vb + 1))) if row <= 6 else 0)
        # slot 2: the diagonal, always valid, no sign, no upper
        s, x_out, value, work = _oracle_probe(oracles, 3, 2, row, 2)
        assert value == angle_quarter
        # slot 3: padded, identity and no load
        s, x_out, value, work = _oracle_probe(oracles, 3, 3, row, 0)
        assert (x_out, value, work) == (row, 0, 0)
        s, x_out, value, work = _oracle_probe(oracles, 3, 3, row, 2)
        assert (x_out, value, work) == (row, 0, 0)
    # The location oracle shifts mod 2^n and inverts.
    for row in range(8):
        assert _oracle_probe(oracles, 3, 0, row, 0)[1] == (row + 1) % 8
        assert _oracle_probe(oracles, 3, 1, row, 0)[1] == (row - 1) % 8
        assert _oracle_probe(oracles, 3, 0, row, 1)[1] == row


# ----------------------------------------------------------------------
# 3. Consumers on a QROM-backed encoding
# ----------------------------------------------------------------------

# One matching {0<->3, 1<->2} with signed values: d = 1, the cheapest
# QROM-backed register, for the consumer checks.
CONSUMER = np.array([[0.0, 0.0, 0.0, -0.8], [0.0, 0.0, 0.5, 0.0],
                     [0.0, 0.5, 0.0, 0.0], [-0.8, 0.0, 0.0, 0.0]])


def _qrom_case():
    oracles = qrom_oracles(CONSUMER, VB)
    encoding = SparseOracleEncoding(oracles, num_system=2)
    assert encoding.d == 1
    scaled = quantized_dense(CONSUMER, 0.8, VB) / encoding.alpha
    return encoding, scaled


def _controlled_circuit(encoding, control_value):
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


def test_qrom_encoding_controlled_apply_conventions():
    encoding, scaled = _qrom_case()
    ket = random_ket(4, 3)
    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    expected = np.zeros_like(at_zero)
    expected[:4] = ket
    np.testing.assert_allclose(at_zero, expected, atol=1e-10)
    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:4], scaled @ ket, atol=1e-10)


def test_qrom_encoding_walk_powers_are_chebyshev():
    encoding, scaled = _qrom_case()
    ket = random_ket(4, 5)
    walk = Walk(encoding)
    t1 = -scaled
    t2 = 2.0 * scaled @ scaled - np.eye(4)
    for power, reference in ((1, t1), (2, t2)):
        block = block_of(encoding, walk.kernel(power=power), ket)
        np.testing.assert_allclose(block, reference @ ket, atol=1e-10)


def test_qrom_encoding_qsvt_matches_signal_model():
    from test_qsvt import reference_response

    encoding, scaled = _qrom_case()
    ket = random_ket(4, 7)
    sequence = PhaseSequence([0.31, -0.2, 0.47])
    block = block_of(encoding, QSVT(encoding).kernel(sequence), ket)
    eigenvalues, vectors = np.linalg.eigh(scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


# ----------------------------------------------------------------------
# 4. Reverse-slot branches
# ----------------------------------------------------------------------


def _banded_case(offsets, values, num_system):
    oracles = banded_oracles(offsets, values, num_system, VB)
    encoding = SparseOracleEncoding(oracles, num_system=num_system)
    dim = 1 << num_system
    ideal = np.zeros((dim, dim))
    for o, v in zip(offsets, values):
        for j in range(dim):
            if 0 <= j + o < dim:
                ideal[j + o, j] = v
    return encoding, quantized_dense(ideal, oracles.h, VB) / encoding.alpha


def test_two_nontrivial_reverse_slot_pairs():
    # Offsets +-1 and +-2: two bit-0-adjacent pairs, no diagonal, d = 4.
    encoding, scaled = _banded_case([1, -1, 2, -2], [0.5, 0.5, -0.25, -0.25],
                                    3)
    assert encoding.oracles.slot_flip == [1, 0, 3, 2]
    ket = random_ket(8, 9)
    np.testing.assert_allclose(block_of(encoding, encoding.encode_kernel(),
                                        ket),
                               scaled @ ket,
                               atol=1e-10)


def test_one_qubit_slot_register_pair():
    # Offsets +-1 alone: d = 2, a single slot qubit, and the reverse-slot
    # flip is a plain X on it.
    encoding, scaled = _banded_case([1, -1], [0.4, 0.4], 2)
    assert encoding.d_padded == 2 and encoding.oracles.slot_flip == [1, 0]
    np.testing.assert_allclose(encoded_matrix(encoding), scaled, atol=1e-10)
    block2 = block_of(encoding, encoding.walk_kernel(2), random_ket(4, 11))
    np.testing.assert_allclose(
        block2, (2.0 * scaled @ scaled - np.eye(4)) @ random_ket(4, 11),
        atol=1e-10)


# ----------------------------------------------------------------------
# 5. The Hermitian dilation
# ----------------------------------------------------------------------


def _dilation_case():
    # A: three bands (d = 3, padded to 4), signed, with a negative diagonal
    # and no symmetry; 2 system qubits, so the dilation acts on 3.
    offsets = [0, 1, -1]
    values = [-0.6, 0.45, -0.3]
    oracles = banded_oracles(offsets,
                             values,
                             num_system=2,
                             value_bits=VB,
                             hermitian=False)
    encoding = SparseOracleEncoding.from_general_oracles(oracles, num_system=2)
    a = np.zeros((4, 4))
    for o, v in zip(offsets, values):
        for j in range(4):
            if 0 <= j + o < 4:
                a[j + o, j] = v
    a_q = quantized_dense(a, oracles.h, VB)
    zero = np.zeros((4, 4))
    scaled = np.block([[zero, a_q], [a_q.T, zero]]) / encoding.alpha
    return encoding, scaled, a


def test_dilation_with_padded_slots_and_signed_input():
    encoding, scaled, _ = _dilation_case()
    assert encoding.d == 3 and encoding.d_padded == 4
    assert encoding.num_system == 3
    ket = random_ket(8, 13)
    np.testing.assert_allclose(block_of(encoding, encoding.encode_kernel(),
                                        ket),
                               scaled @ ket,
                               atol=1e-10)


def test_dilation_controlled_walk_and_qsvt():
    from test_qsvt import reference_response

    encoding, scaled, _ = _dilation_case()
    ket = random_ket(8, 17)
    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    expected = np.zeros_like(at_zero)
    expected[:8] = ket
    np.testing.assert_allclose(at_zero, expected, atol=1e-10)
    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:8], scaled @ ket, atol=1e-10)
    block2 = block_of(encoding, Walk(encoding).kernel(power=2), ket)
    np.testing.assert_allclose(block2,
                               (2.0 * scaled @ scaled - np.eye(8)) @ ket,
                               atol=1e-10)
    sequence = PhaseSequence([0.15, 0.6, -0.35])
    block = block_of(encoding, QSVT(encoding).kernel(sequence), ket)
    eigenvalues, vectors = np.linalg.eigh(scaled)
    coefficients = vectors.conj().T @ ket
    reference = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, reference, atol=1e-10)


# ----------------------------------------------------------------------
# 6. Control coherence
# ----------------------------------------------------------------------


def test_controlled_apply_with_a_superposed_control():
    encoding, _ = _qrom_case()
    controlled = encoding.controlled_apply_kernel()
    plain = encoding.apply_kernel()
    n_anc = encoding.num_ancilla
    ket = random_ket(4, 23)

    @cudaq.kernel
    def superposed(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        h(control_and_ancilla[0])
        controlled(control_and_ancilla, system)

    @cudaq.kernel
    def reference(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        plain(ancilla, system)

    out = np.array(cudaq.get_state(superposed, state_from(ket)))
    full = np.array(cudaq.get_state(reference, state_from(ket)))
    # Layout [system][control][ancilla], little-endian: the control is
    # qubit index num_system. Split the output on it.
    dim_s = 1 << encoding.num_system
    out = out.reshape(1 << n_anc, 2, dim_s)
    identity_branch = np.zeros((1 << n_anc, dim_s), dtype=np.complex128)
    identity_branch[0] = ket
    np.testing.assert_allclose(out[:, 0, :],
                               identity_branch / math.sqrt(2.0),
                               atol=1e-10)
    np.testing.assert_allclose(out[:, 1, :],
                               full.reshape(1 << n_anc, dim_s) /
                               math.sqrt(2.0),
                               atol=1e-10)


# ----------------------------------------------------------------------
# 7. Input width
# ----------------------------------------------------------------------


def test_state_from_checks_the_register_width():
    encoding, _ = _qrom_case()
    good = state_from(random_ket(4, 29), num_qubits=encoding.num_system)
    assert np.array(cudaq.get_state(
        encoding.encode_kernel(),
        good)).size == 1 << (encoding.num_system + encoding.num_ancilla)
    for bad in (np.array([1.0, 0.0]), np.eye(8)[0], np.array([1.0, 0.0, 0.0])):
        with pytest.raises(ValueError, match="amplitudes"):
            state_from(bad, num_qubits=encoding.num_system)
    with pytest.raises(ValueError, match="amplitudes"):
        state_from(np.eye(2), num_qubits=2)
