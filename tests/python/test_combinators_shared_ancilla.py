# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum encodings must satisfy the protocol on the entire shared ancilla pool."""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT, SumEncoding, Walk, state_from
from dense_references import random_ket
from test_combinators import _block, _chebyshev
from test_qsvt import reference_response


def _noncommuting_sum(sign):
    return SumEncoding([(1.0, PauliLCU({"X": 1.0})),
                        (sign * 0.7, PauliLCU({"Y": 1.0})),
                        (0.5, PauliLCU({"Z": 1.0}))])


def _apply_to_joint_state(encoding):
    apply = encoding.apply_kernel()
    n = encoding.num_system

    @cudaq.kernel
    def circuit(state: cudaq.State):
        qubits = cudaq.qvector(state)
        apply(qubits[n:], qubits[:n])

    return circuit


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sum_controlled_apply_disabled_preserves_entangled_ancillas(sign):
    # The one-term case also has an unused slot. The larger operand leaves
    # its pool entangled with the system, as happens in nested sums.
    encoding = SumEncoding([(sign, PauliLCU({"I": 1.0, "X": 0.7, "Z": -0.4}))])
    controlled = encoding.controlled_apply_kernel()
    n = encoding.num_system
    dim_s, dim_a = 1 << n, 1 << encoding.num_ancilla
    joint = random_ket(n + encoding.num_ancilla, seed=31)
    expected = np.zeros((dim_a, 2, dim_s), dtype=np.complex128)
    expected[:, 0, :] = joint.reshape(dim_a, dim_s)
    expected = expected.reshape(-1)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        qubits = cudaq.qvector(state)
        controlled(qubits[n:], qubits[:n])

    out = np.asarray(cudaq.get_state(circuit, state_from(expected)))
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sum_coherent_control_is_exact_controlled_unitary(sign):
    encoding = _noncommuting_sum(sign)
    controlled = encoding.controlled_apply_kernel()
    n = encoding.num_system
    dim_s, dim_a = 1 << n, 1 << encoding.num_ancilla
    # Entangle the control with arbitrary flag/pool/system amplitudes. A
    # relative phase between the two control branches must remain visible.
    joint = random_ket(n + 1 + encoding.num_ancilla, seed=34)
    expected = joint.reshape(dim_a, 2, dim_s).copy()
    on = expected[:, 1, :].reshape(-1)
    on_norm = np.linalg.norm(on)
    on_out = np.asarray(
        cudaq.get_state(_apply_to_joint_state(encoding),
                        state_from(on / on_norm)))
    expected[:, 1, :] = (on_norm * on_out).reshape(dim_a, dim_s)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        qubits = cudaq.qvector(state)
        controlled(qubits[n:], qubits[:n])

    out = np.asarray(cudaq.get_state(circuit, state_from(joint)))
    np.testing.assert_allclose(out, expected.reshape(-1), atol=1e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sum_full_unitary_is_hermitian_and_involutory(sign):
    # Three noncommuting operands expose products on inactive slots when
    # flag=1. Every column includes the pool, flag and padded slot sectors.
    encoding = _noncommuting_sum(sign)
    circuit = _apply_to_joint_state(encoding)
    identity = np.eye(1 << (encoding.num_system + encoding.num_ancilla),
                      dtype=np.complex128)
    unitary = np.column_stack([
        np.asarray(cudaq.get_state(circuit, state_from(column.copy())))
        for column in identity.T
    ])
    np.testing.assert_allclose(unitary.conj().T @ unitary,
                               identity,
                               atol=1e-12)
    np.testing.assert_allclose(unitary @ unitary, identity, atol=1e-12)
    np.testing.assert_allclose(unitary, unitary.conj().T, atol=1e-12)


def test_sum_padded_slot_is_identity_for_arbitrary_flag_and_pool():
    encoding = _noncommuting_sum(-1.0)
    prepare = encoding.prepare_kernel()
    unprepare = encoding.unprepare_kernel()
    apply = encoding.apply_kernel()
    n = encoding.num_system
    joint = random_ket(n + encoding.num_ancilla, seed=32)
    # For K=3, slot |11> has no summand. Keep arbitrary entanglement among
    # the flag, pool and system, including flag=1 amplitudes.
    indices = np.arange(len(joint))
    joint[((indices >> n) & 3) != 3] = 0.0
    joint /= np.linalg.norm(joint)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        qubits = cudaq.qvector(state)
        ancilla = qubits[n:]
        # Change to the SELECT frame through the public protocol kernels.
        unprepare(ancilla)
        apply(ancilla, qubits[:n])
        prepare(ancilla)

    out = np.asarray(cudaq.get_state(circuit, state_from(joint)))
    np.testing.assert_allclose(out, joint, atol=1e-12)


def test_sum_controlled_walk_roundtrip_on_entangled_control_and_ancillas():
    encoding = _noncommuting_sum(-1.0)
    step = encoding.controlled_walk_step_kernel()
    adjoint = encoding.controlled_adjoint_walk_step_kernel()
    n = encoding.num_system
    joint = random_ket(n + 1 + encoding.num_ancilla, seed=33)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        qubits = cudaq.qvector(state)
        step(qubits[n:], qubits[:n])
        adjoint(qubits[n:], qubits[:n])

    out = np.asarray(cudaq.get_state(circuit, state_from(joint)))
    np.testing.assert_allclose(out, joint, atol=1e-12)


def _nested_sum(coefficient):
    wide = PauliLCU({"I": 1.0, "X": 0.7, "Z": -0.4})
    nested = SumEncoding([(1.0, PauliLCU({"X": 1.0}))])
    encoding = SumEncoding([(1.0, wide), (coefficient, nested)])
    # Independent dense reference: H = I + (0.7 + coefficient) X - 0.4 Z.
    scaled = np.array([[0.6, 0.7 + coefficient], [0.7 + coefficient, 1.4]
                       ]) / 2.7
    return encoding, scaled


@pytest.mark.parametrize("coefficient", [0.6, -0.6])
@pytest.mark.parametrize("power", [2, 3, 4])
def test_nested_sum_clean_walk_powers_match_dense(coefficient, power):
    encoding, scaled = _nested_sum(coefficient)
    ket = np.array([1.0, 0.0], dtype=np.complex128)
    block = _block(encoding, Walk(encoding).kernel(power=power), ket)
    np.testing.assert_allclose(block,
                               _chebyshev(-scaled, power, ket),
                               atol=1e-12)


@pytest.mark.parametrize("coefficient", [0.6, -0.6])
@pytest.mark.parametrize("phases",
                         [[0.0, 0.0, 0.0], [0.23, -0.41, 0.11, 0.37]])
def test_nested_sum_clean_qsvt_matches_signal_model(coefficient, phases):
    encoding, scaled = _nested_sum(coefficient)
    ket = np.array([1.0, 0.0], dtype=np.complex128)
    sequence = PhaseSequence(phases)
    block = _block(encoding, QSVT(encoding).kernel(sequence), ket)
    eigenvalues, vectors = np.linalg.eigh(scaled)
    response = np.array(
        [reference_response(sequence, float(ev)) for ev in eigenvalues])
    expected = (vectors * response) @ (vectors.conj().T @ ket)
    np.testing.assert_allclose(block, expected, atol=1e-12)
