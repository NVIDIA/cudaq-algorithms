# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import cudaq_algorithms as algorithms
import numpy as np


def _zero_signal_component(full_state, num_system, num_signal):
    state_vector = np.asarray(full_state, dtype=np.complex128)
    system_dimension = 1 << num_system
    expected_dimension = 1 << (num_system + num_signal)
    assert state_vector.shape == (expected_dimension, )
    return state_vector[:system_dimension].copy()


def test_tensor_to_jordan_wigner_to_pauli_lcu_qsvt_path():
    cudaq.set_target("qpp-cpu")

    one_body = np.zeros((2, 2), dtype=np.complex128)
    one_body[0, 0] = -0.80
    one_body[1, 1] = -0.35
    one_body[0, 1] = 0.12
    one_body[1, 0] = 0.12

    two_body = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    two_body[0, 1, 1, 0] = 0.18
    two_body[1, 0, 0, 1] = 0.18

    spin_op = algorithms.fermion.jordan_wigner(one_body,
                                               two_body,
                                               scalar_offset=0.21,
                                               tolerance=1e-12)
    matrix = np.asarray(spin_op.to_matrix(), dtype=np.complex128)
    num_system = 2

    rng = np.random.default_rng(314)
    initial_ket = rng.normal(size=1 << num_system)
    initial_ket = initial_ket / np.linalg.norm(initial_ket)
    initial_ket = initial_ket.astype(np.complex128)
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = algorithms.PauliLCU(spin_op,
                                   num_qubits=num_system,
                                   include_identity=False)
    shifted_matrix = (
        matrix -
        encoding.constant_term * np.eye(1 << num_system, dtype=np.complex128))

    phases, walk_directions, angles, term_controls, term_ops, term_lengths, \
        term_signs = algorithms.qsvt.pauli_lcu_kernel_args(
            algorithms.qsvt.phase_sequence([0.0, 0.0]),
            encoding.kernel_data())
    num_signal = encoding.num_ancilla

    @cudaq.kernel
    def qsvt_degree_one(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_signal)
        algorithms.qsvt.apply_phase_sequence(signal, system, phases,
                                             walk_directions, angles,
                                             term_controls, term_ops,
                                             term_lengths, term_signs)

    full_state = cudaq.get_state(qsvt_degree_one, initial_state)
    good_component = _zero_signal_component(full_state, num_system, num_signal)
    expected = -(shifted_matrix @ initial_ket) / encoding.normalization

    assert np.vdot(good_component, good_component).real > 1e-14
    assert np.allclose(good_component, expected, atol=1e-10)

    cudaq.reset_target()
