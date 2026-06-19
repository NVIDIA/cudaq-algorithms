# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
import numpy as np
import pytest

import cudaq_algorithms as algorithms


def test_qubitization_observable_builders_are_exposed_to_python():
    hamiltonian = 0.6 * spin.x(0) + 0.8 * spin.z(0)
    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=1)

    projector = algorithms.qubitization.build_ancilla_zero_projector(
        encoding.num_ancilla)
    reflection = algorithms.qubitization.build_qubitization_reflection_observable(
        encoding.num_ancilla)
    select = algorithms.qubitization.build_lcu_select_observable(encoding)

    assert isinstance(projector, cudaq.SpinOperator)
    assert isinstance(reflection, cudaq.SpinOperator)
    assert isinstance(select, cudaq.SpinOperator)

    assert len(projector.serialize()) > 0
    assert len(reflection.serialize()) > 0
    assert len(select.serialize()) > 0


def test_controlled_qubitization_walk_matches_uncontrolled_action():
    cudaq.set_target("qpp-cpu")

    hamiltonian = 0.6 * spin.x(0) + 0.8 * spin.z(0)
    hamiltonian_matrix = np.array([[0.8, 0.6], [0.6, -0.8]],
                                  dtype=np.complex128)
    initial_ket = np.array([0.3 + 0.2j, -0.4 + 0.1j], dtype=np.complex128)
    initial_ket = initial_ket / np.linalg.norm(initial_ket)
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=1)
    angles, term_controls, term_ops, term_lengths, term_signs = (
        encoding.kernel_data().unpack())
    num_ancilla = encoding.num_ancilla
    num_system = encoding.num_system
    system_dimension = 1 << num_system

    def control_component(full_state, control_bit):
        state_vector = np.asarray(full_state, dtype=np.complex128)
        offset = control_bit << (num_system + num_ancilla)
        return state_vector[offset:offset + system_dimension].copy()

    @cudaq.kernel
    def control_off(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_ancilla)
        control = cudaq.qubit()
        algorithms.block_encoding.prepare(ancilla, angles)
        algorithms.qubitization.controlled_apply_walk(control, ancilla, system,
                                                      angles, term_controls,
                                                      term_ops, term_lengths,
                                                      term_signs)
        algorithms.block_encoding.unprepare(ancilla, angles)

    off_state = cudaq.get_state(control_off, initial_state)
    assert np.allclose(control_component(off_state, 0),
                       initial_ket,
                       atol=1e-10)

    @cudaq.kernel
    def control_on(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_ancilla)
        control = cudaq.qubit()
        x(control)
        algorithms.block_encoding.prepare(ancilla, angles)
        algorithms.qubitization.controlled_apply_walk(control, ancilla, system,
                                                      angles, term_controls,
                                                      term_ops, term_lengths,
                                                      term_signs)
        algorithms.block_encoding.unprepare(ancilla, angles)

    on_state = cudaq.get_state(control_on, initial_state)
    expected = -(hamiltonian_matrix @ initial_ket) / encoding.normalization
    observed = control_component(on_state, 1)
    assert np.vdot(observed,
                   observed).real == pytest.approx(np.vdot(expected,
                                                           expected).real,
                                                   abs=1e-10)
    assert np.allclose(observed, expected, atol=1e-10)

    cudaq.reset_target()
