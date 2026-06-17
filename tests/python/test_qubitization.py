# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin

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
