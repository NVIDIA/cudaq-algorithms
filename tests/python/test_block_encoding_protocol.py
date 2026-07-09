# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Walk and QSVT are generic over the BlockEncoding protocol.

A foreign encoding class — not a PauliLCU, inheriting from nothing, with no
``kernel_args`` — must drive Walk and QSVT to results identical to the
PauliLCU it wraps. This pins the dependency-injection seam: the consumers
may reach the encoding only through the protocol surface.
"""

import numpy as np

import cudaq

from cudaq_algorithms import (BlockEncoding, PauliLCU, PhaseSequence, QSVT,
                              Walk)
from cudaq_algorithms import sim_utils as sim

from dense_references import random_ket

HAMILTONIAN = {"ZI": 0.7, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


class ForeignEncoding:
    """Satisfies BlockEncoding structurally; hides PauliLCU behind it.

    Wraps a PauliLCU's circuits (so the physics is known-correct) but
    exposes only the protocol surface — no ``kernel_args``, no PauliLCU
    inheritance, no ``terms``.
    """

    def __init__(self, inner: PauliLCU):
        self.num_system = inner.num_system
        self.num_ancilla = inner.num_ancilla
        self.alpha = inner.alpha
        self._prepare = inner.prepare_kernel()
        self._unprepare = inner.unprepare_kernel()
        self._apply = inner.apply_kernel()
        self._controlled_apply = inner.controlled_apply_kernel()
        self._walk_step = inner.walk_step_kernel()
        self._adjoint_walk_step = inner.adjoint_walk_step_kernel()
        self._controlled_walk_step = inner.controlled_walk_step_kernel()
        self._controlled_adjoint_walk_step = (
            inner.controlled_adjoint_walk_step_kernel())
        self._select_observable = inner.select_observable()

    def prepare_kernel(self):
        return self._prepare

    def unprepare_kernel(self):
        return self._unprepare

    def apply_kernel(self):
        return self._apply

    def controlled_apply_kernel(self):
        return self._controlled_apply

    def walk_step_kernel(self):
        return self._walk_step

    def adjoint_walk_step_kernel(self):
        return self._adjoint_walk_step

    def controlled_walk_step_kernel(self):
        return self._controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self):
        return self._controlled_adjoint_walk_step

    def select_observable(self):
        return self._select_observable


def test_pauli_lcu_satisfies_protocol():
    # PauliLCU conforms structurally — deliberately no inheritance.
    assert isinstance(PauliLCU(HAMILTONIAN), BlockEncoding)
    assert BlockEncoding not in type(PauliLCU(HAMILTONIAN)).__mro__


def test_foreign_encoding_satisfies_protocol():
    foreign = ForeignEncoding(PauliLCU(HAMILTONIAN))
    assert isinstance(foreign, BlockEncoding)
    assert not isinstance(foreign, PauliLCU)


def test_walk_moments_encoding_generic():
    lcu = PauliLCU(HAMILTONIAN)
    foreign = ForeignEncoding(lcu)
    psi = random_ket(lcu.num_system, seed=3)
    for order in range(5):
        reference = Walk(lcu).moment(psi, order)
        via_protocol = Walk(foreign).moment(psi, order)
        np.testing.assert_allclose(via_protocol, reference, atol=1e-12)


def test_walk_controlled_roundtrip_encoding_generic():
    foreign = ForeignEncoding(PauliLCU(HAMILTONIAN))
    psi = random_ket(foreign.num_system, seed=11)
    kernel = Walk(foreign).controlled_roundtrip_kernel(power=2)
    state = np.asarray(cudaq.get_state(kernel, sim.state_from(psi)))
    dimension = len(psi)
    # Control was flipped on and the roundtrip is the identity: the state
    # returns to |control=1, ancilla=0...0> x |psi>.
    np.testing.assert_allclose(state[dimension:2 * dimension], psi,
                               atol=1e-10)


def test_qsvt_sequence_encoding_generic():
    lcu = PauliLCU(HAMILTONIAN)
    foreign = ForeignEncoding(lcu)
    psi = random_ket(lcu.num_system, seed=3)
    sequence = PhaseSequence([0.4, -0.2, 0.7, 0.1])
    reference = sim.transform(QSVT(lcu), psi, sequence)
    state = np.asarray(
        cudaq.get_state(QSVT(foreign).kernel(sequence), sim.state_from(psi)))
    np.testing.assert_allclose(state[:len(psi)], reference, atol=1e-12)
