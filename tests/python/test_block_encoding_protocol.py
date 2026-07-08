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

HAMILTONIAN = {"ZI": 0.7, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


class ForeignEncoding:
    """Satisfies BlockEncoding structurally; hides PauliLCU behind it.

    Wraps a PauliLCU's circuits (so the physics is known-correct) but
    exposes only the protocol surface — no ``kernel_args``, no PauliLCU
    inheritance, no ``terms``.
    """

    def __init__(self, inner: PauliLCU):
        self._select_observable = inner.select_observable()
        self._kernels = {
            "prepare": inner.prepare_kernel(),
            "unprepare": inner.unprepare_kernel(),
            "apply": inner.apply_kernel(),
            "controlled_apply": inner.controlled_apply_kernel(),
            "step": inner.walk_step_kernel(),
            "adjoint_step": inner.adjoint_walk_step_kernel(),
            "controlled_step": inner.controlled_walk_step_kernel(),
            "controlled_adjoint_step":
                inner.controlled_adjoint_walk_step_kernel(),
        }
        self._geometry = (inner.num_system, inner.num_ancilla, inner.alpha)

    @property
    def num_system(self):
        return self._geometry[0]

    @property
    def num_ancilla(self):
        return self._geometry[1]

    @property
    def alpha(self):
        return self._geometry[2]

    def prepare_kernel(self):
        return self._kernels["prepare"]

    def unprepare_kernel(self):
        return self._kernels["unprepare"]

    def apply_kernel(self):
        return self._kernels["apply"]

    def controlled_apply_kernel(self):
        return self._kernels["controlled_apply"]

    def walk_step_kernel(self):
        return self._kernels["step"]

    def adjoint_walk_step_kernel(self):
        return self._kernels["adjoint_step"]

    def controlled_walk_step_kernel(self):
        return self._kernels["controlled_step"]

    def controlled_adjoint_walk_step_kernel(self):
        return self._kernels["controlled_adjoint_step"]

    def select_observable(self):
        return self._select_observable


def _random_ket(dimension, seed=3):
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dimension).astype(np.complex128)
    return ket / np.linalg.norm(ket)


def test_foreign_encoding_satisfies_protocol():
    foreign = ForeignEncoding(PauliLCU(HAMILTONIAN))
    assert isinstance(foreign, BlockEncoding)
    assert not isinstance(foreign, PauliLCU)


def test_walk_moments_encoding_generic():
    lcu = PauliLCU(HAMILTONIAN)
    foreign = ForeignEncoding(lcu)
    psi = _random_ket(1 << lcu.num_system)
    for order in range(5):
        reference = Walk(lcu).moment(psi, order)
        via_protocol = Walk(foreign).moment(psi, order)
        np.testing.assert_allclose(via_protocol, reference, atol=1e-12)


def test_walk_controlled_roundtrip_encoding_generic():
    foreign = ForeignEncoding(PauliLCU(HAMILTONIAN))
    psi = _random_ket(1 << foreign.num_system, seed=11)
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
    psi = _random_ket(1 << lcu.num_system)
    sequence = PhaseSequence([0.4, -0.2, 0.7, 0.1])
    reference = sim.transform(QSVT(lcu), psi, sequence)
    state = np.asarray(
        cudaq.get_state(QSVT(foreign).kernel(sequence), sim.state_from(psi)))
    np.testing.assert_allclose(state[:len(psi)], reference, atol=1e-12)
