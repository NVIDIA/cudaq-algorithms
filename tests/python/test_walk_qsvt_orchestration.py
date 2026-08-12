# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Walk/QSVT orchestration tests against a mock encoding.

The foreign-encoding protocol tests prove the consumers reach an encoding
only through the BlockEncoding surface; these tests prove the consumers
*sequence* the injected kernels correctly. The mock's circuits are chosen
so composition is directly countable: the walk step is a fixed rotation,
so ``power=k`` must produce exactly ``k`` accumulated rotations, and the
QSVT interleaving is checked against an explicit 4x4 matrix product.

The mock is not a block encoding of any Hamiltonian, so ``Walk.moment``
semantics (Chebyshev moments) are out of scope here — those are pinned by
the PauliLCU test suites against dense references.
"""

from collections import Counter

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import (ADJOINT, FORWARD, BlockEncoding, PhaseSequence,
                              QSVT, Walk)
from cudaq_algorithms import sim_utils as sim

THETA = 0.37  # walk-step / apply rotation angle on the system qubit
BETA = 0.53  # apply's rotation angle on the ancilla (breaks commutation)
PREP = 0.81  # prepare rotation angle on the ancilla


class MockBlockEncoding:
    """Minimal protocol implementation with countable circuits.

    One system qubit, one ancilla. The walk step is ``rx(THETA)`` on the
    system only; ``apply`` also rotates the ancilla by ``ry(BETA)`` so the
    QSVT direction ordering (reflect-then-apply vs apply-then-reflect)
    is observable. ``calls`` counts host-side factory invocations.
    """

    num_system = 1
    num_ancilla = 1
    alpha = 1.0

    def __init__(self):
        self.calls = Counter()

    def prepare_kernel(self):
        self.calls["prepare"] += 1
        angle = PREP

        @cudaq.kernel
        def prep(ancilla: cudaq.qview):
            ry(angle, ancilla[0])

        return prep

    def unprepare_kernel(self):
        self.calls["unprepare"] += 1
        angle = PREP

        @cudaq.kernel
        def unprep(ancilla: cudaq.qview):
            ry(-angle, ancilla[0])

        return unprep

    def apply_kernel(self):
        self.calls["apply"] += 1
        theta = THETA
        beta = BETA

        @cudaq.kernel
        def apply_mock(ancilla: cudaq.qview, system: cudaq.qview):
            rx(theta, system[0])
            ry(beta, ancilla[0])

        return apply_mock

    def controlled_apply_kernel(self):
        self.calls["controlled_apply"] += 1
        theta = THETA
        beta = BETA

        @cudaq.kernel
        def capply(control_and_ancilla: cudaq.qview, system: cudaq.qview):
            rx.ctrl(theta, control_and_ancilla[0], system[0])
            ry.ctrl(beta, control_and_ancilla[0], control_and_ancilla[1])

        return capply

    def walk_step_kernel(self):
        self.calls["walk_step"] += 1
        theta = THETA

        @cudaq.kernel
        def step(ancilla: cudaq.qview, system: cudaq.qview):
            rx(theta, system[0])

        return step

    def adjoint_walk_step_kernel(self):
        self.calls["adjoint_walk_step"] += 1
        theta = THETA

        @cudaq.kernel
        def astep(ancilla: cudaq.qview, system: cudaq.qview):
            rx(-theta, system[0])

        return astep

    def controlled_walk_step_kernel(self):
        self.calls["controlled_walk_step"] += 1
        theta = THETA

        @cudaq.kernel
        def cstep(control_and_ancilla: cudaq.qview, system: cudaq.qview):
            rx.ctrl(theta, control_and_ancilla[0], system[0])

        return cstep

    def controlled_adjoint_walk_step_kernel(self):
        self.calls["controlled_adjoint_walk_step"] += 1
        theta = THETA

        @cudaq.kernel
        def castep(control_and_ancilla: cudaq.qview, system: cudaq.qview):
            rx.ctrl(-theta, control_and_ancilla[0], system[0])

        return castep

    def select_observable(self):
        from cudaq import spin

        return spin.z(self.num_system)


class ZeroAncillaMock(MockBlockEncoding):
    num_ancilla = 0


# --------------------------------------------------------------------------
# Dense single-qubit references (state index = system + 2 * ancilla, i.e.
# the system register is allocated first / least significant).
# --------------------------------------------------------------------------


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def _on_system(op):
    return np.kron(np.eye(2), op)


def _on_ancilla(op):
    return np.kron(op, np.eye(2))


def _state(kernel):
    return np.asarray(cudaq.get_state(kernel, sim.state_from([1.0, 0.0])),
                      dtype=np.complex128)


def test_mock_satisfies_protocol():
    assert isinstance(MockBlockEncoding(), BlockEncoding)


def test_walk_applies_step_power_times():
    for power in (1, 3):
        state = _state(Walk(MockBlockEncoding()).kernel(power=power))
        # Ancilla uncomputed to |0>; system rotated by exactly power * THETA.
        expected = np.zeros(4, dtype=np.complex128)
        expected[:2] = _rx(power * THETA) @ [1.0, 0.0]
        np.testing.assert_allclose(state, expected, atol=1e-12)


def test_walk_adjoint_uses_the_adjoint_step():
    state = _state(Walk(MockBlockEncoding()).adjoint_kernel(power=2))
    expected = np.zeros(4, dtype=np.complex128)
    expected[:2] = _rx(-2 * THETA) @ [1.0, 0.0]
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_walk_uncompute_toggles_unprepare():
    state = _state(Walk(MockBlockEncoding()).kernel(power=1, uncompute=False))
    # PREPARE left in place: ancilla in ry(PREP)|0>, system rotated once.
    anc = _ry(PREP) @ [1.0, 0.0]
    sys = _rx(THETA) @ [1.0, 0.0]
    np.testing.assert_allclose(state, np.kron(anc, sys), atol=1e-12)


def test_walk_roundtrip_is_identity():
    state = _state(Walk(MockBlockEncoding()).roundtrip_kernel(power=3))
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_controlled_walk_control_conventions():
    # Register order: system (1 qubit), then [control, ancilla];
    # index = system + 2 * control + 4 * ancilla.
    off = np.asarray(
        cudaq.get_state(
            Walk(MockBlockEncoding()).controlled_kernel(power=2,
                                                        control_state=0),
            sim.state_from([1.0, 0.0])))
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0  # identity: nothing rotated, control stays |0>
    np.testing.assert_allclose(off, expected, atol=1e-12)

    on = np.asarray(
        cudaq.get_state(
            Walk(MockBlockEncoding()).controlled_kernel(power=2,
                                                        control_state=1),
            sim.state_from([1.0, 0.0])))
    expected = np.zeros(8, dtype=np.complex128)
    expected[2:4] = _rx(2 * THETA) @ [1.0, 0.0]  # control=1 block, anc=0
    np.testing.assert_allclose(on, expected, atol=1e-12)


def test_qsvt_interleaves_phases_and_directions():
    phases = [0.3, 0.7, -0.4, 0.2]
    directions = [FORWARD, ADJOINT, FORWARD]
    state = _state(
        QSVT(MockBlockEncoding()).kernel(
            PhaseSequence(phases, walk_directions=directions)))

    u = _on_ancilla(_ry(BETA)) @ _on_system(_rx(THETA))
    reflect = _on_ancilla(np.diag([-1.0, 1.0]))

    def phase(p):
        return _on_ancilla(np.diag([np.exp(1j * p), 1.0]))

    # Kernel order: phase(p0); then per step (fwd: apply,reflect /
    # adj: reflect,apply); phase(p_i). Matrix product is right-to-left.
    m = phase(phases[0])
    for i, direction in enumerate(directions):
        step = reflect @ u if direction == FORWARD else u @ reflect
        m = phase(phases[i + 1]) @ step @ m
    expected = m @ np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_factory_call_accounting():
    encoding = MockBlockEncoding()
    Walk(encoding).kernel(power=5)
    assert encoding.calls == Counter({
        "prepare": 1,
        "unprepare": 1,
        "walk_step": 1,
    })

    encoding = MockBlockEncoding()
    QSVT(encoding).kernel(PhaseSequence([0.1, 0.2, 0.3]))
    assert encoding.calls == Counter({"apply": 1})


def test_consumers_reject_zero_ancilla_encodings():
    from cudaq_algorithms import reflection_observable

    with pytest.raises(ValueError):
        Walk(ZeroAncillaMock())
    with pytest.raises(ValueError):
        QSVT(ZeroAncillaMock())
    with pytest.raises(ValueError):
        reflection_observable(ZeroAncillaMock())
