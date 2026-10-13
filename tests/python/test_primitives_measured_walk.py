# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured SELECT regressions on coherent, complex-amplitude inputs.

Repeated native trajectories complement the independent exhaustive Kraus
checks in test_primitives_measured_channel.py; they are not a proof by
sampling alone. Execution uses the explicitly constructed control path.
The trajectory tests seed CUDA-Q's generator so a failure reproduces.
"""

import numpy as np
import pytest

import cudaq
from cudaq import spin

from cudaq_algorithms.primitives import (MeasuredUnaryIterationKernels,
                                         measured_unary_iteration_kernels,
                                         unary_iteration_kernels)
from cudaq_algorithms.primitives._unary_iteration import (
    _OP_AND_TT, _OP_CCX, _OP_CCX_CTRL, _OP_MEAS_UNCOMPUTE_CTRL,
    _TOFFOLI_OPCODES, _mint_interpreter)
from cudaq_algorithms.primitives._unary_iteration_measured import (
    _MEASURED_OPCODES, _emit_measured_walk, _measured_walk_toffoli_count)
from dense_references import dense_matrix
from measured_harness import resources_of

_BODIES = {
    0: [("x", 0), ("y", 1)],
    1: [("z", 0)],
    2: [("y", 0), ("x", 1), ("z", 1)],
    4: [("x", 1)],
    5: [("y", 1), ("z", 0)],
}


def _body(k):
    return _BODIES.get(k, [])


def _harness(num_bits, controlled, first, second=None):
    """Compile one entry kernel that prepares the superposed reference
    input (uniform address with per-bit phases, rotated control and
    targets, so amplitudes are complex and a misplaced fix-up CZ shows up
    as a sign flip on part of the superposition) and then applies
    ``first`` and, when given, ``second``. Compile once, launch many
    times: each launch re-samples the measurements independently."""
    two = second is not None
    if controlled and two:

        @cudaq.kernel
        def run():
            control = cudaq.qvector(1)
            address_reg = cudaq.qvector(num_bits)
            ladder = cudaq.qvector(num_bits)
            target = cudaq.qvector(2)
            ry(0.7, control[0])
            rz(0.41, control[0])
            for i in range(num_bits):
                h(address_reg[i])
                rz(0.23 + 0.31 * i, address_reg[i])
            ry(0.3, target[0])
            rz(0.57, target[0])
            ry(1.1, target[1])
            rz(1.37, target[1])
            first(control, address_reg, ladder, target)
            second(control, address_reg, ladder, target)
    elif controlled:

        @cudaq.kernel
        def run():
            control = cudaq.qvector(1)
            address_reg = cudaq.qvector(num_bits)
            ladder = cudaq.qvector(num_bits)
            target = cudaq.qvector(2)
            ry(0.7, control[0])
            rz(0.41, control[0])
            for i in range(num_bits):
                h(address_reg[i])
                rz(0.23 + 0.31 * i, address_reg[i])
            ry(0.3, target[0])
            rz(0.57, target[0])
            ry(1.1, target[1])
            rz(1.37, target[1])
            first(control, address_reg, ladder, target)
    elif two:

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(num_bits)
            ladder = cudaq.qvector(num_bits)
            target = cudaq.qvector(2)
            for i in range(num_bits):
                h(address_reg[i])
                rz(0.23 + 0.31 * i, address_reg[i])
            ry(0.3, target[0])
            rz(0.57, target[0])
            ry(1.1, target[1])
            rz(1.37, target[1])
            first(address_reg, ladder, target)
            second(address_reg, ladder, target)
    else:

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(num_bits)
            ladder = cudaq.qvector(num_bits)
            target = cudaq.qvector(2)
            for i in range(num_bits):
                h(address_reg[i])
                rz(0.23 + 0.31 * i, address_reg[i])
            ry(0.3, target[0])
            rz(0.57, target[0])
            ry(1.1, target[1])
            rz(1.37, target[1])
            first(address_reg, ladder, target)

    return run


def _prepared_reference(num_bits, controlled):
    """The reference input alone (a one-address walk with an empty body is
    the identity)."""
    identity = unary_iteration_kernels(num_bits,
                                       1,
                                       lambda k: [],
                                       controlled=controlled,
                                       include_adjoint=False).kernel
    return np.array(cudaq.get_state(_harness(num_bits, controlled, identity)))


# ----------------------------------------------------------------------
# 1. Behavioral equivalence with the coherent walk (the crux)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("num_bits,num_items", [(2, 4), (2, 3), (3, 8), (3, 5),
                                                (4, 11)])
def test_measured_walk_equals_coherent_walk_on_superpositions(
        num_bits, num_items, controlled):
    # Superposed address x superposed target (x rotated control): the
    # fix-up CZ placement is invisible on computational-basis inputs and
    # shows up here as a relative sign. Absolute tolerance 1e-12 on the
    # raw complex vector; six independently sampled trajectories.
    cudaq.set_random_seed(1000 * num_bits + num_items)
    coherent = unary_iteration_kernels(num_bits,
                                       num_items,
                                       _body,
                                       controlled=controlled,
                                       include_adjoint=False)
    measured = measured_unary_iteration_kernels(num_bits,
                                                num_items,
                                                _body,
                                                controlled=controlled,
                                                include_adjoint=False)
    assert measured.num_measurements > 0
    reference = np.array(
        cudaq.get_state(_harness(num_bits, controlled, coherent.kernel)))
    run = _harness(num_bits, controlled, measured.kernel)
    for _ in range(6):
        state = np.array(cudaq.get_state(run))
        np.testing.assert_allclose(state, reference, atol=1e-12, rtol=0)


def test_measured_walk_ladder_returns_to_zero_mid_walk_branches():
    # A partial tree whose transitions exercise multi-level retraces
    # (fix-ups firing mid-walk) on a basis input: amplitude must sit
    # entirely in the ladder-|0> block afterwards. One compiled harness
    # launched per address (CUDA-Q does not re-specialize per int value).
    cudaq.set_random_seed(36)
    walk = measured_unary_iteration_kernels(3, 6, _body, include_adjoint=False)
    kernel = walk.kernel

    @cudaq.kernel
    def run(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        target = cudaq.qvector(2)
        for i in range(3):
            if ((address >> i) & 1) == 1:
                x(address_reg[i])
        kernel(address_reg, ladder, target)

    for address in range(8):
        state = np.array(cudaq.get_state(run, address))
        # Layout: address [0,3), ladder [3,6), target [6,8).
        for index in np.nonzero(np.abs(state) > 1e-12)[0]:
            assert (index >> 3) & 0b111 == 0, (
                f"ladder not |0> at address {address}")


# ----------------------------------------------------------------------
# 2. Cost bookkeeping: N - 2 / N - 1 AND computes, one measurement each
# ----------------------------------------------------------------------


def test_measured_walk_toffoli_and_measurement_accounting():
    # Emit-only: the tape, not a minted kernel, carries the counts. With an
    # empty body every Toffoli is a tree AND, so measurements == Toffolis
    # is the load-bearing identity on every tree shape.
    for num_bits in range(1, 8):
        capacity = 1 << num_bits
        for controlled in (False, True):
            for num_items in range(1, capacity + 1):
                ops = _emit_measured_walk(num_bits, num_items, controlled,
                                          [[] for _ in range(num_items)],
                                          False)
                toffolis = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)
                measurements = sum(1 for op in ops
                                   if op[0] in _MEASURED_OPCODES)
                expected = _measured_walk_toffoli_count(
                    num_bits, num_items, controlled)
                assert toffolis == measurements == expected
                if num_items == capacity:
                    assert toffolis == capacity - 2 + int(controlled)
    # The minted record carries the same numbers.
    for controlled in (False, True):
        walk = measured_unary_iteration_kernels(3,
                                                8,
                                                lambda k: [("x", 0)],
                                                controlled=controlled,
                                                include_adjoint=False)
        assert walk.toffoli_count == walk.num_measurements == 6 + int(
            controlled)


def test_measured_walk_costs_match_the_compiler():
    # CUDA-Q 0.16 counts unconditional Toffolis and measurements. Fixups
    # are conditional and their count can depend on the traced trajectory.
    walk = measured_unary_iteration_kernels(4,
                                            16,
                                            lambda k: [("x", 0)],
                                            include_adjoint=False)
    resources = resources_of(walk.kernel, 4, 4, 1)
    assert resources.count_controls("x", 2) == walk.toffoli_count == 14
    assert resources.count("mz") == walk.num_measurements == 14
    assert 0 <= resources.count_controls("z", 1) <= walk.num_measurements


# ----------------------------------------------------------------------
# 3. The framework boundary: cudaq.control / cudaq.sample on measuring
#    kernels
# ----------------------------------------------------------------------


def test_cudaq_control_on_measuring_kernel_is_not_the_controlled_channel():
    # The physics allows controlled measured uncomputation (fold the
    # control into the tree root: nothing measured is ever controlled, and
    # the fix-up CZ is the identity on the control-off branch, which is
    # exactly what controlled=True mints). The framework path is the
    # problem: on CUDA-Q 0.16 cudaq.control on a measuring kernel compiles
    # and runs, but mz has no controlled form, so each run measures the
    # control register and the state differs from the controlled channel
    # (and from run to run). Pin whichever failure mode this CUDA-Q
    # exhibits: an outright rejection, or the wrong channel.
    cudaq.set_random_seed(5)
    measured = measured_unary_iteration_kernels(2,
                                                4,
                                                _body,
                                                include_adjoint=False)
    inner = measured.kernel
    correct = measured_unary_iteration_kernels(2,
                                               4,
                                               _body,
                                               controlled=True,
                                               include_adjoint=False)
    reference = np.array(cudaq.get_state(_harness(2, True, correct.kernel)))

    @cudaq.kernel
    def naive_controlled():
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        ry(0.7, control[0])
        rz(0.41, control[0])
        for i in range(2):
            h(address_reg[i])
            rz(0.23 + 0.31 * i, address_reg[i])
        ry(0.3, target[0])
        rz(0.57, target[0])
        ry(1.1, target[1])
        rz(1.37, target[1])
        cudaq.control(inner, control[0], address_reg, ladder, target)

    try:
        for _ in range(6):
            state = np.array(cudaq.get_state(naive_controlled))
            assert not np.allclose(state, reference, atol=1e-9), (
                "cudaq.control on a measuring kernel now implements the "
                "controlled channel: revisit the explicit controlled "
                "construction and this pin")
    except RuntimeError:
        # Also acceptable: a CUDA-Q that rejects the construction
        # outright (the boundary enforced instead of silent).
        pass


@cudaq.kernel
def _direct_feedback():
    q = cudaq.qvector(2)
    h(q[0])
    fired = mz(q[0])
    if fired:
        x(q[1])


def test_cudaq_sample_rejects_measurement_branching_kernel():
    # cudaq.sample refuses kernels that branch on measurement results. The
    # check is CUDA-Q's, not this package's; pin it where it is enforced.
    with pytest.raises(RuntimeError, match="feedback|measurement"):
        cudaq.sample(_direct_feedback, shots_count=10)


def test_cudaq_sample_rejects_nested_measurement_feedback():
    # CUDA-Q 0.16 also rejects the feedback when it sits in a minted
    # sub-kernel of a sampled entry kernel (older releases let it through
    # and returned samples that were not the logical lookup).
    walk = measured_unary_iteration_kernels(2, 4, _body, include_adjoint=False)
    kernel = walk.kernel

    @cudaq.kernel
    def entry():
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        h(address_reg[0])
        kernel(address_reg, ladder, target)
        mz(target)

    with pytest.raises(RuntimeError, match="feedback|measurement"):
        cudaq.sample(entry, shots_count=10)


# ----------------------------------------------------------------------
# 4. The inverse walk
# ----------------------------------------------------------------------


@pytest.mark.parametrize("controlled", [False, True])
def test_measured_walk_adjoint_composes_to_identity(controlled):
    # Non-commuting multi-gate bodies so the walk is not an involution;
    # only the reverse-order walk with reversed bodies restores the
    # state. Superposed inputs, repeated runs.
    cudaq.set_random_seed(11)

    def body(k):
        if k % 2 == 1:
            return [("x", 0), ("z", 0), ("y", 1)]
        return [("x", 1)]

    walk = measured_unary_iteration_kernels(3, 6, body, controlled=controlled)
    reference = _prepared_reference(3, controlled)
    run = _harness(3, controlled, walk.kernel, walk.kernel_adj)
    for _ in range(5):
        state = np.array(cudaq.get_state(run))
        np.testing.assert_allclose(state, reference, atol=1e-12, rtol=0)


# ----------------------------------------------------------------------
# 5. Surface parity with the coherent walk
# ----------------------------------------------------------------------


def test_measured_walk_validation_raises():
    body = lambda k: []
    with pytest.raises(ValueError, match="num_address_bits must be a"):
        measured_unary_iteration_kernels(0, 1, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        measured_unary_iteration_kernels(2, 0, body)
    with pytest.raises(ValueError, match=r"num_items must be an integer in"):
        measured_unary_iteration_kernels(2, 5, body)
    with pytest.raises(ValueError, match="unsupported gate 'q'"):
        measured_unary_iteration_kernels(2, 4, lambda k: [("q", 0)])
    with pytest.raises(ValueError, match="invalid target qubit index"):
        measured_unary_iteration_kernels(2, 4, lambda k: [("x", -1)])


def test_measured_describe_decodes_the_tape():
    walk = measured_unary_iteration_kernels(2,
                                            4,
                                            lambda k: [("x", 0)],
                                            include_adjoint=False)
    text = walk.describe()
    lines = text.splitlines()
    assert lines[0].startswith("measured unary-iteration walk: 4 addresses")
    assert "2 measurements" in lines[0]
    assert sum("# Toffoli" in line for line in lines) == walk.toffoli_count
    gadget_lines = [line for line in lines if "# measured uncompute" in line]
    assert len(gadget_lines) == walk.num_measurements
    assert "mz -> x(ladder" in gadget_lines[0]
    assert "fixup z.ctrl" in gadget_lines[0]
    body_lines = [line for line in lines if line.startswith("x(target[0])")]
    assert len(body_lines) == 4


def test_measured_walk_extended_body_vocabulary_ports():
    # The extended gates are unitary gates on target/work and orthogonal
    # to the uncompute strategy: a body using the AND-ladder gadgets and
    # the sign leaf must agree with the coherent walk gate for gate.
    cudaq.set_random_seed(13)

    def body(k):
        if k == 1:
            return [("sign", )]
        if k == 2:
            return [("and_tt", 0, 1, 0), ("z_w", 0), ("and_tt", 0, 1, 0)]
        return [("free_x", 0), ("x", 1), ("free_x", 0)]

    coherent = unary_iteration_kernels(2, 4, body, include_adjoint=False)
    measured = measured_unary_iteration_kernels(2,
                                                4,
                                                body,
                                                include_adjoint=False)
    assert measured.num_work == coherent.num_work == 1
    kc = coherent.kernel
    km = measured.kernel

    @cudaq.kernel
    def run_coherent():
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        work = cudaq.qvector(1)
        for i in range(2):
            h(address_reg[i])
        ry(0.3, target[0])
        ry(1.1, target[1])
        kc(address_reg, ladder, target, work)

    @cudaq.kernel
    def run_measured():
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        work = cudaq.qvector(1)
        for i in range(2):
            h(address_reg[i])
        ry(0.3, target[0])
        ry(1.1, target[1])
        km(address_reg, ladder, target, work)

    reference = np.array(cudaq.get_state(run_coherent))
    for _ in range(4):
        state = np.array(cudaq.get_state(run_measured))
        np.testing.assert_allclose(state, reference, atol=1e-12, rtol=0)


# ----------------------------------------------------------------------
# 6. Review hardening: cross-implementation composition and observe
# ----------------------------------------------------------------------


@pytest.mark.parametrize("controlled", [False, True])
def test_measured_walk_then_coherent_adjoint_is_identity(controlled):
    # The strongest single equivalence statement available: the measured
    # walk composed with the COHERENT walk's hand-written inverse must be
    # the identity on complex-amplitude inputs. It couples the two
    # implementations' channels AND their global-phase conventions in one
    # assertion (a same-implementation round trip would cancel a shared
    # phase error; this cannot).
    cudaq.set_random_seed(17)
    measured = measured_unary_iteration_kernels(3,
                                                8,
                                                _body,
                                                controlled=controlled,
                                                include_adjoint=False)
    coherent = unary_iteration_kernels(3, 8, _body, controlled=controlled)
    reference = _prepared_reference(3, controlled)
    run = _harness(3, controlled, measured.kernel, coherent.kernel_adj)
    for _ in range(3):
        state = np.array(cudaq.get_state(run))
        np.testing.assert_allclose(state, reference, atol=1e-12, rtol=0)


def test_cudaq_observe_accepts_measured_walk_and_agrees():
    cudaq.set_random_seed(19)
    measured = measured_unary_iteration_kernels(2,
                                                4,
                                                _body,
                                                include_adjoint=False)
    m_kernel = measured.kernel

    @cudaq.kernel
    def m_harness():
        address_reg = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        h(address_reg[0])
        ry(0.81, address_reg[1])
        rz(0.23, address_reg[0])
        rz(0.54, address_reg[1])
        ry(0.3, target[0])
        rz(0.57, target[0])
        ry(1.1, target[1])
        rz(1.37, target[1])
        m_kernel(address_reg, ladder, target)

    # Independently build the logical four-qubit input, apply address
    # projectors times body Pauli matrices, and contract X(address0)Z(target0).
    def single(theta, phi):
        return np.array([
            np.cos(theta / 2) * np.exp(-0.5j * phi),
            np.sin(theta / 2) * np.exp(0.5j * phi)
        ])

    initial = np.kron(
        single(1.1, 1.37),
        np.kron(single(0.3, 0.57),
                np.kron(single(0.81, 0.54), single(np.pi / 2, 0.23))))
    paulis = {
        "x": np.array([[0, 1], [1, 0]]),
        "y": np.array([[0, -1j], [1j, 0]]),
        "z": np.diag([1, -1])
    }
    expected = np.zeros(16, complex)
    for address in range(4):
        target = initial[address::4].copy()
        for name, bit in _body(address):
            gate = np.kron(paulis[name], np.eye(2)) if bit else np.kron(
                np.eye(2), paulis[name])
            target = gate @ target
        expected[address::4] = target
    # Logical qubits: address0, address1, target0, target1 (little-endian).
    observable_dense = dense_matrix([(1.0, "XIZI")], 4)
    reference = np.vdot(expected, observable_dense @ expected).real
    assert abs(reference) > 0.2  # A non-vacuous address-coherence witness.
    observable = spin.x(0) * spin.z(4)
    for _ in range(3):
        measured_value = cudaq.observe(m_harness, observable).expectation()
        assert abs(measured_value - reference) < 1e-12


def test_tree_and_count_excludes_body_toffolis():

    def body(k):
        if k == 0:
            return [("and_tt", 0, 1, 0), ("x_w", 0, 2), ("and_tt", 0, 1, 0)]
        return []

    walk = measured_unary_iteration_kernels(2, 4, body, include_adjoint=False)
    assert walk.toffoli_count == 5
    assert walk.tree_and_count == walk.num_measurements == 2
    assert walk.body_toffoli_count == 3


def test_tree_and_count_is_counted_from_the_tape():
    # Full tree, empty body: every Toffoli is a tree AND and every tree AND
    # is measured away exactly once. The two numbers are computed from
    # different opcodes, so this is not a tautology.
    for controlled in (False, True):
        walk = measured_unary_iteration_kernels(3,
                                                8,
                                                lambda k: [],
                                                controlled=controlled,
                                                include_adjoint=False)
        ands = sum(1 for op in walk.ops if op[0] in (_OP_CCX, _OP_CCX_CTRL))
        assert walk.tree_and_count == ands == walk.num_measurements
        assert walk.tree_and_count == 6 + int(controlled)


def test_describe_tolerates_a_hand_built_record():
    record = MeasuredUnaryIterationKernels(kernel=None,
                                           kernel_adj=None,
                                           num_address=2,
                                           num_ladder=2,
                                           num_items=4,
                                           controlled=False,
                                           toffoli_count=0,
                                           num_measurements=0)
    assert record.describe().startswith("measured unary-iteration walk")
    assert record.tree_and_count == 0


@pytest.mark.parametrize("controlled,has_work", [(False, False), (True, False),
                                                 (False, True), (True, True)])
def test_shared_interpreter_rejects_out_of_set_measured_tapes(
        controlled, has_work):
    with pytest.raises(ValueError, match="outside the interpreter variant"):
        _mint_interpreter([(999, 0, 0, 0)], controlled, has_work)
    if not controlled:
        with pytest.raises(ValueError,
                           match="outside the interpreter variant"):
            _mint_interpreter([(_OP_MEAS_UNCOMPUTE_CTRL, 0, 0, 0)], controlled,
                              has_work)
    if not has_work:
        with pytest.raises(ValueError,
                           match="outside the interpreter variant"):
            _mint_interpreter([(_OP_AND_TT, 0, 1, 0)], controlled, has_work)


@pytest.mark.parametrize("n,m,c,expected", [(4, 1, False, 3), (4, 5, False, 6),
                                            (4, 11, False, 11),
                                            (4, 11, True, 12),
                                            (128, 1, False, 127)])
def test_partial_tree_full_address_contract_cost(n, m, c, expected):
    assert _measured_walk_toffoli_count(n, m, c) == expected
