# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral pins for the extended body vocabulary and external control.

Two advertised capabilities of ``unary_iteration_kernels`` get their
tests here:

1. The extended body gates (``free_x``, ``free_cx``, ``and_tt``,
   ``and_wt``, ``copy_tw``, ``x_w``, ``z_w``, ``sign``). Every one is a
   classical bit permutation with a phase, so each test states its
   expected action as a tiny ``bits -> (bits, phase)`` function applied
   at the active address only (free gates fire everywhere but are
   emitted as conjugation pairs that cancel on inactive addresses), and
   the walk's statevector on a phased superposed address must equal the
   assembled sum exactly. Work registers must return to ``|0>``.

2. ``cudaq.control`` applied by the *user* to a minted kernel. The
   factory's ``controlled=True`` folds the control into the tree; an
   external ``cudaq.control`` wrap is a different mechanism entirely and
   is promised to work ("cudaq.control-compatible by construction").
   The two routes are pinned against each other and against an analytic
   reference in the same run.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import unary_iteration_kernels

cudaq.set_target("qpp-cpu")

_NUM_ADDR = 2
_NUM_ITEMS = 4
# Distinct per-bit phases so every address amplitude is different: an
# address-dependent phase error cannot cancel in the comparison.
_PHI = (0.4, 1.1)


def _address_amplitude(k: int) -> complex:
    """Amplitude of |k> after h + rz(phi) on each address bit."""
    amp = 1.0 + 0.0j
    for j, phi in enumerate(_PHI):
        bit = (k >> j) & 1
        amp *= np.exp(1j * phi / 2 if bit else -1j * phi / 2) / np.sqrt(2)
    return amp


def _walk_state(walk, num_target: int, num_work: int,
                target_basis: int) -> np.ndarray:
    """One application on |phased address> (x) |target_basis>, |0> work."""
    kernel = walk.kernel

    if num_work == 0:

        @cudaq.kernel
        def run(t_basis: int):
            address = cudaq.qvector(_NUM_ADDR)
            ladder = cudaq.qvector(_NUM_ADDR)
            target = cudaq.qvector(num_target)
            for j in range(_NUM_ADDR):
                h(address[j])
            rz(0.4, address[0])
            rz(1.1, address[1])
            for t in range(num_target):
                if (t_basis >> t) & 1:
                    x(target[t])
            kernel(address, ladder, target)
    else:

        @cudaq.kernel
        def run(t_basis: int):
            address = cudaq.qvector(_NUM_ADDR)
            ladder = cudaq.qvector(_NUM_ADDR)
            target = cudaq.qvector(num_target)
            work = cudaq.qvector(num_work)
            for j in range(_NUM_ADDR):
                h(address[j])
            rz(0.4, address[0])
            rz(1.1, address[1])
            for t in range(num_target):
                if (t_basis >> t) & 1:
                    x(target[t])
            kernel(address, ladder, target, work)

    return np.array(cudaq.get_state(run, target_basis))


def _expected_state(action, num_target: int, num_work: int,
                    target_basis: int) -> np.ndarray:
    """Assemble sum_k a_k |k>|0 ladder>|new bits>|0 work>.

    ``action(k, bits) -> (new_bits, phase)`` is the claimed effect of the
    body at address ``k`` on target bits ``bits`` (little-endian ints).
    Layout (qubit 0 = LSB of the state index, allocation order): address
    [0, A), ladder [A, 2A), target [2A, 2A + T), work above that — the
    expected vector is nonzero only where ladder and work are 0.
    """
    total = 2 * _NUM_ADDR + num_target + num_work
    expected = np.zeros(1 << total, dtype=np.complex128)
    for k in range(_NUM_ITEMS):
        new_bits, phase = action(k, target_basis)
        index = k + (new_bits << (2 * _NUM_ADDR))
        expected[index] += _address_amplitude(k) * phase
    return expected


def _check_walk(body, action, num_target: int, expected_work: int):
    walk = unary_iteration_kernels(_NUM_ADDR, _NUM_ITEMS, body)
    assert walk.num_work == expected_work
    for target_basis in range(1 << num_target):
        state = _walk_state(walk, num_target, walk.num_work, target_basis)
        expected = _expected_state(action, num_target, walk.num_work,
                                   target_basis)
        np.testing.assert_allclose(state, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The extended vocabulary, gate by gate
# ---------------------------------------------------------------------------


def test_sign_is_a_phase_on_the_active_address_only():
    marked = (1, 3)

    def body(k):
        return [("sign", )] if k in marked else []

    def action(k, bits):
        return bits, -1.0 if k in marked else 1.0

    _check_walk(body, action, num_target=1, expected_work=0)


def test_free_x_conjugation_inverts_a_leaf_controlled_z():
    # X Z X = -Z: the conjugated core phases the |0> branch of target 0
    # at the active address; the unconditional free_x pair must cancel
    # exactly on every other address.
    def body(k):
        return [("free_x", 0), ("z", 0), ("free_x", 0)]

    def action(k, bits):
        return bits, -1.0 if (bits & 1) == 0 else 1.0

    _check_walk(body, action, num_target=2, expected_work=0)


def test_free_cx_conjugation_builds_a_parity_phase():
    # CX(0 -> 1) Z_1 CX(0 -> 1) = Z_0 Z_1: phase -1 exactly when the two
    # target bits differ... on bit1 XOR bit0 = 1.
    def body(k):
        return [("free_cx", 0, 1), ("z", 1), ("free_cx", 0, 1)]

    def action(k, bits):
        parity = ((bits >> 1) ^ bits) & 1
        return bits, -1.0 if parity else 1.0

    _check_walk(body, action, num_target=2, expected_work=0)


def test_and_tt_with_z_w_is_a_ccz_at_the_active_address():
    # Compute t0 AND t1 into work 0, phase on it, uncompute: CCZ(t0, t1)
    # delivered only at the marked addresses; work back to |0> always.
    marked = (0, 2)

    def body(k):
        if k not in marked:
            return []
        return [("and_tt", 0, 1, 0), ("z_w", 0), ("and_tt", 0, 1, 0)]

    def action(k, bits):
        both = (bits & 1) and (bits >> 1) & 1
        return bits, -1.0 if (k in marked and both) else 1.0

    _check_walk(body, action, num_target=2, expected_work=1)


def test_and_wt_chains_to_a_cccz_at_the_active_address():
    # work0 := t0 AND t1 (and_tt), work1 := work0 AND t2 (and_wt), phase
    # on work1, uncompute both: CCCZ(t0, t1, t2) at every address here.
    def body(k):
        return [
            ("and_tt", 0, 1, 0),
            ("and_wt", 0, 2, 1),
            ("z_w", 1),
            ("and_wt", 0, 2, 1),
            ("and_tt", 0, 1, 0),
        ]

    def action(k, bits):
        all_three = (bits & 1) and (bits >> 1) & 1 and (bits >> 2) & 1
        return bits, -1.0 if all_three else 1.0

    _check_walk(body, action, num_target=3, expected_work=2)


def test_copy_tw_with_x_w_is_a_leaf_controlled_cx():
    # work0 := copy of t0, x_w flips t1 when (leaf AND work0): the net
    # body is CX(t0 -> t1) at the active address; the copy uncomputes.
    marked = (1, 2)

    def body(k):
        if k not in marked:
            return []
        return [("copy_tw", 0, 0), ("x_w", 0, 1), ("copy_tw", 0, 0)]

    def action(k, bits):
        if k in marked and (bits & 1):
            return bits ^ 2, 1.0
        return bits, 1.0

    _check_walk(body, action, num_target=2, expected_work=1)


def test_work_gate_toffoli_count_matches_the_compiler():
    # The emitter's toffoli_count must stay equal to the compiled count
    # when the body itself contributes Toffolis (and_tt / and_wt / x_w
    # are 2-controlled gates on top of the walk's own ladder Toffolis).
    pytest.importorskip("cudaq", reason="resource estimation probe")
    if not hasattr(cudaq, "estimate_resources"):
        pytest.skip("cudaq.estimate_resources is not available")

    def body(k):
        return [("and_tt", 0, 1, 0), ("z_w", 0), ("and_tt", 0, 1, 0)]

    walk = unary_iteration_kernels(_NUM_ADDR, _NUM_ITEMS, body)
    kernel = walk.kernel

    @cudaq.kernel
    def harness():
        address = cudaq.qvector(_NUM_ADDR)
        ladder = cudaq.qvector(_NUM_ADDR)
        target = cudaq.qvector(2)
        work = cudaq.qvector(1)
        kernel(address, ladder, target, work)

    resources = cudaq.estimate_resources(harness)
    compiled = resources.count_controls("x", 2)
    assert compiled == walk.toffoli_count


# ---------------------------------------------------------------------------
# External cudaq.control vs the factory's controlled=True
# ---------------------------------------------------------------------------


def test_external_cudaq_control_matches_builtin_controlled():
    """The two ways to control a walk must implement the same unitary.

    ``controlled=True`` folds the control into the tree (3N/2 - 1
    Toffolis); ``cudaq.control`` on the plain kernel lets CUDA-Q add a
    control to every gate (more expensive, but promised to be correct:
    the minted kernel is flat by construction). Same prep, both routes,
    equality to each other AND to the analytic state — so they cannot
    agree by being wrong the same way.
    """
    marked = (0, 3)

    def body(k):
        return [("z", 0)] if k in marked else []

    plain = unary_iteration_kernels(_NUM_ADDR, _NUM_ITEMS, body)
    builtin = unary_iteration_kernels(_NUM_ADDR,
                                      _NUM_ITEMS,
                                      body,
                                      controlled=True)
    plain_kernel = plain.kernel
    builtin_kernel = builtin.kernel
    theta = 0.7

    @cudaq.kernel
    def run_wrapped():
        control = cudaq.qvector(1)
        address = cudaq.qvector(_NUM_ADDR)
        ladder = cudaq.qvector(_NUM_ADDR)
        target = cudaq.qvector(1)
        ry(theta, control[0])
        for j in range(_NUM_ADDR):
            h(address[j])
        rz(0.4, address[0])
        rz(1.1, address[1])
        x(target[0])
        cudaq.control(plain_kernel, control[0], address, ladder, target)

    @cudaq.kernel
    def run_builtin():
        control = cudaq.qvector(1)
        address = cudaq.qvector(_NUM_ADDR)
        ladder = cudaq.qvector(_NUM_ADDR)
        target = cudaq.qvector(1)
        ry(theta, control[0])
        for j in range(_NUM_ADDR):
            h(address[j])
        rz(0.4, address[0])
        rz(1.1, address[1])
        x(target[0])
        builtin_kernel(control, address, ladder, target)

    wrapped = np.array(cudaq.get_state(run_wrapped))
    built = np.array(cudaq.get_state(run_builtin))

    # Analytic reference. Layout: control 0, address [1, 3), ladder
    # [3, 5), target 5; target holds |1> so the leaf-controlled Z kicks
    # a -1 exactly on (control = 1) x (marked address).
    c = (np.cos(theta / 2), np.sin(theta / 2))
    expected = np.zeros(1 << (2 * _NUM_ADDR + 2), dtype=np.complex128)
    for control_bit in range(2):
        for k in range(_NUM_ITEMS):
            sign = -1.0 if (control_bit == 1 and k in marked) else 1.0
            index = control_bit + (k << 1) + (1 << (2 * _NUM_ADDR + 1))
            expected[index] = c[control_bit] * _address_amplitude(k) * sign

    np.testing.assert_allclose(wrapped, expected, atol=1e-12)
    np.testing.assert_allclose(built, expected, atol=1e-12)
    np.testing.assert_allclose(wrapped, built, atol=1e-12)
