# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composition as a supported surface: minted kernels as building blocks.

The package docstring promises that minted kernels may be called from
user kernels, and that ``cudaq.control`` applied to the whole composition
yields the controlled operation. These tests pin both halves of that
promise:

- plain composition (a user kernel calling minted kernels) involves no
  control variant at all and runs everywhere;
- controlled composition equals the factory's ``controlled=True``
  variant AND the analytic reference, so the two routes cannot agree by
  being wrong the same way. This half carries the ``controlled_composition``
  marker (conftest.py): it fails on a CUDA-Q that regresses control
  propagation and skips only below the supported 0.16 line.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import unary_iteration_kernels

_MARKED = (0, 3)


def _z_body(k):
    return [("z", 0)] if k in _MARKED else []


def test_user_kernel_composes_minted_walk_and_its_adjoint():
    # Plain composition, no control: a user kernel calls the minted walk
    # and its hand-written inverse back to back — identity on every
    # branch of a superposed address. No control variant is involved;
    # this is the un-gated half of the composition promise.
    def body(k):
        if k % 2 == 1:
            return [("x", 0), ("z", 0), ("y", 1)]
        return [("x", 1)]

    walk = unary_iteration_kernels(2, 4, body)
    kernel = walk.kernel
    kernel_adj = walk.kernel_adj

    @cudaq.kernel
    def user_program(address: cudaq.qview, ladder: cudaq.qview,
                     target: cudaq.qview):
        kernel(address, ladder, target)
        kernel_adj(address, ladder, target)

    @cudaq.kernel
    def run():
        address = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(2)
        x(target[0])
        for j in range(2):
            h(address[j])
        user_program(address, ladder, target)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << 6, dtype=np.complex128)
    for address in range(4):
        # Layout: address [0, 2), ladder [2, 4), target [4, 6); target
        # holds |01> (bit 0 set) throughout.
        expected[address + (1 << 4)] = 0.5
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.controlled_composition
def test_control_of_composition_matches_builtin_controlled_walk():
    # The controlled half: cudaq.control over a PARENT kernel that calls
    # the minted (uncontrolled) walk equals the factory's controlled=True
    # walk, and both equal the analytic state — on a superposed control
    # and superposed addresses, where a branch-relative phase error
    # would be invisible to classical control values.
    plain = unary_iteration_kernels(2, 4, _z_body)
    builtin = unary_iteration_kernels(2, 4, _z_body, controlled=True)
    plain_kernel = plain.kernel
    builtin_kernel = builtin.kernel
    theta = 0.7

    @cudaq.kernel
    def parent(address: cudaq.qview, ladder: cudaq.qview, target: cudaq.qview):
        plain_kernel(address, ladder, target)

    @cudaq.kernel
    def run_wrapped():
        control = cudaq.qvector(1)
        address = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(1)
        ry(theta, control[0])
        for j in range(2):
            h(address[j])
        x(target[0])
        cudaq.control(parent, control[0], address, ladder, target)

    @cudaq.kernel
    def run_builtin():
        control = cudaq.qvector(1)
        address = cudaq.qvector(2)
        ladder = cudaq.qvector(2)
        target = cudaq.qvector(1)
        ry(theta, control[0])
        for j in range(2):
            h(address[j])
        x(target[0])
        builtin_kernel(control, address, ladder, target)

    wrapped = np.array(cudaq.get_state(run_wrapped))
    built = np.array(cudaq.get_state(run_builtin))

    amp = (np.cos(theta / 2), np.sin(theta / 2))
    expected = np.zeros(1 << 6, dtype=np.complex128)
    for control_bit in range(2):
        for address in range(4):
            sign = -1.0 if (control_bit == 1 and address in _MARKED) else 1.0
            # Layout: control 0, address [1, 3), ladder [3, 5), target 5.
            expected[control_bit + (address << 1) + (1 << 5)] = \
                sign * amp[control_bit] / 2.0

    np.testing.assert_allclose(wrapped, expected, atol=1e-12)
    np.testing.assert_allclose(built, expected, atol=1e-12)
    np.testing.assert_allclose(wrapped, built, atol=1e-12)
