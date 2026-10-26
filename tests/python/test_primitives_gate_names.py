# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller kernels must not shadow the interpreter's measurement gates."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import (measured_unary_iteration_kernels,
                                         unary_iteration_kernels)


@pytest.mark.parametrize("name", ["h", "mz", "x", "z"])
@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("controlled,has_work", [(False, False), (True, False),
                                                 (False, True), (True, True)])
def test_caller_gate_names_do_not_change_lookup(name, measured, controlled,
                                                has_work):
    # CUDA-Q keeps decorated kernels in process-global registries. Isolate
    # the deliberately colliding names so other tests cannot inherit them.
    result = subprocess.run([
        sys.executable,
        str(pathlib.Path(__file__).resolve()), name,
        str(int(measured)),
        str(int(controlled)),
        str(int(has_work))
    ],
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_lookup(name, measured, controlled, has_work):
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    cudaq.set_random_seed(731)

    # The caller's kernels are valid, but their signatures differ from the
    # intrinsics. Neither an unused coherent arm nor actual measured erasure
    # may resolve to these definitions when compiling the shared interpreter.
    if name == "h":

        @cudaq.kernel
        def h(qubits: cudaq.qview):
            x(qubits[0])

    elif name == "x":

        @cudaq.kernel
        def x(qubits: cudaq.qview):
            z(qubits[0])

    elif name == "z":

        @cudaq.kernel
        def z(qubits: cudaq.qview):
            x(qubits[0])

    else:

        @cudaq.kernel
        def mz(qubits: cudaq.qview):
            x(qubits[0])

    data = [1, 0, 1, 0, 0, 1]
    factory = (measured_unary_iteration_kernels
               if measured else unary_iteration_kernels)
    walk = factory(3,
                   len(data),
                   lambda k: [("x", 0)] if data[k] else [],
                   controlled=controlled,
                   include_adjoint=False,
                   num_work=int(has_work))
    lookup = walk.kernel
    num_ladder = walk.num_ladder
    if measured:
        assert walk.num_measurements > 0

    # Logical register: address[0:3], target[3], optional control[4].
    if controlled and has_work:

        @cudaq.kernel
        def run(initial: cudaq.State):
            logical = cudaq.qvector(initial)
            ladder = cudaq.qvector(num_ladder)
            work = cudaq.qvector(1)
            lookup(logical[4:5], logical[0:3], ladder, logical[3:4], work)

    elif controlled:

        @cudaq.kernel
        def run(initial: cudaq.State):
            logical = cudaq.qvector(initial)
            ladder = cudaq.qvector(num_ladder)
            lookup(logical[4:5], logical[0:3], ladder, logical[3:4])

    elif has_work:

        @cudaq.kernel
        def run(initial: cudaq.State):
            logical = cudaq.qvector(initial)
            ladder = cudaq.qvector(num_ladder)
            work = cudaq.qvector(1)
            lookup(logical[0:3], ladder, logical[3:4], work)

    else:

        @cudaq.kernel
        def run(initial: cudaq.State):
            logical = cudaq.qvector(initial)
            ladder = cudaq.qvector(num_ladder)
            lookup(logical[0:3], ladder, logical[3:4])

    # An independent XOR oracle checks complex phases, a superposed external
    # control, out-of-table addresses, and clean ladder/work registers.
    logical_width = 4 + int(controlled)
    rng = np.random.default_rng(731)
    initial = (rng.normal(size=1 << logical_width) +
               1j * rng.normal(size=1 << logical_width))
    initial /= np.linalg.norm(initial)
    expected = np.zeros(1 << (logical_width + num_ladder + int(has_work)),
                        dtype=np.complex128)
    for index, amplitude in enumerate(initial):
        address = index & 7
        enabled = not controlled or bool(index & 16)
        flip = data[address] if enabled and address < len(data) else 0
        expected[index ^ (flip << 3)] = amplitude
    for _ in range(4 if measured else 1):
        actual = np.asarray(
            cudaq.get_state(run, cudaq.State.from_data(initial)))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == "__main__":
    _run_lookup(sys.argv[1], *(bool(int(value)) for value in sys.argv[2:]))
