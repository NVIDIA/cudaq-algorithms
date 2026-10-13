# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lookup rotation words, use them as controls, and coherently unload them.

Run from the checkout with PYTHONPATH=python python3
docs/sphinx/examples/python/qrom_controlled_rotations.py.
"""
# [Begin Documentation]

import os

import cudaq
import numpy as np

from cudaq_algorithms.primitives import QROMChain


def rotation_chain(fused):
    tables = [[0, 1, 3, 2, 1, 0, 2, 3], [3, 1, 0, 2, 2, 3, 1, 0],
              [1, 2, 1, 3, 0, 1, 3, 2]]
    chain = QROMChain(tables,
                      3,
                      2,
                      fused=fused,
                      variant="select_copy",
                      block_size=2)
    load, transition1, transition2, unload = chain.kernels()
    workspace = chain.num_ladder

    @cudaq.kernel
    def run():
        address = cudaq.qvector(3)
        ladder = cudaq.qvector(workspace)
        word = cudaq.qvector(2)
        target = cudaq.qvector(1)
        h(address)
        load(address, ladder, word)
        for bit in range(2):
            ry.ctrl(0.17 * (1 << bit), word[bit], target[0])
        transition1(address, ladder, word)
        for bit in range(2):
            rz.ctrl(-0.23 * (1 << bit), word[bit], target[0])
        transition2(address, ladder, word)
        for bit in range(2):
            ry.ctrl(0.31 * (1 << bit), word[bit], target[0])
        unload(address, ladder, word)

    actual = np.asarray(cudaq.get_state(run))
    expected = np.zeros_like(actual)
    target_bit = 3 + workspace + 2
    for address in range(8):
        state = np.array([1.0, 0.0], dtype=complex)
        for table, angle, axis in zip(tables, (0.17, -0.23, 0.31), "yzy"):
            theta = angle * table[address]
            if axis == "y":
                c, s = np.cos(theta / 2), np.sin(theta / 2)
                matrix = np.array([[c, -s], [s, c]])
            else:
                matrix = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
            state = matrix @ state
        expected[address] = state[0] / np.sqrt(8)
        expected[address | (1 << target_bit)] = state[1] / np.sqrt(8)
    error = np.max(np.abs(actual - expected))
    # The full-vector oracle checks phases and zero ladder/output registers.
    assert error < 1e-12
    print(f"fused={fused}: lookup Toffolis={chain.toffoli_count}, "
          f"clean workspace={workspace}, max vector error={error:.3e}")
    return actual


if __name__ == "__main__":
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    separate = rotation_chain(False)
    fused = rotation_chain(True)
    np.testing.assert_allclose(fused, separate, atol=1e-12)
