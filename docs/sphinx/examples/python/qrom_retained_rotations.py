# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain a bank across noncommuting data rotations; check a dense oracle.

Run this file with CUDA-Q 0.16 and the source package on PYTHONPATH.
"""

# [Begin Documentation]
import numpy as np
import cudaq
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import RetainedQROMChain


def main():
    cudaq.set_target("qpp-cpu", precision="fp64")
    tables = ((1, 3, 2), (2, 1, 3, 0), (3, 2, 1))
    angles = (0.19, -0.27, 0.31)
    chain = RetainedQROMChain(tables,
                              2,
                              2,
                              block_size=2,
                              controlled=True,
                              dirty_banks=True)
    begin, end = chain.begin_kernel, chain.end_kernel
    first, second, third, last = chain.kernels()
    # Address bits 0:2; word 2:4; data 4; control 5; reference 6;
    # clean prefix (including cache) then borrowed suffix from bit 7.
    ladder_start = 7
    ladder_stop = ladder_start + chain.num_ladder
    clean_mask = ((1 << chain.num_clean) - 1) << ladder_start
    indices = np.arange(1 << ladder_stop)
    rng = np.random.default_rng(812)
    initial = rng.normal(
        size=len(indices)) + 1j * rng.normal(size=len(indices))
    initial[(indices & (clean_mask | (3 << 2))) != 0] = 0
    initial /= np.linalg.norm(initial)
    # The borrowed bank has arbitrary joint entanglement with the address,
    # data, control and reference; the output word and clean prefix are zero.

    @cudaq.kernel
    def experiment(state: cudaq.State):
        q = cudaq.qvector(state)
        begin(q[5:6], q[0:2], q[ladder_start:ladder_stop], q[2:4])
        first(q[5:6], q[0:2], q[ladder_start:ladder_stop], q[2:4])
        ry.ctrl(0.19, q[2], q[4])
        ry.ctrl(0.38, q[3], q[4])
        second(q[5:6], q[0:2], q[ladder_start:ladder_stop], q[2:4])
        rz.ctrl(-0.27, q[2], q[4])
        rz.ctrl(-0.54, q[3], q[4])
        third(q[5:6], q[0:2], q[ladder_start:ladder_stop], q[2:4])
        rx.ctrl(0.31, q[2], q[4])
        rx.ctrl(0.62, q[3], q[4])
        last(q[5:6], q[0:2], q[ladder_start:ladder_stop], q[2:4])
        end(q[5:6], q[0:2], q[ladder_start:ladder_stop], q[2:4])
        # Address/control changes become legal after end, before a new begin.

    paulis = (np.array([[0, -1j],
                        [1j, 0]]), np.diag([1, -1]), np.array([[0, 1], [1,
                                                                        0]]))
    expected = initial.copy()
    # For each address/control block, apply the intended 2x2 data rotations.
    # This reference uses original tables, not XOR differences or QROM plans.
    for index in range(len(initial)):
        if (index >> 4) & 1:
            continue
        address, control = index & 3, (index >> 5) & 1
        rotation = np.eye(2, dtype=complex)
        for table, angle, pauli in zip(tables, angles, paulis):
            word = table[address] if address < len(table) else 0
            theta = control * word * angle
            step = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(
                theta / 2) * pauli
            rotation = step @ rotation
        pair = [index, index | (1 << 4)]
        expected[pair] = rotation @ initial[pair]

    actual = np.asarray(cudaq.get_state(experiment, state_from(initial)))
    error = np.max(np.abs(actual - expected))
    assert error < 1e-12  # No phase alignment, postselection or renormalization.
    assert np.max(np.abs(actual[(indices
                                 & (clean_mask | (3 << 2))) != 0])) < 1e-12
    resources = chain.resources
    print("target: qpp-cpu, fp64; raw complex-vector tolerance: 1e-12")
    print(f"max full complex-vector error: {error:.3e}")
    print(f"epoch: begin, {len(chain.kernels())} difference steps, end")
    print(
        f"workspace: {chain.num_clean} clean (including {chain.num_cache} cache) "
        f"+ {chain.num_dirty} borrowed; peak {chain.num_ladder}")
    print(f"complete epoch: {resources.toffoli_count} logical Toffolis, "
          f"{resources.temporary_and_count} temporary ANDs, "
          f"{resources.num_measurements} measurements")
    print(
        "output and clean work returned to zero; bank/reference correlations checked"
    )


if __name__ == "__main__":
    main()
