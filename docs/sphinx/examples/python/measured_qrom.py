# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured clean QROM: verify a complex XOR map, then read a classical bit.

Run as a file with CUDA-Q 0.16: ``python measured_qrom.py``.
"""

import numpy as np
import cudaq
from cudaq_algorithms.primitives import MeasuredQROM


def main():
    cudaq.set_target("qpp-cpu")
    data = [3, 0, 1, 2, 1]
    lookup = MeasuredQROM(data, 3, 2, variant="select_swap", block_size=2)
    apply_lookup = lookup.kernel
    n_ladder = lookup.num_ladder
    rng = np.random.default_rng(27)
    psi = rng.normal(size=32) + 1j * rng.normal(size=32)
    psi /= np.linalg.norm(psi)
    initial = cudaq.State.from_data(psi)

    @cudaq.kernel
    def experiment(state: cudaq.State):
        logical = cudaq.qvector(state)
        ladder = cudaq.qvector(n_ladder)
        apply_lookup(logical[0:3], ladder, logical[3:5])

    # Address occupies the low three bits, then the output, then workspace.
    # The independent reference preserves phases and pads missing entries.
    reference = np.zeros(1 << (5 + n_ladder), complex)
    for i, amplitude in enumerate(psi):
        address = i & 7
        word = data[address] if address < len(data) else 0
        reference[i ^ (word << 3)] = amplitude
    actual = np.asarray(cudaq.get_state(experiment, initial))
    error = np.max(np.abs(actual - reference))
    assert error < 1e-12

    @cudaq.kernel
    def read_bit() -> bool:
        address = cudaq.qvector(3)
        ladder = cudaq.qvector(n_ladder)
        output = cudaq.qvector(2)
        x(address[0])
        x(address[1])
        apply_lookup(address, ladder, output)
        return mz(output[1])  # data[3] == 2: bit 1 is always one.

    values = list(cudaq.run(read_bit, shots_count=16))
    assert values == [True] * 16
    print(lookup)
    print(f"max complex-vector error: {error:.3e}")
    print(
        f"clean ancillas: {lookup.num_ladder}; tree ANDs: {lookup.tree_and_count}"
    )
    print(f"data[3] bit 1: {sum(values)}/{len(values)} ones")


if __name__ == "__main__":
    main()
