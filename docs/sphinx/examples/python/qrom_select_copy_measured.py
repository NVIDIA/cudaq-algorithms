# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured packet XOR and image-only erasure, checked on complex states.

Run this file with CUDA-Q 0.16 and the source package on PYTHONPATH.
"""

# [Begin Documentation]
import numpy as np
import cudaq
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import MeasuredQROM


def main():
    cudaq.set_target("qpp-cpu", precision="fp64")
    table = (9, 3, 14)  # The fourth address is padded with zero.
    lookup = MeasuredQROM(table,
                          2,
                          4,
                          variant="select_copy",
                          block_size=2,
                          packet_width=3,
                          controlled=True,
                          dirty_banks=True)
    load, erase = lookup.kernel, lookup.erase_kernel

    # Little-endian order: address, output, control, reference, load ladder,
    # erase ladder. Separate ladders avoid incompatible clean-prefix widths.
    load_start = 8
    load_stop = load_start + lookup.num_ladder
    erase_stop = load_stop + lookup.num_erase_ladder
    clean_bits = tuple(range(load_start, load_start + lookup.num_clean))
    clean_bits += tuple(range(load_stop, load_stop + lookup.num_erase_clean))
    indices = np.arange(1 << erase_stop)
    clean_mask = sum(1 << bit for bit in clean_bits)
    rng = np.random.default_rng(811)
    arbitrary = rng.normal(
        size=len(indices)) + 1j * rng.normal(size=len(indices))
    arbitrary[(indices & clean_mask) != 0] = 0
    arbitrary /= np.linalg.norm(arbitrary)
    # Complex amplitudes jointly entangle address, output, both borrowed
    # suffixes, control and reference; only declared clean work starts zero.
    image_input = arbitrary.copy()
    image_input[((indices >> 2) & 15) != 0] = 0
    image_input /= np.linalg.norm(image_input)

    @cudaq.kernel
    def apply_load(state: cudaq.State):
        q = cudaq.qvector(state)
        load(q[6:7], q[0:2], q[load_start:load_stop], q[2:6])

    @cudaq.kernel
    def load_then_erase(state: cudaq.State):
        q = cudaq.qvector(state)
        load(q[6:7], q[0:2], q[load_start:load_stop], q[2:6])
        # Legal only because output now holds control * table[address].
        erase(q[6:7], q[0:2], q[load_stop:erase_stop], q[2:6])

    def xor_reference(state):
        # Independent basis permutation; no planner or generated circuit.
        expected = np.zeros_like(state)
        for index, amplitude in enumerate(state):
            address = index & 3
            word = table[address] if address < len(table) else 0
            if not ((index >> 6) & 1):
                word = 0
            expected[index ^ (word << 2)] = amplitude
        return expected

    xor_state = np.asarray(cudaq.get_state(apply_load, state_from(arbitrary)))
    image_state = np.asarray(
        cudaq.get_state(apply_load, state_from(image_input)))
    erased = np.asarray(
        cudaq.get_state(load_then_erase, state_from(image_input)))
    errors = (np.max(np.abs(xor_state - xor_reference(arbitrary))),
              np.max(np.abs(image_state - xor_reference(image_input))),
              np.max(np.abs(erased - image_input)))
    # Compare raw complex amplitudes: no phase alignment or renormalization.
    assert max(errors) < 1e-12
    print("target: qpp-cpu, fp64; raw complex-vector tolerance: 1e-12")
    print(f"fixed packet widths: {lookup.packets.widths}")
    print("balanced alpha=2 would have widths (2, 2)")
    print("max errors (arbitrary XOR, image load, image erasure): " +
          ", ".join(f"{error:.3e}" for error in errors))
    for name, resources in (("load", lookup.resources),
                            ("erase", lookup.erase_resources)):
        print(
            f"{name}: {resources.num_clean} clean + {resources.num_dirty} borrowed; "
            f"{resources.toffoli_count} logical Toffolis, "
            f"{resources.temporary_and_count} temporary ANDs, "
            f"{resources.num_measurements} measurements, "
            f"{resources.classical_bits} persistent outcome bits")
    print(
        f"peak allocated workspace: {lookup.num_ladder + lookup.num_erase_ladder}; "
        f"total qubits including control/reference: {erase_stop}")


if __name__ == "__main__":
    main()
