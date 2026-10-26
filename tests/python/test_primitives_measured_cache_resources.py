# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler totals and an independent address-coherence witness for cache erasure."""
import numpy as np
import pytest
import cudaq
from cudaq import spin
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import MeasuredQROM
from measured_harness import resources_of


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("phase_size", [1, 2])
def test_cache_counts_match_compiler(controlled, phase_size):
    lookup = MeasuredQROM([3, 0, 1, 2, 1],
                          3,
                          2,
                          variant="select_swap",
                          block_size=2,
                          controlled=controlled,
                          cache_uncompute="measure",
                          phase_block_size=phase_size)
    widths = ([1] if controlled else []) + [3, lookup.num_ladder, 2]
    compiled = resources_of(lookup.kernel, *widths)
    assert compiled.count_controls("x", 2) == lookup.toffoli_count
    assert compiled.count("mz") == lookup.num_measurements


def test_cache_address_coherence_observe():
    data = [0, 0, 1, 1, 0]
    lookup = MeasuredQROM(data,
                          3,
                          1,
                          variant="select_swap",
                          block_size=2,
                          cache_uncompute="measure",
                          phase_block_size=2)
    apply_lookup = lookup.kernel
    n_ladder = lookup.num_ladder

    @cudaq.kernel
    def run(initial: cudaq.State):
        logical = cudaq.qvector(initial)
        ladder = cudaq.qvector(n_ladder)
        apply_lookup(logical[0:3], ladder, logical[3:4])

    # Independent nonuniform, complex product input and classical XOR oracle.
    psi = np.array([1.0 + 0j])
    for angle, phase in [(0.29, 0.37), (0.63, 0.19), (0.44, 0.23),
                         (0.32, 0.41)]:
        psi = np.kron([np.cos(angle), np.exp(1j * phase) * np.sin(angle)], psi)
    expected = np.zeros(16, complex)
    for i, amplitude in enumerate(psi):
        address = i & 7
        word = data[address] if address < len(data) else 0
        expected[i ^ (word << 3)] = amplitude
    # X on high address bit 1, computed without CUDA-Q.
    witness = sum(np.conj(expected[i ^ 2]) * expected[i]
                  for i in range(16)).real
    assert abs(witness) > 0.1
    for seed in [13, 51, 97]:
        cudaq.set_random_seed(seed)
        observed = cudaq.observe(run, spin.x(1), state_from(psi)).expectation()
        assert abs(observed - witness) < 1e-12
