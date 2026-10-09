# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured lookups retain the full complex XOR map on clean workspace."""

import numpy as np
import pytest
import cudaq
from cudaq_algorithms import primitives


def _make(*args, **kwargs):
    assert hasattr(primitives,
                   "MeasuredQROM"), "explicit measured lookup API missing"
    return primitives.MeasuredQROM(*args, **kwargs)


@pytest.mark.parametrize("variant,block_size", [("select", None),
                                                ("select_swap", 2),
                                                ("select_swap", 4)])
@pytest.mark.parametrize("controlled", [False, True])
def test_measured_qrom_complex_xor_map(variant, block_size, controlled):
    # Missing fixup, wrong low/high address split, missing XOR, or discarded
    # padding amplitudes all change this independent permutation oracle.
    data = [3, 0, 1, 2, 1]
    qrom = _make(data,
                 3,
                 2,
                 variant=variant,
                 block_size=block_size,
                 controlled=controlled)
    n_control = int(controlled)
    logical_bits = n_control + 3 + 2
    rng = np.random.default_rng(421)
    logical = rng.normal(
        size=1 << logical_bits) + 1j * rng.normal(size=1 << logical_bits)
    logical /= np.linalg.norm(logical)
    state = cudaq.State.from_data(logical)
    lookup = qrom.kernel
    n_ladder = qrom.num_ladder

    if controlled:

        @cudaq.kernel
        def run(initial: cudaq.State):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(n_ladder)
            lookup(q[0:1], q[1:4], ladder, q[4:6])
    else:

        @cudaq.kernel
        def run(initial: cudaq.State):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(n_ladder)
            lookup(q[0:3], ladder, q[3:5])

    expected = np.zeros(1 << (logical_bits + n_ladder), complex)
    for i, amp in enumerate(logical):
        address = (i >> n_control) & 7
        enabled = not controlled or i & 1
        word = data[address] if enabled and address < len(data) else 0
        expected[i ^ (word << (n_control + 3))] = amp
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(cudaq.get_state(run, state)),
                                   expected,
                                   atol=1e-12)


def test_measured_qrom_resource_choice_and_tree_split():
    qrom = _make([0, 1] * 32, 6, 1)
    assert qrom.variant == "select_swap"
    candidates = [_make([0, 1] * 32, 6, 1, variant="select")]
    candidates += [
        _make([0, 1] * 32, 6, 1, variant="select_swap", block_size=b)
        for b in (2, 4, 8, 16, 32)
    ]
    assert qrom.toffoli_count == min(q.toffoli_count for q in candidates)
    # B=4: two uncontrolled 16-leaf walks, plus 6 routing Toffolis.
    blocked = candidates[2]
    assert blocked.num_ladder == 8
    assert blocked.tree_and_count == blocked.num_measurements == 28
    assert blocked.toffoli_count == 34
    controlled = _make([0, 1] * 32,
                       6,
                       1,
                       variant="select_swap",
                       block_size=4,
                       controlled=True)
    assert controlled.tree_and_count == 30
    assert controlled.toffoli_count == 36


def test_measured_qrom_normalizes_iterable_and_integral_configuration():
    qrom = _make((v for v in [1.0, 2.0, 3.0]),
                 3.0,
                 2.0,
                 variant="select_swap",
                 block_size=2.0)
    assert qrom.data == (1, 2, 3)
    assert qrom.num_address == 3 and qrom.num_output == 2
    assert qrom.block_size == 2


@pytest.mark.parametrize("data,a,b,kwargs", [
    ([], 2, 2, {}),
    ([0.5], 2, 2, {}),
    ([-1], 2, 2, {}),
    ([4], 2, 2, {}),
    ([0] * 5, 2, 2, {}),
    ([0], 0, 2, {}),
    ([0], 2, 0, {}),
    ([0], 2, 2, {
        "variant": "bad"
    }),
    ([0], 2, 2, {
        "block_size": 2
    }),
    ([0], 1, 1, {
        "variant": "select_swap"
    }),
    ([0], 3, 1, {
        "variant": "select_swap",
        "block_size": 3
    }),
    ([0], 3, 1, {
        "variant": "select_swap",
        "block_size": 8
    }),
])
def test_measured_qrom_rejects_invalid_input(data, a, b, kwargs):
    with pytest.raises(ValueError):
        _make(data, a, b, **kwargs)


@pytest.mark.parametrize("variant,block_size,dirty",
                         [("select", None, False), ("select_swap", 2, False),
                          ("select_swap", 2, True)])
@pytest.mark.parametrize("controlled", [False, True])
def test_measured_qrom_counts_match_compiler(variant, block_size, dirty,
                                             controlled):
    qrom = _make([3, 0, 1, 2, 1],
                 3,
                 2,
                 variant=variant,
                 block_size=block_size,
                 controlled=controlled,
                 dirty_banks=dirty)
    lookup = qrom.kernel
    n_ladder = qrom.num_ladder
    if controlled:

        @cudaq.kernel
        def run():
            control = cudaq.qvector(1)
            address = cudaq.qvector(3)
            ladder = cudaq.qvector(n_ladder)
            output = cudaq.qvector(2)
            lookup(control, address, ladder, output)
    else:

        @cudaq.kernel
        def run():
            address = cudaq.qvector(3)
            ladder = cudaq.qvector(n_ladder)
            output = cudaq.qvector(2)
            lookup(address, ladder, output)

    resources = cudaq.estimate_resources(run)
    assert resources.count_controls("x", 2) == qrom.toffoli_count
    assert resources.count("mz") == qrom.num_measurements


@pytest.mark.parametrize("variant", ["select", "select_swap"])
def test_measured_qrom_run_returning_entry(variant):
    qrom = _make([2, 0, 1], 2, 2, variant=variant)
    lookup = qrom.kernel
    n_ladder = qrom.num_ladder

    @cudaq.kernel
    def read_bit() -> bool:
        address = cudaq.qvector(2)
        ladder = cudaq.qvector(n_ladder)
        output = cudaq.qvector(2)
        x(address[1])
        lookup(address, ladder, output)
        return mz(output[0])

    assert list(cudaq.run(read_bit, shots_count=16)) == [True] * 16


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("n,m,b,block_size", [(2, 3, 2, 2), (3, 5, 1, 2),
                                              (3, 5, 1, 4)])
def test_dirty_bank_echo_preserves_entanglement(controlled, n, m, b,
                                                block_size):
    # A random joint state includes all control/address/output/bank bits and
    # an untouched reference. Clean walk bits alone are projected to zero.
    # This detects borrowed-bank corruption, phases, padding, and control-off
    # leakage, including when the bank is entangled with a spectator.
    data = [(3 * i + 1) % (1 << b) for i in range(m)]
    qrom = _make(data,
                 n,
                 b,
                 variant="select_swap",
                 block_size=block_size,
                 controlled=controlled,
                 dirty_banks=True)
    c = int(controlled)
    clean = n - (block_size.bit_length() - 1)
    dirty = block_size * b
    assert qrom.num_clean == clean
    assert qrom.num_dirty == dirty
    assert qrom.num_ladder == clean + dirty
    ladder_start = c + n
    output_start = ladder_start + clean + dirty
    total = output_start + b + 1
    ids = np.arange(1 << total)
    rng = np.random.default_rng(738)
    psi = rng.normal(size=1 << total) + 1j * rng.normal(size=1 << total)
    psi[((ids >> ladder_start) & ((1 << clean) - 1)) != 0] = 0
    psi /= np.linalg.norm(psi)
    expected = np.zeros_like(psi)
    for i, amplitude in enumerate(psi):
        address = (i >> c) & ((1 << n) - 1)
        word = data[address] if address < m and (not c or i & 1) else 0
        expected[i ^ (word << output_start)] = amplitude
    initial = cudaq.State.from_data(psi)
    kernel = qrom.kernel
    if controlled:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            kernel(q[0:1], q[1:ladder_start], q[ladder_start:output_start],
                   q[output_start:output_start + b])
    else:

        @cudaq.kernel
        def run(state: cudaq.State):
            q = cudaq.qvector(state)
            kernel(q[0:ladder_start], q[ladder_start:output_start],
                   q[output_start:output_start + b])

    for _ in range(3):
        np.testing.assert_allclose(np.asarray(cudaq.get_state(run, initial)),
                                   expected,
                                   atol=1e-12)


@pytest.mark.parametrize("variant", ["auto", "select"])
def test_dirty_banks_require_explicit_blocked_variant(variant):
    with pytest.raises(ValueError, match="dirty_banks.*select_swap"):
        _make([0, 1], 2, 1, variant=variant, dirty_banks=True)


def test_dirty_bank_cost_selection_and_counts():
    data = [0, 1] * 32
    qrom = _make(data,
                 6,
                 1,
                 variant="select_swap",
                 block_size=4,
                 dirty_banks=True)
    assert qrom.toffoli_count == 40  # 28 tree + 12 coherent routing.
    assert qrom.tree_and_count == qrom.num_measurements == 28
    assert qrom.num_clean == 4 and qrom.num_dirty == 4
    auto_block = _make(data, 6, 1, variant="select_swap", dirty_banks=True)
    candidates = [
        _make(data,
              6,
              1,
              variant="select_swap",
              block_size=b,
              dirty_banks=True) for b in (2, 4, 8, 16, 32)
    ]
    assert auto_block.toffoli_count == min(q.toffoli_count for q in candidates)
