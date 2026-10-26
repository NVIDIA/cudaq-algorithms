# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured lookups retain the full complex XOR map on clean workspace."""

import re

import numpy as np
import pytest
import cudaq

from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import MeasuredQROM
from cudaq_algorithms.primitives._qrom_measured import (
    _MEASURED_ADDRESS_OPERANDS, _select_swap_cost)
from cudaq_algorithms.primitives._unary_iteration import _OP_DESCRIPTIONS
from cudaq_algorithms.primitives._unary_iteration_measured import (
    _measured_walk_toffoli_count)
from dense_references import random_ket
from measured_harness import resources_of


def _xor_permutation(logical, data, address_bits, n_control, n_ladder):
    """Independent oracle: XOR data[address] into the output for every
    amplitude of ``logical`` (control, address, output; little-endian),
    leaving the clean ladder at zero."""
    n_logical = int(np.log2(len(logical)))
    expected = np.zeros(1 << (n_logical + n_ladder), complex)
    for i, amp in enumerate(logical):
        address = (i >> n_control) & ((1 << address_bits) - 1)
        enabled = not n_control or i & 1
        word = data[address] if enabled and address < len(data) else 0
        expected[i ^ (word << (n_control + address_bits))] = amp
    return expected


@pytest.mark.parametrize("variant,block_size,address_bits",
                         [("select", None, 3), ("select_swap", 2, 3),
                          ("select_swap", 4, 4)])
@pytest.mark.parametrize("controlled", [False, True])
def test_measured_qrom_complex_xor_map(variant, block_size, address_bits,
                                       controlled):
    # Missing fixup, wrong low/high address split, missing XOR, or discarded
    # padding amplitudes all change this independent permutation oracle.
    # Every shape here emits at least one measurement (block_size=4 at
    # three address bits would leave a linear root and no gadget).
    cudaq.set_random_seed(421)
    data = [3, 0, 1, 2, 1]
    qrom = MeasuredQROM(data,
                        address_bits,
                        2,
                        variant=variant,
                        block_size=block_size,
                        controlled=controlled)
    assert qrom.num_measurements > 0
    n_control = int(controlled)
    logical = random_ket(n_control + address_bits + 2, 421)
    lookup = qrom.kernel
    n_ladder = qrom.num_ladder
    a_end = n_control + address_bits

    if controlled:

        @cudaq.kernel
        def run(initial: cudaq.State):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(n_ladder)
            lookup(q[0:1], q[1:a_end], ladder, q[a_end:a_end + 2])
    else:

        @cudaq.kernel
        def run(initial: cudaq.State):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(n_ladder)
            lookup(q[0:a_end], ladder, q[a_end:a_end + 2])

    expected = _xor_permutation(logical, data, address_bits, n_control,
                                n_ladder)
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(logical))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


@pytest.mark.parametrize("variant,block_size,dirty,controlled", [
    ("select", None, False, False),
    ("select", None, False, True),
    ("select_swap", 2, False, False),
    ("select_swap", 2, True, True),
])
def test_measured_qrom_is_self_inverse(variant, block_size, dirty, controlled):
    # Applying the lookup twice is the identity on the declared workspace
    # subspace: clean prefix at zero, bank arbitrary when borrowed.
    cudaq.set_random_seed(77)
    data = [3, 0, 1, 2, 1]
    qrom = MeasuredQROM(data,
                        3,
                        2,
                        variant=variant,
                        block_size=block_size,
                        controlled=controlled,
                        dirty_banks=dirty)
    c = int(controlled)
    ladder_start = c + 3
    output_start = ladder_start + qrom.num_ladder
    total = output_start + 2
    psi = random_ket(total, 77)
    ids = np.arange(1 << total)
    clean_mask = ((ids >> ladder_start) & ((1 << qrom.num_clean) - 1)) != 0
    psi[clean_mask] = 0
    psi /= np.linalg.norm(psi)
    kernel = qrom.kernel
    if controlled:

        @cudaq.kernel
        def twice(state: cudaq.State):
            q = cudaq.qvector(state)
            kernel(q[0:1], q[1:ladder_start], q[ladder_start:output_start],
                   q[output_start:total])
            kernel(q[0:1], q[1:ladder_start], q[ladder_start:output_start],
                   q[output_start:total])
    else:

        @cudaq.kernel
        def twice(state: cudaq.State):
            q = cudaq.qvector(state)
            kernel(q[0:ladder_start], q[ladder_start:output_start],
                   q[output_start:total])
            kernel(q[0:ladder_start], q[ladder_start:output_start],
                   q[output_start:total])

    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(twice, state_from(psi))),
                               psi,
                               atol=1e-12,
                               rtol=0)


@pytest.mark.parametrize(
    "dirty,expected_toffolis,expected_clean,expected_dirty",
    [(False, 34, 8, 0), (True, 40, 4, 4)])
def test_measured_qrom_resource_choice_and_tree_split(dirty, expected_toffolis,
                                                      expected_clean,
                                                      expected_dirty):
    data = [0, 1] * 32
    auto = MeasuredQROM(data,
                        6,
                        1,
                        variant="auto" if not dirty else "select_swap",
                        dirty_banks=dirty)
    candidates = [] if dirty else [MeasuredQROM(data, 6, 1, variant="select")]
    candidates += [
        MeasuredQROM(data,
                     6,
                     1,
                     variant="select_swap",
                     block_size=b,
                     dirty_banks=dirty) for b in (2, 4, 8, 16, 32)
    ]
    if not dirty:
        candidates += [
            MeasuredQROM(data, 6, 1, variant="select_copy", block_size=b)
            for b in (2, 4, 8, 16, 32)
        ]
    assert auto.toffoli_count == min(q.toffoli_count for q in candidates)
    # B=4: two uncontrolled 16-leaf walks, plus 6 (12 dirty) routing
    # Toffolis.
    blocked = MeasuredQROM(data,
                           6,
                           1,
                           variant="select_swap",
                           block_size=4,
                           dirty_banks=dirty)
    assert blocked.num_ladder == 8
    assert blocked.num_clean == expected_clean
    assert blocked.num_dirty == expected_dirty
    assert blocked.tree_and_count == blocked.num_measurements == 28
    assert blocked.toffoli_count == expected_toffolis
    controlled = MeasuredQROM(data,
                              6,
                              1,
                              variant="select_swap",
                              block_size=4,
                              controlled=True,
                              dirty_banks=dirty)
    assert controlled.tree_and_count == 30
    assert controlled.toffoli_count == expected_toffolis + 2


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("block_size", [2, 4, 8, 16, 32])
def test_measured_qrom_select_swap_tape_matches_formula(
        block_size, dirty, controlled):
    # The closed-form price that drives ``auto`` must equal the tape
    # recount, for every candidate, in both bank modes, with and without
    # the external control.
    data = [0, 1] * 32
    qrom = MeasuredQROM(data,
                        6,
                        1,
                        variant="select_swap",
                        block_size=block_size,
                        controlled=controlled,
                        dirty_banks=dirty)
    assert qrom.toffoli_count == _select_swap_cost(6, 64, 1, block_size,
                                                   controlled, dirty)
    high_bits = 6 - (block_size.bit_length() - 1)
    blocks = -(-64 // block_size)
    walks = 2 * _measured_walk_toffoli_count(high_bits, blocks, controlled)
    assert qrom.num_measurements == qrom.tree_and_count == walks
    assert qrom.describe().count("# Toffoli") == qrom.toffoli_count


def test_measured_qrom_normalizes_iterable_and_integral_configuration():
    qrom = MeasuredQROM((v for v in [1.0, 2.0, 3.0]),
                        3.0,
                        2.0,
                        variant="select_swap",
                        block_size=2.0)
    assert qrom.data == (1, 2, 3)
    assert qrom.num_address == 3 and qrom.num_output == 2
    assert qrom.block_size == 2


@pytest.mark.parametrize("data,a,b,kwargs,reason", [
    ([], 2, 2, {}, "non-empty"),
    ([0.5], 2, 2, {}, "integers"),
    (iter([0, 1.5]), 2, 2, {}, "integers"),
    ([-1], 2, 2, {}, "non-negative"),
    ([4], 2, 2, {}, "does not fit"),
    ([0] * 5, 2, 2, {}, "addresses only 4"),
    ([0], 0, 2, {}, "address_bits must"),
    ([0], 2, 0, {}, "output_bits must"),
    ([0], 2, 2, {
        "variant": "bad"
    }, "variant must be one of"),
    ([0], 2, 2, {
        "block_size": 2
    }, "block_size is only valid"),
    ([0], 1, 1, {
        "variant": "select_swap"
    }, "requires address_bits >= 2"),
    ([0], 3, 1, {
        "variant": "select_swap",
        "block_size": 3
    }, "power of two"),
    ([0], 3, 1, {
        "variant": "select_swap",
        "block_size": 8
    }, "needs at least one"),
    ([0, 1], 2, 1, {
        "variant": "select",
        "dirty_banks": True
    }, "dirty_banks=True requires"),
    ([0, 1, 2, 3], 2, 2, {
        "controlled": "False"
    }, "controlled must be a bool"),
    ([0, 1, 2, 3], 2, 2, {
        "variant": "select_swap",
        "dirty_banks": 1
    }, "dirty_banks must be a bool"),
])
def test_measured_qrom_rejects_invalid_input(data, a, b, kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        MeasuredQROM(data, a, b, **kwargs)


@pytest.mark.parametrize("variant,block_size,dirty",
                         [("select", None, False), ("select_swap", 2, False),
                          ("select_swap", 2, True)])
@pytest.mark.parametrize("controlled", [False, True])
def test_measured_qrom_counts_match_compiler(variant, block_size, dirty,
                                             controlled):
    qrom = MeasuredQROM([3, 0, 1, 2, 1],
                        3,
                        2,
                        variant=variant,
                        block_size=block_size,
                        controlled=controlled,
                        dirty_banks=dirty)
    widths = ([1] if controlled else []) + [3, qrom.num_ladder, 2]
    resources = resources_of(qrom.kernel, *widths)
    assert resources.count_controls("x", 2) == qrom.toffoli_count
    assert resources.count("mz") == qrom.num_measurements


@pytest.mark.parametrize("variant,data,address_bits",
                         [("select", [2, 0, 1], 2),
                          ("select_swap", [2, 0, 1, 3, 1], 3)])
def test_measured_qrom_run_returning_entry(variant, data, address_bits):
    qrom = MeasuredQROM(data, address_bits, 2, variant=variant)
    assert qrom.num_measurements > 0  # the run really crosses a gadget
    lookup = qrom.kernel
    n_ladder = qrom.num_ladder

    @cudaq.kernel
    def read_bit() -> bool:
        address = cudaq.qvector(address_bits)
        ladder = cudaq.qvector(n_ladder)
        output = cudaq.qvector(2)
        x(address[1])  # address 2: data[2] == 1, bit 0 is set
        lookup(address, ladder, output)
        return mz(output[0])

    assert list(cudaq.run(read_bit, shots_count=16)) == [True] * 16


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("n,m,b,block_size", [(3, 6, 2, 2), (3, 5, 1, 2),
                                              (4, 9, 1, 4)])
def test_dirty_bank_echo_preserves_entanglement(controlled, n, m, b,
                                                block_size):
    # A random joint state includes all control/address/output/bank bits and
    # an untouched reference. Clean walk bits alone are projected to zero.
    # This detects borrowed-bank corruption, phases, padding, and control-off
    # leakage, including when the bank is entangled with a spectator. Every
    # shape keeps at least two high address bits so a gadget is emitted.
    cudaq.set_random_seed(738)
    data = [(3 * i + 1) % (1 << b) for i in range(m)]
    qrom = MeasuredQROM(data,
                        n,
                        b,
                        variant="select_swap",
                        block_size=block_size,
                        controlled=controlled,
                        dirty_banks=True)
    assert qrom.num_measurements > 0
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
    psi = random_ket(total, 738)
    psi[((ids >> ladder_start) & ((1 << clean) - 1)) != 0] = 0
    psi /= np.linalg.norm(psi)
    expected = np.zeros_like(psi)
    for i, amplitude in enumerate(psi):
        address = (i >> c) & ((1 << n) - 1)
        word = data[address] if address < m and (not c or i & 1) else 0
        expected[i ^ (word << output_start)] = amplitude
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
        np.testing.assert_allclose(np.asarray(
            cudaq.get_state(run, state_from(psi))),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


def test_address_operand_table_covers_every_address_opcode():
    # The SELECT-SWAP rebase shifts every operand that indexes the address
    # register; an opcode missing from the table would be rebased onto a
    # routing wire silently. Derive the expected positions from the
    # description templates themselves.
    for opcode, text in _OP_DESCRIPTIONS.items():
        expected = tuple("abc".index(m)
                         for m in re.findall(r"address\[\{([abc])\}\]", text))
        if expected or opcode in _MEASURED_ADDRESS_OPERANDS:
            assert _MEASURED_ADDRESS_OPERANDS.get(opcode,
                                                  ()) == expected, opcode


def test_measured_qrom_describe_and_independent_tree_count():
    qrom = MeasuredQROM([0, 1] * 32, 6, 1, variant="select_swap", block_size=4)
    text = qrom.describe()
    assert text.splitlines()[0].startswith("MeasuredQROM lookup:")
    assert text.count("# Toffoli") == qrom.toffoli_count == 34
    assert text.count("# measured uncompute") == qrom.num_measurements == 28
    # Two measured walks over 16 blocks on 4 high bits: 2 * (16 + 4 - 2 - 0)
    # tree ANDs, counted from the walk tape, not copied from the
    # measurement count.
    assert qrom.tree_and_count == 28
