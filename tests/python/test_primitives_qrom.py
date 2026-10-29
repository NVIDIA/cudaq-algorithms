# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the QROM lookup (select and select_swap variants).

Correctness is pinned exhaustively: for random tables every address is
read out via basis-state extraction (`cudaq.get_state`), including the
out-of-range addresses of non-power-of-two tables (which must read
zero), for both constructions. Both variants are exactly self-inverse —
that property is tested on superposed addresses instead of a duplicate
adjoint kernel — and the SELECT-SWAP lookup is checked to be
behaviorally identical to the plain unary-iteration lookup on the same
table (equal (address, output) amplitudes with the clean ancillas traced
as |0>).
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import QROM


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


# Register layout in the harnesses: address at [0, A), ladder at
# [A, A + L), output at [A + L, A + L + W) — so the expected state on
# reading address k is the basis vector k + (data[k] << (A + L)), the
# ladder having returned to 0.


def _qrom_readout_state(qrom: QROM, address: int) -> np.ndarray:
    lookup = qrom.kernel
    num_address = qrom.num_address
    num_ladder = qrom.num_ladder
    num_output = qrom.num_output

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(num_address)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(num_output)
        for b in range(num_address):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        lookup(address_reg, ladder, output)

    return np.array(cudaq.get_state(run))


def _exhaustive_readout(qrom: QROM, data):
    total = qrom.num_address + qrom.num_ladder + qrom.num_output
    shift = qrom.num_address + qrom.num_ladder
    for address in range(1 << qrom.num_address):
        # Out-of-range addresses of a short table must read zero.
        expected_word = data[address] if address < len(data) else 0
        state = _qrom_readout_state(qrom, address)
        expected = _basis(address + (expected_word << shift), total)
        np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("address_bits,num_entries", [(1, 2), (2, 4), (2, 3),
                                                      (3, 8), (3, 5), (4, 16),
                                                      (4, 11)])
@pytest.mark.parametrize("output_bits", [1, 3, 6])
def test_qrom_select_exhaustive_readout(address_bits, num_entries,
                                        output_bits):
    rng = np.random.default_rng(7 * address_bits + num_entries + output_bits)
    data = [int(v) for v in rng.integers(0, 1 << output_bits, num_entries)]
    qrom = QROM(data, address_bits, output_bits, variant="select")
    assert qrom.variant == "select"
    assert qrom.num_ladder == address_bits
    _exhaustive_readout(qrom, data)


@pytest.mark.parametrize("address_bits,num_entries,output_bits,block_size",
                         [(2, 4, 3, 2), (2, 3, 3, 2), (3, 8, 2, 2),
                          (3, 5, 2, 2), (3, 8, 1, 4), (4, 16, 1, 4),
                          (4, 11, 1, 2), (4, 9, 1, 8), (2, 2, 2, 2),
                          (3, 4, 1, 4)])
def test_qrom_select_swap_exhaustive_readout(address_bits, num_entries,
                                             output_bits, block_size):
    rng = np.random.default_rng(3 * address_bits + num_entries + block_size)
    data = [int(v) for v in rng.integers(0, 1 << output_bits, num_entries)]
    qrom = QROM(data,
                address_bits,
                output_bits,
                variant="select_swap",
                block_size=block_size)
    assert qrom.variant == "select_swap"
    assert qrom.block_size == block_size
    low_bits = block_size.bit_length() - 1
    assert qrom.num_ladder == (address_bits - low_bits +
                               block_size * output_bits)
    _exhaustive_readout(qrom, data)


@pytest.mark.parametrize("variant,block_size", [("select", None),
                                                ("select_swap", 2),
                                                ("select_swap", 4)])
def test_qrom_is_self_inverse(variant, block_size):
    # X-only write phase: applying the lookup twice XORs the table twice
    # (and the select_swap sandwich W S C S^-1 W squares to identity; at
    # B = 4 the routing network has two stages, so S^-1 is the genuinely
    # reversed Fredkin list, not S itself).
    # Run on all addresses at once (H on the address) with a non-zero
    # initial output word to pin |y XOR d XOR d> = |y>.
    data = [5, 3, 0, 6, 7]
    qrom = QROM(data,
                address_bits=3,
                output_bits=3,
                variant=variant,
                block_size=block_size)
    lookup = qrom.kernel
    num_ladder = qrom.num_ladder
    initial = 0b101

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            if ((initial >> b) & 1) == 1:
                x(output[b])
        lookup(address_reg, ladder, output)
        lookup(address_reg, ladder, output)

    total = 3 + num_ladder + 3
    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << total, dtype=np.complex128)
    for address in range(8):
        expected[address + (initial << (3 + num_ladder))] = 1.0 / np.sqrt(8.0)
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("block_size", [2, 4])
def test_qrom_select_swap_equals_select_on_superposed_addresses(block_size):
    # Same table through both constructions: the (address, output)
    # amplitudes must agree exactly, the ancillas being |0> in both.
    data = [3, 0, 5, 6, 1, 7]
    address_bits, output_bits = 3, 3

    def joint_amplitudes(qrom: QROM) -> np.ndarray:
        lookup = qrom.kernel
        num_ladder = qrom.num_ladder

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(address_bits)
            ladder = cudaq.qvector(num_ladder)
            output = cudaq.qvector(output_bits)
            for b in range(address_bits):
                h(address_reg[b])
            lookup(address_reg, ladder, output)

        state = np.array(cudaq.get_state(run))
        tensor = state.reshape(1 << output_bits, 1 << num_ladder,
                               1 << address_bits)
        # Ancillas must be exactly |0>.
        np.testing.assert_allclose(np.delete(tensor, 0, axis=1),
                                   0.0,
                                   atol=1e-12)
        return tensor[:, 0, :]

    select = QROM(data, address_bits, output_bits, variant="select")
    swap = QROM(data,
                address_bits,
                output_bits,
                variant="select_swap",
                block_size=block_size)
    np.testing.assert_allclose(joint_amplitudes(select),
                               joint_amplitudes(swap),
                               atol=1e-12)


@pytest.mark.controlled_composition
def test_qrom_select_swap_external_control_on_the_composition():
    # The promise behind building select_swap as composed kernels:
    # cudaq.control applied to the WHOLE composition — the parent kernel
    # that calls the walk and routing sub-kernels — implements the
    # controlled lookup. Pinned on a superposed control and superposed
    # addresses against the dense reference
    # |c>|k>|0...0>|y> -> |c>|k>|0...0>|y XOR c * data[k]>.
    data = [3, 0, 5, 6, 1]
    address_bits, output_bits = 3, 3
    qrom = QROM(data,
                address_bits,
                output_bits,
                variant="select_swap",
                block_size=2)
    lookup = qrom.kernel
    num_ladder = qrom.num_ladder
    theta = 2 * np.arccos(0.6)  # control amplitudes (0.6, 0.8)

    @cudaq.kernel
    def run():
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(address_bits)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(output_bits)
        ry(theta, control[0])
        for b in range(address_bits):
            h(address_reg[b])
        cudaq.control(lookup, control[0], address_reg, ladder, output)

    state = np.array(cudaq.get_state(run))
    total = 1 + address_bits + num_ladder + output_bits
    shift = 1 + address_bits + num_ladder
    amp = {0: 0.6, 1: 0.8}
    expected = np.zeros(1 << total, dtype=np.complex128)
    for c in range(2):
        for k in range(1 << address_bits):
            word = data[k] if (c == 1 and k < len(data)) else 0
            index = c + (k << 1) + (word << shift)
            expected[index] = amp[c] / np.sqrt(1 << address_bits)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_qrom_auto_dispatch_picks_the_cheaper_variant():
    # Big table, narrow output: routing is cheap, the block walk is a
    # fraction of the full walk -> select_swap wins.
    wide = QROM(list(range(2)) * 32, 6, 1)
    assert wide.variant == "select_swap"
    assert wide.block_size is not None
    # Tiny table, wide output: every swapped register costs output_bits
    # Toffolis per direction -> the plain walk wins.
    narrow = QROM([200, 3, 118, 25], 2, 8)
    assert narrow.variant == "select"
    assert narrow.block_size is None
    # Whatever auto picks must be priced no worse than the alternatives.
    for block_size in (2, 4, 8, 16, 32):
        forced = QROM(list(range(2)) * 32,
                      6,
                      1,
                      variant="select_swap",
                      block_size=block_size)
        assert wide.toffoli_count <= forced.toffoli_count
    assert wide.toffoli_count <= QROM(list(range(2)) * 32,
                                      6,
                                      1,
                                      variant="select").toffoli_count


def test_qrom_select_swap_default_block_size_is_cost_optimal():
    data = [int(v) for v in np.arange(32) % 4]
    auto_block = QROM(data, 5, 2, variant="select_swap")
    assert auto_block.block_size is not None
    for block_size in (2, 4, 8, 16):
        forced = QROM(data, 5, 2, variant="select_swap", block_size=block_size)
        assert auto_block.toffoli_count <= forced.toffoli_count


def test_qrom_toffoli_count_bound():
    data = list(range(16))
    qrom = QROM(data, address_bits=4, output_bits=4, variant="select")
    assert 0 < qrom.toffoli_count <= 2 * (len(data) - 1)
    short = QROM(data[:5], address_bits=4, output_bits=4, variant="select")
    assert short.toffoli_count < qrom.toffoli_count


def test_qrom_select_single_address_bit():
    # Regression for cuda-quantum#5280: a one-address-bit select lookup
    # has a Toffoli-free walk, so the interpreter's Toffoli dispatch arm
    # is classically dead — and CUDA-Q's CSE aliased the dead arm's
    # control/target wires into illegal IR (an infinite compiler loop
    # through 0.15, a hard compile error once fixed upstream). Dispatch
    # pruning removes the dead arm at mint time; this pins that the
    # minted lookup constructs and reads out correctly.
    data = [5, 2]
    qrom = QROM(data, address_bits=1, output_bits=3, variant="select")
    assert qrom.toffoli_count == 0
    _exhaustive_readout(qrom, data)


def test_qrom_validation_raises():
    with pytest.raises(ValueError, match="data must be non-empty"):
        QROM([], address_bits=2, output_bits=2)
    with pytest.raises(ValueError, match="addresses only 4"):
        QROM([1] * 5, address_bits=2, output_bits=2)
    with pytest.raises(ValueError, match="non-negative"):
        QROM([1, -2], address_bits=1, output_bits=2)
    with pytest.raises(ValueError, match="does not fit in output_bits"):
        QROM([1, 4], address_bits=1, output_bits=2)
    with pytest.raises(ValueError, match="address_bits must be a positive"):
        QROM([1], address_bits=0, output_bits=1)
    with pytest.raises(ValueError, match="output_bits must be a positive"):
        QROM([1], address_bits=1, output_bits=0)
    with pytest.raises(ValueError, match="entries must be integers"):
        QROM([1.5], address_bits=1, output_bits=2)
    # A single-pass iterable must get the same float rejection as a list
    # (regression: the check once ran on an already-exhausted iterator).
    with pytest.raises(ValueError, match="entries must be integers"):
        QROM((v for v in [1.5, 2]), address_bits=1, output_bits=2)
    with pytest.raises(ValueError, match="requires address_bits >= 2"):
        QROM([1, 2], address_bits=1, output_bits=2, variant="select_swap")
    # ... and a valid single-pass iterable must load fully.
    assert QROM((v for v in [1, 2]), address_bits=1,
                output_bits=2).data == (1, 2)
    with pytest.raises(ValueError, match="variant must be one of"):
        QROM([1, 2], address_bits=1, output_bits=2, variant="qroam")
    with pytest.raises(ValueError, match="only valid with variant="):
        QROM([1, 2], address_bits=1, output_bits=2, block_size=2)
    with pytest.raises(ValueError, match="power of two"):
        QROM([1] * 8,
             address_bits=3,
             output_bits=1,
             variant="select_swap",
             block_size=3)
    with pytest.raises(ValueError, match="at least one block-index"):
        QROM([1] * 8,
             address_bits=3,
             output_bits=1,
             variant="select_swap",
             block_size=8)
    # The control flag is a bool, not a truthy value: a stray string or
    # int would otherwise mint a controlled (or plain) lookup silently.
    with pytest.raises(ValueError, match="controlled must be a bool"):
        QROM([1, 2], address_bits=1, output_bits=2, controlled=1)
    with pytest.raises(ValueError, match="controlled must be a bool"):
        QROM([1, 2], address_bits=1, output_bits=2, controlled="no")


# ----------------------------------------------------------------------
# Externally controlled lookups
# ----------------------------------------------------------------------

from cudaq_algorithms.primitives._qrom import _select_swap_cost
from cudaq_algorithms.primitives._unary_iteration import _walk_toffoli_count

_CTRL_DATA = [5, 3, 0, 6, 7]

# (data, address_bits, output_bits, variant, block_size, preloaded output
# word): both variants on the fixture table, a one-bit address (select
# only), a single block-index bit (B = 2^(address_bits - 1)) and a
# zero-padded last block at B > 2.
_CTRL_GEOMETRIES = [
    pytest.param(_CTRL_DATA, 3, 3, "select", None, 0b101, id="select-3b-5e"),
    pytest.param(_CTRL_DATA, 3, 3, "select_swap", 2, 0b101,
                 id="swap-3b-5e-B2"),
    pytest.param([5, 2], 1, 3, "select", None, 0b011, id="select-1b-2e"),
    pytest.param([1, 3, 0, 2, 3],
                 3,
                 2,
                 "select_swap",
                 4,
                 0b01,
                 id="swap-3b-5e-B4-one-high-bit"),
    pytest.param([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                 4,
                 1,
                 "select_swap",
                 2,
                 0b1,
                 id="swap-4b-11e-B2"),
    pytest.param([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
                 4,
                 1,
                 "select_swap",
                 8,
                 0b1,
                 id="swap-4b-11e-B8-padded-block"),
]


def _controlled_fixture(data, address_bits, output_bits, variant, block_size):
    plain = QROM(data,
                 address_bits,
                 output_bits,
                 variant=variant,
                 block_size=block_size)
    ctrl = QROM(data,
                address_bits,
                output_bits,
                variant=variant,
                block_size=block_size,
                controlled=True)
    assert plain.controlled is False and ctrl.controlled is True
    assert ctrl.variant == plain.variant
    assert ctrl.num_ladder == plain.num_ladder
    return plain, ctrl


def _superposed_input_state(address_bits, num_ladder, output_bits, preload):
    """H on every address bit, output preloaded with ``preload``, ladder |0>."""

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(address_bits)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(output_bits)
        for b in range(address_bits):
            h(address_reg[b])
        for t in range(output_bits):
            if ((preload >> t) & 1) == 1:
                x(output[t])

    return np.array(cudaq.get_state(run))


def _expected_lookup_state(data, address_bits, num_ladder, output_bits,
                           preload):
    """The table itself: ``|k>|0>|preload XOR data[k]>`` over all ``k``."""
    state = np.zeros(1 << (address_bits + num_ladder + output_bits),
                     dtype=complex)
    amplitude = 1.0 / np.sqrt(1 << address_bits)
    for k in range(1 << address_bits):
        word = preload ^ (data[k] if k < len(data) else 0)
        state[k | (word << (address_bits + num_ladder))] = amplitude
    return state


def _plain_lookup_state(qrom: QROM, preload):
    lookup = qrom.kernel
    address_bits = qrom.num_address
    num_ladder = qrom.num_ladder
    output_bits = qrom.num_output

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(address_bits)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(output_bits)
        for b in range(address_bits):
            h(address_reg[b])
        for t in range(output_bits):
            if ((preload >> t) & 1) == 1:
                x(output[t])
        lookup(address_reg, ladder, output)

    return np.array(cudaq.get_state(run))


def _controlled_lookup_state(qrom: QROM, preload, control_mode: int):
    """control_mode: 0 -> |0>, 1 -> |1>, 2 -> |+>; the control is qubit 0."""
    lookup = qrom.kernel
    address_bits = qrom.num_address
    num_ladder = qrom.num_ladder
    output_bits = qrom.num_output

    @cudaq.kernel
    def run(mode: int):
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(address_bits)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(output_bits)
        if mode == 1:
            x(control[0])
        if mode == 2:
            h(control[0])
        for b in range(address_bits):
            h(address_reg[b])
        for t in range(output_bits):
            if ((preload >> t) & 1) == 1:
                x(output[t])
        lookup(control, address_reg, ladder, output)

    # Reshape so column c holds the amplitudes with the control in |c>.
    return np.array(cudaq.get_state(run, control_mode)).reshape(-1, 2)


@pytest.mark.parametrize(
    "data,address_bits,output_bits,variant,block_size,preload",
    _CTRL_GEOMETRIES)
def test_controlled_qrom_control_conventions(data, address_bits, output_bits,
                                             variant, block_size, preload):
    # Control |0>: the whole register is untouched (superposed addresses,
    # a non-zero output word). Control |1>: exactly the plain lookup,
    # which is the table itself. Control |+>: the two branches side by
    # side, each with weight 1/2.
    plain, ctrl = _controlled_fixture(data, address_bits, output_bits, variant,
                                      block_size)
    before = _superposed_input_state(address_bits, ctrl.num_ladder,
                                     output_bits, preload)
    after = _plain_lookup_state(plain, preload)
    np.testing.assert_allclose(after,
                               _expected_lookup_state(data, address_bits,
                                                      ctrl.num_ladder,
                                                      output_bits, preload),
                               atol=1e-12)

    at_zero = _controlled_lookup_state(ctrl, preload, 0)
    np.testing.assert_allclose(at_zero[:, 0], before, atol=1e-12)
    np.testing.assert_allclose(at_zero[:, 1], 0.0, atol=1e-12)

    at_one = _controlled_lookup_state(ctrl, preload, 1)
    np.testing.assert_allclose(at_one[:, 1], after, atol=1e-12)
    np.testing.assert_allclose(at_one[:, 0], 0.0, atol=1e-12)

    at_plus = _controlled_lookup_state(ctrl, preload, 2)
    np.testing.assert_allclose(at_plus[:, 0], before / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(at_plus[:, 1], after / np.sqrt(2), atol=1e-12)


@pytest.mark.parametrize(
    "data,address_bits,output_bits,variant,block_size,preload",
    _CTRL_GEOMETRIES)
def test_controlled_qrom_is_self_inverse(data, address_bits, output_bits,
                                         variant, block_size, preload):
    _, ctrl = _controlled_fixture(data, address_bits, output_bits, variant,
                                  block_size)
    lookup = ctrl.kernel
    num_ladder = ctrl.num_ladder

    @cudaq.kernel
    def run():
        control = cudaq.qvector(1)
        address_reg = cudaq.qvector(address_bits)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(output_bits)
        h(control[0])
        for b in range(address_bits):
            h(address_reg[b])
        for t in range(output_bits):
            if ((preload >> t) & 1) == 1:
                x(output[t])
        lookup(control, address_reg, ladder, output)
        lookup(control, address_reg, ladder, output)

    state = np.array(cudaq.get_state(run)).reshape(-1, 2)
    before = _superposed_input_state(address_bits, num_ladder, output_bits,
                                     preload)
    np.testing.assert_allclose(state[:, 0], before / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(state[:, 1], before / np.sqrt(2), atol=1e-12)


def test_controlled_qrom_toffoli_counts_follow_the_documented_contract():
    # Only the walk passes carry the control, so the price is the
    # controlled walk count in place of W; the routing and copy are
    # unchanged (the select_swap delta is twice the per-walk delta).
    plain_select, ctrl_select = _controlled_fixture(_CTRL_DATA, 3, 3, "select",
                                                    None)
    assert ctrl_select.toffoli_count == _walk_toffoli_count(3, 5, True)
    assert plain_select.toffoli_count == _walk_toffoli_count(3, 5, False)
    plain_swap, ctrl_swap = _controlled_fixture(_CTRL_DATA, 3, 3,
                                                "select_swap", 2)
    assert ctrl_swap.toffoli_count == _select_swap_cost(3, 5, 3, 2, True)
    assert plain_swap.toffoli_count == _select_swap_cost(3, 5, 3, 2, False)
    per_walk = ctrl_select.toffoli_count - plain_select.toffoli_count
    assert per_walk > 0
    assert ctrl_swap.toffoli_count - plain_swap.toffoli_count == 2 * (
        _walk_toffoli_count(2, 3, True) - _walk_toffoli_count(2, 3, False))
    assert "controlled=True" in repr(ctrl_swap)
    assert "controlled=False" in repr(plain_swap)


def test_controlled_qrom_auto_prices_controlled_candidates():
    # variant="auto" on a controlled lookup compares the controlled costs
    # against each other, never a controlled count against a plain one.
    # Seven one-bit entries over three address bits make the choice
    # visible: plain pricing picks select_swap with B = 2, controlled
    # pricing picks B = 4 (controlled walks over few high bits are
    # relatively dearer, so the control pushes the optimum toward larger
    # blocks); either mixed pricing would land on B = 2 or on select.
    table = [1, 0, 1, 1, 0, 1, 1]
    plain = QROM(table, 3, 1)
    ctrl = QROM(table, 3, 1, controlled=True)
    assert (plain.variant, plain.block_size) == ("select_swap", 2)
    assert (ctrl.variant, ctrl.block_size) == ("select_swap", 4)
    assert plain.toffoli_count == _select_swap_cost(3, 7, 1, 2, False)
    assert ctrl.toffoli_count == _select_swap_cost(3, 7, 1, 4, True)
    assert ctrl.toffoli_count == min(
        _walk_toffoli_count(3, 7, True),
        min(_select_swap_cost(3, 7, 1, size, True) for size in (2, 4)))
