# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the QROM lookup (select, select_swap, select_copy and the
auto choice among the three constructions; select_copy's packet,
borrowed-bank, budget and chain oracles live in
test_primitives_qrom_extensions.py and test_primitives_qrom_chain.py).

Correctness is pinned exhaustively: for random tables every address is
read out via basis-state extraction (`cudaq.get_state`), including the
out-of-range addresses of non-power-of-two tables (which must read
zero), for all three constructions. The select and select_swap lookups
are self-inverse on clean ancillas — that property is tested on
superposed addresses instead of a duplicate adjoint kernel — and the
SELECT-SWAP and SELECT-COPY lookups are checked to be behaviorally
identical to the plain unary-iteration lookup on the same table (equal
(address, output) amplitudes with the clean ancillas traced as |0>).
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
        np.testing.assert_allclose(state, expected, atol=1e-12, rtol=0)


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
                          (4, 11, 1, 2), (4, 9, 1, 8)])
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


@pytest.mark.parametrize("address_bits,num_entries,output_bits,block_size",
                         [(2, 4, 3, 2), (3, 5, 2, 2), (3, 8, 1, 4),
                          (4, 9, 1, 8), (4, 13, 2, 4)])
def test_qrom_select_copy_exhaustive_readout(address_bits, num_entries,
                                             output_bits, block_size):
    rng = np.random.default_rng(5 * address_bits + num_entries + block_size)
    data = [int(v) for v in rng.integers(0, 1 << output_bits, num_entries)]
    qrom = QROM(data,
                address_bits,
                output_bits,
                variant="select_copy",
                block_size=block_size)
    assert qrom.variant == "select_copy"
    low_bits = block_size.bit_length() - 1
    assert qrom.num_ladder == (max(address_bits - low_bits, low_bits) +
                               (block_size - 1) * output_bits)
    _exhaustive_readout(qrom, data)


@pytest.mark.parametrize("variant,block_size", [("select", None),
                                                ("select_swap", 2)])
def test_qrom_is_self_inverse(variant, block_size):
    # From a clean ladder, applying the lookup twice XORs the table twice.
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
    np.testing.assert_allclose(state, expected, atol=1e-12, rtol=0)


def test_qrom_select_swap_equals_select_on_superposed_addresses():
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
                                   atol=1e-12,
                                   rtol=0)
        return tensor[:, 0, :]

    select = QROM(data, address_bits, output_bits, variant="select")
    swap = QROM(data,
                address_bits,
                output_bits,
                variant="select_swap",
                block_size=2)
    np.testing.assert_allclose(joint_amplitudes(select),
                               joint_amplitudes(swap),
                               atol=1e-12,
                               rtol=0)


def test_qrom_variants_agree_on_superposed_addresses():
    # Same table through all three constructions: the (address, output)
    # amplitudes must agree exactly, the ancillas being |0> in each.
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
                                   atol=1e-12,
                                   rtol=0)
        return tensor[:, 0, :]

    select = QROM(data, address_bits, output_bits, variant="select")
    reference = joint_amplitudes(select)
    for variant in ("select_swap", "select_copy"):
        for block_size in (2, 4):
            other = QROM(data,
                         address_bits,
                         output_bits,
                         variant=variant,
                         block_size=block_size)
            np.testing.assert_allclose(reference,
                                       joint_amplitudes(other),
                                       atol=1e-12,
                                       rtol=0)


def test_qrom_auto_dispatch_picks_the_cheapest_variant():
    # Big table, narrow output: routing is cheap, the block walk is a
    # fraction of the full walk -> a blocked variant wins; at b = 1 and
    # N = 64 select_copy and select_swap tie (both 28 at B = 8) and the
    # tie-break prefers select_copy (no routing-CNOT storm).
    wide = QROM(list(range(2)) * 32, 6, 1)
    assert wide.variant == "select_copy"
    assert wide.block_size is not None
    # Tiny table, wide output: every routed bit costs Toffolis -> the
    # plain walk wins.
    narrow = QROM([200, 3, 118, 25], 2, 8)
    assert narrow.variant == "select"
    assert narrow.block_size is None
    # Multi-bit output: the copy's halved routing beats the swap network
    # at every block size, so auto picks select_copy outright.
    multi = QROM([int(v) % 4 for v in range(64)], 6, 2)
    assert multi.variant == "select_copy"
    # Single-bit output with a large optimal block: the coherent copy
    # walk (3B/2 - 5) outgrows b (B - 1) at b = 1, B > 8 and select_swap
    # is strictly cheapest again (see the one-output-bit regime in the
    # module docstring).
    single = QROM([int(v) % 2 for v in range(128)], 7, 1)
    assert single.variant == "select_swap"
    # Whatever auto picks must be priced no worse than the alternatives.
    for variant in ("select_swap", "select_copy"):
        for block_size in (2, 4, 8, 16, 32):
            forced = QROM(list(range(2)) * 32,
                          6,
                          1,
                          variant=variant,
                          block_size=block_size)
            assert wide.toffoli_count <= forced.toffoli_count
    assert wide.toffoli_count <= QROM(list(range(2)) * 32,
                                      6,
                                      1,
                                      variant="select").toffoli_count


@pytest.mark.parametrize("variant", ["select_swap", "select_copy"])
def test_qrom_default_block_size_is_cost_optimal(variant):
    data = [int(v) for v in np.arange(32) % 4]
    auto_block = QROM(data, 5, 2, variant=variant)
    assert auto_block.block_size is not None
    for block_size in (2, 4, 8, 16):
        forced = QROM(data, 5, 2, variant=variant, block_size=block_size)
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
