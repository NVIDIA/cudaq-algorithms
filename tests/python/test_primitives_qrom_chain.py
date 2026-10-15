# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QROMChain invariants and the SELECT-COPY packet pins Scott Thornton
wrote for the experimental branch, ported to the property API: fused vs
separate equivalence under interleaved operations, per-prefix table
readout, step self-inverse, the (m+1)C vs 2mC prices, option forwarding
and validation."""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import QROM, QROMChain
from test_primitives_resources import _compiled_toffolis_of

# Register layout in the harnesses: address at [0, A), ladder at
# [A, A + L), output at [A + L, A + L + W).


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


# ---------------------------------------------------------------------------
# Behavioral: the chain (Sec. II.C)
# ---------------------------------------------------------------------------


def _chain_tables():
    rng = np.random.default_rng(42)
    lengths = (8, 6, 7)  # non-power-of-two lengths included
    return [[int(v) for v in rng.integers(0, 8, n)] for n in lengths]


def _interleaved_state(kernel_list, num_ladder, thetas):
    """Full state of: prep superposed address, then kernel / caller-op
    alternation with per-step rotations + entangling CNOT on the output.

    ``kernel_list`` has 4 kernels (3 tables); the caller ops between
    them are phase-carrying, so any ordering or fusion error shows up in
    the amplitudes exactly.
    """
    k0, k1, k2, k3 = kernel_list
    t0, t1, t2 = thetas

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            rz(0.3 + 0.2 * b, address_reg[b])
        k0(address_reg, ladder, output)
        ry(t0, output[0])
        cx(output[0], output[1])
        rz(t0, output[2])
        k1(address_reg, ladder, output)
        ry(t1, output[1])
        cx(output[1], output[2])
        rz(t1, output[0])
        k2(address_reg, ladder, output)
        ry(t2, output[2])
        cx(output[2], output[0])
        rz(t2, output[1])
        k3(address_reg, ladder, output)

    return np.array(cudaq.get_state(run))


# "auto" is deliberately absent: its dispatch decision is pinned by the
# factory tests, and each parametrization here retains ~1 GB of compiled
# kernels for the life of the process (CI runners have 7 GB).
@pytest.mark.parametrize("variant", ["select", "select_copy"])
@pytest.mark.parametrize("fused", [True, False])
def test_chain_matches_independent_qroms_with_interleaved_ops(fused, variant):
    # The semantics contract: chain fused == chain naive == m independent
    # QROMs (load, caller op, unload each), with arbitrary caller
    # operations on the output between the steps.
    tables = _chain_tables()
    thetas = (0.7, 1.3, 2.1)
    chain = QROMChain(tables, 3, 3, fused=fused, variant=variant)
    kernels = chain.kernels()
    assert len(kernels) == chain.num_tables + 1 == 4
    state = _interleaved_state(kernels, chain.num_ladder, thetas)

    # Reference: three independent lookups, each applied twice around
    # its caller op (load + self-inverse unload), on the same ladder
    # width (unused ladder lines are never touched).
    reference_qroms = [QROM(table, 3, 3, variant=variant) for table in tables]
    q0, q1, q2 = [q.kernel for q in reference_qroms]
    t0, t1, t2 = thetas
    num_ladder = chain.num_ladder

    @cudaq.kernel
    def reference():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            h(address_reg[b])
            rz(0.3 + 0.2 * b, address_reg[b])
        q0(address_reg, ladder, output)
        ry(t0, output[0])
        cx(output[0], output[1])
        rz(t0, output[2])
        q0(address_reg, ladder, output)
        q1(address_reg, ladder, output)
        ry(t1, output[1])
        cx(output[1], output[2])
        rz(t1, output[0])
        q1(address_reg, ladder, output)
        q2(address_reg, ladder, output)
        ry(t2, output[2])
        cx(output[2], output[0])
        rz(t2, output[1])
        q2(address_reg, ladder, output)

    expected = np.array(cudaq.get_state(reference))
    np.testing.assert_allclose(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("variant,block_size", [("select", None),
                                                ("select_copy", 2)])
def test_chain_prefix_readout_holds_each_table(fused, variant, block_size):
    # After step kernel j the output must hold tables[j][k] (zero for
    # out-of-range addresses of a short table); after the final kernel
    # it must be restored to zero. Exhaustive over addresses.
    tables = _chain_tables()
    chain = QROMChain(tables,
                      3,
                      3,
                      fused=fused,
                      variant=variant,
                      block_size=block_size)
    kernels = chain.kernels()
    num_ladder = chain.num_ladder
    shift = 3 + num_ladder
    total = shift + 3
    # One harness per prefix length applying that many step kernels
    # (explicit unrolled calls — kernels cannot be looped over inside a
    # CUDA-Q kernel). The address is a RUNTIME ARGUMENT: a closure-baked
    # address would mint (and retain, compiled) a fresh kernel per
    # address — 8x the JIT memory for identical circuits.
    k0, k1, k2, k3 = kernels

    @cudaq.kernel
    def run1(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)

    @cudaq.kernel
    def run2(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)
        k1(address_reg, ladder, output)

    @cudaq.kernel
    def run3(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)
        k1(address_reg, ladder, output)
        k2(address_reg, ladder, output)

    @cudaq.kernel
    def run4(address: int):
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        k0(address_reg, ladder, output)
        k1(address_reg, ladder, output)
        k2(address_reg, ladder, output)
        k3(address_reg, ladder, output)

    prefix_kernels = {1: run1, 2: run2, 3: run3, 4: run4}

    def prefix_state(address, prefix):
        return np.array(cudaq.get_state(prefix_kernels[prefix], address))

    for prefix in (1, 2, 3, 4):
        for address in range(8):
            if prefix <= 3:
                table = tables[prefix - 1]
                word = table[address] if address < len(table) else 0
            else:
                word = 0
            expected = _basis(address + (word << shift), total)
            np.testing.assert_allclose(prefix_state(address, prefix),
                                       expected,
                                       atol=1e-12,
                                       rtol=0)


def test_chain_step_kernels_self_inverse_on_clean_sector():
    # Each step kernel XORs a fixed table into the output and restores
    # the ladder, so applying it twice from a clean ladder is the
    # identity — pinned on superposed addresses with a non-zero initial
    # output word.
    tables = _chain_tables()
    chain = QROMChain(tables, 3, 3, fused=True)
    num_ladder = chain.num_ladder
    initial = 0b011
    total = 3 + num_ladder + 3
    expected = np.zeros(1 << total, dtype=np.complex128)
    for address in range(8):
        expected[address + (initial << (3 + num_ladder))] = 1.0 / np.sqrt(8.0)
    for step in chain.kernels():

        @cudaq.kernel
        def run():
            address_reg = cudaq.qvector(3)
            ladder = cudaq.qvector(num_ladder)
            output = cudaq.qvector(3)
            for b in range(3):
                h(address_reg[b])
                if ((initial >> b) & 1) == 1:
                    x(output[b])
            step(address_reg, ladder, output)
            step(address_reg, ladder, output)

        np.testing.assert_allclose(np.array(cudaq.get_state(run)),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


# ---------------------------------------------------------------------------
# Resource contracts
# ---------------------------------------------------------------------------

_HAS_RESOURCES = hasattr(cudaq, "estimate_resources")
needs_resources = pytest.mark.skipif(
    not _HAS_RESOURCES,
    reason="cudaq.estimate_resources is not available in this CUDA-Q")


@needs_resources
def test_chain_fused_costs_m_plus_1_lookups_and_naive_2m():
    # Equal-length tables: the per-lookup price C is shape-only, so
    # fused == (m + 1) C, naive == 2 m C, margin exactly (m - 1) C —
    # and the compiler agrees with the bookkeeping for every step
    # kernel of both modes.
    rng = np.random.default_rng(9)
    m = 4
    tables = [[int(v) for v in rng.integers(0, 8, 16)] for _ in range(m)]
    per_lookup = QROM(tables[0], 4, 3, variant="select").toffoli_count
    fused = QROMChain(tables, 4, 3, fused=True, variant="select")
    naive = QROMChain(tables, 4, 3, fused=False, variant="select")
    assert fused.toffoli_count == (m + 1) * per_lookup
    assert naive.toffoli_count == 2 * m * per_lookup
    assert naive.toffoli_count - fused.toffoli_count == (m - 1) * per_lookup
    assert fused.toffoli_count < naive.toffoli_count
    for chain in (fused, naive):
        assert len(chain.step_toffoli_counts) == m + 1
        assert sum(chain.step_toffoli_counts) == chain.toffoli_count
        for kernel, claimed in zip(chain.kernels(), chain.step_toffoli_counts):
            compiled = _compiled_toffolis_of(kernel, chain.num_address,
                                             chain.num_ladder,
                                             chain.num_output)
            assert compiled == claimed


def test_chain_fused_strictly_cheaper_for_blocked_variants_too():
    # The (m + 1) vs 2 m load count is variant-independent: it also
    # holds when the steps are select_copy lookups (same forced shape).
    rng = np.random.default_rng(17)
    m = 3
    tables = [[int(v) for v in rng.integers(0, 8, 32)] for _ in range(m)]
    per_lookup = QROM(tables[0], 5, 3, variant="select_copy",
                      block_size=4).toffoli_count
    fused = QROMChain(tables,
                      5,
                      3,
                      fused=True,
                      variant="select_copy",
                      block_size=4)
    naive = QROMChain(tables,
                      5,
                      3,
                      fused=False,
                      variant="select_copy",
                      block_size=4)
    assert fused.toffoli_count == (m + 1) * per_lookup
    assert naive.toffoli_count == 2 * m * per_lookup


def test_chain_and_packet_validation_raises():
    with pytest.raises(ValueError, match="alpha .* only valid with"):
        QROM([1, 2], 1, 2, alpha=2)
    with pytest.raises(ValueError, match="alpha .* only valid with"):
        QROM([1, 2], 2, 2, variant="select_swap", block_size=2, alpha=2)
    # Blocked variants need a block-index address bit; that check comes
    # before the alpha range check.
    with pytest.raises(ValueError, match="requires address_bits >= 2"):
        QROM([1, 2], 1, 2, variant="select_copy", alpha=3)
    with pytest.raises(ValueError, match="requires address_bits >= 2"):
        QROM([1, 2], 1, 2, variant="select_copy", alpha=0)
    with pytest.raises(ValueError, match="max_ancillas must be a positive"):
        QROM([1, 2], 1, 2, max_ancillas=0)
    with pytest.raises(ValueError, match="fits max_ancillas"):
        QROM([1] * 16, 4, 1, max_ancillas=1)
    with pytest.raises(ValueError, match="fits max_ancillas"):
        QROM([1] * 16, 4, 4, variant="select_swap", max_ancillas=5)
    with pytest.raises(ValueError, match="sequence of lookup tables"):
        QROMChain(7, 2, 2)
    with pytest.raises(ValueError, match="at least one table"):
        QROMChain([], 2, 2)
    with pytest.raises(ValueError, match=r"tables\[1\] must be a sequence"):
        QROMChain([[1, 2], 5], 2, 2)
    # Range errors are located: the offending table is named.
    with pytest.raises(ValueError, match="table 1.*does not fit"):
        QROMChain([[1, 0], [1, 4]], 1, 2, fused=False)
    with pytest.raises(ValueError, match="table 1.*non-negative"):
        QROMChain([[1, 0], [1, -2]], 1, 2, fused=True)
    with pytest.raises(ValueError, match="addresses only 4"):
        QROMChain([[1] * 5], 2, 2)


def test_chain_repr_and_describe_reflect_the_choice():
    tables = _chain_tables()
    chain = QROMChain(tables, 3, 3, fused=True)
    assert "fused=True" in repr(chain)
    text = chain.describe()
    assert "fused unload tables[0] + load tables[1]" in text
    assert text.count("QROM(") == chain.num_tables + 1
    naive = QROMChain(tables, 3, 3, fused=False)
    assert "two lookups replayed back to back" in naive.describe()
    packets = QROM([int(v) for v in np.arange(16) % 8],
                   4,
                   3,
                   variant="select_copy",
                   block_size=4,
                   alpha=3)
    assert "alpha=3" in repr(packets)
    assert "# Toffoli" in packets.describe()


# ---------------------------------------------------------------------------
# Compiled step counts, option forwarding, container validation,
# single-table reuse and the naive pair kernel
# ---------------------------------------------------------------------------


def test_chain_step_kernels_compile_to_their_step_counts():
    tables = [[0, 1, 3, 2, 1, 0, 2, 3], [3, 1, 0, 2, 2, 3, 1, 0],
              [1, 2, 1, 3, 0, 1, 3, 2]]
    for fused in (True, False):
        chain = QROMChain(tables, 3, 2, fused=fused)
        for kernel, claimed in zip(chain.kernels(), chain.step_toffoli_counts):
            assert _compiled_toffolis_of(kernel, 3, chain.num_ladder,
                                         2) == claimed


def test_chain_forwards_strategy_options():
    chain = QROMChain([[1, 0, 3, 2], [2, 2, 0, 1]],
                      3,
                      2,
                      variant="select_swap",
                      block_size=4)
    assert all(step.variant == "select_swap" and step.block_size == 4
               for step in chain.steps)
    assert chain.num_ladder == 1 + 4 * 2
    for fused in (True, False):
        with pytest.raises(ValueError,
                           match=r"^QROMChain: no construction fits "
                           r"max_ancillas=1"):
            QROMChain([[1, 0, 3, 2], [2, 2, 0, 1]],
                      3,
                      2,
                      fused=fused,
                      max_ancillas=1)
        with pytest.raises(ValueError, match=r"^QROMChain: alpha must"):
            QROMChain([[1, 0]],
                      2,
                      1,
                      fused=fused,
                      variant="select_copy",
                      alpha=5)


def test_chain_accepts_array_tables_and_rejects_strings_and_bools():
    chain = QROMChain([np.array([1, 0, 3, 2]), np.array([2, 2, 0, 1])], 3, 2)
    assert chain.num_tables == 2
    assert QROMChain(np.array([[1, 0], [0, 1]]), 2, 1).num_tables == 2
    with pytest.raises(ValueError, match="sequence of lookup tables"):
        QROMChain("01", 2, 1)
    with pytest.raises(ValueError, match=r"tables\[0\] must be a sequence"):
        QROMChain([b"\x01\x00"], 2, 1)
    with pytest.raises(ValueError, match="fused must be a bool"):
        QROMChain([[1, 0]], 2, 1, fused="False")
    with pytest.raises(ValueError, match=r"^address_bits must be"):
        QROMChain([[1]], 0, 1)
    with pytest.raises(ValueError, match=r"^output_bits must be"):
        QROMChain([[1]], 1, 0)
    with pytest.raises(ValueError, match=r"^QROMChain: "):
        QROMChain([[1, 0]], 2, 1, variant="select_copy", block_size=[2])


def test_single_table_chain_reuses_its_lookup():
    chain = QROMChain([[1, 0, 3, 2]], 2, 2)
    assert len(chain.kernels()) == 2
    assert chain.kernels()[0] is chain.kernels()[1]
    assert chain.steps[0] is chain.steps[1]
    assert chain.toffoli_count == 2 * chain.steps[0].toffoli_count


def test_naive_interior_kernel_equals_two_lookups():
    tables = [[0, 1, 3, 2, 1, 0, 2, 3], [3, 1, 0, 2, 2, 3, 1, 0]]
    chain = QROMChain(tables, 3, 2, fused=False)
    pair = chain.kernels()[1]
    first = chain.steps[0].kernel
    second = chain.steps[1].kernel
    width = chain.num_ladder

    @cudaq.kernel
    def via_pair():
        address = cudaq.qvector(3)
        ladder = cudaq.qvector(width)
        output = cudaq.qvector(2)
        h(address)
        ry(0.4, output[1])
        pair(address, ladder, output)

    @cudaq.kernel
    def via_steps():
        address = cudaq.qvector(3)
        ladder = cudaq.qvector(width)
        output = cudaq.qvector(2)
        h(address)
        ry(0.4, output[1])
        first(address, ladder, output)
        second(address, ladder, output)

    np.testing.assert_allclose(np.asarray(cudaq.get_state(via_pair)),
                               np.asarray(cudaq.get_state(via_steps)),
                               atol=1e-12,
                               rtol=0)
