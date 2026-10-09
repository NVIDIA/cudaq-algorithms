# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent coherent QROM oracles, including the public consumer seam."""

import numpy as np
import pytest
import cudaq

from cudaq_algorithms import state_from
from cudaq_algorithms import primitives
from cudaq_algorithms.primitives import QROM


def _chain(*args, **kwargs):
    assert hasattr(primitives, "QROMChain"), "QROMChain must be exported"
    return primitives.QROMChain(*args, **kwargs)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("bad,reason", [([], "non-empty"), ([1.5], "integers"),
                                        ([-1], "non-negative"),
                                        ([4], "does not fit")])
def test_chain_validates_original_interior_table(fused, bad, reason):
    with pytest.raises(ValueError, match="table 1.*" + reason):
        _chain([[0], bad, [1]], 2, 2, fused=fused)


@pytest.mark.parametrize("fused", [True, False])
def test_chain_normalizes_integral_inputs_before_xor(fused):
    chain = _chain([[0.0], [1.0]], 2.0, 1.0, fused=fused)
    assert type(chain.num_address) is int
    assert type(chain.num_output) is int
    assert len(chain.kernels()) == 3


@pytest.mark.parametrize("variant", ["select_swap", "select_copy"])
def test_blocked_qrom_normalizes_block_size(variant):
    qrom = QROM([0, 1], 2, 1, variant=variant, block_size=2.0)
    assert qrom.block_size == 2
    assert type(qrom.block_size) is int


@pytest.mark.parametrize("variant", ["select_swap", "select_copy"])
def test_blocked_qrom_requires_block_index_bit(variant):
    with pytest.raises(ValueError, match="requires address_bits >= 2"):
        QROM([0, 1], 1, 1, variant=variant, max_ancillas=4)


def test_select_copy_preserves_generator_validation():
    with pytest.raises(ValueError, match="integers"):
        QROM(iter([0, 1.5]), 2, 2, variant="select_copy")
    qrom = QROM(iter([0, 1]), 2, 2, variant="select_copy")
    assert qrom.num_output == 2


@pytest.mark.parametrize("block_size,alpha", [(2, 1), (2, 2), (4, 2), (4, 3)])
def test_controlled_packets_match_complex_permutation(block_size, alpha):
    # All addresses (including padding), all output words, and both
    # external-control sectors carry independent complex amplitudes.
    data = [7, 0, 5, 3, 1]
    qrom = QROM(data,
                3,
                3,
                variant="select_copy",
                block_size=block_size,
                alpha=alpha)
    lookup = qrom.kernel
    ladder = qrom.num_ladder
    shift = 3 + ladder
    control = shift + 3
    rng = np.random.default_rng(443)
    compact = rng.normal(size=128) + 1j * rng.normal(size=128)
    compact /= np.linalg.norm(compact)
    initial = np.zeros(1 << (control + 1), dtype=np.complex128)
    expected = np.zeros_like(initial)
    for j, amplitude in enumerate(compact):
        address, output, enabled = j & 7, (j >> 3) & 7, j >> 6
        source = address | (output << shift) | (enabled << control)
        word = data[address] if enabled and address < len(data) else 0
        initial[source] = amplitude
        expected[source ^ (word << shift)] = amplitude

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        cudaq.control(lookup, q[control], q[0:3], q[3:shift], q[shift:control])

    @cudaq.kernel
    def twice(state: cudaq.State):
        q = cudaq.qvector(state)
        lookup(q[0:3], q[3:shift], q[shift:control])
        lookup(q[0:3], q[3:shift], q[shift:control])

    actual = np.asarray(cudaq.get_state(run, state_from(initial)))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(twice, state_from(initial))),
                               initial,
                               atol=1e-12)

    @cudaq.kernel
    def resources():
        q = cudaq.qvector(control)
        lookup(q[0:3], q[3:shift], q[shift:control])

    compiled = cudaq.estimate_resources(resources).count_controls("x", 2)
    assert compiled == qrom.toffoli_count
    assert qrom.describe().count("# Toffoli") == compiled
    if (block_size, alpha) == (4, 2):
        assert compiled == 13


def _xor_table(state, table, shift):
    indices = np.arange(len(state))
    values = np.array(
        [table[j & 7] if (j & 7) < len(table) else 0 for j in indices])
    result = np.empty_like(state)
    result[indices ^ (values << shift)] = state
    return result


def _single_qubit(state, bit, matrix):
    result = state.copy()
    indices = np.arange(len(state))
    low = indices[(indices & (1 << bit)) == 0]
    high = low | (1 << bit)
    result[low], result[high] = matrix @ np.stack([state[low], state[high]])
    return result


@pytest.mark.parametrize("fused", [True, False])
def test_chain_with_output_mutations_matches_independent_circuit(fused):
    # Unequal lengths make auto choose different constructions. H/Ry on
    # the encoded word deliberately invalidate the table-value invariant,
    # but must preserve equivalence to separate load/operate/unload calls.
    tables = [[1], [1, 3, 2, 0, 1, 2, 3, 0], [2, 1]]
    chain = _chain(tables, 3, 2, fused=fused)
    k0, k1, k2, k3 = chain.kernels()
    shift = 3 + chain.num_ladder
    reference = shift + 2
    rng = np.random.default_rng(193)
    compact = rng.normal(size=64) + 1j * rng.normal(size=64)
    compact /= np.linalg.norm(compact)
    initial = np.zeros(1 << (reference + 1), dtype=np.complex128)
    for j, amplitude in enumerate(compact):
        initial[(j & 7) | ((j >> 3) << shift)] = amplitude
    expected = initial.copy()
    for j, table in enumerate(tables):
        expected = _xor_table(expected, table, shift)
        if j == 0:
            expected = _single_qubit(expected, shift,
                                     np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        elif j == 1:
            c, s = np.cos(0.731 / 2), np.sin(0.731 / 2)
            expected = _single_qubit(expected, shift + 1,
                                     np.array([[c, -s], [s, c]]))
        else:
            indices = np.arange(len(expected))
            permuted = indices ^ (((indices >> (shift + 1)) & 1) << reference)
            expected = expected[permuted]
            expected = _single_qubit(
                expected, shift, np.diag([np.exp(-0.215j),
                                          np.exp(0.215j)]))
        expected = _xor_table(expected, table, shift)

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        k0(q[0:3], q[3:shift], q[shift:reference])
        h(q[shift])
        k1(q[0:3], q[3:shift], q[shift:reference])
        ry(0.731, q[shift + 1])
        k2(q[0:3], q[3:shift], q[shift:reference])
        cx(q[shift + 1], q[reference])
        rz(0.43, q[shift])
        k3(q[0:3], q[3:shift], q[shift:reference])

    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(run, state_from(initial))),
                               expected,
                               atol=1e-12)
    assert chain.toffoli_count == (20 if fused else 28)


def test_ancilla_budget_selects_packets_and_rejects_impossible_budget():
    data = list(range(16)) * 64
    tight = QROM(data, 10, 4, max_ancillas=24)
    free = QROM(data, 10, 4)
    assert (tight.variant, tight.block_size, tight.alpha,
            tight.toffoli_count) == ("select_copy", 16, 4, 591)
    assert (free.variant, free.block_size, free.alpha,
            free.toffoli_count) == ("select_copy", 32, 1, 253)
    with pytest.raises(ValueError, match="fits max_ancillas"):
        QROM(data, 10, 4, max_ancillas=1)


@pytest.mark.parametrize("options,reason", [
    ({
        "alpha": 2
    }, "only valid"),
    ({
        "variant": "select_copy",
        "alpha": 0
    }, "alpha must"),
    ({
        "variant": "select_copy",
        "alpha": 4
    }, "alpha must"),
    ({
        "variant": "select_copy",
        "block_size": 3
    }, "power of two"),
    ({
        "variant": "select_copy",
        "block_size": 4
    }, "block-index"),
    ({
        "max_ancillas": 0
    }, "positive integer"),
    ({
        "variant": "select_copy",
        "max_ancillas": 1
    }, "fits max_ancillas"),
])
def test_strategy_validation(options, reason):
    with pytest.raises(ValueError, match=reason):
        QROM([0, 1, 7], 2, 3, **options)


def test_chain_validates_container_and_reports_roles():
    for tables, reason in [(7, "sequence of lookup tables"),
                           ([], "at least one table"),
                           ([[1], 5], r"tables\[1\].*sequence"),
                           ([[1] * 5], "addresses only 4")]:
        with pytest.raises(ValueError, match=reason):
            _chain(tables, 2, 2)
    for fused in (False, True):
        chain = _chain([[1, 0], [0, 1]], 2, 2, fused=fused)
        assert sum(chain.step_toffoli_counts) == chain.toffoli_count
        assert len(chain.step_toffoli_counts) == 3
        assert len(chain.steps) == (3 if fused else 2)
        assert "fused=" + str(fused) in repr(chain)
        assert "unload tables[0]" in chain.describe()


def test_auto_search_matches_all_candidates_across_budgets():
    # Enumerate forced constructions rather than using the optimizer's
    # helper functions. No JIT compilation is needed for this host policy.
    data = list(range(16)) * 16
    candidates = [QROM(data, 8, 4, variant="select")]
    for block in (2, 4, 8, 16, 32, 64, 128):
        candidates.append(
            QROM(data, 8, 4, variant="select_swap", block_size=block))
        for alpha in (1, 2, 3, 4):
            candidates.append(
                QROM(data,
                     8,
                     4,
                     variant="select_copy",
                     block_size=block,
                     alpha=alpha))
    for budget in (8, 10, 12, 16, 24, 32, 64, 128):
        auto = QROM(data, 8, 4, max_ancillas=budget)
        eligible = [
            q.toffoli_count for q in candidates if q.num_ladder <= budget
        ]
        assert auto.num_ladder <= budget
        assert auto.toffoli_count == min(eligible)


def test_alias_sampling_accepts_select_copy_property():
    from test_primitives_alias_sampling import (_brute_force_marginal,
                                                _circuit_marginal,
                                                _prepare_resources)

    prep = primitives.AliasSamplingPrepare([1.0, 2.0, 3.0, 1.0],
                                           mu=1,
                                           variant="select_copy",
                                           block_size=2,
                                           alpha=3,
                                           max_ancillas=2)
    np.testing.assert_allclose(_circuit_marginal(prep),
                               _brute_force_marginal(prep),
                               atol=1e-12)
    prepare, unprepare = prep.kernel(), prep.adjoint_kernel()
    width = prep.num_garbage

    @cudaq.kernel
    def inverse():
        index = cudaq.qvector(2)
        garbage = cudaq.qvector(width)
        prepare(index, garbage)
        unprepare(index, garbage)

    state = np.asarray(cudaq.get_state(inverse))
    expected = np.zeros_like(state)
    expected[0] = 1
    np.testing.assert_allclose(state, expected, atol=1e-12)
    for adjoint in (False, True):
        resources = _prepare_resources(prep, adjoint)
        assert resources.count_controls("x", 2) == prep.qrom.toffoli_count + 1


@pytest.mark.parametrize("variant,block_size,alpha", [("select_swap", 2, None),
                                                      ("select_copy", 4, 2)])
def test_controlled_lookup_restores_entangled_borrowed_bank(
        variant, block_size, alpha):
    data = [7, 0, 5, 3, 1]
    qrom = QROM(data,
                3,
                3,
                variant=variant,
                block_size=block_size,
                alpha=alpha,
                dirty_banks=True)
    clean = QROM(data,
                 3,
                 3,
                 variant=variant,
                 block_size=block_size,
                 alpha=alpha)
    assert qrom.dirty_banks
    assert qrom.num_clean + qrom.num_dirty == qrom.num_ladder
    assert qrom.num_dirty > 0
    assert clean.num_clean == clean.num_ladder and clean.num_dirty == 0
    assert not clean.dirty_banks
    lookup = qrom.kernel
    bank_start = 3 + qrom.num_clean
    shift = 3 + qrom.num_ladder
    control, reference = shift + 3, shift + 4
    rng = np.random.default_rng(718)
    # Bank, output, address, control and spectator are generally entangled.
    size = 1 << (3 + qrom.num_dirty + 3 + 2)
    compact = rng.normal(size=size) + 1j * rng.normal(size=size)
    compact /= np.linalg.norm(compact)
    initial = np.zeros(1 << (reference + 1), dtype=np.complex128)
    expected = np.zeros_like(initial)
    for j, amplitude in enumerate(compact):
        source = (j & 7) | ((j >> 3) << bank_start)
        address, enabled = source & 7, (source >> control) & 1
        word = data[address] if enabled and address < len(data) else 0
        initial[source] = amplitude
        expected[source ^ (word << shift)] = amplitude

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        cudaq.control(lookup, q[control], q[0:3], q[3:shift], q[shift:control])

    @cudaq.kernel
    def twice(state: cudaq.State):
        q = cudaq.qvector(state)
        lookup(q[0:3], q[3:shift], q[shift:control])
        lookup(q[0:3], q[3:shift], q[shift:control])

    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(run, state_from(initial))),
                               expected,
                               atol=1e-12)
    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(twice, state_from(initial))),
                               initial,
                               atol=1e-12)

    @cudaq.kernel
    def resources():
        q = cudaq.qvector(control)
        lookup(q[0:3], q[3:shift], q[shift:control])

    compiled = cudaq.estimate_resources(resources).count_controls("x", 2)
    assert qrom.toffoli_count == compiled
    assert qrom.describe().count("# Toffoli") == compiled
    if variant == "select_copy":
        assert compiled == 24
    else:
        assert compiled == clean.toffoli_count + 2 * 3 * (block_size - 1)


@pytest.mark.parametrize("variant", ["auto", "select"])
def test_dirty_banks_require_explicit_blocked_variant(variant):
    with pytest.raises(ValueError, match="dirty_banks.*requires"):
        QROM([0, 1], 2, 1, variant=variant, dirty_banks=True)


@pytest.mark.parametrize("variant", ["select_swap", "select_copy"])
def test_borrowed_strategy_search_prices_echo_and_total_budget(variant):
    data = list(range(8)) * 4
    candidates = []
    for block in (2, 4, 8, 16):
        for alpha in ((1, 2, 3) if variant == "select_copy" else (None, )):
            candidates.append(
                QROM(data,
                     5,
                     3,
                     variant=variant,
                     block_size=block,
                     alpha=alpha,
                     dirty_banks=True))
    for budget in (8, 16, 32):
        eligible = [
            q.toffoli_count for q in candidates if q.num_ladder <= budget
        ]
        if not eligible:
            with pytest.raises(ValueError, match="fits max_ancillas"):
                QROM(data,
                     5,
                     3,
                     variant=variant,
                     dirty_banks=True,
                     max_ancillas=budget)
            continue
        chosen = QROM(data,
                      5,
                      3,
                      variant=variant,
                      dirty_banks=True,
                      max_ancillas=budget)
        assert chosen.toffoli_count == min(eligible)
        assert chosen.num_clean + chosen.num_dirty <= budget
