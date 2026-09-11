# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the coherent alias-sampling PREPARE.

Two-level validation: the circuit's index-register marginal (amplitude
magnitudes squared, summed over every garbage configuration) is pinned at
simulator precision against an *independent* brute-force enumeration of
the classical keep/alias branches — for bin k and reference sigma the
branch lands on k if sigma < keep[k] else on alias[k], each branch
carrying probability 1 / (K 2^mu) — and that exact table distribution is
pinned against the ideal w / lambda within the derived discretization
bound (integer weights W_k = round(p_k K 2^mu) plus at most one unit of
residual redistribution: |W_k - ideal| <= 3/2, so
|p_table - p_ideal| <= 1.5 / (K 2^mu) per bin; no bare constants).

PREPARE followed by the hand-written PREPARE-dagger is the identity on the
full (index + garbage) register.

The resource tests at the bottom hold the documented cost against the
compiler: each PREPARE (and its adjoint) costs exactly
``qrom.toffoli_count + 2 mu`` Toffolis plus ``num_index`` Fredkins
— the lookup at the QROM's *own reported* price (whatever variant
``"auto"`` minted) plus one CDKM register comparator
(``cmp_ge_register``) on the ``mu``-bit operands at ``2 mu`` Toffolis.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import AliasSamplingPrepare


def _brute_force_marginal(prep: AliasSamplingPrepare) -> np.ndarray:
    """Enumerate the classical (bin, reference) branches independently."""
    num_bins = prep.num_bins
    unit = 1 << prep.mu
    marginal = np.zeros(num_bins)
    for k in range(num_bins):
        for sigma in range(unit):
            j = k if sigma < prep.keep[k] else prep.alias[k]
            marginal[j] += 1.0 / (num_bins * unit)
    return marginal


def _circuit_marginal(prep: AliasSamplingPrepare) -> np.ndarray:
    prepare = prep.kernel()
    num_index = prep.num_index
    num_garbage = prep.num_garbage

    @cudaq.kernel
    def run():
        index = cudaq.qvector(num_index)
        garbage = cudaq.qvector(num_garbage)
        prepare(index, garbage)

    state = np.array(cudaq.get_state(run))
    # Index register at qubits [0, num_index): its value is the low bits
    # of the basis index, so the marginal sums |amplitude|^2 over the
    # garbage (high) bits.
    return np.sum(np.abs(state.reshape(-1, 1 << num_index))**2, axis=0)


def _padded_ideal(weights, num_bins: int) -> np.ndarray:
    values = np.zeros(num_bins)
    values[:len(weights)] = np.asarray(weights, dtype=float)
    return values / values.sum()


CASES = [
    ([0.7, 0.2, 1.4], 2),
    ([0.7, 0.2, 1.4], 4),
    ([1.0, 3.5, 0.25, 2.0, 0.8], 3),
    ([2.5], 3),  # single weight: deterministic |0> index
    ([0.0, 0.0, 7.0, 0.0], 3),  # single nonzero weight
    ([1.0] * 8, 2),  # uniform: every bin full, exactly representable
    ([0.9, 0.0, 2.2, 0.0, 0.4, 1.3, 0.0, 3.1], 2),  # zero-weight entries
]


@pytest.mark.parametrize("weights,mu", CASES)
def test_alias_sampling_index_marginal(weights, mu):
    prep = AliasSamplingPrepare(weights, mu)
    brute = _brute_force_marginal(prep)
    # Level 1: circuit vs the exact integer-table distribution, at
    # simulator precision.
    np.testing.assert_allclose(_circuit_marginal(prep), brute, atol=1e-10)
    # The class's own table bookkeeping agrees with the enumeration.
    np.testing.assert_allclose(prep.table_probabilities, brute, atol=1e-15)
    # Level 2: table vs ideal within the derived mu-bit bound.
    assert prep.discretization_bound == \
        1.5 / (prep.num_bins * (1 << prep.mu))
    ideal = _padded_ideal(weights, prep.num_bins)
    np.testing.assert_allclose(prep.probabilities, ideal, atol=1e-15)
    assert np.max(np.abs(brute - ideal)) <= prep.discretization_bound + 1e-15


def test_alias_sampling_larger_random_table():
    rng = np.random.default_rng(23)
    weights = rng.random(16) * 3.0
    prep = AliasSamplingPrepare(weights, mu=2)
    assert prep.num_index == 4 and prep.num_bins == 16
    brute = _brute_force_marginal(prep)
    np.testing.assert_allclose(_circuit_marginal(prep), brute, atol=1e-10)
    ideal = _padded_ideal(weights, 16)
    assert np.max(np.abs(brute - ideal)) <= prep.discretization_bound + 1e-15


def test_alias_sampling_exact_dyadic_weights():
    # w / lambda = [1/4, 1/4, 1/2, 0]: representable exactly at any mu,
    # so the table distribution must match the ideal to fp precision.
    prep = AliasSamplingPrepare([1.0, 1.0, 2.0], mu=2)
    brute = _brute_force_marginal(prep)
    np.testing.assert_allclose(brute, [0.25, 0.25, 0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(_circuit_marginal(prep), brute, atol=1e-10)


def test_alias_sampling_uniform_is_exact():
    prep = AliasSamplingPrepare([1.0] * 8, mu=2)
    np.testing.assert_allclose(_brute_force_marginal(prep),
                               np.full(8, 1.0 / 8.0),
                               atol=1e-15)
    # Full bins are self-aliased with the keep clamped into mu bits.
    assert prep.alias == tuple(range(8))


@pytest.mark.parametrize("weights,mu", [([0.7, 0.2, 1.4], 2),
                                        ([1.0, 3.5, 0.25, 2.0, 0.8], 3),
                                        ([2.5], 3)])
def test_alias_sampling_prepare_then_adjoint_is_identity(weights, mu):
    prep = AliasSamplingPrepare(weights, mu)
    prepare = prep.kernel()
    unprepare = prep.adjoint_kernel()
    num_index = prep.num_index
    num_garbage = prep.num_garbage

    @cudaq.kernel
    def run():
        index = cudaq.qvector(num_index)
        garbage = cudaq.qvector(num_garbage)
        prepare(index, garbage)
        unprepare(index, garbage)

    state = np.array(cudaq.get_state(run))
    expected = np.zeros(1 << (num_index + num_garbage), dtype=np.complex128)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-10)


def test_alias_sampling_register_accounting():
    prep = AliasSamplingPrepare([1.0, 2.0, 3.0], mu=4)
    assert prep.num_index == 2
    assert prep.num_bins == 4
    # [alias(m) | keep(mu) | ref(mu) | flag |
    #  ladder(qrom.num_ladder) | carry] = m + 2mu + 2 + num_ladder.
    assert prep.num_garbage == 2 + 2 * 4 + 2 + prep.qrom.num_ladder
    assert prep.ladder_offset == 2 + 2 * 4 + 1
    # A 4-entry table prices out to the plain select walk, whose ladder
    # is one line per address bit — the layout then closes at
    # 2m + 2mu + 2.
    assert prep.qrom.variant == "select"
    assert prep.qrom.num_ladder == prep.num_index
    assert prep.num_garbage == 2 * 2 + 2 * 4 + 2
    assert prep.lam == pytest.approx(6.0)
    assert len(prep.keep) == 4 and len(prep.alias) == 4
    assert all(0 <= v < (1 << 4) for v in prep.keep)
    assert all(0 <= a < 4 for a in prep.alias)
    assert "mu=4" in repr(prep)


def test_alias_sampling_validation_raises():
    with pytest.raises(ValueError, match="weights must be non-empty"):
        AliasSamplingPrepare([], mu=2)
    with pytest.raises(ValueError, match="weights must be non-negative"):
        AliasSamplingPrepare([1.0, -0.5], mu=2)
    with pytest.raises(ValueError, match="weights must be finite"):
        AliasSamplingPrepare([1.0, float("inf")], mu=2)
    with pytest.raises(ValueError, match="sum to zero"):
        AliasSamplingPrepare([0.0, 0.0], mu=2)
    with pytest.raises(ValueError, match="mu must be a positive integer"):
        AliasSamplingPrepare([1.0, 2.0], mu=0)
    with pytest.raises(ValueError, match="mu must be a positive integer"):
        AliasSamplingPrepare([1.0, 2.0], mu=1.5)
    # bools are ints in Python but are rejected explicitly; integral
    # numpy ints stay accepted.
    with pytest.raises(ValueError, match="mu must be a positive integer"):
        AliasSamplingPrepare([1.0, 2.0], mu=True)
    assert AliasSamplingPrepare([1.0, 2.0], mu=np.int64(2)).mu == 2


# ----------------------------------------------------------------------
# Compiler-pinned resource contracts
# ----------------------------------------------------------------------

_RESOURCES = pytest.mark.skipif(
    not hasattr(cudaq, "estimate_resources"),
    reason="cudaq.estimate_resources is not available in this CUDA-Q")


def _prepare_resources(prep: AliasSamplingPrepare, adjoint: bool):
    kernel = prep.adjoint_kernel() if adjoint else prep.kernel()
    num_index = prep.num_index
    num_garbage = prep.num_garbage

    @cudaq.kernel
    def harness():
        index = cudaq.qvector(num_index)
        garbage = cudaq.qvector(num_garbage)
        kernel(index, garbage)

    return cudaq.estimate_resources(harness)


@_RESOURCES
@pytest.mark.parametrize("weights,mu", [([0.7, 0.2, 1.4], 2),
                                        ([1.0, 3.5, 0.25, 2.0, 0.8], 3),
                                        ([2.5], 3), ([1.0] * 8, 2)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_alias_sampling_cost_is_qrom_price_plus_comparator(
        weights, mu, adjoint):
    # The lookup cost is not re-derived here: it is asserted consistent
    # with the QROM's own reported count (whatever construction "auto"
    # priced in), on top of which sits exactly the one CDKM register
    # comparator (cmp_ge_register) — 2 mu Toffolis on the mu-bit
    # operands. The alias swap is num_index Fredkins (counted as cswap,
    # not ccx — ``count_controls`` is arity-aware; ``count("ccx")``
    # matches nothing).
    prep = AliasSamplingPrepare(weights, mu)
    resources = _prepare_resources(prep, adjoint)
    # Not a tautology: qrom.toffoli_count is classical bookkeeping in
    # QROM, deliberately cross-pinned here against the compiled circuit.
    assert resources.count_controls("x", 2) == \
        prep.qrom.toffoli_count + 2 * mu
    assert resources.count_controls("swap", 1) == prep.num_index
    # The mu reference Hadamards and the num_index bin Hadamards. (An
    # incidental property of the current QROM gate choices — the lookup
    # happens to add no h gates — not a contract.)
    assert resources.count("h") == prep.num_index + prep.mu


@_RESOURCES
def test_alias_sampling_forced_select_swap_variant():
    # Pins the docstring's variant-agnostic claim: force the lookup onto
    # the select_swap construction through the new passthrough at the
    # smallest simulable size (m = 2, mu = 2, B = 2: the ladder is
    # 1 + 2 * (m + mu) = 9 lines, 19 qubits total) and re-assert the
    # full PREPARE contract.
    weights = [0.7, 0.2, 1.4, 0.5]
    prep = AliasSamplingPrepare(weights,
                                mu=2,
                                variant="select_swap",
                                block_size=2)
    assert prep.qrom.variant == "select_swap"
    assert prep.num_index + prep.num_garbage <= 24
    # (i) Exact marginal amplitudes, against the independent enumeration.
    brute = _brute_force_marginal(prep)
    np.testing.assert_allclose(_circuit_marginal(prep), brute, atol=1e-10)
    np.testing.assert_allclose(prep.table_probabilities, brute, atol=1e-15)
    # (ii) PREPARE then the hand-written adjoint is the identity.
    prepare = prep.kernel()
    unprepare = prep.adjoint_kernel()
    num_index = prep.num_index
    num_garbage = prep.num_garbage

    @cudaq.kernel
    def roundtrip():
        index = cudaq.qvector(num_index)
        garbage = cudaq.qvector(num_garbage)
        prepare(index, garbage)
        unprepare(index, garbage)

    state = np.array(cudaq.get_state(roundtrip))
    expected = np.zeros(1 << (num_index + num_garbage), dtype=np.complex128)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-10)
    # (iii) The Toffoli cost identity holds for this variant too.
    for adjoint in (False, True):
        resources = _prepare_resources(prep, adjoint)
        assert resources.count_controls("x", 2) == \
            prep.qrom.toffoli_count + 2 * prep.mu


def test_alias_sampling_passthrough_validation_propagates():
    # QROM's own validation errors surface unchanged.
    with pytest.raises(ValueError, match="block_size"):
        AliasSamplingPrepare([1.0, 2.0, 3.0], mu=2, block_size=2)


@_RESOURCES
def test_alias_sampling_adjoint_costs_exactly_the_forward_price():
    # The hand-written adjoint is the literal gate-reversal: same gate
    # multiset, gate for gate.
    prep = AliasSamplingPrepare([0.9, 0.0, 2.2, 0.0, 0.4, 1.3, 0.0, 3.1], 2)
    forward = _prepare_resources(prep, adjoint=False)
    backward = _prepare_resources(prep, adjoint=True)
    for name, controls in (("x", 2), ("x", 1), ("swap", 1), ("x", 0)):
        assert forward.count_controls(name, controls) == \
            backward.count_controls(name, controls)
    assert forward.count("h") == backward.count("h")
