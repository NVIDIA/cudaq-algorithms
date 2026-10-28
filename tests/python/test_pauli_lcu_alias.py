# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PauliLCU ``backend="alias"``: the alias-sampled PREPARE + unary-iteration
(QROM) SELECT path, exercised through the PauliLCU API.

Correctness is validated against independent dense references over the
Pauli sum, two ways: the circuit against the exactly-encoded matrix (alias
table weights, dense Pauli-word unitaries) at simulator precision (1e-10),
and the exactly-encoded matrix against the ideal Hamiltonian within the
derived discretization bound. A small ``mu`` keeps the ancilla register
simulable (qubits grow with ``mu``); the tight 1e-10 check is
``mu``-independent. The walk runs in the PREPARE frame, so ``Walk.moment``
is pinned too: odd orders through the derived ``select_observable``, even
orders through the ``2^num_ancilla``-term reflection observable, on
instances small enough for the latter.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import (BlockEncoding, PauliLCU, PhaseSequence, QSVT,
                              Walk, select_observable)
from cudaq_algorithms.common_kernels import state_from
from test_lcu_block_encoding import (chebyshev_moment, encoded_block,
                                     encoded_matrix, random_ket)
from test_lcu_block_encoding import pauli_matrix as pauli_word_matrix

# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------


def pauli_matrix(word: str, n: int) -> np.ndarray:
    """Dense operator for a Pauli string, char ``word[q]`` on qubit ``q``."""
    return pauli_word_matrix(
        {
            q: ch.lower()
            for q, ch in enumerate(word) if ch != "I"
        }, n)


NUM_SYSTEM = 2
# Mixed-sign Pauli sum with an identity term (five terms -> a 3-bit index).
HAM = {"ZI": 0.6, "IY": -0.4, "XX": 0.5, "ZZ": -0.25, "II": 0.3}
MU = 4
# A one-qubit sum with non-commuting terms: small enough for Walk.moment's
# reflection observable (2^num_ancilla terms).
SMALL = {"Z": 0.6, "X": -0.4}


def build(hamiltonian=HAM, mu=MU, **kwargs) -> PauliLCU:
    return PauliLCU(hamiltonian, backend="alias", mu=mu, **kwargs)


def ideal_hamiltonian(hamiltonian=HAM, n=NUM_SYSTEM) -> np.ndarray:
    dim = 1 << n
    total = np.zeros((dim, dim), dtype=np.complex128)
    for word, coeff in hamiltonian.items():
        total += coeff * pauli_matrix(word, n)
    return total


def discretized_dense(enc: PauliLCU) -> np.ndarray:
    """The exactly encoded matrix: alias-table weights, dense Pauli-word
    unitaries, in PREPARE bin order (padding bins carry zero weight)."""
    n = enc.num_system
    dim = 1 << n
    probabilities = enc._lcu.preparation.table_probabilities
    np.testing.assert_array_equal(probabilities[enc.num_terms:], 0.0)
    total = np.zeros((dim, dim), dtype=np.complex128)
    for k, (coeff, word) in enumerate(enc.terms):
        sign = 1.0 if coeff >= 0.0 else -1.0
        total += enc.alpha * probabilities[k] * sign * pauli_matrix(word, n)
    return total


def discretization_atol(enc: PauliLCU) -> float:
    return (enc.alpha * enc._lcu.preparation.num_bins *
            enc._lcu.preparation.discretization_bound)


# ----------------------------------------------------------------------
# 1. Backend selection and validation
# ----------------------------------------------------------------------


def test_backend_selection_and_validation():
    exact = PauliLCU(HAM)
    assert exact.backend == "exact"
    assert exact.mu is None
    alias = build()
    assert alias.backend == "alias"
    assert alias.mu == MU
    assert "backend='alias'" in repr(alias)
    assert f"mu={MU}" in repr(alias)
    assert PauliLCU(HAM, backend="alias").mu == 8  # the alias default
    with pytest.raises(ValueError, match="backend"):
        PauliLCU(HAM, backend="bogus")
    # mu belongs to the alias table; the exact backend has none.
    with pytest.raises(ValueError, match="mu"):
        PauliLCU(HAM, mu=4)
    # mu is validated, not coerced.
    with pytest.raises(ValueError, match="mu must be an integer"):
        build(mu=2.5)
    with pytest.raises(ValueError, match="mu must be an integer"):
        build(mu=True)
    with pytest.raises(ValueError, match="mu must be >= 1"):
        build(mu=0)


def test_kernel_args_unavailable_for_alias():
    enc = build()
    with pytest.raises(NotImplementedError, match="exact"):
        enc.kernel_args


# ----------------------------------------------------------------------
# 2. One-norm, shape, block-encoding correctness
# ----------------------------------------------------------------------


def test_alpha_and_shape():
    enc = build()
    one_norm = 0.6 + 0.4 + 0.5 + 0.25 + 0.3
    assert enc.alpha == pytest.approx(one_norm, abs=1e-12)
    assert enc.num_system == NUM_SYSTEM
    assert enc.num_terms == 5
    assert enc.constant_term == pytest.approx(0.3, abs=1e-12)
    # Alias ancilla geometry: index + alias garbage + SELECT work.
    assert enc._lcu.num_index == 3  # ceil(log2(5))
    assert enc._lcu.num_garbage == 16  # 2 m + 2 mu + 2
    assert enc._lcu.num_select_work == 1  # Pauli bodies: the minimum
    assert enc.num_ancilla == 20
    assert PauliLCU(HAM).num_ancilla == 3


def test_block_encodes_hamiltonian():
    enc = build()
    block = encoded_matrix(enc) * enc.alpha
    np.testing.assert_allclose(block, discretized_dense(enc), atol=1e-10)
    np.testing.assert_allclose(block,
                               ideal_hamiltonian(),
                               atol=discretization_atol(enc))


SMALL_CASES = [
    pytest.param({"X": 1.0}, {}, id="single-term"),
    pytest.param({"XZ": -0.7}, {}, id="single-negative-term"),
    pytest.param({"II": 0.5}, {}, id="identity-only"),
    pytest.param({
        "ZI": 0.6,
        "II": -0.3
    }, {},
                 id="negative-identity-among-terms"),
    pytest.param({
        "ZI": 0.6,
        "II": 0.3
    }, {"include_identity": False},
                 id="identity-excluded"),
]


@pytest.mark.parametrize("hamiltonian, kwargs", SMALL_CASES)
def test_small_hamiltonians_encode_exactly(hamiltonian, kwargs):
    # Single terms (a one-bit index with a padding bin), identity-only and
    # negative identity terms, and an excluded identity (reported as the
    # constant, absent from the block), all at mu = 1.
    enc = build(hamiltonian, mu=1, **kwargs)
    block = encoded_matrix(enc) * enc.alpha
    np.testing.assert_allclose(block, discretized_dense(enc), atol=1e-10)
    kept = {
        w: c
        for w, c in hamiltonian.items() if not (
            kwargs.get("include_identity", True) is False and set(w) == {"I"})
    }
    np.testing.assert_allclose(block,
                               ideal_hamiltonian(kept, enc.num_system),
                               atol=discretization_atol(enc))
    if kwargs.get("include_identity", True) is False:
        assert enc.constant_term == pytest.approx(0.3, abs=1e-12)


def test_spin_operator_input_matches_mapping_input():
    spin_op = (0.6 * cudaq.spin.z(0) - 0.4 * cudaq.spin.y(1) +
               0.5 * cudaq.spin.x(0) * cudaq.spin.x(1) -
               0.25 * cudaq.spin.z(0) * cudaq.spin.z(1) +
               0.3 * cudaq.spin.i(0))
    from_operator = build(spin_op, mu=2)
    from_mapping = build(mu=2)
    assert from_operator.terms == from_mapping.terms
    ket = random_ket(1 << NUM_SYSTEM, seed=5)
    np.testing.assert_allclose(encoded_block(from_operator,
                                             from_operator.encode_kernel(),
                                             ket),
                               encoded_block(from_mapping,
                                             from_mapping.encode_kernel(),
                                             ket),
                               atol=1e-12)


def test_state_prep_injection_matches_explicit_state():
    enc = build(mu=2)
    quantized = discretized_dense(enc)

    @cudaq.kernel
    def prep(qubits: cudaq.qview):
        h(qubits[0])
        x(qubits[1])

    out = np.array(cudaq.get_state(enc.encode_kernel(state_prep=prep)))
    block = out[:1 << NUM_SYSTEM] * enc.alpha
    # q0=|+> (LSB), q1=|1> (MSB): index = q1*2 + q0 -> {2, 3}.
    ket = np.zeros(1 << NUM_SYSTEM, dtype=np.complex128)
    ket[0b10] = ket[0b11] = 1 / np.sqrt(2)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)


# ----------------------------------------------------------------------
# 3. Walk and QSVT through the consumers
# ----------------------------------------------------------------------


def test_walk_kernel_applies_chebyshev_of_minus_h():
    enc = build()
    scaled = discretized_dense(enc) / enc.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=9)
    block1 = encoded_block(enc, enc.walk_kernel(1), ket)
    np.testing.assert_allclose(block1, -scaled @ ket, atol=1e-10)
    block2 = encoded_block(enc, enc.walk_kernel(2), ket)
    t2 = 2.0 * scaled @ scaled - np.eye(1 << NUM_SYSTEM)
    np.testing.assert_allclose(block2, t2 @ ket, atol=1e-10)


def test_walk_moments_match_chebyshev_sequence():
    # Odd orders measure the derived select_observable in the PREPARE
    # frame, even orders the geometry-only reflection observable; both must
    # agree with <T_k(H/alpha)> of the exactly encoded H.
    enc = build(SMALL, mu=2)
    scaled = discretized_dense(enc) / enc.alpha
    ket = random_ket(2, seed=3)
    moments = Walk(enc).moments(ket, 6)
    expected = [chebyshev_moment(scaled, ket, order) for order in range(6)]
    np.testing.assert_allclose(moments, expected, atol=1e-10)
    # The module-level function serves this backend too.
    assert str(select_observable(enc)) == str(enc.select_observable())


@pytest.mark.parametrize("order", [1, 3])
def test_odd_moments_on_multi_qubit_words(order):
    enc = build(mu=2)
    scaled = discretized_dense(enc) / enc.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=7)
    assert Walk(enc).moment(ket, order) == pytest.approx(chebyshev_moment(
        scaled, ket, order),
                                                         abs=1e-10)


def test_consumer_roundtrips_are_identity():
    enc = build(mu=2)
    walk = Walk(enc)
    dim = 1 << NUM_SYSTEM
    ket = random_ket(dim, seed=13)
    out = np.array(
        cudaq.get_state(walk.roundtrip_kernel(power=2), state_from(ket)))
    expected = np.zeros_like(out)
    expected[:dim] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)
    # The controlled roundtrip flips the control on and leaves it there:
    # |control=1, ancilla=0...0> x |ket> is the second block of amplitudes.
    out = np.array(
        cudaq.get_state(walk.controlled_roundtrip_kernel(power=1),
                        state_from(ket)))
    expected = np.zeros_like(out)
    expected[dim:2 * dim] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


@pytest.mark.parametrize("directions", [None, ["forward", "adjoint"]])
def test_qsvt_sequence_matches_signal_model(directions):
    from test_qsvt import reference_response

    enc = build()
    h_scaled = discretized_dense(enc) / enc.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=23)
    sequence = PhaseSequence([0.23, -0.41, 0.11], walk_directions=directions)
    kernel = QSVT(enc).kernel(sequence)
    block = encoded_block(enc, kernel, ket)
    eigenvalues, vectors = np.linalg.eigh(h_scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


# ----------------------------------------------------------------------
# 4. Protocol conformance and delegation
# ----------------------------------------------------------------------

FACTORIES = [
    "prepare_kernel",
    "unprepare_kernel",
    "apply_kernel",
    "controlled_apply_kernel",
    "walk_step_kernel",
    "adjoint_walk_step_kernel",
    "controlled_walk_step_kernel",
    "controlled_adjoint_walk_step_kernel",
]


def test_satisfies_block_encoding_protocol_and_delegates():
    enc = build()
    assert isinstance(enc, BlockEncoding)
    # Every protocol kernel factory is the LCUBlockEncoding's own kernel.
    for name in FACTORIES:
        assert getattr(enc, name)() is getattr(enc._lcu, name)(), name
    assert str(enc.select_observable()) == str(enc._lcu.select_observable())
