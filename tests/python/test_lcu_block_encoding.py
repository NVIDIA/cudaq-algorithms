# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LCUBlockEncoding: correctness against independent dense references.

The generic LCU primitive is exercised here through a linear combination
of Pauli words ``H = sum_k c_k P_k`` (each ``P_k`` a Hermitian involution),
the family the ``cudaq_algorithms.primitives`` SELECT body vocabulary
expresses directly. Two-level validation, matching the library's culture:

- the circuit is compared against the *exactly encoded* matrix — the term
  unitaries built densely in NumPy, weighted by the alias tables'
  ``table_probabilities`` (integer-exact) — at simulator precision
  (1e-10), catching circuit/convention bugs undiluted by discretization;
- the exactly encoded matrix is compared against the ideal ``H`` within the
  derived bound ``alpha * num_bins * discretization_bound``;
- the term one-norm ``alpha = sum_k |c_k|`` is pinned classically.

``Walk.moment`` is not exercised: the even-moment reflection observable
expands to ``2^num_ancilla`` Pauli terms. Walk circuits are pinned densely
through ``walk_kernel`` powers instead.
"""

import functools

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import BlockEncoding, LCUBlockEncoding, PhaseSequence, QSVT, Walk
from cudaq_algorithms.common_kernels import state_from

# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------

_PAULI = {
    "i": np.eye(2, dtype=np.complex128),
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_matrix(word: dict, n: int) -> np.ndarray:
    """Dense ``P`` for ``word`` (``{qubit: 'x'/'y'/'z'}``), qubit 0 the
    least significant bit (CUDA-Q little-endian statevector convention)."""
    factors = [_PAULI[word.get(t, "i")] for t in reversed(range(n))]
    return functools.reduce(np.kron, factors)


# The shared fixture: a 2-qubit mixed-sign Hamiltonian with an X, a Y, a
# two-qubit XX, a ZZ and an identity term (five terms -> a 3-bit index with
# three zero-weight padding bins).
NUM_SYSTEM = 2
TERMS = [
    (0.6, {
        0: "z"
    }),
    (-0.4, {
        1: "y"
    }),
    (0.5, {
        0: "x",
        1: "x"
    }),
    (-0.25, {
        0: "z",
        1: "z"
    }),
    (0.3, {}),
]


def pauli_body(coeff: float, word: dict) -> list:
    body = [(word[q], q) for q in sorted(word)]
    if coeff < 0.0:
        body.append(("sign", ))
    return body


def build_encoding(terms=TERMS,
                   n=NUM_SYSTEM,
                   mu=4,
                   **kwargs) -> LCUBlockEncoding:
    weights = [abs(c) for c, _ in terms]
    bodies = [pauli_body(c, word) for c, word in terms]
    return LCUBlockEncoding(n, weights, bodies, mu=mu, **kwargs)


def ideal_hamiltonian(terms=TERMS, n=NUM_SYSTEM) -> np.ndarray:
    dim = 1 << n
    total = np.zeros((dim, dim), dtype=np.complex128)
    for coeff, word in terms:
        total += coeff * pauli_matrix(word, n)
    return total


def discretized_dense(encoding, terms=TERMS, n=NUM_SYSTEM) -> np.ndarray:
    """The exactly encoded matrix: alias-table weights, dense unitaries.

    Padding bins (index >= num_terms) act as the identity in SELECT; the
    integer Vose residual can leave them a nonzero table probability, so
    they are included.
    """
    dim = 1 << n
    probabilities = encoding.preparation.table_probabilities
    total = np.zeros((dim, dim), dtype=np.complex128)
    for k, (coeff, word) in enumerate(terms):
        sign = 1.0 if coeff >= 0.0 else -1.0
        total += (encoding.alpha * probabilities[k] * sign *
                  pauli_matrix(word, n))
    for k in range(len(terms), len(probabilities)):
        total += encoding.alpha * probabilities[k] * np.eye(dim)
    return total


def discretization_atol(encoding) -> float:
    return (encoding.alpha * encoding.preparation.num_bins *
            encoding.discretization_bound)


def random_ket(dim: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


def encoded_block(encoding, kernel, ket: np.ndarray) -> np.ndarray:
    """<0|_anc U |0_anc, ket>: the first 2**num_system amplitudes."""
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << encoding.num_system]


def encoded_matrix(encoding) -> np.ndarray:
    """Extract the encoded block column-by-column via cudaq.get_state."""
    dim = 1 << encoding.num_system
    kernel = encoding.encode_kernel()
    block = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        column = np.zeros(dim, dtype=np.complex128)
        column[j] = 1.0
        block[:, j] = encoded_block(encoding, kernel, column)
    return block


# ----------------------------------------------------------------------
# 1. The one-norm
# ----------------------------------------------------------------------


def test_alpha_is_the_term_one_norm():
    encoding = build_encoding()
    one_norm = 0.6 + 0.4 + 0.5 + 0.25 + 0.3
    assert encoding.alpha == pytest.approx(one_norm, abs=1e-12)
    # table_probabilities sum to 1, so the discretized one-norm is exact.
    assert np.sum(encoding.discretized_weights) == pytest.approx(one_norm,
                                                                 abs=1e-12)
    # alpha is a valid block-encoding normalization: >= the spectral norm.
    spectral = np.abs(np.linalg.eigvalsh(ideal_hamiltonian())).max()
    assert encoding.alpha >= spectral - 1e-12
    assert encoding.num_terms == 5


# ----------------------------------------------------------------------
# 2. Block-encoding correctness
# ----------------------------------------------------------------------


def test_block_encodes_the_hamiltonian():
    encoding = build_encoding()
    block = encoded_matrix(encoding) * encoding.alpha
    quantized = discretized_dense(encoding)
    # Circuit vs exact-encoded: undiluted by discretization.
    np.testing.assert_allclose(block, quantized, atol=1e-10)
    # Exact-encoded vs ideal: within the derived discretization bound.
    np.testing.assert_allclose(block,
                               ideal_hamiltonian(),
                               atol=discretization_atol(encoding))


def test_block_dense_action_on_random_ket():
    encoding = build_encoding()
    quantized = discretized_dense(encoding)
    ket = random_ket(1 << NUM_SYSTEM, seed=7)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block * encoding.alpha,
                               quantized @ ket,
                               atol=1e-10)


# ----------------------------------------------------------------------
# 3. Self-adjoint involution, walk Chebyshev, controlled conventions
# ----------------------------------------------------------------------


def test_apply_is_involution():
    encoding = build_encoding()
    apply_u = encoding.apply_kernel()
    n_anc = encoding.num_ancilla

    @cudaq.kernel
    def twice(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        apply_u(ancilla, system)
        apply_u(ancilla, system)

    ket = random_ket(1 << NUM_SYSTEM, seed=5)
    out = np.array(cudaq.get_state(twice, state_from(ket)))
    expected = np.zeros_like(out)
    expected[:1 << NUM_SYSTEM] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_walk_kernel_applies_chebyshev_of_minus_h():
    encoding = build_encoding()
    scaled = discretized_dense(encoding) / encoding.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=9)
    block1 = encoded_block(encoding, encoding.walk_kernel(1), ket)
    np.testing.assert_allclose(block1, -scaled @ ket, atol=1e-10)
    block2 = encoded_block(encoding, encoding.walk_kernel(2), ket)
    t2 = 2.0 * scaled @ scaled - np.eye(1 << NUM_SYSTEM)
    np.testing.assert_allclose(block2, t2 @ ket, atol=1e-10)


def test_walk_roundtrip_through_consumer_is_identity():
    encoding = build_encoding()
    walk = Walk(encoding)
    ket = random_ket(1 << NUM_SYSTEM, seed=13)
    out = np.array(
        cudaq.get_state(walk.roundtrip_kernel(power=2), state_from(ket)))
    expected = np.zeros_like(out)
    expected[:1 << NUM_SYSTEM] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def _controlled_circuit(encoding, control_value: int):
    controlled = encoding.controlled_apply_kernel()
    n_anc = encoding.num_ancilla
    flip = control_value

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        if flip == 1:
            x(control_and_ancilla[0])
        controlled(control_and_ancilla, system)
        if flip == 1:
            x(control_and_ancilla[0])

    return circuit


def test_controlled_apply_control_conventions():
    encoding = build_encoding()
    quantized = discretized_dense(encoding)
    ket = random_ket(1 << NUM_SYSTEM, seed=21)

    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    identity_expected = np.zeros_like(at_zero)
    identity_expected[:1 << NUM_SYSTEM] = ket
    np.testing.assert_allclose(at_zero, identity_expected, atol=1e-10)

    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:1 << NUM_SYSTEM],
                               (quantized @ ket) / encoding.alpha,
                               atol=1e-10)


def test_qsvt_sequence_matches_signal_model():
    from test_qsvt import reference_response

    encoding = build_encoding()
    h_scaled = discretized_dense(encoding) / encoding.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=23)

    sequence = PhaseSequence([0.23, -0.41, 0.11])
    kernel = QSVT(encoding).kernel(sequence)
    block = encoded_block(encoding, kernel, ket)

    eigenvalues, vectors = np.linalg.eigh(h_scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


# ----------------------------------------------------------------------
# 4. State-prep injection, inspection, protocol conformance
# ----------------------------------------------------------------------


def test_state_prep_injection_matches_explicit_state():
    encoding = build_encoding()
    quantized = discretized_dense(encoding)

    @cudaq.kernel
    def prep(qubits: cudaq.qview):
        h(qubits[0])
        x(qubits[1])

    kernel = encoding.encode_kernel(state_prep=prep)
    out = np.array(cudaq.get_state(kernel))
    block = out[:1 << NUM_SYSTEM] * encoding.alpha
    # q0=|+> (LSB), q1=|1> (MSB): index = q1*2 + q0 -> {2, 3}.
    ket = np.zeros(1 << NUM_SYSTEM, dtype=np.complex128)
    ket[0b10] = 1 / np.sqrt(2)
    ket[0b11] = 1 / np.sqrt(2)
    np.testing.assert_allclose(block, quantized @ ket, atol=1e-10)


def test_inspection_and_repr():
    encoding = build_encoding()
    assert encoding.num_system == NUM_SYSTEM
    assert encoding.num_index == 3  # ceil(log2(5))
    assert (encoding.num_ancilla == encoding.num_index + encoding.num_garbage +
            encoding.num_select_work)
    assert encoding.num_select_work >= 1
    assert encoding.mu == 4
    assert encoding.weights[:5] == pytest.approx([0.6, 0.4, 0.5, 0.25, 0.3])
    assert "LCUBlockEncoding" in repr(encoding)


def test_satisfies_block_encoding_protocol():
    encoding = build_encoding()
    assert isinstance(encoding, BlockEncoding)
    # Structural (Protocol) conformance, not inheritance.
    assert BlockEncoding not in type(encoding).__mro__


# ----------------------------------------------------------------------
# 5. select_observable: opt-in, otherwise unavailable
# ----------------------------------------------------------------------


def test_select_observable_unavailable_by_default():
    encoding = build_encoding()
    with pytest.raises(NotImplementedError, match="select_observable"):
        encoding.select_observable()


def test_select_observable_returns_supplied_operator():
    observable = cudaq.spin.z(0)
    encoding = build_encoding(select_observable=observable)
    assert encoding.select_observable() == observable


# ----------------------------------------------------------------------
# 6. Factory-boundary validation raises loudly
# ----------------------------------------------------------------------


def test_validation_raises():
    weights = [abs(c) for c, _ in TERMS]
    bodies = [pauli_body(c, w) for c, w in TERMS]
    with pytest.raises(ValueError, match="num_system"):
        LCUBlockEncoding(0, weights, bodies)
    with pytest.raises(ValueError, match="same length"):
        LCUBlockEncoding(NUM_SYSTEM, weights[:-1], bodies)
    with pytest.raises(ValueError, match="no terms"):
        LCUBlockEncoding(NUM_SYSTEM, [], [])
    with pytest.raises(ValueError, match="work"):
        # A body needing work[2] (num_work 3) capped at 2.
        LCUBlockEncoding(NUM_SYSTEM, [1.0], [[("and_wt", 0, 0, 2)]],
                         num_work=2)
    with pytest.raises(ValueError, match="mu"):
        LCUBlockEncoding(NUM_SYSTEM, weights, bodies, mu=0)
