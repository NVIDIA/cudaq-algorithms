# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PauliLCU ("alias", "unary") backend: the alias-sampled PREPARE +
unary-iteration (QROM) SELECT path, exercised through the PauliLCU API.

Correctness is validated against independent dense references over the
Pauli sum, two ways: the circuit against the exactly-encoded matrix (alias
table weights, dense Pauli-word unitaries) at simulator precision (1e-10),
and the exactly-encoded matrix against the ideal Hamiltonian within the
derived discretization bound. A small ``mu`` keeps the ancilla register
simulable (qubits grow with ``mu``); the tight 1e-10 check is
``mu``-independent.
"""

import functools

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import BlockEncoding, PauliLCU, PhaseSequence, QSVT, Walk
from cudaq_algorithms.common_kernels import state_from

# ----------------------------------------------------------------------
# Dense references (independent NumPy constructions)
# ----------------------------------------------------------------------

_P = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_matrix(word: str, n: int) -> np.ndarray:
    """Dense operator for a Pauli word, char ``word[q]`` on qubit ``q``,
    qubit 0 the least significant bit (CUDA-Q little-endian convention)."""
    factors = [_P[word[t]] for t in reversed(range(n))]
    return functools.reduce(np.kron, factors)


NUM_SYSTEM = 2
# Mixed-sign Pauli sum with an identity term (five terms -> a 3-bit index).
HAM = {"ZI": 0.6, "IY": -0.4, "XX": 0.5, "ZZ": -0.25, "II": 0.3}
MU = 4


def build(**kwargs) -> PauliLCU:
    return PauliLCU(HAM, prepare="alias", select="unary", mu=MU, **kwargs)


def ideal_hamiltonian() -> np.ndarray:
    dim = 1 << NUM_SYSTEM
    total = np.zeros((dim, dim), dtype=np.complex128)
    for word, coeff in HAM.items():
        total += coeff * pauli_matrix(word, NUM_SYSTEM)
    return total


def discretized_dense(enc: PauliLCU) -> np.ndarray:
    """The exactly encoded matrix: alias-table weights, dense Pauli-word
    unitaries, in PREPARE bin order (padding bins act as identity)."""
    dim = 1 << NUM_SYSTEM
    probabilities = enc._lcu.preparation.table_probabilities
    total = np.zeros((dim, dim), dtype=np.complex128)
    for k, (coeff, word) in enumerate(enc.terms):
        sign = 1.0 if coeff >= 0.0 else -1.0
        total += enc.alpha * probabilities[k] * sign * pauli_matrix(
            word, NUM_SYSTEM)
    for k in range(len(enc.terms), len(probabilities)):
        total += enc.alpha * probabilities[k] * np.eye(dim)
    return total


def discretization_atol(enc: PauliLCU) -> float:
    return (enc.alpha * enc._lcu.preparation.num_bins *
            enc._lcu.preparation.discretization_bound)


def random_ket(dim: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


def encoded_block(enc, kernel, ket):
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << enc.num_system]


def encoded_matrix(enc) -> np.ndarray:
    dim = 1 << enc.num_system
    kernel = enc.encode_kernel()
    block = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        column = np.zeros(dim, dtype=np.complex128)
        column[j] = 1.0
        block[:, j] = encoded_block(enc, kernel, column)
    return block


# ----------------------------------------------------------------------
# 1. Backend selection and validation
# ----------------------------------------------------------------------


def test_backend_selection_and_validation():
    exact = PauliLCU(HAM)
    assert exact._mode == "exact"
    alias = build()
    assert alias._mode == "alias"
    assert "alias" in repr(alias)
    # Mixed pairs are rejected.
    with pytest.raises(ValueError, match="mixed"):
        PauliLCU(HAM, prepare="alias", select="multiplexed")
    with pytest.raises(ValueError, match="mixed"):
        PauliLCU(HAM, prepare="exact", select="unary")
    # Bad names.
    with pytest.raises(ValueError, match="prepare"):
        PauliLCU(HAM, prepare="bogus", select="unary")
    with pytest.raises(ValueError, match="select"):
        PauliLCU(HAM, prepare="alias", select="bogus")


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
    # Alias ancilla: index + garbage + work, wider than the exact backend.
    assert enc.num_ancilla == enc._lcu.num_ancilla
    assert enc.num_ancilla > PauliLCU(HAM).num_ancilla


def test_block_encodes_hamiltonian():
    enc = build()
    block = encoded_matrix(enc) * enc.alpha
    np.testing.assert_allclose(block, discretized_dense(enc), atol=1e-10)
    np.testing.assert_allclose(block,
                               ideal_hamiltonian(),
                               atol=discretization_atol(enc))


def test_walk_kernel_applies_chebyshev_of_minus_h():
    enc = build()
    scaled = discretized_dense(enc) / enc.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=9)
    block1 = encoded_block(enc, enc.walk_kernel(1), ket)
    np.testing.assert_allclose(block1, -scaled @ ket, atol=1e-10)
    block2 = encoded_block(enc, enc.walk_kernel(2), ket)
    t2 = 2.0 * scaled @ scaled - np.eye(1 << NUM_SYSTEM)
    np.testing.assert_allclose(block2, t2 @ ket, atol=1e-10)


def test_qsvt_sequence_matches_signal_model():
    from test_qsvt import reference_response

    enc = build()
    h_scaled = discretized_dense(enc) / enc.alpha
    ket = random_ket(1 << NUM_SYSTEM, seed=23)
    sequence = PhaseSequence([0.23, -0.41, 0.11])
    kernel = QSVT(enc).kernel(sequence)
    block = encoded_block(enc, kernel, ket)
    eigenvalues, vectors = np.linalg.eigh(h_scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


def test_satisfies_block_encoding_protocol():
    enc = build()
    assert isinstance(enc, BlockEncoding)


# ----------------------------------------------------------------------
# 3. select_observable / Walk.moment are unavailable for the alias backend
# ----------------------------------------------------------------------


def test_select_observable_unavailable():
    enc = build()
    with pytest.raises(NotImplementedError, match="alias"):
        enc.select_observable()
    # Walk.moment (odd order) routes through select_observable, so it also
    # refuses rather than returning a wrong value.
    with pytest.raises(NotImplementedError):
        Walk(enc).moment(random_ket(1 << NUM_SYSTEM), 1)
