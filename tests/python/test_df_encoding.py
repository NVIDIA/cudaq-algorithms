# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""DoubleFactorizedEncoding: correctness against dense fermionic references.

The reference Hamiltonian is built from dense Jordan-Wigner ladder
operators (interleaved spins, qubit 0 least significant — the convention
of ``cudaq_algorithms.chemistry``), entirely in NumPy, so these tests
need no compiled extension and no external chemistry package.
"""

import numpy as np
import pytest

import cudaq

import cudaq_algorithms as algorithms
from cudaq_algorithms import BlockEncoding, PhaseSequence, QSVT, Walk
from cudaq_algorithms.common_kernels import state_from
from cudaq_algorithms.df_encoding import DoubleFactorizedEncoding

df = algorithms.double_factorization

# ----------------------------------------------------------------------
# Dense fermionic reference (JW ladders, interleaved spins)
# ----------------------------------------------------------------------

_I2 = np.eye(2)
_Z2 = np.diag([1.0, -1.0])
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])


def _annihilator(mode: int, num_modes: int) -> np.ndarray:
    ops = ([_Z2] * mode + [_LOWER] + [_I2] * (num_modes - mode - 1))[::-1]
    out = np.array([[1.0]])
    for op in ops:
        out = np.kron(out, op)
    return out


def dense_hamiltonian(one_body: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """H = sum h_pq E_pq + 1/2 sum (pq|rs) (E_pq E_rs - delta_qr E_ps)."""
    n = one_body.shape[0]
    m = 2 * n
    lower = [_annihilator(j, m) for j in range(m)]
    raise_ = [a.conj().T for a in lower]

    def excite(p, q):
        return (raise_[2 * p] @ lower[2 * q] +
                raise_[2 * p + 1] @ lower[2 * q + 1])

    dim = 1 << m
    h = np.zeros((dim, dim), dtype=complex)
    for p in range(n):
        for q in range(n):
            h += one_body[p, q] * excite(p, q)
            for r in range(n):
                for s in range(n):
                    h += 0.5 * eri[p, q, r, s] * (excite(p, q) @ excite(r, s) -
                                                  (q == r) * excite(p, s))
    return h


def random_system(seed: int, n: int = 2):
    """Random symmetric one-body matrix and PSD 8-fold-symmetric ERI."""
    rng = np.random.default_rng(seed)
    one_body = rng.normal(size=(n, n))
    one_body = 0.5 * (one_body + one_body.T)
    eri = np.zeros((n, n, n, n))
    for _ in range(2 * n):
        s = rng.normal(size=(n, n))
        s = 0.5 * (s + s.T)
        eri += float(rng.uniform(0.1, 1.0)) * np.einsum('pq,rs->pqrs', s, s)
    return one_body, eri


def random_ket(dim: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    return (ket / np.linalg.norm(ket)).astype(np.complex128)


def encoded_block(encoding, kernel, ket: np.ndarray) -> np.ndarray:
    """<0|_anc U |0_anc, ket>: the first 2**num_system amplitudes."""
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << encoding.num_system]


# ----------------------------------------------------------------------
# Protocol and construction
# ----------------------------------------------------------------------


def test_satisfies_block_encoding_protocol():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    assert isinstance(encoding, BlockEncoding)
    assert encoding.num_system == 4
    assert encoding.num_ancilla >= 1
    assert encoding.num_frames == 1 + encoding.factorization.num_leaves


def test_input_validation():
    one_body, eri = random_system(7)
    with pytest.raises(ValueError, match="square"):
        DoubleFactorizedEncoding(np.zeros((2, 3)), eri)
    with pytest.raises(ValueError, match="symmetric"):
        DoubleFactorizedEncoding(np.array([[0.0, 1.0], [0.0, 0.0]]), eri)
    with pytest.raises(ValueError, match="chemist-notation"):
        DoubleFactorizedEncoding(one_body, np.zeros((3, 3, 3, 3)))
    with pytest.raises(ValueError, match="orbitals"):
        DoubleFactorizedEncoding(
            np.eye(3), df.explicit_double_factorization(eri, threshold=0.0))
    with pytest.raises(ValueError, match="no retained terms"):
        DoubleFactorizedEncoding(np.zeros((2, 2)), np.zeros((2, 2, 2, 2)))


def test_alpha_matches_published_lcu_one_norm():
    one_body, eri = random_system(3, n=3)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    factorization = encoding.factorization

    kappa = one_body.copy()
    for rotation, core in zip(factorization.leaf_rotations,
                              factorization.leaf_cores):
        absorbed = core.sum(axis=1) - 0.5 * np.diag(core)
        kappa += (rotation * absorbed) @ rotation.T
    lam = df.double_factorization_one_norm(factorization,
                                           np.linalg.eigvalsh(kappa), "lcu")
    identity_weight = sum(
        abs(c) for c, qubits, _ in encoding.terms if not qubits)
    assert encoding.alpha - identity_weight == pytest.approx(lam, rel=1e-12)


# ----------------------------------------------------------------------
# Encode block == H / alpha
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 21])
def test_encode_block_is_h_over_alpha(seed):
    one_body, eri = random_system(seed)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-12)


def test_encode_block_three_orbitals():
    one_body, eri = random_system(5, n=3)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(64)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-12)


def test_scalar_offset_shifts_the_encoded_operator():
    one_body, eri = random_system(7)
    offset = 0.7137
    encoding = DoubleFactorizedEncoding(one_body, eri, scalar_offset=offset)
    h = dense_hamiltonian(one_body, eri) + offset * np.eye(16)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-12)


def test_truncated_factorization_encodes_truncated_hamiltonian():
    one_body, eri = random_system(7)
    truncated = df.explicit_double_factorization(eri, max_num_leaves=1)
    encoding = DoubleFactorizedEncoding(one_body, truncated)
    h = dense_hamiltonian(one_body, df.reconstruct_eri(truncated))
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-12)

    exact_alpha = DoubleFactorizedEncoding(one_body, eri).alpha
    assert encoding.alpha < exact_alpha


# ----------------------------------------------------------------------
# Walk and QSVT consumers
# ----------------------------------------------------------------------


def _chebyshev(dense_scaled: np.ndarray, order: int) -> np.ndarray:
    t_prev = np.eye(dense_scaled.shape[0], dtype=complex)
    t_cur = dense_scaled.copy()
    if order == 0:
        return t_prev
    for _ in range(order - 1):
        t_prev, t_cur = t_cur, 2.0 * dense_scaled @ t_cur - t_prev
    return t_cur


def test_walk_kernel_applies_chebyshev_of_minus_h():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    scaled = -dense_hamiltonian(one_body, eri) / encoding.alpha
    ket = random_ket(16)
    for power in (1, 2, 3):
        block = encoded_block(encoding, encoding.walk_kernel(power), ket)
        np.testing.assert_allclose(block,
                                   _chebyshev(scaled, power) @ ket,
                                   atol=1e-12)


def test_walk_even_moments_match_dense():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    walk = Walk(encoding)
    scaled = dense_hamiltonian(one_body, eri) / encoding.alpha
    ket = random_ket(16)
    for order in (0, 2, 4):
        expected = float(np.real(ket.conj() @ _chebyshev(scaled, order) @ ket))
        assert walk.moment(ket, order) == pytest.approx(expected, abs=1e-10)


def test_odd_moments_are_unavailable():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    with pytest.raises(NotImplementedError, match="select_observable"):
        encoding.select_observable()
    with pytest.raises(NotImplementedError):
        Walk(encoding).moment(random_ket(16), 1)


def test_qsvt_sequence_matches_signal_model():
    """A mixed-direction sequence against the shared 2x2 signal model."""
    from test_qsvt import reference_response

    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    h_scaled = dense_hamiltonian(one_body, eri) / encoding.alpha
    ket = random_ket(16)

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
# Controlled variants
# ----------------------------------------------------------------------


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
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(16)

    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    identity_expected = np.zeros_like(at_zero)
    identity_expected[:16] = ket
    np.testing.assert_allclose(at_zero, identity_expected, atol=1e-12)

    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:16], (h @ ket) / encoding.alpha,
                               atol=1e-12)


def test_walk_roundtrip_through_consumer_is_identity():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    walk = Walk(encoding)
    ket = random_ket(16)
    kernel = walk.roundtrip_kernel(power=2)
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    expected = np.zeros_like(out)
    expected[:16] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_identity_only_encoding():
    encoding = DoubleFactorizedEncoding(np.zeros((2, 2)),
                                        np.zeros((2, 2, 2, 2)),
                                        scalar_offset=1.5)
    assert encoding.num_terms == 1
    assert encoding.alpha == pytest.approx(1.5)
    assert encoding.constant_term == pytest.approx(1.5)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, ket, atol=1e-12)


def test_diagonal_system():
    # Diagonal one-body and a diagonal-core ERI: frames are (at most)
    # permutations of the computational orbitals.
    eri = np.einsum('pq,rs->pqrs', np.diag([0.4, 0.9]), np.diag([0.4, 0.9]))
    encoding = DoubleFactorizedEncoding(np.diag([0.7, -0.3]), eri)
    h = dense_hamiltonian(np.diag([0.7, -0.3]), eri)
    ket = random_ket(16)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-12)


# ----------------------------------------------------------------------
# state_prep injection (twin-pinning, as in test_state_prep_injection)
# ----------------------------------------------------------------------

_T0, _T1, _T2, _T3 = 0.37, -0.52, 0.21, -0.83


@cudaq.kernel
def _product_prep(qubits: cudaq.qview):
    rx(_T0, qubits[0])
    ry(_T1, qubits[1])
    rx(_T2, qubits[2])
    ry(_T3, qubits[3])


def _injected_ket() -> np.ndarray:
    """The dense statevector _product_prep produces (little-endian)."""
    rx0 = np.array([np.cos(0.5 * _T0), -1.0j * np.sin(0.5 * _T0)])
    ry1 = np.array([np.cos(0.5 * _T1), np.sin(0.5 * _T1)])
    rx2 = np.array([np.cos(0.5 * _T2), -1.0j * np.sin(0.5 * _T2)])
    ry3 = np.array([np.cos(0.5 * _T3), np.sin(0.5 * _T3)])
    out = np.array([1.0])
    for factor in (rx0, ry1, rx2, ry3):
        out = np.kron(factor, out)
    return out.astype(np.complex128)


def test_encode_kernel_prep_injection():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    via_prep = np.array(
        cudaq.get_state(encoding.encode_kernel(state_prep=_product_prep)))
    via_state = np.array(
        cudaq.get_state(encoding.encode_kernel(), state_from(_injected_ket())))
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)


def test_walk_kernel_prep_injection():
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    via_prep = np.array(
        cudaq.get_state(encoding.walk_kernel(power=2,
                                             state_prep=_product_prep)))
    via_state = np.array(
        cudaq.get_state(encoding.walk_kernel(power=2),
                        state_from(_injected_ket())))
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)


def test_prep_mode_returns_zero_argument_kernels():
    # The injected forms must be directly sampleable: no arguments at
    # all, through the sample launcher (a different marshaling path than
    # get_state).
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    for kernel in (
            encoding.encode_kernel(state_prep=_product_prep),
            encoding.walk_kernel(power=1, state_prep=_product_prep),
    ):
        counts = cudaq.sample(kernel, shots_count=100)
        assert sum(counts.values()) == 100
