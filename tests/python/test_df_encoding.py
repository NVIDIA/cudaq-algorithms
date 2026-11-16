# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DoubleFactorizedEncoding: correctness against dense fermionic references.

The encoding under test is the bring-your-own-encoding example
(``docs/sphinx/examples/python/df_encoding.py``) — a complete
``BlockEncoding`` built on the public API. Keeping its dense-reference
suite in CI both validates the example and pins the ``BlockEncoding``
protocol from a consumer's perspective.

The reference Hamiltonian is built from dense Jordan-Wigner ladder
operators (interleaved spins, qubit 0 least significant — the convention
of ``cudaq_algorithms.chemistry``), entirely in NumPy, so these tests
need no compiled extension and no external chemistry package.
"""

import math
import pathlib
import sys

import numpy as np
import pytest

import cudaq

import cudaq_algorithms as algorithms
from cudaq_algorithms import BlockEncoding, PhaseSequence, QSVT, Walk
from cudaq_algorithms.common_kernels import state_from

# The encoding under test is the bring-your-own-encoding example (CUDA-Q
# kernels need real .py files, so a plain path-based import works).
_EXAMPLE_DIR = (pathlib.Path(__file__).resolve().parents[2] / "docs" /
                "sphinx" / "examples" / "python")
sys.path.insert(0, str(_EXAMPLE_DIR))
from df_encoding import DoubleFactorizedEncoding

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


@pytest.mark.parametrize("seed,n", [(7, 2), (21, 2), (5, 3)])
def test_encode_block_is_h_over_alpha(seed, n):
    one_body, eri = random_system(seed, n=n)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(1 << (2 * n))
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


# ----------------------------------------------------------------------
# Review-driven coverage: symmetry, frame invariant, controlled walk step,
# single-ancilla sign
# ----------------------------------------------------------------------


def test_nonsymmetric_leaf_core_matches_symmetrized():
    # The physical leaf depends only on the symmetric part of its core, so an
    # encoding built from a user-supplied non-symmetric core must encode the
    # same operator as its symmetrized version (the one-body centering uses
    # the symmetric 1/2 (rowsum + colsum), matching the ZZ expansion).
    rng = np.random.default_rng(3)
    n = 2
    rotation, _ = np.linalg.qr(rng.normal(size=(n, n)))
    core = rng.normal(size=(n, n))  # deliberately non-symmetric
    one_body = 0.5 * rng.normal(size=(n, n))
    one_body = 0.5 * (one_body + one_body.T)
    asymmetric = df.DoubleFactorization(num_orbitals=n,
                                        leaf_rotations=[rotation],
                                        leaf_cores=[core],
                                        method="X-DF")
    symmetric = df.DoubleFactorization(num_orbitals=n,
                                       leaf_rotations=[rotation],
                                       leaf_cores=[0.5 * (core + core.T)],
                                       method="X-DF")
    ket = random_ket(1 << (2 * n))
    encoding_a = DoubleFactorizedEncoding(one_body, asymmetric)
    encoding_s = DoubleFactorizedEncoding(one_body, symmetric)
    block_a = encoded_block(encoding_a, encoding_a.encode_kernel(), ket)
    block_s = encoded_block(encoding_s, encoding_s.encode_kernel(), ket)
    np.testing.assert_allclose(block_a, block_s, atol=1e-10)


def test_num_frames_invariant_when_kappa_frame_is_elided():
    # Frame 0 (kappa eigenbasis) is retained as a zero-term placeholder even
    # when all its one-body words vanish, so num_frames == 1 + num_leaves and
    # frame 0 stays the kappa eigenbasis regardless of thresholding.
    rng = np.random.default_rng(9)
    n = 2
    rotation, _ = np.linalg.qr(rng.normal(size=(n, n)))
    core = rng.normal(size=(n, n))
    core = 0.5 * (core + core.T)
    factorization = df.DoubleFactorization(num_orbitals=n,
                                           leaf_rotations=[rotation],
                                           leaf_cores=[core],
                                           method="X-DF")
    # Choose one_body so kappa == 0 -> frame 0's one-body words all vanish.
    absorbed = (0.5 * (core.sum(axis=1) + core.sum(axis=0)) -
                0.5 * np.diag(core))
    one_body = -(rotation * absorbed) @ rotation.T
    encoding = DoubleFactorizedEncoding(one_body, factorization)
    assert encoding.num_frames == 1 + factorization.num_leaves
    # ...and it still block-encodes the right operator.
    eri = df.reconstruct_eri(factorization)
    h = dense_hamiltonian(one_body, eri)
    ket = random_ket(1 << (2 * n))
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, (h @ ket) / encoding.alpha, atol=1e-10)


@pytest.mark.parametrize("control_value", [0, 1])
def test_controlled_walk_step_roundtrip_is_identity(control_value):
    # Directly pins controlled_walk_step and controlled_adjoint_walk_step (and
    # the underlying controlled_walk / controlled_adjoint_walk): the composition
    # W then W-dagger is identity at control |1|, and both controlled ops are
    # identity at control |0|.
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    step = encoding.controlled_walk_step_kernel()
    adjoint_step = encoding.controlled_adjoint_walk_step_kernel()
    n_anc = encoding.num_ancilla
    flip = control_value

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        if flip == 1:
            x(control_and_ancilla[0])
        step(control_and_ancilla, system)
        adjoint_step(control_and_ancilla, system)
        if flip == 1:
            x(control_and_ancilla[0])

    ket = random_ket(16)
    out = np.array(cudaq.get_state(circuit, state_from(ket)))
    expected = np.zeros_like(out)
    expected[:16] = ket
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_single_ancilla_negative_sign_select():
    # scalar_offset=-1.5 with zero integrals: identity-only, one negative term
    # with num_ancilla == 1, which exercises the n_anc == 1 z(ancilla[0]) SELECT
    # sign branch (skipped by the +1.5 identity-only test).
    encoding = DoubleFactorizedEncoding(np.zeros((2, 2)),
                                        np.zeros((2, 2, 2, 2)),
                                        scalar_offset=-1.5)
    assert encoding.num_ancilla == 1
    assert encoding.alpha == pytest.approx(1.5)
    ket = random_ket(1 << encoding.num_system)
    block = encoded_block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, -ket, atol=1e-10)  # constant/alpha = -1


# ----------------------------------------------------------------------
# Pins for the example's inlined helper copies (the package keeps private
# twins; these mirror the package-side tests so the copies cannot silently
# diverge on load-bearing behavior).
# ----------------------------------------------------------------------


def test_example_prepare_angles_keep_tiny_sibling_terms():
    # Mirror of test_pauli_lcu.test_prepare_angles_keep_tiny_sibling_terms
    # against the example's inlined copy: the zero-subtree guard must fire
    # only for exact-zero (padded) subtrees; two retained sibling terms whose
    # combined probability is below any threshold still split 50/50.
    from df_encoding import _prepare_angles
    kept = [2e-12, 2e-12, 1e6]
    total = sum(kept)
    probabilities = [c / total for c in kept] + [0.0]
    angles = _prepare_angles(probabilities)
    assert angles[1] == pytest.approx(2.0 * math.asin(math.sqrt(0.5)))
    # Exact-zero padding subtrees still produce zero rotations.
    assert _prepare_angles([0.5, 0.5, 0.0, 0.0])[2] == 0.0


def test_walk_kernel_rejects_negative_power():
    # Pins the example's inlined _validate_power (mirrors the package twin):
    # both rejection branches, so a copy "simplified" to int(power) plus a
    # negativity check cannot silently truncate fractional powers.
    one_body, eri = random_system(7)
    encoding = DoubleFactorizedEncoding(one_body, eri)
    with pytest.raises(ValueError, match="non-negative integer"):
        encoding.walk_kernel(power=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        encoding.walk_kernel(power=1.5)


# ----------------------------------------------------------------------
# QROM frames: independent determinant-minor reference, no PySCF dependency.
# ----------------------------------------------------------------------


def qrom_fixture(**kwargs):
    theta = .37
    v = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    a = (v * [.6, -.2]) @ v.T
    one_body = np.array([[-.9, .17], [.17, -.4]])
    eri = np.einsum("pq,rs->pqrs", a, a)
    return DoubleFactorizedEncoding(one_body,
                                    eri,
                                    rotation_backend="qrom",
                                    **kwargs), one_body, eri


def fock_minors(orbital):
    occupied = [[p for p in range(len(orbital)) if j >> p & 1]
                for j in range(1 << len(orbital))]
    result = np.zeros((len(occupied), len(occupied)), dtype=complex)
    for s, rows in enumerate(occupied):
        for t, cols in enumerate(occupied):
            if len(rows) == len(cols):
                result[s, t] = np.linalg.det(orbital[np.ix_(rows, cols)])
    return result


def dense_terms(encoding, represented=True):
    frames = (encoding.represented_frame_matrices
              if represented else encoding.ideal_frame_matrices)
    lifts = [fock_minors(np.kron(f, np.eye(2))) for f in frames]
    operators = []
    for coefficient, qubits, frame in encoding.terms:
        diagonal = np.array([(-1)**sum((j >> q) & 1 for q in qubits)
                             for j in range(1 << encoding.num_system)])
        g = lifts[frame]
        operators.append(np.sign(coefficient) * (g * diagonal) @ g.conj().T)
    h = sum(abs(term[0]) * op for term, op in zip(encoding.terms, operators))
    return h, operators


def select_state(encoding, branches, *, controls=None):
    """Reusable full-state SELECT oracle; branches indexed by logical term."""
    width = encoding.rotation_report["term_index_width"]
    system_width = encoding.num_system
    enable_width = 0 if controls is None else 1
    total = system_width + encoding.num_ancilla + enable_width
    selector = encoding.select_kernel(controlled=controls is not None)
    initial = np.zeros(1 << total, dtype=complex)
    expected = np.zeros_like(initial)
    _, operators = dense_terms(encoding)
    control_amplitudes = [1.] if controls is None else controls
    for control, amplitude in enumerate(control_amplitudes):
        for j, ket in enumerate(branches):
            # Physical index qubit zero carries the logical MSB.
            address = int(f"{j:0{width}b}"[::-1], 2)
            indices = np.arange(1 << system_width) + (
                (control + (address << enable_width)) << system_width)
            initial[indices] = amplitude * ket
            op = operators[j] if j < len(operators) else np.eye(len(ket))
            if controls is not None and control == 0:
                op = np.eye(len(ket))
            expected[indices] = amplitude * (op @ ket)

    @cudaq.kernel
    def run(state: cudaq.State):
        q = cudaq.qvector(state)
        selector(q[system_width:total], q[0:system_width])

    actual = np.asarray(cudaq.get_state(run, state_from(initial)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_qrom_metadata_and_validation():
    encoding, one_body, eri = qrom_fixture()
    fixed = DoubleFactorizedEncoding(one_body, eri)
    assert encoding.terms == fixed.terms
    assert encoding.alpha == fixed.alpha
    report = encoding.rotation_report
    assert report["mixing_slots"] == 1
    assert report["program_width"] == 4
    assert report["total_ancillas"] == sum(report[k]
                                           for k in ("term_index_width",
                                                     "program_width",
                                                     "ladder_width"))
    assert report["table_size"] == 1 << report["term_index_width"]
    assert fixed.rotation_report["program_width"] == 0
    assert fixed.rotation_report["ladder_width"] == 0
    # Public arrays and metadata cannot mutate an encoding.
    copy = encoding.ideal_frame_matrices[0]
    copy[:] = 0
    assert np.linalg.norm(encoding.ideal_frame_matrices[0]) > 1
    report["program_width"] = -1
    assert encoding.rotation_report["program_width"] == 4
    with pytest.raises(ValueError, match="rotation_backend"):
        DoubleFactorizedEncoding(one_body, eri, rotation_backend="other")
    for bits in (0, 53, True, 2.5):
        with pytest.raises(ValueError, match="angle_bits"):
            qrom_fixture(angle_bits=bits)


def test_qrom_qr_reconstruction_spin_lift_and_bound():
    from df_encoding import _spatial_matrix, _spatial_program
    # Three spatial modes check chronology, and interleaved spectator parity.
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    modes, angles, f = _spatial_program(rotation)
    assert len(modes) == 3
    np.testing.assert_allclose(_spatial_matrix(3, modes, angles),
                               f,
                               rtol=0,
                               atol=1e-10)
    from cudaq_algorithms.stateprep import spatial_givens_network
    lift = fock_minors(np.kron(f, np.eye(2)))
    ket = random_ket(64, seed=7)

    @cudaq.kernel
    def circuit(state: cudaq.State):
        q = cudaq.qvector(state)
        spatial_givens_network(q, modes, angles)

    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(circuit, state_from(ket))),
                               lift @ ket,
                               rtol=0,
                               atol=1e-10)
    encoding, one_body, eri = qrom_fixture()
    kept, _ = dense_terms(encoding, represented=False)
    represented, _ = dense_terms(encoding)
    np.testing.assert_allclose(kept,
                               dense_hamiltonian(one_body, eri),
                               rtol=0,
                               atol=1e-10)
    bound = sum(
        abs(c) *
        min(2., 2 * encoding.rotation_report["frame_error_bounds"][frame])
        for c, qubits, frame in encoding.terms if qubits)
    error = np.linalg.norm(represented - kept, 2)
    assert 1e-3 < error <= bound + 1e-10


def test_qrom_each_term_bit_reversal_padding_and_coherent_control():
    encoding, _, _ = qrom_fixture()
    width = encoding.rotation_report["term_index_width"]
    branches = np.eye(1 << width, 16, dtype=complex)  # distinct system inputs
    # Check every physical address, including padded identities.
    for j in range(1 << width):
        isolated = np.zeros_like(branches)
        isolated[j] = random_ket(16, seed=7 + j)
        select_state(encoding, isolated)
    rng = np.random.default_rng(7)
    branches = rng.normal(size=(1 << width,
                                16)) + 1j * rng.normal(size=(1 << width, 16))
    branches /= np.linalg.norm(branches)
    select_state(encoding, branches)
    for controls in ([1., 0.], [0., 1.], [1 / np.sqrt(2), 1j / np.sqrt(2)]):
        select_state(encoding, branches, controls=controls)
    assert any(c < 0 for c, _, _ in encoding.terms)
    rows = encoding.rotation_report["angle_rows"]
    words = encoding.rotation_report["frame_words"]
    for j in range(1 << width):
        address = int(f"{j:0{width}b}"[::-1], 2)
        expected = 0
        if j < encoding.num_terms and encoding.terms[j][1]:
            expected = words[encoding.terms[j][2]][0]
        assert rows[address] == expected


def test_qrom_complete_block_and_consumers():
    from test_qsvt import reference_response
    encoding, _, _ = qrom_fixture()
    represented, _ = dense_terms(encoding)
    # All columns, rather than a single favorable initial state.
    block = np.column_stack([
        encoded_block(encoding, encoding.encode_kernel(),
                      np.eye(16, dtype=complex)[j]) for j in range(16)
    ])
    np.testing.assert_allclose(block,
                               represented / encoding.alpha,
                               rtol=0,
                               atol=1e-10)
    np.testing.assert_allclose(block, block.conj().T, rtol=0, atol=1e-10)
    ket = random_ket(16, seed=7)
    for order in (1, 2):
        actual = encoded_block(encoding, encoding.walk_kernel(order), ket)
        np.testing.assert_allclose(
            actual,
            _chebyshev(-represented / encoding.alpha, order) @ ket,
            rtol=0,
            atol=1e-10)
    sequence = PhaseSequence([.3, -.4, .25])
    eigenvalues, vectors = np.linalg.eigh(represented / encoding.alpha)
    expected = (vectors *
                [reference_response(sequence, float(x))
                 for x in eigenvalues]) @ (vectors.conj().T @ ket)
    np.testing.assert_allclose(encoded_block(encoding,
                                             QSVT(encoding).kernel(sequence),
                                             ket),
                               expected,
                               rtol=0,
                               atol=1e-10)


@pytest.mark.parametrize(
    "controls", [[1., 0.], [0., 1.], [1 / np.sqrt(2), 1j / np.sqrt(2)]])
def test_qrom_controlled_apply_full_phase(controls):
    encoding, _, _ = qrom_fixture()
    operation = encoding.controlled_apply_kernel()
    n_anc = encoding.num_ancilla
    ket = random_ket(16, seed=7)
    uncontrolled = np.asarray(
        cudaq.get_state(encoding.encode_kernel(), state_from(ket)))
    # System first, followed by enable, index, words, ladder.
    initial = np.zeros(1 << (5 + n_anc), dtype=complex)
    initial[:16] = controls[0] * ket
    initial[16:32] = controls[1] * ket
    expected = np.zeros_like(initial)
    expected[:16] = controls[0] * ket
    for j in range(len(uncontrolled) // 16):
        expected[32 * j + 16:32 * j +
                 32] = controls[1] * uncontrolled[16 * j:16 * j + 16]

    @cudaq.kernel
    def circuit(state: cudaq.State):
        q = cudaq.qvector(state)
        operation(q[4:5 + n_anc], q[0:4])

    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(circuit, state_from(initial))),
                               expected,
                               rtol=0,
                               atol=1e-10)


def test_qrom_prepare_injection_and_walk_adjoint():
    encoding, _, _ = qrom_fixture()
    prepare_index, unprepare_index = encoding.prepare_kernel(
    ), encoding.unprepare_kernel()
    n_anc = encoding.num_ancilla

    @cudaq.kernel
    def prepare_roundtrip(state: cudaq.State):
        q = cudaq.qvector(state)
        prepare_index(q)
        unprepare_index(q)

    ket = random_ket(1 << n_anc, seed=7)
    np.testing.assert_allclose(np.asarray(
        cudaq.get_state(prepare_roundtrip, state_from(ket))),
                               ket,
                               rtol=0,
                               atol=1e-10)
    for factory in (encoding.encode_kernel, encoding.walk_kernel):
        via_prep = np.asarray(
            cudaq.get_state(factory(state_prep=_product_prep)))
        via_state = np.asarray(
            cudaq.get_state(factory(), state_from(_injected_ket())))
        np.testing.assert_allclose(via_prep, via_state, rtol=0, atol=1e-10)
    ket = random_ket(16, seed=7)
    for controlled in (False, True):
        step = (encoding.controlled_walk_step_kernel()
                if controlled else encoding.walk_step_kernel())
        inverse = (encoding.controlled_adjoint_walk_step_kernel()
                   if controlled else encoding.adjoint_walk_step_kernel())
        total = n_anc + int(controlled)

        @cudaq.kernel
        def roundtrip(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(total)
            if controlled:
                h(ancilla[0])
                r1(.61, ancilla[0])
            step(ancilla, system)
            inverse(ancilla, system)
            if controlled:
                r1(-.61, ancilla[0])
                h(ancilla[0])

        actual = np.asarray(cudaq.get_state(roundtrip, state_from(ket)))
        expected = np.zeros_like(actual)
        expected[:16] = ket
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("n", [1, 2])
def test_qrom_identity_and_diagonal_edges(n):
    eri = np.zeros((n, ) * 4)
    for one_body, offset in ((np.zeros(
        (n, n)), -1.5), (np.diag(np.arange(n) + .2), .31)):
        encoding = DoubleFactorizedEncoding(one_body,
                                            eri,
                                            scalar_offset=offset,
                                            rotation_backend="qrom")
        if n == 1:
            assert encoding.rotation_report["program_width"] == 0
            assert encoding.rotation_report["ladder_width"] == 0
            assert encoding.rotation_report["table_size"] == 0
        ket = random_ket(1 << (2 * n), seed=7)
        expected = (dense_hamiltonian(one_body, eri) +
                    offset * np.eye(len(ket))) @ ket / encoding.alpha
        np.testing.assert_allclose(encoded_block(encoding,
                                                 encoding.encode_kernel(),
                                                 ket),
                                   expected,
                                   rtol=0,
                                   atol=1e-10)


def test_qrom_compressed_core_and_thresholded_empty_frame():
    rotation = np.array([[np.cos(.37), -np.sin(.37)],
                         [np.sin(.37), np.cos(.37)]])
    core = np.array([[.6, -.12], [-.12, .31]])
    factorization = df.DoubleFactorization(num_orbitals=2,
                                           leaf_rotations=[rotation],
                                           leaf_cores=[core],
                                           method="C-DF")
    absorbed = core.sum(axis=1) - .5 * np.diag(core)
    one_body = -(rotation *
                 absorbed) @ rotation.T  # kappa=0; frame 0 has no Z words
    fixed = DoubleFactorizedEncoding(one_body,
                                     factorization,
                                     coefficient_threshold=.01)
    qrom = DoubleFactorizedEncoding(one_body,
                                    factorization,
                                    coefficient_threshold=.01,
                                    rotation_backend="qrom")
    assert qrom.num_frames == 2
    assert qrom.terms == fixed.terms
    assert qrom.alpha == fixed.alpha
    assert not any(qubits and frame == 0 for _, qubits, frame in qrom.terms)
    kept, _ = dense_terms(qrom, represented=False)
    np.testing.assert_allclose(kept,
                               dense_hamiltonian(
                                   one_body,
                                   df.reconstruct_eri(factorization)),
                               rtol=0,
                               atol=1e-10)
    h, _ = dense_terms(qrom)
    ket = random_ket(16, seed=7)
    np.testing.assert_allclose(encoded_block(qrom, qrom.encode_kernel(), ket),
                               h @ ket / qrom.alpha,
                               rtol=0,
                               atol=1e-10)
