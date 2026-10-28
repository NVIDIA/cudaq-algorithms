# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the block-encoding combinators.

Every reference is a dense NumPy matrix built from the operands' Pauli
terms (or, for orbital rotations, from an independent Jordan-Wigner
transform of the rotated one-body tensor); the combinators themselves
never feed a reference.
"""

import warnings

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import (BlockEncoding, PauliLCU, PhaseSequence, QSVT,
                              RotatedEncoding, SumEncoding, Walk, chemistry,
                              fermion, state_from)
from dense_references import dense_matrix, random_ket

NUM_QUBITS = 3

# ----------------------------------------------------------------------
# Operands: PauliLCUs of different ancilla widths on one 3-qubit system
# ----------------------------------------------------------------------

TERMS_A = {"ZII": 0.5, "IXI": 0.3, "XXZ": -0.2}  # 3 terms -> 2 ancillas
TERMS_B = {"IIZ": 0.7, "YYI": 0.1}  # 2 terms -> 1 ancilla
TERMS_C = {
    "ZZZ": 0.4,
    "XIX": 0.25,
    "IYZ": 0.15,
    "ZIX": 0.05,
    "IIX": 0.1
}  # 5 terms -> 3 ancillas


def _encoding(terms):
    return PauliLCU(terms)


def _dense(encoding):
    return dense_matrix(encoding.terms, encoding.num_system)


def _block(encoding, kernel, ket):
    out = np.array(cudaq.get_state(kernel, state_from(ket)))
    return out[:1 << encoding.num_system]


def _chebyshev(scaled, order, ket):
    previous, current = ket, scaled @ ket
    if order == 0:
        return ket
    for _ in range(order - 1):
        previous, current = current, 2.0 * (scaled @ current) - previous
    return current


def _controlled_circuit(encoding, control_value):
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


def _controlled_roundtrip(encoding, control_value):
    step = encoding.controlled_walk_step_kernel()
    adjoint = encoding.controlled_adjoint_walk_step_kernel()
    n_anc = encoding.num_ancilla
    flip = control_value

    @cudaq.kernel
    def circuit(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(n_anc + 1)
        if flip == 1:
            x(control_and_ancilla[0])
        step(control_and_ancilla, system)
        adjoint(control_and_ancilla, system)
        if flip == 1:
            x(control_and_ancilla[0])

    return circuit


SUM_CASES = {
    "scale": [(2.5, TERMS_A)],
    "two-positive": [(1.0, TERMS_A), (1.0, TERMS_B)],
    "two-mixed-sign": [(1.0, TERMS_A), (-0.7, TERMS_B)],
    "three-mixed-widths": [(0.3, TERMS_A), (-1.2, TERMS_B), (0.5, TERMS_C)],
    "five-repeated": [(1.0, TERMS_A), (-1.0, TERMS_B), (2.0, TERMS_C),
                      (-0.5, TERMS_A), (0.25, TERMS_B)],
}


def _sum_case(name):
    terms = [(c, _encoding(t)) for c, t in SUM_CASES[name]]
    dense = sum(c * _dense(e) for c, e in terms)
    return SumEncoding(terms), dense, terms


# ----------------------------------------------------------------------
# SumEncoding: protocol, geometry and normalization
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,num_slots", [("scale", 1), ("two-positive", 1),
                                            ("three-mixed-widths", 2),
                                            ("five-repeated", 3)])
def test_sum_protocol_and_geometry(name, num_slots):
    encoding, _, terms = _sum_case(name)
    assert isinstance(encoding, BlockEncoding)
    assert encoding.num_system == NUM_QUBITS
    assert encoding.num_slots == num_slots
    pool = max(e.num_ancilla for _, e in terms)
    assert encoding.num_ancilla == num_slots + 1 + pool
    assert encoding.alpha == pytest.approx(
        sum(abs(c) * e.alpha for c, e in terms))
    assert encoding.coefficients == tuple(c for c, _ in terms)
    assert len(encoding.summands) == len(terms)
    assert "SumEncoding" in repr(encoding)


@pytest.mark.parametrize("name", list(SUM_CASES))
def test_sum_block_is_weighted_sum_over_alpha(name):
    encoding, dense, _ = _sum_case(name)
    ket = random_ket(NUM_QUBITS, 3)
    block = _block(encoding, encoding.encode_kernel(), ket)
    np.testing.assert_allclose(block, dense @ ket / encoding.alpha, atol=1e-12)
    # The PauliLCU-shaped convenience makes sim_utils.action work too.
    np.testing.assert_allclose(sim.action(encoding, ket),
                               dense @ ket / encoding.alpha,
                               atol=1e-12)


def test_sum_of_pauli_lcus_matches_the_merged_pauli_lcu():
    """The degenerate case pins the normalization arithmetic exactly."""
    a, b = _encoding(TERMS_A), _encoding(TERMS_B)
    merged = PauliLCU({
        **{
            w: 0.6 * c
            for w, c in TERMS_A.items()
        },
        **{
            w: -1.5 * c
            for w, c in TERMS_B.items()
        }
    })
    combined = SumEncoding([(0.6, a), (-1.5, b)])
    assert combined.alpha == pytest.approx(merged.alpha)
    ket = random_ket(NUM_QUBITS, 4)
    np.testing.assert_allclose(_block(combined, combined.encode_kernel(), ket),
                               _block(merged, merged.encode_kernel(), ket),
                               atol=1e-12)


def test_sum_encode_kernel_with_injected_state_prep():
    encoding, dense, _ = _sum_case("two-mixed-sign")

    @cudaq.kernel
    def prep(qubits: cudaq.qview):
        x(qubits[0])
        h(qubits[2])

    ket = np.zeros(1 << NUM_QUBITS, dtype=np.complex128)
    ket[0b001] = ket[0b101] = 1.0 / np.sqrt(2.0)
    block = _block(encoding, encoding.encode_kernel(state_prep=prep), ket)
    np.testing.assert_allclose(block, dense @ ket / encoding.alpha, atol=1e-12)


# ----------------------------------------------------------------------
# SumEncoding: controlled variants, walk and QSVT consumers
# ----------------------------------------------------------------------


def test_sum_controlled_apply_control_conventions():
    encoding, dense, _ = _sum_case("three-mixed-widths")
    ket = random_ket(NUM_QUBITS, 5)
    dim = 1 << NUM_QUBITS

    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 0), state_from(ket)))
    expected = np.zeros_like(at_zero)
    expected[:dim] = ket
    np.testing.assert_allclose(at_zero, expected, atol=1e-12)

    at_one = np.array(
        cudaq.get_state(_controlled_circuit(encoding, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:dim],
                               dense @ ket / encoding.alpha,
                               atol=1e-12)


@pytest.mark.parametrize("control_value", [0, 1])
def test_sum_controlled_walk_roundtrip_is_identity(control_value):
    encoding, _, _ = _sum_case("three-mixed-widths")
    ket = random_ket(NUM_QUBITS, 6)
    out = np.array(
        cudaq.get_state(_controlled_roundtrip(encoding, control_value),
                        state_from(ket)))
    expected = np.zeros_like(out)
    expected[:1 << NUM_QUBITS] = ket
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_sum_walk_kernel_applies_chebyshev_of_minus_h():
    encoding, dense, _ = _sum_case("three-mixed-widths")
    walk = Walk(encoding)
    scaled = -dense / encoding.alpha
    ket = random_ket(NUM_QUBITS, 7)
    for power in (1, 2, 3, 4):  # odd powers covered on the statevector
        block = _block(encoding, walk.kernel(power=power), ket)
        np.testing.assert_allclose(block,
                                   _chebyshev(scaled, power, ket),
                                   atol=1e-12)


def test_sum_walk_even_moments_match_dense():
    encoding, dense, _ = _sum_case("five-repeated")
    walk = Walk(encoding)
    scaled = dense / encoding.alpha
    ket = random_ket(NUM_QUBITS, 8)
    for order in (0, 2, 4):
        expected = float(np.real(ket.conj() @ _chebyshev(scaled, order, ket)))
        assert walk.moment(ket, order) == pytest.approx(expected, abs=1e-10)


def test_sum_odd_moments_are_unavailable():
    encoding, _, _ = _sum_case("two-positive")
    with pytest.raises(NotImplementedError, match="select_observable"):
        encoding.select_observable()
    with pytest.raises(NotImplementedError):
        Walk(encoding).moment(random_ket(NUM_QUBITS, 1), 1)


def test_sum_qsvt_sequence_matches_signal_model():
    from test_qsvt import reference_response

    encoding, dense, _ = _sum_case("two-mixed-sign")
    scaled = dense / encoding.alpha
    ket = random_ket(NUM_QUBITS, 9)
    sequence = PhaseSequence([0.23, -0.41, 0.11, 0.37])
    block = _block(encoding, QSVT(encoding).kernel(sequence), ket)

    eigenvalues, vectors = np.linalg.eigh(scaled)
    coefficients = vectors.conj().T @ ket
    expected = (vectors * np.array(
        [reference_response(sequence, float(ev))
         for ev in eigenvalues])) @ coefficients
    np.testing.assert_allclose(block, expected, atol=1e-10)


# ----------------------------------------------------------------------
# RotatedEncoding
# ----------------------------------------------------------------------


def _rotated_case(n, seed):
    """A one-body operator h = V diag(eps) V^T on n modes (2n qubits).

    The inner encoding is diagonal in the rotated modes; the reference is
    the Jordan-Wigner transform of the spin-expanded one-body tensor.
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2.0
    eps, modes = np.linalg.eigh(h)
    nq = 2 * n
    words = {"I" * nq: float(np.sum(eps))}
    for m, e in enumerate(eps):
        for spin in range(2):
            word = ["I"] * nq
            word[2 * m + spin] = "Z"
            words["".join(word)] = -0.5 * float(e)
    diagonal = PauliLCU(words)
    rotated = RotatedEncoding.from_orbital_rotation(diagonal, modes)
    one_so, two_so = chemistry.spin_orbital_tensors(h, np.zeros((n, n, n, n)))
    reference = PauliLCU(fermion.jordan_wigner(one_so, two_so))
    return rotated, _dense(reference), nq


@pytest.mark.parametrize("n,seed", [(2, 5), (3, 6)])
def test_rotated_block_is_rotated_operator_over_alpha(n, seed):
    rotated, dense, nq = _rotated_case(n, seed)
    assert isinstance(rotated, BlockEncoding)
    assert rotated.num_system == nq
    assert rotated.num_ancilla == rotated.inner.num_ancilla
    assert rotated.alpha == pytest.approx(rotated.inner.alpha)
    ket = random_ket(nq, 9)
    block = _block(rotated, rotated.encode_kernel(), ket)
    np.testing.assert_allclose(block, dense @ ket / rotated.alpha, atol=1e-12)


def test_rotated_controlled_and_walk_kernels():
    rotated, dense, nq = _rotated_case(2, 5)
    ket = random_ket(nq, 10)
    dim = 1 << nq

    at_zero = np.array(
        cudaq.get_state(_controlled_circuit(rotated, 0), state_from(ket)))
    np.testing.assert_allclose(at_zero[:dim], ket, atol=1e-12)
    at_one = np.array(
        cudaq.get_state(_controlled_circuit(rotated, 1), state_from(ket)))
    np.testing.assert_allclose(at_one[:dim],
                               dense @ ket / rotated.alpha,
                               atol=1e-12)

    for control_value in (0, 1):
        out = np.array(
            cudaq.get_state(_controlled_roundtrip(rotated, control_value),
                            state_from(ket)))
        np.testing.assert_allclose(out[:dim], ket, atol=1e-12)

    walk = Walk(rotated)
    for power in (1, 3):
        block = _block(rotated, walk.kernel(power=power), ket)
        np.testing.assert_allclose(block,
                                   _chebyshev(-dense / rotated.alpha, power,
                                              ket),
                                   atol=1e-12)
    expected = float(
        np.real(ket.conj() @ _chebyshev(dense / rotated.alpha, 2, ket)))
    assert walk.moment(ket, 2) == pytest.approx(expected, abs=1e-10)
    with pytest.raises(NotImplementedError, match="select_observable"):
        rotated.select_observable()


def test_rotated_with_explicit_kernels():
    """A user-supplied basis change: conjugate Z words by Hadamards."""
    inner = PauliLCU({"ZI": 0.6, "IZ": -0.4})
    reference = PauliLCU({"XI": 0.6, "IX": -0.4})

    @cudaq.kernel
    def hadamards(system: cudaq.qview):
        for i in range(system.size()):
            h(system[i])

    rotated = RotatedEncoding(inner, hadamards, hadamards)
    ket = random_ket(2, 11)
    block = _block(rotated, rotated.encode_kernel(), ket)
    np.testing.assert_allclose(block,
                               _dense(reference) @ ket / rotated.alpha,
                               atol=1e-12)


def test_rotated_orbital_rotation_validation():
    inner = PauliLCU({"ZIII": 0.5})
    with pytest.raises(ValueError, match="square"):
        RotatedEncoding.from_orbital_rotation(inner, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="system qubits"):
        RotatedEncoding.from_orbital_rotation(inner, np.eye(3))
    with pytest.raises(ValueError, match="orthonormal"):
        RotatedEncoding.from_orbital_rotation(inner, np.ones((2, 2)))


# ----------------------------------------------------------------------
# Nesting: the combinators compose
# ----------------------------------------------------------------------


def test_nested_sum_of_sum_and_rotated():
    rotated, dense_t, nq = _rotated_case(2, 5)
    interaction = PauliLCU({"ZZII": 0.8, "IIZZ": 0.8})
    dense_u = _dense(interaction)
    inner_sum = SumEncoding([(1.0, rotated), (1.0, interaction)])
    outer = SumEncoding([(0.5, inner_sum), (-0.3, interaction),
                         (2.0, rotated)])
    dense = 0.5 * (dense_t + dense_u) - 0.3 * dense_u + 2.0 * dense_t
    assert isinstance(outer, BlockEncoding)
    assert outer.alpha == pytest.approx(0.5 * inner_sum.alpha +
                                        0.3 * interaction.alpha +
                                        2.0 * rotated.alpha)
    ket = random_ket(nq, 12)
    block = _block(outer, outer.encode_kernel(), ket)
    np.testing.assert_allclose(block, dense @ ket / outer.alpha, atol=1e-12)
    expected = float(
        np.real(ket.conj() @ _chebyshev(dense / outer.alpha, 2, ket)))
    assert Walk(outer).moment(ket, 2) == pytest.approx(expected, abs=1e-10)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_sum_input_validation():
    a = _encoding(TERMS_A)
    with pytest.raises(ValueError, match="at least one"):
        SumEncoding([])
    with pytest.raises(TypeError, match="pair"):
        SumEncoding([a])
    with pytest.raises(TypeError, match="real number"):
        SumEncoding([(True, a)])
    with pytest.raises(ValueError, match="finite and nonzero"):
        SumEncoding([(0.0, a)])
    with pytest.raises(ValueError, match="finite and nonzero"):
        SumEncoding([(float("nan"), a)])
    with pytest.raises(ValueError, match="system qubits"):
        SumEncoding([(1.0, a), (1.0, PauliLCU({"ZZ": 1.0}))])
    with pytest.raises(TypeError, match="BlockEncoding protocol"):
        SumEncoding([(1.0, object())])
    with pytest.raises(TypeError, match="BlockEncoding protocol"):
        RotatedEncoding(object(), None, None)


# ----------------------------------------------------------------------
# Controlled consumers respect the control (not just W W-dagger = I)
# ----------------------------------------------------------------------


def _assert_controlled_respects_control(encoding, make_kernel, make_controlled,
                                        ket):
    """Control |1>: the control=1 half equals the uncontrolled state and the
    control=0 half is empty. Control |0>: the identity (input state,
    ancillas in |0>). Layout: [system][control][ancilla], q[0] the LSB."""
    dim_s, dim_a = 1 << encoding.num_system, 1 << encoding.num_ancilla
    reference = np.asarray(cudaq.get_state(make_kernel(), state_from(ket)))
    on_state = np.asarray(cudaq.get_state(make_controlled(1), state_from(ket)))
    off_state = np.asarray(cudaq.get_state(make_controlled(0),
                                           state_from(ket)))
    on = on_state.reshape(dim_a, 2, dim_s)
    off = off_state.reshape(dim_a, 2, dim_s)
    np.testing.assert_allclose(on[:, 1, :].reshape(-1), reference, atol=1e-10)
    np.testing.assert_allclose(on[:, 0, :], 0.0, atol=1e-10)
    expected_off = np.zeros((dim_a, dim_s), dtype=np.complex128)
    expected_off[0] = ket
    np.testing.assert_allclose(off[:, 0, :], expected_off, atol=1e-10)
    np.testing.assert_allclose(off[:, 1, :], 0.0, atol=1e-10)


def _combinator(which):
    if which == "sum":
        return _sum_case("three-mixed-widths")[0]
    return _rotated_case(2, 5)[0]


@pytest.mark.parametrize("which", ["sum", "rotated"])
def test_controlled_walk_respects_control(which):
    encoding = _combinator(which)
    walk = Walk(encoding)
    ket = random_ket(encoding.num_system, 13)
    _assert_controlled_respects_control(
        encoding, lambda: walk.kernel(power=2),
        lambda c: walk.controlled_kernel(power=2, control_state=c), ket)


@pytest.mark.parametrize("which", ["sum", "rotated"])
def test_controlled_qsvt_respects_control(which):
    encoding = _combinator(which)
    transformer = QSVT(encoding)
    sequence = PhaseSequence([0.3, -0.4, 0.25],
                             walk_directions=["forward", "adjoint"])
    ket = random_ket(encoding.num_system, 16)
    _assert_controlled_respects_control(
        encoding, lambda: transformer.kernel(sequence),
        lambda c: transformer.controlled_kernel(sequence, control_state=c),
        ket)


# ----------------------------------------------------------------------
# More shapes: a signed rescaling, a rotation around a sum
# ----------------------------------------------------------------------


def test_single_negative_term_is_a_signed_rescaling():
    a = _encoding(TERMS_A)
    encoding = SumEncoding([(-2.0, a)])
    assert encoding.alpha == pytest.approx(2.0 * a.alpha)
    assert encoding.num_ancilla == 1 + 1 + a.num_ancilla
    dense = -2.0 * _dense(a)
    ket = random_ket(NUM_QUBITS, 14)
    np.testing.assert_allclose(_block(encoding, encoding.encode_kernel(), ket),
                               dense @ ket / encoding.alpha,
                               atol=1e-12)
    scaled = -dense / encoding.alpha
    walk = Walk(encoding)
    for power in (1, 2, 3):
        np.testing.assert_allclose(_block(encoding, walk.kernel(power=power),
                                          ket),
                                   _chebyshev(scaled, power, ket),
                                   atol=1e-12)


def test_rotated_around_sum():
    inner_sum, dense_inner, _ = _sum_case("two-mixed-sign")

    @cudaq.kernel
    def hadamards(system: cudaq.qview):
        for i in range(system.size()):
            h(system[i])

    rotated = RotatedEncoding(inner_sum, hadamards, hadamards)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    basis_change = np.kron(np.kron(hadamard, hadamard), hadamard)
    dense = basis_change @ dense_inner @ basis_change.T
    ket = random_ket(NUM_QUBITS, 15)
    np.testing.assert_allclose(_block(rotated, rotated.encode_kernel(), ket),
                               dense @ ket / rotated.alpha,
                               atol=1e-12)
    expected = float(
        np.real(ket.conj() @ _chebyshev(dense / rotated.alpha, 2, ket)))
    assert Walk(rotated).moment(ket, 2) == pytest.approx(expected, abs=1e-10)


# ----------------------------------------------------------------------
# The Givens network realizes V up to a sign per mode, on both spins
# ----------------------------------------------------------------------


def _one_particle_matrix(kernel, num_modes, spin):
    """The matrix a system-side kernel realizes on single-excitation states
    of one spin sector, plus the largest amplitude leaking out of it."""
    nq = 2 * num_modes
    matrix = np.zeros((num_modes, num_modes))
    leak = 0.0
    sector = [1 << (2 * q + spin) for q in range(num_modes)]
    for p in range(num_modes):
        ket = np.zeros(1 << nq, dtype=np.complex128)
        ket[sector[p]] = 1.0

        @cudaq.kernel
        def apply(state: cudaq.State):
            system = cudaq.qvector(state)
            kernel(system)

        out = np.asarray(cudaq.get_state(apply, state_from(ket)))
        assert np.abs(out.imag).max() < 1e-12
        matrix[:, p] = out.real[sector]
        outside = np.ones(out.size, dtype=bool)
        outside[sector] = False
        leak = max(leak, float(np.abs(out[outside]).max()))
    return matrix, leak


def test_orbital_rotation_network_realizes_the_matrix_up_to_mode_signs():
    # A mode swap has determinant -1, so the Givens sweep must end in a
    # sign matrix D != I: the realized one-particle matrix is V D with
    # D = diag(+-1), identically on both spins, and enter is its transpose.
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    rotated = RotatedEncoding.from_orbital_rotation(PauliLCU({"ZIII": 0.5}),
                                                    swap)
    assert rotated.num_rotations == 1
    realized = {}
    for spin in (0, 1):
        exit_matrix, leak = _one_particle_matrix(rotated._exit, 2, spin)
        assert leak < 1e-12
        np.testing.assert_allclose(np.abs(exit_matrix),
                                   np.abs(swap),
                                   atol=1e-12)
        signs = np.array(
            [exit_matrix[1, 0] / swap[1, 0], exit_matrix[0, 1] / swap[0, 1]])
        np.testing.assert_allclose(np.abs(signs), 1.0, atol=1e-12)
        np.testing.assert_allclose(exit_matrix,
                                   swap @ np.diag(signs),
                                   atol=1e-12)
        enter_matrix, leak = _one_particle_matrix(rotated._enter, 2, spin)
        assert leak < 1e-12
        np.testing.assert_allclose(enter_matrix, exit_matrix.T, atol=1e-12)
        realized[spin] = exit_matrix
    np.testing.assert_allclose(realized[0], realized[1], atol=1e-12)


# ----------------------------------------------------------------------
# Guards added at the factory boundary
# ----------------------------------------------------------------------


class _ZeroAncilla:
    """A protocol-shaped wrapper claiming no ancilla qubits."""

    num_ancilla = 0

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_sum_rejects_complex_string_and_ancilla_free_operands():
    a = _encoding(TERMS_A)
    with pytest.raises(ValueError, match="complex"):
        SumEncoding([(np.complex128(1.0 + 2.0j), a)])
    with pytest.raises(TypeError, match="real number"):
        SumEncoding([("2.0", a)])
    with pytest.raises(ValueError, match="at least one ancilla"):
        SumEncoding([(1.0, _ZeroAncilla(a))])


def test_rotated_rejects_non_kernels_and_degenerate_rotations():
    inner = PauliLCU({"ZI": 0.5})
    with pytest.raises(TypeError, match="enter"):
        RotatedEncoding(inner, None, None)
    with pytest.raises(ValueError, match="at least 2 modes"):
        RotatedEncoding.from_orbital_rotation(inner, np.array([[1.0]]))
    with pytest.raises(ValueError, match="real matrix"):
        RotatedEncoding.from_orbital_rotation(PauliLCU({"ZIII": 0.5}),
                                              np.eye(2, dtype=complex))


def test_from_orbital_rotation_warns_for_off_diagonal_inner():
    # The network reproduces V up to a sign per mode; a PauliLCU inner with
    # X or Y words sees those signs, a diagonal (I/Z) inner does not.
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    diagonal = PauliLCU({"ZIII": 0.5, "IZII": -0.25, "IIZZ": 0.75})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        RotatedEncoding.from_orbital_rotation(diagonal, rotation)
    hopping = PauliLCU({"XZXI": 0.5, "ZIII": 0.25})
    with pytest.warns(RuntimeWarning, match="X or Y factors"):
        RotatedEncoding.from_orbital_rotation(hopping, rotation)
