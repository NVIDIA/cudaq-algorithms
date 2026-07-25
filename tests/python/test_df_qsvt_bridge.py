# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Double factorization -> qubit Hamiltonian -> quantum primitives.

Pins the chemistry bridge (spin expansion + Jordan-Wigner) against
literature H2/STO-3G values, and the DF-compression path end to end:
factorize, reconstruct, bridge, and feed PauliLCU/Walk.
"""

import numpy as np
import pytest

import cudaq_algorithms as algorithms
from cudaq_algorithms import PauliLCU, Walk, chemistry

df = algorithms.double_factorization

# Only the qubit_hamiltonian path depends on fermion.jordan_wigner (imported
# lazily). Gate those tests on the real dependency via a try/except import,
# not on the compiled extension: keying to `_pycudaq_algorithms` would become
# a permanent skip once fermion is pure Python (None on every install then),
# silently disabling this whole file. spin_orbital_tensors is pure NumPy, so
# its tests stay ungated and run everywhere.
try:
    from cudaq_algorithms.fermion import jordan_wigner as _jw  # noqa: F401
    _HAS_FERMION = True
except Exception:
    _HAS_FERMION = False

_needs_fermion = pytest.mark.skipif(
    not _HAS_FERMION,
    reason="qubit_hamiltonian requires the fermion subpackage")

# H2 / STO-3G at R = 0.7414 A: standard MO-basis integrals (chemist
# notation, spatial orbitals); FCI total energy -1.137270 Ha.
H1 = np.array([[-1.25246357, 0.0], [0.0, -0.47594871]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = 0.67449876
ERI[1, 1, 1, 1] = 0.69716349
ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.66347258
ERI[0, 1, 0, 1] = ERI[1, 0, 1, 0] = 0.18128881
ERI[0, 1, 1, 0] = ERI[1, 0, 0, 1] = 0.18128881
E_NUCLEAR = 0.71375697
FCI_ENERGY = -1.137270


@_needs_fermion
def test_bridge_reproduces_h2_ground_state():
    spin_op = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    assert spin_op.qubit_count == 4
    ground = float(np.min(np.linalg.eigvalsh(spin_op.to_matrix())))
    assert ground == pytest.approx(FCI_ENERGY, abs=5e-5)


def test_bridge_validates_shapes():
    with pytest.raises(ValueError, match="square"):
        chemistry.spin_orbital_tensors(np.zeros((2, 3)), ERI)
    with pytest.raises(ValueError, match="chemist-notation"):
        chemistry.spin_orbital_tensors(H1, np.zeros((3, 3, 3, 3)))


@_needs_fermion
def test_full_rank_factorization_roundtrips_the_hamiltonian():
    factorization = df.explicit_double_factorization(ERI, threshold=0.0)
    assert df.factorization_error(ERI, factorization) < 1e-10

    direct = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    reconstructed = chemistry.qubit_hamiltonian(
        H1, df.reconstruct_eri(factorization), scalar_offset=E_NUCLEAR)

    direct_ground = float(np.min(np.linalg.eigvalsh(direct.to_matrix())))
    rebuilt_ground = float(
        np.min(np.linalg.eigvalsh(reconstructed.to_matrix())))
    assert rebuilt_ground == pytest.approx(direct_ground, abs=1e-8)

    assert PauliLCU(reconstructed).alpha == pytest.approx(
        PauliLCU(direct).alpha, abs=1e-8)


@_needs_fermion
def test_compression_reduces_alpha_and_bounds_energy_error():
    exact_alpha = PauliLCU(
        chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)).alpha

    truncated = df.explicit_double_factorization(ERI, max_num_leaves=1)
    truncated_h = chemistry.qubit_hamiltonian(H1,
                                              df.reconstruct_eri(truncated),
                                              scalar_offset=E_NUCLEAR)
    truncated_alpha = PauliLCU(truncated_h).alpha

    # Truncation drops LCU weight and shifts the spectrum by an amount
    # controlled by the reconstruction error.
    assert truncated_alpha < exact_alpha
    ground = float(np.min(np.linalg.eigvalsh(truncated_h.to_matrix())))
    tensor_error = df.factorization_error(ERI, truncated)
    assert abs(ground - FCI_ENERGY) < 4.0 * tensor_error + 5e-5


@_needs_fermion
def test_bridged_hamiltonian_feeds_the_walk():
    spin_op = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    encoding = PauliLCU(spin_op)
    walk = Walk(encoding)

    dense = np.asarray(spin_op.to_matrix())
    rng = np.random.default_rng(11)
    ket = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

    expected_t1 = float(np.real(ket.conj() @ (dense @ ket))) / encoding.alpha
    assert walk.moment(ket, 1) == pytest.approx(expected_t1, abs=1e-8)
    assert walk.moment(ket, 0) == pytest.approx(1.0, abs=1e-8)


def test_spin_orbital_tensors_direct_values():
    # Hand-computed spin expansion for a tiny input, independent of any
    # eigensolver -- catches index/transpose typos the H2 minimum-eigenvalue
    # assertion is blind to. (Pure NumPy, so it runs without fermion.)
    one, two = chemistry.spin_orbital_tensors(H1, ERI)
    # One-body: each spatial entry replicates onto the interleaved diagonal.
    assert one[0, 0] == pytest.approx(H1[0, 0])
    assert one[1, 1] == pytest.approx(H1[0, 0])
    assert one[2, 2] == pytest.approx(H1[1, 1])
    assert one[3, 3] == pytest.approx(H1[1, 1])
    # Two-body: coefficient of adag_p adag_q a_r a_s is 0.5 * eri reordered to
    # (p, r, s, q); check the four spin placements for the (0,0,0,0) block.
    value = 0.5 * ERI.transpose(0, 2, 3, 1)[0, 0, 0, 0]
    assert two[0, 0, 0, 0] == pytest.approx(value)
    assert two[1, 1, 1, 1] == pytest.approx(value)
    assert two[0, 1, 1, 0] == pytest.approx(value)
    assert two[1, 0, 0, 1] == pytest.approx(value)


def test_spin_orbital_tensors_rejects_asymmetric_eri_and_scalar():
    asymmetric = np.zeros((2, 2, 2, 2))
    asymmetric[0, 0, 1, 1] = 1.0  # symmetric partners unset
    with pytest.raises(ValueError, match="permutation symmetry"):
        chemistry.spin_orbital_tensors(H1, asymmetric)
    # ...but a caller who knows better can opt out.
    chemistry.spin_orbital_tensors(H1, asymmetric, validate_symmetry=False)
    # Scalar one_body gives a clear ValueError, not an IndexError.
    with pytest.raises(ValueError, match="square"):
        chemistry.spin_orbital_tensors(np.array(1.0), ERI)


@_needs_fermion
def test_bridged_operator_is_hermitian():
    # Independent of the min-eigenvalue check: a dropped +1 in the spin
    # expansion yields a non-Hermitian operator whose minimum eigenvalue is
    # unchanged to ~6 digits but whose full spectrum and LCU alpha are wrong.
    matrix = np.asarray(
        chemistry.qubit_hamiltonian(H1, ERI,
                                    scalar_offset=E_NUCLEAR).to_matrix())
    assert np.allclose(matrix, matrix.conj().T, atol=1e-10)


@_needs_fermion
def test_generic_symmetric_integrals_give_hermitian_operator():
    # A non-H2, n=3 case with generic (Cholesky-symmetric) integrals, so
    # index/transpose typos that cancel under H2's sparsity are exposed.
    rng = np.random.default_rng(5)
    n = 3
    one = rng.normal(size=(n, n))
    one = 0.5 * (one + one.T)
    eri = np.zeros((n, n, n, n))
    for _ in range(2 * n):
        factor = rng.normal(size=(n, n))
        factor = 0.5 * (factor + factor.T)
        eri += np.einsum("pq,rs->pqrs", factor, factor)  # 8-fold symmetric
    matrix = np.asarray(chemistry.qubit_hamiltonian(one, eri).to_matrix())
    assert matrix.shape == (1 << (2 * n), 1 << (2 * n))
    assert np.allclose(matrix, matrix.conj().T, atol=1e-10)


@_needs_fermion
def test_complex_dtype_eri_matches_real_path():
    # The bridge coerces to complex128 unconditionally; a complex-dtype eri
    # must give the same operator as the real one (exercises the coercion).
    real = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    complex_dtype = chemistry.qubit_hamiltonian(H1.astype(complex),
                                                ERI.astype(complex),
                                                scalar_offset=E_NUCLEAR)
    difference = np.max(
        np.abs(
            np.asarray(real.to_matrix()) -
            np.asarray(complex_dtype.to_matrix())))
    assert difference < 1e-12
