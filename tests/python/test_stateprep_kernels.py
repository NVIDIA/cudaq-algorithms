# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Validate the state-preparation kernels against dense references.

Each ansatz kernel is checked against an independently computed dense
reference: the ordered product of single-term matrix exponentials
``prod_j exp(i * theta_k * c_kj * P_kj)`` applied to the Hartree-Fock
determinant, with the terms taken from the operator pool in kernel
parameter order. The terms within one generator mutually commute, so the
reference is exact even where the circuits interleave the term
exponentials in a different order; the check pins the circuits, the pool
contents, and the parameter ordering.

Invalid-input tests pin the host-side validation: the error cases the
C++ bindings defined (odd qubit counts, odd electrons at spin 0, odd
spin-orbital counts) plus explicit guards where the C++ had undefined
behavior (counts that are negative, fractional, or mutually
inconsistent).
"""

import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms

from dense_references import dense_matrix

# ----------------------------------------------------------------------------
# Entry kernels
# ----------------------------------------------------------------------------


@cudaq.kernel
def _uccsd_entry(num_qubits: int, occupied: list[int], thetas: list[float],
                 num_electrons: int, spin: int):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.uccsd(q, thetas, num_electrons, spin)


@cudaq.kernel
def _uccgsd_entry(num_qubits: int, occupied: list[int], thetas: list[float],
                  words: list[list[cudaq.pauli_word]],
                  coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.uccgsd(q, thetas, words, coeffs)


@cudaq.kernel
def _upccgsd_entry(num_qubits: int, occupied: list[int], thetas: list[float],
                   words: list[list[cudaq.pauli_word]],
                   coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.upccgsd(q, thetas, words, coeffs)


@cudaq.kernel
def _ceo_entry(num_qubits: int, occupied: list[int], thetas: list[float],
               words: list[list[cudaq.pauli_word]], coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.ceo(q, thetas, words, coeffs)


# ----------------------------------------------------------------------------
# Dense reference machinery
# ----------------------------------------------------------------------------


def _hf_occupation(num_electrons, spin):
    if spin == 0:
        return list(range(num_electrons))
    n_occ_beta = (num_electrons - spin) // 2
    n_occ_alpha = num_electrons - n_occ_beta
    return sorted([2 * i for i in range(n_occ_alpha)] +
                  [2 * i + 1 for i in range(n_occ_beta)])


def _hf_ket(num_qubits, occupation):
    index = sum(1 << orbital for orbital in occupation)
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


def _thetas(seed, count):
    rng = np.random.default_rng(seed)
    return (0.4 * rng.standard_normal(count)).tolist()


def _pool_term_groups(ops, num_qubits):
    """(word, coefficient) term lists per pool operator, in term order."""
    return [[(str(term.get_pauli_word(num_qubits)),
              float(term.evaluate_coefficient().real))
             for term in cudaq.SpinOperator(op)] for op in ops]


def _dense_product_reference(term_groups, thetas, num_qubits, ket, scale=1.0):
    """Apply prod_k prod_j exp(i scale theta_k c_kj P_kj) to ket, in order.

    exp_pauli(angle, qubits, word) implements exp(i * angle * P), so the
    uccgsd/upccgsd/ceo kernels realize scale = +1. The uccsd CNOT-ladder
    circuit uses rz(0.5 * theta) / rz(0.125 * theta) gadgets, which come
    out as exp(-i * (theta / 2) * c * P) in the pool convention:
    scale = -1/2.
    """
    from scipy.linalg import expm

    for theta, terms in zip(thetas, term_groups):
        for word, coefficient in terms:
            generator = dense_matrix([(1.0, word)], num_qubits)
            ket = expm(1.0j * scale * theta * coefficient * generator) @ ket
    return ket


def _state(kernel, *args):
    return np.array(cudaq.get_state(kernel, *args))


def _assert_close(actual, reference):
    # fp32 targets (e.g. the default `nvidia` simulator) land around 5e-6.
    single_precision = np.dtype(cudaq.complex()) == np.dtype(np.complex64)
    tol = 5e-5 if single_precision else 1e-12
    assert np.max(np.abs(actual - reference)) < tol


# ----------------------------------------------------------------------------
# Dense-exponential validation
# ----------------------------------------------------------------------------


def _uccsd_circuit_signs(num_qubits, num_electrons, spin):
    """Per-excitation theta signs applied by the double_excitation circuit.

    The circuit negates theta for the index patterns (p < q, r > s) and
    (p > q, r < s) while the pool operators carry no such sign, so the
    reference has to apply it explicitly (this convention is inherited
    from the C++ implementation, where the discrepancy between the pool
    and the kernel is identical).
    """
    (singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
     doubles_beta) = algorithms.stateprep.get_uccsd_excitations(
         num_qubits, num_electrons, spin)
    signs = [1.0] * (len(singles_alpha) + len(singles_beta))
    for p, q, r, s in doubles_mixed + doubles_alpha + doubles_beta:
        flipped = (p < q and r > s) or (p > q and r < s)
        signs.append(-1.0 if flipped else 1.0)
    return signs


@pytest.mark.parametrize("num_qubits,num_electrons,spin",
                         [(4, 2, 0), (6, 3, 1), (8, 4, 0)])
def test_uccsd_kernel_matches_dense_exponential(num_qubits, num_electrons,
                                                spin):
    pool = algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_qubits * 100 + num_electrons, len(pool))
    signs = _uccsd_circuit_signs(num_qubits, num_electrons, spin)
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_uccsd_entry, num_qubits, occupation, thetas,
                    num_electrons, spin)
    reference = _dense_product_reference(
        term_groups, [t * s for t, s in zip(thetas, signs)],
        num_qubits,
        _hf_ket(num_qubits, occupation),
        scale=-0.5)
    _assert_close(actual, reference)


@pytest.mark.parametrize("num_qubits", [4, 6])
def test_uccgsd_kernel_matches_dense_exponential(num_qubits):
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(num_qubits)
    pool = algorithms.stateprep.make_uccgsd_operator_pool(num_qubits)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_qubits, len(words))
    occupation = list(range(num_qubits // 2))

    actual = _state(_uccgsd_entry, num_qubits, occupation, thetas, words,
                    coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


@pytest.mark.parametrize("num_qubits", [4, 8])
def test_upccgsd_kernel_matches_dense_exponential(num_qubits):
    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(num_qubits)
    pool = algorithms.stateprep.make_upccgsd_operator_pool(num_qubits)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_qubits + 7, len(words))
    occupation = list(range(num_qubits // 2))

    actual = _state(_upccgsd_entry, num_qubits, occupation, thetas, words,
                    coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


# num_orbitals = 4 is the smallest size with same-spin doubles
# (they need four descending spatial orbitals).
@pytest.mark.parametrize("num_orbitals", [2, 3, 4])
def test_ceo_kernel_matches_dense_exponential(num_orbitals):
    num_qubits = 2 * num_orbitals
    words, coeffs = algorithms.stateprep.get_ceo_pauli_lists(num_orbitals)
    pool = algorithms.stateprep.make_ceo_operator_pool(num_orbitals)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_orbitals + 13, len(words))
    occupation = list(range(num_qubits // 2))

    actual = _state(_ceo_entry, num_qubits, occupation, thetas, words, coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


# ----------------------------------------------------------------------------
# Host-side input validation
# ----------------------------------------------------------------------------


def test_invalid_inputs_raise():
    stateprep = algorithms.stateprep

    # Error cases the C++ bindings also defined.
    with pytest.raises(RuntimeError, match="should be even"):
        stateprep.get_uccsd_excitations(5, 2, 0)
    with pytest.raises(RuntimeError, match="spin multiplicity"):
        stateprep.get_uccsd_excitations(4, 3, 0)
    with pytest.raises(ValueError, match="even number"):
        stateprep.make_upccgsd_operator_pool(7)

    # Guards where the C++ had undefined behavior (unsigned underflow)
    # or rejected the value at the size_t type level.
    with pytest.raises(ValueError, match="num_electrons cannot exceed"):
        stateprep.get_uccsd_excitations(4, 6, 0)
    with pytest.raises(ValueError, match="spin cannot exceed"):
        stateprep.get_uccsd_excitations(4, 1, 3)
    with pytest.raises(ValueError, match="does not fit"):
        stateprep.get_uccsd_excitations(4, 4, 2)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.get_uccsd_excitations(4, 2.5, 0)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_uccgsd_operator_pool(-4)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_ceo_operator_pool(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.get_ceo_pauli_lists(1.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_ceo_operator_pool(True)
