# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Hartree-Fock reference states and fixed-parameter UCC.

The ``fixed_parameter_ucc`` kernel is validated against dense matrix
exponentials of the operator pool built with pure numpy/scipy (an
independent reference, not another cudaq kernel), and the open-shell
occupation is cross-checked against the determinant implied by
``get_uccsd_excitations`` — the interleaved alpha/beta layout the two
must share.
"""

import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms

from dense_references import dense_matrix

stateprep = algorithms.stateprep

# ----------------------------------------------------------------------------
# Entry kernels and references
# ----------------------------------------------------------------------------


@cudaq.kernel
def _hartree_fock_entry(num_qubits: int, num_electrons: int):
    q = cudaq.qvector(num_qubits)
    stateprep.hartree_fock(q, num_electrons)


@cudaq.kernel
def _occupation_entry(num_qubits: int, occupied_orbitals: list[int]):
    q = cudaq.qvector(num_qubits)
    stateprep.hartree_fock_occupation(q, occupied_orbitals)


@cudaq.kernel
def _hf_ucc_entry(num_qubits: int, num_electrons: int, thetas: list[float],
                  words: list[list[cudaq.pauli_word]],
                  coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    stateprep.hartree_fock(q, num_electrons)
    stateprep.fixed_parameter_ucc(q, thetas, words, coeffs)


def _state(kernel, *args):
    return np.array(cudaq.get_state(kernel, *args))


def _assert_basis_state(state, index):
    expected = np.zeros_like(state, dtype=complex)
    expected[index] = 1.0
    np.testing.assert_allclose(state, expected, atol=1.0e-12)


def _assert_close(actual, reference):
    single_precision = np.dtype(cudaq.complex()) == np.dtype(np.complex64)
    tol = 5e-5 if single_precision else 1e-12
    assert np.max(np.abs(actual - reference)) < tol


def _dense_pool_evolution(pool, thetas, num_qubits, ket):
    """Independent dense reference: prod_g exp(i theta_g G_g) |ket>."""
    from scipy.linalg import expm

    state = np.array(ket, dtype=np.complex128)
    for theta, op in zip(thetas, pool):
        terms = [(float(term.evaluate_coefficient().real),
                  str(term.get_pauli_word(num_qubits)))
                 for term in cudaq.SpinOperator(op)]
        generator = dense_matrix(terms, num_qubits)
        state = expm(1.0j * theta * generator) @ state
    return state


# ----------------------------------------------------------------------------
# Hartree-Fock occupations and kernels
# ----------------------------------------------------------------------------


def test_hartree_fock_host_helpers():
    assert stateprep.make_hartree_fock_occupation(6, 4) == [0, 1, 2, 3]

    resources = stateprep.estimate_hartree_fock_resources(6, 4)
    assert resources.num_qubits == 6
    assert resources.num_electrons == 4
    assert resources.num_x_gates == 4

    explicit = stateprep.estimate_hartree_fock_occupation_resources(
        6, [0, 2, 5])
    assert explicit.num_electrons == 3
    assert explicit.num_x_gates == 3

    with pytest.raises(ValueError, match="num_electrons cannot exceed"):
        stateprep.make_hartree_fock_occupation(2, 3)
    with pytest.raises(ValueError, match="exceeds num_qubits"):
        stateprep.validate_hartree_fock_occupation(4, [0, 4])
    with pytest.raises(ValueError, match="unique"):
        stateprep.validate_hartree_fock_occupation(4, [0, 2, 2])
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.validate_hartree_fock_occupation(4, [0, -1])
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_hartree_fock_occupation(4, 2.5)


def test_hartree_fock_canonical_statevector():
    state = _state(_hartree_fock_entry, 4, 3)
    _assert_basis_state(state, 0b0111)


def test_hartree_fock_explicit_occupation_statevector():
    state = _state(_occupation_entry, 4, [0, 2])
    _assert_basis_state(state, 0b0101)


def test_hartree_fock_open_shell_occupation():
    # Closed shell stays contiguous.
    assert stateprep.make_hartree_fock_occupation(8, 4, 0) == [0, 1, 2, 3]

    # Open shell (4 electrons, spin=2) must use the interleaved alpha/beta
    # convention of get_uccsd_excitations: {0, 1, 2, 4}, NOT {0, 1, 2, 3}.
    occupation = stateprep.make_hartree_fock_occupation(8, 4, 2)
    assert occupation == [0, 1, 2, 4]

    # Cross-check against the determinant implied by the UCCSD excitations.
    singles_alpha, singles_beta, _, _, _ = stateprep.get_uccsd_excitations(
        8, 4, 2)
    implied = sorted({s[0]
                      for s in singles_alpha}
                     | {s[0]
                        for s in singles_beta})
    assert occupation == implied

    state = _state(_occupation_entry, 8, occupation)
    _assert_basis_state(state, 0b00010111)  # qubits {0, 1, 2, 4} set

    with pytest.raises(ValueError, match="must be even"):
        stateprep.make_hartree_fock_occupation(7, 3, 1)
    with pytest.raises(ValueError, match="spin cannot exceed"):
        stateprep.make_hartree_fock_occupation(8, 2, 3)
    with pytest.raises(ValueError, match="does not fit"):
        stateprep.make_hartree_fock_occupation(4, 4, 2)


# ----------------------------------------------------------------------------
# Fixed-parameter UCC host helpers
# ----------------------------------------------------------------------------


def test_fixed_parameter_ucc_pauli_lists_from_pool():
    pool = stateprep.make_uccsd_operator_pool(4, 2, 0)
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(pool, 4)

    assert len(words) == len(coeffs) == 3
    assert [len(group) for group in words] == [2, 2, 8]
    assert all(
        len(word_group) == len(coeff_group)
        for word_group, coeff_group in zip(words, coeffs))
    # Same group shapes and coefficients as the pool-specific helper.
    uccgsd_pool = stateprep.make_uccgsd_operator_pool(4)
    generic_words, generic_coeffs = \
        stateprep.get_fixed_parameter_ucc_pauli_lists(uccgsd_pool, 4)
    helper_words, helper_coeffs = stateprep.get_uccgsd_pauli_lists(4)
    assert [len(g) for g in generic_words] == [len(g) for g in helper_words]
    assert generic_coeffs == helper_coeffs


def test_fixed_parameter_ucc_pauli_lists_filter_and_reject():
    from cudaq import spin

    # Near-zero terms are dropped...
    op = 0.5 * spin.x(0) * spin.y(1) + 1.0e-15 * spin.z(0) * spin.z(1)
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists([op], 2)
    assert [len(group) for group in words] == [1]
    assert coeffs == [[0.5]]

    # ...unless the tolerance says otherwise.
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(
        [op], 2, coefficient_tolerance=0.0)
    assert [len(group) for group in words] == [2]

    # Complex coefficients cannot become exp_pauli angles.
    with pytest.raises(ValueError, match="real"):
        stateprep.get_fixed_parameter_ucc_pauli_lists(
            [0.5j * spin.x(0) * spin.y(1)], 2)
    with pytest.raises(ValueError, match="non-negative"):
        stateprep.get_fixed_parameter_ucc_pauli_lists([op], 2, -1.0)


def test_fixed_parameter_ucc_validation_and_resources():
    words, coeffs = stateprep.get_uccgsd_pauli_lists(4)
    parameters = [0.05 * (i + 1) for i in range(len(words))]

    stateprep.validate_fixed_parameter_ucc(4, parameters, words, coeffs)

    resources = stateprep.estimate_fixed_parameter_ucc_resources(4, words)
    assert resources.num_qubits == 4
    assert resources.num_excitations == len(parameters)
    assert resources.num_pauli_rotations == sum(len(g) for g in words)
    assert resources.max_pauli_rotations_per_excitation == 8

    with pytest.raises(ValueError, match="same length"):
        stateprep.validate_fixed_parameter_ucc(4, parameters[:-1], words,
                                               coeffs)
    with pytest.raises(ValueError, match="coefficient group"):
        stateprep.validate_fixed_parameter_ucc(4, parameters, words,
                                               [g[:-1] for g in coeffs])
    with pytest.raises(ValueError, match="exceeds num_qubits"):
        stateprep.validate_fixed_parameter_ucc(2, [0.1], [["XYZ"]], [[1.0]])
    with pytest.raises(ValueError, match="unsupported Pauli"):
        stateprep.validate_fixed_parameter_ucc(2, [0.1], [["XQ"]], [[1.0]])


# ----------------------------------------------------------------------------
# Fixed-parameter UCC kernel vs dense pool exponentials
# ----------------------------------------------------------------------------


def test_fixed_parameter_ucc_kernel_accepts_grouped_arguments():
    words, coeffs = stateprep.get_upccgsd_pauli_lists(4)
    parameters = [0.1, -0.2, 0.3]

    state = _state(_hf_ucc_entry, 4, 2, parameters, words, coeffs)
    assert np.isclose(np.linalg.norm(state), 1.0)


@pytest.mark.parametrize("num_qubits,num_electrons,spin",
                         [(4, 2, 0), (6, 3, 1), (8, 4, 2)])
def test_fixed_parameter_ucc_matches_dense_pool_exponential(
        num_qubits, num_electrons, spin):
    pool = stateprep.make_uccsd_operator_pool(num_qubits, num_electrons, spin)
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(
        pool, num_qubits)
    rng = np.random.default_rng(num_qubits * 100 + num_electrons)
    parameters = (0.3 * rng.standard_normal(len(pool))).tolist()
    occupation = stateprep.make_hartree_fock_occupation(
        num_qubits, num_electrons, spin)

    @cudaq.kernel
    def entry(num_qubits: int, occupied: list[int], thetas: list[float],
              pauli_words: list[list[cudaq.pauli_word]],
              coefficients: list[list[float]]):
        q = cudaq.qvector(num_qubits)
        stateprep.hartree_fock_occupation(q, occupied)
        stateprep.fixed_parameter_ucc(q, thetas, pauli_words, coefficients)

    actual = _state(entry, num_qubits, occupation, parameters, words, coeffs)

    hf_ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    hf_ket[sum(1 << orbital for orbital in occupation)] = 1.0
    reference = _dense_pool_evolution(pool, parameters, num_qubits, hf_ket)
    _assert_close(actual, reference)


# ----------------------------------------------------------------------------
# hartree_fock_ucc_kernel factory
# ----------------------------------------------------------------------------


def _prepared_state(num_qubits, state_prep):
    """Run a (qubits: qview) state-prep kernel on a fresh register."""

    @cudaq.kernel
    def entry():
        q = cudaq.qvector(num_qubits)
        state_prep(q)

    return _state(entry)


def test_factory_matches_grouped_argument_kernel():
    pool = stateprep.make_uccsd_operator_pool(4, 2, 0)
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(pool, 4)
    parameters = [0.1, -0.2, 0.3]

    manual = _state(_hf_ucc_entry, 4, 2, parameters, words, coeffs)
    factory = stateprep.hartree_fock_ucc_kernel(4,
                                                parameters,
                                                words,
                                                coeffs,
                                                num_electrons=2)
    np.testing.assert_allclose(_prepared_state(4, factory), manual, atol=1e-12)


def test_factory_open_shell_reference():
    pool = stateprep.make_uccsd_operator_pool(8, 4, 2)
    words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(pool, 8)
    zeros = [0.0] * len(pool)

    # num_electrons + spin and explicit occupation agree.
    by_spin = stateprep.hartree_fock_ucc_kernel(8,
                                                zeros,
                                                words,
                                                coeffs,
                                                num_electrons=4,
                                                spin=2)
    occupation = stateprep.make_hartree_fock_occupation(8, 4, 2)
    by_occupation = stateprep.hartree_fock_ucc_kernel(
        8, zeros, words, coeffs, occupied_orbitals=occupation)

    state = _prepared_state(8, by_spin)
    _assert_basis_state(state, 0b00010111)
    np.testing.assert_allclose(_prepared_state(8, by_occupation),
                               state,
                               atol=1e-12)


def test_factory_string_words_and_hf_only_shapes():
    # Plain string words work, and empty word groups degrade to HF-only.
    factory = stateprep.hartree_fock_ucc_kernel(2, [0.0], [["XY"]], [[1.0]],
                                                num_electrons=1)
    state = _prepared_state(2, factory)
    np.testing.assert_allclose(np.abs(state[0b01]), 1.0, atol=1e-6)

    hf_only = stateprep.hartree_fock_ucc_kernel(2, [], [], [], num_electrons=1)
    _assert_basis_state(_prepared_state(2, hf_only), 0b01)

    identity = stateprep.hartree_fock_ucc_kernel(2, [], [], [],
                                                 num_electrons=0)
    _assert_basis_state(_prepared_state(2, identity), 0b00)


def test_factory_validates_inputs():
    words, coeffs = stateprep.get_upccgsd_pauli_lists(4)

    with pytest.raises(ValueError, match="same length"):
        stateprep.hartree_fock_ucc_kernel(4, [0.1],
                                          words,
                                          coeffs,
                                          num_electrons=2)
    parameters = [0.0] * len(words)
    with pytest.raises(ValueError, match="exactly one"):
        stateprep.hartree_fock_ucc_kernel(4, parameters, words, coeffs)
    with pytest.raises(ValueError, match="exactly one"):
        stateprep.hartree_fock_ucc_kernel(4,
                                          parameters,
                                          words,
                                          coeffs,
                                          num_electrons=2,
                                          occupied_orbitals=[0, 1])
    with pytest.raises(ValueError, match="spin only applies"):
        stateprep.hartree_fock_ucc_kernel(4,
                                          parameters,
                                          words,
                                          coeffs,
                                          spin=2,
                                          occupied_orbitals=[0, 1])
    with pytest.raises(ValueError, match="exceeds num_qubits"):
        stateprep.hartree_fock_ucc_kernel(4,
                                          parameters,
                                          words,
                                          coeffs,
                                          occupied_orbitals=[0, 4])
    with pytest.raises(ValueError, match="num_electrons cannot exceed"):
        stateprep.hartree_fock_ucc_kernel(4,
                                          parameters,
                                          words,
                                          coeffs,
                                          num_electrons=6)
