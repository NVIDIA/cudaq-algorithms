import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms


def _assert_basis_state(state, index):
    expected = np.zeros_like(state, dtype=complex)
    expected[index] = 1.0
    np.testing.assert_allclose(state, expected, atol=1.0e-12)


def test_hartree_fock_host_helpers():
    occupation = algorithms.stateprep.make_hartree_fock_occupation(6, 4)
    assert occupation == [0, 1, 2, 3]

    resources = algorithms.stateprep.estimate_hartree_fock_resources(6, 4)
    assert resources.num_qubits == 6
    assert resources.num_electrons == 4
    assert resources.num_x_gates == 4

    explicit_resources = algorithms.stateprep.estimate_hartree_fock_occupation_resources(
        6, [0, 2, 5])
    assert explicit_resources.num_electrons == 3
    assert explicit_resources.num_x_gates == 3

    with pytest.raises(ValueError, match="num_electrons"):
        algorithms.stateprep.make_hartree_fock_occupation(2, 3)
    with pytest.raises(ValueError, match="exceeds"):
        algorithms.stateprep.validate_hartree_fock_occupation(4, [0, 4])
    with pytest.raises(ValueError, match="unique"):
        algorithms.stateprep.validate_hartree_fock_occupation(4, [0, 2, 2])


def test_hartree_fock_canonical_statevector():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock(q, 3)

    state = np.asarray(cudaq.get_state(kernel), dtype=complex)
    _assert_basis_state(state, 0b0111)


def test_hartree_fock_explicit_occupation_statevector():
    occupation = [0, 2]

    @cudaq.kernel
    def kernel(occupied_orbitals: list[int]):
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock_occupation(q, occupied_orbitals)

    state = np.asarray(cudaq.get_state(kernel, occupation), dtype=complex)
    _assert_basis_state(state, 0b0101)


def test_fixed_parameter_ucc_plan_helpers():
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(4)
    parameters = [0.05 * (i + 1) for i in range(len(words))]

    plan = algorithms.stateprep.make_fixed_parameter_ucc_plan(
        words, coeffs, parameters, 4)
    assert plan.num_qubits == 4
    assert plan.parameters == pytest.approx(parameters)
    assert len(plan.pauli_words) == len(parameters)
    assert len(plan.coefficients) == len(parameters)

    resources = algorithms.stateprep.estimate_fixed_parameter_ucc_resources(
        plan)
    assert resources.num_qubits == 4
    assert resources.num_excitations == len(parameters)
    assert resources.num_pauli_rotations == sum(len(group) for group in words)
    assert resources.max_pauli_rotations_per_excitation == 8


def test_fixed_parameter_uccsd_plan_helper():
    parameters = [0.1, -0.2, 0.3]
    plan = algorithms.stateprep.make_fixed_parameter_uccsd_plan(
        4, 2, parameters)
    assert plan.num_qubits == 4
    assert plan.parameters == pytest.approx(parameters)
    assert len(plan.pauli_words) == 3
    assert len(plan.coefficients) == 3

    with pytest.raises(ValueError, match="same length"):
        algorithms.stateprep.make_fixed_parameter_uccsd_plan(4, 2, [0.1, 0.2])


def test_fixed_parameter_ucc_kernel_matches_uccgsd_kernel():
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(4)
    parameters = [0.03 * (i + 1) for i in range(len(words))]

    @cudaq.kernel
    def fixed_kernel(thetas: list[float], pauli_words: list[list[cudaq.pauli_word]],
                     coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock(q, 2)
        algorithms.stateprep.fixed_parameter_ucc(q, thetas, pauli_words,
                                                 coefficients)

    @cudaq.kernel
    def uccgsd_kernel(thetas: list[float], pauli_words: list[list[cudaq.pauli_word]],
                      coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock(q, 2)
        algorithms.stateprep.uccgsd(q, thetas, pauli_words, coefficients)

    fixed_state = np.asarray(cudaq.get_state(fixed_kernel, parameters, words,
                                             coeffs),
                             dtype=complex)
    uccgsd_state = np.asarray(cudaq.get_state(uccgsd_kernel, parameters, words,
                                              coeffs),
                              dtype=complex)

    np.testing.assert_allclose(fixed_state, uccgsd_state, atol=1.0e-12)


def test_fixed_parameter_ucc_kernel_accepts_plan_data():
    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(4)
    parameters = [0.1, -0.2, 0.3]
    plan = algorithms.stateprep.make_fixed_parameter_upccgsd_plan(4, parameters)

    assert len(plan.pauli_words) == len(words)
    assert [len(group) for group in plan.pauli_words] == [
        len(group) for group in words
    ]
    assert plan.coefficients == coeffs

    @cudaq.kernel
    def kernel(thetas: list[float], pauli_words: list[list[cudaq.pauli_word]],
               coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock(q, 2)
        algorithms.stateprep.fixed_parameter_ucc(q, thetas, pauli_words,
                                                 coefficients)

    state = np.asarray(
        cudaq.get_state(kernel, plan.parameters, plan.pauli_words,
                        plan.coefficients),
        dtype=complex)
    assert np.isclose(np.linalg.norm(state), 1.0)
