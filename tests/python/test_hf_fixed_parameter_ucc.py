import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms


def _assert_basis_state(state, index):
    expected = np.zeros_like(state, dtype=complex)
    expected[index] = 1.0
    np.testing.assert_allclose(state, expected, atol=1.0e-12)


@pytest.fixture
def double_precision_target():
    # Default simulator is single precision (~1e-7); the dense-reference
    # comparison below needs double precision for a tight tolerance.
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _operator_exp_evolve(operator_pool, parameters, ket, num_qubits):
    """Independent reference: apply prod_g exp(+i theta_g G_g) to ket using dense
    matrix exponentials of the pool operators G_g.

    The fixed_parameter_ucc kernel applies, per excitation group, the product of
    exp_pauli(theta*c_i, P_i) = exp(+i theta c_i P_i). The Pauli words within one
    UCC excitation generator mutually commute, so that product equals
    exp(+i theta sum_i c_i P_i) = exp(+i theta G_g). Computed purely with NumPy
    (no cudaq state-prep kernels), so it independently checks the kernel's sign,
    angle, grouping, and qubit ordering.
    """
    # Multiply by the full-width identity so every operator's dense matrix spans
    # all num_qubits in a consistent basis (some pool operators act on fewer).
    full_identity = cudaq.spin.i(0)
    for qubit in range(1, num_qubits):
        full_identity = full_identity * cudaq.spin.i(qubit)

    state = np.array(ket, dtype=np.complex128)
    for theta, op in zip(parameters, operator_pool):
        generator = np.array((op * full_identity).to_matrix(),
                             dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        propagator = eigenvectors @ (np.exp(
            1.0j * theta * eigenvalues)[:, None] * eigenvectors.conj().T)
        state = propagator @ state
    return state


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


def test_hartree_fock_open_shell_occupation():
    # Closed shell stays contiguous.
    assert list(algorithms.stateprep.make_hartree_fock_occupation(
        8, 4, 0)) == [0, 1, 2, 3]

    # Open shell (4 electrons, spin=2) must use the interleaved alpha/beta
    # convention of get_uccsd_excitations: occupied = {0, 1, 2, 4}, NOT the
    # contiguous {0, 1, 2, 3}.
    occupation = list(
        algorithms.stateprep.make_hartree_fock_occupation(8, 4, 2))
    assert occupation == [0, 1, 2, 4]

    # Cross-check against the determinant implied by the UCCSD excitations.
    singles_alpha, singles_beta, _, _, _ = (
        algorithms.stateprep.get_uccsd_excitations(8, 4, 2))
    implied = sorted(
        {s[0] for s in singles_alpha} | {s[0] for s in singles_beta})
    assert occupation == implied

    @cudaq.kernel
    def kernel(occupied_orbitals: list[int]):
        q = cudaq.qvector(8)
        algorithms.stateprep.hartree_fock_occupation(q, occupied_orbitals)

    state = np.asarray(cudaq.get_state(kernel, occupation), dtype=complex)
    _assert_basis_state(state, 0b00010111)  # qubits {0,1,2,4} set


def test_fixed_parameter_ucc_matches_dense_operator_reference(
        double_precision_target):
    # Independent check of the UCC kernel against dense matrix exponentials of
    # the operator pool (not against another cudaq kernel).
    num_qubits, num_electrons = 4, 2
    pool = algorithms.stateprep.make_uccsd_operator_pool(num_qubits,
                                                         num_electrons, 0)
    parameters = [0.13 * (i + 1) for i in range(len(pool))]
    plan = algorithms.stateprep.make_fixed_parameter_uccsd_plan(
        num_qubits, num_electrons, parameters)

    @cudaq.kernel
    def hf_only():
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock(q, 2)

    @cudaq.kernel
    def hf_then_ucc(thetas: list[float],
                    pauli_words: list[list[cudaq.pauli_word]],
                    coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.hartree_fock(q, 2)
        algorithms.stateprep.fixed_parameter_ucc(q, thetas, pauli_words,
                                                 coefficients)

    hf_ket = np.asarray(cudaq.get_state(hf_only), dtype=np.complex128)
    expected = _operator_exp_evolve(pool, parameters, hf_ket, num_qubits)
    actual = np.asarray(cudaq.get_state(hf_then_ucc, plan.parameters,
                                        plan.pauli_words, plan.coefficients),
                        dtype=np.complex128)
    np.testing.assert_allclose(actual, expected, atol=1.0e-9)


def test_fixed_parameter_ucc_plan_composes_with_validate_and_estimate():
    # Regression for the plan/validate interop bug: a plan from
    # make_fixed_parameter_ucc_plan must be accepted by validate and estimate.
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(4)
    parameters = [0.05 * (i + 1) for i in range(len(words))]
    plan = algorithms.stateprep.make_fixed_parameter_ucc_plan(
        words, coeffs, parameters, 4)

    algorithms.stateprep.validate_fixed_parameter_ucc_plan(plan)
    resources = algorithms.stateprep.estimate_fixed_parameter_ucc_resources(
        plan)
    assert resources.num_excitations == len(parameters)

    # C++ plans (from the uccsd/uccgsd/upccgsd makers) must validate too.
    cpp_plan = algorithms.stateprep.make_fixed_parameter_uccsd_plan(
        4, 2, [0.1, -0.2, 0.3])
    algorithms.stateprep.validate_fixed_parameter_ucc_plan(cpp_plan)
