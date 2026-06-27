import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms


def _reference_slater_state(occupied_orbitals):
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=complex)
    num_orbitals, num_electrons = occupied_orbitals.shape
    state = np.zeros(2**num_orbitals, dtype=complex)

    for basis_index in range(2**num_orbitals):
        occupied = [
            orbital for orbital in range(num_orbitals)
            if (basis_index >> orbital) & 1
        ]
        if len(occupied) == num_electrons:
            state[basis_index] = np.linalg.det(occupied_orbitals[np.ix_(
                occupied, range(num_electrons))])

    return state


def _assert_allclose_up_to_global_phase(actual, expected, atol=1.0e-10):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    pivot = int(np.argmax(np.abs(expected)))
    phase = 1.0
    if abs(expected[pivot]) > atol:
        phase = actual[pivot] / expected[pivot]
        phase /= abs(phase)
    np.testing.assert_allclose(actual, phase * expected, atol=atol)


def _prepare_real_slater_state(occupied_orbitals):
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=float)
    num_orbitals, num_electrons = occupied_orbitals.shape
    schedule = algorithms.stateprep.make_givens_rotation_schedule(
        occupied_orbitals)
    indices = algorithms.stateprep.get_givens_rotation_indices(schedule)
    angles = algorithms.stateprep.get_givens_rotation_angles(schedule)

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float]):
        q = cudaq.qvector(num_orbitals)
        algorithms.stateprep.prepare_slater_determinant(
            q, orbital_indices, rotation_angles, num_electrons)

    return np.asarray(cudaq.get_state(kernel, indices, angles))


def _prepare_complex_slater_state(occupied_orbitals):
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=complex)
    num_orbitals, num_electrons = occupied_orbitals.shape
    schedule = algorithms.stateprep.make_givens_rotation_schedule(
        occupied_orbitals)
    indices = algorithms.stateprep.get_givens_rotation_indices(schedule)
    angles = algorithms.stateprep.get_givens_rotation_angles(schedule)
    phases = algorithms.stateprep.get_givens_rotation_phases(schedule)

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float],
               rotation_phases: list[float], final_phases: list[float]):
        q = cudaq.qvector(num_orbitals)
        algorithms.stateprep.prepare_complex_slater_determinant(
            q, orbital_indices, rotation_angles, rotation_phases, final_phases,
            num_electrons)

    return np.asarray(
        cudaq.get_state(kernel, indices, angles, phases,
                        schedule.final_phases))


def test_givens_schedule_two_orbital_statevector():
    theta = 0.37
    occupied_orbitals = [[np.cos(theta)], [np.sin(theta)]]
    schedule = algorithms.stateprep.make_givens_rotation_schedule(
        occupied_orbitals)

    assert schedule.num_orbitals == 2
    assert schedule.num_electrons == 1
    assert len(schedule.rotations) == 1
    assert schedule.rotations[0].first_orbital == 0
    assert schedule.rotations[0].second_orbital == 1
    assert np.isclose(schedule.rotations[0].theta, theta)

    indices = algorithms.stateprep.get_givens_rotation_indices(schedule)
    angles = algorithms.stateprep.get_givens_rotation_angles(schedule)

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float]):
        q = cudaq.qvector(2)
        algorithms.stateprep.prepare_slater_determinant(
            q, orbital_indices, rotation_angles, 1)

    state = np.asarray(cudaq.get_state(kernel, indices, angles))
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected)


def test_prepare_random_real_slater_determinant_statevector():
    rng = np.random.default_rng(13)
    occupied_orbitals, _ = np.linalg.qr(rng.normal(size=(4, 2)))

    schedule = algorithms.stateprep.make_givens_rotation_schedule(
        occupied_orbitals)
    indices = algorithms.stateprep.get_givens_rotation_indices(schedule)
    angles = algorithms.stateprep.get_givens_rotation_angles(schedule)

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float]):
        q = cudaq.qvector(4)
        algorithms.stateprep.prepare_slater_determinant(
            q, orbital_indices, rotation_angles, 2)

    state = np.asarray(cudaq.get_state(kernel, indices, angles))
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_real_slater_determinant_sign_convention():
    theta = 0.52
    occupied_orbitals = np.array([[np.cos(theta), 0.0], [0.0, 1.0],
                                  [np.sin(theta), 0.0]])

    state = _prepare_real_slater_state(occupied_orbitals)
    expected = _reference_slater_state(occupied_orbitals)
    assert np.isclose(expected[0b011], np.cos(theta))
    assert np.isclose(expected[0b110], -np.sin(theta))
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_random_real_five_orbital_three_electron_statevector():
    rng = np.random.default_rng(19)
    occupied_orbitals, _ = np.linalg.qr(rng.normal(size=(5, 3)))

    state = _prepare_real_slater_state(occupied_orbitals)
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_complex_one_electron_slater_determinant_statevector():
    theta = 0.41
    phase = 0.73
    occupied_orbitals = np.array([[np.cos(theta)],
                                  [np.exp(1j * phase) * np.sin(theta)]])
    schedule = algorithms.stateprep.make_givens_rotation_schedule(
        occupied_orbitals)
    indices = algorithms.stateprep.get_givens_rotation_indices(schedule)
    angles = algorithms.stateprep.get_givens_rotation_angles(schedule)
    phases = algorithms.stateprep.get_givens_rotation_phases(schedule)

    assert np.isclose(schedule.rotations[0].theta, theta)
    assert np.isclose(schedule.rotations[0].phase, phase)

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float],
               rotation_phases: list[float], final_phases: list[float]):
        q = cudaq.qvector(2)
        algorithms.stateprep.prepare_complex_slater_determinant(
            q, orbital_indices, rotation_angles, rotation_phases, final_phases,
            1)

    state = np.asarray(
        cudaq.get_state(kernel, indices, angles, phases,
                        schedule.final_phases))
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_complex_slater_determinant_relative_phase_and_sign():
    theta = 0.52
    phase = 0.73
    occupied_orbitals = np.array([[np.cos(theta), 0.0], [0.0, 1.0],
                                  [np.exp(1j * phase) * np.sin(theta), 0.0]])

    state = _prepare_complex_slater_state(occupied_orbitals)
    expected = _reference_slater_state(occupied_orbitals)
    assert np.isclose(expected[0b011], np.cos(theta))
    assert np.isclose(expected[0b110], -np.exp(1j * phase) * np.sin(theta))
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_random_complex_slater_determinant_statevector():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    occupied_orbitals, _ = np.linalg.qr(raw)

    state = _prepare_complex_slater_state(occupied_orbitals)
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_random_complex_five_orbital_three_electron_statevector():
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    occupied_orbitals, _ = np.linalg.qr(raw)

    state = _prepare_complex_slater_state(occupied_orbitals)
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_slater_determinant_preserves_particle_number():
    rng = np.random.default_rng(7)
    occupied_orbitals, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    schedule = algorithms.stateprep.make_givens_rotation_schedule(
        occupied_orbitals.tolist())
    indices = algorithms.stateprep.get_givens_rotation_indices(schedule)
    angles = algorithms.stateprep.get_givens_rotation_angles(schedule)

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float]):
        q = cudaq.qvector(4)
        algorithms.stateprep.prepare_slater_determinant(
            q, orbital_indices, rotation_angles, 2)

    state = np.asarray(cudaq.get_state(kernel, indices, angles))
    probabilities = np.abs(state)**2

    for basis_index, probability in enumerate(probabilities):
        if probability < 1.0e-12:
            continue
        assert basis_index.bit_count() == 2


def test_slater_determinant_plan_real_statevector_and_resources():
    rng = np.random.default_rng(29)
    occupied_orbitals, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    plan = algorithms.stateprep.make_slater_determinant_plan(occupied_orbitals)

    assert plan.num_orbitals == 4
    assert plan.num_electrons == 2
    assert not plan.is_complex
    assert len(plan.orbital_indices) == 2 * len(plan.angles)
    assert len(plan.phases) == len(plan.angles)
    assert len(plan.final_phases) == plan.num_electrons

    resources = algorithms.stateprep.estimate_givens_stateprep_resources(plan)
    assert resources.num_givens_rotations == len(plan.angles)
    assert resources.num_exp_pauli_calls == 2 * len(plan.angles)
    assert resources.num_phase_rotations == 0
    assert resources.two_qubit_gate_count_proxy == resources.num_exp_pauli_calls
    assert resources.depth_proxy == resources.num_exp_pauli_calls

    num_orbitals = plan.num_orbitals
    num_electrons = plan.num_electrons

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float]):
        q = cudaq.qvector(num_orbitals)
        algorithms.stateprep.prepare_slater_determinant(
            q, orbital_indices, rotation_angles, num_electrons)

    state = np.asarray(
        cudaq.get_state(kernel, plan.orbital_indices, plan.angles))
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_slater_determinant_plan_complex_statevector_and_resources():
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    occupied_orbitals, _ = np.linalg.qr(raw)
    plan = algorithms.stateprep.make_slater_determinant_plan(occupied_orbitals)

    assert plan.num_orbitals == 5
    assert plan.num_electrons == 3
    assert plan.is_complex
    assert len(plan.orbital_indices) == 2 * len(plan.angles)
    assert len(plan.phases) == len(plan.angles)
    assert len(plan.final_phases) == plan.num_electrons

    resources = algorithms.stateprep.estimate_givens_stateprep_resources(plan)
    assert resources.num_givens_rotations == len(plan.angles)
    assert resources.num_exp_pauli_calls == 2 * len(plan.angles)
    assert resources.num_phase_rotations == len(
        plan.angles) + plan.num_electrons
    assert resources.two_qubit_gate_count_proxy == resources.num_exp_pauli_calls
    assert resources.depth_proxy == (resources.num_exp_pauli_calls +
                                     resources.num_phase_rotations)

    num_orbitals = plan.num_orbitals
    num_electrons = plan.num_electrons

    @cudaq.kernel
    def kernel(orbital_indices: list[int], rotation_angles: list[float],
               rotation_phases: list[float], final_phases: list[float]):
        q = cudaq.qvector(num_orbitals)
        algorithms.stateprep.prepare_complex_slater_determinant(
            q, orbital_indices, rotation_angles, rotation_phases, final_phases,
            num_electrons)

    state = np.asarray(
        cudaq.get_state(kernel, plan.orbital_indices, plan.angles, plan.phases,
                        plan.final_phases))
    expected = _reference_slater_state(occupied_orbitals)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_validate_slater_determinant_plan_rejects_non_adjacent_rotation():
    plan = algorithms.stateprep.SlaterDeterminantPlan()
    plan.num_orbitals = 3
    plan.num_electrons = 1
    plan.orbital_indices = [0, 2]
    plan.angles = [0.25]
    plan.phases = [0.0]
    plan.final_phases = [0.0]

    with pytest.raises(ValueError, match="adjacent rotations"):
        algorithms.stateprep.validate_slater_determinant_plan(plan)


def test_dispatch_complex_dtype_all_real_routes_complex():
    # A complex-dtype array whose values are all real must still route to the
    # complex path (dtype.kind == 'c'); the prepared state must be correct.
    rng = np.random.default_rng(101)
    real_q, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    occupied = real_q.astype(complex)
    plan = algorithms.stateprep.make_slater_determinant_plan(occupied)
    assert plan.is_complex

    state = _prepare_complex_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_dispatch_python_list_with_complex_entries_routes_complex():
    # A nested Python list with complex entries must route to the complex path
    # via the _contains_complex recursion (no numpy dtype to inspect).
    occupied = [[0.6 + 0.0j, 0.0 + 0.0j], [0.8 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 1.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
    plan = algorithms.stateprep.make_slater_determinant_plan(occupied)
    assert plan.is_complex

    state = _prepare_complex_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_dispatch_real_python_list_stays_real():
    occupied = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    plan = algorithms.stateprep.make_slater_determinant_plan(occupied)
    assert not plan.is_complex


def test_estimate_givens_rotation_schedule_resources():
    rng = np.random.default_rng(53)
    occupied, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    schedule = algorithms.stateprep.make_givens_rotation_schedule(occupied)
    resources = (
        algorithms.stateprep.estimate_givens_rotation_schedule_resources(
            schedule))
    assert resources.num_givens_rotations == len(schedule.rotations)
    assert resources.num_exp_pauli_calls == 2 * len(schedule.rotations)
    assert resources.num_phase_rotations == 0  # is_complex defaults to False


def test_localized_signed_basis_prepares_correct_determinant():
    # A localized, sign-flipped occupied orbital leaves a negative pivot in the
    # real reduction. The real kernel omits the 0/pi "final phase", but that is
    # only a global phase, so the prepared determinant is still correct up to
    # global phase (verified numerically across many inputs).
    occupied = [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]]
    state = _prepare_real_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)
