import numpy as np

import cudaq
import cudaq_algorithms as algorithms


def _reference_slater_state(occupied_orbitals):
    occupied_orbitals = np.asarray(occupied_orbitals, dtype=float)
    num_orbitals, num_electrons = occupied_orbitals.shape
    state = np.zeros(2**num_orbitals, dtype=complex)

    for basis_index in range(2**num_orbitals):
        occupied = [
            orbital for orbital in range(num_orbitals)
            if (basis_index >> orbital) & 1
        ]
        if len(occupied) == num_electrons:
            state[basis_index] = np.linalg.det(
                occupied_orbitals[np.ix_(occupied, range(num_electrons))])

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
