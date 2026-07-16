# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the Givens-rotation Slater determinant preparation.

Statevector tests check the prepared state against two independent
references: the dense minor expansion (amplitude of ``|S>`` is
``det(Q[S, :])``) and, for one spot-check, a second-quantized
construction that applies Jordan-Wigner creation-operator matrices to
the vacuum. Host-side tests pin the schedule construction (elimination
order, sign/phase conventions, real/complex dispatch), the validation
errors, and the resource estimates.
"""

import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms
from cudaq_algorithms import PauliLCU, stateprep
from cudaq_algorithms import sim_utils as sim

# ----------------------------------------------------------------------------
# Entry kernels
# ----------------------------------------------------------------------------


@cudaq.kernel
def _real_entry(num_spin_orbitals: int, orbital_indices: list[int],
                angles: list[float], num_electrons: int):
    q = cudaq.qvector(num_spin_orbitals)
    algorithms.stateprep.slater_determinant(q, orbital_indices, angles,
                                            num_electrons)


@cudaq.kernel
def _complex_entry(num_spin_orbitals: int, orbital_indices: list[int],
                   angles: list[float], phases: list[float],
                   final_phases: list[float], num_electrons: int):
    q = cudaq.qvector(num_spin_orbitals)
    algorithms.stateprep.complex_slater_determinant(q, orbital_indices, angles,
                                                    phases, final_phases,
                                                    num_electrons)


# ----------------------------------------------------------------------------
# References and helpers
# ----------------------------------------------------------------------------


def _reference_slater_state(orbital_coefficients):
    """Dense minor expansion: amplitude of |S> is det(Q[S, :])."""
    orbital_coefficients = np.asarray(orbital_coefficients, dtype=complex)
    num_spin_orbitals, num_electrons = orbital_coefficients.shape
    state = np.zeros(2**num_spin_orbitals, dtype=complex)

    for basis_index in range(2**num_spin_orbitals):
        occupied = [
            orbital for orbital in range(num_spin_orbitals)
            if (basis_index >> orbital) & 1
        ]
        if len(occupied) == num_electrons:
            state[basis_index] = np.linalg.det(orbital_coefficients[np.ix_(
                occupied, range(num_electrons))])

    return state


def _second_quantized_slater_state(orbital_coefficients):
    """Independent reference: JW creation-operator matrices on the vacuum.

    Builds ``b_k^dag = sum_p Q[p, k] c_p^dag`` from dense Jordan-Wigner
    creation operators (little-endian, Z string on the qubits below) and
    applies ``b_0^dag ... b_{ne-1}^dag`` to ``|0...0>``; no determinant
    identity is used anywhere.
    """
    matrix = np.asarray(orbital_coefficients, dtype=complex)
    num_spin_orbitals, num_electrons = matrix.shape
    dimension = 2**num_spin_orbitals

    def creation(p):
        operator = np.zeros((dimension, dimension), dtype=complex)
        for column in range(dimension):
            if not (column >> p) & 1:
                parity = (column & ((1 << p) - 1)).bit_count()
                operator[column | (1 << p), column] = (-1.0)**parity
        return operator

    state = np.zeros(dimension, dtype=complex)
    state[0] = 1.0
    for k in range(num_electrons - 1, -1, -1):
        mode = sum(matrix[p, k] * creation(p)
                   for p in range(num_spin_orbitals))
        state = mode @ state
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


def _prepare_real_slater_state(orbital_coefficients):
    orbital_coefficients = np.asarray(orbital_coefficients, dtype=float)
    num_spin_orbitals, num_electrons = orbital_coefficients.shape
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
    return np.asarray(
        cudaq.get_state(_real_entry, num_spin_orbitals,
                        stateprep.get_givens_rotation_indices(schedule),
                        stateprep.get_givens_rotation_angles(schedule),
                        num_electrons))


def _prepare_complex_slater_state(orbital_coefficients):
    orbital_coefficients = np.asarray(orbital_coefficients, dtype=complex)
    num_spin_orbitals, num_electrons = orbital_coefficients.shape
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
    return np.asarray(
        cudaq.get_state(_complex_entry, num_spin_orbitals,
                        stateprep.get_givens_rotation_indices(schedule),
                        stateprep.get_givens_rotation_angles(schedule),
                        stateprep.get_givens_rotation_phases(schedule),
                        list(schedule.final_phases), num_electrons))


# ----------------------------------------------------------------------------
# Statevector correctness
# ----------------------------------------------------------------------------


def test_givens_schedule_two_orbital_statevector():
    theta = 0.37
    orbital_coefficients = [[np.cos(theta)], [np.sin(theta)]]
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)

    assert schedule.num_spin_orbitals == 2
    assert schedule.num_electrons == 1
    assert len(schedule.rotations) == 1
    assert schedule.rotations[0].first_orbital == 0
    assert schedule.rotations[0].second_orbital == 1
    assert np.isclose(schedule.rotations[0].theta, theta)

    state = _prepare_real_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected)


def test_prepare_random_real_slater_determinant_statevector():
    rng = np.random.default_rng(13)
    orbital_coefficients, _ = np.linalg.qr(rng.normal(size=(4, 2)))

    state = _prepare_real_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_real_slater_determinant_sign_convention():
    theta = 0.52
    orbital_coefficients = np.array([[np.cos(theta), 0.0], [0.0, 1.0],
                                     [np.sin(theta), 0.0]])

    state = _prepare_real_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    assert np.isclose(expected[0b011], np.cos(theta))
    assert np.isclose(expected[0b110], -np.sin(theta))
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_random_real_five_orbital_three_electron_statevector():
    rng = np.random.default_rng(19)
    orbital_coefficients, _ = np.linalg.qr(rng.normal(size=(5, 3)))

    state = _prepare_real_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_complex_one_electron_slater_determinant_statevector():
    theta = 0.41
    phase = 0.73
    orbital_coefficients = np.array([[np.cos(theta)],
                                     [np.exp(1j * phase) * np.sin(theta)]])
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)

    assert np.isclose(schedule.rotations[0].theta, theta)
    assert np.isclose(schedule.rotations[0].phase, phase)

    state = _prepare_complex_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_complex_slater_determinant_relative_phase_and_sign():
    theta = 0.52
    phase = 0.73
    orbital_coefficients = np.array([[np.cos(theta), 0.0], [0.0, 1.0],
                                     [np.exp(1j * phase) * np.sin(theta),
                                      0.0]])

    state = _prepare_complex_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    assert np.isclose(expected[0b011], np.cos(theta))
    assert np.isclose(expected[0b110], -np.exp(1j * phase) * np.sin(theta))
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_random_complex_slater_determinant_statevector():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    orbital_coefficients, _ = np.linalg.qr(raw)

    state = _prepare_complex_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_prepare_random_complex_five_orbital_three_electron_statevector():
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    orbital_coefficients, _ = np.linalg.qr(raw)

    state = _prepare_complex_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_slater_determinant_preserves_particle_number():
    rng = np.random.default_rng(7)
    orbital_coefficients, _ = np.linalg.qr(rng.normal(size=(4, 2)))

    state = _prepare_real_slater_state(orbital_coefficients.tolist())
    probabilities = np.abs(state)**2

    for basis_index, probability in enumerate(probabilities):
        if probability < 1.0e-12:
            continue
        assert basis_index.bit_count() == 2


def test_second_quantized_reference_spot_check():
    # The two independent references must agree exactly, and the circuit
    # must match them up to global phase.
    rng = np.random.default_rng(37)
    raw = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    orbital_coefficients, _ = np.linalg.qr(raw)

    minors = _reference_slater_state(orbital_coefficients)
    second_quantized = _second_quantized_slater_state(orbital_coefficients)
    np.testing.assert_allclose(second_quantized, minors, atol=1.0e-12)

    state = _prepare_complex_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, second_quantized, atol=1.0e-6)


def test_localized_signed_basis_prepares_correct_determinant():
    # A localized, sign-flipped occupied orbital leaves a negative pivot in
    # the real reduction. The real kernel omits the 0/pi "final phase", but
    # that is only a global phase, so the prepared determinant is still
    # correct up to global phase.
    occupied = [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]]
    state = _prepare_real_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_complex_kernel_with_zero_phases_matches_real_kernel():
    # For a real schedule the complex kernel (all phases zero) must reduce
    # to the real kernel exactly.
    rng = np.random.default_rng(41)
    occupied, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    schedule = stateprep.make_givens_rotation_schedule(occupied)
    assert not schedule.is_complex

    indices = stateprep.get_givens_rotation_indices(schedule)
    angles = stateprep.get_givens_rotation_angles(schedule)
    phases = stateprep.get_givens_rotation_phases(schedule)
    assert phases == [0.0] * len(angles)
    assert list(schedule.final_phases) == [0.0] * schedule.num_electrons

    real_state = np.asarray(cudaq.get_state(_real_entry, 4, indices, angles,
                                            2))
    complex_state = np.asarray(
        cudaq.get_state(_complex_entry, 4, indices, angles, phases,
                        list(schedule.final_phases), 2))
    np.testing.assert_allclose(complex_state, real_state, atol=1.0e-12)


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


def test_basis_determinant_schedule_has_no_rotations():
    # Occupied orbitals already in the computational basis: zero rotations,
    # and the kernels must still prepare |...0011> from empty flattened
    # arrays.
    occupied = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    schedule = stateprep.make_givens_rotation_schedule(occupied)
    assert schedule.rotations == []
    assert stateprep.estimate_givens_resources(
        schedule).num_givens_rotations == 0

    state = np.asarray(cudaq.get_state(_real_entry, 4, [], [], 2))
    expected = np.zeros(16, dtype=complex)
    expected[0b0011] = 1.0
    np.testing.assert_allclose(state, expected, atol=1.0e-12)


def test_single_orbital_single_electron():
    schedule = stateprep.make_givens_rotation_schedule([[1.0]])
    assert schedule.num_spin_orbitals == 1
    assert schedule.num_electrons == 1
    assert schedule.rotations == []

    state = np.asarray(cudaq.get_state(_real_entry, 1, [], [], 1))
    np.testing.assert_allclose(state, [0.0, 1.0], atol=1.0e-12)


def test_full_filling_prepares_all_ones_determinant():
    # num_electrons == num_spin_orbitals: the only determinant is |1...1| with
    # amplitude det(Q), a pure global phase.
    rng = np.random.default_rng(43)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    occupied, _ = np.linalg.qr(raw)

    state = _prepare_complex_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    assert np.isclose(abs(expected[0b111]), 1.0)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------


def test_dispatch_complex_dtype_all_real_routes_complex():
    # A complex-dtype array whose values are all real must still route to the
    # complex path (dtype.kind == 'c'); the prepared state must be correct.
    rng = np.random.default_rng(101)
    real_q, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    occupied = real_q.astype(complex)
    schedule = stateprep.make_givens_rotation_schedule(occupied)
    assert schedule.is_complex

    state = _prepare_complex_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_dispatch_python_list_with_complex_entries_routes_complex():
    # A nested Python list with complex entries must route to the complex
    # path (no numpy dtype to inspect).
    occupied = [[0.6 + 0.0j, 0.0 + 0.0j], [0.8 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 1.0j], [0.0 + 0.0j, 0.0 + 0.0j]]
    schedule = stateprep.make_givens_rotation_schedule(occupied)
    assert schedule.is_complex

    state = _prepare_complex_slater_state(occupied)
    expected = _reference_slater_state(occupied)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_dispatch_real_python_list_stays_real():
    occupied = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    schedule = stateprep.make_givens_rotation_schedule(occupied)
    assert not schedule.is_complex


# ----------------------------------------------------------------------------
# Schedule shape, resources, and validation
# ----------------------------------------------------------------------------


def test_real_schedule_shape_and_resources():
    rng = np.random.default_rng(29)
    orbital_coefficients, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)

    assert schedule.num_spin_orbitals == 4
    assert schedule.num_electrons == 2
    assert not schedule.is_complex
    indices = stateprep.get_givens_rotation_indices(schedule)
    angles = stateprep.get_givens_rotation_angles(schedule)
    phases = stateprep.get_givens_rotation_phases(schedule)
    assert len(indices) == 2 * len(angles)
    assert len(phases) == len(angles)
    assert len(schedule.final_phases) == schedule.num_electrons

    resources = stateprep.estimate_givens_resources(schedule)
    assert resources.num_spin_orbitals == 4
    assert resources.num_electrons == 2
    assert resources.num_givens_rotations == len(angles)
    assert resources.num_exp_pauli_calls == 2 * len(angles)
    assert resources.num_phase_rotations == 0
    assert resources.two_qubit_gate_count_proxy == resources.num_exp_pauli_calls
    assert resources.depth_proxy == resources.num_exp_pauli_calls

    state = _prepare_real_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_complex_schedule_shape_and_resources():
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    orbital_coefficients, _ = np.linalg.qr(raw)
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)

    assert schedule.num_spin_orbitals == 5
    assert schedule.num_electrons == 3
    assert schedule.is_complex
    angles = stateprep.get_givens_rotation_angles(schedule)
    assert len(
        stateprep.get_givens_rotation_indices(schedule)) == 2 * len(angles)
    assert len(stateprep.get_givens_rotation_phases(schedule)) == len(angles)
    assert len(schedule.final_phases) == schedule.num_electrons

    resources = stateprep.estimate_givens_resources(schedule)
    assert resources.num_spin_orbitals == 5
    assert resources.num_electrons == 3
    assert resources.num_givens_rotations == len(angles)
    assert resources.num_exp_pauli_calls == 2 * len(angles)
    assert resources.num_phase_rotations == (len(angles) +
                                             schedule.num_electrons)
    assert resources.two_qubit_gate_count_proxy == resources.num_exp_pauli_calls
    assert resources.depth_proxy == (resources.num_exp_pauli_calls +
                                     resources.num_phase_rotations)

    state = _prepare_complex_slater_state(orbital_coefficients)
    expected = _reference_slater_state(orbital_coefficients)
    _assert_allclose_up_to_global_phase(state, expected, atol=1.0e-6)


def test_validate_schedule_rejects_non_adjacent_rotation():
    schedule = stateprep.GivensRotationSchedule(
        num_spin_orbitals=3,
        num_electrons=1,
        rotations=[stateprep.GivensRotation(0, 2, 0.25)],
        final_phases=[0.0])

    with pytest.raises(ValueError, match="adjacent rotations"):
        stateprep.validate_givens_rotation_schedule(schedule)


def test_validate_schedule_rejects_invalid_counts_and_indices():
    with pytest.raises(ValueError, match="num_spin_orbitals"):
        stateprep.validate_givens_rotation_schedule(
            stateprep.GivensRotationSchedule(num_spin_orbitals=0,
                                             num_electrons=1))
    with pytest.raises(ValueError, match="num_electrons must"):
        stateprep.validate_givens_rotation_schedule(
            stateprep.GivensRotationSchedule(num_spin_orbitals=2,
                                             num_electrons=0))
    with pytest.raises(ValueError, match="cannot exceed"):
        stateprep.validate_givens_rotation_schedule(
            stateprep.GivensRotationSchedule(num_spin_orbitals=1,
                                             num_electrons=2))
    with pytest.raises(ValueError, match="out of range"):
        stateprep.validate_givens_rotation_schedule(
            stateprep.GivensRotationSchedule(
                num_spin_orbitals=2,
                num_electrons=1,
                rotations=[stateprep.GivensRotation(1, 2, 0.1)]))
    with pytest.raises(ValueError, match="final phase per electron"):
        stateprep.validate_givens_rotation_schedule(
            stateprep.GivensRotationSchedule(num_spin_orbitals=2,
                                             num_electrons=1,
                                             is_complex=True,
                                             final_phases=[]))
    with pytest.raises(ValueError, match="empty or"):
        stateprep.validate_givens_rotation_schedule(
            stateprep.GivensRotationSchedule(num_spin_orbitals=2,
                                             num_electrons=1,
                                             final_phases=[0.0, 0.0]))


def test_orbital_coefficient_validation_errors():
    with pytest.raises(ValueError,
                       match="orbital_coefficients must not be "
                       "empty"):
        stateprep.make_givens_rotation_schedule([])
    with pytest.raises(ValueError, match="at least one"):
        stateprep.make_givens_rotation_schedule([[], []])
    with pytest.raises(ValueError, match="cannot exceed"):
        stateprep.make_givens_rotation_schedule([[1.0, 0.0]])
    with pytest.raises(ValueError, match="rectangular"):
        stateprep.make_givens_rotation_schedule([[1.0, 0.0], [0.0]])
    with pytest.raises(ValueError, match="normalized"):
        stateprep.make_givens_rotation_schedule([[0.5], [0.5]])
    with pytest.raises(ValueError, match="orthogonal"):
        s = 1.0 / np.sqrt(2.0)
        stateprep.make_givens_rotation_schedule([[1.0, s], [0.0, s]])


# ----------------------------------------------------------------------------
# slater_determinant_kernel factory
# ----------------------------------------------------------------------------


def _prepared_state(num_qubits, state_prep):
    """Run a (qubits: qview) state-prep kernel on a fresh register."""

    @cudaq.kernel
    def entry():
        q = cudaq.qvector(num_qubits)
        state_prep(q)

    return np.asarray(cudaq.get_state(entry))


def test_factory_matches_real_kernel_path():
    rng = np.random.default_rng(53)
    orbital_coefficients, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
    assert not schedule.is_complex

    prep = stateprep.slater_determinant_kernel(schedule)
    direct = _prepare_real_slater_state(orbital_coefficients)
    np.testing.assert_allclose(_prepared_state(schedule.num_spin_orbitals,
                                               prep),
                               direct,
                               atol=1.0e-12)


def test_factory_matches_complex_kernel_path():
    rng = np.random.default_rng(59)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    orbital_coefficients, _ = np.linalg.qr(raw)
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
    assert schedule.is_complex

    prep = stateprep.slater_determinant_kernel(schedule)
    direct = _prepare_complex_slater_state(orbital_coefficients)
    np.testing.assert_allclose(_prepared_state(schedule.num_spin_orbitals,
                                               prep),
                               direct,
                               atol=1.0e-12)


def test_factory_rotation_free_shapes():
    # Basis determinants build rotation-free schedules; the factory must
    # avoid the empty-list-capture shapes and still prepare |...0011>
    # (real) and its final-phased twin (complex).
    real_schedule = stateprep.make_givens_rotation_schedule([[1.0, 0.0],
                                                             [0.0, 1.0],
                                                             [0.0, 0.0],
                                                             [0.0, 0.0]])
    assert real_schedule.rotations == []
    expected = np.zeros(16, dtype=complex)
    expected[0b0011] = 1.0
    real_prep = stateprep.slater_determinant_kernel(real_schedule)
    np.testing.assert_allclose(_prepared_state(4, real_prep),
                               expected,
                               atol=1.0e-12)

    matrix = np.array([[1.0, 0.0], [0.0, 1.0j], [0.0, 0.0], [0.0, 0.0]])
    complex_schedule = stateprep.make_givens_rotation_schedule(matrix)
    assert complex_schedule.is_complex
    assert complex_schedule.rotations == []
    complex_prep = stateprep.slater_determinant_kernel(complex_schedule)
    np.testing.assert_allclose(_prepared_state(4, complex_prep),
                               _prepare_complex_slater_state(matrix),
                               atol=1.0e-12)


def test_factory_injects_as_state_prep_kernel():
    # Live injection: the factory kernel drives PauliLCU.encode_kernel
    # and must match the State-taking twin fed the identical prepared
    # ket as data (the test_state_prep_injection pattern).
    theta = 0.37
    schedule = stateprep.make_givens_rotation_schedule([[np.cos(theta)],
                                                        [np.sin(theta)]])
    prep = stateprep.slater_determinant_kernel(schedule)

    enc = PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
    assert enc.num_system == schedule.num_spin_orbitals

    ket = _prepared_state(schedule.num_spin_orbitals, prep)
    via_prep = np.asarray(cudaq.get_state(enc.encode_kernel(state_prep=prep)))
    via_state = np.asarray(
        cudaq.get_state(enc.encode_kernel(), sim.state_from(ket)))
    np.testing.assert_allclose(via_prep, via_state, atol=1.0e-12)
