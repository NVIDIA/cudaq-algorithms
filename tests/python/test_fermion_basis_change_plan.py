# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent orbital-matrix oracles for the immutable Givens planner."""

import copy
import decimal
import itertools
import pickle
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

from cudaq_algorithms.basis_change._encoding import AngleEncoding
from cudaq_algorithms.basis_change._metrics import \
    estimate_fermion_basis_change_error
from cudaq_algorithms.basis_change._plan import (GivensBasisChangeSchedule,
                                                 GivensOperation,
                                                 GivensStructure,
                                                 compile_givens_basis_change)

INVALID_TOLERANCES = [
    0, -1, np.nan, np.inf, 1j, "bad", True, np.True_, "1e-3", b"1e-3",
    np.complex128(1e-3), 1., 10., 10**400,
    np.array(True),
    np.array(1e-3j),
    np.array([1e-3])
]


def relocated_block(seed):
    """Real orthogonal ``[[0, Q], [1, 0]]``.

    Its transpose ``roll(I, 1, 0) @ blockdiag(Q.T, 1)`` relocates one mode.
    """
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = np.zeros((4, 4), dtype=complex)
    matrix[:3, 1:] = q * np.sign(np.diag(r))
    matrix[3, 0] = 1.
    return matrix


def phase_slots(schedule):
    """Number-phase and relative-phase values, without mixing angles."""
    return [
        value
        for name, value in zip(schedule.parameter_names, schedule.parameters)
        if not name.startswith("theta")
    ]


def codes(schedule, width):
    encoding = AngleEncoding(width)
    return tuple(encoding.encode(value) for value in schedule.parameters)


def tape_matrix(schedule, parameters=None):
    """Dense gate multiplication independent of planner reconstruction."""
    values = schedule.parameters if parameters is None else parameters
    result = np.eye(schedule.structure.num_modes, dtype=complex)
    for op in schedule.operations:
        gate = np.eye(len(result), dtype=complex)
        angle = values[op.parameter_index] * op.scale
        if op.kind == "phase":
            gate[op.first, op.first] = np.exp(1j * angle)
        else:
            a, b = op.first, op.second
            gate[np.ix_([a, b], [a, b])] = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        result = gate @ result
    return result


@pytest.mark.parametrize("m", [2, 3, 5, 9])
@pytest.mark.parametrize("complex_input", [False, True])
def test_full_random_unitary_keeps_every_column_phase(m, complex_input):
    rng = np.random.default_rng(730 + m)
    raw = rng.normal(size=(m, m))
    if complex_input:
        raw = raw + 1j * rng.normal(size=(m, m))
    unitary, _ = np.linalg.qr(raw)
    schedule = compile_givens_basis_change(unitary)
    np.testing.assert_allclose(tape_matrix(schedule), unitary, atol=2e-14)
    np.testing.assert_allclose(schedule.orbital_matrix(), unitary, atol=2e-14)
    assert schedule.reconstruction_error < 2e-14
    assert schedule.num_parameters == m * m


@pytest.mark.parametrize("unitary", [
    np.diag([-1., 1.]),
    np.array([[0., -1.], [1., 0.]]),
    np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]]),
    np.diag([1j, -1., -1j]),
    np.diag([-1., -1.])
])
def test_signed_permutations_and_diagonal_phases_are_exact(unitary):
    schedule = compile_givens_basis_change(unitary)
    np.testing.assert_allclose(tape_matrix(schedule), unitary, atol=1e-14)


@pytest.mark.parametrize("seed", range(3))
def test_value_equal_real_inputs_compile_to_identical_parameters(seed):
    # U.T and U.conj().T differ only in zero sign bits. Gauge phases once came
    # from rounding noise, so equal values gave different tables.
    matrix = relocated_block(seed)
    first = compile_givens_basis_change(matrix.T)
    second = compile_givens_basis_change(matrix.conj().T)
    assert np.array_equal(matrix.T, matrix.conj().T)
    assert first.parameters == second.parameters
    for width in range(1, 17):
        assert codes(first, width) == codes(second, width)
    np.testing.assert_allclose(tape_matrix(first), matrix.T, atol=1e-14)
    # Real pivots use exact +-1 phase factors, so no slot carries residue.
    assert set(phase_slots(first)) <= {0., np.pi, -np.pi}


@pytest.mark.parametrize("m", [3, 4])
def test_signed_permutations_keep_phase_slots_on_multiples_of_pi(m):
    for order in itertools.permutations(range(m)):
        for signs in itertools.product([1., -1.], repeat=m):
            matrix = np.eye(m)[list(order)] * signs
            schedule = compile_givens_basis_change(matrix)
            np.testing.assert_allclose(tape_matrix(schedule),
                                       matrix,
                                       atol=1e-14)
            assert set(phase_slots(schedule)) <= {0., np.pi, -np.pi}


@pytest.mark.parametrize("seed", range(2))
def test_one_ulp_entry_change_keeps_width_eight_codes(seed):
    matrix = relocated_block(seed).T.copy()
    expected = codes(compile_givens_basis_change(matrix), 8)
    for row, col in np.argwhere(matrix != 0):
        bumped = matrix.copy()
        bumped[row, col] = np.nextafter(matrix[row, col].real, np.inf)
        assert codes(compile_givens_basis_change(bumped), 8) == expected


@pytest.mark.parametrize(
    "tiny",
    [1e-310, -1e-310,
     complex(6e-311, 8e-311),
     complex(3e-322, 4e-322)])
def test_subnormal_entries_compile_accurately(tiny):
    # A unit phase computed as z / abs(z) overflows for these entries.
    matrix = np.diag(np.exp([0., .3j, -1.1j]))
    matrix[2, 0] = tiny
    schedule = compile_givens_basis_change(matrix)
    assert schedule.reconstruction_error < 1e-15
    np.testing.assert_allclose(tape_matrix(schedule), matrix, atol=1e-15)


@pytest.mark.parametrize("tiny", [5e-324, 1e-323, 1e-321, 1e-315])
@pytest.mark.parametrize("complex_entry", [False, True])
def test_two_subnormal_pivot_magnitudes_keep_rotations_normalised(
        tiny, complex_entry):
    # Both magnitudes of one pivot pair are subnormal; an unscaled hypot
    # would round so that c = s = 1, and the row update would not rotate.
    rng = np.random.default_rng(3)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = np.zeros((4, 4), dtype=complex)
    matrix[0, 0] = 1.
    matrix[1:, 1:] = q * np.sign(np.diag(r))
    matrix[2, 0] = 1j * tiny if complex_entry else tiny
    matrix[3, 0] = tiny
    schedule = compile_givens_basis_change(matrix)
    assert schedule.reconstruction_error < 1e-15
    np.testing.assert_allclose(tape_matrix(schedule), matrix, atol=1e-15)


@pytest.mark.parametrize("column", [
    [0., -1.],
    [1j, 0.],
    [-1., 0., 0.],
    [0., 0., -1.],
    [0., 1j, 0.],
    [.5, -.5, .5j, -.5j],
    [1e-15j, np.sqrt(.5), -np.sqrt(.5), 0.],
])
def test_partial_preserves_column_phase_for_every_reference(column):
    column = np.asarray(column, dtype=complex)
    for reference in range(len(column)):
        structure = GivensStructure.partial(len(column),
                                            reference_mode=reference)
        for basis in (column, column[:, None]):
            schedule = compile_givens_basis_change(basis, structure=structure)
            matrix = tape_matrix(schedule)
            np.testing.assert_allclose(matrix[:, reference],
                                       column,
                                       atol=2e-14)
            np.testing.assert_allclose(matrix.conj().T @ matrix,
                                       np.eye(len(column)),
                                       atol=2e-14)
            np.testing.assert_allclose(schedule.orbital_matrix(),
                                       matrix,
                                       atol=2e-14)
            assert schedule.num_parameters == 2 * len(column) - 1
            assert schedule.reconstruction_error < 2e-14


@pytest.mark.parametrize("m", [2, 5, 8])
def test_random_partial_with_nondefault_reference(m):
    rng = np.random.default_rng(100 + m)
    vector = rng.normal(size=m) + 1j * rng.normal(size=m)
    vector /= np.linalg.norm(vector)
    for reference in range(m):
        schedule = compile_givens_basis_change(
            vector,
            structure=GivensStructure.partial(m, reference_mode=reference))
        np.testing.assert_allclose(tape_matrix(schedule)[:, reference],
                                   vector,
                                   atol=2e-14)


def test_shape_inference_and_logical_mode_map_order():
    vector = np.array([0., 1j, 0.])
    for basis in (vector, vector[:, None]):
        assert compile_givens_basis_change(basis).structure.kind == "partial"
    assert compile_givens_basis_change(np.eye(3)).structure.kind == "full"
    mapping = [7, 1, 4]
    structure = GivensStructure.partial(3, mode_map=mapping)
    mapping[0] = 99
    assert structure.mode_map == (7, 1, 4)
    assert structure.num_system_qubits == 8
    schedule = compile_givens_basis_change(vector, structure=structure)
    np.testing.assert_allclose(tape_matrix(schedule)[:, 0], vector, atol=1e-14)


@pytest.mark.parametrize("kind", ["full", "partial"])
def test_zero_slots_remain_programmable_without_basis(kind):
    structure = getattr(GivensStructure, kind)(3)
    basis = np.eye(3) if kind == "full" else np.array([1., 0., 0.])
    bound = compile_givens_basis_change(basis, structure=structure)
    unbound = compile_givens_basis_change(structure=structure)
    assert unbound.parameters is None
    assert unbound.reconstruction_error is None
    assert bound.operations == unbound.operations
    assert len(set(bound.parameter_names)) == bound.num_parameters
    assert len(structure.mode_pairs) == (3 if kind == "full" else 2)
    assert len(bound.operations) == (12 if kind == "full" else 5)
    with pytest.raises(ValueError):
        unbound.orbital_matrix()
    for slot in range(bound.num_parameters):
        parameters = np.zeros(bound.num_parameters)
        parameters[slot] = .3
        # A relative phase matters when its corresponding mixing angle is nonzero.
        if kind == "full" and "phi" in bound.parameter_names[slot]:
            parameters[slot - 1] = .4
        actual = unbound.orbital_matrix(parameters)
        np.testing.assert_allclose(actual,
                                   tape_matrix(unbound, parameters),
                                   atol=1e-14)
        assert not np.allclose(actual, np.eye(3))
    assert bound.parameters == tuple(bound.parameters)
    with pytest.raises(FrozenInstanceError):
        bound.parameters = ()
    with pytest.raises(FrozenInstanceError):
        structure.num_modes = 4
    with pytest.raises(FrozenInstanceError):
        bound.operations[0].scale = 2
    matrix = bound.orbital_matrix()
    matrix[:] = 99
    np.testing.assert_allclose(bound.orbital_matrix(), np.eye(3), atol=1e-14)


@pytest.mark.parametrize(
    "basis", [[], [[]], 1., [0., 0.], [2., 0.], [[1., 1.], [0., 0.]],
              np.ones((3, 2)),
              np.ones((1, 2)),
              np.ones((2, 2, 1)), [np.nan], [np.inf], [[1., 0.], [0., 1.01]],
              [[1., 0.], [1.]], [1., 1e-3]])
def test_rejects_invalid_basis_without_projection(basis):
    with pytest.raises(ValueError):
        compile_givens_basis_change(basis)


@pytest.mark.parametrize(
    "basis", [[np.nan, 0.], [0., np.inf], [[1., 0.], [0., np.nan]]])
def test_rejects_nonfinite_basis(basis):
    # Two modes, so the one-mode rule cannot reject these first.
    with pytest.raises(ValueError, match="basis must be a finite"):
        compile_givens_basis_change(basis)


@pytest.mark.parametrize("tolerance", INVALID_TOLERANCES)
def test_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        compile_givens_basis_change(np.eye(2), tolerance=tolerance)


def test_reconstruction_error_is_the_maximum_entry_residual():
    # Accepted inexact input compiles to the unitary its angles define.
    full_input = (1 + 1e-7) * np.eye(3)
    full = compile_givens_basis_change(full_input, tolerance=1e-5)
    residual = np.max(np.abs(tape_matrix(full) - full_input))
    assert full.reconstruction_error == pytest.approx(residual, rel=1e-6)
    assert full.reconstruction_error == pytest.approx(1e-7, rel=1e-6)
    column = (1 + 1e-7) * np.array([1., 1j, -1.]) / np.sqrt(3)
    partial = compile_givens_basis_change(column, tolerance=1e-5)
    residual = np.max(np.abs(tape_matrix(partial)[:, 0] - column))
    assert partial.reconstruction_error == pytest.approx(residual, rel=1e-6)
    assert partial.reconstruction_error == pytest.approx(1e-7 / np.sqrt(3),
                                                         rel=1e-6)


def test_unitarity_tolerance_bounds_the_maximum_gram_entry():
    # Each diagonal Gram entry is off by (1+e)**2 - 1; a Frobenius norm of
    # U†U-I would double that for four modes.
    compile_givens_basis_change((1 + 4.5e-7) * np.eye(4), tolerance=1e-6)
    with pytest.raises(ValueError, match="unitary"):
        compile_givens_basis_change((1 + 5.5e-7) * np.eye(4), tolerance=1e-6)


@pytest.mark.parametrize("m", [0, 1, -1, 2.5, True])
def test_rejects_invalid_mode_count(m):
    with pytest.raises(ValueError):
        GivensStructure.full(m)
    with pytest.raises(ValueError):
        GivensStructure.partial(m)


@pytest.mark.parametrize("basis", [[1.], [[1.]], [[np.exp(.4j)]], [-1j]])
def test_rejects_single_mode_basis(basis):
    # One mode admits only a number phase; callers apply r1 directly.
    with pytest.raises(ValueError, match="at least 2"):
        compile_givens_basis_change(basis)


@pytest.mark.parametrize("mapping",
                         [[0], [0, 0], [-1, 2], [0., 1], [True, 1], {0, 1},
                          frozenset({0, 1}),
                          dict.fromkeys((0, 1)), (0, 2**31), "01", b"01",
                          bytearray([0, 1]), 1,
                          iter((0, 1)),
                          iter({0, 1}),
                          dict.fromkeys((0, 1)).keys(),
                          (index for index in (0, 1))])
def test_rejects_invalid_mode_map(mapping):
    # Only ordered sequences qualify: iterators and views may follow hash
    # order, and bytes are not index sequences.
    with pytest.raises(ValueError):
        GivensStructure.full(2, mode_map=mapping)


def test_mode_map_accepts_ordered_numpy_indices_up_to_the_limit():
    structure = GivensStructure.full(2, mode_map=np.array([3, 1]))
    assert structure.mode_map == (3, 1)
    assert all(type(index) is int for index in structure.mode_map)
    widest = GivensStructure.partial(2, mode_map=(np.uint32(2**31 - 1), 0))
    assert widest.num_system_qubits == 2**31


@pytest.mark.parametrize("reference", [-1, 2, .5, True])
def test_rejects_invalid_reference_mode(reference):
    with pytest.raises(ValueError):
        GivensStructure.partial(2, reference_mode=reference)


def test_rejects_inconsistent_structure_or_missing_basis():
    for basis, structure in [
        (np.eye(2), GivensStructure.full(3)),
        ([1., 0.], GivensStructure.full(2)),
        (np.eye(2), GivensStructure.partial(2)),
        (np.eye(2), "full"),
    ]:
        with pytest.raises(ValueError):
            compile_givens_basis_change(basis, structure=structure)
    with pytest.raises(ValueError):
        compile_givens_basis_change()


@pytest.mark.parametrize(
    "parameters",
    [[], [0.] * 3, [0.] * 5, [0., 0., 0., np.nan], [0., 0., 0., np.inf],
     [0., 0., 0., 1j], [[0., 0.], [0., 0.]]])
def test_rejects_invalid_parameter_override(parameters):
    schedule = compile_givens_basis_change(np.eye(2))
    with pytest.raises(ValueError):
        schedule.orbital_matrix(parameters)


@pytest.mark.parametrize(
    "parameters",
    [("1.5", "0", "0"), (b"1", b"0", b"0"), (True, False, True),
     np.array([True, False, True]), (.1, True, .3), [0, np.True_, 0],
     np.array([0., True, 0.], dtype=object), [np.array(True), 0., 0.],
     [0., 0., np.array(1j)], [0., "0", 0.],
     bytearray([1, 0, 0])])
def test_rejects_non_numeric_parameter_vectors(parameters):
    # Text, bytes and bool entries are not real angles, even when float()
    # could coerce them or numpy would upcast them in a mixed list.
    schedule = compile_givens_basis_change([1., 0.])
    with pytest.raises(ValueError):
        GivensBasisChangeSchedule(schedule.structure, parameters)
    with pytest.raises(ValueError):
        schedule.orbital_matrix(parameters)
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_error(schedule, parameters)


@pytest.mark.parametrize("parameters", [(1, 0, 2),
                                        np.array([1, 0, 2]),
                                        np.array([1, 0, 2], dtype=np.uint8)])
def test_integer_parameter_vectors_are_real_angles(parameters):
    schedule = compile_givens_basis_change([1., 0.])
    bound = GivensBasisChangeSchedule(schedule.structure, parameters)
    assert bound.parameters == (1., 0., 2.)
    assert all(type(value) is float for value in bound.parameters)
    np.testing.assert_allclose(schedule.orbital_matrix(parameters),
                               tape_matrix(bound),
                               atol=1e-14)
    report = estimate_fermion_basis_change_error(schedule, parameters)
    assert len(report.parameter_deltas) == 3


@pytest.mark.parametrize("parameters", [
    (Fraction(1, 2), 0, Fraction(-3, 4)),
    np.array([.5, 0., -.75], dtype=object),
    [np.float32(.5), decimal.Decimal(0),
     np.array(-.75)],
])
def test_real_scalar_parameter_entries_are_converted(parameters):
    # Entries convert one by one, so object arrays and Fraction, Decimal or
    # zero-dimensional array entries are real angles.
    schedule = compile_givens_basis_change([1., 0.])
    bound = GivensBasisChangeSchedule(schedule.structure, parameters)
    assert bound.parameters == (.5, 0., -.75)
    assert all(type(value) is float for value in bound.parameters)
    np.testing.assert_allclose(schedule.orbital_matrix(parameters),
                               tape_matrix(bound),
                               atol=1e-14)
    report = estimate_fermion_basis_change_error(schedule, parameters)
    assert report.parameter_deltas == pytest.approx((.5, 0., -.75))


def test_integer_entries_beyond_int64_must_fit_a_double():
    structure = GivensStructure.partial(2)
    wide = GivensBasisChangeSchedule(structure, [2**64, 0, -2**70])
    assert wide.parameters == (2.**64, 0., -2.**70)
    with pytest.raises(ValueError, match="finite"):
        GivensBasisChangeSchedule(structure, [10**400, 0, 0])


def test_finite_input_whose_gram_matrix_overflows_is_rejected():
    # NaN comparisons must not accidentally accept an unnormalized matrix.
    with pytest.raises(ValueError):
        compile_givens_basis_change(np.full((2, 2), complex(1e308, 1e308)))


@pytest.mark.parametrize(
    "zero",
    [complex(0., 0.),
     complex(-0., 0.),
     complex(0., -0.),
     complex(-0., -0.)])
def test_signed_zero_amplitudes_produce_canonical_parameter_tables(zero):
    # The phase of an absent amplitude must not depend on its zero sign bits,
    # and a real negative amplitude has phase pi whatever its zero imaginary
    # sign. No parameter is -0.0, for any reference mode.
    for amplitude, reference in itertools.product([1., -1.], range(3)):
        column = np.full(3, zero, dtype=complex)
        column[reference] = complex(amplitude, zero.imag)
        canonical_column = np.zeros(3)
        canonical_column[reference] = amplitude
        structure = GivensStructure.partial(3, reference_mode=reference)
        partial = compile_givens_basis_change(column, structure=structure)
        canonical = compile_givens_basis_change(canonical_column,
                                                structure=structure)
        assert partial.parameters == canonical.parameters
        assert not any(
            np.signbit(p)
            for p in partial.parameters + canonical.parameters if p == 0.)
    for amplitude in (1., -1.):
        full_input = np.full((3, 3), zero, dtype=complex)
        np.fill_diagonal(full_input, complex(amplitude, zero.imag))
        full = compile_givens_basis_change(full_input)
        canonical = compile_givens_basis_change(np.diag([amplitude] * 3))
        assert full.parameters == canonical.parameters
        assert not any(np.signbit(p) for p in full.parameters if p == 0.)


def test_tolerance_validation_is_shared_with_factory():
    from cudaq_algorithms.basis_change._plan import _validate_tolerance
    for value in (np.float64(1e-9), np.array(1e-9), decimal.Decimal("1e-9")):
        assert _validate_tolerance(value) == 1e-9
        assert type(_validate_tolerance(value)) is float
    for value in INVALID_TOLERANCES:
        with pytest.raises(ValueError, match="tolerance"):
            _validate_tolerance(value)
        with pytest.raises(ValueError, match="tolerance"):
            compile_givens_basis_change(structure=GivensStructure.full(2),
                                        tolerance=value)


@pytest.mark.parametrize("zero", [complex(-0., 0.), complex(-0., -0.)])
def test_full_elimination_canonicalizes_zero_upper_pivot_phase(zero):
    matrix = np.array([[zero, -1.], [1., 0.]])
    canonical = compile_givens_basis_change([[0., -1.], [1., 0.]])
    schedule = compile_givens_basis_change(matrix)
    assert schedule.parameters == canonical.parameters
    np.testing.assert_allclose(tape_matrix(schedule), matrix, atol=1e-14)


@pytest.mark.parametrize(
    "error",
    ["nope", np.nan, -1., np.inf, True, 1j, [0.], 10**400,
     np.array(True)])
def test_rejects_invalid_reconstruction_error(error):
    compiled = compile_givens_basis_change(np.eye(2))
    with pytest.raises(ValueError, match="reconstruction_error"):
        GivensBasisChangeSchedule(compiled.structure, compiled.parameters,
                                  error)


def test_structure_only_schedule_rejects_a_reconstruction_error():
    for error in (.25, 0.):
        with pytest.raises(ValueError, match="reconstruction_error"):
            GivensBasisChangeSchedule(GivensStructure.partial(2), None, error)
    # The documented structure-only form of a compiled schedule.
    compiled = compile_givens_basis_change([.6, .8])
    structure_only = GivensBasisChangeSchedule(compiled.structure)
    assert structure_only.parameters is None
    assert structure_only.reconstruction_error is None
    with pytest.raises(ValueError, match="reconstruction_error"):
        replace(compiled, parameters=None)
    assert replace(compiled, parameters=None,
                   reconstruction_error=None) == structure_only


def test_pickles_and_copies_rebuild_through_validation():
    structure = GivensStructure.partial(3,
                                        mode_map=(4, 0, 2),
                                        reference_mode=1)
    schedule = compile_givens_basis_change(np.array([.6, 0., .8j]),
                                           structure=structure)
    for value in (structure, schedule, GivensBasisChangeSchedule(structure)):
        for clone in (pickle.loads(pickle.dumps(value)), copy.copy(value),
                      copy.deepcopy(value)):
            assert clone == value
    restored = pickle.loads(pickle.dumps(schedule))
    assert restored.reconstruction_error == schedule.reconstruction_error
    # Loading runs __post_init__, so tampered state is rejected, not restored.
    crafted = GivensStructure.full(2)
    object.__setattr__(crafted, "num_modes", 1)
    object.__setattr__(crafted, "mode_map", (0, ))
    with pytest.raises(ValueError, match="at least 2"):
        pickle.loads(pickle.dumps(crafted))
    tampered = compile_givens_basis_change(np.eye(2))
    object.__setattr__(tampered, "parameters", (0., 0., 0.))
    with pytest.raises(ValueError, match="parameters"):
        pickle.loads(pickle.dumps(tampered))


def test_equality_and_hash_ignore_the_reconstruction_diagnostic():
    from cudaq_algorithms.basis_change import fermion_basis_change
    rng = np.random.default_rng(17)
    unitary, _ = np.linalg.qr(
        rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    compiled = compile_givens_basis_change(unitary)
    hand_built = GivensBasisChangeSchedule(compiled.structure,
                                           compiled.parameters)
    annotated = GivensBasisChangeSchedule(compiled.structure,
                                          compiled.parameters, np.float64(.5))
    assert hand_built.reconstruction_error is None
    assert type(annotated.reconstruction_error) is float
    assert hand_built == compiled == annotated
    assert hash(hand_built) == hash(compiled) == hash(annotated)
    assert compiled != GivensBasisChangeSchedule(compiled.structure)
    assert (fermion_basis_change(hand_built) == fermion_basis_change(compiled)
            == fermion_basis_change(unitary))


def test_operation_records_are_public_and_use_logical_modes():
    import cudaq_algorithms.basis_change as basis_change
    assert basis_change.GivensOperation is GivensOperation
    assert "GivensOperation" in basis_change.__all__
    structure = GivensStructure.full(3, mode_map=(4, 0, 2))
    operations = compile_givens_basis_change(structure=structure).operations
    assert all(isinstance(op, GivensOperation) for op in operations)
    # Records hold logical modes; physical qubit 4 never appears.
    assert {op.first for op in operations} == {0, 1, 2}
    assert all(op.second is None for op in operations if op.kind == "phase")
    # Each complex mixing block is P_b(-phi) G(theta) P_b(phi); scale -1
    # marks only the leading phase.
    blocks = [operations[i:i + 3] for i in range(3, len(operations), 3)]
    for (lead, mix, tail), (a, b) in zip(blocks, structure.mode_pairs):
        assert (mix.kind, mix.first, mix.second) == ("givens", a, b)
        assert (lead.kind, lead.first, lead.scale) == ("phase", b, -1.)
        assert (tail.kind, tail.first, tail.scale) == ("phase", b, 1.)
        assert {lead.parameter_index,
                tail.parameter_index} == {mix.parameter_index + 1}
    assert [op.scale for op in operations].count(-1.) == len(blocks)


@pytest.mark.parametrize("owner", [GivensStructure, GivensBasisChangeSchedule])
def test_public_properties_have_docstrings(owner):
    # The API reference documents only members that carry a docstring.
    names = [
        name for name, member in vars(owner).items()
        if isinstance(member, property) and not name.startswith("_")
    ]
    assert names
    assert [name for name in names if not getattr(owner, name).__doc__] == []
