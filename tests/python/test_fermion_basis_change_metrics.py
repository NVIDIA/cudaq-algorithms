# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent matrix-norm and static-tape checks of Givens estimates."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from itertools import combinations

import cudaq
import numpy as np
import pytest

from cudaq_algorithms.basis_change._factory import fermion_basis_change
from cudaq_algorithms.basis_change._plan import (GivensStructure,
                                                 compile_givens_basis_change)
from cudaq_algorithms.basis_change._metrics import (
    estimate_fermion_basis_change_error,
    estimate_fermion_basis_change_resources)


def fock_lift(orbital_matrix):
    """Independent exterior-power oracle, including vacuum and filled state."""
    m = len(orbital_matrix)
    result = np.zeros((2**m, 2**m), dtype=complex)
    result[0, 0] = 1.
    for number in range(1, m + 1):
        for rows in combinations(range(m), number):
            for columns in combinations(range(m), number):
                result[sum(1 << i for i in rows),
                       sum(1 << i for i in columns)] = np.linalg.det(
                           orbital_matrix[np.ix_(rows, columns)])
    return result


def emitted_gates(operation, size):
    """Count gates with CUDA-Q itself, independent of the host gate tape."""

    @cudaq.kernel
    def run():
        register = cudaq.qvector(size)
        operation(register)

    return cudaq.estimate_resources(run).to_dict()


@pytest.mark.parametrize("kind", ["full", "partial"])
@pytest.mark.parametrize("step", [1e-5, .03, 2.])
def test_error_bounds_dominate_independent_fock_and_reflection_norms(
        kind, step):
    rng = np.random.default_rng(921)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    schedule = compile_givens_basis_change(q if kind == "full" else q[:, 0])
    represented = np.array(
        schedule.parameters) + step * rng.normal(size=schedule.num_parameters)
    report = estimate_fermion_basis_change_error(schedule, represented)
    exact = fock_lift(schedule.orbital_matrix())
    approximation = fock_lift(schedule.orbital_matrix(represented))
    assert np.linalg.norm(exact - approximation,
                          2) <= report.basis_change_error_bound + 1e-14
    reflection = np.diag([1. if i & 1 else -1. for i in range(16)])
    exact_reflection = exact @ reflection @ exact.conj().T
    approximate_reflection = approximation @ reflection @ approximation.conj(
    ).T
    assert np.linalg.norm(exact_reflection - approximate_reflection,
                          2) <= report.reflection_error_bound + 1e-14
    assert 0 <= report.basis_change_error_bound <= 2
    assert 0 <= report.reflection_error_bound <= 2


def test_phase_occurrences_and_separate_synthesis_contribution_are_counted():
    schedule = compile_givens_basis_change(np.eye(2))
    # Two diagonal phases, one theta, and phi appearing twice in the tape.
    report = estimate_fermion_basis_change_error(schedule, [.1, -.2, .3, -.4],
                                                 synthesis_error_bound=.07)
    assert report.parameter_error_sum == pytest.approx(1.4)
    assert report.synthesis_error_bound == .07
    assert report.basis_change_error_bound == pytest.approx(1.47)
    assert report.reflection_error_bound == 2.
    assert report.parameter_deltas == pytest.approx((.1, -.2, .3, -.4))
    zero = estimate_fermion_basis_change_error(schedule, schedule.parameters)
    assert zero.basis_change_error_bound == zero.reflection_error_bound == 0.
    with pytest.raises(FrozenInstanceError):
        report.parameter_error_sum = 0.


def test_partial_negative_angles_and_output_phases_share_error_budget():
    schedule = compile_givens_basis_change([.6, .8j],
                                           structure=GivensStructure.partial(
                                               2, reference_mode=1))
    represented = np.asarray(schedule.parameters) + [.02, -.03, .04]
    report = estimate_fermion_basis_change_error(schedule,
                                                 represented,
                                                 synthesis_error_bound=.01)
    represented[:] = 100.
    assert report.parameter_error_sum == pytest.approx(.09)
    assert report.basis_change_error_bound == pytest.approx(.10)
    assert report.reflection_error_bound == pytest.approx(.20)
    assert report.parameter_deltas == pytest.approx((.02, -.03, .04))


@pytest.mark.parametrize("kind,mapping,expected", [
    ("full", [0, 1, 2], (3, 9, 15, 12, 36)),
    ("full", [4, 0, 2], (3, 9, 15, 32, 36)),
    ("partial", [4, 0, 2], (2, 3, 7, 24, 24)),
    ("full", [7, 3], (1, 4, 6, 16, 12)),
    ("partial", [0, 5], (1, 2, 4, 20, 12)),
])
def test_resources_count_static_slots_and_physical_parity(
        kind, mapping, expected):
    structure = getattr(GivensStructure, kind)(len(mapping), mode_map=mapping)
    schedule = compile_givens_basis_change(structure=structure)
    report = estimate_fermion_basis_change_resources(schedule,
                                                     num_program_qubits=11)
    assert (report.num_givens_rotations, report.num_number_phases,
            report.num_elementary_rotations, report.num_parity_cnots,
            report.num_clifford_basis_changes) == expected
    assert report.num_hadamards == 8 * expected[0]
    assert report.num_s_phase_gates == 4 * expected[0]
    assert report.num_system_qubits == max(mapping) + 1
    assert report.num_program_qubits == 11
    assert report.num_workspace_qubits == 0
    assert report.num_parameters == schedule.num_parameters
    with pytest.raises(FrozenInstanceError):
        report.num_workspace_qubits = 1


def test_resource_report_does_not_remove_numeric_zero_operations():
    structure = GivensStructure.full(4, mode_map=[1, 3, 2, 5])
    concrete = compile_givens_basis_change(np.eye(4), structure=structure)
    abstract = compile_givens_basis_change(structure=structure)
    assert estimate_fermion_basis_change_resources(
        concrete) == estimate_fermion_basis_change_resources(abstract)


# Every case has all-zero host angles or none, so no slot may be dropped.
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("basis,structure,bits,total", [
    (np.eye(3), None, None, 63),
    (np.eye(2), GivensStructure.full(2, mode_map=(0, 3)), None, 30),
    (np.eye(2), None, 2, 28),
    (None, GivensStructure.partial(3, mode_map=(4, 0, 2),
                                   reference_mode=1), 1, 55),
])
def test_resource_estimate_matches_gates_emitted_by_plan_kernels(
        basis, structure, bits, total, adjoint):
    plan = fermion_basis_change(basis,
                                structure=structure,
                                programmable=bits is not None,
                                angle_bits=bits)
    assert basis is None or not any(plan.schedule.parameters)
    gates = emitted_gates(plan.adjoint_kernel() if adjoint else plan.kernel(),
                          plan.num_qubits)
    report = estimate_fermion_basis_change_resources(plan.schedule,
                                                     angle_bits=bits)
    assert gates.get("h", 0) == report.num_hadamards
    assert gates.get("cx", 0) == report.num_parity_cnots
    # CUDA-Q reports S-dagger separately, as "sdg".
    assert gates.get("s", 0) + gates.get("sdg", 0) == report.num_s_phase_gates
    if bits is None:
        rotations = gates.get("rz", 0) + gates.get("r1", 0)
        assert rotations == report.num_elementary_rotations
    else:
        rotations = gates.get("crz", 0) + gates.get("cr1", 0)
        assert rotations == report.num_bit_controlled_rotations
    # The estimator accounts for every emitted gate and nothing else.
    assert sum(gates.values()) == total == (
        report.num_hadamards + report.num_s_phase_gates +
        report.num_parity_cnots + rotations)
    assert (report.num_system_qubits +
            report.num_program_qubits == plan.num_qubits)


@pytest.mark.parametrize("bits", [None, 2])
def test_external_control_adds_the_enable_to_every_counted_gate(bits):
    plan = fermion_basis_change(np.eye(3) if bits is None else np.eye(2),
                                programmable=bits is not None,
                                angle_bits=bits)
    operation, size = plan.kernel(), plan.num_qubits

    @cudaq.kernel
    def controlled(register: cudaq.qview):
        cudaq.control(operation, register[0], register[1:size + 1])

    # H -> CH, S -> CS, CNOT -> Toffoli, Rz -> CRz, CRz -> CCRz, and so on.
    expected = {
        "c" + name: count
        for name, count in emitted_gates(operation, size).items()
    }
    assert emitted_gates(controlled, size + 1) == expected


@pytest.mark.parametrize("bits", [None, 2])
def test_controlled_kernel_adds_the_enable_to_rotations_only(bits):
    plan = fermion_basis_change(np.eye(3) if bits is None else np.eye(2),
                                programmable=bits is not None,
                                angle_bits=bits)
    rotations = ("crz", "cr1") if bits else ("rz", "r1")
    expected = {
        ("c" + name if name in rotations else name): count
        for name, count in emitted_gates(plan.kernel(),
                                         plan.num_qubits).items()
    }
    assert emitted_gates(plan.controlled_kernel(),
                         plan.num_qubits + 1) == expected
    assert emitted_gates(plan.controlled_kernel(adjoint=True),
                         plan.num_qubits + 1) == expected


def test_plan_resources_include_explicit_register_gaps():
    layout = ((10, 11), (3, 4), (8, 9), (5, 6))
    plan = fermion_basis_change(np.eye(2),
                                programmable=True,
                                angle_bits=2,
                                parameter_registers=layout)
    report = estimate_fermion_basis_change_resources(plan)
    assert plan.num_qubits == 12
    assert report.num_system_qubits + report.num_program_qubits == 12
    assert report == estimate_fermion_basis_change_resources(
        plan.schedule, angle_bits=2, num_program_qubits=10)
    classical = fermion_basis_change(np.eye(3))
    assert estimate_fermion_basis_change_resources(
        classical) == estimate_fermion_basis_change_resources(
            classical.schedule)
    with pytest.raises(ValueError, match="plan supplies"):
        estimate_fermion_basis_change_resources(plan, angle_bits=2)
    with pytest.raises(ValueError, match="plan supplies"):
        estimate_fermion_basis_change_resources(plan, num_program_qubits=10)


@pytest.mark.parametrize("kind", ["full", "partial"])
def test_plan_quantization_bound_dominates_fock_operator_norm(kind):
    rng = np.random.default_rng(57)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    plan = fermion_basis_change(q if kind == "full" else q[:, 0],
                                programmable=True,
                                angle_bits=6)
    exact = fock_lift(plan.schedule.orbital_matrix())
    represented = fock_lift(
        plan.schedule.orbital_matrix(plan.represented_parameters))
    error = np.linalg.norm(exact - represented, 2)
    assert 0 < error <= plan.quantization_error_bound + 1e-14
    assert plan.quantization_error_bound < 2


@pytest.mark.parametrize("value", [
    -1, -1e-300, np.nan, np.inf, 1j, "bad", True, np.True_, "0.1", b"0.1",
    np.complex128(.1 + 1j), 10**400,
    np.array(True),
    np.array(.1j),
    np.array([.1])
])
def test_rejects_invalid_synthesis_bound(value):
    schedule = compile_givens_basis_change([1., 0.])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        estimate_fermion_basis_change_error(schedule, [0., 0., 0.],
                                            synthesis_error_bound=value)


@pytest.mark.parametrize("value,expected", [(0, 0.), (-0., 0.),
                                            (np.float32(.25), .25),
                                            (np.int64(1), 1.),
                                            (Fraction(1, 8), .125),
                                            (Decimal("0.125"), .125),
                                            (np.array(.25), .25),
                                            (np.array(1), 1.)])
def test_accepts_real_scalar_synthesis_bound(value, expected):
    schedule = compile_givens_basis_change([1., 0.])
    report = estimate_fermion_basis_change_error(schedule, [0., 0., 0.],
                                                 synthesis_error_bound=value)
    assert type(report.synthesis_error_bound) is float
    assert report.synthesis_error_bound == expected
    # A -0.0 bound is reported as +0.0.
    assert not np.signbit(report.synthesis_error_bound)
    assert report.basis_change_error_bound == expected


@pytest.mark.parametrize("value", [-1, .5, True, np.inf])
def test_rejects_invalid_program_width(value):
    schedule = compile_givens_basis_change([1., 0.])
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_resources(schedule,
                                                num_program_qubits=value)


@pytest.mark.parametrize("bits,width", [(1, 0), (3, 0), (3, 2),
                                        (3, np.int64(2)), (52, 51)])
def test_program_width_override_must_hold_one_word(bits, width):
    schedule = compile_givens_basis_change(structure=GivensStructure.full(3))
    with pytest.raises(ValueError, match="at least angle_bits"):
        estimate_fermion_basis_change_resources(schedule,
                                                angle_bits=bits,
                                                num_program_qubits=width)


@pytest.mark.parametrize("bits,width", [(1, 1), (3, 3), (3, 4), (52, 52),
                                        (3, 100), (None, 0), (None, 2)])
def test_program_width_override_accepts_one_word_or_more(bits, width):
    schedule = compile_givens_basis_change(structure=GivensStructure.full(3))
    report = estimate_fermion_basis_change_resources(schedule,
                                                     angle_bits=bits,
                                                     num_program_qubits=width)
    assert report.num_program_qubits == width
    # Live word qubits never change the rotations the tape applies.
    assert report.num_bit_controlled_rotations == (bits or 0) * 15


@pytest.mark.parametrize(
    "values", [[], [0., 0.], [0., 0., 1j], [0., 0., np.nan], [0., 0., np.inf]])
def test_rejects_invalid_represented_parameters(values):
    schedule = compile_givens_basis_change([1., 0.])
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_error(schedule, values)


def test_error_report_requires_original_numerical_parameters():
    schedule = compile_givens_basis_change(structure=GivensStructure.full(2))
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_error(schedule, [0.] * 4)


@pytest.mark.parametrize("schedule", [None, np.eye(2), "full"])
def test_metrics_require_a_schedule(schedule):
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_resources(schedule)
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_error(schedule, [0.])


@pytest.mark.parametrize("kind", ["full", "partial"])
def test_error_uses_shortest_circular_parameter_deltas(kind):
    basis = np.eye(2) if kind == "full" else [1., 0.]
    schedule = compile_givens_basis_change(basis)
    offsets = np.array([.01, -.02, .03, -.04][:schedule.num_parameters])
    turns = np.array([1, -2, 3, -4][:schedule.num_parameters])
    represented = np.array(schedule.parameters) + 2 * np.pi * turns + offsets
    report = estimate_fermion_basis_change_error(schedule,
                                                 represented,
                                                 synthesis_error_bound=.05)
    assert report.parameter_deltas == pytest.approx(offsets)
    expected_sum = .14 if kind == "full" else .06
    assert report.parameter_error_sum == pytest.approx(expected_sum)
    assert report.basis_change_error_bound == pytest.approx(expected_sum + .05)
    assert report.synthesis_error_bound == .05
    exact = fock_lift(schedule.orbital_matrix())
    approximation = fock_lift(schedule.orbital_matrix(represented))
    assert np.linalg.norm(exact - approximation,
                          2) <= report.parameter_error_sum


def test_negative_phase_wrapped_to_full_turn_has_small_error():
    schedule = compile_givens_basis_change([np.exp(-.001j), 0.])
    # Slots theta[0], phase[0], phase[1]: only phase[0] is nonzero.
    report = estimate_fermion_basis_change_error(schedule, [0., 0., 0.])
    assert report.parameter_deltas == pytest.approx((0., .001, 0.))
    assert report.basis_change_error_bound == pytest.approx(.001)
    wrapped = estimate_fermion_basis_change_error(schedule,
                                                  [0., 2 * np.pi - .001, 0.])
    assert wrapped.basis_change_error_bound < 1e-14


@pytest.mark.parametrize("bits", [1, 3, 16, 52, np.int64(8)])
def test_resources_count_each_bit_controlled_elementary_rotation(bits):
    schedule = compile_givens_basis_change(structure=GivensStructure.full(3))
    report = estimate_fermion_basis_change_resources(schedule, angle_bits=bits)
    assert report.num_bit_controlled_rotations == int(bits) * 15
    assert report.num_program_qubits == int(bits) * 9
    explicit = estimate_fermion_basis_change_resources(schedule,
                                                       angle_bits=bits,
                                                       num_program_qubits=2 *
                                                       int(bits))
    assert explicit.num_program_qubits == 2 * int(bits)
    assert report.num_parity_cnots == 12
    assert report.num_workspace_qubits == 0
    classical = estimate_fermion_basis_change_resources(schedule)
    assert classical.num_bit_controlled_rotations == 0
    assert classical.num_program_qubits == 0


@pytest.mark.parametrize("bits", [0, -1, 53, True, 2.5, np.inf, "3"])
def test_resources_reject_invalid_angle_width(bits):
    schedule = compile_givens_basis_change([1., 0.])
    with pytest.raises(ValueError):
        estimate_fermion_basis_change_resources(schedule, angle_bits=bits)


@pytest.mark.parametrize("magnitude", [1e16, 1e20, 1e308])
@pytest.mark.parametrize("sign", [-1, 1])
def test_large_angle_error_bound_uses_actual_phase_not_float_tau_remainder(
        magnitude, sign):
    import math
    from cudaq_algorithms.basis_change._plan import GivensBasisChangeSchedule
    angle = sign * magnitude
    schedule = GivensBasisChangeSchedule(GivensStructure.full(2),
                                         (angle, 0., 0., 0.))
    # Floating tau reduction is not a phase-preserving operation for a large
    # radian angle; the estimator must detect the resulting physical error.
    represented = math.remainder(angle, math.tau)
    report = estimate_fermion_basis_change_error(schedule,
                                                 [represented, 0., 0., 0.])
    # Only mode 0 carries a phase, so this 2x2 diagonal has the same operator
    # norm as the full two-mode Fock difference.
    exact = np.diag([1., np.exp(1j * angle)])
    approximate = np.diag([1., np.exp(1j * represented)])
    measured = np.linalg.norm(exact - approximate, ord=2)
    assert measured > .1
    assert measured <= report.basis_change_error_bound
    unchanged = estimate_fermion_basis_change_error(schedule,
                                                    [angle, 0., 0., 0.])
    assert unchanged.basis_change_error_bound == 0.


def test_opposite_large_angle_operands_do_not_overflow_error_estimate():
    from cudaq_algorithms.basis_change._plan import GivensBasisChangeSchedule
    angle = 1e308
    schedule = GivensBasisChangeSchedule(GivensStructure.full(2),
                                         (angle, 0., 0., 0.))
    report = estimate_fermion_basis_change_error(schedule,
                                                 [-angle, 0., 0., 0.])
    measured = abs(np.exp(1j * angle) - np.exp(-1j * angle))
    assert measured <= report.basis_change_error_bound
    assert np.isfinite(report.parameter_deltas[0])
