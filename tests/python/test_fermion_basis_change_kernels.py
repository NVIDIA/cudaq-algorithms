# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete-state checks for the existing-gate fermionic kernel backend."""

import os
import subprocess
import sys

import cudaq
import numpy as np
import pytest

from cudaq_algorithms import state_from
from cudaq_algorithms.basis_change._kernels import (classical_kernel,
                                                    programmable_kernel)
from cudaq_algorithms.basis_change._plan import (GivensStructure,
                                                 compile_givens_basis_change)


def _fock_matrix(orbital):
    """Independent determinant minors, with qubit zero least significant."""
    orbital = np.asarray(orbital, dtype=complex)
    occupied = [[p for p in range(len(orbital)) if bits & (1 << p)]
                for bits in range(1 << len(orbital))]
    result = np.zeros((len(occupied), len(occupied)), dtype=complex)
    for row, rows in enumerate(occupied):
        for col, cols in enumerate(occupied):
            if len(rows) == len(cols):
                result[row, col] = np.linalg.det(orbital[np.ix_(rows, cols)])
    return result


def _creation(mode, size):
    result = np.zeros((1 << size, 1 << size), dtype=complex)
    for bits in range(1 << size):
        if not bits & (1 << mode):
            parity = (bits & ((1 << mode) - 1)).bit_count()
            result[bits | (1 << mode), bits] = (-1)**parity
    return result


def _random_state(size, seed=711):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << size) + 1j * rng.normal(size=1 << size)
    return state / np.linalg.norm(state)


def _runner(operation, size, *, controlled=False, generic_adjoint=False):
    """Compile one route once; the returned map simulates many inputs."""

    @cudaq.kernel
    def controlled_operation(register: cudaq.qview):
        cudaq.control(operation, register[0], register[1:size])

    selected = controlled_operation if controlled else operation

    @cudaq.kernel
    def inverted_operation(register: cudaq.qview):
        cudaq.adjoint(selected, register)

    applied = inverted_operation if generic_adjoint else selected

    @cudaq.kernel
    def run(state: cudaq.State):
        register = cudaq.qvector(state)
        applied(register)

    return lambda initial: np.asarray(
        cudaq.get_state(run, state_from(np.ascontiguousarray(initial))))


def _apply(operation, initial, **kwargs):
    return _runner(operation, int(np.log2(len(initial))), **kwargs)(initial)


def _matrix(operation, size, **kwargs):
    run = _runner(operation, size, **kwargs)
    return np.column_stack(
        [run(column.copy()) for column in np.eye(1 << size, dtype=complex).T])


def _two_mode_orbital(parameters, kind):
    """Closed form independent of the host planner and its operation tape."""
    if kind == "full":
        a, b, theta, phi = parameters
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -np.exp(-1j * phi) * s], [
            np.exp(1j * phi) * s, c
        ]]) @ np.diag(np.exp(1j * np.array([a, b])))
    theta, a, b = parameters
    return np.diag(np.exp(1j * np.array([a, b]))) @ np.array([[
        np.cos(theta), -np.sin(theta)
    ], [np.sin(theta), np.cos(theta)]])


def _mixing(first, second, theta, phi=0.):
    """Three-mode embedding of [[c, -exp(-i phi) s], [exp(i phi) s, c]]."""
    result = np.eye(3, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    pair = (first, second)
    result[np.ix_(pair, pair)] = [[c, -np.exp(-1j * phi) * s],
                                  [np.exp(1j * phi) * s, c]]
    return result


def _three_mode_full(values):
    """Closed form: diagonal phases, then mixings (1,2), (0,1), (1,2)."""
    result = np.diag(np.exp(1j * np.asarray(values[:3])))
    for k, pair in enumerate(((1, 2), (0, 1), (1, 2))):
        result = _mixing(*pair, values[3 + 2 * k], values[4 + 2 * k]) @ result
    return result


def _three_mode_partial(values, reference_mode=0):
    """Closed form: chain (0,1), (1,2), reversed when rooted at mode 2."""
    pairs = ((1, 2), (0, 1)) if reference_mode == 2 else ((0, 1), (1, 2))
    result = np.eye(3, dtype=complex)
    for theta, pair in zip(values[:2], pairs):
        result = _mixing(*pair, theta) @ result
    return np.diag(np.exp(1j * np.asarray(values[2:]))) @ result


def _physical(orbital, mode_map):
    """Embed logical modes on their physical qubits; spectators stay fixed."""
    physical = np.eye(max(mode_map) + 1, dtype=complex)
    physical[np.ix_(mode_map, mode_map)] = orbital
    return physical


def _layout(structure, width):
    start = structure.num_system_qubits
    return tuple(
        tuple(range(start + k * width, start + (k + 1) * width))
        for k in range(structure.num_parameters))


def _joint_reference(initial,
                     structure,
                     layout,
                     weights,
                     orbital_factory,
                     *,
                     controlled=False,
                     adjoint=False):
    system_width = structure.num_system_qubits
    width = int(np.log2(len(initial))) - int(controlled)
    expected = initial.copy()
    for upper in range(1 << (width - system_width)):
        index = upper << system_width
        values = [
            sum(weight * ((index >> qubit) & 1)
                for qubit, weight in zip(word, weights)) for word in layout
        ]
        physical = np.eye(system_width, dtype=complex)
        physical[np.ix_(structure.mode_map,
                        structure.mode_map)] = (orbital_factory(values))
        operator = _fock_matrix(physical)
        if adjoint:
            operator = operator.conj().T
        positions = np.arange(1 << system_width) + index
        if controlled:
            positions = 2 * positions + 1
        expected[positions] = operator @ initial[positions]
    return expected


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("complex_input", [False, True])
def test_classical_full_transform_matches_every_fock_sector(
        size, complex_input):
    rng = np.random.default_rng(804 + size)
    data = rng.normal(size=(size, size))
    if complex_input:
        data = data + 1j * rng.normal(size=(size, size))
    unitary, _ = np.linalg.qr(data)
    operation = classical_kernel(compile_givens_basis_change(unitary))
    np.testing.assert_allclose(_matrix(operation, size),
                               _fock_matrix(unitary),
                               atol=3e-12,
                               rtol=0)


@pytest.mark.parametrize("unitary", [
    np.diag([-1., 1.]),
    np.diag(np.exp(1j * np.array([0.31, -0.73]))),
    np.array([[0., 1.], [1., 0.]]),
    np.array([[0., np.exp(0.43j)], [1., 0.]]),
])
def test_classical_keeps_vacuum_and_determinant_phases(unitary):
    operation = classical_kernel(compile_givens_basis_change(unitary))
    np.testing.assert_allclose(_matrix(operation, len(unitary)),
                               _fock_matrix(unitary),
                               atol=3e-12,
                               rtol=0)


@pytest.mark.parametrize("mode_map", [(0, 2), (2, 0), (1, 3)])
def test_classical_nonadjacent_modes_keep_spectator_parity(mode_map):
    unitary = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    structure = GivensStructure.full(2, mode_map=mode_map)
    operation = classical_kernel(
        compile_givens_basis_change(unitary, structure=structure))
    physical = np.eye(max(mode_map) + 1, dtype=complex)
    physical[np.ix_(mode_map, mode_map)] = unitary
    np.testing.assert_allclose(_matrix(operation, len(physical)),
                               _fock_matrix(physical),
                               atol=3e-12,
                               rtol=0)


@pytest.mark.parametrize("mode_map", [(2, 0, 1), (4, 0, 2)])
def test_classical_parity_strings_cross_active_modes(mode_map):
    # Logical pair (0, 1) is physically separated by mapped logical mode 2.
    rng = np.random.default_rng(1205)
    unitary, _ = np.linalg.qr(
        rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    structure = GivensStructure.full(3, mode_map=mode_map)
    schedule = compile_givens_basis_change(unitary, structure=structure)
    # Generic data: no mixing angle is near a multiple of pi.
    assert np.min(np.abs(np.sin(schedule.parameters[3::2]))) > 0.5
    physical = _physical(unitary, mode_map)
    np.testing.assert_allclose(_matrix(classical_kernel(schedule),
                                       len(physical)),
                               _fock_matrix(physical),
                               atol=3e-12,
                               rtol=0)


@pytest.mark.parametrize("reference_mode", [0, 1, 2])
def test_classical_partial_maps_creation_operator_with_complex_phase(
        reference_mode):
    vector = np.array([1., 1j, -1.]) / np.sqrt(3)
    structure = GivensStructure.partial(3, reference_mode=reference_mode)
    operation = classical_kernel(
        compile_givens_basis_change(vector, structure=structure))
    actual = _matrix(operation, 3)
    expected = sum(value * _creation(p, 3) for p, value in enumerate(vector))
    np.testing.assert_allclose(
        actual @ _creation(reference_mode, 3) @ actual.conj().T,
        expected,
        atol=4e-12,
        rtol=0)
    np.testing.assert_allclose(actual[:, 0],
                               np.eye(8)[:, 0],
                               atol=3e-12,
                               rtol=0)


def test_classical_partial_parity_string_crosses_active_mode():
    # Logical pair (0, 1) sits on qubits 0 and 2 around mapped logical mode 2.
    rng = np.random.default_rng(1203)
    vector = rng.normal(size=3) + 1j * rng.normal(size=3)
    vector /= np.linalg.norm(vector)
    mode_map = (0, 2, 1)
    structure = GivensStructure.partial(3, reference_mode=1, mode_map=mode_map)
    schedule = compile_givens_basis_change(vector, structure=structure)
    assert np.min(np.abs(np.sin(schedule.parameters[:2]))) > 0.5
    orbital = _three_mode_partial(schedule.parameters, reference_mode=1)
    np.testing.assert_allclose(orbital[:, 1], vector, atol=1e-12, rtol=0)
    actual = _matrix(classical_kernel(schedule), 3)
    np.testing.assert_allclose(actual,
                               _fock_matrix(_physical(orbital, mode_map)),
                               atol=3e-12,
                               rtol=0)
    creation = [_creation(qubit, 3) for qubit in mode_map]
    np.testing.assert_allclose(
        actual @ creation[1] @ actual.conj().T,
        sum(value * operator for value, operator in zip(vector, creation)),
        atol=4e-12,
        rtol=0)


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("inverse", ["explicit", "generic"])
def test_classical_inverse_and_control_compose_exactly(controlled, inverse):
    unitary = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    unitary = np.diag(np.exp(1j * np.array([0.31, -0.73]))) @ unitary
    operation = classical_kernel(compile_givens_basis_change(unitary),
                                 adjoint=inverse == "explicit")
    expected = _fock_matrix(unitary).conj().T
    if controlled:
        full = np.eye(8, dtype=complex)
        full[1::2, 1::2] = expected
        expected = full
    np.testing.assert_allclose(_matrix(operation,
                                       2 + int(controlled),
                                       controlled=controlled,
                                       generic_adjoint=inverse == "generic"),
                               expected,
                               atol=4e-12,
                               rtol=0)


@pytest.mark.parametrize("kind", ["full", "partial"])
@pytest.mark.parametrize("controlled, inverse", [
    (False, "forward"),
    (False, "explicit"),
    (False, "generic"),
    (True, "forward"),
    (True, "explicit"),
    (True, "generic"),
])
def test_programmed_superposed_words_preserve_all_relative_phases(
        kind, controlled, inverse):
    structure = getattr(GivensStructure, kind)(2)
    schedule = compile_givens_basis_change(structure=structure)
    # Internal constructor takes precomputed weights. A non-grid value also
    # exercises mixing for a one-bit word rather than only integer pi signs.
    weights = (0.63, )
    layout = _layout(structure, len(weights))
    operation = programmable_kernel(schedule,
                                    layout,
                                    weights,
                                    adjoint=inverse == "explicit")
    initial = _random_state(2 + structure.num_parameters + int(controlled))
    expected = _joint_reference(initial,
                                structure,
                                layout,
                                weights,
                                lambda values: _two_mode_orbital(values, kind),
                                controlled=controlled,
                                adjoint=inverse != "forward")
    np.testing.assert_allclose(_apply(operation,
                                      initial,
                                      controlled=controlled,
                                      generic_adjoint=inverse == "generic"),
                               expected,
                               atol=8e-12,
                               rtol=0)


def test_quantum_words_override_every_classically_zero_slot():
    schedule = compile_givens_basis_change(np.eye(2))
    structure = schedule.structure
    weights = (np.pi / 2, np.pi)
    layout = _layout(structure, 2)
    operation = programmable_kernel(schedule, layout, weights)
    initial = _random_state(10)
    expected = _joint_reference(
        initial, structure, layout, weights,
        lambda values: _two_mode_orbital(values, "full"))
    assert np.linalg.norm(expected - initial) > 0.1
    np.testing.assert_allclose(_apply(operation, initial),
                               expected,
                               atol=8e-12,
                               rtol=0)


@pytest.mark.parametrize("inverse", ["forward", "explicit", "generic"])
def test_two_bit_words_keep_reordered_modes_and_spectator_parity(inverse):
    structure = GivensStructure.partial(2, mode_map=(2, 0))
    schedule = compile_givens_basis_change(structure=structure)
    weights = (np.pi / 2, np.pi)
    layout = _layout(structure, 2)
    initial = _random_state(9)
    operation = programmable_kernel(schedule,
                                    layout,
                                    weights,
                                    adjoint=inverse == "explicit")
    expected = _joint_reference(
        initial,
        structure,
        layout,
        weights,
        lambda values: _two_mode_orbital(values, "partial"),
        adjoint=inverse != "forward")
    np.testing.assert_allclose(_apply(operation,
                                      initial,
                                      generic_adjoint=inverse == "generic"),
                               expected,
                               atol=8e-12,
                               rtol=0)


def test_program_registers_can_be_reordered_with_unused_gaps():
    structure = GivensStructure.partial(2)
    schedule = compile_givens_basis_change(structure=structure)
    weights = (0.5, )
    layout = ((5, ), (2, ), (4, ))
    operation = programmable_kernel(schedule, layout, weights)
    initial = _random_state(6)
    expected = _joint_reference(
        initial, structure, layout, weights,
        lambda values: _two_mode_orbital(values, "partial"))
    np.testing.assert_allclose(_apply(operation, initial),
                               expected,
                               atol=8e-12,
                               rtol=0)


@pytest.mark.parametrize("weight", [np.pi, 0.43])
def test_three_mode_programmed_partial_matches_complete_joint_state(weight):
    structure = GivensStructure.partial(3)
    schedule = compile_givens_basis_change(structure=structure)
    weights = (weight, )
    layout = _layout(structure, 1)
    operation = programmable_kernel(schedule, layout, weights)
    initial = _random_state(8)
    expected = _joint_reference(initial, structure, layout, weights,
                                _three_mode_partial)
    np.testing.assert_allclose(_apply(operation, initial),
                               expected,
                               atol=8e-12,
                               rtol=0)


@pytest.mark.parametrize("mode_map, controlled, inverse", [
    (None, False, "forward"),
    (None, False, "explicit"),
    (None, True, "forward"),
    ((2, 0, 1), False, "forward"),
])
def test_programmed_three_mode_full_matches_complete_joint_state(
        mode_map, controlled, inverse):
    # Map (2, 0, 1) runs logical pair (0, 1) across the active qubit 1.
    structure = GivensStructure.full(3, mode_map=mode_map)
    schedule = compile_givens_basis_change(structure=structure)
    weights = (0.63, )
    layout = _layout(structure, 1)
    operation = programmable_kernel(schedule,
                                    layout,
                                    weights,
                                    adjoint=inverse == "explicit")
    initial = _random_state(12 + int(controlled))
    expected = _joint_reference(initial,
                                structure,
                                layout,
                                weights,
                                _three_mode_full,
                                controlled=controlled,
                                adjoint=inverse == "explicit")
    np.testing.assert_allclose(_apply(operation,
                                      initial,
                                      controlled=controlled),
                               expected,
                               atol=8e-12,
                               rtol=0)


@pytest.mark.parametrize("reference_mode", [1, 2])
@pytest.mark.parametrize("inverse", ["forward", "explicit"])
def test_programmed_partial_words_wrap_negative_left_chain_angles(
        reference_mode, inverse):
    structure = GivensStructure.partial(3, reference_mode=reference_mode)
    schedule = compile_givens_basis_change(structure=structure)
    layout = _layout(structure, 1)
    # Left-chain angles are negative. An unsigned full-turn word stores -0.43
    # as 2*pi - 0.43, which must apply exactly the negative rotation.
    stored, signed = (2 * np.pi - 0.43, ), (-0.43, )
    operation = programmable_kernel(schedule,
                                    layout,
                                    stored,
                                    adjoint=inverse == "explicit")
    initial = _random_state(8)
    expected = _joint_reference(
        initial,
        structure,
        layout,
        signed,
        lambda values: _three_mode_partial(values, reference_mode),
        adjoint=inverse == "explicit")
    np.testing.assert_allclose(_apply(operation, initial),
                               expected,
                               atol=8e-12,
                               rtol=0)


def test_programmed_forward_then_explicit_inverse_restores_entangled_state():
    structure = GivensStructure.partial(2, mode_map=(2, 0))
    schedule = compile_givens_basis_change(structure=structure)
    layout = _layout(structure, 2)
    weights = (0.41, 0.82)
    forward = programmable_kernel(schedule, layout, weights)
    inverse = programmable_kernel(schedule, layout, weights, adjoint=True)
    initial = _random_state(9)

    @cudaq.kernel
    def roundtrip(register: cudaq.qview):
        forward(register)
        inverse(register)

    np.testing.assert_allclose(_apply(roundtrip, initial),
                               initial,
                               atol=8e-12,
                               rtol=0)


def test_compiled_programmed_kernel_uses_existing_controlled_rotations():
    structure = GivensStructure.full(2)
    schedule = compile_givens_basis_change(structure=structure)
    operation = programmable_kernel(schedule, _layout(structure, 2),
                                    (np.pi / 2, np.pi))
    ir = str(operation)
    assert "quake.rz" in ir
    assert "quake.r1" in ir
    assert "programmable" not in ir.lower()
    assert "angleformat" not in ir.lower()


@pytest.mark.parametrize(
    "structure, counts",
    [
        # Three mixings with separations 2, 4, 2 and nine number phases.
        (GivensStructure.full(3, mode_map=(4, 0, 2)), (12, 18, 32, 24, 6, 6)),
        # Two mixings with separations 1, 2 and three number phases.
        (GivensStructure.partial(3, reference_mode=2, mode_map=(0, 2, 1)),
         (8, 6, 12, 16, 4, 4)),
    ])
def test_programmed_kernel_emits_every_bit_controlled_rotation(
        structure, counts):
    # Two-bit words on all-zero host data: per mixing 2b controlled Rz, 4d
    # CNOT, 8 H and 2 S + 2 S-dagger; per number phase b controlled R1.
    basis = np.eye(3) if structure.kind == "full" else np.eye(3)[:, 2]
    schedule = compile_givens_basis_change(basis, structure=structure)
    assert not any(schedule.parameters)
    layout = _layout(structure, 2)
    operation = programmable_kernel(schedule, layout, (np.pi / 2, np.pi))
    size = layout[-1][-1] + 1

    @cudaq.kernel
    def allocate():
        register = cudaq.qvector(size)
        operation(register)

    expected = dict(zip(("crz", "cr1", "cx", "h", "s", "sdg"), counts))
    assert cudaq.estimate_resources(allocate).to_dict() == expected


_GENERIC_ADJOINT_SCRIPT = """
import os

import cudaq
import numpy as np

from cudaq_algorithms import fermion_basis_change, state_from
from cudaq_algorithms.basis_change import GivensStructure

cudaq.set_target(os.environ["CUDAQ_DEFAULT_SIMULATOR"])
rng = np.random.default_rng(97)
data = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
unitary = np.linalg.qr(data)[0]
vector = rng.normal(size=3) + 1j * rng.normal(size=3)
partial = GivensStructure.partial(3, reference_mode=1, mode_map=(0, 2, 1))
plans = (fermion_basis_change(unitary, programmable=True, angle_bits=2),
         fermion_basis_change(vector / np.linalg.norm(vector),
                              structure=partial,
                              programmable=True,
                              angle_bits=1))


def run(operation, initial):

    @cudaq.kernel
    def driver(state: cudaq.State):
        register = cudaq.qvector(state)
        operation(register)

    return np.asarray(cudaq.get_state(driver, state_from(initial)))


def generic_inverse(operation):

    @cudaq.kernel
    def inverse(register: cudaq.qview):
        cudaq.adjoint(operation, register)

    return inverse


for plan in plans:
    size = 1 << plan.num_qubits
    initial = rng.normal(size=size) + 1j * rng.normal(size=size)
    initial /= np.linalg.norm(initial)
    explicit = run(plan.adjoint_kernel(), initial)
    generic = run(generic_inverse(plan.kernel()), initial)
    print("adjoint", np.max(np.abs(generic - explicit)),
          np.linalg.norm(explicit - initial))
"""


def test_generic_adjoint_of_programmed_plan_compiles_without_diagnostics(
        tmp_path):
    # A fresh process compiles the reversal itself. Failed reversal prints
    # compiler errors to the process stderr even when the result is correct.
    # Kernels need inspectable source, so the script runs from a file.
    script = tmp_path / "generic_adjoint.py"
    script.write_text(_GENERIC_ADJOINT_SCRIPT)
    env = dict(os.environ)
    env.setdefault("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")
    command = [sys.executable, "-W", "ignore", str(script)]
    result = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    # Match the MLIR adjoint diagnostics only, so unrelated runtime messages
    # (for example from another simulator target) cannot fail this test.
    assert "could not reverse" not in result.stderr
    assert "not reversible" not in result.stderr
    assert "auto-generation of adjoint" not in result.stderr
    rows = [
        tuple(float(value) for value in line.split()[1:])
        for line in result.stdout.splitlines() if line.startswith("adjoint ")
    ]
    assert len(rows) == 2
    for deviation, change in rows:
        assert deviation < 1e-12
        assert change > 0.1
