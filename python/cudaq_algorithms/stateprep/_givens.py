# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Givens-rotation Slater determinant preparation.

Prepares the Slater determinant of an orthonormal orbital-coefficient
matrix ``Q`` (``num_spin_orbitals x num_electrons``) on the
Jordan-Wigner / little-endian qubit layout: the amplitude of basis state
``|S>`` with occupied set ``S`` is ``det(Q[S, :])``, up to a global
phase.

Host side, ``make_givens_rotation_schedule`` reduces ``Q`` to the
computational-basis determinant with adjacent (nearest-neighbor) Givens
row rotations, bottom-up per column; the state-preparation kernels then
apply the inverse rotations in reverse order to the Hartree-Fock-style
determinant ``|1...10...0>``. Real and complex matrices dispatch
automatically (a complex dtype routes complex even when every value is
real); complex schedules carry a relative phase per rotation plus one
final phase per electron.

The kernels are module-level and composable from user kernels, like the
ansatz kernels in ``_kernels``; the host helpers
(``get_givens_rotation_indices`` / ``..._angles`` / ``..._phases``)
supply the flattened arrays that cross the kernel boundary, and
``slater_determinant_kernel`` packages a schedule as a ready-to-inject
``(qubits: qview)`` kernel. Device kernels have no error channel, so
mismatched flattened inputs are a no-op — validation belongs on the host
(``validate_givens_rotation_schedule`` runs automatically for built
schedules).

Kernel-language constraints (they shape the code below):

* Python ``exp_pauli`` only accepts the ``(angle, register, word)``
  form — individual qubit operands are rejected. Runtime-contiguous
  ``qview`` slices (``qubits[a:a + 2]``) are valid registers, and the
  schedule only ever emits *adjacent* rotations (validated on the
  host), so the two-qubit rotation maps onto slices exactly. A
  non-adjacent variant would need host-side full-width Pauli words.
* Guards use positive ``if`` blocks, never early ``return`` (kernel
  ``return`` is silently ignored by the compiler,
  https://github.com/NVIDIA/cuda-quantum/issues/4845).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import numpy as np

import cudaq

# ============================================================================
# Device kernels (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def givens_rotation(qubits: cudaq.qview, theta: float, first_orbital: int,
                    second_orbital: int):
    """Apply an adjacent real fermionic Givens rotation.

    CUDA-Q's built-in Givens convention maps ``|10>`` to
    ``cos(theta)|10> - sin(theta)|01>``; the state-preparation convention
    here uses the opposite sign, so this inlines the built-in rotation
    at -theta. Non-adjacent orbital pairs are a no-op, matching the
    host-side adjacency validation.
    """
    if first_orbital + 1 == second_orbital:
        pair = qubits[first_orbital:first_orbital + 2]
        exp_pauli(0.5 * theta, pair, "YX")
        exp_pauli(-0.5 * theta, pair, "XY")
    elif second_orbital + 1 == first_orbital:
        pair = qubits[second_orbital:second_orbital + 2]
        exp_pauli(-0.5 * theta, pair, "YX")
        exp_pauli(0.5 * theta, pair, "XY")


@cudaq.kernel
def phase_givens_rotation(qubits: cudaq.qview, theta: float, phase: float,
                          first_orbital: int, second_orbital: int):
    """Apply an adjacent phase-aware fermionic Givens rotation."""
    givens_rotation(qubits, theta, first_orbital, second_orbital)
    # rz(phase) equals exp(i * phase * n) on this qubit up to global phase.
    rz(phase, qubits[second_orbital])


@cudaq.kernel
def slater_determinant(qubits: cudaq.qview, orbital_indices: list[int],
                       angles: list[float], num_electrons: int):
    """Prepare a real Slater determinant from a flattened Givens schedule.

    ``orbital_indices`` holds two entries per angle
    (``get_givens_rotation_indices``); mismatched flattened inputs are a
    no-op (positive guard — see the module docstring).
    """
    if len(orbital_indices) == 2 * len(angles):
        for i in range(num_electrons):
            x(qubits[i])
        for i in range(len(angles)):
            givens_rotation(qubits, angles[i], orbital_indices[2 * i],
                            orbital_indices[2 * i + 1])


@cudaq.kernel
def complex_slater_determinant(qubits: cudaq.qview, orbital_indices: list[int],
                               angles: list[float], phases: list[float],
                               final_phases: list[float], num_electrons: int):
    """Prepare a complex Slater determinant from a flattened Givens schedule."""
    if (len(orbital_indices) == 2 * len(angles) and len(phases) == len(angles)
            and len(final_phases) >= num_electrons):
        for i in range(num_electrons):
            x(qubits[i])
        for i in range(num_electrons):
            rz(final_phases[i], qubits[i])
        for i in range(len(angles)):
            phase_givens_rotation(qubits, angles[i], phases[i],
                                  orbital_indices[2 * i],
                                  orbital_indices[2 * i + 1])


# ============================================================================
# Host-side schedule construction
# ============================================================================


@dataclass(frozen=True)
class GivensRotation:
    """One adjacent Givens rotation between two orbitals.

    ``phase`` is the relative phase applied after the rotation
    (``exp(i * phase * n_second)``); it is 0 for real schedules.
    """

    first_orbital: int
    second_orbital: int
    theta: float
    phase: float = 0.0


@dataclass(frozen=True)
class GivensRotationSchedule:
    """A Givens rotation sequence preparing a Slater determinant.

    ``rotations`` are in application order (the reverse of the
    elimination order). ``final_phases`` holds one phase per electron,
    applied to the occupied qubits before the rotations; it is all-zero
    for real schedules.
    """

    num_spin_orbitals: int
    num_electrons: int
    is_complex: bool = False
    rotations: list[GivensRotation] = field(default_factory=list)
    final_phases: list[float] = field(default_factory=list)


def _as_matrix(orbital_coefficients):
    """Normalize input to a list of rows; detect complex entries."""
    if hasattr(orbital_coefficients, "tolist"):
        is_complex = getattr(getattr(orbital_coefficients, "dtype", None),
                             "kind", None) == "c"
        rows = orbital_coefficients.tolist()
    else:
        is_complex = False
        rows = [list(row) for row in orbital_coefficients]
    if not is_complex:
        # np.complex64 is not a subclass of Python ``complex`` (only
        # np.complex128 is), so a nested list of np.complex64 scalars would
        # otherwise misdispatch to the real branch and silently drop phases.
        is_complex = any(
            isinstance(value, (complex, np.complexfloating)) for row in rows
            for value in row)
    return rows, is_complex


def _validate_orbital_coefficients(rows, tolerance, is_complex):
    if not rows:
        raise ValueError("orbital_coefficients must not be empty")

    num_electrons = len(rows[0])
    if num_electrons == 0:
        raise ValueError(
            "orbital_coefficients must contain at least one occupied orbital")
    if num_electrons > len(rows):
        raise ValueError(
            "number of occupied orbitals cannot exceed number of spin "
            "orbitals")
    for row in rows:
        if len(row) != num_electrons:
            raise ValueError(
                "orbital_coefficients must be a rectangular matrix")

    for col in range(num_electrons):
        norm = sum(abs(row[col])**2 for row in rows)
        if abs(norm - 1.0) > 100.0 * tolerance:
            raise ValueError("orbital_coefficients columns must be normalized")
        for other in range(col + 1, num_electrons):
            if is_complex:
                overlap = sum(
                    complex(row[col]).conjugate() * row[other] for row in rows)
            else:
                overlap = sum(row[col] * row[other] for row in rows)
            if abs(overlap) > 100.0 * tolerance:
                raise ValueError(
                    "orbital_coefficients columns must be orthogonal")


def _argument_or_zero(value, tolerance):
    if abs(value) <= tolerance:
        return 0.0
    return cmath.phase(value)


def make_givens_rotation_schedule(orbital_coefficients,
                                  tolerance=1.0e-12) -> GivensRotationSchedule:
    """Build the Givens rotation schedule preparing a Slater determinant.

    ``orbital_coefficients`` is a (num_spin_orbitals x num_electrons)
    matrix (numpy array or nested lists) whose orthonormal columns are
    the occupied orbitals. Real and complex inputs dispatch automatically
    (a complex dtype routes complex even when all values are real). For
    interleaved-spin systems the matrix rows must follow the package's
    alpha (even) / beta (odd) spin-orbital ordering, so the prepared
    determinant composes with ``hartree_fock_occupation`` references and
    the UCCSD excitation conventions built at the same spin.
    """
    rows, is_complex = _as_matrix(orbital_coefficients)
    _validate_orbital_coefficients(rows, tolerance, is_complex)

    num_spin_orbitals = len(rows)
    num_electrons = len(rows[0])
    work = [[complex(value) for value in row] for row in rows]
    eliminations = []

    # Zero the sub-diagonal of each column bottom-up with rotations of
    # adjacent rows; the surviving diagonal is real for real inputs and a
    # per-column phase (the final phases) for complex inputs.
    for col in range(num_electrons):
        for row in range(num_spin_orbitals - 1, col, -1):
            upper_row = row - 1
            upper = work[upper_row][col]
            lower = work[row][col]

            if abs(lower) <= tolerance:
                continue

            upper_magnitude = abs(upper)
            lower_magnitude = abs(lower)
            radius = math.hypot(upper_magnitude, lower_magnitude)

            cosine = upper_magnitude / radius
            sine = lower_magnitude / radius
            theta = math.atan2(sine, cosine)
            if is_complex:
                phase = (_argument_or_zero(lower, tolerance) -
                         _argument_or_zero(upper, tolerance))
            else:
                # The real path works on signed values directly.
                cosine = upper.real / radius
                sine = lower.real / radius
                theta = math.atan2(sine, cosine)
                phase = 0.0

            lower_phase = cmath.exp(-1.0j * phase)
            for k in range(num_electrons):
                upper_value = work[upper_row][k]
                lower_value = lower_phase * work[row][k]
                work[upper_row][k] = cosine * upper_value + sine * lower_value
                work[row][k] = -sine * upper_value + cosine * lower_value

            eliminations.append(GivensRotation(upper_row, row, theta, phase))

    if is_complex:
        final_phases = [
            _argument_or_zero(work[col][col], tolerance)
            for col in range(num_electrons)
        ]
    else:
        final_phases = [0.0] * num_electrons

    # State preparation applies the inverse of the row rotations that
    # reduce the orbital-coefficient matrix to the basis determinant.
    schedule = GivensRotationSchedule(num_spin_orbitals=num_spin_orbitals,
                                      num_electrons=num_electrons,
                                      is_complex=is_complex,
                                      rotations=list(reversed(eliminations)),
                                      final_phases=final_phases)
    validate_givens_rotation_schedule(schedule)
    return schedule


def validate_givens_rotation_schedule(schedule: GivensRotationSchedule):
    """Validate a schedule against the state-preparation kernel contract.

    ``make_givens_rotation_schedule`` output always passes; this guards
    hand-built schedules (the kernels themselves cannot raise).
    """
    if schedule.num_spin_orbitals <= 0:
        raise ValueError("num_spin_orbitals must be greater than zero")
    if schedule.num_electrons <= 0:
        raise ValueError("num_electrons must be greater than zero")
    if schedule.num_electrons > schedule.num_spin_orbitals:
        raise ValueError("num_electrons cannot exceed num_spin_orbitals")

    for rotation in schedule.rotations:
        first = rotation.first_orbital
        second = rotation.second_orbital
        if (min(first, second) < 0
                or max(first, second) >= schedule.num_spin_orbitals):
            raise ValueError("Givens rotation orbital index is out of range")
        if abs(first - second) != 1:
            raise ValueError(
                "Givens state-preparation kernels require adjacent rotations")

    if schedule.is_complex:
        if len(schedule.final_phases) != schedule.num_electrons:
            raise ValueError(
                "complex Givens schedules require one final phase per "
                "electron")
    elif schedule.final_phases and len(
            schedule.final_phases) != schedule.num_electrons:
        raise ValueError("real Givens schedule final phases must be empty or "
                         "match num_electrons")


# ============================================================================
# Flattened kernel arguments
# ============================================================================


def get_givens_rotation_indices(schedule: GivensRotationSchedule) -> list[int]:
    """Flattened (first, second) orbital pairs, two entries per rotation."""
    indices = []
    for rotation in schedule.rotations:
        indices.append(int(rotation.first_orbital))
        indices.append(int(rotation.second_orbital))
    return indices


def get_givens_rotation_angles(
        schedule: GivensRotationSchedule) -> list[float]:
    """Rotation angles in application order."""
    return [float(rotation.theta) for rotation in schedule.rotations]


def get_givens_rotation_phases(
        schedule: GivensRotationSchedule) -> list[float]:
    """Relative phases in application order (all zero for real schedules)."""
    return [float(rotation.phase) for rotation in schedule.rotations]


# ============================================================================
# Kernel factory
# ============================================================================


def slater_determinant_kernel(schedule: GivensRotationSchedule):
    """A ``(qubits: qview)`` kernel preparing the schedule's determinant.

    The returned kernel expects a ``schedule.num_spin_orbitals``-wide
    register in ``|0...0>`` and is directly injectable as a ``state_prep``
    kernel (e.g. into ``PauliLCU.encode_kernel``). It dispatches on
    ``schedule.is_complex`` to the ``slater_determinant`` /
    ``complex_slater_determinant`` kernel path. The schedule is flattened
    into plain index/angle/phase arrays before capture — nested
    structures marshal as kernel *arguments* but cannot be
    closure-*captured* by Python kernels.
    """
    validate_givens_rotation_schedule(schedule)
    orbital_indices = get_givens_rotation_indices(schedule)
    angles = get_givens_rotation_angles(schedule)
    phases = get_givens_rotation_phases(schedule)
    final_phases = [float(value) for value in schedule.final_phases]
    num_electrons = int(schedule.num_electrons)

    # Positive guards choose a kernel shape whose captured lists are all
    # non-empty: empty list captures fail to launch (cuda-quantum#4847),
    # and num_electrons >= 1 always holds for a valid schedule.
    if schedule.is_complex:
        if len(angles) > 0:

            @cudaq.kernel
            def complex_prep(qubits: cudaq.qview):
                complex_slater_determinant(qubits, orbital_indices, angles,
                                           phases, final_phases, num_electrons)

            return complex_prep

        # Rotation-free complex schedule: occupation plus final phases.
        @cudaq.kernel
        def complex_basis_prep(qubits: cudaq.qview):
            for i in range(num_electrons):
                x(qubits[i])
            for i in range(num_electrons):
                rz(final_phases[i], qubits[i])

        return complex_basis_prep

    if len(angles) > 0:

        @cudaq.kernel
        def real_prep(qubits: cudaq.qview):
            slater_determinant(qubits, orbital_indices, angles, num_electrons)

        return real_prep

    # Rotation-free real schedule: the basis determinant ``|1...10...0>``.
    @cudaq.kernel
    def basis_prep(qubits: cudaq.qview):
        for i in range(num_electrons):
            x(qubits[i])

    return basis_prep


# ============================================================================
# Resource estimation
# ============================================================================


@dataclass(frozen=True)
class GivensResourceEstimate:
    """Lightweight circuit-cost summary for a Givens schedule.

    ``num_spin_orbitals`` and ``num_electrons`` echo the schedule.
    ``num_exp_pauli_calls`` counts the two-qubit ``exp_pauli`` rotations
    (two per Givens rotation); ``num_phase_rotations`` counts the
    single-qubit ``rz`` gates of a complex preparation (one per rotation
    plus one per electron). The proxies are decomposition-independent
    upper bounds, not transpiled gate counts.
    """

    num_spin_orbitals: int
    num_electrons: int
    num_givens_rotations: int
    num_exp_pauli_calls: int
    num_phase_rotations: int
    two_qubit_gate_count_proxy: int
    depth_proxy: int


def estimate_givens_resources(
        schedule: GivensRotationSchedule) -> GivensResourceEstimate:
    """Resource estimate for preparing a schedule's Slater determinant."""
    validate_givens_rotation_schedule(schedule)
    num_rotations = len(schedule.rotations)
    num_exp_pauli_calls = 2 * num_rotations
    num_phase_rotations = (num_rotations + schedule.num_electrons
                           if schedule.is_complex else 0)
    return GivensResourceEstimate(
        num_spin_orbitals=schedule.num_spin_orbitals,
        num_electrons=schedule.num_electrons,
        num_givens_rotations=num_rotations,
        num_exp_pauli_calls=num_exp_pauli_calls,
        num_phase_rotations=num_phase_rotations,
        two_qubit_gate_count_proxy=num_exp_pauli_calls,
        depth_proxy=num_exp_pauli_calls + num_phase_rotations)
