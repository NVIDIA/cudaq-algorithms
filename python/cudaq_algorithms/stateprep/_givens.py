# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Givens-rotation Slater determinant preparation.

Prepares the Slater determinant of an orthonormal orbital-coefficient
matrix ``Q`` (``num_spin_orbitals x num_electrons``) on the
Jordan-Wigner / little-endian qubit layout: the amplitude of basis state
``|S>`` with occupied set ``S`` is ``det(Q[S, :])``, up to a global
phase.

The construction follows Jiang et al., Phys. Rev. Applied 9, 044036
(2018) (arXiv:1711.05395, Sec. III): apply the Givens network that
reduces ``Q`` to the computational-basis determinant, inverted, to
``|1...10...0>``. The strictly adjacent (nearest-neighbor) rotation
scheduling is the linear-depth arrangement of Kivlichan et al., Phys.
Rev. Lett. 120, 110501 (2018) (arXiv:1711.04789) -- and is also what
lets every rotation map onto a contiguous two-qubit ``exp_pauli`` slice
(see the kernel-language constraints below).

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


# ============================================================================
# Orbital rotations: a real orthogonal one-particle matrix as an adjacent
# Givens network applied to both spin sectors
# ============================================================================


@cudaq.kernel
def spatial_givens_rotation(system: cudaq.qview, theta: float, mode: int):
    """Rotate adjacent spatial modes ``mode`` and ``mode + 1`` by ``theta``
    on both spin sectors of the interleaved register.

    Mode ``p`` lives on qubits ``2p`` (up) and ``2p + 1`` (down). On each
    spin the rotation ``exp(theta (a^dag_p a_{p+1} - a^dag_{p+1} a_p))`` is
    an ``XZY`` / ``YZX`` ``exp_pauli`` pair on the contiguous three-qubit
    window starting at the mode's qubit, whose middle qubit carries the
    Jordan-Wigner parity of the other spin. A zero angle is the identity
    without accessing either window, so mode-zero padding is safe even
    on a single spatial mode (two qubits).
    """
    if theta != 0.0:
        up = system[2 * mode:2 * mode + 3]
        exp_pauli(0.5 * theta, up, "XZY")
        exp_pauli(-0.5 * theta, up, "YZX")
        down = system[2 * mode + 1:2 * mode + 4]
        exp_pauli(0.5 * theta, down, "XZY")
        exp_pauli(-0.5 * theta, down, "YZX")


@cudaq.kernel
def spatial_givens_segment(system: cudaq.qview, modes: list[int],
                           angles: list[float], start: int, count: int):
    """Apply ``spatial_givens_rotation(system, angles[i], modes[i])`` for
    ``i`` from ``start`` to ``start + count - 1`` in order: one segment of
    a flat rotation list shared by several frames (a kernel cannot slice a
    sub-list to hand to another kernel)."""
    for i in range(count):
        spatial_givens_rotation(system, angles[start + i], modes[start + i])


@cudaq.kernel
def spatial_givens_network(system: cudaq.qview, modes: list[int],
                           angles: list[float]):
    """Apply ``spatial_givens_rotation(system, angles[i], modes[i])`` for
    each ``i`` in order (the flattened form of an
    ``orbital_rotation_program``)."""
    spatial_givens_segment(system, modes, angles, 0, len(modes))


@dataclass(frozen=True)
class OrbitalRotationSchedule:
    """An adjacent Givens sweep of a real orthogonal matrix ``V``.

    ``rotations`` are ``(mode, theta)`` plane rotations on modes
    ``(mode, mode + 1)`` in elimination order: applied to the rows of
    ``V`` in turn they reduce it to a diagonal of signs,
    ``R_m ... R_1 V = diag(signs)``. ``signs`` is that residual, one
    ``+-1`` per mode: the networks built from the schedule reproduce
    ``V diag(signs)``, i.e. ``V`` up to a sign per mode. That is
    immaterial for operators diagonal in the rotated modes (number
    operators); ``RotatedEncoding.from_orbital_rotation`` cancels it with
    a ``Z`` on both spin orbitals of each mode with sign ``-1``
    (``Z_p a_p Z_p = -a_p`` under Jordan-Wigner), so its frame kernels
    realize ``V`` exactly.
    """

    num_modes: int
    rotations: tuple[tuple[int, float], ...]
    signs: tuple[int, ...]


def make_orbital_rotation_schedule(orthogonal,
                                   tolerance: float = 1.0e-12
                                   ) -> OrbitalRotationSchedule:
    """Decompose a real orthogonal ``n x n`` matrix (target modes as
    columns) into adjacent plane rotations.

    Column by column, bottom-up, each rotation on rows ``(p, p + 1)``
    zeroes the lower entry — the orthogonal-matrix analogue of
    ``make_givens_rotation_schedule``, acting on spatial modes rather than
    spin orbitals. Raises ``ValueError`` for a complex, non-square or
    non-orthonormal input.
    """
    if np.iscomplexobj(orthogonal):
        raise ValueError("orthogonal must be a real matrix")
    work = np.array(orthogonal, dtype=float, copy=True)
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError("orthogonal must be a square matrix")
    n = work.shape[0]
    if not np.allclose(work.T @ work, np.eye(n), atol=tolerance):
        raise ValueError("orthogonal must have orthonormal columns")
    rotations: list[tuple[int, float]] = []
    for column in range(n):
        for row in range(n - 1, column, -1):
            upper, lower = work[row - 1, column], work[row, column]
            if abs(lower) > 1e-14:
                theta = math.atan2(lower, upper)
                c, s = math.cos(theta), math.sin(theta)
                rotation = np.array([[c, s], [-s, c]])
                work[row - 1:row + 1, :] = rotation @ work[row - 1:row + 1, :]
                rotations.append((row - 1, theta))
    signs = tuple(1 if work[i, i] >= 0.0 else -1 for i in range(n))
    if not np.allclose(work, np.diag(signs), atol=1e3 * tolerance):
        raise ValueError("the Givens sweep did not reduce orthogonal to a "
                         "diagonal of signs; is it orthonormal?")
    return OrbitalRotationSchedule(num_modes=n,
                                   rotations=tuple(rotations),
                                   signs=signs)


def orbital_rotation_program(schedule: OrbitalRotationSchedule, *,
                             inverse: bool) -> tuple[list[int], list[float]]:
    """Flattened ``(modes, angles)`` lists for ``spatial_givens_network``.

    With ``inverse=False`` the network realizes ``V diag(signs)`` on the
    one-particle space of each spin: from ``R_m ... R_1 V = D`` follows
    ``V D = R_1^T ... R_m^T``, so the sweep is applied reversed with negated
    angles (the last elimination first in circuit time). With
    ``inverse=True`` it realizes the transpose, ``D V^T``: the sweep in
    elimination order. A diagonal ``V`` has no rotations; the lists are
    then the single zero-angle rotation ``[0], [0.0]`` — an identity on any
    register, including a single spatial mode — so that they are never
    empty when they cross the kernel boundary (cuda-quantum#4847).
    """
    if inverse:
        pairs = list(schedule.rotations)
    else:
        pairs = [(mode, -theta)
                 for mode, theta in reversed(schedule.rotations)]
    if not pairs:
        return [0], [0.0]
    return [mode for mode, _ in pairs], [theta for _, theta in pairs]
