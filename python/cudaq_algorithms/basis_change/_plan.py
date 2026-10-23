# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable host plans for vacuum-preserving fermionic basis changes.

Columns are orbitals: ``Gamma(U) a_j† Gamma(U)† = sum_p U[p,j] a_p†``.
The chronological tape uses exact number phases ``exp(i alpha n)`` and
real Givens matrices ``[[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]``.
No determinant phase is discarded. Logical mode ``j`` is the Jordan-Wigner
mode on physical qubit ``mode_map[j]``, whose parity string runs over every
lower-numbered physical qubit, mapped or spectator. Many-body determinant
minors therefore take occupied sets in increasing physical-qubit order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Integral, Real

import numpy as np

# Largest accepted qubit index. CUDA-Q 0.16 silently wraps indices near
# 2**61 onto low qubits, so far larger values must be rejected on the host.
_MAX_QUBIT_INDEX = 2**31 - 1


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _index(value, name):
    """Return a nonnegative qubit index within the supported range."""
    index = _integer(value, name)
    if not 0 <= index <= _MAX_QUBIT_INDEX:
        raise ValueError(
            f"{name} must be a qubit index from 0 to {_MAX_QUBIT_INDEX}")
    return index


def _sequence(values, name):
    """Return a tuple from an ordered sequence; reject unordered containers.

    Lists, tuples, ranges and arrays qualify. Sets, mappings, iterators,
    generators and text do not, so hash order never decides a placement.
    """
    if (isinstance(values, (str, bytes, bytearray))
            or not isinstance(values, (Sequence, np.ndarray))):
        raise ValueError(f"{name} must be an ordered sequence")
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered sequence") from exc


def _real_scalar(value):
    """Return ``float(value)`` for a real scalar; raise TypeError otherwise.

    Real numbers, ``Decimal`` and zero-dimensional integer or floating arrays
    are real scalars; bool, text, bytes and complex values are not. Errors
    from the conversion itself, such as OverflowError, propagate.
    """
    if (isinstance(value, np.ndarray) and value.ndim == 0
            and value.dtype.kind in "iuf"):
        value = value[()]
    if (isinstance(value, (bool, np.bool_, str, bytes, bytearray))
            or not isinstance(value, (Real, Decimal))):
        raise TypeError("not a real scalar")
    return float(value)


def _real(value, name):
    """Return a finite float from a real scalar that is not bool or text."""
    try:
        value = _real_scalar(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


@dataclass(frozen=True)
class GivensOperation:
    """One chronological logical operation with an indexed real parameter.

    ``kind`` is ``"phase"``, the number phase ``exp(i*angle*n)`` on mode
    ``first``, or ``"givens"``, the real rotation ``[[c, -s], [s, c]]`` on
    the mode pair ``(first, second)`` with ``c, s = cos(angle), sin(angle)``.
    ``first`` and ``second`` are logical mode indices; the physical qubit of
    mode ``first`` is ``structure.mode_map[first]``. ``second`` is ``None``
    for a phase. ``parameter_index`` indexes the schedule parameters, and the
    applied angle is ``scale * parameters[parameter_index]``: ``scale = -1``
    marks the leading ``P_b(-phi)`` of a complex mixing block.
    """

    kind: str
    first: int
    second: int | None
    parameter_index: int
    scale: float = 1.0


@dataclass(frozen=True)
class GivensStructure:
    """Static topology and parameter schema, independent of numerical values.

    Full schemas start with ``m`` diagonal phases, followed by a ``theta,
    phi`` pair per chronological mixing slot. Partial schemas contain
    ``m-1`` mixing angles followed by ``m`` output number phases. All slots
    remain present for zero angles and real input data. ``num_modes`` must
    be at least two: a single mode has no mixing and admits only a number
    phase, which callers apply directly with ``r1``. ``mode_map`` is an
    ordered sequence (list, tuple, range or one-dimensional array) of
    distinct physical qubit indices, one per logical mode, each from 0 to
    ``2**31 - 1``.
    """

    #: Number of logical fermionic modes, at least two.
    num_modes: int
    #: Schedule kind, either ``"full"`` or ``"partial"``.
    kind: str
    #: Physical qubit of each logical mode; defaults to ``0 .. m-1``.
    mode_map: tuple[int, ...] | None = None
    #: Logical reference mode of a partial schedule, otherwise None.
    reference_mode: int | None = None

    def __post_init__(self):
        m = _integer(self.num_modes, "num_modes")
        if m < 2:
            raise ValueError(
                "num_modes must be at least 2; a single mode admits only a "
                "number phase, so apply r1 directly")
        if self.kind not in ("full", "partial"):
            raise ValueError("kind must be 'full' or 'partial'")
        mapping = (tuple(range(m)) if self.mode_map is None else tuple(
            _index(i, "mode_map index")
            for i in _sequence(self.mode_map, "mode_map")))
        if len(mapping) != m or len(set(mapping)) != m:
            raise ValueError(
                "mode_map must contain num_modes distinct nonnegative indices")
        reference = self.reference_mode
        if self.kind == "partial":
            reference = _integer(0 if reference is None else reference,
                                 "reference_mode")
            if reference < 0 or reference >= m:
                raise ValueError("reference_mode must index a logical mode")
        elif reference is not None:
            raise ValueError(
                "reference_mode is only defined for partial structures")
        object.__setattr__(self, "num_modes", m)
        object.__setattr__(self, "mode_map", mapping)
        object.__setattr__(self, "reference_mode", reference)

    def __reduce__(self):
        # Rebuild through the constructor, so unpickling validates again.
        return (type(self), (self.num_modes, self.kind, self.mode_map,
                             self.reference_mode))

    @classmethod
    def full(cls, num_modes, *, mode_map=None):
        """Construct the fixed adjacent QR topology for a square unitary."""
        return cls(num_modes, "full", mode_map)

    @classmethod
    def partial(cls, num_modes, *, mode_map=None, reference_mode=0):
        """Construct a magnitude-distribution chain rooted at reference_mode."""
        return cls(num_modes, "partial", mode_map, reference_mode)

    @property
    def num_system_qubits(self):
        """Minimum physical register width containing every mapped mode."""
        return max(self.mode_map) + 1

    @property
    def mode_pairs(self):
        """Logical mixing pairs in chronological application order."""
        if self.kind == "full":
            elimination_pairs = tuple(
                (row - 1, row) for col in range(self.num_modes - 1)
                for row in range(self.num_modes - 1, col, -1))
            return elimination_pairs[::-1]
        reference = self.reference_mode
        return (tuple(
            (i, i + 1) for i in range(reference - 1, -1, -1)) + tuple(
                (i, i + 1) for i in range(reference, self.num_modes - 1)))

    @property
    def parameter_names(self):
        """Stable names in the numerical parameter-vector order."""
        if self.kind == "full":
            return (tuple(f"diagonal_phase[{i}]"
                          for i in range(self.num_modes)) +
                    tuple(name for k in range(len(self.mode_pairs))
                          for name in (f"theta[{k}]", f"phi[{k}]")))
        return (tuple(f"theta[{k}]" for k in range(self.num_modes - 1)) +
                tuple(f"phase[{i}]" for i in range(self.num_modes)))

    @property
    def parameter_schema(self):
        """Immutable ordered parameter names; every parameter is a real angle."""
        return self.parameter_names

    @property
    def num_parameters(self):
        """Real parameter slot count: ``m**2`` full, ``2*m - 1`` partial."""
        return self.num_modes**2 if self.kind == "full" else 2 * self.num_modes - 1

    @property
    def operations(self):
        """Exact number phases and real mixing gates in chronological order."""
        if self.kind == "partial":
            return (tuple(
                GivensOperation("givens", a, b, k)
                for k, (a, b) in enumerate(self.mode_pairs)) + tuple(
                    GivensOperation("phase", i, None, self.num_modes - 1 + i)
                    for i in range(self.num_modes)))
        operations = [
            GivensOperation("phase", i, None, i) for i in range(self.num_modes)
        ]
        for k, (a, b) in enumerate(self.mode_pairs):
            theta = self.num_modes + 2 * k
            # P_b(phi) G(theta) P_b(-phi) has the required complex entries.
            operations.extend(
                (GivensOperation("phase", b, None, theta + 1,
                                 -1.), GivensOperation("givens", a, b, theta),
                 GivensOperation("phase", b, None, theta + 1)))
        return tuple(operations)


def _parameters(values, count):
    """Return a finite float tuple from a one-dimensional real vector.

    Integer and floating arrays convert as a whole. Other vectors, including
    object arrays, convert entry by entry, so a bool, text, bytes or complex
    entry is rejected even where numpy would upcast it to a number.
    """
    message = "parameters must be a one-dimensional real vector"
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    numeric = isinstance(values, np.ndarray) and values.dtype.kind in "iuf"
    try:
        # Object entries keep their own types, so True never becomes 1.0.
        array = np.asarray(values, dtype=None if numeric else object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if array.shape != (count, ):
        raise ValueError(message)
    try:
        result = (array.astype(float).tolist()
                  if numeric else [_real_scalar(value) for value in array])
    except TypeError as exc:
        raise ValueError(message) from exc
    except (ValueError, OverflowError) as exc:
        raise ValueError("parameters must be finite") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError("parameters must be finite")
    return tuple(result)


@dataclass(frozen=True)
class GivensBasisChangeSchedule:
    """Immutable static tape with optional numerical angle values.

    ``reconstruction_error`` is the maximum absolute orbital-matrix entry
    error (the specified column only for partial input), a finite
    nonnegative diagnostic. It is ``None`` for structure-only plans and
    takes no part in equality or hashing, which depend only on
    ``structure`` and ``parameters``. No input or output ndarray is retained.
    ``GivensBasisChangeSchedule(schedule.structure)`` is the structure-only
    form of any schedule. ``dataclasses.replace(schedule, parameters=None)``
    raises for a schedule with a ``reconstruction_error`` unless
    ``reconstruction_error=None`` is also passed, and replacing the
    parameters keeps the old residual, so recompile instead when the
    angles change.
    """

    #: Static topology and parameter schema of the tape.
    structure: GivensStructure
    #: Host angles in radians in parameter order, or None if structure-only.
    parameters: tuple[float, ...] | None = None
    #: Maximum orbital-matrix entry residual, or None if structure-only.
    reconstruction_error: float | None = field(default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.structure, GivensStructure):
            raise ValueError("structure must be a GivensStructure")
        if self.parameters is not None:
            object.__setattr__(
                self, "parameters",
                _parameters(self.parameters, self.num_parameters))
        error = self.reconstruction_error
        if error is not None:
            if self.parameters is None:
                raise ValueError(
                    "reconstruction_error requires numerical parameters")
            error = _real(error, "reconstruction_error")
            if error < 0:
                raise ValueError("reconstruction_error must be nonnegative")
            object.__setattr__(self, "reconstruction_error", error + 0.)

    def __reduce__(self):
        # Rebuild through the constructor, so unpickling validates again.
        return (type(self), (self.structure, self.parameters,
                             self.reconstruction_error))

    @property
    def operations(self):
        """Chronological :class:`GivensOperation` tape of the structure."""
        return self.structure.operations

    @property
    def parameter_names(self):
        """Stable names in the numerical parameter-vector order."""
        return self.structure.parameter_names

    @property
    def num_parameters(self):
        """Number of real parameter slots in the schedule."""
        return self.structure.num_parameters

    def orbital_matrix(self, parameters=None):
        """Reconstruct the complete orbital unitary using optional overrides.

        A partial plan specifies only its reference column; the other columns
        are a deterministic unitary completion. The returned array is fresh.
        Structure-only plans require an explicit parameter vector.
        """
        values = self.parameters if parameters is None else parameters
        if values is None:
            raise ValueError("a structure-only plan requires parameters")
        values = _parameters(values, self.num_parameters)
        matrix = np.eye(self.structure.num_modes, dtype=complex)
        for op in self.operations:
            angle = values[op.parameter_index] * op.scale
            if op.kind == "phase":
                matrix[op.first] *= np.exp(1j * angle)
            else:
                a, b = op.first, op.second
                upper, lower = matrix[a].copy(), matrix[b].copy()
                c, s = math.cos(angle), math.sin(angle)
                matrix[a] = c * upper - s * lower
                matrix[b] = s * upper + c * lower
        return matrix


def _validate_tolerance(tolerance):
    """Return a real tolerance in (0, 1) shared by planner and factory."""
    tolerance = _real(tolerance, "tolerance")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must lie strictly between 0 and 1")
    return tolerance


def _argument(value):
    """Canonical arg(value) in (-pi, pi]: arg(0)=0, zero signs are ignored."""
    return 0. if value == 0 else float(np.arctan2(value.imag + 0., value.real))


def _full_parameters(matrix):
    work = matrix.copy()
    eliminations = []
    m = len(work)
    for col in range(m - 1):
        for row in range(m - 1, col, -1):
            upper, lower = work[row - 1, col], work[row, col]
            if lower == 0:
                eliminations.append((0., 0.))
                continue
            # Exact coefficients keep structural zeros exactly zero, so no
            # later pivot takes its gauge phase from rounding noise.
            au, al = abs(upper), abs(lower)
            # Scale first: hypot of two subnormals rounds to subnormal
            # precision, which would leave (c, s) unnormalised.
            big = max(au, al)
            r = math.hypot(au / big, al / big)
            c, s = au / big / r, al / big / r
            phi = _argument(lower) - _argument(upper)
            if upper.imag == 0 and lower.imag == 0:
                # Real pivots get an exact +-1 factor. Others use exp(i*phi)
                # because z / abs(z) overflows for subnormal z.
                rel = (math.copysign(1., lower.real) *
                       (math.copysign(1., upper.real) if au else 1.))
            else:
                rel = np.exp(1j * phi)
            a, b = work[row - 1].copy(), work[row].copy()
            work[row - 1] = c * a + s * np.conj(rel) * b
            work[row] = -s * rel * a + c * b
            eliminations.append((math.atan2(al, au), phi))
    return (tuple(_argument(work[i, i])
                  for i in range(m)) + tuple(value
                                             for pair in reversed(eliminations)
                                             for value in pair))


def _partial_parameters(vector, structure):
    magnitudes = np.abs(vector)
    reference = structure.reference_mode
    angles = []
    # Distribute the left subtree first, leaving the right subtree at root.
    for i in range(reference - 1, -1, -1):
        remaining = (float(np.linalg.norm(magnitudes[reference:]))
                     if i == reference - 1 else float(magnitudes[i + 1]))
        left = float(np.linalg.norm(magnitudes[:i + 1]))
        # Subtracting from zero never yields a -0.0 parameter.
        angles.append(0. - math.atan2(left, remaining))
    for i in range(reference, structure.num_modes - 1):
        angles.append(
            math.atan2(float(np.linalg.norm(magnitudes[i + 1:])),
                       float(magnitudes[i])))
    return tuple(angles) + tuple(_argument(value) for value in vector)


def compile_givens_basis_change(basis=None,
                                *,
                                structure=None,
                                tolerance=1e-12):
    """Compile a square unitary or a normalized orbital column on the host.

    ``basis`` may have shape ``(m,m)``, ``(m,)`` or ``(m,1)`` with
    ``m >= 2``. ``basis=None`` requires a structure and allocates the same
    tape and parameter schema without numerical values. Validation uses the
    absolute ``tolerance``, with ``0 < tolerance < 1``, for the column norm
    or maximum entry of U†U-I, and rejects inputs outside it. Accepted input
    is compiled from its elimination angles, so an inexact input is replaced
    by the unitary (or unit column) those angles define.
    ``reconstruction_error`` reports the maximum entry discrepancy; check it
    when loosening ``tolerance``.
    """
    tolerance = _validate_tolerance(tolerance)
    if structure is not None and not isinstance(structure, GivensStructure):
        raise ValueError("structure must be a GivensStructure")
    if basis is None:
        if structure is None:
            raise ValueError("basis or structure must be supplied")
        return GivensBasisChangeSchedule(structure)
    try:
        array = np.asarray(basis, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "basis must be a square matrix or column vector") from exc
    if array.ndim not in (1, 2) or not array.size or not np.all(
            np.isfinite(array)):
        raise ValueError(
            "basis must be a finite nonempty square matrix or column vector")
    m = array.shape[0]
    is_full = array.ndim == 2 and array.shape == (m, m)
    is_partial = array.ndim == 1 or array.shape == (m, 1)
    if not (is_full or is_partial):
        raise ValueError("basis must be a square matrix or column vector")
    if structure is None:
        structure = GivensStructure.full(
            m) if is_full else GivensStructure.partial(m)
    if (structure.num_modes != m or (structure.kind == "full" and not is_full)
            or (structure.kind == "partial" and not is_partial)):
        raise ValueError("basis shape is inconsistent with structure")
    if structure.kind == "full":
        with np.errstate(over="ignore", invalid="ignore"):
            gram_error = float(
                np.max(np.abs(array.conj().T @ array - np.eye(m))))
        if not math.isfinite(gram_error) or gram_error > tolerance:
            raise ValueError("basis must be unitary within tolerance")
        parameters = _full_parameters(array)
        target = array
    else:
        target = array.reshape(m)
        if abs(np.linalg.norm(target) - 1.) > tolerance:
            raise ValueError(
                "basis column must be normalized within tolerance")
        parameters = _partial_parameters(target, structure)
    schedule = GivensBasisChangeSchedule(structure, parameters)
    reconstructed = schedule.orbital_matrix()
    if structure.kind == "partial":
        reconstructed = reconstructed[:, structure.reference_mode]
    error = float(np.max(np.abs(reconstructed - target)))
    return GivensBasisChangeSchedule(structure, parameters, error)
