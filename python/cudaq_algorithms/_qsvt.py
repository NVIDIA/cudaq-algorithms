# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from dataclasses import dataclass

from . import QSVTPhaseConvention, qsvt

_FORWARD_WALK = 0
_ADJOINT_WALK = 1


@dataclass(frozen=True)
class _PhaseSequence:
    """Python-facing QSVT phase sequence data.

    The CUDA-Q kernels consume plain phase and walk-direction arrays. This
    helper keeps those arrays together without exposing the lower-level C++
    planning structs in the public Python API.
    """

    phases: tuple
    walk_directions: tuple
    convention: object

    @property
    def degree(self):
        return len(self.phases) - 1

    @property
    def phase_data(self):
        return list(self.phases)

    @property
    def walk_direction_data(self):
        return list(self.walk_directions)

    def projector_phase_data(self):
        """Return the phases in the projector convention used by the kernels.

        The device kernels (apply_phase_sequence) only implement projector
        phases diag(exp(i*phi), 1). A qsp-tagged sequence stores Z-rotation
        phases diag(exp(i*phi), exp(-i*phi)), which are equivalent to
        projector phases 2*phi up to a global phase, so they are converted
        here. phase_data stays raw for host-side helpers like phases_to_poly
        that honor the convention themselves.
        """
        if self.convention == QSVTPhaseConvention.qsp:
            return _projector_phases_from_qsp(self.phases)
        return self.phase_data

    def kernel_data(self):
        return {
            "phases": self.projector_phase_data(),
            "walk_directions": self.walk_direction_data,
        }


@dataclass(frozen=True)
class _PolyError:
    """Sampled error between a QSVT polynomial and a target function."""

    max_abs_error: float
    rms_error: float
    max_error_x: float
    num_samples: int


def _phase_convention(convention):
    if convention is None:
        return QSVTPhaseConvention.qsvt
    if isinstance(convention, str):
        try:
            return getattr(QSVTPhaseConvention, convention.lower())
        except AttributeError as exc:
            raise ValueError(
                "convention must be 'qsvt', 'qsp', or a QSVTPhaseConvention"
            ) from exc
    return convention


def _walk_direction_code(direction):
    if isinstance(direction, str):
        key = direction.lower()
        if key == "forward":
            return _FORWARD_WALK
        if key in ("adjoint", "backward", "reverse"):
            return _ADJOINT_WALK
    if direction in (_FORWARD_WALK, _ADJOINT_WALK):
        return int(direction)
    raise ValueError("walk direction must be 'forward', 'adjoint', 0, or 1")


def _forward_walk_directions(degree):
    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return [_FORWARD_WALK] * degree


def _alternating_walk_directions(degree, first="forward"):
    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    first_code = _walk_direction_code(first)
    second_code = _ADJOINT_WALK if first_code == _FORWARD_WALK else _FORWARD_WALK
    return [first_code if i % 2 == 0 else second_code for i in range(degree)]


def _phase_sequence(phases, walk_directions=None, convention=None):
    if isinstance(phases, _PhaseSequence):
        if walk_directions is not None:
            raise ValueError(
                "walk_directions cannot override an existing PhaseSequence")
        if convention is None or _phase_convention(
                convention) == phases.convention:
            return phases
        return _PhaseSequence(phases.phases, phases.walk_directions,
                              _phase_convention(convention))

    phase_data = tuple(float(phase) for phase in phases)
    if len(phase_data) == 0:
        raise ValueError("phases must contain at least one value")

    degree = len(phase_data) - 1
    if walk_directions is None:
        direction_data = tuple(_forward_walk_directions(degree))
    else:
        direction_data = tuple(
            _walk_direction_code(direction) for direction in walk_directions)
        if len(direction_data) != degree:
            raise ValueError(
                "walk_directions must contain len(phases) - 1 entries")

    return _PhaseSequence(phase_data, direction_data,
                          _phase_convention(convention))


def _pauli_lcu_kernel_tuple(kernel_data):
    """Return the flattened PauliLCU arrays needed by device interop helpers."""

    if hasattr(kernel_data, "unpack"):
        return tuple(kernel_data.unpack())
    if hasattr(kernel_data, "as_tuple"):
        return tuple(kernel_data.as_tuple())
    if isinstance(kernel_data, dict):
        return (kernel_data["angles"], kernel_data["term_controls"],
                kernel_data["term_ops"], kernel_data["term_lengths"],
                kernel_data["term_signs"])
    if isinstance(kernel_data, (tuple, list)) and len(kernel_data) == 5:
        return tuple(kernel_data)
    return (kernel_data.angles, kernel_data.term_controls,
            kernel_data.term_ops, kernel_data.term_lengths,
            kernel_data.term_signs)


def _pauli_lcu_sequence_kernel_args(phases,
                                    kernel_data,
                                    walk_directions=None,
                                    convention=None):
    """Pack QSVT phase data and PauliLCU layout for apply_phase_sequence().

    Phases from a qsp-tagged sequence (or a raw list with convention="qsp")
    are converted to the projector convention the kernels implement; see
    _PhaseSequence.projector_phase_data.
    """

    sequence = _phase_sequence(phases, walk_directions, convention)
    angles, term_controls, term_ops, term_lengths, term_signs = (
        _pauli_lcu_kernel_tuple(kernel_data))
    return (sequence.projector_phase_data(), sequence.walk_direction_data,
            list(angles), list(term_controls), list(term_ops),
            list(term_lengths), list(term_signs))


def _projector_phases_from_qsp(phases):
    """Convert QSPPACK/QSP phases to projector phases used by QSVT kernels."""

    return [2.0 * float(phase) for phase in phases]


def _recover_real_time_evolution(cos_state, sin_state, cos_qsp_phases,
                                 sin_qsp_phases):
    """Recover exp(-iHt)|psi> from cosine/sine QSPPACK QSVT components.

    This helper is intended for simulation validation workflows that already
    extracted the good-subspace statevectors. Hardware-oriented workflows
    should estimate observables or probabilities instead of calling get_state().
    """

    import numpy as np

    cos_state = np.asarray(cos_state, dtype=np.complex128)
    sin_state = np.asarray(sin_state, dtype=np.complex128)
    cos_state = cos_state * np.exp(-1.0j * np.sum(cos_qsp_phases))
    sin_state = sin_state * np.exp(-1.0j * np.sum(sin_qsp_phases))
    return 2.0 * (cos_state.real + 1.0j * sin_state.imag)


_cpp_phases_to_poly = qsvt.phases_to_poly
_cpp_estimate_poly_error = qsvt.estimate_poly_error


def _phases_to_poly(phases, convention=None):
    if isinstance(phases, _PhaseSequence):
        if convention is None:
            convention = phases.convention
        phases = phases.phase_data
    else:
        phases = list(phases)
    return _cpp_phases_to_poly(phases, _phase_convention(convention))


def _estimate_poly_error(poly, target, domain=(-1.0, 1.0), num_points=101):
    error = _cpp_estimate_poly_error(poly, target, domain, num_points)
    return _PolyError(max_abs_error=float(error.max_abs_error),
                      rms_error=float(error.rms_error),
                      max_error_x=float(error.max_error_x),
                      num_samples=int(error.num_samples))


qsvt.FORWARD = _FORWARD_WALK
qsvt.ADJOINT = _ADJOINT_WALK
qsvt.PhaseConvention = QSVTPhaseConvention
qsvt.PhaseSequence = _PhaseSequence
qsvt.PolyError = _PolyError
qsvt.phase_sequence = _phase_sequence
qsvt.forward_walk_directions = _forward_walk_directions
qsvt.alternating_walk_directions = _alternating_walk_directions
qsvt.pauli_lcu_kernel_args = _pauli_lcu_sequence_kernel_args
qsvt.projector_phases_from_qsp = _projector_phases_from_qsp
qsvt.recover_real_time_evolution = _recover_real_time_evolution
qsvt.phases_to_poly = _phases_to_poly
qsvt.estimate_poly_error = _estimate_poly_error
