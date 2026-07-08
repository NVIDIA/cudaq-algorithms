# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Quantum singular value transformation over a PauliLCU block encoding.

Provides a ``PhaseSequence`` value type with qsvt/qsp phase-convention
handling built in, the phase/walk sequence kernels (plain and controlled),
and a ``QSVT`` object with the corresponding kernel factories.

Each walk step is the full block encoding (PREPARE, SELECT, PREPARE dagger)
composed with a reflection about the all-zero signal state, and projector
phases ``diag(e^{i phi}, 1)`` act on the same |0...0> signal subspace. The
signal register starts at |0...0>.

The walk block encodes ``-H/alpha``; the circuits fold the sign in, so on an
eigenstate of H with eigenvalue lambda the good-subspace block implements
``p(lambda / alpha)`` — the polynomial defined by the phase sequence at the
plain scaled eigenvalue, with no caller-side negation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import cudaq

from .common_kernels import (controlled_reflect_about_zero,
                             controlled_signal_phase, reflect_about_zero,
                             signal_phase)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    from .block_encoding import BlockEncoding
    from .pauli_lcu import Kernel

FORWARD = 0
ADJOINT = 1

_DIRECTION_CODES = {
    "forward": FORWARD,
    "adjoint": ADJOINT,
    "backward": ADJOINT,
    "reverse": ADJOINT,
    FORWARD: FORWARD,
    ADJOINT: ADJOINT,
}


def _direction_code(direction: int | str) -> int:
    key = direction.lower() if isinstance(direction, str) else direction
    try:
        return _DIRECTION_CODES[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "walk direction must be 'forward', 'adjoint', 0, or 1") from exc


# ============================================================================
# Phase sequences
# ============================================================================


class PhaseSequence:
    """A validated QSVT/QSP phase sequence.

    Parameters
    ----------
    phases
        d + 1 phase angles for a degree-d polynomial.
    walk_directions
        Optional; one direction ('forward'/'adjoint' or 0/1) per walk,
        length d. Defaults to all forward.
    convention
        "qsvt" (projector phases ``diag(e^{i phi}, 1)``, the default) or
        "qsp" (Z-rotation phases ``diag(e^{i phi}, e^{-i phi})``, the
        QSPPACK convention). qsp-tagged phases are converted automatically
        wherever a circuit is built; ``phases`` always stays raw.
    """

    phases: tuple[float, ...]
    walk_directions: tuple[int, ...]
    convention: str

    def __init__(self,
                 phases: Iterable[float],
                 walk_directions: Iterable[int | str] | None = None,
                 convention: str = "qsvt") -> None:
        self.phases = tuple(float(p) for p in phases)
        if not self.phases:
            raise ValueError("phases must contain at least one value")
        if not all(math.isfinite(p) for p in self.phases):
            raise ValueError("phases must be finite")

        convention = str(convention).lower()
        if convention not in ("qsvt", "qsp"):
            raise ValueError("convention must be 'qsvt' or 'qsp'")
        self.convention = convention

        if walk_directions is None:
            self.walk_directions = (FORWARD,) * self.degree
        else:
            self.walk_directions = tuple(
                _direction_code(d) for d in walk_directions)
            if len(self.walk_directions) != self.degree:
                raise ValueError(
                    "walk_directions must contain len(phases) - 1 entries")

    @property
    def degree(self) -> int:
        return len(self.phases) - 1

    @property
    def projector_phases(self) -> list[float]:
        """Phases in the projector convention the circuits implement.

        qsp phases are doubled (equivalent up to a global phase of
        ``exp(i * sum(phases))``; see ``recover_real_time_evolution``).
        """
        if self.convention == "qsp":
            return [2.0 * p for p in self.phases]
        return list(self.phases)

    def __repr__(self) -> str:
        return (f"PhaseSequence(degree={self.degree}, "
                f"convention={self.convention!r})")


def _as_sequence(sequence: PhaseSequence | Iterable[float],
                 convention: str | None = None) -> PhaseSequence:
    if isinstance(sequence, PhaseSequence):
        if convention is not None and convention != sequence.convention:
            return PhaseSequence(sequence.phases, sequence.walk_directions,
                                 convention)
        return sequence
    return PhaseSequence(sequence, convention=convention or "qsvt")


# ============================================================================
# Device kernels
# ============================================================================








# ============================================================================
# The user-facing object
# ============================================================================


class QSVT:
    """Quantum singular value transformation over a block encoding.

    Generic over the ``BlockEncoding`` protocol: encoding-specific
    circuits are delegated to the injected encoding (``PauliLCU`` is the
    provided implementation).
    """

    def __init__(self, encoding: BlockEncoding) -> None:
        if encoding.num_ancilla == 0:
            raise ValueError(
                "QSVT requires an encoding with at least one ancilla "
                "(two or more LCU terms); the 0-ancilla case is degenerate")
        self.encoding = encoding

    def __repr__(self) -> str:
        return f"QSVT({self.encoding!r})"

    def kernel(self,
               sequence: PhaseSequence | Iterable[float],
               convention: str | None = None) -> Kernel:
        """A ``@cudaq.kernel(state)`` applying the phase/walk sequence.

        ``sequence`` may be a PhaseSequence or a plain list of phases
        (optionally with ``convention="qsp"``). The signal register is
        allocated in |0...0> after the system register.
        """
        seq = _as_sequence(sequence, convention)
        phases = seq.projector_phases
        # A degree-0 sequence has no walks; pad with one unused entry because
        # empty list captures cannot be marshaled.
        directions = list(seq.walk_directions) or [FORWARD]
        n_anc = self.encoding.num_ancilla
        u_a = self.encoding.apply_kernel()

        @cudaq.kernel
        def qsvt_kernel(state: cudaq.State):
            system = cudaq.qvector(state)
            signal = cudaq.qvector(n_anc)
            signal_phase(signal, phases[0])
            for i in range(1, len(phases)):
                if directions[i - 1] == 1:
                    reflect_about_zero(signal)
                    u_a(signal, system)
                else:
                    u_a(signal, system)
                    reflect_about_zero(signal)
                signal_phase(signal, phases[i])

        return qsvt_kernel

    def controlled_kernel(self,
                          sequence: PhaseSequence | Iterable[float],
                          convention: str | None = None,
                          control_state: int = 1) -> Kernel:
        """``@cudaq.kernel(state)`` applying the sequence controlled.

        Allocates the system register from ``state``, then one register
        holding [control, signal] (a CUDA-Q Python control set cannot mix a
        bare qubit with a separate register). With control |0> the sequence
        is the identity.
        """
        seq = _as_sequence(sequence, convention)
        phases = seq.projector_phases
        directions = list(seq.walk_directions) or [FORWARD]
        n_anc = self.encoding.num_ancilla
        flip_control = int(control_state) == 1
        controlled_u_a = self.encoding.controlled_apply_kernel()

        @cudaq.kernel
        def controlled_qsvt_kernel(state: cudaq.State):
            system = cudaq.qvector(state)
            control_and_signal = cudaq.qvector(1 + n_anc)
            if flip_control:
                x(control_and_signal[0])
            controlled_signal_phase(control_and_signal, phases[0])
            for i in range(1, len(phases)):
                if directions[i - 1] == 1:
                    controlled_reflect_about_zero(control_and_signal)
                    controlled_u_a(control_and_signal, system)
                else:
                    controlled_u_a(control_and_signal, system)
                    controlled_reflect_about_zero(control_and_signal)
                controlled_signal_phase(control_and_signal, phases[i])

        return controlled_qsvt_kernel


def recover_real_time_evolution(
        cos_state: ArrayLike, sin_state: ArrayLike,
        cos_phases: Sequence[float],
        sin_phases: Sequence[float]) -> NDArray[np.complex128]:
    """Combine cosine/sine QSP components into exp(-i H t)|psi>.

    ``cos_state`` and ``sin_state`` are good-subspace statevectors produced
    by running qsp-convention sequences through the QSVT circuit (which
    executes doubled projector phases); the per-sequence global phase
    ``exp(i * sum(phases))`` is removed here. Valid for real Hamiltonians and
    real input states, where the cosine/sine parts live in the real/imaginary
    components.
    """
    import numpy as np

    cos_state = np.asarray(cos_state, dtype=np.complex128)
    sin_state = np.asarray(sin_state, dtype=np.complex128)
    cos_state = cos_state * np.exp(-1.0j * np.sum(cos_phases))
    sin_state = sin_state * np.exp(-1.0j * np.sum(sin_phases))
    return 2.0 * (cos_state.real + 1.0j * sin_state.imag)
