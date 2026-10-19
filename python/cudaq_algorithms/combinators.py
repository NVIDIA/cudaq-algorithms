# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combinators that build new ``BlockEncoding`` objects out of existing ones.

Two constructions, both generic over the ``BlockEncoding`` protocol (any
conforming encoding can be an operand) and both producing an object that
satisfies the protocol again, so ``Walk`` and ``QSVT`` consume the result
unchanged and the combinators nest:

``SumEncoding``
    Encodes ``sum_i c_i H_i`` from encodings of the ``H_i`` with real
    coefficients ``c_i``; ``alpha = sum_i |c_i| alpha_i``. One extra *slot*
    register selects the summand, one *flag* qubit drives the selected
    summand's own controlled block encoding, and the summands' ancillas
    share one pool: SELECT is block-diagonal over the slot register, each
    block a unitary on the whole pool, and the encoded block is the
    all-zero subspace of the entire ancilla register. A single summand is
    the "scale by a constant" case.

``RotatedEncoding``
    Encodes ``V H V^dagger`` from an encoding of ``H`` and a pair of
    system-side kernels implementing ``V^dagger`` (enter) and ``V`` (exit).
    Because the rotation touches only the system register it commutes with
    every ancilla-side operation, so each protocol kernel of the inner
    encoding is simply sandwiched between the two rotations; the controlled
    variants come for free.

The payoff is choosing *where* each piece is cheap to encode: a hopping
term is diagonal in momentum space (``alpha_T = sum_k |eps_k|``) while the
interaction is diagonal in real space, and the sum of the two encodings
has a smaller ``alpha`` than one flat Pauli LCU of the whole Hamiltonian
(see ``docs/sphinx/examples/python/09_hubbard_combined_encoding.py``).

What is **not** provided: odd-moment ``select_observable`` hooks. The sum's
SELECT drives each summand's full block encoding, PREPARE trees included,
so it has no compact ``cudaq.SpinOperator`` form; a rotated encoding's
terms are not computational-frame Pauli words. Even moments
(``Walk.moment`` with even order), ``Walk.kernel`` statevectors at any
power, and QSVT sequences are all available; ``Walk.moments``, which
alternates even and odd orders, is not.

Every operand's ``apply_kernel`` must be Hermitian (hence an involution),
as the protocol requires and every encoding in this package satisfies;
``SumEncoding`` builds its SELECT from the operands' ``U_A`` and relies on
it for walk powers ``>= 2``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

import cudaq

from .common_kernels import (_real_coefficient, controlled_reflect_about_zero,
                             reflect_about_zero)
from .pauli_lcu import _prepare_angles, prepare, unprepare

if TYPE_CHECKING:
    from .block_encoding import BlockEncoding, Kernel

__all__ = ["RotatedEncoding", "SumEncoding"]

_PROTOCOL_MEMBERS = ("num_system", "num_ancilla", "alpha", "prepare_kernel",
                     "unprepare_kernel", "apply_kernel",
                     "controlled_apply_kernel", "walk_step_kernel",
                     "adjoint_walk_step_kernel", "controlled_walk_step_kernel",
                     "controlled_adjoint_walk_step_kernel",
                     "select_observable")


def _require_encoding(value, label: str):
    missing = [m for m in _PROTOCOL_MEMBERS if not hasattr(value, m)]
    if missing:
        raise TypeError(f"{label} does not satisfy the BlockEncoding protocol "
                        f"(missing {', '.join(missing)})")
    if int(value.num_ancilla) < 1:
        raise ValueError(f"{label} must have at least one ancilla qubit")
    if not (float(value.alpha) > 0.0 and math.isfinite(float(value.alpha))):
        raise ValueError(f"{label} must have a positive finite alpha")


def _chain(kernels: Sequence) -> "Kernel":
    """One ``(ancilla, system)`` kernel applying ``kernels`` in order."""
    chained = kernels[0]
    for following in kernels[1:]:
        chained = _pair(chained, following)
    return chained


def _pair(first, second):

    @cudaq.kernel
    def both(ancilla: cudaq.qview, system: cudaq.qview):
        first(ancilla, system)
        second(ancilla, system)

    return both


# ============================================================================
# SumEncoding
# ============================================================================


class SumEncoding:
    """Block encoding of ``sum_i c_i H_i`` from encodings of the ``H_i``.

    Ancilla layout is ``[slot (num_slots qubits)][flag][shared pool]`` with
    ``num_slots = max(1, ceil(log2 K))`` and a pool as wide as the widest
    summand, so ``num_ancilla = num_slots + 1 + max_i num_ancilla_i``.
    PREPARE puts amplitude ``sqrt(|c_i| alpha_i / alpha)`` on slot ``|i>``
    (the same binary rotation tree ``PauliLCU`` uses); SELECT computes
    ``flag <- (slot == i)``, phases the branch by ``-1`` when ``c_i < 0``,
    and lets the flag drive summand ``i``'s ``controlled_apply_kernel`` on
    the contiguous ``[flag | pool]`` view. SELECT is block-diagonal over the
    slot register, each block a unitary on the whole pool, so summands of
    different widths share the pool and the encoded block is the all-zero
    subspace of the entire ancilla register, which the walk's reflection
    covers.

    Each summand's ``apply_kernel`` must be Hermitian, hence an involution
    (``PREPARE-dagger SELECT PREPARE`` with an involutory SELECT is; so is
    every encoding in this package and every combinator of them). Then each
    slot factor and SELECT itself are involutions and the standard walk
    construction applies. A protocol-conforming encoding that is not
    Hermitian still yields the right block and first walk step, but wrong
    walk powers ``>= 2`` and QSVT sequences; the protocol cannot check
    this, so verify ``U_A^2 = I`` for a foreign encoding before summing it.

    Parameters
    ----------
    terms
        ``[(c_1, encoding_1), ..., (c_K, encoding_K)]`` with real, nonzero,
        finite coefficients and ``BlockEncoding`` operands acting on the
        same number of system qubits. ``K >= 1``.
    """

    def __init__(self, terms: Iterable[tuple[float, "BlockEncoding"]]):
        pairs = list(terms)
        if not pairs:
            raise ValueError("SumEncoding needs at least one (coefficient, "
                             "encoding) term")
        coefficients: list[float] = []
        summands: list = []
        for index, pair in enumerate(pairs):
            try:
                coefficient, encoding = pair
            except (TypeError, ValueError) as exc:
                raise TypeError("each term must be a (coefficient, encoding) "
                                "pair") from exc
            if (isinstance(coefficient, (bool, np.bool_))
                    or not isinstance(coefficient, numbers.Number)):
                raise TypeError(f"term {index}: coefficient must be a real "
                                f"number, got {coefficient!r}")
            try:
                coefficient = _real_coefficient(coefficient)
            except ValueError as exc:
                raise ValueError(f"term {index}: {exc}") from exc
            if not math.isfinite(coefficient) or coefficient == 0.0:
                raise ValueError(f"term {index}: coefficient must be finite "
                                 "and nonzero")
            _require_encoding(encoding, f"term {index}")
            coefficients.append(coefficient)
            summands.append(encoding)

        num_system = int(summands[0].num_system)
        for index, encoding in enumerate(summands[1:], start=1):
            if int(encoding.num_system) != num_system:
                raise ValueError(
                    f"term {index} acts on {encoding.num_system} system "
                    f"qubits, term 0 on {num_system}")

        self._coefficients = tuple(coefficients)
        self._summands = tuple(summands)
        self._num_system = num_system
        self._pool = max(int(e.num_ancilla) for e in summands)
        self._num_slots = max(1, (len(summands) - 1).bit_length())
        weights = [
            abs(c) * float(e.alpha) for c, e in zip(coefficients, summands)
        ]
        self._alpha = float(sum(weights))
        padded = weights + [0.0] * ((1 << self._num_slots) - len(weights))
        self._angles = _prepare_angles([w / self._alpha for w in padded])

    # ------------------------------------------------------------------
    # Geometry and data
    # ------------------------------------------------------------------

    @property
    def num_system(self) -> int:
        return self._num_system

    @property
    def num_ancilla(self) -> int:
        """``num_slots + 1 (flag) + shared pool``."""
        return self._num_slots + 1 + self._pool

    @property
    def num_slots(self) -> int:
        """Width of the slot register selecting the summand."""
        return self._num_slots

    @property
    def alpha(self) -> float:
        """``sum_i |c_i| alpha_i``."""
        return self._alpha

    @property
    def coefficients(self) -> tuple[float, ...]:
        return self._coefficients

    @property
    def summands(self) -> tuple:
        return self._summands

    def __repr__(self) -> str:
        return (f"SumEncoding(num_terms={len(self._summands)}, "
                f"num_system={self._num_system}, "
                f"num_ancilla={self.num_ancilla}, alpha={self._alpha:.6g})")

    # ------------------------------------------------------------------
    # PREPARE over the slot register
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> "Kernel":
        """``(ancilla: qview)``: PREPARE the slot register."""
        angles = self._angles
        m = self._num_slots

        @cudaq.kernel
        def prepare_slots(ancilla: cudaq.qview):
            prepare(ancilla.front(m), angles)

        return prepare_slots

    def unprepare_kernel(self) -> "Kernel":
        """``(ancilla: qview)``: PREPARE dagger on the slot register."""
        angles = self._angles
        m = self._num_slots

        @cudaq.kernel
        def unprepare_slots(ancilla: cudaq.qview):
            unprepare(ancilla.front(m), angles)

        return unprepare_slots

    # ------------------------------------------------------------------
    # SELECT: one flag-driven factor per slot
    # ------------------------------------------------------------------

    def _slot_kernel(self, index: int):
        """``(ancilla, system)``: summand ``index`` applied where slot == index."""
        m = self._num_slots
        negative = self._coefficients[index] < 0.0
        controlled = self._summands[index].controlled_apply_kernel()
        width = 1 + int(self._summands[index].num_ancilla)
        # Slot qubit b holds bit (m - 1 - b) of the slot value (qubit 0 is
        # the most significant), matching the PREPARE tree's leaf order.
        zero_bits = [b for b in range(m) if ((index >> (m - 1 - b)) & 1) == 0]
        if not zero_bits:
            zero_bits = [-1]  # empty lists cannot cross the kernel boundary

        @cudaq.kernel
        def slot(ancilla: cudaq.qview, system: cudaq.qview):
            for b in zero_bits:
                if b >= 0:
                    x(ancilla[b])
            x.ctrl(ancilla.front(m), ancilla[m])
            for b in zero_bits:
                if b >= 0:
                    x(ancilla[b])
            if negative:
                z(ancilla[m])
            controlled(ancilla[m:m + width], system)
            for b in zero_bits:
                if b >= 0:
                    x(ancilla[b])
            x.ctrl(ancilla.front(m), ancilla[m])
            for b in zero_bits:
                if b >= 0:
                    x(ancilla[b])

        return slot

    def _controlled_slot_kernel(self, index: int):
        """Layout ``[control][slot][flag][pool]``: flag <- control AND (slot == index)."""
        m = self._num_slots
        negative = self._coefficients[index] < 0.0
        controlled = self._summands[index].controlled_apply_kernel()
        width = 1 + int(self._summands[index].num_ancilla)
        zero_bits = [
            1 + b for b in range(m) if ((index >> (m - 1 - b)) & 1) == 0
        ]
        if not zero_bits:
            zero_bits = [-1]

        @cudaq.kernel
        def controlled_slot(control_and_ancilla: cudaq.qview,
                            system: cudaq.qview):
            for b in zero_bits:
                if b >= 0:
                    x(control_and_ancilla[b])
            x.ctrl(control_and_ancilla.front(1 + m),
                   control_and_ancilla[1 + m])
            for b in zero_bits:
                if b >= 0:
                    x(control_and_ancilla[b])
            if negative:
                z(control_and_ancilla[1 + m])
            controlled(control_and_ancilla[1 + m:1 + m + width], system)
            for b in zero_bits:
                if b >= 0:
                    x(control_and_ancilla[b])
            x.ctrl(control_and_ancilla.front(1 + m),
                   control_and_ancilla[1 + m])
            for b in zero_bits:
                if b >= 0:
                    x(control_and_ancilla[b])

        return controlled_slot

    def _select_kernel(self):
        return _chain(
            [self._slot_kernel(i) for i in range(len(self._summands))])

    def _controlled_select_kernel(self):
        return _chain([
            self._controlled_slot_kernel(i) for i in range(len(self._summands))
        ])

    # ------------------------------------------------------------------
    # Protocol kernels
    # ------------------------------------------------------------------

    def apply_kernel(self) -> "Kernel":
        """``(ancilla, system)``: PREPARE, SELECT, PREPARE dagger."""
        angles = self._angles
        m = self._num_slots
        select = self._select_kernel()

        @cudaq.kernel
        def apply(ancilla: cudaq.qview, system: cudaq.qview):
            prepare(ancilla.front(m), angles)
            select(ancilla, system)
            unprepare(ancilla.front(m), angles)

        return apply

    def controlled_apply_kernel(self) -> "Kernel":
        """``(control_and_ancilla, system)``: the block encoding controlled
        by qubit 0. The PREPARE pair stays uncontrolled: it cancels when
        the control is ``|0>`` because the controlled SELECT is the identity."""
        angles = self._angles
        m = self._num_slots
        controlled_select = self._controlled_select_kernel()

        @cudaq.kernel
        def controlled_apply(control_and_ancilla: cudaq.qview,
                             system: cudaq.qview):
            prepare(control_and_ancilla[1:1 + m], angles)
            controlled_select(control_and_ancilla, system)
            unprepare(control_and_ancilla[1:1 + m], angles)

        return controlled_apply

    def walk_step_kernel(self) -> "Kernel":
        """``(ancilla, system)``: SELECT, then reflect about PREPARE."""
        angles = self._angles
        m = self._num_slots
        select = self._select_kernel()

        @cudaq.kernel
        def walk(ancilla: cudaq.qview, system: cudaq.qview):
            select(ancilla, system)
            unprepare(ancilla.front(m), angles)
            reflect_about_zero(ancilla)
            prepare(ancilla.front(m), angles)

        return walk

    def adjoint_walk_step_kernel(self) -> "Kernel":
        """``(ancilla, system)``: reflect about PREPARE, then SELECT."""
        angles = self._angles
        m = self._num_slots
        select = self._select_kernel()

        @cudaq.kernel
        def adjoint_walk(ancilla: cudaq.qview, system: cudaq.qview):
            unprepare(ancilla.front(m), angles)
            reflect_about_zero(ancilla)
            prepare(ancilla.front(m), angles)
            select(ancilla, system)

        return adjoint_walk

    def controlled_walk_step_kernel(self) -> "Kernel":
        angles = self._angles
        m = self._num_slots
        controlled_select = self._controlled_select_kernel()

        @cudaq.kernel
        def controlled_walk(control_and_ancilla: cudaq.qview,
                            system: cudaq.qview):
            controlled_select(control_and_ancilla, system)
            unprepare(control_and_ancilla[1:1 + m], angles)
            controlled_reflect_about_zero(control_and_ancilla)
            prepare(control_and_ancilla[1:1 + m], angles)

        return controlled_walk

    def controlled_adjoint_walk_step_kernel(self) -> "Kernel":
        angles = self._angles
        m = self._num_slots
        controlled_select = self._controlled_select_kernel()

        @cudaq.kernel
        def controlled_adjoint_walk(control_and_ancilla: cudaq.qview,
                                    system: cudaq.qview):
            unprepare(control_and_ancilla[1:1 + m], angles)
            controlled_reflect_about_zero(control_and_ancilla)
            prepare(control_and_ancilla[1:1 + m], angles)
            controlled_select(control_and_ancilla, system)

        return controlled_adjoint_walk

    def select_observable(self):
        raise NotImplementedError(
            "SumEncoding has no select_observable: its SELECT drives each "
            "summand's full block encoding (PREPARE trees included), which "
            "has no compact SpinOperator form. Even moments are available "
            "through Walk.moment; for odd orders take the good-subspace "
            "block of Walk(encoding).kernel(power=k) on a statevector "
            "simulator.")

    # ------------------------------------------------------------------
    # Convenience (not protocol members)
    # ------------------------------------------------------------------

    def encode_kernel(self, state_prep: "Kernel | None" = None) -> "Kernel":
        """A kernel applying the full block encoding (``PauliLCU`` shape).

        Without ``state_prep``: ``(state: cudaq.State)``, allocating the
        system register from ``state`` and the ancillas after it. With
        ``state_prep`` (a ``(qubits: qview)`` kernel): zero-argument,
        preparing the system register with ``state_prep`` first.
        """
        return _encode_kernel(self, state_prep)


# ============================================================================
# RotatedEncoding
# ============================================================================


class RotatedEncoding:
    """Block encoding of ``V H V^dagger`` from an encoding of ``H``.

    Use :meth:`from_orbital_rotation` for a real orbital rotation in the
    interleaved spin-orbital layout.

    Parameters
    ----------
    inner
        A ``BlockEncoding`` of ``H``.
    enter, exit
        ``(system: cudaq.qview)`` kernels implementing ``V^dagger`` and
        ``V`` on the system register. Every ``(ancilla, system)`` kernel of
        ``inner`` is wrapped as ``enter; inner; exit``; the ancilla-only
        PREPARE hooks pass through, and ``alpha`` and the ancilla count are
        unchanged.
    """

    def __init__(self, inner: "BlockEncoding", enter: "Kernel",
                 exit: "Kernel") -> None:
        _require_encoding(inner, "inner")
        for label, kernel in (("enter", enter), ("exit", exit)):
            if not callable(kernel):
                raise TypeError(f"{label} must be a (system: cudaq.qview) "
                                f"kernel, got {kernel!r}")
        self._inner = inner
        self._enter = enter
        self._exit = exit
        self._num_rotations: int | None = None

    @classmethod
    def from_orbital_rotation(cls, inner: "BlockEncoding",
                              orthogonal) -> "RotatedEncoding":
        """Rotate by a real orthogonal one-particle matrix ``V``.

        ``orthogonal`` is ``n x n`` with the target modes as columns and
        ``2 n == inner.num_system`` (spin orbital ``2p`` is mode ``p`` spin
        up, ``2p + 1`` spin down). The rotation is compiled to a network of
        adjacent plane rotations applied to both spin sectors, exactly as
        the double-factorized example does. The network reproduces ``V``
        up to a sign per mode, which is immaterial when the inner encoding
        is diagonal in the rotated modes (number operators) — the intended
        use — and must be accounted for otherwise.
        """
        _require_encoding(inner, "inner")
        if np.iscomplexobj(orthogonal):
            raise ValueError("orthogonal must be a real matrix")
        matrix = np.asarray(orthogonal, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("orthogonal must be a square matrix")
        n = matrix.shape[0]
        if n < 2:
            raise ValueError("an orbital rotation needs at least 2 modes (4 "
                             "system qubits); a 1 x 1 orthogonal matrix is a "
                             "sign, not a rotation")
        if 2 * n != int(inner.num_system):
            raise ValueError(
                f"a {n} x {n} orbital rotation needs {2 * n} system qubits "
                f"(interleaved spin orbitals); the encoding has "
                f"{inner.num_system}")
        if not np.allclose(matrix.T @ matrix, np.eye(n), atol=1e-10):
            raise ValueError("orthogonal must have orthonormal columns")
        sweep = _givens_sweep(matrix)
        enter_orbitals, enter_angles = _rotation_lists(sweep, reverse=False)
        exit_orbitals, exit_angles = _rotation_lists(sweep, reverse=True)

        @cudaq.kernel
        def enter(system: cudaq.qview):
            _spin_orbital_rotations(system, enter_orbitals, enter_angles)

        @cudaq.kernel
        def exit(system: cudaq.qview):
            _spin_orbital_rotations(system, exit_orbitals, exit_angles)

        rotated = cls(inner, enter, exit)
        rotated._num_rotations = len(sweep)
        return rotated

    @property
    def inner(self) -> "BlockEncoding":
        return self._inner

    @property
    def num_rotations(self) -> int | None:
        """Adjacent Givens rotations per spin sector in a
        ``from_orbital_rotation`` network, applied on entry and again on
        exit; ``None`` for user-supplied kernels."""
        return self._num_rotations

    @property
    def num_system(self) -> int:
        return int(self._inner.num_system)

    @property
    def num_ancilla(self) -> int:
        return int(self._inner.num_ancilla)

    @property
    def alpha(self) -> float:
        return float(self._inner.alpha)

    def __repr__(self) -> str:
        return f"RotatedEncoding({self._inner!r})"

    def prepare_kernel(self) -> "Kernel":
        return self._inner.prepare_kernel()

    def unprepare_kernel(self) -> "Kernel":
        return self._inner.unprepare_kernel()

    def _sandwich(self, inner_kernel):
        enter = self._enter
        exit = self._exit

        @cudaq.kernel
        def rotated(ancilla: cudaq.qview, system: cudaq.qview):
            enter(system)
            inner_kernel(ancilla, system)
            exit(system)

        return rotated

    def apply_kernel(self) -> "Kernel":
        return self._sandwich(self._inner.apply_kernel())

    def controlled_apply_kernel(self) -> "Kernel":
        # At control |0> the inner kernel is the identity and the
        # uncontrolled sandwich telescopes away.
        return self._sandwich(self._inner.controlled_apply_kernel())

    def walk_step_kernel(self) -> "Kernel":
        return self._sandwich(self._inner.walk_step_kernel())

    def adjoint_walk_step_kernel(self) -> "Kernel":
        return self._sandwich(self._inner.adjoint_walk_step_kernel())

    def controlled_walk_step_kernel(self) -> "Kernel":
        return self._sandwich(self._inner.controlled_walk_step_kernel())

    def controlled_adjoint_walk_step_kernel(self) -> "Kernel":
        return self._sandwich(
            self._inner.controlled_adjoint_walk_step_kernel())

    def select_observable(self):
        raise NotImplementedError(
            "RotatedEncoding has no select_observable: the inner encoding's "
            "terms live in the rotated frame and are not computational-frame "
            "Pauli words. Even moments are available through Walk.moment; "
            "for odd orders take the good-subspace block of "
            "Walk(encoding).kernel(power=k) on a statevector simulator.")

    def encode_kernel(self, state_prep: "Kernel | None" = None) -> "Kernel":
        """A kernel applying the full rotated block encoding (``PauliLCU``
        shape; see ``SumEncoding.encode_kernel``)."""
        return _encode_kernel(self, state_prep)


# ============================================================================
# Shared helpers
# ============================================================================


def _encode_kernel(encoding, state_prep):
    apply = encoding.apply_kernel()
    n_anc = int(encoding.num_ancilla)
    n_sys = int(encoding.num_system)

    if state_prep is not None:

        @cudaq.kernel
        def prep_encoded():
            system = cudaq.qvector(n_sys)
            state_prep(system)
            ancilla = cudaq.qvector(n_anc)
            apply(ancilla, system)

        return prep_encoded

    @cudaq.kernel
    def encoded(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        apply(ancilla, system)

    return encoded


def _givens_sweep(orthogonal: np.ndarray) -> list[tuple[int, float]]:
    """Adjacent plane rotations ``(p, theta)`` on rows ``(p, p + 1)`` that
    reduce ``orthogonal`` to a diagonal of signs (column by column)."""
    work = np.array(orthogonal, dtype=float, copy=True)
    n = work.shape[0]
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
    return rotations


def _rotation_lists(sweep, reverse: bool):
    pairs = ([(p, -theta)
              for p, theta in reversed(sweep)] if reverse else list(sweep))
    if not pairs:
        pairs = [(0, 0.0)]  # empty lists cannot cross the kernel boundary
    return [p for p, _ in pairs], [theta for _, theta in pairs]


@cudaq.kernel
def _spin_orbital_rotations(system: cudaq.qview, orbitals: list[int],
                            angles: list[float]):
    """Adjacent spatial plane rotations lifted to both spin sectors.

    Mode ``p`` lives on qubits ``2p`` (up) and ``2p + 1`` (down); the
    rotation on modes ``(p, p + 1)`` is ``exp(-i theta/2 (X Z Y - Y Z X))``
    on the three-qubit window of each spin. Zero angles are the identity.
    """
    for i in range(len(orbitals)):
        p = orbitals[i]
        theta = angles[i]
        up = system[2 * p:2 * p + 3]
        exp_pauli(0.5 * theta, up, "XZY")
        exp_pauli(-0.5 * theta, up, "YZX")
        down = system[2 * p + 1:2 * p + 4]
        exp_pauli(0.5 * theta, down, "XZY")
        exp_pauli(-0.5 * theta, down, "YZX")
