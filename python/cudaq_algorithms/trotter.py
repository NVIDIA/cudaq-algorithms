# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Suzuki-Trotter Hamiltonian simulation.

Product-formula time evolution for Hamiltonians expressed as sums of Pauli
strings: term extraction, host-side planning and ordering, resource
estimation, and the circuit primitive itself.

The typical workflow builds a validated plan on the host and either uses
its kernel factory or composes the ``apply_trotter`` primitive inside a
custom kernel::

    from cudaq_algorithms import trotter

    evolution = trotter.Trotter(hamiltonian)
    kernel = evolution.kernel(time=0.8, steps=4, order=2)  # @cudaq.kernel()
    resources = evolution.resources(steps=4, order=2)

Identity terms: for ``H = c I + H'``, ``apply_trotter`` implements the
product formula for ``H'`` only. The omitted ``exp(-i c t)`` is an
unobservable global phase for one unconditioned evolution but a real
relative phase for controlled or interference-based algorithms;
``identity_coefficient`` is reported on the ``Trotter`` object so callers
can account for it (the simulation helper ``sim_utils.evolve``
reintroduces it host-side).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

import cudaq

from .pauli_lcu import state_from

#: Supported product-formula orders.
FIRST_ORDER_TROTTER: int = 1
SECOND_ORDER_TROTTER: int = 2
FOURTH_ORDER_TROTTER: int = 4

#: Forest-Ruth fourth-order splitting weights: the order-4 step is the
#: symmetric second-order step applied with time fractions w1, w0, w1,
#: where w1 = 1/(2 - 2**(1/3)) and w0 = 1 - 2*w1.
FOREST_RUTH_W1: float = 1.3512071919596578
FOREST_RUTH_W0: float = -1.7024143839193153

#: Accepted Hamiltonian input forms: a ``cudaq.SpinOperator`` (or a single
#: spin term), a ``{"XZI...": coefficient}`` mapping, or an iterable of
#: ``(coefficient, word)`` pairs.
HamiltonianLike = Union[Mapping[str, complex], Iterable[tuple], Any]


# ============================================================================
# Device kernel (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def apply_trotter(coefficients: list[float], words: list[cudaq.pauli_word],
                  time: float, steps: int, order: int, qubits: cudaq.qview):
    """Apply Suzuki-Trotter evolution exp(-i H' t) to a live register.

    Identity terms are intentionally omitted (see the module docstring).
    Invalid runtime inputs are no-ops: zero steps, mismatched
    coefficient/word lengths, or an unsupported order leave the register
    unchanged.

    NOTE: the whole body is one positively-guarded if-block rather than
    early-return guards because ``return`` inside a Python ``@cudaq.kernel``
    is silently ignored by the compiler (gates after it still execute); see
    https://github.com/NVIDIA/cuda-quantum/issues/4845.
    """
    valid_order = order == 1 or order == 2 or order == 4
    if steps > 0 and len(coefficients) == len(words) and valid_order:
        n = len(words)
        dt = time / steps
        for _ in range(steps):
            if order == 1:
                for i in range(n):
                    exp_pauli(-dt * coefficients[i], qubits, words[i])
            elif order == 4:
                # Forest-Ruth: symmetric second-order steps with time
                # fractions w1, w0, w1.
                for i in range(n):
                    exp_pauli(-0.5 * FOREST_RUTH_W1 * dt * coefficients[i],
                              qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * FOREST_RUTH_W1 * dt * coefficients[i],
                              qubits, words[i])
                for i in range(n):
                    exp_pauli(-0.5 * FOREST_RUTH_W0 * dt * coefficients[i],
                              qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * FOREST_RUTH_W0 * dt * coefficients[i],
                              qubits, words[i])
                for i in range(n):
                    exp_pauli(-0.5 * FOREST_RUTH_W1 * dt * coefficients[i],
                              qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * FOREST_RUTH_W1 * dt * coefficients[i],
                              qubits, words[i])
            else:
                for i in range(n):
                    exp_pauli(-0.5 * dt * coefficients[i], qubits, words[i])
                for j in range(n):
                    i = n - 1 - j
                    exp_pauli(-0.5 * dt * coefficients[i], qubits, words[i])


# state_from lives with the block-encoding module; re-exported here since
# Trotter workflows need the same precision-aware state construction.


# ============================================================================
# Host-side term extraction
# ============================================================================


def _maybe_call(value: Any) -> Any:
    """Return ``value()`` if callable (property-vs-method API tolerance)."""
    return value() if callable(value) else value


def _is_spin_like(value: Any) -> bool:
    """Return True for cudaq spin operators and spin terms (duck-typed)."""
    return (hasattr(value, "evaluate_coefficient")
            or hasattr(value, "term_count")
            or hasattr(value, "get_term_count"))


def _term_qubit_extent(term: Any) -> int:
    """Number of qubits a spin term touches (max degree + 1)."""
    max_degree = _maybe_call(getattr(term, "max_degree", -1))
    return max_degree + 1 if max_degree >= 0 else 0


def _real_coefficient(coefficient: complex, tolerance: float) -> float:
    """Coerce a coefficient to its real part, rejecting imaginary content.

    Raises ``ValueError`` if the imaginary part exceeds ``tolerance``.
    """
    coefficient = complex(coefficient)
    if abs(coefficient.imag) > tolerance:
        raise ValueError(
            "trotter error - only real Hamiltonian coefficients are "
            "supported.")
    return float(coefficient.real)


def _word_pairs_from_input(
        hamiltonian: HamiltonianLike,
        tolerance: float) -> tuple[list[tuple[float, str]], int]:
    """Normalize any accepted Hamiltonian form to ``(coefficient, word)``
    pairs plus the register width.

    Accepted forms:

    * a mapping ``{"XZI...": coefficient}``;
    * a ``cudaq.SpinOperator`` or a single spin term (wrapped via
      ``cudaq.SpinOperator`` so elementary products normalize the same
      way);
    * an iterable of ``(coefficient, word)`` pairs.

    Every word is validated to contain only I/X/Y/Z, all words must share
    one width, and coefficients must be real to within ``tolerance``.

    Returns ``(pairs, num_qubits)`` where ``pairs`` is a list of
    ``(float, str)`` and ``num_qubits`` is the common word width.
    """
    if isinstance(hamiltonian, Mapping):
        pairs = [(_real_coefficient(c, tolerance), str(w))
                 for w, c in hamiltonian.items()]
        width = max((len(w) for _, w in pairs), default=0)
    elif _is_spin_like(hamiltonian):
        # Wrapping canonicalizes: elementary products and full operators
        # both become a SpinOperator whose iteration yields terms with
        # evaluate_coefficient() and get_pauli_word().
        terms = list(cudaq.SpinOperator(hamiltonian))
        # Fix the register width once so every word is padded to the same
        # length.
        width = 0
        for term in terms:
            width = max(width, _term_qubit_extent(term))
        pairs = [(_real_coefficient(term.evaluate_coefficient(), tolerance),
                  str(term.get_pauli_word(width))) for term in terms]
    elif isinstance(hamiltonian, Iterable):
        pairs = [(_real_coefficient(c, tolerance), str(w))
                 for c, w in hamiltonian]
        width = max((len(w) for _, w in pairs), default=0)
    else:
        raise TypeError(
            "hamiltonian must be a cudaq spin operator or term, a "
            "{word: coeff} mapping, or an iterable of (coeff, word) pairs")

    for _, word in pairs:
        if len(word) != width:
            raise ValueError("all Pauli words must have the same length")
        if any(ch not in "IXYZ" for ch in word):
            raise ValueError(f"unsupported Pauli word: {word!r}")
    return pairs, width


def make_trotter_terms(
        hamiltonian: HamiltonianLike,
        coefficient_tolerance: float = 1e-12
) -> tuple[list[float], list[str], float, int]:
    """Return flattened terms for the Suzuki-Trotter circuit primitive.

    Returns ``(coefficients, words, identity_coefficient, num_qubits)``
    where ``words`` are padded plain strings: readable, comparable, and
    accepted directly as ``list[cudaq.pauli_word]`` kernel arguments.
    (Only kernel-captured words need explicit ``cudaq.pauli_word``
    conversion, which ``Trotter.kernel`` performs internally.)
    """
    if coefficient_tolerance < 0.0:
        raise ValueError(
            "trotter error - coefficient tolerance must be non-negative.")

    pairs, num_qubits = _word_pairs_from_input(hamiltonian,
                                               coefficient_tolerance)

    coefficients: list[float] = []
    words: list[str] = []
    identity_coefficient = 0.0
    for coefficient, word in pairs:
        if set(word) <= {"I"}:
            identity_coefficient += coefficient
            continue
        coefficients.append(coefficient)
        words.append(word)

    return coefficients, words, identity_coefficient, num_qubits


# ============================================================================
# Planning, ordering, resources
# ============================================================================


class TrotterOrdering(Enum):
    """Term-ordering strategies for the product formula.

    ``PRESERVE_INPUT`` keeps the extraction order;
    ``COEFFICIENT_MAGNITUDE_DESCENDING`` applies the largest-magnitude
    terms first, a common heuristic for reducing Trotter error.
    """

    PRESERVE_INPUT = "preserve_input"
    COEFFICIENT_MAGNITUDE_DESCENDING = "coefficient_magnitude_descending"


def _validate_order(order: int) -> int:
    """Validate and return a supported product-formula order (1, 2, or 4)."""
    if order not in (FIRST_ORDER_TROTTER, SECOND_ORDER_TROTTER,
                     FOURTH_ORDER_TROTTER):
        raise ValueError("order must be one of {1, 2, 4}")
    return int(order)


def _validate_steps(steps: int) -> int:
    """Validate and return a positive Trotter step count."""
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be greater than zero")
    return steps


def _validate_time(time: float) -> float:
    """Validate and return a finite evolution time."""
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be a finite number")
    return time


def _coerce_ordering(ordering: TrotterOrdering | str) -> TrotterOrdering:
    """Coerce a ``TrotterOrdering`` or its string value to the enum."""
    if isinstance(ordering, TrotterOrdering):
        return ordering
    try:
        return TrotterOrdering(str(ordering))
    except ValueError as exc:
        raise ValueError(f"unsupported Trotter ordering: {ordering}") from exc


def _ordered_terms(
        coefficients: list[float], words: list[str],
        ordering: TrotterOrdering) -> tuple[list[float], list[str]]:
    """Apply the ordering strategy to parallel coefficient/word lists."""
    coefficients = list(coefficients)
    words = list(words)
    if ordering == TrotterOrdering.PRESERVE_INPUT:
        return coefficients, words
    ordered = sorted(zip(coefficients, words),
                     key=lambda item: abs(item[0]),
                     reverse=True)
    if not ordered:
        return [], []
    ordered_coefficients, ordered_words = zip(*ordered)
    return list(ordered_coefficients), list(ordered_words)


def _pauli_weight(word: str) -> int:
    """Number of non-identity Paulis in a word."""
    return sum(1 for op in str(word) if op != "I")


def _rotations_per_step(order: int) -> int:
    """Pauli rotations each term contributes per Trotter step."""
    return {1: 1, 2: 2, 4: 6}[_validate_order(order)]


@dataclass(frozen=True)
class TrotterResourceEstimate:
    """Lightweight circuit-cost summary for a Trotter sequence.

    ``estimated_cx_count`` is a decomposition proxy: two CNOTs per
    additional non-identity Pauli in each rotation.
    """

    num_terms: int
    steps: int
    order: int
    pauli_rotations: int
    estimated_cx_count: int
    identity_coefficient: float


class Trotter:
    """Suzuki-Trotter product formulas for a Pauli-sum Hamiltonian.

    Term extraction, validation, and ordering happen once at
    construction (identity terms are split off into
    ``identity_coefficient``); the evolution parameters ``time``,
    ``steps``, and ``order`` are supplied per kernel request, mirroring
    the other primitives (``Walk.kernel(power=...)``,
    ``QSVT.kernel(sequence)``).

    The object is hardware-shaped: nothing here executes a
    simulator-only API (see ``sim_utils.evolve`` for statevector-based
    evolution).
    """

    def __init__(self,
                 hamiltonian: HamiltonianLike,
                 ordering: TrotterOrdering | str = (
                     TrotterOrdering.PRESERVE_INPUT),
                 *,
                 coefficient_tolerance: float = 1e-12) -> None:
        ordering = _coerce_ordering(ordering)
        coefficients, words, identity, num_qubits = make_trotter_terms(
            hamiltonian, coefficient_tolerance)
        coefficients, words = _ordered_terms(coefficients, words, ordering)
        self._coefficients = coefficients
        self._words = words
        self._identity = identity
        self._num_qubits = num_qubits
        self._ordering = ordering

    @property
    def coefficients(self) -> list[float]:
        """Retained non-identity coefficients, in application order."""
        return list(self._coefficients)

    @property
    def words(self) -> list[str]:
        """Retained Pauli words, parallel to ``coefficients``."""
        return list(self._words)

    @property
    def identity_coefficient(self) -> float:
        """Sum of identity-term coefficients (not realizable in circuit)."""
        return self._identity

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_terms(self) -> int:
        """Number of retained non-identity terms."""
        return len(self._coefficients)

    @property
    def ordering(self) -> TrotterOrdering:
        return self._ordering

    def __repr__(self) -> str:
        return (f"Trotter(terms={self.num_terms}, "
                f"qubits={self.num_qubits}, "
                f"identity_coefficient={self.identity_coefficient:.6g})")

    def kernel(self,
               time: float,
               steps: int = 1,
               order: int = SECOND_ORDER_TROTTER) -> Any:
        """Return a ``@cudaq.kernel()`` applying the product formula.

        (``Any`` because CUDA-Q exposes no stable public Python type for
        compiled kernel objects.)

        The kernel allocates ``num_qubits`` qubits in |0...0> and applies
        the ``order``-order formula for ``time`` over ``steps`` steps. The
        identity phase is not included (it cannot be, in a circuit); track
        ``identity_coefficient`` when it matters.
        """
        time = _validate_time(time)
        steps = _validate_steps(steps)
        order = _validate_order(order)
        coefficients = [float(c) for c in self._coefficients]
        words = [cudaq.pauli_word(str(w)) for w in self._words]
        num_qubits = int(self._num_qubits)

        if not words:
            # Identity-only Hamiltonian: the circuit is the identity.
            # Special-cased because captured empty lists cannot be
            # marshaled across the kernel boundary
            # (https://github.com/NVIDIA/cuda-quantum/issues/4847).
            @cudaq.kernel
            def evolve_identity():
                cudaq.qvector(num_qubits)

            return evolve_identity

        @cudaq.kernel
        def evolve():
            qubits = cudaq.qvector(num_qubits)
            apply_trotter(coefficients, words, time, steps, order, qubits)

        return evolve

    def resources(self,
                  steps: int = 1,
                  order: int = SECOND_ORDER_TROTTER) -> TrotterResourceEstimate:
        """Resource estimate for ``steps`` steps of the ``order`` formula."""
        return estimate_trotter_resources(
            self._coefficients, self._words, steps=steps, order=order,
            identity_coefficient=self._identity)


def estimate_trotter_resources(
        coefficients: list[float],
        words: list[str],
        steps: int = 1,
        order: int = SECOND_ORDER_TROTTER,
        identity_coefficient: float = 0.0) -> TrotterResourceEstimate:
    """Return a lightweight resource estimate for a Trotter sequence."""
    coefficients = list(coefficients)
    words = list(words)
    steps = _validate_steps(steps)
    order = _validate_order(order)

    if len(coefficients) != len(words):
        raise ValueError("coefficients and words must have equal length")

    rotations = len(words) * steps * _rotations_per_step(order)
    cx_per_ordered_step = sum(
        max(0, 2 * (_pauli_weight(word) - 1)) for word in words)
    estimated_cx_count = cx_per_ordered_step * steps * _rotations_per_step(
        order)
    return TrotterResourceEstimate(
        num_terms=len(words),
        steps=steps,
        order=order,
        pauli_rotations=rotations,
        estimated_cx_count=estimated_cx_count,
        identity_coefficient=float(identity_coefficient))


__all__ = [
    "FIRST_ORDER_TROTTER",
    "SECOND_ORDER_TROTTER",
    "FOURTH_ORDER_TROTTER",
    "FOREST_RUTH_W1",
    "FOREST_RUTH_W0",
    "HamiltonianLike",
    "Trotter",
    "TrotterOrdering",
    "TrotterResourceEstimate",
    "apply_trotter",
    "estimate_trotter_resources",
    "make_trotter_terms",
    "state_from",
]
