import math
from dataclasses import dataclass
from enum import Enum

from cudaq_algorithms._pycudaq_algorithms import hamiltonian_simulation as _cpp_hamiltonian_simulation

apply_trotter = _cpp_hamiltonian_simulation.apply_trotter
_cpp_make_trotter_terms = _cpp_hamiltonian_simulation._make_trotter_terms

FIRST_ORDER_TROTTER = 1
SECOND_ORDER_TROTTER = 2
FOURTH_ORDER_TROTTER = 4


class TrotterOrdering(Enum):
    PRESERVE_INPUT = "preserve_input"
    COEFFICIENT_MAGNITUDE_DESCENDING = "coefficient_magnitude_descending"


@dataclass(frozen=True)
class TrotterPlan:
    coefficients: list[float]
    words: list
    identity_coefficient: float
    num_qubits: int
    time: float
    steps: int
    order: int
    ordering: TrotterOrdering


@dataclass(frozen=True)
class TrotterResourceEstimate:
    num_terms: int
    steps: int
    order: int
    pauli_rotations: int
    estimated_cx_count: int
    identity_coefficient: float


def _maybe_call(value):
    return value() if callable(value) else value


def _is_python_spin_operator(value):
    return hasattr(value, "term_count") and hasattr(value, "__iter__")


def _is_python_spin_term(value):
    return hasattr(value, "evaluate_coefficient") and hasattr(
        value, "get_pauli_word")


def _validate_order(order):
    if order not in (FIRST_ORDER_TROTTER, SECOND_ORDER_TROTTER,
                     FOURTH_ORDER_TROTTER):
        raise ValueError("order must be one of {1, 2, 4}")
    return int(order)


def _validate_steps(steps):
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be greater than zero")
    return steps


def _validate_time(time):
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be a finite number")
    return time


def _coerce_ordering(ordering):
    if isinstance(ordering, TrotterOrdering):
        return ordering
    try:
        return TrotterOrdering(str(ordering))
    except ValueError as exc:
        raise ValueError(f"unsupported Trotter ordering: {ordering}") from exc


def _ordered_terms(coefficients, words, ordering):
    coefficients = list(coefficients)
    words = list(words)
    if ordering == TrotterOrdering.PRESERVE_INPUT:
        return coefficients, words
    if ordering == TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING:
        ordered = sorted(zip(coefficients, words),
                         key=lambda item: abs(item[0]),
                         reverse=True)
        if not ordered:
            return [], []
        ordered_coefficients, ordered_words = zip(*ordered)
        return list(ordered_coefficients), list(ordered_words)
    raise ValueError(f"unsupported Trotter ordering: {ordering}")


def _pauli_weight(word):
    return sum(1 for op in str(word) if op != "I")


def _rotations_per_step(order):
    return {1: 1, 2: 2, 4: 6}[_validate_order(order)]


def _term_coefficient(term, coefficient_tolerance):
    coefficient = term.evaluate_coefficient()
    if abs(coefficient.imag) > coefficient_tolerance:
        raise ValueError(
            "trotter error - only real Hamiltonian coefficients are supported."
        )
    return float(coefficient.real)


def _term_qubit_extent(term):
    max_degree = _maybe_call(getattr(term, "max_degree", -1))
    return max_degree + 1 if max_degree >= 0 else 0


def make_trotter_terms(hamiltonian, coefficient_tolerance=1e-12):
    """Return flattened terms for Suzuki-Trotter circuit primitives.

    Returns ``(coefficients, words, identity_coefficient, num_qubits)`` where
    ``words`` are padded Pauli strings suitable for CUDA-Q kernel arguments.

    ``apply_trotter`` omits identity terms. For ``H = c I + H'``, it applies a
    product-formula approximation to ``exp(-i H' t)`` and leaves the phase
    ``exp(-i c t)`` to the caller. This phase cancels in ordinary expectation
    values of one unconditioned evolved state, but it can matter for controlled
    evolution, overlaps, phase estimation, Krylov/QEL moments, and other
    interference-based algorithms.
    """
    if coefficient_tolerance < 0.0:
        raise ValueError(
            "trotter error - coefficient tolerance must be non-negative.")

    if _is_python_spin_term(hamiltonian):
        num_qubits = _term_qubit_extent(hamiltonian)
        terms = [hamiltonian]
    elif _is_python_spin_operator(hamiltonian):
        num_qubits = int(_maybe_call(getattr(hamiltonian, "qubit_count", 0)))
        terms = list(hamiltonian)
    else:
        return _cpp_make_trotter_terms(hamiltonian, coefficient_tolerance)

    # Determine the full register width once so every Pauli word is padded to the
    # same length. Computing it inside the term loop would leave earlier words
    # padded to a shorter width than a later, wider term, producing ragged words
    # whose qubit indices no longer line up. This mirrors the C++ path, which
    # fixes num_qubits = hamiltonian.num_qubits() before padding.
    for term in terms:
        num_qubits = max(num_qubits, _term_qubit_extent(term))

    coefficients = []
    words = []
    identity_coefficient = 0.0

    for term in terms:
        coefficient = _term_coefficient(term, coefficient_tolerance)
        if term.is_identity():
            identity_coefficient += coefficient
            continue
        words.append(term.get_pauli_word(num_qubits))
        coefficients.append(coefficient)

    return coefficients, words, identity_coefficient, num_qubits


def make_trotter_plan(hamiltonian,
                      time,
                      steps=1,
                      order=SECOND_ORDER_TROTTER,
                      ordering=TrotterOrdering.PRESERVE_INPUT,
                      coefficient_tolerance=1e-12):
    """Build a host-side plan for Suzuki-Trotter evolution.

    The returned plan is intentionally simple: it contains flattened data that
    can be passed directly to ``apply_trotter``
    from a CUDA-Q kernel.
    """
    time = _validate_time(time)
    steps = _validate_steps(steps)
    order = _validate_order(order)
    ordering = _coerce_ordering(ordering)
    coefficients, words, identity, num_qubits = make_trotter_terms(
        hamiltonian, coefficient_tolerance)
    coefficients, words = _ordered_terms(coefficients, words, ordering)
    return TrotterPlan(coefficients=coefficients,
                       words=words,
                       identity_coefficient=identity,
                       num_qubits=num_qubits,
                       time=time,
                       steps=steps,
                       order=order,
                       ordering=ordering)


def estimate_trotter_resources(plan_or_coefficients,
                               words=None,
                               steps=None,
                               order=None,
                               identity_coefficient=0.0):
    """Return a lightweight resource estimate for a Trotter sequence.

    The CNOT count is a decomposition proxy based on two CNOTs per additional
    non-identity Pauli in each Pauli rotation.
    """
    if isinstance(plan_or_coefficients, TrotterPlan):
        coefficients = plan_or_coefficients.coefficients
        words = plan_or_coefficients.words
        steps = plan_or_coefficients.steps
        order = plan_or_coefficients.order
        identity_coefficient = plan_or_coefficients.identity_coefficient
    else:
        coefficients = list(plan_or_coefficients)
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


_cpp_hamiltonian_simulation.make_trotter_terms = make_trotter_terms
_cpp_hamiltonian_simulation.make_trotter_plan = make_trotter_plan
_cpp_hamiltonian_simulation.estimate_trotter_resources = estimate_trotter_resources
_cpp_hamiltonian_simulation.TrotterOrdering = TrotterOrdering
_cpp_hamiltonian_simulation.TrotterPlan = TrotterPlan
_cpp_hamiltonian_simulation.TrotterResourceEstimate = TrotterResourceEstimate

__all__ = [
    "FIRST_ORDER_TROTTER",
    "SECOND_ORDER_TROTTER",
    "FOURTH_ORDER_TROTTER",
    "TrotterOrdering",
    "TrotterPlan",
    "TrotterResourceEstimate",
    "apply_trotter",
    "make_trotter_terms",
    "make_trotter_plan",
    "estimate_trotter_resources",
]
