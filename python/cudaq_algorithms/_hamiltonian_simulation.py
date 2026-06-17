from cudaq_algorithms._pycudaq_algorithms import hamiltonian_simulation as _cpp_hamiltonian_simulation

apply_trotter = _cpp_hamiltonian_simulation.apply_trotter
_cpp_make_trotter_terms = _cpp_hamiltonian_simulation._make_trotter_terms


def _maybe_call(value):
    return value() if callable(value) else value


def _is_python_spin_operator(value):
    return hasattr(value, "term_count") and hasattr(value, "__iter__")


def _is_python_spin_term(value):
    return hasattr(value, "evaluate_coefficient") and hasattr(
        value, "get_pauli_word")


def _term_coefficient(term, coefficient_tolerance):
    coefficient = term.evaluate_coefficient()
    if abs(coefficient.imag) > coefficient_tolerance:
        raise ValueError(
            "trotter error - only real Hamiltonian coefficients are supported.")
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

    coefficients = []
    words = []
    identity_coefficient = 0.0

    for term in terms:
        coefficient = _term_coefficient(term, coefficient_tolerance)
        term_extent = _term_qubit_extent(term)
        num_qubits = max(num_qubits, term_extent)
        if term.is_identity():
            identity_coefficient += coefficient
            continue
        words.append(term.get_pauli_word(num_qubits))
        coefficients.append(coefficient)

    return coefficients, words, identity_coefficient, num_qubits


_cpp_hamiltonian_simulation.make_trotter_terms = make_trotter_terms

__all__ = ["apply_trotter", "make_trotter_terms"]
