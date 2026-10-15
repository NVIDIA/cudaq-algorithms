"""Independent small dense references; never imported by the worker."""

import numpy as np


class CheckFailure(ValueError):
    """An executed artifact did not satisfy a registered numerical check."""


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def real_array(payload, key, shape):
    require(isinstance(payload, dict) and key in payload, f"missing {key}")
    try:
        value = np.asarray(payload[key])
        require(value.dtype.kind in 'iuf', f"{key}: expected real numbers")
        require(value.shape == tuple(shape), f"{key}: expected shape {shape}")
        value = value.astype(float)
        require(np.all(np.isfinite(value)), f"{key}: nonfinite number")
        return value
    except (TypeError, ValueError) as exc:
        raise CheckFailure(f"{key}: {exc}") from exc


def complex_array(payload, key, shape):
    values = real_array(payload, key, (*shape, 2))
    return values[..., 0] + 1j * values[..., 1]


def close(actual, expected, atol, label, strict=False):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f"{label}: shape mismatch")
    require(
        np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
        f"{label}: nonfinite value")
    error = float(np.max(np.abs(a - b), initial=0))
    require(
        error < atol if strict else error <= atol,
        f"{label}: absolute error {error:.12g} exceeds tolerance {atol:g}")


def pauli(n, terms):
    """Sum (coefficient, {qubit: letter}) terms, q0 least significant."""
    matrices = {
        'I': np.eye(2),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.diag([1, -1])
    }
    result = np.zeros((2**n, 2**n), dtype=complex)
    for coefficient, operators in terms:
        term = np.ones((1, 1), dtype=complex)
        for q in reversed(range(n)):
            term = np.kron(term, matrices[operators.get(q, 'I')])
        result += coefficient * term
    return result


def annihilators(n):
    result = []
    for q in range(n):
        a = np.zeros((2**n, 2**n), dtype=complex)
        for state in range(2**n):
            if state & (1 << q):
                a[state ^ (1 << q),
                  state] = (-1)**((state & ((1 << q) - 1)).bit_count())
        result.append(a)
    return result


def sector(n, particles, spin=None):
    return np.array([
        s for s in range(2**n)
        if s.bit_count() == particles and (spin is None or (sum(
            (s >> q) & 1
            for q in range(0, n, 2)), sum(
                (s >> q) & 1 for q in range(1, n, 2))) == tuple(spin))
    ],
                    dtype=int)
