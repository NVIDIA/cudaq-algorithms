# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent bounded E05 mathematics; no artifact or CUDA-Q imports.

Four distinct determinants share one orthonormal orbital basis. The reference
is the normalized coherent sum of exterior products, in little-endian Fock
order. Arbitrary nonorthogonal determinant families are outside this slice.
These helpers alone do not establish kernel, resource or Walk compatibility.
"""
from itertools import combinations
import operator

import numpy as np

ATOL = 1e-10
MAX_QUBITS = 8


def _inputs(orbital_basis, occupations, coefficients):
    basis = np.array(orbital_basis, dtype=complex, copy=True)
    if (basis.ndim != 2 or basis.shape[0] != basis.shape[1]
            or not 1 <= len(basis) <= MAX_QUBITS
            or not np.isfinite(basis).all()):
        raise ValueError("requires a finite, small square orbital basis")
    if not np.allclose(
            basis.conj().T @ basis, np.eye(len(basis)), atol=1e-12, rtol=0):
        raise ValueError("orbital basis must be orthonormal")
    try:
        determinants = []
        for occupied in occupations:
            indices = []
            for index in occupied:
                if isinstance(index, (bool, np.bool_)):
                    raise ValueError("boolean orbital index")
                indices.append(operator.index(index))
            if (not indices or indices != sorted(set(indices))
                    or indices[0] < 0 or indices[-1] >= len(basis)):
                raise ValueError(
                    "requires ascending distinct in-range orbital indices")
            determinants.append(tuple(indices))
    except TypeError as error:
        raise ValueError("invalid determinant indices") from error
    if (len(determinants) != 4 or len(set(determinants)) != 4
            or len({len(d)
                    for d in determinants}) != 1):
        raise ValueError(
            "requires four distinct equal-particle-number determinants")
    weights = np.array(coefficients, dtype=complex, copy=True)
    if weights.shape != (
            4, ) or not np.isfinite(weights).all() or not np.any(weights):
        raise ValueError(
            "requires four finite nonzero-as-a-vector coefficients")
    return basis, determinants, weights


def case_inputs(orbital_basis, occupations, coefficients):
    """Data-only views for explicit API bindings; never normalize artifact input."""
    basis, determinants, weights = _inputs(orbital_basis, occupations,
                                           coefficients)
    return {
        "orbital_basis": basis,
        "occupations": [list(d) for d in determinants],
        "orbital_matrices": [basis[:, d].copy() for d in determinants],
        "coefficients": weights
    }


def expected_state(orbital_basis, occupations, coefficients):
    """Compute sum_j c_j det(U[S,D_j]); normalize the complete coherent sum."""
    basis, determinants, weights = _inputs(orbital_basis, occupations,
                                           coefficients)
    # Scaling all coefficients by a positive real number preserves the target
    # ray and avoids overflow/underflow in the small reference computation.
    scale = max(np.abs(weights.real).max(), np.abs(weights.imag).max())
    weights = weights.real / scale + 1j * (weights.imag / scale)
    state = np.zeros(1 << len(basis), complex)
    for occupied in combinations(range(len(basis)), len(determinants[0])):
        index = sum(1 << site for site in occupied)
        state[index] = sum(
            weight * np.linalg.det(basis[np.ix_(occupied, determinant)])
            for weight, determinant in zip(weights, determinants))
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid reference norm")
    return state / norm


def assert_prepared_state(actual, expected, *, atol=ATOL):
    """Compare the complete state with ONE global phase, without fixing its norm."""
    if type(atol) not in (float,
                          int) or not np.isfinite(atol) or not 0 < atol < 1:
        raise ValueError("invalid comparison tolerance")
    expected = np.asarray(expected, dtype=complex)
    if (expected.ndim != 1 or not 2 <= expected.size <= 1 << MAX_QUBITS
            or expected.size & (expected.size - 1)
            or not np.isfinite(expected).all()
            or abs(np.linalg.norm(expected) - 1) > 1e-12):
        raise ValueError(
            "expected state must be finite, normalized and bounded")
    actual = np.asarray(actual, dtype=complex)
    assert actual.shape == expected.shape, "wrong full-register shape"
    assert np.isfinite(actual).all(), "nonfinite prepared state"
    norm = np.linalg.norm(actual)
    norm_error = float(abs(norm - 1))
    assert norm_error <= atol, "prepared state is not normalized"
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if overlap != 0 else 1
    error = float(np.max(np.abs(actual - phase * expected)))
    assert error <= atol, "incorrect coherent amplitudes (one overall phase allowed)"
    return {
        "norm_error": norm_error,
        "max_amplitude_error": error,
        "fidelity": min(1., float(abs(overlap / norm)**2))
    }


def assert_clean_ancillas(actual, expected_system, *, num_ancilla, atol=ATOL):
    """For zero-step/roundtrip Walk only: added high-order ancillas must be zero."""
    system = np.asarray(expected_system, dtype=complex)
    if (type(num_ancilla) is not int or not 0 <= num_ancilla <= MAX_QUBITS
            or system.ndim != 1
            or system.size << num_ancilla > 1 << MAX_QUBITS):
        raise ValueError("invalid or excessive register width")
    expected = np.zeros(system.size << num_ancilla, complex)
    expected[:system.size] = system
    return assert_prepared_state(actual, expected, atol=atol)
