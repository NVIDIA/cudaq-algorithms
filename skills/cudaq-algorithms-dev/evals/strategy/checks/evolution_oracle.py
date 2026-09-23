# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent bounded E08 mathematics; no artifact or CUDA-Q imports.

This module computes a small dense evolution reference and compares a complete
state vector under an explicitly selected exact-vector or physical-ray policy.
It does not establish artifact provenance, QSVT use, or hardware execution.
"""
import numbers

import numpy as np

MAX_QUBITS = 4
INPUT_TOLERANCE = 1e-12
MAX_COMPARISON_TOLERANCE = 0.01


def _bounded_state(value, *, role):
    """Copy and validate evaluator-owned state input."""
    try:
        state = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{role} state must be a complex vector") from error
    size = state.size if state.ndim == 1 else 0
    if (state.ndim != 1 or not 2 <= size <= 1 << MAX_QUBITS
            or size & (size - 1) or not np.isfinite(state).all()):
        raise ValueError(
            f"{role} state must be a finite bounded power-of-two vector")
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or abs(norm - 1.0) > INPUT_TOLERANCE:
        raise ValueError(f"{role} state must be normalized")
    return state


def _bounded_hamiltonian(value):
    try:
        hamiltonian = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Hamiltonian must be a complex matrix") from error
    dimension = hamiltonian.shape[0] if hamiltonian.ndim == 2 else 0
    if (hamiltonian.ndim != 2 or hamiltonian.shape != (dimension, dimension)
            or not 2 <= dimension <= 1 << MAX_QUBITS
            or dimension & (dimension - 1)
            or not np.isfinite(hamiltonian).all()):
        raise ValueError(
            "Hamiltonian must be a finite bounded power-of-two square matrix")
    if not np.allclose(
            hamiltonian, hamiltonian.conj().T, atol=INPUT_TOLERANCE, rtol=0):
        raise ValueError("Hamiltonian must be Hermitian")
    return hamiltonian


def _finite_real(value, *, name, upper=None):
    if isinstance(value,
                  (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if upper is not None and not 0 < result <= upper:
        raise ValueError(f"{name} must be positive and at most {upper}")
    return result


def expected_state(hamiltonian, initial_state, time):
    """Return ``exp(-1j * H * time) @ initial_state`` for one to four qubits."""
    matrix = _bounded_hamiltonian(hamiltonian)
    state = _bounded_state(initial_state, role="initial")
    if state.shape != (len(matrix), ):
        raise ValueError("initial state dimension must match Hamiltonian")
    duration = _finite_real(time, name="time")

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("Hamiltonian eigendecomposition failed") from error
    with np.errstate(over="ignore", invalid="ignore"):
        phase_arguments = eigenvalues * duration
    if not np.isfinite(phase_arguments).all():
        raise ValueError("nonfinite evolution phase argument")
    with np.errstate(over="ignore", invalid="ignore"):
        phases = np.exp(-1j * phase_arguments)
        evolved = eigenvectors @ (phases * (eigenvectors.conj().T @ state))
    if not np.isfinite(phases).all() or not np.isfinite(evolved).all():
        raise ValueError("nonfinite dense evolution result")
    return evolved


def _actual_state(value, shape):
    try:
        state = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssertionError(
            "actual state must be a complex vector") from error
    if state.shape != shape:
        raise AssertionError("actual state has wrong shape")
    if not np.isfinite(state).all():
        raise AssertionError("actual state contains nonfinite values")
    return state


def assert_evolved_state(actual, expected, *, atol, phase_policy):
    """Assert full-state agreement without ever normalizing artifact output.

    ``phase_policy='exact'`` gates on the raw L2 vector error. ``'ray'`` gates
    on an L2 error after aligning exactly one global phase. Both errors are
    returned so the selected convention remains visible in saved evidence.
    """
    tolerance = _finite_real(atol, name="atol", upper=MAX_COMPARISON_TOLERANCE)
    if type(phase_policy) is not str or phase_policy not in ("exact", "ray"):
        raise ValueError("phase_policy must be 'exact' or 'ray'")
    target = _bounded_state(expected, role="expected")
    observed = _actual_state(actual, target.shape)

    norm = np.linalg.norm(observed)
    norm_error = float(abs(norm - 1.0))
    if not np.isfinite(norm):
        raise AssertionError("actual state norm is nonfinite")
    if norm_error > tolerance:
        raise AssertionError("actual state is not normalized")

    overlap = np.vdot(target, observed)
    phase = overlap / abs(overlap) if overlap != 0 else 1.0 + 0.0j
    raw_l2_error = float(np.linalg.norm(observed - target))
    aligned_l2_error = float(np.linalg.norm(observed - phase * target))
    selected_error = raw_l2_error if phase_policy == "exact" else aligned_l2_error
    if not np.isfinite(selected_error):
        raise AssertionError("actual state error is nonfinite")
    if selected_error > tolerance:
        raise AssertionError("actual state has incorrect amplitudes")

    fidelity = 0.0 if norm == 0 else float(abs(overlap)**2 / norm**2)
    fidelity = min(1.0, max(0.0, fidelity))
    return {
        "raw_l2_error": raw_l2_error,
        "aligned_l2_error": aligned_l2_error,
        "norm_error": norm_error,
        "fidelity": fidelity,
        "phase_policy": phase_policy,
    }
