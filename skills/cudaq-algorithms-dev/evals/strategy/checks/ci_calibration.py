# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator-only E05 calibration factories, never worker acceptance evidence.

For four spin orbitals and two electrons, independently apply Jordan-Wigner
creation operators, then synthesize a 16x16 unitary. This deliberately does not
exercise product Slater/Givens reuse or substantiate resource/scalability claims.
CUDA-Q is imported only when a factory is called; pure-reference tests never
construct or execute a quantum kernel.
"""
import uuid

import numpy as np


def _validated_inputs(orbital_basis, occupations, coefficients, normalize):
    basis = np.asarray(orbital_basis, dtype=np.complex128)
    if basis.shape != (4, 4) or not np.isfinite(basis).all():
        raise ValueError("calibration requires a finite four-orbital basis")
    if not np.allclose(basis.conj().T @ basis, np.eye(4), atol=1e-10, rtol=0):
        raise ValueError("orbital basis must be unitary")
    if len(occupations) != 4:
        raise ValueError("calibration requires four determinants")
    pairs = []
    for pair in occupations:
        if len(pair) != 2:
            raise ValueError(
                "each determinant must contain two orbital indices")
        if any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer)) or not 0 <= i < 4
                for i in pair):
            raise ValueError("determinant indices must be integers in [0, 4)")
        if pair[0] >= pair[1]:
            raise ValueError(
                "determinant indices must be distinct and ascending")
        pairs.append(tuple(int(i) for i in pair))
    if len(set(pairs)) != 4:
        raise ValueError("calibration requires four distinct determinants")
    weights = np.asarray(coefficients, dtype=np.complex128)
    if weights.shape != (4, ) or not np.isfinite(weights).all():
        raise ValueError("coefficients must be four finite values")
    norm = float(np.linalg.norm(weights))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("coefficient norm must be finite and nonzero")
    if normalize:
        weights = weights / norm
    elif abs(norm - 1) > 1e-10:
        raise ValueError("coefficients must be normalized")
    return basis, pairs, weights


def _create_mode(state, orbital_column):
    """Apply sum_p Q[p] a_p^dagger directly in the occupation basis."""
    result = np.zeros(16, dtype=np.complex128)
    for occupied, amplitude in enumerate(state):
        for orbital in range(4):
            if occupied & (1 << orbital):
                continue
            lower_occupancy = (occupied & ((1 << orbital) - 1)).bit_count()
            sign = -1 if lower_occupancy % 2 else 1
            result[occupied | (
                1 << orbital)] += sign * orbital_column[orbital] * amplitude
    return result


def creation_reference(orbital_basis,
                       occupations,
                       coefficients,
                       *,
                       normalize=False,
                       defect=None):
    """Independent CI vector from ordered creation operators, without minors."""
    basis, pairs, weights = _validated_inputs(orbital_basis, occupations,
                                              coefficients, normalize)
    weights = weights.copy()
    if defect == "wrong_relative_phase":
        weights[1] *= 1j
    elif defect == "dropped_imaginary":
        weights = weights.real.astype(np.complex128)
        norm = np.linalg.norm(weights)
        if norm == 0:
            raise ValueError(
                "imaginary-dropping mutant erased every coefficient")
        weights /= norm
    elif defect is not None:
        raise ValueError("unknown calibration defect")
    total = np.zeros(16, dtype=np.complex128)
    for pair, weight in zip(pairs, weights):
        determinant = np.zeros(16, dtype=np.complex128)
        determinant[0] = 1
        # a_j^dagger a_k^dagger |vacuum> applies k before j for j < k.
        for occupied_orbital in reversed(pair):
            determinant = _create_mode(determinant, basis[:, occupied_orbital])
        total += weight * determinant
    return total


def unitary_from_ci_state(state):
    """A complex Householder reflection whose first column is the CI state.

    Two-electron states have zero vacuum amplitude. For w = |vacuum> - |psi>,
    w^dagger w = 2, so I - w w^dagger is unitary and maps the vacuum to psi.
    """
    state = np.asarray(state, dtype=np.complex128)
    if state.shape != (16, ) or not np.isfinite(state).all() or abs(
            state[0]) > 1e-12:
        raise ValueError(
            "expected a finite 16-vector orthogonal to the vacuum")
    norm = np.linalg.norm(state)
    if abs(norm - 1) > 1e-10:
        raise ValueError("CI state must be normalized")
    direction = -(state / norm)
    direction[0] += 1
    return np.eye(16, dtype=np.complex128) - np.outer(direction,
                                                      direction.conj())


def _kernel(state, *, extra_ancilla=False):
    import cudaq
    from cudaq.kernel.kernel_decorator import mk_decorator

    matrix = unitary_from_ci_state(state)
    operation = "e05calibration" + uuid.uuid4().hex
    cudaq.register_operation(operation, matrix)
    kernel, qubits = cudaq.make_kernel(cudaq.qview)
    # register_operation uses MSB target order; the reference is little-endian.
    getattr(kernel, operation)(qubits[3], qubits[2], qubits[1], qubits[0])
    if extra_ancilla:
        scratch = kernel.qalloc(1)
        kernel.x(scratch[0])
    # The pinned AST call path accepts decorated kernels, not captured builders.
    # Convert inside the positive/negative fixture, never in the outcome binder.
    return mk_decorator(kernel)


def positive(orbital_basis, occupations, coefficients):
    """Positive numerical control with strict normalized-coefficient inputs."""
    return _kernel(creation_reference(orbital_basis, occupations,
                                      coefficients))


def positive_normalize(orbital_basis, occupations, coefficients):
    """Positive numerical control accepting nonunit nonzero coefficient norms."""
    return _kernel(
        creation_reference(orbital_basis,
                           occupations,
                           coefficients,
                           normalize=True))


def wrong_relative_phase(orbital_basis, occupations, coefficients):
    """Mutant: multiply only the second CI coefficient by i."""
    return _kernel(
        creation_reference(orbital_basis,
                           occupations,
                           coefficients,
                           defect="wrong_relative_phase"))


def dropped_imaginary(orbital_basis, occupations, coefficients):
    """Mutant: discard imaginary weights, retaining a normalized wrong state."""
    return _kernel(
        creation_reference(orbital_basis,
                           occupations,
                           coefficients,
                           defect="dropped_imaginary"))


def extra_ancilla(orbital_basis, occupations, coefficients):
    """Mutant: correct system state plus an undeclared internal |1> ancilla."""
    return _kernel(creation_reference(orbital_basis, occupations,
                                      coefficients),
                   extra_ancilla=True)
