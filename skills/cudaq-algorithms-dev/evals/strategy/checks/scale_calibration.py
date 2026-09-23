# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator-only controls, NEVER a fallback for a missing worker scale API.

These deliberately use simulator tomography to build known-small test outputs.
They are not valid production implementations and earn no worker credit.
The expected Hamiltonians remain literal, separately specified oracle fixtures.
"""
import itertools

import numpy as np


def pauli_terms(matrix, factor, padding=0.):
    """Calibration-only Pauli decomposition; padding adds a cancelling pair."""
    matrix = np.asarray(matrix, complex)
    ns = matrix.shape[0].bit_length() - 1
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1])
    }
    terms = []
    for word in itertools.product("IXYZ", repeat=ns):
        pauli = np.ones((1, 1))
        for letter in reversed(word):
            pauli = np.kron(pauli, matrices[letter])
        coefficient = factor * np.trace(pauli @ matrix).real / len(matrix)
        if abs(coefficient) > 1e-12:
            terms.append((float(coefficient), "".join(word)))
    if padding or not terms:
        amount = padding if padding else .5
        terms.extend([(amount, "I" * ns), (-amount, "I" * ns)])
    return terms


def _scale(encoding, factor, padding):
    from cudaq_algorithms import PauliLCU
    from scale_oracle import full_matrix
    if not np.isscalar(factor) or not np.isreal(factor) or not np.isfinite(
            factor):
        raise ValueError("factor must be real and finite")
    dim = 1 << encoding.num_system
    matrix = encoding.alpha * full_matrix(encoding, "apply_kernel")[:dim, :dim]
    return PauliLCU(pauli_terms(matrix, float(factor), padding))


def compact(encoding, factor):
    """alpha_out = abs(factor) * alpha_in on these distinct-word fixtures."""
    return _scale(encoding, factor, 0.)


def padded(*, base, multiplier):
    """Same operator with alpha_out larger by one, plus possible extra ancillas."""
    return _scale(base, multiplier, .5)


class _Proxy:

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _negated_walk(operation):
    import cudaq

    @cudaq.kernel
    def negated(ancilla: cudaq.qview, system: cudaq.qview):
        operation(ancilla, system)
        x(ancilla[0])
        z(ancilla[0])
        x(ancilla[0])
        z(ancilla[0])

    return negated


def _negated_controlled_walk(operation):
    import cudaq

    @cudaq.kernel
    def negated(control_and_ancilla: cudaq.qview, system: cudaq.qview):
        operation(control_and_ancilla, system)
        z(control_and_ancilla[0])

    return negated


def _bad_subspace_apply(operation):
    import cudaq
    from cudaq_algorithms.common_kernels import reflect_about_zero

    @cudaq.kernel
    def bad_phase(ancilla: cudaq.qview, system: cudaq.qview):
        # D=-R: +1 on good, -1 elsewhere. Right multiplication U D.
        reflect_about_zero(ancilla)
        x(ancilla[0])
        z(ancilla[0])
        x(ancilla[0])
        z(ancilla[0])
        operation(ancilla, system)

    return bad_phase


def _bad_subspace_controlled_apply(operation):
    import cudaq
    from cudaq_algorithms.common_kernels import controlled_reflect_about_zero

    @cudaq.kernel
    def bad_phase(control_and_ancilla: cudaq.qview, system: cudaq.qview):
        controlled_reflect_about_zero(control_and_ancilla)
        z(control_and_ancilla[0])
        operation(control_and_ancilla, system)

    return bad_phase


def mutant(encoding, factor, name):
    """One fault at a time for named numerical assertions."""
    if name == "ignore_factor":
        return compact(encoding, 1.)
    if name == "abs_sign":
        return compact(encoding, abs(factor))
    if name == "builtin_only":
        from cudaq_algorithms import PauliLCU
        if not isinstance(encoding, PauliLCU):
            raise TypeError("requires builtin")
    if name == "nonfinite_acceptance":
        return encoding
    if name == "zero_division":
        return 1. / factor
    output = _Proxy(compact(encoding, factor))
    if name == "alpha_only":
        output.alpha = 1.7 * output.alpha
    elif name == "adjoint":
        output.adjoint_walk_step_kernel = output.walk_step_kernel
    elif name == "control_phase":
        import cudaq
        original = output.controlled_apply_kernel()

        @cudaq.kernel
        def phase_fault(control_and_ancilla: cudaq.qview, system: cudaq.qview):
            z(control_and_ancilla[0])
            original(control_and_ancilla, system)

        output.controlled_apply_kernel = lambda: phase_fault
    elif name == "ordering":
        import cudaq
        original = output.apply_kernel()

        @cudaq.kernel
        def order_fault(ancilla: cudaq.qview, system: cudaq.qview):
            swap(system[0], system[1])
            original(ancilla, system)
            swap(system[0], system[1])

        output.apply_kernel = lambda: order_fault
    elif name == "walk_sign":
        forward = _negated_walk(output.walk_step_kernel())
        adjoint = _negated_walk(output.adjoint_walk_step_kernel())
        controlled = _negated_controlled_walk(
            output.controlled_walk_step_kernel())
        controlled_adjoint = _negated_controlled_walk(
            output.controlled_adjoint_walk_step_kernel())
        output.walk_step_kernel = lambda: forward
        output.adjoint_walk_step_kernel = lambda: adjoint
        output.controlled_walk_step_kernel = lambda: controlled
        output.controlled_adjoint_walk_step_kernel = lambda: controlled_adjoint
    elif name == "bad_subspace_phase":
        apply = _bad_subspace_apply(output.apply_kernel())
        controlled = _bad_subspace_controlled_apply(
            output.controlled_apply_kernel())
        output.apply_kernel = lambda: apply
        output.controlled_apply_kernel = lambda: controlled
    elif name != "builtin_only":
        raise ValueError("unknown calibration mutant")
    return output
