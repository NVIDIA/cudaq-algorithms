# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Simulation-only helpers.

Everything here depends on statevector access (``cudaq.get_state`` /
postselection slicing), which only exists on simulators. The module ships
with the package as a clearly-labeled companion, but it is not part of the
hardware-shaped API: the library classes (encodings, kernel factories,
observables, ``Walk.moment`` via ``cudaq.observe``) never execute
``get_state``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import cudaq

# Re-exported: precision-aware initial-state construction.
from .pauli_lcu import state_from
from .trotter import TrotterPlan

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    from .pauli_lcu import PauliLCU
    from .qsvt import PhaseSequence, QSVT

__all__ = ["state_from", "good_subspace", "action", "transform", "evolve"]


def good_subspace(encoding: PauliLCU,
                  state: ArrayLike) -> NDArray[np.complex128]:
    """Postselect the all-zero-ancilla block of a simulated statevector.

    The kernel factories allocate the system register first, so with
    CUDA-Q's little-endian statevector order (q[0] = least-significant bit)
    the good subspace is the first contiguous block of 2**num_system
    amplitudes.
    """
    import numpy as np

    vector = np.asarray(state, dtype=np.complex128)
    expected = 1 << (encoding.num_system + encoding.num_ancilla)
    if vector.shape != (expected, ):
        raise ValueError(f"expected a statevector of dimension {expected}, "
                         f"got shape {vector.shape}")
    return vector[:1 << encoding.num_system].copy()


def action(encoding: PauliLCU, ket: ArrayLike) -> NDArray[np.complex128]:
    """Return (H/alpha)|ket> by simulating the encoding and postselecting.

    Multiply by ``encoding.alpha`` to recover H|ket>.
    """
    state = cudaq.get_state(encoding.encode_kernel(), state_from(ket))
    return good_subspace(encoding, state)


def transform(transformer: QSVT,
              ket: ArrayLike,
              sequence: PhaseSequence | Iterable[float],
              convention: str | None = None) -> NDArray[np.complex128]:
    """Return the good-subspace state after a QSVT sequence.

    For an eigenstate of H with eigenvalue lambda the result is
    ``p(lambda / alpha)`` times the input, where ``p`` is the polynomial
    the phase sequence implements.
    """
    state = cudaq.get_state(transformer.kernel(sequence, convention),
                            state_from(ket))
    return good_subspace(transformer.encoding, state)


def evolve(plan: TrotterPlan,
           ket: ArrayLike,
           include_identity_phase: bool = True) -> NDArray[np.complex128]:
    """Simulate a Trotter plan on ``ket`` and return the evolved statevector.

    Unlike the circuit primitive, this can reintroduce the identity phase
    ``exp(-i * identity_coefficient * time)`` (on by default), so the
    result approximates the full ``exp(-i H t)|ket>``.

    The plan's kernel prepares |0...0> and evolves; to evolve an arbitrary
    ``ket``, the input state is loaded through ``cudaq.get_state``'s
    initial-state support via a state-taking wrapper kernel.
    """
    import numpy as np

    coefficients = [float(c) for c in plan.coefficients]
    words = [cudaq.pauli_word(str(w)) for w in plan.words]
    time = float(plan.time)
    steps = int(plan.steps)
    order = int(plan.order)

    if words:
        from .trotter import apply_trotter

        @cudaq.kernel
        def evolve_state(state: cudaq.State):
            qubits = cudaq.qvector(state)
            apply_trotter(coefficients, words, time, steps, order, qubits)

        state = np.asarray(cudaq.get_state(evolve_state, state_from(ket)),
                           dtype=np.complex128)
    else:
        state = np.asarray(ket, dtype=np.complex128).copy()

    if include_identity_phase and plan.identity_coefficient != 0.0:
        state = state * np.exp(-1.0j * plan.identity_coefficient * time)
    return state
