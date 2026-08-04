# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from .common_kernels import state_from
from .trotter import SECOND_ORDER_TROTTER, Trotter

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


def evolve(evolution: Trotter,
           ket: ArrayLike,
           time: float,
           steps: int = 1,
           order: int = SECOND_ORDER_TROTTER,
           include_identity_phase: bool = True) -> NDArray[np.complex128]:
    """Simulate a Trotter evolution on ``ket``; return the evolved statevector.

    Unlike the circuit primitive, this can reintroduce the identity phase
    ``exp(-i * identity_coefficient * time)`` (on by default), so the
    result approximates the full ``exp(-i H t)|ket>``.

    Delegates to ``Trotter.state_kernel`` — the same validation (finite
    time, positive integral steps, order in {1, 2, 4}) and marshaling as
    ``Trotter.kernel``, raising ``ValueError`` for invalid parameters
    instead of silently returning an unevolved state.
    """
    import numpy as np

    ket_array = np.asarray(ket, dtype=np.complex128)
    if ket_array.ndim != 1 or ket_array.size != (1 << evolution.num_qubits):
        raise ValueError(
            f"ket must be a 1-D statevector of dimension "
            f"{1 << evolution.num_qubits} for {evolution.num_qubits} "
            f"qubit(s); got shape {ket_array.shape}")

    kernel = evolution.state_kernel(time, steps, order)
    state = np.asarray(cudaq.get_state(kernel, state_from(ket_array)),
                       dtype=np.complex128)
    if include_identity_phase and evolution.identity_coefficient != 0.0:
        state = state * np.exp(
            -1.0j * evolution.identity_coefficient * float(time))
    return state
