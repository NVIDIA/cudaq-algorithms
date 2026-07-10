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

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    from .pauli_lcu import PauliLCU
    from .qsvt import PhaseSequence, QSVT

__all__ = ["state_from", "good_subspace", "action", "transform"]


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
