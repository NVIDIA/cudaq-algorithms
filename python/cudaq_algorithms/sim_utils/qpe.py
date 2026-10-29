# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statevector shortcuts for `PhaseEstimation` (simulators only).

The class builds circuits and decodes readouts; these helpers execute
its measurement-free circuit with ``cudaq.get_state`` on a large-stack
worker and hand the exact phase-register marginal to its decoders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudaq

from ..common_kernels import state_from
from ._stack import run_on_large_stack

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..phase_estimation import PhaseEstimation

__all__ = ["phase_probabilities", "energies"]


def phase_probabilities(estimator: PhaseEstimation,
                        num_bits: int,
                        *,
                        ket=None,
                        state_prep=None) -> NDArray[np.float64]:
    """The exact phase-register marginal of a phase-estimation circuit:
    ``2^num_bits`` probabilities, entry ``j`` for readout ``j``, summing
    to 1 (no shot noise).

    Simulates ``estimator.kernel(num_bits, state_prep=...)`` with
    ``cudaq.get_state`` on a large-stack worker (`sim_utils.run_on_large_stack`) and
    traces out the system and ancilla registers. Provide exactly one of
    ``ket`` (array-like / ``cudaq.State``) or ``state_prep``. Decode with
    `PhaseEstimation.energies_from_probabilities`, or call `energies`.
    """
    import numpy as np

    if (ket is None) == (state_prep is None):
        raise ValueError("provide exactly one of ket or state_prep")
    kernel = estimator.kernel(num_bits, state_prep=state_prep)
    m = int(num_bits)
    if state_prep is not None:
        state = np.array(run_on_large_stack(cudaq.get_state, kernel))
    else:
        source = ket if isinstance(ket, cudaq.State) else state_from(ket)
        state = np.array(run_on_large_stack(cudaq.get_state, kernel, source))
    n_sys = estimator.encoding.num_system
    n_anc = estimator.encoding.num_ancilla
    # Little-endian layout: system (lowest bits) | phase | ancilla.
    reshaped = state.reshape(1 << n_anc, 1 << m, 1 << n_sys)
    return np.sum(np.abs(reshaped)**2, axis=(0, 2))


def energies(estimator: PhaseEstimation,
             num_bits: int,
             *,
             ket=None,
             state_prep=None,
             top: int | None = None) -> list[tuple[float, float]]:
    """Ranked ``(energy, probability)`` pairs from the simulated
    phase-estimation circuit: `phase_probabilities` decoded by
    `PhaseEstimation.energies_from_probabilities` (mirror readouts folded,
    most probable first, ``top`` keeps the leading entries)."""
    from ..phase_estimation import _validate_top

    top = _validate_top(top)  # before the simulation, not after it
    return estimator.energies_from_probabilities(phase_probabilities(
        estimator, num_bits, ket=ket, state_prep=state_prep),
                                                 num_bits,
                                                 top=top)
