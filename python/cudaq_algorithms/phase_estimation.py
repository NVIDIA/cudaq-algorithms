# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantum phase estimation over a block encoding.

``PhaseEstimation`` estimates eigenvalues of ``H`` from a ``BlockEncoding``
of ``H / alpha`` by phase-estimating the encoding's qubitization walk — the
same walk ``Walk`` exposes. The walk ``W`` has eigenvalues ``e^{±iθ}`` with
``cos θ = −E/alpha`` on each eigenspace of ``H`` (``E`` the eigenvalue), so an
``m``-bit phase readout ``j`` maps back to an energy by

    E = −alpha · cos(2π j / 2^m).

Estimating the walk directly needs no Hamiltonian time evolution and no
external phase-angle synthesis: the controlled walk, the inverse QFT, and
state preparation are all in the library. The ``±θ`` sign ambiguity is
harmless (``cos`` is even), and an input prepared near an eigenstate
concentrates the readout on that eigenvalue (a general input spreads over the
spectrum weighted by overlap, the usual phase-estimation behaviour).

This is the textbook form: an ``m``-qubit phase register, a controlled
``W^{2^k}`` ladder, and an inverse QFT. (An iterative single-ancilla form is
a planned follow-up.)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cudaq
import numpy as np

from .common_kernels import state_from
from .primitives import iqft

if TYPE_CHECKING:
    from .block_encoding import BlockEncoding, Kernel

__all__ = ["PhaseEstimation"]

# Minted kernels are closures; keep references so CUDA-Q does not collect
# them while a returned factory kernel is still in use.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


def _validate_num_bits(num_bits: int) -> int:
    if int(num_bits) != num_bits or num_bits < 1:
        raise ValueError(
            f"num_bits must be a positive integer, got {num_bits}")
    return int(num_bits)


class PhaseEstimation:
    """Textbook quantum phase estimation over a ``BlockEncoding``.

    Parameters
    ----------
    encoding
        Any object satisfying the ``BlockEncoding`` protocol (``PauliLCU``,
        ``LCUBlockEncoding``, ...): it supplies the controlled qubitization
        walk and the one-norm ``alpha``.

    The phase register holds ``num_bits`` qubits; the energy resolution is
    set by how finely ``2π j / 2^num_bits`` samples the walk eigenphase.
    """

    def __init__(self, encoding: BlockEncoding):
        if encoding.num_ancilla < 1:
            raise ValueError("encoding.num_ancilla must be >= 1")
        self._encoding = encoding
        self._kernel_cache: dict = {}

    @property
    def encoding(self) -> BlockEncoding:
        return self._encoding

    def _encoding_kernel(self, name: str) -> Kernel:
        if name not in self._kernel_cache:
            self._kernel_cache[name] = getattr(self._encoding, name)()
        return self._kernel_cache[name]

    # ------------------------------------------------------------------
    # Energy readout
    # ------------------------------------------------------------------

    def phase_to_energy(self, index: int, num_bits: int) -> float:
        """Map an ``m``-bit phase readout ``j`` to an energy.

        ``E = −alpha · cos(2π j / 2^m)`` — the inverse of the walk's
        ``cos θ = −E/alpha`` eigenphase relation.
        """
        m = _validate_num_bits(num_bits)
        phase = 2.0 * math.pi * index / (1 << m)
        return -self._encoding.alpha * math.cos(phase)

    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------

    def kernel(self,
               num_bits: int,
               *,
               state_prep: Kernel | None = None) -> Kernel:
        """The phase-estimation circuit (no measurement).

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state``. With ``state_prep`` (a
        ``(qubits: qview)`` kernel): a zero-argument kernel that prepares the
        system register itself. In both the phase + ancilla registers start
        in ``|0...0>``; the returned circuit leaves the phase register ready
        to read (measure it, or take its marginal from ``cudaq.get_state``).
        """
        m = _validate_num_bits(num_bits)
        n_anc = self._encoding.num_ancilla
        n_sys = self._encoding.num_system
        prep = self._encoding_kernel("prepare_kernel")
        unprep = self._encoding_kernel("unprepare_kernel")
        controlled_step = self._encoding_kernel("controlled_walk_step_kernel")

        # The phase register and the ancilla share one vector so that the
        # control qubit for each controlled-W^{2^k} is contiguous with the
        # ancilla (CUDA-Q qviews are contiguous, and the control cannot share
        # a control set with a separate register). The last phase qubit is
        # the control slot, adjacent to the ancilla; each phase qubit k is
        # swapped into it, its 2^k controlled walk steps applied (wrapped in
        # the uncontrolled PREPARE pair, so phase-qubit |0> acts as identity),
        # then swapped back.

        if state_prep is not None:

            @cudaq.kernel
            def qpe_prepared():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                work = cudaq.qvector(m + n_anc)
                for i in range(m):
                    h(work[i])
                for k in range(m):
                    if k != m - 1:
                        swap(work[k], work[m - 1])
                    prep(work[m:m + n_anc])
                    for _ in range(1 << (m - 1 - k)):
                        controlled_step(work[m - 1:m + n_anc], system)
                    unprep(work[m:m + n_anc])
                    if k != m - 1:
                        swap(work[k], work[m - 1])
                iqft(work[0:m])

            _retain(qpe_prepared)
            return qpe_prepared

        @cudaq.kernel
        def qpe_state(state: cudaq.State):
            system = cudaq.qvector(state)
            work = cudaq.qvector(m + n_anc)
            for i in range(m):
                h(work[i])
            for k in range(m):
                if k != m - 1:
                    swap(work[k], work[m - 1])
                prep(work[m:m + n_anc])
                for _ in range(1 << (m - 1 - k)):
                    controlled_step(work[m - 1:m + n_anc], system)
                unprep(work[m:m + n_anc])
                if k != m - 1:
                    swap(work[k], work[m - 1])
            iqft(work[0:m])

        _retain(qpe_state)
        return qpe_state

    # ------------------------------------------------------------------
    # Energy estimation (simulation)
    # ------------------------------------------------------------------

    def energies(self,
                 num_bits: int,
                 *,
                 ket=None,
                 state_prep: Kernel | None = None,
                 top: int | None = None) -> list[tuple[float, float]]:
        """Estimated energies with probabilities, most probable first.

        Builds the phase-estimation state, takes the phase-register marginal
        from ``cudaq.get_state`` (deterministic — no shot noise), and maps
        each readout ``j`` to ``(energy, probability)``. Provide exactly one
        of ``ket`` (array-like / ``cudaq.State``) or ``state_prep``.
        ``top`` truncates to the most probable readouts.
        """
        if (ket is None) == (state_prep is None):
            raise ValueError("provide exactly one of ket or state_prep")
        m = _validate_num_bits(num_bits)
        n_sys = self._encoding.num_system
        n_anc = self._encoding.num_ancilla

        kernel = self.kernel(m, state_prep=state_prep)
        if state_prep is not None:
            state = np.array(cudaq.get_state(kernel))
        else:
            source = ket if isinstance(ket, cudaq.State) else state_from(ket)
            state = np.array(cudaq.get_state(kernel, source))

        # Little-endian layout: system (lowest bits) | phase | ancilla.
        reshaped = state.reshape(1 << n_anc, 1 << m, 1 << n_sys)
        probabilities = np.sum(np.abs(reshaped)**2, axis=(0, 2))

        results = [(self.phase_to_energy(j, m), float(probabilities[j]))
                   for j in range(1 << m)]
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results if top is None else results[:top]
