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
state preparation are all in the library. Each eigenvalue appears at the
two mirror readouts ``j`` and ``2^m - j`` (the ``±θ`` pair; ``cos`` is
even), so ``energies`` folds the pair into one entry. An input prepared
near an eigenstate concentrates the readout on that eigenvalue; a general
input spreads over the spectrum weighted by overlap, the usual
phase-estimation behaviour.

This is the textbook form: an ``m``-qubit phase register, a controlled
``W^{2^k}`` ladder, and an inverse QFT, costing ``2^m - 1`` controlled walk
steps. The phase register is little-endian (qubit 0 the least significant
bit), matching ``primitives.iqft``.

Simulating the ladder uses host stack in proportion to its length: the
compiler re-materializes the walk step's captured tables on the stack on
every loop iteration (about 1 kB per step for a ``PauliLCU``), so from
``m = 13`` the circuit outgrows the default 8 MiB thread stack and the
process dies with a segmentation fault. ``phase_probabilities`` and
``energies`` therefore run the simulation on a worker thread with a large
stack (``sim_utils.run_on_large_stack``); a circuit from ``kernel``
executed directly needs the same wrapper, or a raised stack limit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cudaq
import numpy as np

from .block_encoding import mint_cached_kernel
from .common_kernels import state_from
from .primitives import iqft
from .sim_utils import run_on_large_stack

if TYPE_CHECKING:
    from .block_encoding import BlockEncoding, Kernel

__all__ = ["PhaseEstimation"]


def _validate_num_bits(num_bits: int) -> int:
    if (isinstance(num_bits, bool) or int(num_bits) != num_bits
            or num_bits < 1):
        raise ValueError(
            f"num_bits must be a positive integer, got {num_bits!r}")
    return int(num_bits)


class PhaseEstimation:
    """Textbook quantum phase estimation over a ``BlockEncoding``.

    The phase register holds ``num_bits`` qubits; the energy resolution is
    set by how finely ``2π j / 2^num_bits`` samples the walk eigenphase
    (about ``alpha π / 2^num_bits`` at worst). Composed kernels are cached
    per ``(num_bits, state_prep)``.

    Parameters
    ----------
    encoding
        Any object satisfying the ``BlockEncoding`` protocol (``PauliLCU``
        and the other encodings in this package): it supplies the
        controlled qubitization walk, the PREPARE hooks that set the walk's
        frame, and the one-norm ``alpha``.
    """

    def __init__(self, encoding: BlockEncoding):
        if encoding.num_ancilla < 1:
            raise ValueError("encoding.num_ancilla must be >= 1")
        self._encoding = encoding
        self._kernel_cache: dict = {}
        self._composed: dict = {}

    @property
    def encoding(self) -> BlockEncoding:
        return self._encoding

    def __repr__(self) -> str:
        return f"PhaseEstimation({self._encoding!r})"

    def _encoding_kernel(self, name: str) -> Kernel:
        return mint_cached_kernel(self._kernel_cache, self._encoding, name)

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
        in ``|0...0>``. Register layout: system, then ``num_bits`` phase
        qubits (little-endian: phase qubit 0 is the least significant
        readout bit), then the encoding's ancillas. The returned circuit
        leaves the phase register ready to read (measure it, or take its
        marginal through ``phase_probabilities`` / ``energies``). Executing
        a long ladder yourself needs a large thread stack, about 1 kB per
        walk step (``sim_utils.run_on_large_stack(cudaq.sample, kernel,
        ...)``; module docstring).
        """
        m = _validate_num_bits(num_bits)
        key = (m, None if state_prep is None else id(state_prep))
        if key in self._composed:
            return self._composed[key][0]
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
        # swapped into it, its 2^(m-1-k) controlled walk steps applied, then
        # swapped back (the reversed power schedule matches the swap-free
        # iqft). The encoding's PREPARE pair wraps the whole ladder once: it
        # puts the ancilla in the walk's frame, where W acts as a rotation on
        # each eigenspace of H. Identity at a phase qubit in |0> is the
        # protocol contract of controlled_walk_step_kernel itself.

        if state_prep is not None:

            @cudaq.kernel
            def qpe_prepared():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                work = cudaq.qvector(m + n_anc)
                for i in range(m):
                    h(work[i])
                prep(work[m:m + n_anc])
                for k in range(m):
                    if k != m - 1:
                        swap(work[k], work[m - 1])
                    for _ in range(1 << (m - 1 - k)):
                        controlled_step(work[m - 1:m + n_anc], system)
                    if k != m - 1:
                        swap(work[k], work[m - 1])
                unprep(work[m:m + n_anc])
                iqft(work[0:m])

            self._composed[key] = (qpe_prepared, state_prep)
            return qpe_prepared

        @cudaq.kernel
        def qpe_state(state: cudaq.State):
            system = cudaq.qvector(state)
            work = cudaq.qvector(m + n_anc)
            for i in range(m):
                h(work[i])
            prep(work[m:m + n_anc])
            for k in range(m):
                if k != m - 1:
                    swap(work[k], work[m - 1])
                for _ in range(1 << (m - 1 - k)):
                    controlled_step(work[m - 1:m + n_anc], system)
                if k != m - 1:
                    swap(work[k], work[m - 1])
            unprep(work[m:m + n_anc])
            iqft(work[0:m])

        self._composed[key] = (qpe_state, None)
        return qpe_state

    # ------------------------------------------------------------------
    # Energy estimation (simulation)
    # ------------------------------------------------------------------

    def phase_probabilities(self,
                            num_bits: int,
                            *,
                            ket=None,
                            state_prep: Kernel | None = None) -> np.ndarray:
        """The raw phase-register marginal: ``2^num_bits`` probabilities,
        entry ``j`` for readout ``j`` (little-endian), summing to 1.

        Taken from ``cudaq.get_state`` (deterministic, no shot noise), run
        on a worker thread with a large stack (module docstring). Provide
        exactly one of ``ket`` (array-like / ``cudaq.State``) or
        ``state_prep``. Each eigenvalue of ``H`` shows up at the two mirror
        readouts ``j`` and ``2^m - j``; ``energies`` folds them.
        """
        if (ket is None) == (state_prep is None):
            raise ValueError("provide exactly one of ket or state_prep")
        m = _validate_num_bits(num_bits)
        n_sys = self._encoding.num_system
        n_anc = self._encoding.num_ancilla

        kernel = self.kernel(m, state_prep=state_prep)
        if state_prep is not None:
            state = np.array(run_on_large_stack(cudaq.get_state, kernel))
        else:
            source = ket if isinstance(ket, cudaq.State) else state_from(ket)
            state = np.array(
                run_on_large_stack(cudaq.get_state, kernel, source))

        # Little-endian layout: system (lowest bits) | phase | ancilla.
        reshaped = state.reshape(1 << n_anc, 1 << m, 1 << n_sys)
        return np.sum(np.abs(reshaped)**2, axis=(0, 2))

    def energies(self,
                 num_bits: int,
                 *,
                 ket=None,
                 state_prep: Kernel | None = None,
                 top: int | None = None) -> list[tuple[float, float]]:
        """Estimated energies with probabilities, most probable first.

        Folds the mirror readouts ``j`` and ``2^m - j`` (the same energy)
        into one entry, so the list has ``2^(m-1) + 1`` distinct energies
        whose probabilities sum to 1. Provide exactly one of ``ket``
        (array-like / ``cudaq.State``) or ``state_prep``. ``top`` keeps the
        most probable entries.
        """
        if top is not None and (isinstance(top, bool) or int(top) != top
                                or top < 1):
            raise ValueError(f"top must be a positive integer, got {top!r}")
        m = _validate_num_bits(num_bits)
        probabilities = self.phase_probabilities(m,
                                                 ket=ket,
                                                 state_prep=state_prep)
        half = 1 << (m - 1)
        folded = probabilities[:half + 1].copy()
        folded[1:half] += probabilities[-1:half:-1]
        results = [(self.phase_to_energy(j, m), float(folded[j]))
                   for j in range(half + 1)]
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results if top is None else results[:int(top)]
