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
even), so the decoders fold the pair into one entry. An input prepared
near an eigenstate concentrates the readout on that eigenvalue; a general
input spreads over the spectrum weighted by overlap, the usual
phase-estimation behaviour.

The class builds circuits and decodes readouts; executing a circuit is the
caller's choice of executor and target. ``kernel`` returns the circuit, and
with ``measure=True`` the circuit ends by measuring the phase register
alone, so ``cudaq.sample`` on any target returns bitstrings of that
register; ``energies_from_counts`` turns the counts into ranked energies.
The statevector shortcut for simulators — the exact phase-register
marginal — is ``sim_utils.qpe.phase_probabilities`` /
``sim_utils.qpe.energies``.

This is the textbook form: an ``m``-qubit phase register, a controlled
``W^{2^k}`` ladder, and an inverse QFT, costing ``2^m - 1`` controlled walk
steps. The phase register is little-endian (qubit 0 the least significant
bit), matching ``primitives.iqft``; CUDA-Q bitstrings list qubit 0 first,
so a readout string ``b`` is the index ``int(b[::-1], 2)``.

Simulating the ladder uses host stack in proportion to its length: the
compiler re-materializes the walk step's captured tables on the stack on
every loop iteration (about 1 kB per step for a ``PauliLCU``), so from
``m = 13`` the circuit outgrows the default 8 MiB thread stack and the
process dies with a segmentation fault. ``sim_utils.run_on_large_stack``
runs an executor on a worker thread with a large stack; the ``sim_utils``
phase-estimation helpers use it, and a sampled circuit on a simulator needs
it too (``sim_utils.run_on_large_stack(cudaq.sample, circuit, ...)``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import cudaq
import numpy as np

from .block_encoding import mint_cached_kernel
from .primitives import iqft

if TYPE_CHECKING:
    from .block_encoding import BlockEncoding, Kernel

__all__ = ["PhaseEstimation"]


def _validate_num_bits(num_bits: int) -> int:
    if (isinstance(num_bits, bool) or int(num_bits) != num_bits
            or num_bits < 1):
        raise ValueError(
            f"num_bits must be a positive integer, got {num_bits!r}")
    return int(num_bits)


def _validate_top(top) -> int | None:
    if top is None:
        return None
    if isinstance(top, bool) or int(top) != top or top < 1:
        raise ValueError(f"top must be a positive integer, got {top!r}")
    return int(top)


class PhaseEstimation:
    """Textbook quantum phase estimation over a ``BlockEncoding``.

    The phase register holds ``num_bits`` qubits; the energy resolution is
    set by how finely ``2π j / 2^num_bits`` samples the walk eigenphase
    (about ``alpha π / 2^num_bits`` at worst). Composed kernels are cached
    per ``(num_bits, state_prep, measure)``.

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
               state_prep: Kernel | None = None,
               measure: bool = False) -> Kernel:
        """The phase-estimation circuit.

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state``. With ``state_prep`` (a
        ``(qubits: qview)`` kernel): a zero-argument kernel that prepares the
        system register itself. In both the phase + ancilla registers start
        in ``|0...0>``. Register layout: system, then ``num_bits`` phase
        qubits (little-endian: phase qubit 0 is the least significant
        readout bit), then the encoding's ancillas.

        ``measure=False`` leaves the phase register ready to read (take its
        marginal with ``sim_utils.qpe.phase_probabilities``, or measure it
        yourself). ``measure=True`` ends the circuit by measuring the phase
        register alone, so ``cudaq.sample`` returns its bitstrings on any
        target; decode them with ``energies_from_counts``. On a simulator,
        a long ladder needs a large thread stack, about 1 kB per walk step
        (``sim_utils.run_on_large_stack(cudaq.sample, circuit, ...)``;
        module docstring).
        """
        m = _validate_num_bits(num_bits)
        if not isinstance(measure, bool):
            raise ValueError("measure must be a bool")
        key = (m, None if state_prep is None else id(state_prep), measure)
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

        @cudaq.kernel
        def ladder(system: cudaq.qview, work: cudaq.qview):
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

        if state_prep is not None and measure:

            @cudaq.kernel
            def qpe_prepared_measured():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                work = cudaq.qvector(m + n_anc)
                ladder(system, work)
                mz(work[0:m])

            composed = qpe_prepared_measured
        elif state_prep is not None:

            @cudaq.kernel
            def qpe_prepared():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                work = cudaq.qvector(m + n_anc)
                ladder(system, work)

            composed = qpe_prepared
        elif measure:

            @cudaq.kernel
            def qpe_state_measured(state: cudaq.State):
                system = cudaq.qvector(state)
                work = cudaq.qvector(m + n_anc)
                ladder(system, work)
                mz(work[0:m])

            composed = qpe_state_measured
        else:

            @cudaq.kernel
            def qpe_state(state: cudaq.State):
                system = cudaq.qvector(state)
                work = cudaq.qvector(m + n_anc)
                ladder(system, work)

            composed = qpe_state

        self._composed[key] = (composed, state_prep, ladder)
        return composed

    # ------------------------------------------------------------------
    # Classical decoding of readouts
    # ------------------------------------------------------------------

    def phase_probabilities_from_counts(self, counts,
                                        num_bits: int) -> np.ndarray:
        """The phase-register distribution behind sampled bitstrings:
        ``2^num_bits`` relative frequencies, entry ``j`` for readout ``j``.

        ``counts`` is a ``cudaq.SampleResult`` or a mapping from bitstring
        to count. Bitstrings list qubit 0 first. A string of ``num_bits``
        characters is the phase register (``kernel(..., measure=True)``);
        a string over every qubit of the circuit (``kernel`` sampled with
        no explicit measurement) is sliced at the register layout, system
        | phase | ancilla.
        """
        m = _validate_num_bits(num_bits)
        n_sys = self._encoding.num_system
        n_anc = self._encoding.num_ancilla
        items = counts.items() if isinstance(counts, Mapping) else list(
            counts.items())
        probabilities = np.zeros(1 << m)
        total = 0
        for bits, count in items:
            bits = str(bits).replace(" ", "")
            if len(bits) == m:
                phase = bits
            elif len(bits) == n_sys + m + n_anc:
                phase = bits[n_sys:n_sys + m]
            else:
                raise ValueError(
                    f"bitstring {bits!r} has {len(bits)} bits; expected "
                    f"{m} (the phase register) or {n_sys + m + n_anc} "
                    "(every qubit of the circuit)")
            probabilities[int(phase[::-1], 2)] += count
            total += count
        if total <= 0:
            raise ValueError("counts hold no shots")
        return probabilities / total

    def energies_from_probabilities(self,
                                    probabilities,
                                    num_bits: int,
                                    *,
                                    top: int | None = None
                                    ) -> list[tuple[float, float]]:
        """Rank energies from a phase-register distribution.

        Folds the mirror readouts ``j`` and ``2^m - j`` (the same energy)
        into one entry, so the list has ``2^(m-1) + 1`` distinct energies
        whose probabilities sum to that of the input, most probable first.
        ``top`` keeps the most probable entries.
        """
        top = _validate_top(top)
        m = _validate_num_bits(num_bits)
        marginal = np.asarray(probabilities, dtype=float)
        if marginal.shape != (1 << m, ):
            raise ValueError(
                f"probabilities must have shape ({1 << m},) for num_bits="
                f"{m}, got {marginal.shape}")
        half = 1 << (m - 1)
        folded = marginal[:half + 1].copy()
        folded[1:half] += marginal[-1:half:-1]
        results = [(self.phase_to_energy(j, m), float(folded[j]))
                   for j in range(half + 1)]
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results if top is None else results[:top]

    def energies_from_counts(self,
                             counts,
                             num_bits: int,
                             *,
                             top: int | None = None
                             ) -> list[tuple[float, float]]:
        """Ranked energies with probabilities from sampled bitstrings
        (``phase_probabilities_from_counts`` then
        ``energies_from_probabilities``)."""
        return self.energies_from_probabilities(
            self.phase_probabilities_from_counts(counts, num_bits),
            num_bits,
            top=top)
