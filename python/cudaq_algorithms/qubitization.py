# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Qubitization walks over a PauliLCU block encoding.

Provides walk kernels (forward, adjoint, and controlled variants), the
reflection and SELECT observables, and a ``Walk`` object that measures
Chebyshev moments ``<T_k(H/alpha)>`` with the quantum exact Lanczos (QEL)
even/odd convention.

One walk step is SELECT followed by a reflection about the PREPARE state,
the walk block is ``-H/alpha``, and moments are measured as

* even ``k = 2p``:  reflection observable ``2|0..0><0..0| - I`` on the
  ancillas after PREPARE, p walks, UNPREPARE;
* odd ``k = 2p+1``: the SELECT observable after PREPARE and p walks
  (no UNPREPARE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cudaq
from cudaq import spin

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .block_encoding import BlockEncoding
    from .pauli_lcu import Kernel

from .pauli_lcu import (PauliLCU, controlled_select, prepare,
                        reflect_about_zero, select, unprepare, walk)

FORWARD = 0
ADJOINT = 1


# ============================================================================
# Device kernels
# ============================================================================


@cudaq.kernel
def reflect_about_prepare(ancilla: cudaq.qview, angles: list[float]):
    """Reflect about the PREPARE state: PREPARE†, zero reflection, PREPARE."""
    unprepare(ancilla, angles)
    reflect_about_zero(ancilla)
    prepare(ancilla, angles)


@cudaq.kernel
def adjoint_walk(ancilla: cudaq.qview, system: cudaq.qview,
                 angles: list[float], term_controls: list[int],
                 term_ops: list[int], term_lengths: list[int],
                 term_signs: list[int]):
    """Adjoint walk step: reflection first, then SELECT (both self-adjoint)."""
    reflect_about_prepare(ancilla, angles)
    select(ancilla, system, term_controls, term_ops, term_lengths, term_signs)


@cudaq.kernel
def controlled_reflect_about_zero(control_and_register: cudaq.qview):
    """Zero-state reflection on qubits 1.. controlled by qubit 0.

    Qubit 0 of ``control_and_register`` is the external control (see
    controlled_select for why the control shares a register).
    """
    total = control_and_register.size()
    n = total - 1
    for i in range(n):
        x(control_and_register[1 + i])
    if n == 0:
        z(control_and_register[0])
    else:
        z.ctrl(control_and_register.front(total - 1),
               control_and_register[total - 1])
    for i in range(n):
        x(control_and_register[1 + i])


@cudaq.kernel
def controlled_reflect_about_prepare(control_and_ancilla: cudaq.qview,
                                     angles: list[float]):
    """PREPARE-state reflection controlled by qubit 0.

    The PREPARE / PREPARE-dagger pair stays uncontrolled (it cancels when
    the control is |0>); only the zero reflection is controlled.
    """
    n = control_and_ancilla.size() - 1
    unprepare(control_and_ancilla.back(n), angles)
    controlled_reflect_about_zero(control_and_ancilla)
    prepare(control_and_ancilla.back(n), angles)


@cudaq.kernel
def controlled_walk(control_and_ancilla: cudaq.qview, system: cudaq.qview,
                    angles: list[float], term_controls: list[int],
                    term_ops: list[int], term_lengths: list[int],
                    term_signs: list[int]):
    """One walk step controlled by qubit 0 of ``control_and_ancilla``."""
    controlled_select(control_and_ancilla, system, term_controls, term_ops,
                      term_lengths, term_signs)
    controlled_reflect_about_prepare(control_and_ancilla, angles)


@cudaq.kernel
def controlled_adjoint_walk(control_and_ancilla: cudaq.qview,
                            system: cudaq.qview, angles: list[float],
                            term_controls: list[int], term_ops: list[int],
                            term_lengths: list[int], term_signs: list[int]):
    """One adjoint walk step controlled by qubit 0."""
    controlled_reflect_about_prepare(control_and_ancilla, angles)
    controlled_select(control_and_ancilla, system, term_controls, term_ops,
                      term_lengths, term_signs)


# ============================================================================
# Observables (system register at qubits [0, ns), ancillas at [ns, ns+na) —
# matching the factory kernels, which allocate the system register first)
# ============================================================================


def _bit_projector(qubit: int, bit: int) -> cudaq.SpinOperator:
    """|bit><bit| on one qubit as a spin operator."""
    sign = 1.0 - 2.0 * float(bit)
    return 0.5 * (spin.i(qubit) + sign * spin.z(qubit))


def reflection_observable(encoding: PauliLCU) -> cudaq.SpinOperator:
    """R = 2|0...0><0...0| - I on the ancilla register."""
    if encoding.num_ancilla == 0:
        raise ValueError("reflection observable needs at least one ancilla")
    offset = encoding.num_system
    projector = _bit_projector(offset, 0)
    for b in range(1, encoding.num_ancilla):
        projector = projector * _bit_projector(offset + b, 0)
    return 2.0 * projector - spin.i(offset)


def select_observable(encoding: PauliLCU) -> cudaq.SpinOperator:
    """The SELECT operator sum_i sign_i |i><i|_anc x P_i as an observable."""
    if encoding.num_ancilla == 0:
        raise ValueError("select observable needs at least one ancilla")
    offset = encoding.num_system
    n_anc = encoding.num_ancilla

    observable = None
    for index, (coefficient, word) in enumerate(encoding.terms):
        term = 1.0 if coefficient >= 0.0 else -1.0
        for b in range(n_anc):
            bit = (index >> (n_anc - 1 - b)) & 1
            term = term * _bit_projector(offset + b, bit)
        for qubit, label in enumerate(word):
            if label == "X":
                term = term * spin.x(qubit)
            elif label == "Y":
                term = term * spin.y(qubit)
            elif label == "Z":
                term = term * spin.z(qubit)
        observable = term if observable is None else observable + term
    return observable


# ============================================================================
# The user-facing object
# ============================================================================


class Walk:
    """Qubitization walk over a block encoding.

    Generic over the ``BlockEncoding`` protocol: encoding-specific
    circuits and the odd-moment observable are delegated to the injected
    encoding (``PauliLCU`` is the provided implementation).

    Provides walk/adjoint-walk kernel factories and Chebyshev-moment
    measurement in the QEL even/odd convention. Requires a non-degenerate
    encoding (at least one ancilla, i.e. two or more LCU terms).
    """

    def __init__(self, encoding: BlockEncoding) -> None:
        if encoding.num_ancilla == 0:
            raise ValueError(
                "Walk requires an encoding with at least one ancilla "
                "(two or more LCU terms); a single-term encoding is the "
                "signed Pauli word itself")
        self.encoding = encoding

    def __repr__(self) -> str:
        return f"Walk({self.encoding!r})"

    # ------------------------------------------------------------------
    # Kernel factories
    # ------------------------------------------------------------------

    def _factory(self, power: int, uncompute: bool,
                 forward: bool) -> Kernel:
        # Delegate every encoding-specific circuit to the injected encoding;
        # this factory only sequences the returned (data-free) kernels.
        n_anc = self.encoding.num_ancilla
        steps = int(power)
        prep = self.encoding.prepare_kernel()
        unprep = self.encoding.unprepare_kernel()
        step = (self.encoding.walk_step_kernel()
                if forward else self.encoding.adjoint_walk_step_kernel())

        if uncompute:

            @cudaq.kernel
            def walk_and_uncompute(state: cudaq.State):
                system = cudaq.qvector(state)
                ancilla = cudaq.qvector(n_anc)
                prep(ancilla)
                for _ in range(steps):
                    step(ancilla, system)
                unprep(ancilla)

            return walk_and_uncompute

        @cudaq.kernel
        def walk_prepared(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prep(ancilla)
            for _ in range(steps):
                step(ancilla, system)

        return walk_prepared

    def kernel(self, power: int = 1, uncompute: bool = True) -> Kernel:
        """``@cudaq.kernel(state)``: PREPARE, W^power, optionally UNPREPARE."""
        return self._factory(power, uncompute, forward=True)

    def adjoint_kernel(self, power: int = 1,
                       uncompute: bool = True) -> Kernel:
        """``@cudaq.kernel(state)``: PREPARE, (W†)^power, optionally UNPREPARE."""
        return self._factory(power, uncompute, forward=False)

    def roundtrip_kernel(self, power: int = 1) -> Kernel:
        """PREPARE, W^power, (W†)^power, UNPREPARE — the identity, for tests."""
        n_anc = self.encoding.num_ancilla
        steps = int(power)
        prep = self.encoding.prepare_kernel()
        unprep = self.encoding.unprepare_kernel()
        step = self.encoding.walk_step_kernel()
        adjoint_step = self.encoding.adjoint_walk_step_kernel()

        @cudaq.kernel
        def roundtrip(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prep(ancilla)
            for _ in range(steps):
                step(ancilla, system)
            for _ in range(steps):
                adjoint_step(ancilla, system)
            unprep(ancilla)

        return roundtrip

    def controlled_kernel(self, power: int = 1, control_state: int = 1,
                          uncompute: bool = True) -> Kernel:
        """``@cudaq.kernel(state)`` running controlled walks.

        Allocates the system register from ``state``, then one register
        holding [control, ancillas] (the control cannot share a control set
        with a separate register in CUDA-Q Python). The control qubit is
        initialized to ``control_state``; with control |0> the circuit is
        the identity up to the (cancelling) PREPARE pair.
        """
        n_anc = self.encoding.num_ancilla
        steps = int(power)
        flip_control = int(control_state) == 1
        prep = self.encoding.prepare_kernel()
        unprep = self.encoding.unprepare_kernel()
        controlled_step = self.encoding.controlled_walk_step_kernel()

        if uncompute:

            @cudaq.kernel
            def controlled_walked(state: cudaq.State):
                system = cudaq.qvector(state)
                control_and_ancilla = cudaq.qvector(1 + n_anc)
                if flip_control:
                    x(control_and_ancilla[0])
                prep(control_and_ancilla.back(n_anc))
                for _ in range(steps):
                    controlled_step(control_and_ancilla, system)
                unprep(control_and_ancilla.back(n_anc))

            return controlled_walked

        @cudaq.kernel
        def controlled_walked_prepared(state: cudaq.State):
            system = cudaq.qvector(state)
            control_and_ancilla = cudaq.qvector(1 + n_anc)
            if flip_control:
                x(control_and_ancilla[0])
            prep(control_and_ancilla.back(n_anc))
            for _ in range(steps):
                controlled_step(control_and_ancilla, system)

        return controlled_walked_prepared

    def controlled_roundtrip_kernel(self, power: int = 1,
                                    control_state: int = 1) -> Kernel:
        """Controlled W^power then controlled (W dagger)^power — identity."""
        n_anc = self.encoding.num_ancilla
        steps = int(power)
        flip_control = int(control_state) == 1
        prep = self.encoding.prepare_kernel()
        unprep = self.encoding.unprepare_kernel()
        controlled_step = self.encoding.controlled_walk_step_kernel()
        controlled_adjoint_step = (
            self.encoding.controlled_adjoint_walk_step_kernel())

        @cudaq.kernel
        def controlled_roundtrip(state: cudaq.State):
            system = cudaq.qvector(state)
            control_and_ancilla = cudaq.qvector(1 + n_anc)
            if flip_control:
                x(control_and_ancilla[0])
            prep(control_and_ancilla.back(n_anc))
            for _ in range(steps):
                controlled_step(control_and_ancilla, system)
            for _ in range(steps):
                controlled_adjoint_step(control_and_ancilla, system)
            unprep(control_and_ancilla.back(n_anc))

        return controlled_roundtrip

    # ------------------------------------------------------------------
    # Moment measurement (simulation-friendly, but observable-based:
    # the same circuits and operators run on hardware)
    # ------------------------------------------------------------------

    def moment(self, ket: ArrayLike, order: int) -> float:
        """Measure the Chebyshev moment <T_order(H/alpha)> for |ket>."""
        from .pauli_lcu import state_from

        order = int(order)
        if order < 0:
            raise ValueError("order must be non-negative")
        power = order // 2
        if order % 2 == 0:
            # Geometry-only (2|0..0><0..0| - I on the ancillas): derivable
            # for any zero-flagged encoding, no encoding hook needed.
            kernel = self.kernel(power=power, uncompute=True)
            observable = reflection_observable(self.encoding)
        else:
            # Encoding-specific: delegated to the BlockEncoding hook.
            kernel = self.kernel(power=power, uncompute=False)
            observable = self.encoding.select_observable()
        state = state_from(ket)
        return float(cudaq.observe(kernel, observable, state).expectation())

    def moments(self, ket: ArrayLike, count: int) -> list[float]:
        """Measure moments <T_0>, ..., <T_{count-1}> for |ket>."""
        return [self.moment(ket, order) for order in range(int(count))]
