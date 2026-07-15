# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Qubitization walks over a block encoding.

Provides the reflection and SELECT observables and a ``Walk`` object that
composes an injected encoding's walk kernels and measures
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

from .block_encoding import mint_cached_kernel
from .common_kernels import (_bit_projector, _validate_control_state,
                             _validate_power, state_from)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .block_encoding import BlockEncoding, Kernel

# ============================================================================
# Observables (system register at qubits [0, ns), ancillas at [ns, ns+na) —
# matching the factory kernels, which allocate the system register first)
# ============================================================================


def reflection_observable(encoding: BlockEncoding) -> cudaq.SpinOperator:
    """R = 2|0...0><0...0| - I on the ancilla register."""
    if encoding.num_ancilla == 0:
        raise ValueError("reflection observable needs at least one ancilla")
    offset = encoding.num_system
    projector = _bit_projector(offset, 0)
    for b in range(1, encoding.num_ancilla):
        projector = projector * _bit_projector(offset + b, 0)
    return 2.0 * projector - spin.i(offset)


# ============================================================================
# The user-facing object
# ============================================================================


class Walk:
    """Qubitization walk over a block encoding.

    Generic over the ``BlockEncoding`` protocol: encoding-specific
    circuits and the odd-moment observable are delegated to the injected
    encoding (``PauliLCU`` is the provided implementation).

    Provides walk/adjoint-walk kernel factories and Chebyshev-moment
    measurement in the QEL even/odd convention. Requires ``num_ancilla >= 1``
    (``PauliLCU`` always satisfies this; the guard defends against
    degenerate foreign encodings). The encoding is fixed at construction —
    kernels and observables are cached per instance.
    """

    def __init__(self, encoding: BlockEncoding) -> None:
        if encoding.num_ancilla == 0:
            raise ValueError(
                "Walk requires an encoding with num_ancilla >= 1 (the "
                "walk's sign comes from a reflection that is a no-op on an "
                "empty register)")
        self._encoding = encoding
        # Encoding factories and observables are immutable per encoding;
        # mint each once and reuse across kernel builds (a moments() sweep
        # otherwise recompiles identical sub-kernels every call).
        self._kernel_cache: dict = {}
        self._observable_cache: dict = {}

    def _encoding_kernel(self, factory_name: str):
        return mint_cached_kernel(self._kernel_cache, self._encoding,
                                  factory_name)

    @property
    def encoding(self):
        """The injected block encoding (read-only: kernels and observables
        are cached against it, so swapping it would serve stale circuits)."""
        return self._encoding

    def __repr__(self) -> str:
        return f"Walk({self.encoding!r})"

    # ------------------------------------------------------------------
    # Kernel factories
    # ------------------------------------------------------------------

    def _factory(self,
                 power: int,
                 uncompute: bool,
                 forward: bool,
                 state_prep: Kernel | None = None) -> Kernel:
        # Delegate every encoding-specific circuit to the injected encoding;
        # this factory only sequences the returned (data-free) kernels.
        n_anc = self._encoding.num_ancilla
        n_sys = self._encoding.num_system
        steps = _validate_power(power)
        prep = self._encoding_kernel("prepare_kernel")
        unprep = self._encoding_kernel("unprepare_kernel")
        step = self._encoding_kernel(
            "walk_step_kernel" if forward else "adjoint_walk_step_kernel")

        if state_prep is not None:
            if uncompute:

                @cudaq.kernel
                def prep_walk_and_uncompute():
                    system = cudaq.qvector(n_sys)
                    state_prep(system)
                    ancilla = cudaq.qvector(n_anc)
                    prep(ancilla)
                    for _ in range(steps):
                        step(ancilla, system)
                    unprep(ancilla)

                return prep_walk_and_uncompute

            @cudaq.kernel
            def prep_walk_prepared():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                prep(ancilla)
                for _ in range(steps):
                    step(ancilla, system)

            return prep_walk_prepared

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

    def kernel(self,
               power: int = 1,
               uncompute: bool = True,
               state_prep: Kernel | None = None) -> Kernel:
        """PREPARE, W^power, optionally UNPREPARE.

        Without ``state_prep`` the returned kernel takes one
        ``cudaq.State`` argument (the input state as data — the
        simulation-friendly form). With ``state_prep`` — a kernel with
        signature ``(qubits: cudaq.qview)`` — the returned kernel takes
        no arguments: it allocates the system register in |0...0>, runs
        ``state_prep`` on it, and applies the walks. ``state_prep`` must
        act only on that register, whose width is ``num_system`` (a
        documented contract; not verifiable at factory time).
        """
        return self._factory(power,
                             uncompute,
                             forward=True,
                             state_prep=state_prep)

    def adjoint_kernel(self,
                       power: int = 1,
                       uncompute: bool = True,
                       state_prep: Kernel | None = None) -> Kernel:
        """PREPARE, (W†)^power, optionally UNPREPARE (see ``kernel``)."""
        return self._factory(power,
                             uncompute,
                             forward=False,
                             state_prep=state_prep)

    def roundtrip_kernel(self,
                         power: int = 1,
                         state_prep: Kernel | None = None) -> Kernel:
        """PREPARE, W^power, (W†)^power, UNPREPARE — the identity, for tests."""
        n_anc = self._encoding.num_ancilla
        steps = _validate_power(power)
        prep = self._encoding_kernel("prepare_kernel")
        unprep = self._encoding_kernel("unprepare_kernel")
        step = self._encoding_kernel("walk_step_kernel")
        adjoint_step = self._encoding_kernel("adjoint_walk_step_kernel")

        n_sys = self._encoding.num_system
        if state_prep is not None:

            @cudaq.kernel
            def prep_roundtrip():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                prep(ancilla)
                for _ in range(steps):
                    step(ancilla, system)
                for _ in range(steps):
                    adjoint_step(ancilla, system)
                unprep(ancilla)

            return prep_roundtrip

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

    def controlled_kernel(self,
                          power: int = 1,
                          control_state: int = 1,
                          uncompute: bool = True,
                          state_prep: Kernel | None = None) -> Kernel:
        """Controlled walks over the system register.

        Input modes as in ``kernel``: without ``state_prep`` the returned
        kernel takes one ``cudaq.State`` argument; with ``state_prep`` it
        takes no arguments and prepares the system register itself (the
        injected prep runs once, uncontrolled, as in QPE). Either way the
        system register is followed by one register holding
        [control, ancillas] (the control cannot share a control set with
        a separate register in CUDA-Q Python). The control qubit is
        initialized to ``control_state``; with control |0> the circuit is
        the identity up to the (cancelling) PREPARE pair.
        """
        n_anc = self._encoding.num_ancilla
        steps = _validate_power(power)
        flip_control = _validate_control_state(control_state) == 1
        prep = self._encoding_kernel("prepare_kernel")
        unprep = self._encoding_kernel("unprepare_kernel")
        controlled_step = self._encoding_kernel("controlled_walk_step_kernel")
        n_sys = self._encoding.num_system

        if state_prep is not None:
            if uncompute:

                @cudaq.kernel
                def prep_controlled_walked():
                    system = cudaq.qvector(n_sys)
                    state_prep(system)
                    control_and_ancilla = cudaq.qvector(1 + n_anc)
                    if flip_control:
                        x(control_and_ancilla[0])
                    prep(control_and_ancilla.back(n_anc))
                    for _ in range(steps):
                        controlled_step(control_and_ancilla, system)
                    unprep(control_and_ancilla.back(n_anc))

                return prep_controlled_walked

            @cudaq.kernel
            def prep_controlled_walked_prepared():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                control_and_ancilla = cudaq.qvector(1 + n_anc)
                if flip_control:
                    x(control_and_ancilla[0])
                prep(control_and_ancilla.back(n_anc))
                for _ in range(steps):
                    controlled_step(control_and_ancilla, system)

            return prep_controlled_walked_prepared

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

    def controlled_roundtrip_kernel(
            self,
            power: int = 1,
            control_state: int = 1,
            state_prep: Kernel | None = None) -> Kernel:
        """Controlled W^power then controlled (W dagger)^power — identity."""
        n_anc = self._encoding.num_ancilla
        steps = _validate_power(power)
        flip_control = _validate_control_state(control_state) == 1
        prep = self._encoding_kernel("prepare_kernel")
        unprep = self._encoding_kernel("unprepare_kernel")
        controlled_step = self._encoding_kernel("controlled_walk_step_kernel")
        controlled_adjoint_step = self._encoding_kernel(
            "controlled_adjoint_walk_step_kernel")
        n_sys = self._encoding.num_system

        if state_prep is not None:

            @cudaq.kernel
            def prep_controlled_roundtrip():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                control_and_ancilla = cudaq.qvector(1 + n_anc)
                if flip_control:
                    x(control_and_ancilla[0])
                prep(control_and_ancilla.back(n_anc))
                for _ in range(steps):
                    controlled_step(control_and_ancilla, system)
                for _ in range(steps):
                    controlled_adjoint_step(control_and_ancilla, system)
                unprep(control_and_ancilla.back(n_anc))

            return prep_controlled_roundtrip

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

    def moment(self,
               ket: ArrayLike | None,
               order: int,
               *,
               state_prep: Kernel | None = None) -> float:
        """Measure the Chebyshev moment <T_order(H/alpha)>.

        The input state is given either as ``ket`` (array-like or an
        already-built ``cudaq.State`` — the simulation-friendly form) or
        as ``state_prep`` (a ``(qubits: cudaq.qview)`` preparation kernel
        composed into the measured circuit — the hardware-shaped form).
        Provide exactly one.
        """
        if (ket is None) == (state_prep is None):
            raise ValueError("provide exactly one of ket or state_prep")
        if int(order) != order or order < 0:
            raise ValueError("order must be a non-negative integer")
        order = int(order)
        power = order // 2
        if order % 2 == 0:
            # Geometry-only (2|0..0><0..0| - I on the ancillas): derivable
            # for any zero-flagged encoding, no encoding hook needed.
            kernel = self.kernel(power=power,
                                 uncompute=True,
                                 state_prep=state_prep)
            if "reflection" not in self._observable_cache:
                self._observable_cache["reflection"] = reflection_observable(
                    self.encoding)
            observable = self._observable_cache["reflection"]
        else:
            # Encoding-specific: delegated to the BlockEncoding hook.
            kernel = self.kernel(power=power,
                                 uncompute=False,
                                 state_prep=state_prep)
            if "select" not in self._observable_cache:
                self._observable_cache["select"] = (
                    self._encoding.select_observable())
            observable = self._observable_cache["select"]
        if state_prep is not None:
            return float(cudaq.observe(kernel, observable).expectation())
        state = ket if isinstance(ket, cudaq.State) else state_from(ket)
        return float(cudaq.observe(kernel, observable, state).expectation())

    def moments(self,
                ket: ArrayLike | None,
                count: int,
                *,
                state_prep: Kernel | None = None) -> list[float]:
        """Measure moments <T_0>, ..., <T_{count-1}> (see ``moment``)."""
        if (ket is None) == (state_prep is None):
            raise ValueError("provide exactly one of ket or state_prep")
        if int(count) != count or count < 0:
            raise ValueError("count must be a non-negative integer")
        if state_prep is not None:
            return [
                self.moment(None, order, state_prep=state_prep)
                for order in range(int(count))
            ]
        state = ket if isinstance(ket, cudaq.State) else state_from(ket)
        return [self.moment(state, order) for order in range(int(count))]
