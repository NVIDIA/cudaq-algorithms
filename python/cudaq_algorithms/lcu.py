# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear-combination-of-unitaries (LCU) block encoding from term data.

``LCUBlockEncoding`` block-encodes ``H / alpha`` for a Hamiltonian written
as a weighted sum of Hermitian involutory unitaries ``H = sum_k c_k U_k``.
The caller supplies the term *weights* ``|c_k|`` and a SELECT *body* per
term describing ``U_k`` as a gate list; the encoding assembles an
alias-sampled PREPARE over the weights and a unary-iteration SELECT over
the term unitaries. ``alpha`` is the term one-norm ``sum_k |c_k|``.

The ``U_A = PREPARE-dagger . SELECT . PREPARE`` structure is the standard
LCU block encoding (Childs and Wiebe, "Hamiltonian simulation using linear
combinations of unitary operations", arXiv:1202.5822; Berry, Childs,
Cleve, Kothari and Somma, "Simulating Hamiltonian dynamics with a
truncated Taylor series", arXiv:1412.4687): PREPARE loads the term weights
as amplitudes, SELECT applies the indexed term unitary, and the un-PREPARE
projects the weighted sum into the block. The PREPARE (coherent alias
sampling) and SELECT (unary iteration) are the
``cudaq_algorithms.primitives`` gadgets of Babbush et al. (arXiv:1805.03662),
cited where they are implemented.

Term unitaries and SELECT bodies
--------------------------------

Each ``U_k`` must be a Hermitian involution (``U_k^2 = I``), so SELECT is a
Hermitian involution and the whole encoding is exactly self-adjoint (see
below). The term body is a list of gate ops in the unary-iteration
instruction vocabulary (see
``cudaq_algorithms.primitives._unary_iteration``); the leaf-controlled ops
``("x" | "y" | "z", target)`` apply a Pauli to ``system[target]`` when the
index register holds ``k``, ``("sign",)`` carries a ``-1`` phase (a
negative ``c_k``), and the free / AND-ladder ops build multi-controlled
term unitaries using a ``work`` scratch register. A term whose body is
empty is the identity; a Pauli word ``P_k`` is the list of its non-identity
single-qubit factors.

Circuit
-------

``U_A = PREPARE-dagger . SELECT . PREPARE`` over ``ancilla = [index(m) |
garbage(g) | select_work(w)]``:

- PREPARE / PREPARE-dagger: ``AliasSamplingPrepare`` over the term weights
  (with garbage — sound only in this symmetric sandwich; see
  ``cudaq_algorithms.primitives._alias_sampling``). Never placed under
  ``cudaq.control``.
- SELECT: a flat unary-iteration walk
  (``cudaq_algorithms.primitives._unary_iteration``) over the term index,
  applying ``U_k`` controlled on index ``k``. SELECT's ladder reuses the
  PREPARE QROM ladder qubits (both are clean between uses); any AND-ladder
  scratch sits in dedicated clean ``select_work`` at the ancilla tail
  (CUDA-Q cannot deallocate mid-circuit, so it is counted in
  ``num_ancilla``).

Self-adjointness: every term unitary squares to the identity and SELECT is
block-diagonal over the index, so ``SELECT^2 = I`` and ``U_A =
P-dagger S P`` is exactly self-adjoint and involutory **on states whose
ancilla register enters in ``|0...0>``** — the domain every consumer
supplies. Walk reflections cover the **full** ancilla register, garbage
included, and this is a requirement, not a convenience: the block is the
all-``|0...0>`` ancilla subspace, and because SELECT leaves the PREPARE
garbage entangled with the system the index-``|0>`` sector is *not*
garbage-free, so an index-only reflection would misidentify the block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import cudaq
import numpy as np

from .common_kernels import (_validate_power, controlled_reflect_about_zero,
                             reflect_about_zero)
from .primitives import AliasSamplingPrepare, unary_iteration_kernels

if TYPE_CHECKING:
    from .block_encoding import Kernel

__all__ = ["LCUBlockEncoding"]

# Minted kernels are closures; keep references so CUDA-Q does not collect
# them while a returned factory kernel is still in use.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


@cudaq.kernel
def _noop(register: cudaq.qview):
    pass


# Body ops that consume a ``work`` operand, and the position of the work
# index within the op tuple (see the unary-iteration vocabulary).
_WORK_OPS = {"and_tt": 3, "and_wt": 3, "copy_tw": 2}


def _required_work(bodies: Sequence[Sequence[tuple]]) -> int:
    """The ``work`` width the bodies need (>= 1: an empty work view must
    never cross the kernel boundary, cuda-quantum#4847)."""
    width = 1
    for body in bodies:
        for item in body:
            slot = _WORK_OPS.get(item[0])
            if slot is not None:
                width = max(width, int(item[slot]) + 1)
    return width


class LCUBlockEncoding:
    """Block encoding of ``H / alpha`` for ``H = sum_k c_k U_k``.

    Parameters
    ----------
    num_system
        System-register width in qubits (the ``U_k`` act on
        ``system[0:num_system]``).
    weights
        The term weights ``|c_k|``: non-negative and finite, with a
        positive sum. Signs ``c_k < 0`` are carried on the SELECT branch
        (a ``("sign",)`` op in the body), never on the weights.
    bodies
        One SELECT body per term (same length as ``weights``), each a list
        of unary-iteration gate ops realizing the Hermitian involution
        ``U_k``.
    mu
        Alias-table keep precision in bits (>= 1); the per-term probability
        rounding is bounded by ``discretization_bound``.
    num_work
        SELECT work-scratch width. Defaults to the minimum the bodies
        require; a larger value is accepted, a smaller one is rejected.
    select_observable
        Optional ``cudaq.SpinOperator`` for the odd-Chebyshev-moment
        observable (available only when the ``U_k`` are computational-frame
        Pauli words); ``select_observable()`` raises without it.

    Satisfies the ``BlockEncoding`` protocol (``select_observable`` only
    when supplied above). Measure Chebyshev data through ``walk_kernel``
    powers; ``Walk.moment`` even moments need an ``2^num_ancilla``-term
    reflection observable, astronomical at these ancilla widths.
    """

    def __init__(self,
                 num_system: int,
                 weights: Sequence[float],
                 bodies: Sequence[Sequence[tuple]],
                 *,
                 mu: int = 8,
                 num_work: int | None = None,
                 select_observable: Any | None = None):
        num_system = int(num_system)
        if num_system < 1:
            raise ValueError(f"num_system must be >= 1, got {num_system}")
        weights = [float(w) for w in weights]
        bodies = [list(body) for body in bodies]
        if len(weights) != len(bodies):
            raise ValueError(
                f"weights and bodies must have the same length, got "
                f"{len(weights)} and {len(bodies)}")
        if not weights:
            raise ValueError("no terms: weights and bodies are empty")

        required_work = _required_work(bodies)
        if num_work is None:
            num_work = required_work
        else:
            num_work = int(num_work)
            if num_work < required_work:
                raise ValueError(
                    f"num_work={num_work} is too small; the bodies need at "
                    f"least {required_work} work qubits")

        # Pin the "select" QROM variant: it keeps ladder_offset / num_ladder
        # == num_index, the geometry the apply kernel relies on when SELECT
        # reuses the PREPARE QROM ladder.
        preparation = AliasSamplingPrepare(weights, mu, variant="select")
        select = unary_iteration_kernels(preparation.num_index,
                                         len(bodies),
                                         lambda k: bodies[k],
                                         include_adjoint=False,
                                         num_work=num_work)
        controlled_select = unary_iteration_kernels(preparation.num_index,
                                                    len(bodies),
                                                    lambda k: bodies[k],
                                                    controlled=True,
                                                    include_adjoint=False,
                                                    num_work=num_work)

        self._num_system = num_system
        self._weights = tuple(weights)
        self._mu = int(mu)
        self._preparation = preparation
        self._num_work = num_work
        self._select = select
        self._controlled_select = controlled_select
        self._select_observable = select_observable
        self._build_kernels()

    # ------------------------------------------------------------------
    # Kernel construction (all data captured here, factories are data-free)
    # ------------------------------------------------------------------

    def _build_kernels(self) -> None:
        # Unpack into scalar locals: tuples (and self) cannot be
        # closure-captured into kernels.
        m = self._preparation.num_index
        g = self._preparation.num_garbage
        l0 = self._preparation.ladder_offset
        w = self._num_work
        prepare = self._preparation.kernel()
        unprepare = self._preparation.adjoint_kernel()
        select = self._select.kernel
        controlled_select = self._controlled_select.kernel

        @cudaq.kernel
        def lcu_apply(ancilla: cudaq.qview, system: cudaq.qview):
            """U_A = PREPARE-dagger SELECT PREPARE (self-adjoint,
            involutory). SELECT's ladder reuses the PREPARE QROM ladder
            (clean between the sandwich halves)."""
            prepare(ancilla.front(m), ancilla[m:m + g])
            select(ancilla.front(m), ancilla[m + l0:m + l0 + m], system,
                   ancilla[m + g:m + g + w])
            unprepare(ancilla.front(m), ancilla[m:m + g])

        @cudaq.kernel
        def lcu_controlled_apply(control_and_ancilla: cudaq.qview,
                                 system: cudaq.qview):
            """Controlled U_A = PREPARE-dagger (controlled SELECT) PREPARE.

            Only the Hermitian middle factor carries the external control
            (the uncontrolled PREPARE pair cancels at control ``|0>``); the
            control reaches SELECT as a one-qubit view, so no control set
            mixes a qview with a bare qubit, and PREPARE — which calls
            sub-kernels — never sits under a control variant.
            """
            prepare(control_and_ancilla[1:1 + m],
                    control_and_ancilla[1 + m:1 + m + g])
            controlled_select(control_and_ancilla.front(1),
                              control_and_ancilla[1:1 + m],
                              control_and_ancilla[1 + m + l0:1 + m + l0 + m],
                              system,
                              control_and_ancilla[1 + m + g:1 + m + g + w])
            unprepare(control_and_ancilla[1:1 + m],
                      control_and_ancilla[1 + m:1 + m + g])

        @cudaq.kernel
        def lcu_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W = R U_A (block encodes -H/alpha); R covers the full
            ancilla register, garbage included (sound in the sandwiched
            frame — see the module docstring)."""
            lcu_apply(ancilla, system)
            reflect_about_zero(ancilla)

        @cudaq.kernel
        def lcu_adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W-dagger = U_A R (U_A is self-adjoint)."""
            reflect_about_zero(ancilla)
            lcu_apply(ancilla, system)

        @cudaq.kernel
        def lcu_controlled_walk_step(control_and_ancilla: cudaq.qview,
                                     system: cudaq.qview):
            lcu_controlled_apply(control_and_ancilla, system)
            controlled_reflect_about_zero(control_and_ancilla)

        @cudaq.kernel
        def lcu_controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                             system: cudaq.qview):
            controlled_reflect_about_zero(control_and_ancilla)
            lcu_controlled_apply(control_and_ancilla, system)

        _retain(lcu_apply, lcu_controlled_apply, lcu_walk_step,
                lcu_adjoint_walk_step, lcu_controlled_walk_step,
                lcu_controlled_adjoint_walk_step)
        self._apply = lcu_apply
        self._controlled_apply = lcu_controlled_apply
        self._walk_step = lcu_walk_step
        self._adjoint_walk_step = lcu_adjoint_walk_step
        self._controlled_walk_step = lcu_controlled_walk_step
        self._controlled_adjoint_walk_step = lcu_controlled_adjoint_walk_step

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def num_system(self) -> int:
        return self._num_system

    @property
    def num_ancilla(self) -> int:
        """All ancillas: index + PREPARE garbage + SELECT work scratch.

        The block is the all-ancilla-``|0...0>`` subspace, so consumers
        reflect about the **full** register, not the index alone: after
        SELECT the garbage is entangled with the system, so the
        index-``|0>`` sector still carries nonzero garbage and an
        index-only reflection would misidentify the block. The SELECT work
        scratch *is* ``|0>`` at reflection points but is still counted here
        because CUDA-Q cannot deallocate it mid-circuit.
        """
        return (self._preparation.num_index + self._preparation.num_garbage +
                self._num_work)

    @property
    def num_index(self) -> int:
        """Term-index register width (the block ancillas proper)."""
        return self._preparation.num_index

    @property
    def num_garbage(self) -> int:
        """PREPARE garbage width (entangled between the sandwich halves)."""
        return self._preparation.num_garbage

    @property
    def num_select_work(self) -> int:
        """Clean SELECT work scratch (``|0>`` at reflection points)."""
        return self._num_work

    @property
    def num_terms(self) -> int:
        return len(self._weights)

    @property
    def mu(self) -> int:
        return self._mu

    @property
    def weights(self) -> tuple:
        """The term weights ``|c_k|`` in PREPARE bin order (bins
        ``>= num_terms`` are zero-weight padding acting as the identity)."""
        return self._weights

    @property
    def alpha(self) -> float:
        """The term one-norm ``sum_k |c_k|``.

        Exact for the discretized weights too: ``table_probabilities`` sum
        to 1 by construction, so ``sum(discretized_weights)`` equals this
        ideal one-norm identically.
        """
        return self._preparation.lam

    @property
    def discretized_weights(self) -> np.ndarray:
        """The exactly realized per-bin weights ``alpha *
        table_probabilities`` (the dense-reference anchor)."""
        return self.alpha * self._preparation.table_probabilities

    @property
    def discretization_bound(self) -> float:
        """Per-bin probability rounding bound (see
        ``cudaq_algorithms.primitives._alias_sampling``)."""
        return self._preparation.discretization_bound

    @property
    def preparation(self) -> AliasSamplingPrepare:
        """The underlying alias-sampling PREPARE (tables and bounds)."""
        return self._preparation

    def __repr__(self) -> str:
        return (f"LCUBlockEncoding(terms={self.num_terms}, mu={self.mu}, "
                f"system_qubits={self.num_system}, "
                f"ancilla_qubits={self.num_ancilla} "
                f"(index {self.num_index} + garbage {self.num_garbage} + "
                f"work {self.num_select_work}), alpha={self.alpha:.6g})")

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    def encode_kernel(self, state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying the full block encoding.

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state`` and the ancilla register (in
        ``|0...0>``) after it. With ``state_prep`` (a ``(qubits: qview)``
        kernel): a zero-argument kernel that allocates the system register
        in ``|0...0>``, runs ``state_prep`` on it, then applies the encoding.
        """
        apply_u = self._apply
        n_anc = self.num_ancilla
        n_sys = self.num_system

        if state_prep is not None:

            @cudaq.kernel
            def lcu_prep_encoded():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                apply_u(ancilla, system)

            _retain(lcu_prep_encoded)
            return lcu_prep_encoded

        @cudaq.kernel
        def lcu_encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            apply_u(ancilla, system)

        _retain(lcu_encoded)
        return lcu_encoded

    def walk_kernel(self,
                    power: int = 1,
                    state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying ``power`` qubitization walk steps.

        The all-zero-ancilla block of the result is T_power(-H/alpha)
        applied to the input state. Input modes as in ``encode_kernel``.
        """
        step = self._walk_step
        n_anc = self.num_ancilla
        n_sys = self.num_system
        steps = _validate_power(power)

        if state_prep is not None:

            @cudaq.kernel
            def lcu_prep_walked():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                for _ in range(steps):
                    step(ancilla, system)

            _retain(lcu_prep_walked)
            return lcu_prep_walked

        @cudaq.kernel
        def lcu_walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            for _ in range(steps):
                step(ancilla, system)

        _retain(lcu_walked)
        return lcu_walked

    # ------------------------------------------------------------------
    # BlockEncoding protocol: data-free kernel factories
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: trivial — the alias-sampling PREPARE is
        part of ``apply_kernel``'s symmetric sandwich (its garbage is only
        sound there, never left dirty across consumer-visible points)."""
        return _noop

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: trivial (see ``prepare_kernel``)."""
        return _noop

    def apply_kernel(self) -> Kernel:
        """``(ancilla, system)``: the full block encoding U_A."""
        return self._apply

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0."""
        return self._controlled_apply

    def walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one qubitization walk step W = R U_A."""
        return self._walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W-dagger."""
        return self._adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        return self._controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled adjoint step."""
        return self._controlled_adjoint_walk_step

    # ------------------------------------------------------------------
    # Observable hooks
    # ------------------------------------------------------------------

    def select_observable(self) -> Any:
        """The odd-Chebyshev-moment observable, if one was supplied.

        Available only when the term unitaries are computational-frame
        Pauli words (so ``sum_k sign_k |k><k| (x) P_k`` is a
        ``cudaq.SpinOperator``); raises otherwise.
        """
        if self._select_observable is None:
            raise NotImplementedError(
                "LCUBlockEncoding was built without select_observable: the "
                "odd-Chebyshev-moment observable is available only when the "
                "term unitaries are computational-frame Pauli words")
        return self._select_observable
