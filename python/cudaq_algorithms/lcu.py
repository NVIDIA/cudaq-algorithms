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

Walk frame: the qubitization walk step is ``W = PREPARE . R . PREPARE-dagger
. SELECT`` with ``R`` the reflection about ancilla ``|0...0>`` — SELECT,
then reflect about the PREPARE state — exactly as ``PauliLCU`` defines it,
so the ``prepare_kernel`` / ``unprepare_kernel`` hooks are the alias
PREPARE and its adjoint and ``Walk`` composes ``PREPARE . W^p .
PREPARE-dagger = (R U_A)^p``. In this frame the protocol's odd-moment
observable is the SELECT operator itself, ``sum_k sign_k |k><k|_index (x)
P_k`` on the index and system registers (identity on the garbage and
work), available whenever every body is a Pauli word; it agrees with the
SELECT circuit on every state the walk produces, because those states keep
the ladder and work scratch clean.

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

import numbers
from typing import TYPE_CHECKING, Any, Sequence

import cudaq
import numpy as np

from .common_kernels import (_bit_projector, _validate_power,
                             controlled_reflect_about_zero, reflect_about_zero)
from .primitives import AliasSamplingPrepare, unary_iteration_kernels
from .primitives._unary_iteration import _BODY_OPCODES, _EXTENDED_BODY_GATES

if TYPE_CHECKING:
    from .block_encoding import Kernel

__all__ = ["LCUBlockEncoding"]

# Minted kernels are closures; keep references so CUDA-Q does not collect
# them while a returned factory kernel is still in use.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


def _as_count(name: str, value, minimum: int) -> int:
    """Require an integral number ``>= minimum`` (no bools, no silent
    truncation of fractional values)."""
    if (isinstance(value,
                   (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or int(value) != value):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _is_index(value) -> bool:
    """True for a non-negative integral number (bools excluded)."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real) or not np.isfinite(value)):
        return False
    return int(value) == value and value >= 0


# ``sigma_then . sigma_first`` for distinct Paulis, ``first`` applied first:
# (phase, result label).
_PAULI_COMPOSE = {
    ("x", "y"): (-1j, "z"),
    ("y", "x"): (1j, "z"),
    ("y", "z"): (-1j, "x"),
    ("z", "y"): (1j, "x"),
    ("z", "x"): (-1j, "y"),
    ("x", "z"): (1j, "y"),
}


def _compose_pauli(first: str, then: str) -> tuple[complex, str]:
    """``sigma_then . sigma_first`` as ``(phase, label)``, ``"i"`` the
    identity."""
    if first == "i":
        return 1.0, then
    if first == then:
        return 1.0, "i"
    return _PAULI_COMPOSE[(first, then)]


def _operand_kinds(gate: str) -> tuple | None:
    """Operand kinds of a SELECT body op (``"t"`` = system target index,
    ``"w"`` = work index), from the unary-iteration vocabulary tables;
    ``None`` for an unknown op name."""
    if gate in _BODY_OPCODES:
        return ("t", )
    if gate in _EXTENDED_BODY_GATES:
        return _EXTENDED_BODY_GATES[gate][1]
    return None


def _validate_bodies(bodies: Sequence[Sequence[tuple]],
                     num_system: int) -> tuple[int, list]:
    """Check every body op at the factory boundary.

    Returns the ``work`` width the bodies need and, per body, its reduced
    Pauli word ``(sign, {qubit: "x" | "y" | "z"})`` when the body uses only
    the leaf-controlled Pauli and ``("sign",)`` ops, else ``None``.

    Each op must be a known unary-iteration body op with the right number
    of operands, every operand a non-negative integer, and every system
    target index ``< num_system`` (the primitive cannot check the width
    itself). A Pauli-word body whose ops compose to a ``+-i`` phase is
    rejected: its term unitary would not be Hermitian. The returned width
    is ``>= 1``: an empty work view must never cross the kernel boundary
    (cuda-quantum#4847).
    """
    width = 1
    words = []
    for k, body in enumerate(bodies):
        phase = 1.0 + 0.0j
        word: dict = {}
        pauli_only = True
        for i, item in enumerate(body):
            malformed = (not isinstance(item, (tuple, list)) or not item
                         or not isinstance(item[0], str))
            if malformed:
                raise ValueError(f"body {k}, op {i}: expected a tuple "
                                 f"('name', operands...), got {item!r}")
            gate, operands = item[0], item[1:]
            kinds = _operand_kinds(gate)
            if kinds is None:
                raise ValueError(f"body {k}, op {i}: unknown SELECT body op "
                                 f"{gate!r}")
            if len(operands) != len(kinds):
                raise ValueError(
                    f"body {k}, op {i}: {gate!r} takes {len(kinds)} "
                    f"operands, got {len(operands)}")
            for kind, operand in zip(kinds, operands):
                name = "target" if kind == "t" else "work"
                if not _is_index(operand):
                    raise ValueError(
                        f"body {k}, op {i}: {gate!r} {name} index must be a "
                        f"non-negative integer, got {operand!r}")
                if kind == "t" and operand >= num_system:
                    raise ValueError(
                        f"body {k}, op {i}: {gate!r} targets system qubit "
                        f"{int(operand)}, but num_system = {num_system}")
                if kind == "w":
                    width = max(width, int(operand) + 1)
            if gate in _BODY_OPCODES:
                target = int(operands[0])
                factor, label = _compose_pauli(word.get(target, "i"), gate)
                phase *= factor
                if label == "i":
                    word.pop(target, None)
                else:
                    word[target] = label
            elif gate == "sign":
                phase = -phase
            else:
                pauli_only = False
        if pauli_only:
            if phase.imag != 0.0:
                unit = "-i" if phase.imag < 0.0 else "i"
                raise ValueError(
                    f"body {k}: its Pauli ops compose to {unit} times a "
                    f"Pauli word, which is not Hermitian; a term unitary "
                    f"must be a Hermitian involution")
            words.append((int(round(phase.real)), word))
        else:
            words.append(None)
    return width, words


class LCUBlockEncoding:
    """Block encoding of ``H / alpha`` for ``H = sum_k c_k U_k``.

    Satisfies the full ``BlockEncoding`` protocol. ``select_observable``
    (the odd-Chebyshev-moment observable ``Walk.moment`` uses) is derived
    from the bodies when every term is a Pauli word (``"x"``, ``"y"``,
    ``"z"`` and ``("sign",)`` ops only) and raises ``NotImplementedError``
    for bodies using the extended vocabulary. Even ``Walk.moment`` orders
    use the geometry-only reflection observable, whose ``2^num_ancilla``
    Pauli terms are practical only at small ``mu``; dense Chebyshev data
    is also available through ``walk_kernel`` powers.

    Every body op is validated at construction (known op, operand count,
    system targets ``< num_system``, and a Hermitian product for
    Pauli-word bodies); the encoding raises ``ValueError`` rather than
    failing inside a kernel.

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
    num_select_work
        SELECT work-scratch width. Defaults to the minimum the bodies
        require; a larger value is accepted, a smaller one is rejected.
    """

    def __init__(self,
                 num_system: int,
                 weights: Sequence[float],
                 bodies: Sequence[Sequence[tuple]],
                 *,
                 mu: int = 8,
                 num_select_work: int | None = None):
        num_system = _as_count("num_system", num_system, 1)
        mu = _as_count("mu", mu, 1)
        weights = [float(w) for w in weights]
        bodies = [list(body) for body in bodies]
        if len(weights) != len(bodies):
            raise ValueError(
                f"weights and bodies must have the same length, got "
                f"{len(weights)} and {len(bodies)}")
        if not weights:
            raise ValueError("no terms: weights and bodies are empty")

        required_work, pauli_words = _validate_bodies(bodies, num_system)
        if num_select_work is None:
            num_select_work = required_work
        else:
            num_select_work = _as_count("num_select_work", num_select_work, 1)
            if num_select_work < required_work:
                raise ValueError(
                    f"num_select_work={num_select_work} is too small; the "
                    f"bodies need at least {required_work} work qubits")

        # Pin the "select" QROM variant: it keeps ladder_offset / num_ladder
        # == num_index, the geometry the apply kernel relies on when SELECT
        # reuses the PREPARE QROM ladder.
        preparation = AliasSamplingPrepare(weights, mu, variant="select")
        select = unary_iteration_kernels(preparation.num_index,
                                         len(bodies),
                                         lambda k: bodies[k],
                                         include_adjoint=False,
                                         num_work=num_select_work)
        controlled_select = unary_iteration_kernels(preparation.num_index,
                                                    len(bodies),
                                                    lambda k: bodies[k],
                                                    controlled=True,
                                                    include_adjoint=False,
                                                    num_work=num_select_work)

        self._num_system = num_system
        self._weights = tuple(weights)
        self._mu = mu
        self._preparation = preparation
        self._num_work = num_select_work
        self._pauli_words = pauli_words
        self._select = select
        self._controlled_select = controlled_select
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
        def lcu_prepare(ancilla: cudaq.qview):
            """The alias PREPARE on [index | garbage]: the protocol's
            prepare hook (the walk runs in the PREPARE frame)."""
            prepare(ancilla.front(m), ancilla[m:m + g])

        @cudaq.kernel
        def lcu_unprepare(ancilla: cudaq.qview):
            """PREPARE-dagger on [index | garbage]."""
            unprepare(ancilla.front(m), ancilla[m:m + g])

        @cudaq.kernel
        def lcu_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W = PREPARE R PREPARE-dagger SELECT: SELECT, then reflect
            about the PREPARE state. R covers the full ancilla register,
            garbage included (see the module docstring). SELECT's ladder
            and the work scratch are clean here, as after every step."""
            select(ancilla.front(m), ancilla[m + l0:m + l0 + m], system,
                   ancilla[m + g:m + g + w])
            unprepare(ancilla.front(m), ancilla[m:m + g])
            reflect_about_zero(ancilla)
            prepare(ancilla.front(m), ancilla[m:m + g])

        @cudaq.kernel
        def lcu_adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W-dagger = SELECT PREPARE R PREPARE-dagger (SELECT is
            self-adjoint)."""
            unprepare(ancilla.front(m), ancilla[m:m + g])
            reflect_about_zero(ancilla)
            prepare(ancilla.front(m), ancilla[m:m + g])
            select(ancilla.front(m), ancilla[m + l0:m + l0 + m], system,
                   ancilla[m + g:m + g + w])

        @cudaq.kernel
        def lcu_controlled_walk_step(control_and_ancilla: cudaq.qview,
                                     system: cudaq.qview):
            """Controlled W: controlled SELECT, PREPARE-dagger, controlled
            R, PREPARE. Only the two Hermitian factors carry the control;
            the uncontrolled PREPARE pair cancels at control |0>, and
            PREPARE never sits under a control variant."""
            controlled_select(control_and_ancilla.front(1),
                              control_and_ancilla[1:1 + m],
                              control_and_ancilla[1 + m + l0:1 + m + l0 + m],
                              system,
                              control_and_ancilla[1 + m + g:1 + m + g + w])
            unprepare(control_and_ancilla[1:1 + m],
                      control_and_ancilla[1 + m:1 + m + g])
            controlled_reflect_about_zero(control_and_ancilla)
            prepare(control_and_ancilla[1:1 + m],
                    control_and_ancilla[1 + m:1 + m + g])

        @cudaq.kernel
        def lcu_controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                             system: cudaq.qview):
            """Controlled W-dagger."""
            unprepare(control_and_ancilla[1:1 + m],
                      control_and_ancilla[1 + m:1 + m + g])
            controlled_reflect_about_zero(control_and_ancilla)
            prepare(control_and_ancilla[1:1 + m],
                    control_and_ancilla[1 + m:1 + m + g])
            controlled_select(control_and_ancilla.front(1),
                              control_and_ancilla[1:1 + m],
                              control_and_ancilla[1 + m + l0:1 + m + l0 + m],
                              system,
                              control_and_ancilla[1 + m + g:1 + m + g + w])

        _retain(lcu_apply, lcu_controlled_apply, lcu_prepare, lcu_unprepare,
                lcu_walk_step, lcu_adjoint_walk_step, lcu_controlled_walk_step,
                lcu_controlled_adjoint_walk_step)
        self._apply = lcu_apply
        self._controlled_apply = lcu_controlled_apply
        self._prepare = lcu_prepare
        self._unprepare = lcu_unprepare
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
        """The term weights ``|c_k|``, one per term in term order (the first
        ``num_terms`` PREPARE bins). ``discretized_weights`` has one entry
        per bin; the bins past ``num_terms`` are zero-weight padding that
        acts as the identity."""
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

        PREPARE, ``power`` walk steps, PREPARE-dagger — the same
        ``(R U_A)^power`` that ``Walk.kernel`` composes — so the
        all-zero-ancilla block of the result is T_power(-H/alpha) applied
        to the input state. Input modes as in ``encode_kernel``.
        """
        step = self._walk_step
        prep = self._prepare
        unprep = self._unprepare
        n_anc = self.num_ancilla
        n_sys = self.num_system
        steps = _validate_power(power)

        if state_prep is not None:

            @cudaq.kernel
            def lcu_prep_walked():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                prep(ancilla)
                for _ in range(steps):
                    step(ancilla, system)
                unprep(ancilla)

            _retain(lcu_prep_walked)
            return lcu_prep_walked

        @cudaq.kernel
        def lcu_walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prep(ancilla)
            for _ in range(steps):
                step(ancilla, system)
            unprep(ancilla)

        _retain(lcu_walked)
        return lcu_walked

    # ------------------------------------------------------------------
    # BlockEncoding protocol: data-free kernel factories
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: the alias PREPARE on ``[index | garbage]``
        — the walk runs in the PREPARE frame, as ``PauliLCU``'s does."""
        return self._prepare

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE-dagger."""
        return self._unprepare

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
        """The SELECT operator ``sum_k sign_k |k><k|_index (x) P_k`` as a
        ``cudaq.SpinOperator``: the odd-Chebyshev-moment observable.

        Measured after ``prepare_kernel`` and ``p`` walk steps with no
        UNPREPARE (what ``Walk.moment`` does for odd orders), its
        expectation is ``<T_{2p+1}(H/alpha)>``. It acts on the index
        register (qubits ``num_system .. num_system + num_index - 1``,
        little-endian) and the system, and is the identity on the garbage
        and work. Available when every body is a Pauli word (``"x"`` /
        ``"y"`` / ``"z"`` and ``("sign",)`` ops only); raises
        ``NotImplementedError`` for bodies using the extended vocabulary,
        whose term unitaries have no spin-operator form.
        """
        if any(word is None for word in self._pauli_words):
            raise NotImplementedError(
                "LCUBlockEncoding.select_observable needs every SELECT body "
                "to be a Pauli word (x/y/z and ('sign',) ops only); a body "
                "using the extended vocabulary has no SpinOperator form, so "
                "only even Walk.moment orders are available")
        offset = self.num_system
        observable = None
        for k, (sign, word) in enumerate(self._pauli_words):
            term = None
            for b in range(self.num_index):
                factor = _bit_projector(offset + b, (k >> b) & 1)
                term = factor if term is None else term * factor
            for qubit in sorted(word):
                term = term * getattr(cudaq.spin, word[qubit])(qubit)
            term = float(sign) * term
            observable = term if observable is None else observable + term
        return observable
