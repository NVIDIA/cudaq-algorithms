# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pauli LCU block encoding.

Implements the linear-combination-of-unitaries block encoding of a Pauli-sum
Hamiltonian: the LCU decomposition, PREPARE-angle computation, and the
PREPARE/SELECT/walk device kernels (including controlled variants).

``PauliLCU`` is the primary entry point. It accepts a ``cudaq.SpinOperator``,
a ``{"XZI...": coeff}`` mapping, or ``[(coeff, word), ...]`` pairs, and
provides ready-to-run kernel factories::

    enc = PauliLCU({"ZI": 0.7, "XX": 0.19, "IZ": -0.43})
    kernel = enc.encode_kernel()            # @cudaq.kernel(state)

The module-level kernels (``prepare``, ``select``, ``apply``, ...) are
composable from user kernels; ``PauliLCU.kernel_args`` supplies the flattened
arrays they take as arguments. Statevector-based conveniences (postselection,
``action``) live in ``sim_utils``, outside the library surface.

The encoding satisfies ``(<0|_anc ⊗ I) U (|0>_anc ⊗ I) = H / alpha`` with
``alpha = sum(|c_i|)``, and the qubitization walk block is ``-H / alpha``
(the zero reflection phases ``|0...0>`` by -1).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import cudaq

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# A Hamiltonian in any accepted input form: a ``cudaq.SpinOperator``, a
# ``{"XZI...": coefficient}`` mapping, or an iterable of
# ``(coefficient, word)`` pairs.
HamiltonianLike: TypeAlias = Union[cudaq.SpinOperator, Mapping[str, float],
                                   Iterable[tuple[float, str]]]

# A compiled ``@cudaq.kernel``. CUDA-Q does not expose a stable public
# Python type for kernel objects, so kernel factories are annotated with
# this alias.
Kernel: TypeAlias = Any

# The flattened arrays that cross the kernel boundary, in the order the
# module-level kernels take them:
# (angles, term_controls, term_ops, term_lengths, term_signs).
LCUKernelArgs: TypeAlias = tuple[list[float], list[int], list[int],
                                 list[int], list[int]]

_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}


# ============================================================================
# Device kernels (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def prepare(ancilla: cudaq.qview, angles: list[float]):
    """PREPARE: encode sqrt(|c_i| / alpha) amplitudes on the ancilla register."""
    n = ancilla.size()
    if n == 0:
        return
    ry(angles[0], ancilla[0])
    idx = 1
    for layer in range(1, n):
        branches = 1 << layer
        for branch in range(branches):
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(ancilla[layer - 1 - bit])
            ry.ctrl(angles[idx], ancilla.front(layer), ancilla[layer])
            idx += 1
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(ancilla[layer - 1 - bit])


@cudaq.kernel
def unprepare(ancilla: cudaq.qview, angles: list[float]):
    """PREPARE dagger."""
    n = ancilla.size()
    if n == 0:
        return
    idx = len(angles) - 1
    for reverse_layer in range(n - 1):
        layer = n - 1 - reverse_layer
        branches = 1 << layer
        for reverse_branch in range(branches):
            branch = branches - 1 - reverse_branch
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(ancilla[layer - 1 - bit])
            ry.ctrl(-angles[idx], ancilla.front(layer), ancilla[layer])
            idx -= 1
            for bit in range(layer):
                if ((branch >> bit) & 1) == 0:
                    x(ancilla[layer - 1 - bit])
    ry(-angles[0], ancilla[0])


@cudaq.kernel
def select(ancilla: cudaq.qview, system: cudaq.qview, term_controls: list[int],
           term_ops: list[int], term_lengths: list[int],
           term_signs: list[int]):
    """SELECT: apply Pauli term i, controlled on ancilla state |i>."""
    n_anc = ancilla.size()
    ptr_ctrl = 0
    ptr_op = 0
    for i in range(len(term_lengths)):
        for b in range(n_anc):
            if term_controls[ptr_ctrl] == 0:
                x(ancilla[b])
            ptr_ctrl += 1
        for _ in range(term_lengths[i]):
            code = term_ops[ptr_op]
            target = term_ops[ptr_op + 1]
            ptr_op += 2
            if n_anc == 0:
                if code == 1:
                    x(system[target])
                elif code == 2:
                    y(system[target])
                else:
                    z(system[target])
            else:
                if code == 1:
                    x.ctrl(ancilla, system[target])
                elif code == 2:
                    y.ctrl(ancilla, system[target])
                else:
                    z.ctrl(ancilla, system[target])
        if term_signs[i] < 0:
            if n_anc == 0:
                # Single-term LCU: no projected block exists, so the sign is
                # part of the encoded operator. rz(2*pi) is exactly -I.
                rz(6.283185307179586, system[0])
            elif n_anc == 1:
                z(ancilla[0])
            else:
                z.ctrl(ancilla.front(n_anc - 1), ancilla[n_anc - 1])
        back = ptr_ctrl - 1
        for b in range(n_anc):
            if term_controls[back] == 0:
                x(ancilla[n_anc - 1 - b])
            back -= 1


@cudaq.kernel
def controlled_select(control_and_ancilla: cudaq.qview, system: cudaq.qview,
                      term_controls: list[int], term_ops: list[int],
                      term_lengths: list[int], term_signs: list[int]):
    """SELECT controlled by an external qubit.

    CUDA-Q Python kernels cannot mix a bare qubit with a qview in one
    control set, so the external control is qubit 0 of
    ``control_and_ancilla`` and the LCU ancillas are the remaining qubits;
    every control set is then a view of that one register.
    """
    n_anc = control_and_ancilla.size() - 1
    ptr_ctrl = 0
    ptr_op = 0
    for i in range(len(term_lengths)):
        for b in range(n_anc):
            if term_controls[ptr_ctrl] == 0:
                x(control_and_ancilla[1 + b])
            ptr_ctrl += 1
        for _ in range(term_lengths[i]):
            code = term_ops[ptr_op]
            target = term_ops[ptr_op + 1]
            ptr_op += 2
            if code == 1:
                x.ctrl(control_and_ancilla, system[target])
            elif code == 2:
                y.ctrl(control_and_ancilla, system[target])
            else:
                z.ctrl(control_and_ancilla, system[target])
        if term_signs[i] < 0:
            if n_anc == 0:
                # Controlled -I on the system is a Z on the control.
                z(control_and_ancilla[0])
            else:
                total = control_and_ancilla.size()
                z.ctrl(control_and_ancilla.front(total - 1),
                       control_and_ancilla[total - 1])
        back = ptr_ctrl - 1
        for b in range(n_anc):
            if term_controls[back] == 0:
                x(control_and_ancilla[n_anc - b])
            back -= 1


@cudaq.kernel
def apply(ancilla: cudaq.qview, system: cudaq.qview, angles: list[float],
          term_controls: list[int], term_ops: list[int],
          term_lengths: list[int], term_signs: list[int]):
    """Full block encoding: PREPARE, SELECT, PREPARE dagger."""
    prepare(ancilla, angles)
    select(ancilla, system, term_controls, term_ops, term_lengths, term_signs)
    unprepare(ancilla, angles)


@cudaq.kernel
def reflect_about_zero(register: cudaq.qview):
    """I - 2|0...0><0...0| (phases the all-zero state by -1)."""
    n = register.size()
    if n == 0:
        return
    for i in range(n):
        x(register[i])
    if n == 1:
        z(register[0])
    else:
        z.ctrl(register.front(n - 1), register[n - 1])
    for i in range(n):
        x(register[i])


@cudaq.kernel
def walk(ancilla: cudaq.qview, system: cudaq.qview, angles: list[float],
         term_controls: list[int], term_ops: list[int],
         term_lengths: list[int], term_signs: list[int]):
    """One qubitization walk step: SELECT, then reflect about PREPARE."""
    select(ancilla, system, term_controls, term_ops, term_lengths, term_signs)
    unprepare(ancilla, angles)
    reflect_about_zero(ancilla)
    prepare(ancilla, angles)


def state_from(ket: ArrayLike) -> cudaq.State:
    """Build a cudaq.State from array data at the current target's precision.

    fp32 simulators (e.g. the default `nvidia` target) reject complex128
    input ("[sim-state] invalid data precision"); cudaq.complex() reports
    the dtype the active target expects.
    """
    import numpy as np

    return cudaq.State.from_data(np.asarray(ket, dtype=cudaq.complex()))


# ============================================================================
# Host-side decomposition
# ============================================================================


def _terms_from_input(
        hamiltonian: HamiltonianLike,
        num_qubits: int | None) -> tuple[list[tuple[float, str]], int]:
    """Normalize the supported Hamiltonian input forms to (coeff, word) pairs."""
    if isinstance(hamiltonian, Mapping):
        pairs = [(float(c), str(w)) for w, c in hamiltonian.items()]
    elif isinstance(hamiltonian, cudaq.SpinOperator):
        width = num_qubits if num_qubits is not None else int(
            hamiltonian.qubit_count)
        pairs = []
        for term in hamiltonian:
            coeff = term.evaluate_coefficient()
            if abs(coeff.imag) > 1e-10:
                raise ValueError(
                    "complex Hamiltonian coefficients are not supported")
            pairs.append((float(coeff.real), str(term.get_pauli_word(width))))
    elif isinstance(hamiltonian, Iterable):
        pairs = [(float(c), str(w)) for c, w in hamiltonian]
    else:
        raise TypeError(
            "hamiltonian must be a cudaq.SpinOperator, a {word: coeff} "
            "mapping, or an iterable of (coeff, word) pairs")

    if not pairs:
        raise ValueError("hamiltonian has no terms")

    width = len(pairs[0][1])
    for _, word in pairs:
        if len(word) != width:
            raise ValueError("all Pauli words must have the same length")
        if any(ch not in "IXYZ" for ch in word):
            raise ValueError(f"unsupported Pauli word: {word!r}")
    if num_qubits is not None and num_qubits != width:
        raise ValueError(
            f"num_qubits={num_qubits} does not match word length {width}")
    return pairs, width


def _prepare_angles(probabilities: Sequence[float]) -> list[float]:
    """Rotation angles for the binary state-preparation tree."""
    n_leaves = len(probabilities)
    if n_leaves & (n_leaves - 1):
        raise ValueError("probability vector size must be a power of 2")
    n_qubits = n_leaves.bit_length() - 1

    angles = []
    for layer in range(n_qubits):
        step = n_leaves >> (layer + 1)
        for node in range(1 << layer):
            start = node * step * 2
            total = sum(probabilities[start:start + 2 * step])
            if total < 1e-12:
                # Zero-probability subtree from power-of-two padding.
                angles.append(0.0)
            else:
                right = sum(probabilities[start + step:start + 2 * step])
                angles.append(2.0 * math.asin(math.sqrt(right / total)))
    return angles


# ============================================================================
# The user-facing object
# ============================================================================


class PauliLCU:
    """Block encoding of H / alpha via a linear combination of Pauli strings.

    Parameters
    ----------
    hamiltonian
        A ``cudaq.SpinOperator``, a mapping ``{"XZI...": coefficient}``, or an
        iterable of ``(coefficient, word)`` pairs. Words use one I/X/Y/Z
        character per qubit, position = qubit index.
    num_qubits
        Optional; inferred from the input when omitted, validated against it
        when given.
    include_identity
        Whether identity words are retained inside the encoded operator
        (their sum is always reported as ``constant_term``).
    coefficient_threshold
        Terms with ``|coefficient|`` below this are dropped.
    """

    def __init__(self,
                 hamiltonian: HamiltonianLike,
                 num_qubits: int | None = None,
                 *,
                 include_identity: bool = True,
                 coefficient_threshold: float = 1e-12) -> None:
        pairs, width = _terms_from_input(hamiltonian, num_qubits)

        self._num_system = width
        self._constant = 0.0
        kept = []
        for coeff, word in pairs:
            if abs(coeff) < coefficient_threshold:
                continue
            identity = set(word) == {"I"}
            if identity:
                self._constant += coeff
                if not include_identity:
                    continue
            kept.append((coeff, word))
        if not kept:
            raise ValueError("hamiltonian has no retained terms")

        self._terms = kept
        self._alpha = sum(abs(c) for c, _ in kept)
        self._num_ancilla = max(0, (len(kept) - 1).bit_length())

        # Precompute the flattened arrays that cross the kernel boundary.
        probabilities = [abs(c) / self._alpha for c, _ in kept]
        probabilities += [0.0] * ((1 << self._num_ancilla) - len(kept))
        self._angles = _prepare_angles(probabilities)

        self._term_controls = []
        self._term_ops = []
        self._term_lengths = []
        self._term_signs = []
        for index, (coeff, word) in enumerate(kept):
            for b in range(self._num_ancilla):
                self._term_controls.append(
                    (index >> (self._num_ancilla - 1 - b)) & 1)
            ops = [(code, qubit) for qubit, ch in enumerate(word)
                   if (code := _PAULI_CODES.get(ch)) is not None]
            for code, qubit in ops:
                self._term_ops.extend((code, qubit))
            self._term_lengths.append(len(ops))
            self._term_signs.append(-1 if coeff < 0.0 else 1)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def num_system(self) -> int:
        return self._num_system

    @property
    def num_ancilla(self) -> int:
        return self._num_ancilla

    @property
    def num_terms(self) -> int:
        return len(self._terms)

    @property
    def alpha(self) -> float:
        """LCU normalization (1-norm of retained coefficients)."""
        return self._alpha

    normalization = alpha

    @property
    def constant_term(self) -> float:
        return self._constant

    @property
    def terms(self) -> list[tuple[float, str]]:
        return list(self._terms)

    @property
    def kernel_args(self) -> LCUKernelArgs:
        """(angles, term_controls, term_ops, term_lengths, term_signs).

        Escape hatch for composing the module-level kernels inside your own
        ``@cudaq.kernel``; the factory methods below capture these for you.
        """
        return (list(self._angles), list(self._term_controls),
                list(self._term_ops), list(self._term_lengths),
                list(self._term_signs))

    def __repr__(self) -> str:
        return (f"PauliLCU(terms={self.num_terms}, "
                f"system_qubits={self.num_system}, "
                f"ancilla_qubits={self.num_ancilla}, "
                f"alpha={self.alpha:.6g})")

    # ------------------------------------------------------------------
    # Kernel factories
    # ------------------------------------------------------------------

    def _zero_ancilla_kernel(self, repetitions: int) -> Kernel:
        """Single-term (0-ancilla) special case.

        Needed because CUDA-Q cannot marshal empty ``list`` kernel arguments
        ("Cannot infer runtime argument type"), and with zero ancillas the
        flattened control data is empty. The encoding degenerates to the
        signed Pauli word, so apply it directly. One walk step also reduces
        to one signed application (the reflections are empty-register no-ops).
        """
        ops = list(self._term_ops)
        negative = self._term_signs[0] < 0
        steps = int(repetitions)

        if ops:

            @cudaq.kernel
            def single_term(state: cudaq.State):
                system = cudaq.qvector(state)
                for _ in range(steps):
                    for i in range(len(ops) // 2):
                        code = ops[2 * i]
                        target = ops[2 * i + 1]
                        if code == 1:
                            x(system[target])
                        elif code == 2:
                            y(system[target])
                        else:
                            z(system[target])
                    if negative:
                        rz(6.283185307179586, system[0])

            return single_term

        @cudaq.kernel
        def single_identity_term(state: cudaq.State):
            system = cudaq.qvector(state)
            for _ in range(steps):
                if negative:
                    rz(6.283185307179586, system[0])

        return single_identity_term

    def encode_kernel(self) -> Kernel:
        """A ``@cudaq.kernel(state)`` applying the full block encoding.

        The kernel allocates the system register from ``state`` and the
        ancilla register (in |0...0>) after it.
        """
        if self.num_ancilla == 0:
            return self._zero_ancilla_kernel(1)

        angles, controls, ops, lengths, signs = self.kernel_args
        n_anc = self.num_ancilla

        @cudaq.kernel
        def encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            apply(ancilla, system, angles, controls, ops, lengths, signs)

        return encoded

    def walk_kernel(self, power: int = 1) -> Kernel:
        """A ``@cudaq.kernel(state)`` running PREPARE, walk^power, UNPREPARE.

        The all-zero-ancilla block of the result is T_power(-H/alpha) applied
        to the input state (Chebyshev polynomial of the walk block -H/alpha).
        """
        if self.num_ancilla == 0:
            return self._zero_ancilla_kernel(power)

        angles, controls, ops, lengths, signs = self.kernel_args
        n_anc = self.num_ancilla
        steps = int(power)

        @cudaq.kernel
        def walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prepare(ancilla, angles)
            for _ in range(steps):
                walk(ancilla, system, angles, controls, ops, lengths, signs)
            unprepare(ancilla, angles)

        return walked


    # ------------------------------------------------------------------
    # BlockEncoding protocol: data-free kernel factories
    #
    # Each factory returns a kernel with the protocol's fixed signature,
    # with this encoding's flattened arrays captured inside. Walk and QSVT
    # compose these without touching the LCU internals; kernel_args stays
    # available as the LCU-specific escape hatch for user kernels.
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE with this encoding's angles."""
        angles = list(self._angles)

        @cudaq.kernel
        def prepare_ancilla(ancilla: cudaq.qview):
            prepare(ancilla, angles)

        return prepare_ancilla

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE dagger with this encoding's angles."""
        angles = list(self._angles)

        @cudaq.kernel
        def unprepare_ancilla(ancilla: cudaq.qview):
            unprepare(ancilla, angles)

        return unprepare_ancilla

    def apply_kernel(self) -> Kernel:
        """``(ancilla, system)``: the full block encoding U_A."""
        angles, controls, ops, lengths, signs = self.kernel_args

        @cudaq.kernel
        def apply_encoding(ancilla: cudaq.qview, system: cudaq.qview):
            apply(ancilla, system, angles, controls, ops, lengths, signs)

        return apply_encoding

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0.

        Uncontrolled PREPARE pairs wrap the controlled SELECT, so the
        circuit is the identity at control |0>.
        """
        angles, controls, ops, lengths, signs = self.kernel_args
        n_anc = self.num_ancilla

        @cudaq.kernel
        def apply_controlled(control_and_ancilla: cudaq.qview,
                             system: cudaq.qview):
            prepare(control_and_ancilla.back(n_anc), angles)
            controlled_select(control_and_ancilla, system, controls, ops,
                              lengths, signs)
            unprepare(control_and_ancilla.back(n_anc), angles)

        return apply_controlled

    def walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one qubitization walk step W."""
        angles, controls, ops, lengths, signs = self.kernel_args

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            walk(ancilla, system, angles, controls, ops, lengths, signs)

        return walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W†."""
        from .qubitization import adjoint_walk

        angles, controls, ops, lengths, signs = self.kernel_args

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            adjoint_walk(ancilla, system, angles, controls, ops, lengths,
                         signs)

        return adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        from .qubitization import controlled_walk

        angles, controls, ops, lengths, signs = self.kernel_args

        @cudaq.kernel
        def controlled_walk_step(control_and_ancilla: cudaq.qview,
                                 system: cudaq.qview):
            controlled_walk(control_and_ancilla, system, angles, controls,
                            ops, lengths, signs)

        return controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled adjoint walk step."""
        from .qubitization import controlled_adjoint_walk

        angles, controls, ops, lengths, signs = self.kernel_args

        @cudaq.kernel
        def controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                         system: cudaq.qview):
            controlled_adjoint_walk(control_and_ancilla, system, angles,
                                    controls, ops, lengths, signs)

        return controlled_adjoint_walk_step

    def select_observable(self) -> cudaq.SpinOperator:
        """The odd-moment SELECT observable for this encoding.

        BlockEncoding protocol hook; delegates to the module-level
        ``qubitization.select_observable``.
        """
        from .qubitization import select_observable

        return select_observable(self)
