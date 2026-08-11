# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from typing import TYPE_CHECKING, TypeAlias, Union

import cudaq

from .common_kernels import (_bit_projector, _real_coefficient,
                             _validate_power, controlled_reflect_about_zero,
                             controlled_signal_phase, reflect_about_zero,
                             signal_phase, state_from)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .block_encoding import Kernel

# A Hamiltonian in any accepted input form: a ``cudaq.SpinOperator``, a
# ``{"XZI...": coefficient}`` mapping, or an iterable of
# ``(coefficient, word)`` pairs.
HamiltonianLike: TypeAlias = Union[cudaq.SpinOperator, cudaq.SpinOperatorTerm,
                                   Mapping[str, complex],
                                   Iterable[tuple[complex, str]]]

# The flattened arrays that cross the kernel boundary, in the order the
# module-level kernels take them:
# (angles, term_controls, term_ops, term_lengths, term_signs).
LCUKernelArgs: TypeAlias = tuple[list[float], list[int], list[int], list[int],
                                 list[int]]

_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}

# ============================================================================
# Device kernels (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def prepare(ancilla: cudaq.qview, angles: list[float]):
    """PREPARE: encode sqrt(|c_i| / alpha) amplitudes on the ancilla register.

    Positive guard, not early return: kernel ``return`` is silently
    ignored (cuda-quantum#4845).
    """
    n = ancilla.size()
    if n > 0:
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
    """PREPARE dagger.

    Hand-written inverse of ``prepare``: ``cudaq.adjoint(prepare, ...)``
    fails at runtime on this kernel's conditionally-conjugated rotations
    (cuda-quantum#4898), and adjoint autogeneration also silently
    mis-replays loop-carried classical updates (cuda-quantum#4897), so
    the explicit reverse is kept and the inverse property is pinned by
    ``test_unprepare_inverts_prepare``. Guards are positive blocks, not
    early returns (cuda-quantum#4845).
    """
    n = ancilla.size()
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
    if n > 0:
        ry(-angles[0], ancilla[0])


@cudaq.kernel
def select(ancilla: cudaq.qview, system: cudaq.qview, term_controls: list[int],
           term_ops: list[int], term_lengths: list[int],
           term_signs: list[int]):
    """SELECT: apply Pauli term i, controlled on ancilla state |i>.

    Requires a non-empty ancilla register (PauliLCU always provides at
    least one ancilla).
    """
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
            if code == 1:
                x.ctrl(ancilla, system[target])
            elif code == 2:
                y.ctrl(ancilla, system[target])
            else:
                z.ctrl(ancilla, system[target])
        if term_signs[i] < 0:
            if n_anc == 1:
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
def walk(ancilla: cudaq.qview, system: cudaq.qview, angles: list[float],
         term_controls: list[int], term_ops: list[int],
         term_lengths: list[int], term_signs: list[int]):
    """One qubitization walk step: SELECT, then reflect about PREPARE."""
    select(ancilla, system, term_controls, term_ops, term_lengths, term_signs)
    unprepare(ancilla, angles)
    reflect_about_zero(ancilla)
    prepare(ancilla, angles)


# ============================================================================
# LCU composite kernels: walk steps and QSVT sequences over this encoding
# (composable from user kernels; PauliLCU.kernel_args supplies the data)
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


@cudaq.kernel
def apply_phase_sequence(signal: cudaq.qview, system: cudaq.qview,
                         phases: list[float], walk_directions: list[int],
                         angles: list[float], term_controls: list[int],
                         term_ops: list[int], term_lengths: list[int],
                         term_signs: list[int]):
    """Projector-phase QSVT sequence: phase, then (walk step, phase) repeats.

    The signal register must start in |0...0>. A forward step is the full
    block encoding followed by the zero-state reflection; an adjoint step is
    the reverse (both factors are self-adjoint).

    ``walk_directions`` must be non-empty (empty lists cannot cross the
    kernel boundary, cuda-quantum#4847): for a degree-0 sequence pass one
    unused entry, as the QSVT factories do.
    """
    signal_phase(signal, phases[0])
    for i in range(1, len(phases)):
        if walk_directions[i - 1] == 1:
            reflect_about_zero(signal)
            apply(signal, system, angles, term_controls, term_ops,
                  term_lengths, term_signs)
        else:
            apply(signal, system, angles, term_controls, term_ops,
                  term_lengths, term_signs)
            reflect_about_zero(signal)
        signal_phase(signal, phases[i])


@cudaq.kernel
def apply_controlled_phase_sequence(
        control_and_signal: cudaq.qview, system: cudaq.qview,
        phases: list[float], walk_directions: list[int], angles: list[float],
        term_controls: list[int], term_ops: list[int], term_lengths: list[int],
        term_signs: list[int]):
    """QSVT sequence controlled by qubit 0 of ``control_and_signal``.

    The uncontrolled PREPARE / PREPARE-dagger pair wraps a controlled
    SELECT, so each walk step collapses to the identity for control |0>;
    the zero reflection and signal phases are likewise controlled, making
    the full sequence the identity when the control is off.
    """
    n_signal = control_and_signal.size() - 1
    controlled_signal_phase(control_and_signal, phases[0])
    for i in range(1, len(phases)):
        if walk_directions[i - 1] == 1:
            controlled_reflect_about_zero(control_and_signal)
            prepare(control_and_signal.back(n_signal), angles)
            controlled_select(control_and_signal, system, term_controls,
                              term_ops, term_lengths, term_signs)
            unprepare(control_and_signal.back(n_signal), angles)
        else:
            prepare(control_and_signal.back(n_signal), angles)
            controlled_select(control_and_signal, system, term_controls,
                              term_ops, term_lengths, term_signs)
            unprepare(control_and_signal.back(n_signal), angles)
            controlled_reflect_about_zero(control_and_signal)
        controlled_signal_phase(control_and_signal, phases[i])


# Re-exported from common_kernels (encoding-independent; kept importable
# here for compatibility with existing call sites).

# ============================================================================
# Host-side decomposition
# ============================================================================


def _terms_from_input(
        hamiltonian: HamiltonianLike,
        num_qubits: int | None) -> tuple[list[tuple[float, str]], int]:
    """Normalize the supported Hamiltonian input forms to (coeff, word) pairs."""
    if isinstance(hamiltonian, Mapping):
        pairs = [(_real_coefficient(c), str(w))
                 for w, c in hamiltonian.items()]
    elif isinstance(hamiltonian, cudaq.SpinOperatorTerm):
        # A single product term (e.g. 0.5 * spin.x(0) * spin.y(1)):
        # normalize to a one-term operator.
        return _terms_from_input(cudaq.SpinOperator(hamiltonian), num_qubits)
    elif isinstance(hamiltonian, cudaq.SpinOperator):
        terms = list(hamiltonian)
        extent = max((int(term.max_degree) + 1 for term in terms), default=0)
        width = extent if num_qubits is None else num_qubits
        if width < extent:
            raise ValueError(
                f"num_qubits={num_qubits} is smaller than Hamiltonian "
                f"extent {extent}")
        pairs = [(_real_coefficient(term.evaluate_coefficient()),
                  str(term.get_pauli_word(width))) for term in terms]
    elif isinstance(hamiltonian, Iterable) and not isinstance(
            hamiltonian, (str, bytes, bytearray, memoryview)):
        pairs = [(_real_coefficient(c), str(w)) for c, w in hamiltonian]
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
            if total == 0.0:
                # Exactly-zero subtree: only power-of-two padding, since
                # every retained term contributes strictly positive
                # probability. (A threshold here would zero the split of
                # genuinely tiny sibling terms and silently encode a
                # different operator.)
                angles.append(0.0)
            else:
                right = sum(probabilities[start + step:start + 2 * step])
                angles.append(2.0 * math.asin(math.sqrt(right / total)))
    return angles


def select_observable(encoding: PauliLCU) -> cudaq.SpinOperator:
    """The SELECT operator sum_i sign_i |i><i|_anc x P_i as an observable.

    LCU-specific: built from the encoding's signed Pauli terms. Its
    expectation after PREPARE and p walk steps is the odd Chebyshev moment
    <T_{2p+1}(H/alpha)> (the BlockEncoding.select_observable hook).
    """
    from cudaq import spin

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

    ``num_ancilla`` is always at least 1: single-term (and identity-only)
    Hamiltonians get one idle ancilla, so every encoding works uniformly
    with ``Walk``/``QSVT`` and no flattened kernel argument is ever empty.
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
        # At least one ancilla even for a single term (amplitude [1, 0] on
        # one idle qubit): empty flattened lists cannot cross the kernel
        # boundary (cuda-quantum#4847), and a degenerate 0-ancilla encoding
        # would lose the walk's -H/alpha sign (the reflection that supplies
        # it is a no-op on an empty register).
        self._num_ancilla = max(1, (len(kept) - 1).bit_length())

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
                self._term_controls.append((index >>
                                            (self._num_ancilla - 1 - b)) & 1)
            ops = [(code, qubit) for qubit, ch in enumerate(word)
                   if (code := _PAULI_CODES.get(ch)) is not None]
            for code, qubit in ops:
                self._term_ops.extend((code, qubit))
            self._term_lengths.append(len(ops))
            self._term_signs.append(-1 if coeff < 0.0 else 1)
        if not self._term_ops:
            # Identity-only Hamiltonian: every term_length is 0, so this
            # padding is never dereferenced — it exists only because empty
            # lists cannot cross the kernel boundary (cuda-quantum#4847).
            self._term_ops = [0, 0]

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

    @property
    def constant_term(self) -> float:
        return self._constant

    @property
    def terms(self) -> list[tuple[float, str]]:
        return list(self._terms)

    @property
    def _kernel_data(self) -> LCUKernelArgs:
        """Internal, uncopied flattened arrays for the kernel factories."""
        return (self._angles, self._term_controls, self._term_ops,
                self._term_lengths, self._term_signs)

    @property
    def kernel_args(self) -> LCUKernelArgs:
        """(angles, term_controls, term_ops, term_lengths, term_signs).

        Escape hatch for composing the module-level kernels inside your own
        ``@cudaq.kernel``; the factory methods below capture these for you.
        Returns defensive copies.
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

    def encode_kernel(self, state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying the full block encoding.

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state`` and the ancilla register (in
        |0...0>) after it. With ``state_prep`` (a ``(qubits: qview)``
        kernel): a zero-argument kernel that allocates the system register
        in |0...0>, runs ``state_prep`` on it, then applies the encoding.
        """
        angles, controls, ops, lengths, signs = self._kernel_data
        n_anc = self.num_ancilla
        n_sys = self.num_system

        if state_prep is not None:

            @cudaq.kernel
            def prep_encoded():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                apply(ancilla, system, angles, controls, ops, lengths, signs)

            return prep_encoded

        @cudaq.kernel
        def encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            apply(ancilla, system, angles, controls, ops, lengths, signs)

        return encoded

    def walk_kernel(self,
                    power: int = 1,
                    state_prep: Kernel | None = None) -> Kernel:
        """A kernel running PREPARE, walk^power, UNPREPARE.

        The all-zero-ancilla block of the result is T_power(-H/alpha)
        applied to the input state (Chebyshev polynomial of the walk block
        -H/alpha). Input modes as in ``encode_kernel``: a
        ``cudaq.State``-taking kernel, or a zero-argument kernel when
        ``state_prep`` is given.
        """
        angles, controls, ops, lengths, signs = self._kernel_data
        n_anc = self.num_ancilla
        n_sys = self.num_system
        steps = _validate_power(power)

        if state_prep is not None:

            @cudaq.kernel
            def prep_walked():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                prepare(ancilla, angles)
                for _ in range(steps):
                    walk(ancilla, system, angles, controls, ops, lengths,
                         signs)
                unprepare(ancilla, angles)

            return prep_walked

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
        angles = self._angles

        @cudaq.kernel
        def prepare_ancilla(ancilla: cudaq.qview):
            prepare(ancilla, angles)

        return prepare_ancilla

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE dagger with this encoding's angles."""
        angles = self._angles

        @cudaq.kernel
        def unprepare_ancilla(ancilla: cudaq.qview):
            unprepare(ancilla, angles)

        return unprepare_ancilla

    def apply_kernel(self) -> Kernel:
        """``(ancilla, system)``: the full block encoding U_A."""
        angles, controls, ops, lengths, signs = self._kernel_data

        @cudaq.kernel
        def apply_encoding(ancilla: cudaq.qview, system: cudaq.qview):
            apply(ancilla, system, angles, controls, ops, lengths, signs)

        return apply_encoding

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0.

        Uncontrolled PREPARE pairs wrap the controlled SELECT, so the
        circuit is the identity at control |0>.
        """
        angles, controls, ops, lengths, signs = self._kernel_data
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
        angles, controls, ops, lengths, signs = self._kernel_data

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            walk(ancilla, system, angles, controls, ops, lengths, signs)

        return walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W†."""
        angles, controls, ops, lengths, signs = self._kernel_data

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            adjoint_walk(ancilla, system, angles, controls, ops, lengths,
                         signs)

        return adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        angles, controls, ops, lengths, signs = self._kernel_data

        @cudaq.kernel
        def controlled_walk_step(control_and_ancilla: cudaq.qview,
                                 system: cudaq.qview):
            controlled_walk(control_and_ancilla, system, angles, controls, ops,
                            lengths, signs)

        return controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled adjoint walk step."""
        angles, controls, ops, lengths, signs = self._kernel_data

        @cudaq.kernel
        def controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                         system: cudaq.qview):
            controlled_adjoint_walk(control_and_ancilla, system, angles,
                                    controls, ops, lengths, signs)

        return controlled_adjoint_walk_step

    def select_observable(self) -> cudaq.SpinOperator:
        """The odd-moment SELECT observable for this encoding.

        BlockEncoding protocol hook; delegates to the module-level
        ``select_observable``.
        """
        return select_observable(self)
