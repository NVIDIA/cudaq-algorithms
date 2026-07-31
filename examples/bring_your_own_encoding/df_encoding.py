# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Bring your own block encoding: a complete, worked ``BlockEncoding``.

``cudaq_algorithms`` consumers (``Walk``, ``QSVT``) are generic over the
structural ``BlockEncoding`` protocol — no inheritance, no registration.
Any object that exposes ``num_system`` / ``num_ancilla`` / ``alpha`` and
the documented kernel factories (``prepare_kernel``, ``apply_kernel``,
``walk_step_kernel``, the controlled/adjoint variants, ...) plugs into
qubitization and QSVT unchanged. This example is a full-scale
demonstration: a *double-factorized* encoding of the electronic-structure
Hamiltonian, implemented outside the package against the public API only,
and validated against an independent dense reference by
``tests/python/test_df_encoding.py``. Use it as the template for writing
your own encoding; ``df_block_encoding.py`` in this directory drives it
head to head with the built-in ``PauliLCU`` through the same ``Walk``.

Encodes the electronic-structure Hamiltonian directly from its
double-factorized form (von Burg et al., PRX Quantum 2, 030305 (2021);
arXiv:2007.14460) instead of a flat Pauli expansion:

    H = const + sum_k F_k N_k                       (kappa eigenbasis)
        + 1/2 sum_t sum_kl Z^t_kl (N^t_k - 1)(N^t_l - 1)   (leaf frames)

where ``N^t_k`` is the (spin-summed) number operator of leaf ``t``'s
rotated orbital ``k``, and ``F`` are the eigenvalues of the corrected
one-body matrix ``kappa`` — the raw one-body integrals plus the exchange
correction ``-1/2 sum_r (pr|rq)`` and the one-body remainder from
centering the leaf number operators (both evaluated on the *factorized*
tensor, so truncated factorizations encode exactly their truncated
Hamiltonian). The centered form makes each leaf pure ZZ and reproduces
the LCU one-norm of ``double_factorization_one_norm`` as ``alpha``.
Every term is *diagonal in some rotated orbital basis*: the one-body part in
the eigenbasis of ``kappa``, each leaf in its own eigenbasis ``U^t``. The
SELECT oracle therefore applies only ancilla-controlled Z words, conjugated
by *uncontrolled* Givens-rotation networks that change the orbital frame —
the structural cost advantage of the double-factorized encoding over a
Pauli-word SELECT.

Spin orbitals are interleaved (``2p`` up, ``2p + 1`` down), matching the
Jordan-Wigner convention of ``cudaq_algorithms.chemistry`` and the compiled
``fermion.jordan_wigner``. A spatial Givens rotation between adjacent
orbitals ``p, p + 1`` lifts to two three-qubit ``exp_pauli`` pairs
(``XZY``/``YZX`` on contiguous qubit slices), one per spin.

``DoubleFactorizedEncoding`` satisfies the ``BlockEncoding`` protocol, so
``Walk`` and ``QSVT`` consume it unchanged. One protocol hook is
unavailable: ``select_observable`` (the odd-Chebyshev-moment observable) is
LCU-specific — the rotated-frame terms are not Pauli words on the
computational frame — and raises ``NotImplementedError``; even moments and
the full QSVT/walk circuit surface are unaffected.

Kernel-language constraints (cuda-quantum#4845/#4847) shape this module the
same way they shape ``pauli_lcu``: positive guards instead of early
returns, and flattened-list padding so no empty list crosses the kernel
boundary.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

import cudaq

from cudaq_algorithms.block_encoding import Kernel
from cudaq_algorithms.common_kernels import reflect_about_zero
from cudaq_algorithms.double_factorization import (
    DoubleFactorization, explicit_double_factorization)
from cudaq_algorithms.pauli_lcu import (controlled_reflect_about_prepare,
                                        prepare, reflect_about_prepare,
                                        unprepare)

__all__ = [
    "DoubleFactorizedEncoding",
    "apply_basis_rotations",
    "select",
    "controlled_select",
    "apply",
    "walk",
    "adjoint_walk",
    "controlled_walk",
    "controlled_adjoint_walk",
]

# ----------------------------------------------------------------------
# Small host-side helpers, inlined so the example depends only on the
# public package surface (the library keeps private twins of these; the
# dense-reference tests pin both to the same behavior).
# ----------------------------------------------------------------------


def _validate_power(power: int) -> int:
    """Require a non-negative integral walk power (no silent truncation)."""
    steps = int(power)
    if steps != power or steps < 0:
        raise ValueError("power must be a non-negative integer")
    return steps


def _prepare_angles(probabilities) -> list[float]:
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


# ============================================================================
# Device kernels (module level, composable from user kernels)
# ============================================================================


@cudaq.kernel
def apply_basis_rotations(system: cudaq.qview, start: int, count: int,
                          rot_orbitals: list[int], rot_angles: list[float]):
    """Apply ``count`` spatial Givens rotations starting at ``start``.

    Rotation ``i`` mixes adjacent spatial orbitals ``p, p + 1``
    (``p = rot_orbitals[start + i]``) with angle ``rot_angles[start + i]``,
    lifted to both spin sectors of the interleaved Jordan-Wigner register:
    ``exp(theta (a^dag_a a_b - a^dag_b a_a))`` on modes ``(2p, 2p + 2)``
    and ``(2p + 1, 2p + 3)``, each realized as an ``XZY``/``YZX``
    ``exp_pauli`` pair on a contiguous three-qubit slice.
    """
    for i in range(count):
        p = rot_orbitals[start + i]
        theta = rot_angles[start + i]
        up = system[2 * p:2 * p + 3]
        exp_pauli(0.5 * theta, up, "XZY")
        exp_pauli(-0.5 * theta, up, "YZX")
        down = system[2 * p + 1:2 * p + 4]
        exp_pauli(0.5 * theta, down, "XZY")
        exp_pauli(-0.5 * theta, down, "YZX")


@cudaq.kernel
def select(ancilla: cudaq.qview, system: cudaq.qview, term_controls: list[int],
           term_targets: list[int], term_lengths: list[int],
           term_signs: list[int], frame_term_counts: list[int],
           segment_rot_counts: list[int], rot_orbitals: list[int],
           rot_angles: list[float]):
    """Double-factorized SELECT: framed, ancilla-controlled Z words.

    For each frame an uncontrolled Givens segment rotates the system into
    that frame's orbital eigenbasis, the frame's Z words are applied
    controlled on their ancilla index (the PauliLCU control convention:
    X-conjugation maps state ``|i>`` to all-ones, the sign is a phase on
    the all-ones state), and a final segment rotates back to the
    computational frame. Segments between frames are the concatenated
    exit + entry rotations, so with no controls active the whole circuit
    telescopes to the identity. SELECT as a whole is self-adjoint (each
    encoded term is a Hermitian involution on disjoint ancilla indices).

    Requires a non-empty ancilla register (the encoding always provides
    at least one ancilla).
    """
    n_anc = ancilla.size()
    rot_ptr = 0
    ptr_ctrl = 0
    ptr_tgt = 0
    term_i = 0
    for f in range(len(frame_term_counts)):
        apply_basis_rotations(system, rot_ptr, segment_rot_counts[f],
                              rot_orbitals, rot_angles)
        rot_ptr += segment_rot_counts[f]
        for _ in range(frame_term_counts[f]):
            for b in range(n_anc):
                if term_controls[ptr_ctrl] == 0:
                    x(ancilla[b])
                ptr_ctrl += 1
            for _ in range(term_lengths[term_i]):
                z.ctrl(ancilla, system[term_targets[ptr_tgt]])
                ptr_tgt += 1
            if term_signs[term_i] < 0:
                if n_anc == 1:
                    z(ancilla[0])
                else:
                    z.ctrl(ancilla.front(n_anc - 1), ancilla[n_anc - 1])
            back = ptr_ctrl - 1
            for b in range(n_anc):
                if term_controls[back] == 0:
                    x(ancilla[n_anc - 1 - b])
                back -= 1
            term_i += 1
    apply_basis_rotations(system, rot_ptr,
                          segment_rot_counts[len(frame_term_counts)],
                          rot_orbitals, rot_angles)


@cudaq.kernel
def controlled_select(control_and_ancilla: cudaq.qview, system: cudaq.qview,
                      term_controls: list[int], term_targets: list[int],
                      term_lengths: list[int], term_signs: list[int],
                      frame_term_counts: list[int],
                      segment_rot_counts: list[int], rot_orbitals: list[int],
                      rot_angles: list[float]):
    """SELECT controlled by qubit 0 of ``control_and_ancilla``.

    Only the Z words and sign phases carry the external control; the
    Givens segments stay uncontrolled and telescope to the identity when
    the control is |0> (nothing separates consecutive exit/entry pairs).
    """
    n_anc = control_and_ancilla.size() - 1
    rot_ptr = 0
    ptr_ctrl = 0
    ptr_tgt = 0
    term_i = 0
    for f in range(len(frame_term_counts)):
        apply_basis_rotations(system, rot_ptr, segment_rot_counts[f],
                              rot_orbitals, rot_angles)
        rot_ptr += segment_rot_counts[f]
        for _ in range(frame_term_counts[f]):
            for b in range(n_anc):
                if term_controls[ptr_ctrl] == 0:
                    x(control_and_ancilla[1 + b])
                ptr_ctrl += 1
            for _ in range(term_lengths[term_i]):
                z.ctrl(control_and_ancilla, system[term_targets[ptr_tgt]])
                ptr_tgt += 1
            if term_signs[term_i] < 0:
                total = control_and_ancilla.size()
                z.ctrl(control_and_ancilla.front(total - 1),
                       control_and_ancilla[total - 1])
            back = ptr_ctrl - 1
            for b in range(n_anc):
                if term_controls[back] == 0:
                    x(control_and_ancilla[n_anc - b])
                back -= 1
            term_i += 1
    apply_basis_rotations(system, rot_ptr,
                          segment_rot_counts[len(frame_term_counts)],
                          rot_orbitals, rot_angles)


@cudaq.kernel
def apply(ancilla: cudaq.qview, system: cudaq.qview, angles: list[float],
          term_controls: list[int], term_targets: list[int],
          term_lengths: list[int], term_signs: list[int],
          frame_term_counts: list[int], segment_rot_counts: list[int],
          rot_orbitals: list[int], rot_angles: list[float]):
    """Full block encoding: PREPARE, framed SELECT, PREPARE dagger."""
    prepare(ancilla, angles)
    select(ancilla, system, term_controls, term_targets, term_lengths,
           term_signs, frame_term_counts, segment_rot_counts, rot_orbitals,
           rot_angles)
    unprepare(ancilla, angles)


@cudaq.kernel
def walk(ancilla: cudaq.qview, system: cudaq.qview, angles: list[float],
         term_controls: list[int], term_targets: list[int],
         term_lengths: list[int], term_signs: list[int],
         frame_term_counts: list[int], segment_rot_counts: list[int],
         rot_orbitals: list[int], rot_angles: list[float]):
    """One qubitization walk step: SELECT, then reflect about PREPARE."""
    select(ancilla, system, term_controls, term_targets, term_lengths,
           term_signs, frame_term_counts, segment_rot_counts, rot_orbitals,
           rot_angles)
    unprepare(ancilla, angles)
    reflect_about_zero(ancilla)
    prepare(ancilla, angles)


@cudaq.kernel
def adjoint_walk(ancilla: cudaq.qview, system: cudaq.qview,
                 angles: list[float], term_controls: list[int],
                 term_targets: list[int], term_lengths: list[int],
                 term_signs: list[int], frame_term_counts: list[int],
                 segment_rot_counts: list[int], rot_orbitals: list[int],
                 rot_angles: list[float]):
    """Adjoint walk step: reflection first, then SELECT (both self-adjoint)."""
    reflect_about_prepare(ancilla, angles)
    select(ancilla, system, term_controls, term_targets, term_lengths,
           term_signs, frame_term_counts, segment_rot_counts, rot_orbitals,
           rot_angles)


@cudaq.kernel
def controlled_walk(control_and_ancilla: cudaq.qview, system: cudaq.qview,
                    angles: list[float], term_controls: list[int],
                    term_targets: list[int], term_lengths: list[int],
                    term_signs: list[int], frame_term_counts: list[int],
                    segment_rot_counts: list[int], rot_orbitals: list[int],
                    rot_angles: list[float]):
    """One walk step controlled by qubit 0 of ``control_and_ancilla``."""
    controlled_select(control_and_ancilla, system, term_controls, term_targets,
                      term_lengths, term_signs, frame_term_counts,
                      segment_rot_counts, rot_orbitals, rot_angles)
    controlled_reflect_about_prepare(control_and_ancilla, angles)


@cudaq.kernel
def controlled_adjoint_walk(control_and_ancilla: cudaq.qview,
                            system: cudaq.qview, angles: list[float],
                            term_controls: list[int], term_targets: list[int],
                            term_lengths: list[int], term_signs: list[int],
                            frame_term_counts: list[int],
                            segment_rot_counts: list[int],
                            rot_orbitals: list[int], rot_angles: list[float]):
    """One adjoint walk step controlled by qubit 0."""
    controlled_reflect_about_prepare(control_and_ancilla, angles)
    controlled_select(control_and_ancilla, system, term_controls, term_targets,
                      term_lengths, term_signs, frame_term_counts,
                      segment_rot_counts, rot_orbitals, rot_angles)


# ============================================================================
# Host-side construction
# ============================================================================


def _givens_sweep(orthogonal: np.ndarray) -> list[tuple[int, float]]:
    """Adjacent-Givens decomposition of an orthogonal matrix.

    Returns plane rotations ``(p, theta)`` on rows ``(p, p + 1)`` whose
    left-application reduces ``orthogonal`` to a diagonal of signs:
    ``R_m ... R_1 V = diag(+-1)``. The signs are irrelevant here: every
    encoded term is a number operator in the rotated basis, and number
    operators are invariant under per-mode sign flips.
    """
    work = np.array(orthogonal, dtype=float, copy=True)
    n = work.shape[0]
    rotations: list[tuple[int, float]] = []
    for col in range(n):
        for row in range(n - 1, col, -1):
            upper, lower = work[row - 1, col], work[row, col]
            if abs(lower) > 1e-14:
                theta = math.atan2(lower, upper)
                c, s = math.cos(theta), math.sin(theta)
                rotation = np.array([[c, s], [-s, c]])
                work[row - 1:row + 1, :] = rotation @ work[row - 1:row + 1, :]
                rotations.append((row - 1, theta))
    return rotations


def _forward_rotations(
        sweep: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Circuit program whose matrix is the frame's Gaussian rotation ``G``.

    From ``R_m ... R_1 V = D`` follows ``V = R_1^T ... R_m^T D``; lifting
    transposes negates angles, and the matrix product order means the last
    sweep rotation is applied first in circuit time. Conjugating a
    computational Z word ``W`` as ``G W G^dag`` therefore takes the
    *inverse* program before the word and this program after it.
    """
    return [(p, -theta) for p, theta in reversed(sweep)]


def _inverse_rotations(
        sweep: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Circuit program whose matrix is ``G^dag`` (frame entry)."""
    return [(p, theta) for p, theta in sweep]


def _leaf_pauli_expansion(
        core: np.ndarray,
        threshold: float) -> tuple[float, dict[tuple[int, ...], float]]:
    """ZZ expansion of the *centered* leaf operator (von Burg form).

    Writing ``N_k = (N_k - 1) + 1`` splits the leaf into

        1/2 sum_kl Z_kl N_k N_l = 1/2 sum_kl Z_kl (N_k - 1)(N_l - 1)
                                  + sum_k (sum_l Z_kl) N_k
                                  - 1/2 sum_kl Z_kl

    The middle line is a one-body operator the caller absorbs into the
    corrected one-body matrix (its own eigenbasis, no leaf terms). What
    remains here is pure ZZ: with interleaved spins ``N_k - 1 =
    -(Z_{2k} + Z_{2k+1}) / 2``, so each ``k < l`` pair yields four ZZ
    words of coefficient ``core_kl / 4`` and each diagonal ``k`` one
    cross-spin ZZ of ``core_kk / 4`` — which is why the encoding's
    ``alpha`` (minus the identity term) reproduces the LCU one-norm of
    ``double_factorization_one_norm`` exactly.

    Returns the identity coefficient and a map from qubit pairs to ZZ
    coefficients.
    """
    rank = core.shape[0]
    constant = -0.5 * float(core.sum()) + 0.25 * float(np.trace(core))
    # Each (k, l) pair contributes to distinct qubit pairs (diagonal ->
    # (2k, 2k+1); off-diagonal -> four (2k+s, 2l+t) with the second index
    # >= 2l > 2k+1), so no word accumulates from more than one term and the
    # threshold can be applied inline on the single coefficient.
    words: dict[tuple[int, ...], float] = {}
    for k in range(rank):
        c = 0.25 * float(core[k, k])
        if abs(c) >= threshold:
            words[(2 * k, 2 * k + 1)] = c
        for l in range(k + 1, rank):
            c = 0.125 * float(core[k, l] + core[l, k])
            if abs(c) < threshold:
                continue
            for sigma in range(2):
                for tau in range(2):
                    a, b = 2 * k + sigma, 2 * l + tau
                    words[(a, b) if a < b else (b, a)] = c
    return constant, words


class DoubleFactorizedEncoding:
    """Block encoding of the double-factorized electronic Hamiltonian.

    Parameters
    ----------
    one_body
        The ``(n, n)`` symmetric core-Hamiltonian matrix over real spatial
        orbitals (chemist conventions, as in
        ``cudaq_algorithms.chemistry``).
    two_body
        Either a ``DoubleFactorization`` (from
        ``double_factorization.explicit_double_factorization`` or
        ``compressed_double_factorization`` — truncation happens there) or
        a raw ``(n, n, n, n)`` chemist-notation ERI tensor, which is
        factorized exactly (``threshold=0.0``).
    scalar_offset
        Constant added as an identity term (e.g. nuclear repulsion).
    coefficient_threshold
        Encoded terms with ``|coefficient|`` below this are dropped.

    The encoded operator is ``H / alpha`` on the all-zero ancilla block,
    where ``H`` is the Hamiltonian *of the factorized (possibly
    truncated) tensor* — compressing the factorization directly lowers
    ``alpha`` and the SELECT term count.

    Satisfies the ``BlockEncoding`` protocol except ``select_observable``
    (LCU-specific odd-moment hook), which raises ``NotImplementedError``:
    odd Chebyshev moments via the observable trick are unavailable, even
    moments and all walk/QSVT circuits work unchanged.
    """

    def __init__(self,
                 one_body: ArrayLike,
                 two_body: Union[DoubleFactorization, ArrayLike],
                 *,
                 scalar_offset: float = 0.0,
                 coefficient_threshold: float = 1e-12) -> None:
        kappa_base = np.asarray(one_body, dtype=float)
        if kappa_base.ndim != 2 or kappa_base.shape[0] != kappa_base.shape[1]:
            raise ValueError("one_body must be a square (n, n) matrix")
        if not np.allclose(kappa_base, kappa_base.T, atol=1e-10):
            raise ValueError("one_body must be symmetric")
        n = kappa_base.shape[0]

        if isinstance(two_body, DoubleFactorization):
            factorization = two_body
        else:
            eri = np.asarray(two_body, dtype=float)
            if eri.shape != (n, n, n, n):
                raise ValueError(
                    "two_body must be a DoubleFactorization or an "
                    "(n, n, n, n) chemist-notation ERI tensor matching "
                    "one_body")
            # Fast-fail on a non-chemist-symmetric ERI (matching the one_body
            # symmetry check above): otherwise the factorizer symmetrizes it
            # and the encoding would silently block-encode an operator that
            # differs from the tensor passed in.
            for axes in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
                if not np.allclose(eri, eri.transpose(axes), atol=1e-8):
                    raise ValueError(
                        "two_body ERI must obey the chemist-notation "
                        "permutation symmetry (pq|rs) = (qp|rs) = (pq|sr) = "
                        "(rs|pq)")
            factorization = explicit_double_factorization(eri, threshold=0.0)
        if factorization.num_orbitals != n:
            raise ValueError(
                f"factorization has {factorization.num_orbitals} orbitals, "
                f"one_body has {n}")

        self._num_spatial = n
        self._num_system = 2 * n
        self._factorization = factorization

        # Corrected one-body matrix (von Burg's kappa): the exchange
        # correction -1/2 sum_r (pr|rq) (which contracts the *factorized*
        # tensor: U^T U = I collapses it to U diag(Z) U^T per leaf, so the
        # encoded H is exactly the Hamiltonian of the reconstructed ERI,
        # truncation included) plus the absorbed leaf singles
        # sum_l Z^t_kl from centering the number operators (see
        # ``_leaf_pauli_expansion``).
        kappa = kappa_base.copy()
        for rotation, core in zip(factorization.leaf_rotations,
                                  factorization.leaf_cores):
            # The leaf ZZ expansion symmetrizes the core (1/2 (core + core^T)),
            # so the matching one-body centering is 1/2 (rowsum + colsum). For
            # a symmetric core these coincide; the symmetric form keeps a
            # user-built non-symmetric leaf core consistent with what the ZZ
            # terms encode (off by 1/2 (colsum - rowsum) otherwise).
            absorbed = (0.5 * (core.sum(axis=1) + core.sum(axis=0)) -
                        0.5 * np.diag(core))
            kappa += (rotation * absorbed) @ rotation.T
        eigenvalues, eigenvectors = np.linalg.eigh(kappa)

        # Frame data: (sweep rotations, {qubit-tuple: coefficient}).
        constant = float(scalar_offset)
        frames: list[tuple[list[tuple[int, float]], dict[tuple[int, ...],
                                                         float]]] = []

        one_body_words: dict[tuple[int, ...], float] = {}
        for k in range(n):
            f_k = float(eigenvalues[k])
            constant += f_k
            if abs(0.5 * f_k) >= coefficient_threshold:
                one_body_words[(2 * k, )] = -0.5 * f_k
                one_body_words[(2 * k + 1, )] = -0.5 * f_k
        frames.append((_givens_sweep(eigenvectors), one_body_words))

        for rotation, core in zip(factorization.leaf_rotations,
                                  factorization.leaf_cores):
            leaf_constant, leaf_words = _leaf_pauli_expansion(
                core, coefficient_threshold)
            constant += leaf_constant
            # Keep every leaf frame, even if all its words were thresholded
            # away, so num_frames == 1 + num_leaves stays true and frame 0
            # remains the kappa eigenbasis. An empty frame contributes no
            # SELECT terms; its rotation sweep telescopes to the identity.
            frames.append((_givens_sweep(rotation), leaf_words))

        self._constant = constant

        # The identity total is one frameless term; it lives in the first
        # frame's group (rotations act trivially on it). Frames are retained
        # as placeholders (frame 0 is always present), so the count/index
        # invariant holds regardless of thresholding.
        terms: list[tuple[float, tuple[int, ...], int]] = []
        if abs(constant) >= coefficient_threshold:
            terms.append((constant, (), 0))
        if not terms and not any(words for _, words in frames):
            raise ValueError("hamiltonian has no retained terms")

        frame_term_counts = []
        for frame_index, (_, words) in enumerate(frames):
            count = len(words)
            if frame_index == 0 and terms:
                count += 1  # the identity term rides in frame 0
            for qubits, coefficient in words.items():
                terms.append((coefficient, qubits, frame_index))
            frame_term_counts.append(count)

        self._terms = terms
        self._alpha = sum(abs(c) for c, _, _ in terms)
        self._num_ancilla = max(1, (len(terms) - 1).bit_length())

        # Rotation program: each frame's terms are conjugated as
        # G W G^dag, so the segment before a frame's terms applies G^dag
        # (frame entry) and the segment after applies G; between frames
        # the two concatenate, and with no terms applied in between the
        # whole program telescopes to the identity.
        segments: list[list[tuple[int, float]]] = []
        segments.append(_inverse_rotations(frames[0][0]))
        for previous, current in zip(frames, frames[1:]):
            segments.append(
                _forward_rotations(previous[0]) +
                _inverse_rotations(current[0]))
        segments.append(_forward_rotations(frames[-1][0]))

        self._segment_rot_counts = [len(s) for s in segments]
        self._rot_orbitals = [p for s in segments for p, _ in s]
        self._rot_angles = [theta for s in segments for _, theta in s]
        if not self._rot_orbitals:
            # Padding only: no empty list crosses the kernel boundary
            # (cuda-quantum#4847); all segment counts are 0, so these are
            # never dereferenced.
            self._rot_orbitals = [0]
            self._rot_angles = [0.0]
        self._frame_term_counts = frame_term_counts

        probabilities = [abs(c) / self._alpha for c, _, _ in terms]
        probabilities += [0.0] * ((1 << self._num_ancilla) - len(terms))
        self._angles = _prepare_angles(probabilities)

        self._term_controls: list[int] = []
        self._term_targets: list[int] = []
        self._term_lengths: list[int] = []
        self._term_signs: list[int] = []
        for index, (coefficient, qubits, _) in enumerate(terms):
            for b in range(self._num_ancilla):
                self._term_controls.append((index >>
                                            (self._num_ancilla - 1 - b)) & 1)
            self._term_targets.extend(qubits)
            self._term_lengths.append(len(qubits))
            self._term_signs.append(-1 if coefficient < 0.0 else 1)
        if not self._term_targets:
            # Identity-only padding, same rationale as the rotation pad.
            self._term_targets = [0]

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
    def num_spatial_orbitals(self) -> int:
        return self._num_spatial

    @property
    def alpha(self) -> float:
        """Block-encoding normalization (1-norm of retained coefficients)."""
        return self._alpha

    @property
    def constant_term(self) -> float:
        """Total identity coefficient (offset + frame constants)."""
        return self._constant

    @property
    def num_terms(self) -> int:
        return len(self._terms)

    @property
    def num_frames(self) -> int:
        return len(self._frame_term_counts)

    @property
    def num_givens_rotations(self) -> int:
        """Spatial Givens rotations in the full SELECT program."""
        return sum(self._segment_rot_counts)

    @property
    def factorization(self) -> DoubleFactorization:
        return self._factorization

    @property
    def terms(self) -> list[tuple[float, tuple[int, ...], int]]:
        """Encoded terms as ``(coefficient, z_qubits, frame_index)``.

        ``z_qubits`` are the qubits carrying Z *in that frame's rotated
        orbital basis* (empty tuple = identity); frame 0 is the
        ``kappa'`` eigenbasis, frames ``1..`` follow the factorization's
        leaf order.
        """
        return list(self._terms)

    def __repr__(self) -> str:
        return (f"DoubleFactorizedEncoding(terms={self.num_terms}, "
                f"frames={self.num_frames}, "
                f"system_qubits={self.num_system}, "
                f"ancilla_qubits={self.num_ancilla}, "
                f"alpha={self.alpha:.6g})")

    @property
    def _kernel_data(self):
        """Internal, uncopied flattened arrays for the kernel factories."""
        return (self._angles, self._term_controls, self._term_targets,
                self._term_lengths, self._term_signs, self._frame_term_counts,
                self._segment_rot_counts, self._rot_orbitals, self._rot_angles)

    # ------------------------------------------------------------------
    # Convenience factories (mirroring PauliLCU)
    # ------------------------------------------------------------------

    def encode_kernel(self, state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying the full block encoding.

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state`` and the ancilla register (in
        |0...0>) after it. With ``state_prep`` (a ``(qubits: qview)``
        kernel): a zero-argument kernel that allocates the system register
        in |0...0>, runs ``state_prep`` on it, then applies the encoding.
        """
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data
        n_anc = self.num_ancilla
        n_sys = self.num_system

        if state_prep is not None:

            @cudaq.kernel
            def prep_encoded():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                apply(ancilla, system, angles, controls, targets, lengths,
                      signs, frame_counts, seg_counts, orbitals, thetas)

            return prep_encoded

        @cudaq.kernel
        def encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            apply(ancilla, system, angles, controls, targets, lengths, signs,
                  frame_counts, seg_counts, orbitals, thetas)

        return encoded

    def walk_kernel(self,
                    power: int = 1,
                    state_prep: Kernel | None = None) -> Kernel:
        """A kernel running PREPARE, walk^power, UNPREPARE.

        The all-zero-ancilla block of the result is T_power(-H/alpha)
        applied to the input state. Input modes as in ``encode_kernel``: a
        ``cudaq.State``-taking kernel, or a zero-argument kernel when
        ``state_prep`` is given.
        """
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data
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
                    walk(ancilla, system, angles, controls, targets, lengths,
                         signs, frame_counts, seg_counts, orbitals, thetas)
                unprepare(ancilla, angles)

            return prep_walked

        @cudaq.kernel
        def walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            prepare(ancilla, angles)
            for _ in range(steps):
                walk(ancilla, system, angles, controls, targets, lengths,
                     signs, frame_counts, seg_counts, orbitals, thetas)
            unprepare(ancilla, angles)

        return walked

    # ------------------------------------------------------------------
    # BlockEncoding protocol: data-free kernel factories
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
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def apply_encoding(ancilla: cudaq.qview, system: cudaq.qview):
            apply(ancilla, system, angles, controls, targets, lengths, signs,
                  frame_counts, seg_counts, orbitals, thetas)

        return apply_encoding

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0."""
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data
        n_anc = self.num_ancilla

        @cudaq.kernel
        def apply_controlled(control_and_ancilla: cudaq.qview,
                             system: cudaq.qview):
            prepare(control_and_ancilla.back(n_anc), angles)
            controlled_select(control_and_ancilla, system, controls, targets,
                              lengths, signs, frame_counts, seg_counts,
                              orbitals, thetas)
            unprepare(control_and_ancilla.back(n_anc), angles)

        return apply_controlled

    def walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one qubitization walk step W."""
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            walk(ancilla, system, angles, controls, targets, lengths, signs,
                 frame_counts, seg_counts, orbitals, thetas)

        return walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W†."""
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            adjoint_walk(ancilla, system, angles, controls, targets, lengths,
                         signs, frame_counts, seg_counts, orbitals, thetas)

        return adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def controlled_walk_step(control_and_ancilla: cudaq.qview,
                                 system: cudaq.qview):
            controlled_walk(control_and_ancilla, system, angles, controls,
                            targets, lengths, signs, frame_counts, seg_counts,
                            orbitals, thetas)

        return controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled adjoint walk step."""
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                         system: cudaq.qview):
            controlled_adjoint_walk(control_and_ancilla, system, angles,
                                    controls, targets, lengths, signs,
                                    frame_counts, seg_counts, orbitals, thetas)

        return controlled_adjoint_walk_step

    # ------------------------------------------------------------------
    # Observable hooks
    # ------------------------------------------------------------------

    def select_observable(self) -> Any:
        """Unavailable for this encoding (LCU-specific hook).

        The odd-Chebyshev-moment observable requires SELECT's terms to be
        Pauli words in the computational frame; here they are Z words in
        rotated orbital frames. Even moments (``Walk.moment`` with even
        order) and every kernel factory are unaffected.
        """
        raise NotImplementedError(
            "DoubleFactorizedEncoding does not provide select_observable: "
            "its SELECT terms are frame-rotated Z words, not computational-"
            "frame Pauli words; odd Chebyshev moments via the observable "
            "trick are unavailable for this encoding")
