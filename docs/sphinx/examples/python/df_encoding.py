# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Bring your own block encoding: a complete, worked ``BlockEncoding``.

``cudaq_algorithms`` consumers (``Walk``, ``QSVT``) are generic over the
structural ``BlockEncoding`` protocol — no inheritance, no registration.
Any object that exposes ``num_system`` / ``num_ancilla`` / ``alpha`` and
the documented kernel factories (``prepare_kernel``, ``apply_kernel``,
``walk_step_kernel``, the controlled/adjoint variants, ...) plugs into
qubitization and QSVT unchanged. This example is a full-scale
demonstration: a *double-factorized* encoding of the electronic-structure
Hamiltonian, built entirely on the public ``cudaq_algorithms`` API and
validated against an independent dense reference by
``tests/python/test_df_encoding.py``. Use it as the template for writing
your own encoding; ``df_block_encoding.py`` in this directory drives it
head to head with the built-in ``PauliLCU`` through the same ``Walk``.

Encodes the electronic-structure Hamiltonian directly from its
double-factorized form (von Burg et al., Phys. Rev. Research 3, 033055 (2021);
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
the structure exploited by the fixed backend. The optional
``rotation_backend="qrom", angle_bits=4`` instead loads a shared real
spatial frame from the term index, applies programmable inverse/forward
networks around the signed Z word, and erases the angle words. PREPARE
still uses the exact flat coefficient tree. This demonstrates coherent
frame selection, not the papers' nested DF preparation, degree-two
Chebyshev construction, optimized normalization, phase-gradient synthesis,
or optimized inverse-QROM cost (see also Lee et al., PRX Quantum 2,
030305 (2021), arXiv:2011.03494, Appendix C.2 and Figure 16).

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
from cudaq_algorithms.common_kernels import (controlled_reflect_about_zero,
                                             reflect_about_zero)
from cudaq_algorithms.basis_change import AngleEncoding
from cudaq_algorithms.primitives import QROM
from cudaq_algorithms.double_factorization import (
    DoubleFactorization, explicit_double_factorization)
from cudaq_algorithms.pauli_lcu import (controlled_reflect_about_prepare,
                                        prepare, reflect_about_prepare,
                                        unprepare)
from cudaq_algorithms.stateprep import (make_orbital_rotation_schedule,
                                        orbital_rotation_program,
                                        spatial_givens_rotation,
                                        spatial_givens_segment)

__all__ = [
    "DoubleFactorizedEncoding",
    "select",
    "controlled_select",
    "apply",
    "walk",
    "adjoint_walk",
    "controlled_walk",
    "controlled_adjoint_walk",
]

# ----------------------------------------------------------------------
# Small host-side helpers (input guard + state-preparation angle tree),
# kept local so this file is self-contained on the public
# ``cudaq_algorithms`` surface and can be copied wholesale as the
# starting point for a new encoding.
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
        spatial_givens_segment(system, rot_orbitals, rot_angles, rot_ptr,
                               segment_rot_counts[f])
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
    spatial_givens_segment(system, rot_orbitals, rot_angles, rot_ptr,
                           segment_rot_counts[len(frame_term_counts)])


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
    the control is ``|0>`` (nothing separates consecutive exit/entry pairs).
    """
    n_anc = control_and_ancilla.size() - 1
    rot_ptr = 0
    ptr_ctrl = 0
    ptr_tgt = 0
    term_i = 0
    for f in range(len(frame_term_counts)):
        spatial_givens_segment(system, rot_orbitals, rot_angles, rot_ptr,
                               segment_rot_counts[f])
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
    spatial_givens_segment(system, rot_orbitals, rot_angles, rot_ptr,
                           segment_rot_counts[len(frame_term_counts)])


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


def _spatial_program(matrix):
    """Fixed QR topology, retaining zero slots; forward gives F = V D.

    spatial_givens_rotation has orbital block [[cos,sin],[-sin,cos]],
    opposite to basis_change's real mixing gate. Diagonal signs D commute
    with all our Z words and cancel in their conjugation.
    """
    n = len(matrix)
    work = np.array(matrix, dtype=float, copy=True)
    eliminated = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            upper, lower = work[row - 1, col], work[row, col]
            theta = 0. if lower == 0. else math.atan2(lower, upper)
            c, s = math.cos(theta), math.sin(theta)
            work[row - 1:row +
                 1] = np.array([[c, s], [-s, c]]) @ work[row - 1:row + 1]
            eliminated.append((row - 1, theta))
    signs = np.where(np.diag(work) >= 0, 1., -1.)
    # Match the existing spatial schedule's residual allowance. This host
    # decomposition check is separate from the 1e-10 circuit/reference tests.
    np.testing.assert_allclose(work, np.diag(signs), rtol=0, atol=1e-9)
    modes = [mode for mode, _ in reversed(eliminated)]
    angles = [-theta for _, theta in reversed(eliminated)]
    return modes, angles, np.asarray(matrix) * signs


def _spatial_matrix(n, modes, angles):
    """NumPy one-particle product, independent of device amplitudes."""
    matrix = np.eye(n)
    for mode, theta in zip(modes, angles):
        c, s = math.cos(theta), math.sin(theta)
        gate = np.eye(n)
        gate[mode:mode + 2, mode:mode + 2] = [[c, s], [-s, c]]
        matrix = gate @ matrix
    return matrix


@cudaq.kernel
def programmable_spatial_givens(words: cudaq.qview, system: cudaq.qview,
                                modes: list[int], weights: list[float]):
    """One real spatial frame, shared by both interleaved spin sectors.

    Word bit k has weight 2*pi*2**(k-b). The public spatial primitive
    includes the intervening Jordan-Wigner parity qubits.
    """
    bits = len(weights)
    for slot in range(len(modes)):
        for bit in range(bits):
            cudaq.control(spatial_givens_rotation, words[slot * bits + bit],
                          system, weights[bit], modes[slot])


class DoubleFactorizedEncoding:
    """Block encoding of the double-factorized electronic Hamiltonian.

    :param one_body: The ``(n, n)`` symmetric core-Hamiltonian matrix over
        real spatial orbitals (chemist conventions, as in
        ``cudaq_algorithms.chemistry``).
    :param two_body: Either a ``DoubleFactorization`` (from
        ``double_factorization.explicit_double_factorization`` or
        ``compressed_double_factorization`` -- truncation happens there) or
        a raw ``(n, n, n, n)`` chemist-notation ERI tensor, which is
        factorized exactly (``threshold=0.0``).
    :param scalar_offset: Constant added as an identity term (e.g. nuclear
        repulsion).
    :param coefficient_threshold: Encoded terms with ``|coefficient|`` below
        this are dropped.
    :param rotation_backend: ``"fixed"`` (default) sweeps classical frames;
        ``"qrom"`` coherently loads angle words for one shared spatial network.
    :param angle_bits: Unsigned full-turn width for the QROM backend (1..52).
        The default is four; fixed circuits allocate no words or QROM ladder.

    The encoded operator is ``H / alpha`` on the all-zero ancilla block,
    where ``H`` is the Hamiltonian *of the factorized (possibly
    truncated) tensor* after coefficient thresholding for the fixed backend.
    QROM additionally rounds its retained orbital frames. Both backends
    use the same coefficients and alpha; angle error is separate from
    factorization and threshold error.

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
                 coefficient_threshold: float = 1e-12,
                 rotation_backend: str = "fixed",
                 angle_bits: int = 4) -> None:
        if rotation_backend not in ("fixed", "qrom"):
            raise ValueError("rotation_backend must be 'fixed' or 'qrom'")
        self._rotation_backend = rotation_backend
        self._angle_encoding = (AngleEncoding(angle_bits)
                                if rotation_backend == "qrom" else None)
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
        frame_matrices = [eigenvectors] + list(factorization.leaf_rotations)

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
        frames.append(
            (make_orbital_rotation_schedule(eigenvectors), one_body_words))

        for rotation, core in zip(factorization.leaf_rotations,
                                  factorization.leaf_cores):
            leaf_constant, leaf_words = _leaf_pauli_expansion(
                core, coefficient_threshold)
            constant += leaf_constant
            # Keep every leaf frame, even if all its words were thresholded
            # away, so num_frames == 1 + num_leaves stays true and frame 0
            # remains the kappa eigenbasis. An empty frame contributes no
            # SELECT terms; its rotation sweep telescopes to the identity.
            frames.append(
                (make_orbital_rotation_schedule(rotation), leaf_words))

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
        segments: list[tuple[list[int], list[float]]] = []
        segments.append(orbital_rotation_program(frames[0][0], inverse=True))
        for previous, current in zip(frames, frames[1:]):
            exit_modes, exit_angles = orbital_rotation_program(previous[0],
                                                               inverse=False)
            enter_modes, enter_angles = orbital_rotation_program(current[0],
                                                                 inverse=True)
            segments.append(
                (exit_modes + enter_modes, exit_angles + enter_angles))
        segments.append(orbital_rotation_program(frames[-1][0], inverse=False))

        self._segment_rot_counts = [len(modes) for modes, _ in segments]
        self._rot_orbitals = [p for modes, _ in segments for p in modes]
        self._rot_angles = [
            theta for _, angles in segments for theta in angles
        ]
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
        self._configure_rotations(frame_matrices)

    def _configure_rotations(self, frames):
        self._num_index = self._num_ancilla
        programs = [_spatial_program(v) for v in frames]
        self._mixing_modes = programs[0][0]
        self._ideal_frames = [f for _, _, f in programs]
        self._frame_angles = [a for _, a, _ in programs]
        self._represented_frames = [f.copy() for f in self._ideal_frames]
        self._lookup = None
        self._program_width = self._ladder_width = 0
        self._frame_words = [() for _ in frames]
        self._frame_bounds = [0.] * len(frames)
        self._angle_rows = ()
        if self._angle_encoding is not None and self._mixing_modes:
            codec = self._angle_encoding
            self._frame_words = [
                tuple(codec.encode(t) for t in angles)
                for angles in self._frame_angles
            ]
            decoded = [[codec.decode(code) for code in row]
                       for row in self._frame_words]
            self._represented_frames = [
                _spatial_matrix(self._num_spatial, self._mixing_modes, angles)
                for angles in decoded
            ]
            # Two spin sectors double the per-angle Fock bound.
            self._frame_bounds = [
                min(
                    2., 2 * sum(
                        abs(math.atan2(math.sin(a - b), math.cos(a - b)))
                        for a, b in zip(new, old)))
                for new, old in zip(decoded, self._frame_angles)
            ]
            self._program_width = len(self._mixing_modes) * codec.width
            rows = [0] * (1 << self._num_index)
            for j, (_, qubits, frame) in enumerate(self._terms):
                # PREPARE/SELECT index qubit 0 is MSB; QROM qubit 0 is LSB.
                address = int(f"{j:0{self._num_index}b}"[::-1], 2)
                if qubits:  # identity and padded addresses use a zero program
                    rows[address] = sum(
                        code << (codec.width * slot)
                        for slot, code in enumerate(self._frame_words[frame]))
            self._angle_rows = tuple(rows)
            self._lookup = QROM(rows,
                                address_bits=self._num_index,
                                output_bits=self._program_width,
                                variant="select")
            self._ladder_width = self._lookup.num_ladder
        self._num_ancilla += self._program_width + self._ladder_width

    @property
    def ideal_frame_matrices(self):
        """Copies of sign-gauged real spatial frames F=V D, in frame order."""
        return tuple(f.copy() for f in self._ideal_frames)

    @property
    def represented_frame_matrices(self):
        """Copies of F_b from decoded words (F itself for the fixed backend)."""
        return tuple(f.copy() for f in self._represented_frames)

    @property
    def rotation_report(self):
        """Detached metadata; no angles or extra workspace for fixed circuits."""
        bits = None if self._angle_encoding is None else self._angle_encoding.width
        slots = len(self._mixing_modes)
        return dict(
            backend=self._rotation_backend,
            angle_bits=bits,
            mixing_slots=slots,
            term_index_width=self._num_index,
            program_width=self._program_width,
            ladder_width=self._ladder_width,
            qrom_variant=None if self._lookup is None else "select",
            total_ancillas=self.num_ancilla,
            table_size=len(self._angle_rows),
            angle_rows=self._angle_rows,
            frame_words=tuple(self._frame_words),
            frame_error_bounds=tuple(self._frame_bounds),
            prepare_rotations=len(self._angles),
            lookup_calls=0 if self._lookup is None else 2,
            programmable_spatial_calls=(2 * slots * bits
                                        if self._lookup is not None else 0))

    def select_kernel(self, *, controlled=False):
        """Clean-workspace SELECT; optional enable is the first operand qubit.

        Layout is [index | angle words | QROM ladder], or
        [enable | index | angle words | ladder] for the controlled form.
        The index and system may remain entangled after SELECT.
        """
        controls, targets = self._term_controls, self._term_targets
        lengths, signs = self._term_lengths, self._term_signs
        index_width = self._num_index
        shift = int(controlled)
        if self._lookup is None:
            counts = self._frame_term_counts
            segments, modes, angles = (self._segment_rot_counts,
                                       self._rot_orbitals, self._rot_angles)
            if controlled:

                @cudaq.kernel
                def fixed_controlled(register: cudaq.qview,
                                     system: cudaq.qview):
                    controlled_select(register, system, controls, targets,
                                      lengths, signs, counts, segments, modes,
                                      angles)

                return fixed_controlled

            @cudaq.kernel
            def fixed_select(register: cudaq.qview, system: cudaq.qview):
                select(register, system, controls, targets, lengths, signs,
                       counts, segments, modes, angles)

            return fixed_select

        lookup = self._lookup.kernel
        program_end = shift + index_width + self._program_width
        index_end = shift + index_width
        total = shift + self.num_ancilla
        modes = self._mixing_modes
        weights = list(self._angle_encoding.weights)

        @cudaq.kernel
        def selected(register: cudaq.qview, system: cudaq.qview):
            lookup(register[shift:index_end], register[program_end:total],
                   register[index_end:program_end])
            cudaq.adjoint(programmable_spatial_givens,
                          register[index_end:program_end], system, modes,
                          weights)
            ctrl_ptr = 0
            target_ptr = 0
            for term in range(len(lengths)):
                for bit in range(index_width):
                    if controls[ctrl_ptr] == 0:
                        x(register[shift + bit])
                    ctrl_ptr += 1
                for _ in range(lengths[term]):
                    z.ctrl(register.front(index_end),
                           system[targets[target_ptr]])
                    target_ptr += 1
                if signs[term] < 0:
                    if index_end == 1:
                        z(register[0])
                    else:
                        z.ctrl(register.front(index_end - 1),
                               register[index_end - 1])
                for bit in range(index_width):
                    if controls[ctrl_ptr - index_width + bit] == 0:
                        x(register[shift + bit])
            programmable_spatial_givens(register[index_end:program_end],
                                        system, modes, weights)
            cudaq.adjoint(lookup, register[shift:index_end],
                          register[program_end:total],
                          register[index_end:program_end])

        return selected

    def _programmed_step(self,
                         *,
                         controlled=False,
                         walk_step=False,
                         adjoint=False):
        selector = self.select_kernel(controlled=controlled)
        prepare_index, unprepare_index = self.prepare_kernel(
        ), self.unprepare_kernel()
        n_anc = self.num_ancilla
        reflection = (controlled_reflect_about_zero
                      if controlled else reflect_about_zero)

        @cudaq.kernel
        def operation(register: cudaq.qview, system: cudaq.qview):
            if walk_step:
                if adjoint:
                    unprepare_index(register.back(n_anc))
                    reflection(register)
                    prepare_index(register.back(n_anc))
                    selector(register, system)
                else:
                    selector(register, system)
                    unprepare_index(register.back(n_anc))
                    reflection(register)
                    prepare_index(register.back(n_anc))
            else:
                prepare_index(register.back(n_anc))
                selector(register, system)
                unprepare_index(register.back(n_anc))

        return operation

    def _programmed_circuit(self, power=None, state_prep=None):
        n_sys, n_anc = self.num_system, self.num_ancilla
        prepare_index, unprepare_index = self.prepare_kernel(
        ), self.unprepare_kernel()
        operation = (self.apply_kernel()
                     if power is None else self.walk_step_kernel())
        steps = 1 if power is None else _validate_power(power)
        is_walk = power is not None

        @cudaq.kernel
        def run(system: cudaq.qview, ancilla: cudaq.qview):
            if is_walk:
                prepare_index(ancilla)
            for _ in range(steps):
                operation(ancilla, system)
            if is_walk:
                unprepare_index(ancilla)

        if state_prep is not None:

            @cudaq.kernel
            def prepared():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                run(system, ancilla)

            return prepared

        @cudaq.kernel
        def from_state(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            run(system, ancilla)

        return from_state

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
        if self._lookup is not None:
            return 2 * len(self._mixing_modes) * self._angle_encoding.width
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
        ``|0...0>``) after it. With ``state_prep`` (a ``(qubits: qview)``
        kernel): a zero-argument kernel that allocates the system register
        in ``|0...0>``, runs ``state_prep`` on it, then applies the encoding.
        """
        if self._rotation_backend == "qrom":
            return self._programmed_circuit(state_prep=state_prep)
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
        if self._rotation_backend == "qrom":
            return self._programmed_circuit(power=steps, state_prep=state_prep)

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
        n_index = self._num_index

        @cudaq.kernel
        def prepare_ancilla(ancilla: cudaq.qview):
            prepare(ancilla.front(n_index), angles)

        return prepare_ancilla

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE dagger with this encoding's angles."""
        angles = self._angles
        n_index = self._num_index

        @cudaq.kernel
        def unprepare_ancilla(ancilla: cudaq.qview):
            unprepare(ancilla.front(n_index), angles)

        return unprepare_ancilla

    def apply_kernel(self) -> Kernel:
        """``(ancilla, system)``: the full block encoding U_A."""
        if self._rotation_backend == "qrom":
            return self._programmed_step()
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def apply_encoding(ancilla: cudaq.qview, system: cudaq.qview):
            apply(ancilla, system, angles, controls, targets, lengths, signs,
                  frame_counts, seg_counts, orbitals, thetas)

        return apply_encoding

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0."""
        if self._rotation_backend == "qrom":
            return self._programmed_step(controlled=True)
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
        if self._rotation_backend == "qrom":
            return self._programmed_step(walk_step=True)
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            walk(ancilla, system, angles, controls, targets, lengths, signs,
                 frame_counts, seg_counts, orbitals, thetas)

        return walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W†."""
        if self._rotation_backend == "qrom":
            return self._programmed_step(walk_step=True, adjoint=True)
        (angles, controls, targets, lengths, signs, frame_counts, seg_counts,
         orbitals, thetas) = self._kernel_data

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            adjoint_walk(ancilla, system, angles, controls, targets, lengths,
                         signs, frame_counts, seg_counts, orbitals, thetas)

        return adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        if self._rotation_backend == "qrom":
            return self._programmed_step(controlled=True, walk_step=True)
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
        if self._rotation_backend == "qrom":
            return self._programmed_step(controlled=True,
                                         walk_step=True,
                                         adjoint=True)
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
