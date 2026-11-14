# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hartree-Fock reference states and fixed-parameter UCC (host side).

Host helpers for the ``hartree_fock`` / ``hartree_fock_occupation`` /
``fixed_parameter_ucc`` device kernels in ``_kernels``:

* ``make_hartree_fock_occupation`` builds the occupied spin-orbital list
  of the Hartree-Fock determinant with alpha on even orbitals and beta
  on odd orbitals for nonzero spin. Spin zero uses contiguous indices.
* ``get_fixed_parameter_ucc_pauli_lists`` converts any operator pool to
  the grouped (Pauli words, coefficients) form the kernel takes, dropping
  near-zero terms and rejecting complex coefficients.
* ``hartree_fock_ucc_kernel`` returns a ``(qubits: qview)`` kernel that
  prepares the reference determinant and applies the fixed-amplitude UCC
  product — directly injectable as a ``state_prep`` kernel. The grouped
  data is flattened before capture: nested lists marshal as kernel
  *arguments* but cannot be closure-*captured*
  (``getElementType(): incompatible function arguments``).
* Resource estimators, as plain functions returning frozen dataclasses.

There is deliberately no "plan" object and no statevector convenience:
simulate a factory kernel through an entry kernel and ``cudaq.get_state``
(statevector helpers live in ``sim_utils``, outside the library surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import cudaq

from ._kernels import hartree_fock_occupation
from ._pools import _as_count

# ============================================================================
# Device kernel (private): flattened fixed-parameter UCC for the factory
# ============================================================================


@cudaq.kernel
def _fixed_parameter_ucc_flat(qubits: cudaq.qview, angles: list[float],
                              words: list[cudaq.pauli_word]):
    """Flattened ``fixed_parameter_ucc`` used by the kernel factory (nested
    lists cannot be captured); ``angles[j] = theta_group(j) * coeff_j``."""
    for j in range(len(words)):
        exp_pauli(angles[j], qubits, words[j])


# ============================================================================
# Resource summaries
# ============================================================================


@dataclass(frozen=True)
class HartreeFockResourceEstimate:
    """Circuit cost of a Hartree-Fock reference preparation."""

    num_qubits: int
    num_electrons: int
    num_x_gates: int


@dataclass(frozen=True)
class FixedParameterUccResourceEstimate:
    """Circuit cost of a fixed-parameter UCC product (Pauli rotations)."""

    num_qubits: int
    num_excitations: int
    num_pauli_rotations: int
    max_pauli_rotations_per_excitation: int


# ============================================================================
# Hartree-Fock occupations
# ============================================================================


def make_hartree_fock_occupation(num_qubits, num_electrons, spin=0):
    """Occupied spin-orbital indices of the Hartree-Fock reference.

    ``spin`` is the signed integer difference between the alpha and beta
    electron counts (twice the spin projection). Spin zero returns contiguous
    indices starting at zero. For nonzero spin, alpha electrons occupy
    even indices and beta electrons occupy odd indices. For example,
    ``make_hartree_fock_occupation(2, 1, spin=-1)`` returns ``[1]``.
    Nonzero spin requires an even qubit count, and the electron count
    and spin must have the same parity. Each spin population must fit
    in its half of the register.
    """
    num_qubits = _as_count(num_qubits, "num_qubits")
    num_electrons = _as_count(num_electrons, "num_electrons")
    if isinstance(spin, bool):
        raise ValueError("spin must be an integer")
    try:
        spin_number = int(spin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spin must be an integer") from exc
    if spin_number != spin:
        raise ValueError("spin must be an integer")
    if num_electrons > num_qubits:
        raise ValueError("num_electrons cannot exceed num_qubits")

    if spin_number == 0:
        return list(range(num_electrons))

    if num_qubits % 2 != 0:
        raise ValueError("num_qubits must be even when spin is nonzero")
    if abs(spin_number) > num_electrons:
        raise ValueError("abs(spin) cannot exceed num_electrons")
    if (num_electrons - spin_number) % 2 != 0:
        raise ValueError("(num_electrons - spin) must be even when spin "
                         "is nonzero (spin is 2*S_z)")

    num_spatial = num_qubits // 2
    n_occ_beta = (num_electrons - spin_number) // 2
    n_occ_alpha = num_electrons - n_occ_beta
    if max(n_occ_alpha, n_occ_beta) > num_spatial:
        raise ValueError("the requested (num_electrons, spin) does not fit in "
                         "num_qubits spin orbitals")

    occupied = [2 * i for i in range(n_occ_alpha)]
    occupied += [2 * i + 1 for i in range(n_occ_beta)]
    return sorted(occupied)


def validate_hartree_fock_occupation(num_qubits, occupied_orbitals):
    """Reject out-of-range, duplicate, or non-integral orbital indices."""
    num_qubits = _as_count(num_qubits, "num_qubits")
    seen = set()
    for orbital in occupied_orbitals:
        index = _as_count(orbital, "occupied orbital index")
        if index >= num_qubits:
            raise ValueError("occupied orbital index exceeds num_qubits")
        if index in seen:
            raise ValueError("occupied orbital indices must be unique")
        seen.add(index)


def estimate_hartree_fock_resources(num_qubits,
                                    num_electrons,
                                    spin=0) -> HartreeFockResourceEstimate:
    """Resource estimate for the canonical Hartree-Fock reference."""
    occupation = make_hartree_fock_occupation(num_qubits, num_electrons, spin)
    return estimate_hartree_fock_occupation_resources(num_qubits, occupation)


def estimate_hartree_fock_occupation_resources(
        num_qubits, occupied_orbitals) -> HartreeFockResourceEstimate:
    """Resource estimate for an explicit-occupation reference."""
    validate_hartree_fock_occupation(num_qubits, occupied_orbitals)
    num_qubits = _as_count(num_qubits, "num_qubits")
    return HartreeFockResourceEstimate(num_qubits=num_qubits,
                                       num_electrons=len(occupied_orbitals),
                                       num_x_gates=len(occupied_orbitals))


# ============================================================================
# Fixed-parameter UCC host helpers
# ============================================================================


def get_fixed_parameter_ucc_pauli_lists(operator_pool,
                                        num_qubits,
                                        coefficient_tolerance=1.0e-12):
    """Any operator pool as (Pauli word groups, coefficient groups).

    Like the pool-specific ``get_*_pauli_lists`` helpers, but for an
    arbitrary pool (e.g. ``make_uccsd_operator_pool``): one group per pool
    operator, in pool order, ready for the ``fixed_parameter_ucc`` kernel.
    Terms with ``|coefficient| <= coefficient_tolerance`` are dropped
    (they would waste identity rotations); coefficients with an imaginary
    part above the tolerance are rejected — ``exp_pauli`` angles are real.
    """
    num_qubits = _as_count(num_qubits, "num_qubits")
    if coefficient_tolerance < 0.0:
        raise ValueError("coefficient_tolerance must be non-negative")
    words_list = []
    coefficients_list = []
    for op in operator_pool:
        words = []
        coefficients = []
        for term in cudaq.SpinOperator(op):
            coefficient = term.evaluate_coefficient()
            if abs(coefficient.imag) > coefficient_tolerance:
                raise ValueError("only real operator-pool coefficients are "
                                 "supported")
            if abs(coefficient.real) <= coefficient_tolerance:
                continue
            words.append(cudaq.pauli_word(str(
                term.get_pauli_word(num_qubits))))
            coefficients.append(float(coefficient.real))
        words_list.append(words)
        coefficients_list.append(coefficients)
    return words_list, coefficients_list


def validate_fixed_parameter_ucc(num_qubits, parameters, pauli_words,
                                 coefficients):
    """Validate grouped fixed-parameter UCC data against ``num_qubits``.

    Pauli words given as strings are checked for length and alphabet;
    ``cudaq.pauli_word`` objects expose no accessor and are trusted (the
    ``get_*_pauli_lists`` helpers build them at the right width).
    """
    num_qubits = _as_count(num_qubits, "num_qubits")
    if (len(parameters) != len(pauli_words)
            or len(pauli_words) != len(coefficients)):
        raise ValueError("parameters, Pauli-word groups, and coefficient "
                         "groups must have the same length")
    for words, coeffs in zip(pauli_words, coefficients):
        if len(words) != len(coeffs):
            raise ValueError("each Pauli-word group must match its "
                             "coefficient group")
        for word in words:
            if isinstance(word, str):
                # Every valid word spans the whole register (the get_*_pauli_
                # lists helpers pad to full width); a short word would slip
                # past here and fail later inside exp_pauli at kernel launch.
                if len(word) != num_qubits:
                    raise ValueError("Pauli word width must equal num_qubits")
                if any(ch not in "IXYZ" for ch in word):
                    raise ValueError(f"unsupported Pauli word: {word!r}")


def estimate_fixed_parameter_ucc_resources(
        num_qubits, pauli_words) -> FixedParameterUccResourceEstimate:
    """Resource estimate for a fixed-parameter UCC product."""
    group_sizes = [len(group) for group in pauli_words]
    return FixedParameterUccResourceEstimate(
        num_qubits=_as_count(num_qubits, "num_qubits"),
        num_excitations=len(group_sizes),
        num_pauli_rotations=sum(group_sizes),
        max_pauli_rotations_per_excitation=max(group_sizes, default=0))


# ============================================================================
# Kernel factory
# ============================================================================


def hartree_fock_ucc_kernel(num_qubits,
                            parameters,
                            pauli_words,
                            coefficients,
                            *,
                            num_electrons=None,
                            spin=0,
                            occupied_orbitals=None):
    """A ``(qubits: qview)`` kernel: Hartree-Fock reference + UCC product.

    Provide exactly one of ``num_electrons`` (with optional signed ``spin``
    for an open-shell reference) or explicit ``occupied_orbitals``. The
    returned kernel expects a ``num_qubits``-wide register in ``|0...0>`` and
    is directly injectable as a ``state_prep`` kernel (e.g. into
    ``PauliLCU.encode_kernel``). The grouped data is flattened into
    per-rotation angles before capture — nested lists cannot be captured
    by Python kernels.
    """
    num_qubits = _as_count(num_qubits, "num_qubits")
    validate_fixed_parameter_ucc(num_qubits, parameters, pauli_words,
                                 coefficients)
    if (num_electrons is None) == (occupied_orbitals is None):
        raise ValueError(
            "provide exactly one of num_electrons or occupied_orbitals")
    if occupied_orbitals is None:
        occupation = make_hartree_fock_occupation(num_qubits, num_electrons,
                                                  spin)
    else:
        if spin != 0:
            raise ValueError(
                "spin only applies with num_electrons; encode the open-shell "
                "reference directly in occupied_orbitals")
        # Validate the raw indices (rejecting non-integral / bool) rather
        # than int()-coercing first, which would truncate e.g. 2.5 -> 2 into
        # a silently wrong reference determinant.
        occupation = [
            _as_count(index, "occupied orbital index")
            for index in occupied_orbitals
        ]
        validate_hartree_fock_occupation(num_qubits, occupation)

    flat_angles = []
    flat_words = []
    for theta, words, coeffs in zip(parameters, pauli_words, coefficients):
        for word, coeff in zip(words, coeffs):
            flat_angles.append(float(theta) * float(coeff))
            flat_words.append(word if isinstance(word, cudaq.pauli_word) else
                              cudaq.pauli_word(str(word)))

    # Positive guards choose a kernel shape whose captured lists are all
    # non-empty: empty list captures fail to launch (cuda-quantum#4847).
    if len(occupation) > 0 and len(flat_words) > 0:

        @cudaq.kernel
        def hf_then_ucc(qubits: cudaq.qview):
            hartree_fock_occupation(qubits, occupation)
            _fixed_parameter_ucc_flat(qubits, flat_angles, flat_words)

        return hf_then_ucc

    if len(occupation) > 0:

        @cudaq.kernel
        def hf_only(qubits: cudaq.qview):
            hartree_fock_occupation(qubits, occupation)

        return hf_only

    if len(flat_words) > 0:

        @cudaq.kernel
        def ucc_only(qubits: cudaq.qview):
            _fixed_parameter_ucc_flat(qubits, flat_angles, flat_words)

        return ucc_only

    @cudaq.kernel
    def identity_prep(qubits: cudaq.qview):
        pass

    return identity_prep
