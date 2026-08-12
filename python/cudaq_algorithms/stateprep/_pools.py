# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Excitation enumeration and operator pools for state preparation.

Pure-Python port of ``lib/stateprep/excitations.cpp`` and the host-side
helpers of ``lib/stateprep/device/*.cpp``, built with ``cudaq.spin``
algebra. Enumeration orders, index conventions, and coefficient signs
match the C++ implementation exactly (the tests pin them):

* UCCSD excitations follow the interleaved alpha/beta spin-orbital layout
  (alpha even, beta odd) with the closed-shell (spin == 0) and open-shell
  (spin > 0) occupancy rules of the C++ ``get_uccsd_excitations``.
* UCCGSD doubles enumerate the three pairings of every 4-combination of
  qubits, normalized to (high, low) within each pair and deduplicated in
  sorted order (the C++ ``std::set`` iteration order).
* CEO singles and doubles follow the pairing conventions of
  https://arxiv.org/abs/2407.08696; each double contributes two operators.
"""

from __future__ import annotations

import cudaq
from cudaq import spin

# ============================================================================
# Spin-algebra helpers
# ============================================================================


def _as_count(value, name):
    """Validate an integral, non-negative count (no silent coercion).

    The compiled bindings rejected negative and fractional inputs at the
    ``size_t`` type level; this keeps that contract explicit. Booleans
    are rejected too (``True`` is not a qubit count).
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    count = int(value)
    if count != value or count < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return count


def _z_parity(low, high):
    """Product of Z operators strictly between low and high (or None)."""
    parity = None
    for i in range(low + 1, high):
        factor = spin.z(i)
        parity = factor if parity is None else parity * factor
    return parity


def _times(*factors):
    """Product of the non-None factors (None when all are None)."""
    result = None
    for factor in factors:
        if factor is None:
            continue
        result = factor if result is None else result * factor
    return result


# ============================================================================
# UCCSD excitations and pool
# ============================================================================


def get_uccsd_excitations(num_qubits, num_electrons, spin=0):
    """Enumerate UCCSD excitations for the interleaved spin-orbital layout.

    Returns ``(singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
    doubles_beta)`` as lists of index lists, in the C++ enumeration order
    (which fixes the parameter order of the ``uccsd`` kernel).
    """
    num_qubits = _as_count(num_qubits, "num_qubits")
    num_electrons = _as_count(num_electrons, "num_electrons")
    spin_number = _as_count(spin, "spin")
    if num_qubits % 2 != 0:
        raise RuntimeError("The total number of qubits should be even.")
    # The C++ implementation has no defined behavior for these (unsigned
    # underflow); reject them explicitly.
    if num_electrons > num_qubits:
        raise ValueError("num_electrons cannot exceed num_qubits")
    if spin_number > num_electrons:
        raise ValueError("spin cannot exceed num_electrons")

    num_spatial = num_qubits // 2
    if spin_number > 0:
        n_occ_beta = (num_electrons - spin_number) // 2
        n_occ_alpha = num_electrons - n_occ_beta
        if n_occ_alpha > num_spatial:
            raise ValueError(
                "the requested (num_electrons, spin) does not fit in "
                "num_qubits spin orbitals")
        n_virt_alpha = num_spatial - n_occ_alpha
        n_virt_beta = num_spatial - n_occ_beta
        occupied_alpha = [i * 2 for i in range(n_occ_alpha)]
        virtual_alpha = [i * 2 + 2 * n_occ_alpha for i in range(n_virt_alpha)]
        occupied_beta = [i * 2 + 1 for i in range(n_occ_beta)]
        virtual_beta = [i * 2 + 2 * n_occ_beta + 1 for i in range(n_virt_beta)]
    elif num_electrons % 2 == 0 and spin_number == 0:
        n_occ = num_electrons // 2
        n_virt = num_spatial - n_occ
        occupied_alpha = [i * 2 for i in range(n_occ)]
        virtual_alpha = [i * 2 + num_electrons for i in range(n_virt)]
        occupied_beta = [i * 2 + 1 for i in range(n_occ)]
        virtual_beta = [i * 2 + num_electrons + 1 for i in range(n_virt)]
    else:
        raise RuntimeError(
            "Incorrect spin multiplicity. Number of electrons is odd but "
            f"spin is 0 {num_electrons}, {spin_number}")

    singles_alpha = [[p, q] for p in occupied_alpha for q in virtual_alpha]
    singles_beta = [[p, q] for p in occupied_beta for q in virtual_beta]
    doubles_mixed = [[p, q, r, s] for p in occupied_alpha
                     for q in occupied_beta for r in virtual_beta
                     for s in virtual_alpha]

    doubles_alpha = []
    for p in range(len(occupied_alpha) - 1):
        for q in range(p + 1, len(occupied_alpha)):
            for r in range(len(virtual_alpha) - 1):
                for s in range(r + 1, len(virtual_alpha)):
                    doubles_alpha.append([
                        occupied_alpha[p], occupied_alpha[q], virtual_alpha[r],
                        virtual_alpha[s]
                    ])
    doubles_beta = []
    for p in range(len(occupied_beta) - 1):
        for q in range(p + 1, len(occupied_beta)):
            for r in range(len(virtual_beta) - 1):
                for s in range(r + 1, len(virtual_beta)):
                    doubles_beta.append([
                        occupied_beta[p], occupied_beta[q], virtual_beta[r],
                        virtual_beta[s]
                    ])

    return (singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
            doubles_beta)


def get_num_uccsd_parameters(num_qubits, num_electrons, spin=0):
    """Number of UCCSD ansatz parameters (one per excitation)."""
    return sum(
        len(group)
        for group in get_uccsd_excitations(num_qubits, num_electrons, spin))


def _uccsd_single(p, q):
    """0.5 * (Y_p Z... X_q - X_p Z... Y_q) with the parity string in (p, q)."""
    parity = _z_parity(p, q)
    return 0.5 * _times(spin.y(p), parity, spin.x(q)) - 0.5 * _times(
        spin.x(p), parity, spin.y(q))


def _uccsd_double(p, q, r, s):
    """The 8-term UCCSD double-excitation generator (C++ index rules)."""
    if p < q and r < s:
        i_occ, j_occ, a_virt, b_virt = p, q, r, s
    elif p > q and r > s:
        i_occ, j_occ, a_virt, b_virt = q, p, s, r
    elif p < q and r > s:
        i_occ, j_occ, a_virt, b_virt = p, q, s, r
    else:
        i_occ, j_occ, a_virt, b_virt = q, p, r, s

    parity_a = _z_parity(i_occ, j_occ)
    parity_b = _z_parity(a_virt, b_virt)

    def term(op_i, op_j, op_a, op_b):
        return _times(op_i(i_occ), parity_a, op_j(j_occ), op_a(a_virt),
                      parity_b, op_b(b_virt))

    op = term(spin.x, spin.x, spin.x, spin.y)
    op += term(spin.x, spin.x, spin.y, spin.x)
    op += term(spin.x, spin.y, spin.y, spin.y)
    op += term(spin.y, spin.x, spin.y, spin.y)
    op -= term(spin.x, spin.y, spin.x, spin.x)
    op -= term(spin.y, spin.x, spin.x, spin.x)
    op -= term(spin.y, spin.y, spin.x, spin.y)
    op -= term(spin.y, spin.y, spin.y, spin.x)
    return 0.125 * op


def make_uccsd_operator_pool(num_qubits, num_electrons, spin=0):
    """One spin operator per UCCSD excitation, in excitation order."""
    (singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
     doubles_beta) = get_uccsd_excitations(num_qubits, num_electrons, spin)
    ops = []
    for p, q in singles_alpha:
        ops.append(_uccsd_single(p, q))
    for p, q in singles_beta:
        ops.append(_uccsd_single(p, q))
    for p, q, r, s in doubles_mixed:
        ops.append(_uccsd_double(p, q, r, s))
    for p, q, r, s in doubles_alpha:
        ops.append(_uccsd_double(p, q, r, s))
    for p, q, r, s in doubles_beta:
        ops.append(_uccsd_double(p, q, r, s))
    return ops


# ============================================================================
# UCCGSD (generalized) pool
# ============================================================================


def _generate_uccgsd_singles(num_qubits):
    return [(p, q) for p in range(1, num_qubits) for q in range(p)]


def _generate_uccgsd_doubles(num_qubits):
    doubles = set()
    for a in range(num_qubits):
        for b in range(a + 1, num_qubits):
            for c in range(b + 1, num_qubits):
                for d in range(c + 1, num_qubits):
                    for one, two in (((a, b), (c, d)), ((a, c), (b, d)),
                                     ((a, d), (b, c))):
                        one = (max(one), min(one))
                        two = (max(two), min(two))
                        doubles.add((min(one, two), max(one, two)))
    return sorted(doubles)


def _uccgsd_single(p, q):
    """0.5 * (Y_q Z... X_p - X_q Z... Y_p) for p > q."""
    parity = _z_parity(q, p)
    return 0.5 * _times(spin.y(q), parity, spin.x(p)) - 0.5 * _times(
        spin.x(q), parity, spin.y(p))


def _uccgsd_double(p, q, r, s):
    """The 8-term generalized double-excitation generator for p > q, r > s."""
    parity_a = _z_parity(q, p)
    parity_b = _z_parity(s, r)

    def term(op_s, op_r, op_q, op_p):
        return _times(op_s(s), parity_b, op_r(r), op_q(q), parity_a, op_p(p))

    op = term(spin.y, spin.x, spin.x, spin.x)
    op += term(spin.x, spin.y, spin.x, spin.x)
    op += term(spin.y, spin.y, spin.y, spin.x)
    op += term(spin.y, spin.y, spin.x, spin.y)
    op -= term(spin.x, spin.x, spin.y, spin.x)
    op -= term(spin.x, spin.x, spin.x, spin.y)
    op -= term(spin.x, spin.y, spin.y, spin.y)
    op -= term(spin.y, spin.x, spin.y, spin.y)
    return 0.125 * op


def make_uccgsd_operator_pool(num_qubits,
                              only_singles=False,
                              only_doubles=False):
    """Generalized singles and doubles over all qubit pairs/quadruples."""
    num_qubits = _as_count(num_qubits, "num_qubits")
    ops = []
    if not only_doubles:
        for p, q in _generate_uccgsd_singles(num_qubits):
            ops.append(_uccgsd_single(p, q))
    if not only_singles:
        for pq, rs in _generate_uccgsd_doubles(num_qubits):
            ops.append(_uccgsd_double(pq[0], pq[1], rs[0], rs[1]))
    return ops


# ============================================================================
# UpCCGSD (paired) pool
# ============================================================================


def make_upccgsd_operator_pool(num_qubits, only_doubles=False):
    """Spin-preserving singles plus paired (same-spatial-orbital) doubles."""
    num_qubits = _as_count(num_qubits, "num_qubits")
    if num_qubits % 2 != 0:
        raise ValueError("make_upccgsd_operator_pool expects an even number "
                         "of spin orbitals.")
    ops = []
    if not only_doubles:
        for p, q in _generate_uccgsd_singles(num_qubits):
            if p % 2 == q % 2:
                ops.append(_uccgsd_single(p, q))
    num_spatial = num_qubits // 2
    for p in range(num_spatial):
        for q in range(p + 1, num_spatial):
            ops.append(_uccgsd_double(2 * q + 1, 2 * q, 2 * p + 1, 2 * p))
    return ops


# ============================================================================
# CEO (coupled exchange operator) pool
# ============================================================================


def _generate_ceo_singles(num_orbitals, offset):
    """Same-spin singles (p, q), p > q; offset 0 for alpha, 1 for beta."""
    return [(2 * i + offset, 2 * j + offset) for i in range(num_orbitals)
            for j in range(i)]


def _generate_ceo_same_spin_doubles(num_orbitals, offset):
    """The three CEO pairings of each descending orbital 4-combination."""
    doubles = []
    for i in range(num_orbitals):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    p, q, r, s = (2 * i + offset, 2 * j + offset,
                                  2 * k + offset, 2 * l + offset)
                    doubles.append((p, q, r, s))
                    doubles.append((p, r, q, s))
                    doubles.append((q, p, r, s))
    return doubles


def _generate_ceo_mixed_doubles(num_orbitals):
    """(alpha, beta) -> (alpha, beta) pair excitations with p > r, q > s."""
    return [(2 * i, 2 * j + 1, 2 * k, 2 * l + 1) for i in range(num_orbitals)
            for j in range(num_orbitals) for k in range(i) for l in range(j)]


def _ceo_single(p, q):
    """0.5 * (Y_q X_p - X_q Y_p), no parity string."""
    return 0.5 * spin.y(q) * spin.x(p) - 0.5 * spin.x(q) * spin.y(p)


def _ceo_double_pair(p, q, r, s):
    """The two 4-term CEO double-excitation operators for (p, q, r, s)."""
    op_a = spin.x(r) * spin.x(p) * spin.x(s) * spin.y(q)
    op_a -= spin.x(r) * spin.x(p) * spin.y(s) * spin.x(q)
    op_a += spin.y(r) * spin.y(p) * spin.x(s) * spin.y(q)
    op_a -= spin.y(r) * spin.y(p) * spin.y(s) * spin.x(q)

    op_b = spin.x(r) * spin.y(p) * spin.x(s) * spin.x(q)
    op_b += spin.x(r) * spin.y(p) * spin.y(s) * spin.y(q)
    op_b -= spin.y(r) * spin.x(p) * spin.x(s) * spin.x(q)
    op_b -= spin.y(r) * spin.x(p) * spin.y(s) * spin.y(q)
    return 0.25 * op_a, 0.25 * op_b


def make_ceo_operator_pool(num_orbitals):
    """Coupled-exchange-operator pool (arXiv:2407.08696 conventions)."""
    num_orbitals = _as_count(num_orbitals, "num_orbitals")
    ops = []
    for p, q in _generate_ceo_singles(num_orbitals, 0):
        ops.append(_ceo_single(p, q))
    for p, q in _generate_ceo_singles(num_orbitals, 1):
        ops.append(_ceo_single(p, q))
    for p, q, r, s in _generate_ceo_same_spin_doubles(num_orbitals, 0):
        ops.extend(_ceo_double_pair(p, q, r, s))
    for p, q, r, s in _generate_ceo_same_spin_doubles(num_orbitals, 1):
        ops.extend(_ceo_double_pair(p, q, r, s))
    for p, q, r, s in _generate_ceo_mixed_doubles(num_orbitals):
        ops.extend(_ceo_double_pair(p, q, r, s))
    return ops


# ============================================================================
# Pauli word/coefficient lists (kernel-ready form of a pool)
# ============================================================================


def _pauli_lists_from_pool(ops, num_qubits):
    words_list = []
    coefficients_list = []
    for op in ops:
        words = []
        coefficients = []
        for term in cudaq.SpinOperator(op):
            words.append(cudaq.pauli_word(str(
                term.get_pauli_word(num_qubits))))
            coefficients.append(float(term.evaluate_coefficient().real))
        words_list.append(words)
        coefficients_list.append(coefficients)
    return words_list, coefficients_list


def get_uccgsd_pauli_lists(num_qubits, only_singles=False, only_doubles=False):
    """UCCGSD pool as (pauli word groups, coefficient groups)."""
    ops = make_uccgsd_operator_pool(num_qubits, only_singles, only_doubles)
    return _pauli_lists_from_pool(ops, num_qubits)


def get_upccgsd_pauli_lists(num_qubits, only_doubles=False):
    """UpCCGSD pool as (pauli word groups, coefficient groups)."""
    ops = make_upccgsd_operator_pool(num_qubits, only_doubles)
    return _pauli_lists_from_pool(ops, num_qubits)


def get_ceo_pauli_lists(num_orbitals):
    """CEO pool as (pauli word groups, coefficient groups)."""
    ops = make_ceo_operator_pool(num_orbitals)
    return _pauli_lists_from_pool(ops,
                                  2 * _as_count(num_orbitals, "num_orbitals"))
