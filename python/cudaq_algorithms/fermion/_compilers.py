# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Fermion-to-qubit compilers: Jordan-Wigner and Bravyi-Kitaev, pure Python.

Both transforms are instances of one construction. A fermion-to-qubit
encoding is a linear map over GF(2): an invertible binary matrix ``A``
stores the occupation vector ``n`` as qubit bits ``x = A n (mod 2)``.
Jordan-Wigner is ``A = I`` (each qubit stores one occupation);
Bravyi-Kitaev is the Fenwick (binary-indexed-tree) partial-sum matrix,
whose row ``i`` accumulates the occupations of the standard
binary-indexed-tree range ending at mode ``i``.

Everything a ladder operator needs follows from ``A`` and its GF(2)
inverse, as three qubit masks per mode ``j``:

- update mask ``U(j)``  — column ``j`` of ``A``: the qubits that flip
  when occupation ``j`` flips (the X part),
- parity mask ``P(j)``  — the XOR of rows ``0..j-1`` of ``A^-1``: the
  qubits whose parity equals the fermionic parity below ``j`` (the Z
  part carrying the anticommutation sign),
- flip mask ``F(j)``    — row ``j`` of ``A^-1``: the qubits whose
  parity equals occupation ``j`` (the number-operator Z word).

With the Majorana word ``M_j = X_{U(j)} Z_{P(j)}`` (parity read before
the flip), the ladder operators are

    adag_j = (M_j + M_j Z_{F(j)}) / 2
    a_j    = (M_j - M_j Z_{F(j)}) / 2

and a Hamiltonian is compiled by expanding each integral term as a
product of these two-term sums. Pauli words are held in the symplectic
representation — a pair of bitmasks ``(x, z)`` meaning
``i^{popcount(x & z)} X^x Z^z`` (``Y`` on overlap) — so products are two
XORs and a phase.

The Hamiltonian convention matches the C++ transforms this module
replaced:

    H = scalar_offset * I + sum_ij  h[i, j]      adag_i a_j
                          + sum_ijkl V[i, j, k, l] adag_i adag_j a_k a_l

Unlike the retired C++ Bravyi-Kitaev (which assumed additional
tensor structure beyond hermiticity), both transforms here compile the
tensors exactly as given, term by term.

Migration from the compiled extension (behavior differences, not bugs):

- Invalid shapes/ranks raise ``ValueError`` (the Pythonic choice), where
  the compiled binding surfaced C++ ``throw`` as ``RuntimeError``. Code
  with ``except RuntimeError`` around a transform must be updated.
- The returned operator's qubit width tracks the qubits actually touched.
  A Hamiltonian that never couples the highest orbital(s) yields a
  narrower operator, and a fully-pruned or all-zero Hamiltonian yields an
  empty operator; ``to_matrix()`` is then ``2^(touched)``, not ``2^n``.
  Callers needing a fixed ``2^n`` must ensure the top orbital is touched
  (or pad the result themselves).
- Bravyi-Kitaev no longer antisymmetrizes the two-body tensor internally
  (see ``bravyi_kitaev``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import cudaq

__all__ = ["jordan_wigner", "bravyi_kitaev"]

# ----------------------------------------------------------------------
# Symplectic Pauli-word algebra
# ----------------------------------------------------------------------
#
# A word is (x, z): integer bitmasks over the qubits. The operator it
# denotes is i^{popcount(x & z)} X^x Z^z — the canonical word with Y on
# every overlap bit (Y = i X Z). A term is (coefficient, (x, z)).

_I_POWERS = (1.0, 1.0j, -1.0, -1.0j)


def _word_product(x1: int, z1: int, x2: int,
                  z2: int) -> tuple[complex, int, int]:
    """Product of two canonical words: phase and the resulting word.

    From X^x Z^z algebra: commuting Z^{z1} past X^{x2} contributes
    (-1)^{popcount(z1 & x2)}; the i^{popcount(x & z)} canonicalization
    factors of the inputs and the output contribute the rest.
    """
    x = x1 ^ x2
    z = z1 ^ z2
    i_exponent = ((x1 & z1).bit_count() + (x2 & z2).bit_count() -
                  (x & z).bit_count()) % 4
    sign = -1.0 if (z1 & x2).bit_count() & 1 else 1.0
    return sign * _I_POWERS[i_exponent], x, z


def _terms_product(left: list, right: list) -> list:
    """Product of two term lists (sums of coefficient-weighted words)."""
    out = []
    for c1, (x1, z1) in left:
        for c2, (x2, z2) in right:
            phase, x, z = _word_product(x1, z1, x2, z2)
            out.append((c1 * c2 * phase, (x, z)))
    return out


# ----------------------------------------------------------------------
# GF(2) encodings
# ----------------------------------------------------------------------


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a binary matrix over GF(2) by Gauss-Jordan elimination."""
    m = matrix.shape[0]
    work = np.concatenate(
        [matrix.astype(np.uint8) & 1,
         np.eye(m, dtype=np.uint8)], axis=1)
    for col in range(m):
        pivot_rows = np.nonzero(work[col:, col])[0]
        if pivot_rows.size == 0:
            raise ValueError("encoding matrix is singular over GF(2)")
        pivot = col + int(pivot_rows[0])
        if pivot != col:
            work[[col, pivot]] = work[[pivot, col]]
        eliminate = np.nonzero(work[:, col])[0]
        for row in eliminate:
            if row != col:
                work[row] ^= work[col]
    return work[:, m:]


def _row_mask(row: np.ndarray) -> int:
    mask = 0
    for qubit in np.nonzero(row)[0]:
        mask |= 1 << int(qubit)
    return mask


class _Encoding:
    """Update/parity/flip masks derived from a GF(2) encoding matrix."""

    def __init__(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.uint8) & 1
        m = matrix.shape[0]
        inverse = _gf2_inverse(matrix)

        self.num_modes = m
        self.update_masks = [_row_mask(matrix[:, j]) for j in range(m)]
        self.flip_masks = [_row_mask(inverse[j]) for j in range(m)]
        self.parity_masks = []
        prefix = np.zeros(m, dtype=np.uint8)
        for j in range(m):
            self.parity_masks.append(_row_mask(prefix))
            prefix ^= inverse[j]

    def ladder_terms(self, mode: int, dagger: bool) -> list:
        """``adag_mode`` or ``a_mode`` as a two-term word sum."""
        majorana = _terms_product(
            [(1.0, (self.update_masks[mode], 0))],
            [(1.0, (0, self.parity_masks[mode]))],
        )
        flipped = _terms_product(majorana, [(1.0, (0, self.flip_masks[mode]))])
        sign = 1.0 if dagger else -1.0
        return [(0.5 * c, w) for c, w in majorana] + [(sign * 0.5 * c, w)
                                                      for c, w in flipped]


def _identity_matrix(num_modes: int) -> np.ndarray:
    return np.eye(num_modes, dtype=np.uint8)


def _fenwick_matrix(num_modes: int) -> np.ndarray:
    """The Bravyi-Kitaev (binary-indexed-tree) partial-sum matrix.

    In one-based indexing, node ``i`` of a binary indexed tree stores the
    sum over the range ``(i - lowbit(i), i]``; row ``i - 1`` of the
    encoding matrix marks exactly that range of modes.
    """
    matrix = np.zeros((num_modes, num_modes), dtype=np.uint8)
    for i in range(1, num_modes + 1):
        low = i - (i & -i)
        matrix[i - 1, low:i] = 1
    return matrix


# ----------------------------------------------------------------------
# Hamiltonian compilation
# ----------------------------------------------------------------------


def _validate_tensors(
        first: ArrayLike,
        second: Optional[ArrayLike]) -> tuple[np.ndarray, np.ndarray, int]:
    first = np.asarray(first, dtype=np.complex128)

    if first.ndim == 2:
        n = first.shape[0]
        if first.shape != (n, n):
            raise ValueError("one_body dimensions must match")
        one_body = first
        if second is None:
            two_body = np.zeros((0, 0, 0, 0), dtype=np.complex128)
        else:
            two_body = np.asarray(second, dtype=np.complex128)
            if two_body.ndim != 4:
                raise ValueError("two_body has the wrong rank")
            if two_body.shape != (n, n, n, n):
                raise ValueError("one_body and two_body dimensions differ")
        return one_body, two_body, n

    if first.ndim == 4:
        if second is not None:
            raise ValueError("second tensor is invalid when first is rank 4")
        n = first.shape[0]
        if first.shape != (n, n, n, n):
            raise ValueError("two_body dimensions must match")
        return np.zeros((n, n), dtype=np.complex128), first, n

    raise ValueError("expected rank-2 one_body or rank-4 two_body")


def _accumulate(accumulator: dict, coefficient: complex,
                factors: list) -> None:
    terms = factors[0]
    for factor in factors[1:]:
        terms = _terms_product(terms, factor)
    for c, word in terms:
        accumulator[word] = accumulator.get(word, 0.0) + coefficient * c


def _to_spin_operator(accumulator: dict, tolerance: float):
    operators = []
    for (x, z), coefficient in sorted(accumulator.items()):
        if abs(coefficient) < tolerance:
            continue
        term = cudaq.SpinOperator.empty() + 1.0  # identity term
        support = x | z
        qubit = 0
        while support >> qubit:
            bit = 1 << qubit
            if x & bit and z & bit:
                term = term * cudaq.spin.y(qubit)
            elif x & bit:
                term = term * cudaq.spin.x(qubit)
            elif z & bit:
                term = term * cudaq.spin.z(qubit)
            qubit += 1
        operators.append(coefficient * term)
    if not operators:
        return cudaq.SpinOperator.empty()
    # Pairwise (binary-tree) reduction: cudaq operator addition copies both
    # sides, so a left fold over T terms costs O(T^2) term copies while the
    # tree costs O(T log T) — the difference between seconds and hours for
    # multi-thousand-term Hamiltonians.
    while len(operators) > 1:
        paired = []
        for i in range(0, len(operators) - 1, 2):
            paired.append(operators[i] + operators[i + 1])
        if len(operators) & 1:
            paired.append(operators[-1])
        operators = paired
    return operators[0].canonicalize()


def _compile_hamiltonian(encoding: _Encoding, one_body: np.ndarray,
                         two_body: np.ndarray, scalar_offset: float,
                         tolerance: float):

    def negligible(value: complex) -> bool:
        # Magnitude cutoff, matching the docstrings and the output-side prune
        # in _to_spin_operator (a magnitude disk, not a componentwise square).
        return abs(value) < tolerance

    raise_terms = [
        encoding.ladder_terms(j, dagger=True)
        for j in range(encoding.num_modes)
    ]
    lower_terms = [
        encoding.ladder_terms(j, dagger=False)
        for j in range(encoding.num_modes)
    ]

    accumulator: dict = {(0, 0): complex(scalar_offset)}

    for i, j in np.argwhere(one_body):
        coefficient = complex(one_body[i, j])
        if not negligible(coefficient):
            _accumulate(accumulator, coefficient,
                        [raise_terms[i], lower_terms[j]])

    if two_body.size:
        for i, j, k, l in np.argwhere(two_body):
            coefficient = complex(two_body[i, j, k, l])
            if not negligible(coefficient):
                _accumulate(accumulator, coefficient, [
                    raise_terms[i], raise_terms[j], lower_terms[k],
                    lower_terms[l]
                ])

    return _to_spin_operator(accumulator, tolerance)


# ----------------------------------------------------------------------
# Public transforms
# ----------------------------------------------------------------------


def jordan_wigner(one_body_or_two_body: ArrayLike,
                  two_body: Optional[ArrayLike] = None,
                  scalar_offset: float = 0.0,
                  tolerance: float = 1e-15):
    """Jordan-Wigner transform of fermionic integrals to a qubit operator.

    Accepts an ``(n, n)`` one-body tensor, optionally with an
    ``(n, n, n, n)`` two-body tensor, or a two-body tensor alone; entries
    are the coefficients of ``adag_i a_j`` and ``adag_i adag_j a_k a_l``
    over ``n`` spin orbitals. ``scalar_offset`` is added as an identity
    term; input entries and compiled terms with magnitude below
    ``tolerance`` are dropped. Returns a ``cudaq.SpinOperator``.
    """
    one_body, two_body_arr, n = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    return _compile_hamiltonian(_Encoding(_identity_matrix(n)), one_body,
                                two_body_arr, scalar_offset, tolerance)


def bravyi_kitaev(one_body_or_two_body: ArrayLike,
                  two_body: Optional[ArrayLike] = None,
                  scalar_offset: float = 0.0,
                  tolerance: float = 1e-15):
    """Bravyi-Kitaev transform of fermionic integrals to a qubit operator.

    Same input conventions as ``jordan_wigner``; the qubits store
    Fenwick-tree partial sums of the occupations, giving O(log n)-weight
    Pauli words. Returns a ``cudaq.SpinOperator``.

    Migration note: the two-body tensor is compiled *literally*, entry by
    entry as ``V[i,j,k,l] adag_i adag_j a_k a_l`` — identical to
    ``jordan_wigner``. The retired compiled binding instead antisymmetrized
    the tensor internally before transforming; matching ``jordan_wigner``
    here repairs a prior JW/BK inconsistency. A caller who passed a raw,
    non-antisymmetrized (e.g. chemist-ordered) two-body tensor and relied on
    that internal antisymmetrization must now antisymmetrize the input
    themselves, or they will silently get a different operator.
    """
    one_body, two_body_arr, n = _validate_tensors(one_body_or_two_body,
                                                  two_body)
    return _compile_hamiltonian(_Encoding(_fenwick_matrix(n)), one_body,
                                two_body_arr, scalar_offset, tolerance)
