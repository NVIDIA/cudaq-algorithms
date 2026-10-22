# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
tensors exactly as given, term by term. The integrals may be dense arrays
or sparse ``{index tuple: coefficient}`` mappings (``_normalize_integrals``
reduces both to the same sorted records of nonzero entries), so the two
forms compile to the same operator.

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

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
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


@dataclass(frozen=True, eq=False)
class _FermionTerms:
    """Fermionic integrals normalized to their nonzero entries, sorted by
    index.

    ``one_body_indices`` is ``(m1, 2)`` and ``two_body_indices`` ``(m2, 4)``
    (``int64``, lexicographically sorted) with the matching coefficients in
    ``*_values`` (``complex128``); iterate them as ``((i, j), h_ij)`` and
    ``((i, j, k, l), V_ijkl)`` through :attr:`one_body` / :attr:`two_body`.
    Exact zeros and entries below the tolerance are already removed, so a
    dense tensor and a sparse mapping with the same content compile
    identically (same terms, same order). ``num_modes`` is the declared
    mode count -- the dense dimension, or ``num_modes=`` for mapping input.
    It is part of the operator's meaning rather than bookkeeping: it sets
    the qubit width of the linear encodings and the shape of the
    Bravyi-Kitaev tree, so the same term can compile to different Pauli
    words at different counts. Declared modes that carry no term are
    allowed (an all-zero dense row/column always was)."""

    num_modes: int
    one_body_indices: np.ndarray
    one_body_values: np.ndarray
    two_body_indices: np.ndarray
    two_body_values: np.ndarray

    @property
    def one_body(self):
        return zip(map(tuple, self.one_body_indices.tolist()),
                   self.one_body_values.tolist())

    @property
    def two_body(self):
        return zip(map(tuple, self.two_body_indices.tolist()),
                   self.two_body_values.tolist())


def _negligible(coefficient: complex, tolerance: float) -> bool:
    """The one drop rule for every transform: exact zeros and magnitudes
    below ``tolerance`` (an explicit zero behaves like a missing entry)."""
    return coefficient == 0 or abs(coefficient) < tolerance


def _mapping_rank(mapping: Mapping, name: str) -> int:
    """Common key length of a sparse mapping (0 when it is empty)."""
    ranks = set()
    for key in mapping:
        if not isinstance(key, tuple):
            raise ValueError(f"{name} keys must be index tuples, got {key!r}")
        ranks.add(len(key))
    if len(ranks) > 1:
        raise ValueError(
            f"{name} mixes keys of different ranks {sorted(ranks)}; one-body "
            "entries have two indices and two-body entries four")
    return ranks.pop() if ranks else 0


def _empty_records(rank: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, rank), dtype=np.int64), np.zeros(0,
                                                         dtype=np.complex128)


def _records_from_mapping(mapping: Mapping, rank: int, num_modes: int,
                          name: str,
                          tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    records = []
    for key, value in mapping.items():
        indices = []
        for index in key:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError(
                    f"{name} key {key!r} has a non-integer index {index!r}")
            if not 0 <= index < num_modes:
                raise ValueError(f"{name} key {key!r} is out of range for "
                                 f"num_modes={num_modes}")
            indices.append(int(index))
        try:
            coefficient = complex(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} entry {key!r} has a non-numeric "
                             f"coefficient {value!r}") from None
        if not np.isfinite(coefficient):
            raise ValueError(
                f"{name} entry {key!r} has a non-finite coefficient {value!r}")
        if not _negligible(coefficient, tolerance):
            records.append((tuple(indices), coefficient))
    records.sort(key=lambda record: record[0])
    if not records:
        return _empty_records(rank)
    return (np.array([record[0] for record in records], dtype=np.int64),
            np.array([record[1] for record in records], dtype=np.complex128))


def _records_from_dense(array: np.ndarray, name: str,
                        tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite entries")
    keep = (array != 0) & (np.abs(array) >= tolerance)
    # argwhere and boolean indexing both walk the array in C order, so the
    # records come out lexicographically sorted and aligned.
    return np.argwhere(keep).astype(np.int64), array[keep].astype(
        np.complex128)


def _normalize_integrals(first,
                         second=None,
                         *,
                         num_modes: Optional[int] = None,
                         tolerance: float = 1e-15) -> _FermionTerms:
    """Reduce the public integral arguments to :class:`_FermionTerms`.

    ``first`` is the one-body integrals -- an ``(n, n)`` array or a mapping
    with two-index keys -- optionally followed by ``second``, the two-body
    integrals as an ``(n, n, n, n)`` array or a mapping with four-index
    keys; or ``first`` alone is the two-body integrals (rank-4 array or
    four-index mapping) and ``second`` must be omitted. Dense and sparse
    forms may be mixed. The mode count is the dense dimension when any
    array is given (``num_modes`` may repeat it but not contradict it) and
    must be passed as ``num_modes`` when both inputs are mappings. Missing
    mapping entries are zero; explicit zeros and entries below
    ``tolerance`` are dropped in both forms. Nothing here allocates a dense
    tensor for a mapping or for an omitted input."""
    if num_modes is not None:
        if (isinstance(num_modes, bool) or not isinstance(num_modes, Integral)
                or num_modes < 1):
            raise ValueError(
                f"num_modes must be a positive integer, got {num_modes!r}")
        num_modes = int(num_modes)

    one_dense = two_dense = None
    one_sparse = two_sparse = None
    n = None

    if isinstance(first, Mapping):
        rank = _mapping_rank(first, "the first integral argument")
        if rank == 4:
            if second is not None:
                raise ValueError(
                    "second tensor is invalid when first is rank 4 (two-body)")
            two_sparse = first
        elif rank in (0, 2):
            one_sparse = first
        else:
            raise ValueError("expected rank-2 one_body or rank-4 two_body "
                             f"keys, got {rank}-index keys")
    else:
        array = np.asarray(first, dtype=np.complex128)
        if array.ndim == 2:
            if array.shape[0] != array.shape[1]:
                raise ValueError("one_body dimensions must match")
            one_dense, n = array, array.shape[0]
        elif array.ndim == 4:
            if second is not None:
                raise ValueError(
                    "second tensor is invalid when first is rank 4")
            if len(set(array.shape)) != 1:
                raise ValueError("two_body dimensions must match")
            two_dense, n = array, array.shape[0]
        else:
            raise ValueError("expected rank-2 one_body or rank-4 two_body, "
                             f"got rank {array.ndim}")

    if second is not None:
        if isinstance(second, Mapping):
            rank = _mapping_rank(second, "two_body")
            if rank not in (0, 4):
                raise ValueError("two_body has the wrong rank (keys have "
                                 f"{rank} indices, expected 4)")
            two_sparse = second
        else:
            array = np.asarray(second, dtype=np.complex128)
            if array.ndim != 4:
                raise ValueError("two_body has the wrong rank (expected 4, "
                                 f"got {array.ndim})")
            if len(set(array.shape)) != 1:
                raise ValueError("two_body dimensions must match")
            if n is not None and array.shape[0] != n:
                raise ValueError("one_body and two_body dimensions differ "
                                 f"({n} vs {array.shape[0]})")
            two_dense, n = array, array.shape[0]

    if n is None:
        if num_modes is None:
            raise ValueError(
                "num_modes is required when the integrals are given as "
                "mappings: no dense tensor fixes the number of modes, and the "
                "count sets the qubit width of the linear encodings and the "
                "Bravyi-Kitaev tree")
        n = num_modes
    elif num_modes is not None and num_modes != n:
        raise ValueError(f"num_modes={num_modes} disagrees with the dense "
                         f"tensor dimension {n}")

    if one_dense is not None:
        one_body = _records_from_dense(one_dense, "one_body", tolerance)
    elif one_sparse is not None:
        one_body = _records_from_mapping(one_sparse, 2, n, "one_body",
                                         tolerance)
    else:
        one_body = _empty_records(2)
    if two_dense is not None:
        two_body = _records_from_dense(two_dense, "two_body", tolerance)
    elif two_sparse is not None:
        two_body = _records_from_mapping(two_sparse, 4, n, "two_body",
                                         tolerance)
    else:
        two_body = _empty_records(4)
    return _FermionTerms(n, *one_body, *two_body)


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


def _compile_hamiltonian(encoding: _Encoding, terms: _FermionTerms,
                         scalar_offset: float, tolerance: float):
    # Ladder words are built on demand: with sparse input only the modes that
    # carry a term are touched, however many are declared.
    ladders: dict = {}

    def ladder(mode: int, dagger: bool) -> list:
        key = (mode, dagger)
        if key not in ladders:
            ladders[key] = encoding.ladder_terms(mode, dagger)
        return ladders[key]

    accumulator: dict = {(0, 0): complex(scalar_offset)}

    for (i, j), coefficient in terms.one_body:
        _accumulate(accumulator, coefficient,
                    [ladder(i, True), ladder(j, False)])
    for (i, j, k, l), coefficient in terms.two_body:
        _accumulate(accumulator, coefficient, [
            ladder(i, True),
            ladder(j, True),
            ladder(k, False),
            ladder(l, False)
        ])

    return _to_spin_operator(accumulator, tolerance)


# ----------------------------------------------------------------------
# Public transforms
# ----------------------------------------------------------------------


def jordan_wigner(one_body_or_two_body: ArrayLike | Mapping,
                  two_body: Optional[ArrayLike | Mapping] = None,
                  scalar_offset: float = 0.0,
                  tolerance: float = 1e-15,
                  *,
                  num_modes: Optional[int] = None):
    """Jordan-Wigner transform of fermionic integrals to a qubit operator.

    The integrals are the coefficients of ``adag_i a_j`` (one-body) and
    ``adag_i adag_j a_k a_l`` (two-body) over ``n`` spin orbitals,

        H = scalar_offset * I + sum_ij   h[i, j]       adag_i a_j
                              + sum_ijkl V[i, j, k, l] adag_i adag_j a_k a_l

    compiled exactly as given: index order is preserved and no conjugate
    terms, tensor symmetries or prefactors are inserted (Hermiticity is the
    caller's responsibility). Each tensor may be **dense** -- an ``(n, n)``
    one-body array, optionally followed by an ``(n, n, n, n)`` two-body
    array, or a two-body array alone -- or **sparse**, a mapping from index
    tuples to coefficients whose missing entries are zero::

        h = {(0, 1): -1.0, (1, 0): -1.0}     # hopping 0 <-> 1
        V = {(0, 1, 1, 0): 0.5}               # 0.5 * n_0 n_1
        op = jordan_wigner(h, V, num_modes=4)

    Dense and sparse forms may be mixed. The mode count is read from any
    dense tensor; when both inputs are mappings it must be given as
    ``num_modes``, because it is part of the operator's definition rather
    than something to infer from the largest index (it sets the qubit width,
    and for Bravyi-Kitaev the shape of the Fenwick tree). Declared modes
    that carry no term are allowed, as an all-zero dense row/column always
    was. ``scalar_offset`` is added as an identity term; input entries
    (explicit zeros included) and compiled terms with magnitude below
    ``tolerance`` are dropped. Returns a ``cudaq.SpinOperator``.
    """
    terms = _normalize_integrals(one_body_or_two_body,
                                 two_body,
                                 num_modes=num_modes,
                                 tolerance=tolerance)
    return _compile_hamiltonian(_Encoding(_identity_matrix(terms.num_modes)),
                                terms, scalar_offset, tolerance)


def bravyi_kitaev(one_body_or_two_body: ArrayLike | Mapping,
                  two_body: Optional[ArrayLike | Mapping] = None,
                  scalar_offset: float = 0.0,
                  tolerance: float = 1e-15,
                  *,
                  num_modes: Optional[int] = None):
    """Bravyi-Kitaev transform of fermionic integrals to a qubit operator.

    Same input conventions as ``jordan_wigner`` (dense tensors or sparse
    mappings, ``num_modes`` when both inputs are mappings); the qubits store
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
    terms = _normalize_integrals(one_body_or_two_body,
                                 two_body,
                                 num_modes=num_modes,
                                 tolerance=tolerance)
    return _compile_hamiltonian(_Encoding(_fenwick_matrix(terms.num_modes)),
                                terms, scalar_offset, tolerance)
