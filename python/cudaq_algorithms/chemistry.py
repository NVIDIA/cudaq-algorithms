# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Chemistry-input bridges between classical tensors and qubit Hamiltonians.

Connects the double-factorization preprocessing to the quantum primitives:
chemist-notation spatial integrals (the form the DF module consumes and
reconstructs) are spin-expanded and passed through the ``fermion``
subpackage's Jordan-Wigner transform, yielding a ``cudaq.SpinOperator``
ready for ``PauliLCU``/``Walk``/``QSVT``.

``spin_orbital_tensors`` is pure NumPy and always importable and usable.
``qubit_hamiltonian`` additionally uses ``fermion.jordan_wigner`` (imported
lazily); importing this module never raises, and if ``fermion`` is
unavailable the ``ImportError`` surfaces when ``qubit_hamiltonian`` is
called, not at import.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["from_fcidump", "spin_orbital_tensors", "qubit_hamiltonian"]


def spin_orbital_tensors(
        one_body: ArrayLike,
        eri: ArrayLike,
        *,
        validate_symmetry: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Spin-expand chemist-notation spatial integrals.

    ``one_body`` is the ``(n, n)`` core Hamiltonian and ``eri`` the
    ``(n, n, n, n)`` chemist-notation ``(pq|rs)`` two-electron tensor over
    real spatial orbitals — the exact convention the double-factorization
    module documents. Returns ``(one_body_so, two_body_so)`` over ``2n``
    spin orbitals (interleaved spins: ``2p`` up, ``2p + 1`` down), where
    ``two_body_so[p, q, r, s]`` is the coefficient of
    ``a^dag_p a^dag_q a_r a_s`` as consumed by ``fermion.jordan_wigner``.

    ``eri`` must obey the real-orbital chemist permutation symmetry
    (``(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)`` and their compositions) —
    this is what makes the resulting qubit Hamiltonian Hermitian, and it
    holds for reconstructed DF integrals and mean-field integrals. It is
    checked by default; pass ``validate_symmetry=False`` to skip the check
    (e.g. for genuinely complex integrals, whose symmetry differs).
    """
    one_body = np.asarray(one_body, dtype=np.complex128)
    eri = np.asarray(eri, dtype=np.complex128)
    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be a square (n, n) matrix")
    n = one_body.shape[0]
    if eri.shape != (n, n, n, n):
        raise ValueError(
            "eri must be an (n, n, n, n) chemist-notation tensor matching "
            "one_body")
    if validate_symmetry:
        # The three generators of the real-orbital 8-fold symmetry; an
        # asymmetric eri would yield a non-Hermitian operator that the
        # primitives would silently consume as if Hermitian.
        for axes in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
            if not np.allclose(eri, eri.transpose(axes), atol=1e-8):
                raise ValueError(
                    "eri must obey the chemist-notation permutation symmetry "
                    "(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq); pass "
                    "validate_symmetry=False to skip this check")

    # Chemist (pq|rs) -> coefficients of adag_p adag_q a_r a_s.
    reordered = np.ascontiguousarray(eri.transpose(0, 2, 3, 1))

    m = 2 * n
    one_body_so = np.zeros((m, m), dtype=np.complex128)
    two_body_so = np.zeros((m, m, m, m), dtype=np.complex128)
    for p in range(n):
        for q in range(n):
            one_body_so[2 * p, 2 * q] = one_body[p, q]
            one_body_so[2 * p + 1, 2 * q + 1] = one_body[p, q]
            for r in range(n):
                for s in range(n):
                    coefficient = 0.5 * reordered[p, q, r, s]
                    two_body_so[2 * p, 2 * q, 2 * r, 2 * s] = coefficient
                    two_body_so[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                2 * s + 1] = coefficient
                    two_body_so[2 * p, 2 * q + 1, 2 * r + 1,
                                2 * s] = coefficient
                    two_body_so[2 * p + 1, 2 * q, 2 * r,
                                2 * s + 1] = coefficient
    return one_body_so, two_body_so


def qubit_hamiltonian(one_body: ArrayLike,
                      eri: ArrayLike,
                      *,
                      scalar_offset: float = 0.0,
                      tolerance: float = 1e-12,
                      validate_symmetry: bool = True):
    """Qubit Hamiltonian (``cudaq.SpinOperator``) from chemist integrals.

    Spin-expands the spatial integrals (see ``spin_orbital_tensors``, whose
    ``eri`` symmetry precondition and ``validate_symmetry`` flag apply here
    too) and applies the Jordan-Wigner transform. ``scalar_offset`` is
    added as an identity term (e.g. the nuclear repulsion energy);
    ``tolerance`` prunes negligible terms inside the transform.

    Combined with the double-factorization module this closes the
    classical-to-quantum loop::

        factorization = compressed_double_factorization(eri, num_leaves=T)
        h_truncated = qubit_hamiltonian(one_body,
                                        reconstruct_eri(factorization))
        encoding = PauliLCU(h_truncated)   # -> Walk / QSVT
    """
    from . import fermion  # imported lazily; ImportError here if unavailable

    one_body_so, two_body_so = spin_orbital_tensors(
        one_body, eri, validate_symmetry=validate_symmetry)
    return fermion.jordan_wigner(one_body_so,
                                 two_body_so,
                                 scalar_offset=float(scalar_offset),
                                 tolerance=float(tolerance))


# The eight index tuples of the real-orbital chemist symmetry orbit of
# (pq|rs): swap p<->q, r<->s, and (pq)<->(rs). A set dedups the diagonal
# records (e.g. (pp|pp) collapses to one tuple).
def _eri_symmetry_orbit(p, q, r, s):
    return {(p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
            (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)}


def from_fcidump(contents: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse FCIDUMP *contents* into chemist-notation spatial integrals.

    Takes the integral data as a string -- the text of the file, already
    read -- not a path; the caller does the file I/O::

        one_body, eri, core = from_fcidump(Path("mol.fcidump").read_text())

    Keeping the parse pure (no file access) lets callers and tests inject
    the integrals directly. Returns ``(one_body, eri, core_energy)``: the
    ``(n, n)`` core Hamiltonian, the dense ``(n, n, n, n)`` chemist-notation
    ``(pq|rs)`` two-electron tensor (all eight symmetry partners of each
    stored record populated), and the scalar core/constant energy -- the
    same triple ``from_pyscf``/``from_psi4`` return and the exact convention
    ``qubit_hamiltonian`` and ``DoubleFactorizedEncoding`` consume::

        one_body, eri, core = from_fcidump(text)
        hamiltonian = qubit_hamiltonian(one_body, eri, scalar_offset=core)

    FCIDUMP indices are Fortran 1-based; a ``value i j k l`` record with all
    of ``i,j,k,l`` nonzero is a two-electron integral, with ``k == l == 0``
    a one-electron integral ``h_ij`` (its transpose is filled too), and with
    all indices zero the core energy.

    Only real (RHF/ROHF-style) FCIDUMP files are supported; complex/UHF
    variants (``IUHF=1``) have a different index symmetry and are rejected
    downstream by the ``validate_symmetry`` check in ``qubit_hamiltonian``.
    """
    header_lines: list[str] = []
    body_lines: list[str] = []
    state = "pre"
    for raw in contents.splitlines():
        line = raw.strip()
        if state == "body":
            if line and not line.startswith(("#", "!")):
                body_lines.append(line)
            continue
        if not line:
            continue
        if state == "pre":
            lowered = line.lower()
            if not (lowered.startswith("&fci") or lowered.startswith("$fci")):
                raise ValueError(
                    "FCIDUMP contents must begin with an &FCI header namelist")
            state = "header"
            line = line[4:].strip()
            if not line:
                continue
        # state == "header"
        ended = bool(re.search(r"&end|\$end", line, re.IGNORECASE))
        line = re.sub(r"&end|\$end", " ", line, flags=re.IGNORECASE)
        if "/" in line:
            line = line.split("/", 1)[0]
            ended = True
        header_lines.append(line)
        if ended:
            state = "body"
    if state == "pre":
        raise ValueError(
            "FCIDUMP contents must begin with an &FCI header namelist")

    header = " ".join(header_lines)
    match = re.search(r"NORB\s*=\s*(\d+)", header, re.IGNORECASE)
    if match is None:
        raise ValueError("FCIDUMP header must specify NORB")
    n = int(match.group(1))
    if n <= 0:
        raise ValueError("FCIDUMP NORB must be a positive integer")

    one_body = np.zeros((n, n))
    eri = np.zeros((n, n, n, n))
    core_energy = 0.0
    for line in body_lines:
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(
                "FCIDUMP integral line must have 5 fields (value i j k l), "
                f"got: {line!r}")
        try:
            # Some writers emit a Fortran 'D' exponent (1.0D-3).
            value = float(tokens[0].replace("D", "E").replace("d", "e"))
            i, j, k, l = (int(token) for token in tokens[1:])
        except ValueError as error:
            raise ValueError(
                f"malformed FCIDUMP integral line: {line!r}") from error
        if max(i, j, k, l) > n or min(i, j, k, l) < 0:
            raise ValueError(
                f"FCIDUMP orbital index out of range [0, {n}]: {line!r}")
        if i and j and k and l:
            for a, b, c, d in _eri_symmetry_orbit(i - 1, j - 1, k - 1, l - 1):
                eri[a, b, c, d] = value
        elif i and j and not k and not l:
            one_body[i - 1, j - 1] = value
            one_body[j - 1, i - 1] = value
        elif not (i or j or k or l):
            core_energy = value
        else:
            raise ValueError(
                f"unexpected FCIDUMP index pattern (a one-electron record "
                f"must have k = l = 0): {line!r}")
    return (np.ascontiguousarray(one_body), np.ascontiguousarray(eri),
            float(core_energy))
