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

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "spin_orbital_tensors", "qubit_hamiltonian", "from_pyscf", "from_psi4"
]


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


def from_pyscf(mean_field) -> tuple[np.ndarray, np.ndarray, float]:
    """Chemist ``(pq|rs)`` MO integrals + nuclear repulsion from PySCF.

    ``mean_field`` is a converged restricted mean field (e.g. the result
    of ``pyscf.scf.RHF(mol).run()``). Returns
    ``(one_body, eri, nuclear_repulsion)`` in the molecular-orbital basis
    and chemist notation -- exactly the arguments
    :func:`qubit_hamiltonian` expects (pass ``nuclear_repulsion`` as its
    ``scalar_offset``).

    Restricted (single ``mo_coeff`` matrix) references only; the spin
    expansion downstream assumes one spatial set shared by both spins.
    """
    from functools import reduce

    from pyscf import ao2mo

    mol = mean_field.mol
    coefficients = np.asarray(mean_field.mo_coeff)
    num_orbitals = coefficients.shape[1]

    core_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    one_body = reduce(np.dot, (coefficients.T, core_ao, coefficients))

    # ao2mo.full returns chemist-notation MO integrals; restore(1, ...)
    # unpacks the 8-fold-symmetric storage to the dense (n, n, n, n) tensor.
    eri = ao2mo.restore(1, ao2mo.full(mol, coefficients), num_orbitals)

    return (np.ascontiguousarray(one_body), np.ascontiguousarray(eri),
            float(mean_field.energy_nuc()))


def from_psi4(wavefunction) -> tuple[np.ndarray, np.ndarray, float]:
    """Chemist ``(pq|rs)`` MO integrals + nuclear repulsion from Psi4.

    ``wavefunction`` is a converged restricted wavefunction, e.g. the
    second return value of ``psi4.energy("scf", return_wfn=True)``. Returns
    ``(one_body, eri, nuclear_repulsion)`` in the molecular-orbital basis
    and chemist notation -- identical in meaning and convention to
    :func:`from_pyscf`, so either drives :func:`qubit_hamiltonian`
    unchanged.

    Restricted references only (uses ``Ca``); ``mo_eri`` already returns
    the chemist ``(pq|rs)`` ordering, matching the PySCF path.
    """
    import psi4

    coefficients_matrix = wavefunction.Ca()
    coefficients = np.asarray(coefficients_matrix)

    # Core Hamiltonian (kinetic + potential) in the AO basis -> MO basis.
    core_ao = np.asarray(wavefunction.H())
    one_body = coefficients.T @ core_ao @ coefficients

    mints = psi4.core.MintsHelper(wavefunction.basisset())
    eri = np.asarray(
        mints.mo_eri(coefficients_matrix, coefficients_matrix,
                     coefficients_matrix, coefficients_matrix))

    nuclear_repulsion = wavefunction.molecule().nuclear_repulsion_energy()

    return (np.ascontiguousarray(one_body), np.ascontiguousarray(eri),
            float(nuclear_repulsion))
