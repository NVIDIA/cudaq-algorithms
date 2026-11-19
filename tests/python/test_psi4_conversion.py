# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chemistry-input extraction from PySCF and Psi4 agree.

``chemistry.qubit_hamiltonian`` is package-agnostic (it takes chemist
``(pq|rs)`` MO integrals as plain NumPy). These tests confirm the two
extraction helpers feed it correctly:

* PySCF-only: the qubit Hamiltonian built from ``from_pyscf`` reproduces
  the FCI ground-state energy (runs wherever PySCF is installed).
* PySCF vs Psi4: the same molecule/basis extracted from both packages
  yields qubit Hamiltonians with the *same spectrum*. Spectrum, not
  coefficient-by-coefficient equality, is the right invariant: the two
  codes may return canonical MOs that differ by orbital phase or (for
  degeneracies) ordering, a unitary gauge that leaves the spectrum --
  and every physical observable -- unchanged.

Both engines are optional; each test skips if its package is missing.
Psi4 ships via conda-forge (no PyPI wheel), so the cross-check runs in a
conda-enabled environment.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import cudaq_algorithms as algorithms

# (label, basis, geometry as (symbol, (x, y, z)) in angstrom)
_MOLECULES = [
    ("H2", "sto-3g", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))]),
    ("H4", "sto-3g", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474)),
                      ("H", (1.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.7474))]),
]


def _pyscf_mean_field(geometry, basis):
    from pyscf import gto, scf

    mol = gto.M(atom=[(symbol, xyz) for symbol, xyz in geometry],
                basis=basis,
                symmetry=False)
    return scf.RHF(mol).run(verbose=0)


def _psi4_wavefunction(geometry, basis):
    import psi4

    psi4.core.be_quiet()
    lines = [f"{symbol} {x} {y} {z}" for symbol, (x, y, z) in geometry]
    lines += ["units angstrom", "symmetry c1", "no_reorient", "no_com"]
    psi4.geometry("\n".join(lines))
    psi4.set_options({
        "basis": basis,
        "scf_type": "pk",
        "e_convergence": 1e-11,
        "d_convergence": 1e-11,
        "reference": "rhf",
    })
    _, wavefunction = psi4.energy("scf", return_wfn=True)
    return wavefunction


def _spectrum(one_body, eri, nuclear_repulsion):
    operator = algorithms.chemistry.qubit_hamiltonian(
        one_body, eri, scalar_offset=nuclear_repulsion, tolerance=1e-12)
    return np.linalg.eigvalsh(np.asarray(operator.to_matrix()))


@pytest.mark.parametrize("label, basis, geometry", _MOLECULES)
def test_from_pyscf_reproduces_fci(label, basis, geometry):
    pytest.importorskip("pyscf")
    from pyscf import fci

    mean_field = _pyscf_mean_field(geometry, basis)
    one_body, eri, nuclear_repulsion = algorithms.chemistry.from_pyscf(
        mean_field)
    ground = float(_spectrum(one_body, eri, nuclear_repulsion).min())
    reference = float(fci.FCI(mean_field).kernel()[0])
    assert abs(ground - reference) < 1e-8, (
        f"{label}: JW ground {ground} differs from FCI {reference}")


@pytest.mark.parametrize("label, basis, geometry", _MOLECULES)
def test_pyscf_and_psi4_qubit_hamiltonians_match(label, basis, geometry):
    pytest.importorskip("pyscf")
    pytest.importorskip("psi4")

    pyscf_tensors = algorithms.chemistry.from_pyscf(
        _pyscf_mean_field(geometry, basis))
    psi4_tensors = algorithms.chemistry.from_psi4(
        _psi4_wavefunction(geometry, basis))

    # Same geometry, but PySCF and Psi4 use marginally different Bohr-radius
    # constants for the angstrom->bohr conversion, so nuclear repulsion (and
    # the integrals) differ at the ~1e-9 level -- well inside the spectrum
    # tolerance below, which is the quantity that actually matters.
    assert abs(pyscf_tensors[2] - psi4_tensors[2]) < 1e-6

    pyscf_spectrum = _spectrum(*pyscf_tensors)
    psi4_spectrum = _spectrum(*psi4_tensors)
    assert np.allclose(pyscf_spectrum, psi4_spectrum, atol=1e-6), (
        f"{label}: PySCF and Psi4 qubit-Hamiltonian spectra differ by "
        f"{np.max(np.abs(pyscf_spectrum - psi4_spectrum)):.2e}")


def test_from_pyscf_rejects_unrestricted_mean_field(monkeypatch):
    pyscf_stub = ModuleType("pyscf")
    pyscf_stub.ao2mo = ModuleType("pyscf.ao2mo")
    monkeypatch.setitem(sys.modules, "pyscf", pyscf_stub)
    mean_field = SimpleNamespace(mol=None, mo_coeff=np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="mo_coeff with ndim 3"):
        algorithms.chemistry.from_pyscf(mean_field)


@pytest.mark.parametrize(
    "nirrep, same_a_b_orbs, message",
    [(2, True, "C1"), (1, False, "restricted")])
def test_from_psi4_rejects_unsupported_wavefunctions(
        monkeypatch, nirrep, same_a_b_orbs, message):
    monkeypatch.setitem(sys.modules, "psi4", ModuleType("psi4"))
    wavefunction = SimpleNamespace(
        nirrep=lambda: nirrep,
        same_a_b_orbs=lambda: same_a_b_orbs)

    with pytest.raises(ValueError, match=message):
        algorithms.chemistry.from_psi4(wavefunction)


def test_from_psi4_returns_mo_integrals_and_nuclear_repulsion(monkeypatch):
    coefficients_matrix = np.array([[0.8, -0.6], [0.6, 0.8]])
    core_ao = np.array([[1.0, 0.25], [0.25, 2.0]])
    eri_storage = np.arange(32.0).reshape(2, 2, 2, 2, 2)
    eri_reference = eri_storage[..., 0]
    basis_set = object()

    def mo_eri(*matrices):
        assert len(matrices) == 4
        assert all(matrix is coefficients_matrix for matrix in matrices)
        return eri_reference

    def mints_helper(received_basis):
        assert received_basis is basis_set
        return SimpleNamespace(mo_eri=mo_eri)

    psi4_stub = ModuleType("psi4")
    psi4_stub.core = SimpleNamespace(MintsHelper=mints_helper)
    monkeypatch.setitem(sys.modules, "psi4", psi4_stub)

    wavefunction = SimpleNamespace(
        nirrep=lambda: 1,
        same_a_b_orbs=lambda: True,
        Ca=lambda: coefficients_matrix,
        H=lambda: core_ao,
        basisset=lambda: basis_set,
        molecule=lambda: SimpleNamespace(
            nuclear_repulsion_energy=lambda: np.float64(0.715)))

    one_body, eri, nuclear_repulsion = algorithms.chemistry.from_psi4(
        wavefunction)

    np.testing.assert_allclose(one_body,
                               coefficients_matrix.T @ core_ao @
                               coefficients_matrix)
    np.testing.assert_array_equal(eri, eri_reference)
    assert one_body.flags.c_contiguous
    assert eri.flags.c_contiguous
    assert isinstance(nuclear_repulsion, float)
    assert nuclear_repulsion == pytest.approx(0.715)
