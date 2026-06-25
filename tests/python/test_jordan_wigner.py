import os
from functools import reduce

import cudaq_algorithms as algorithms
import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

pytest.importorskip("pyscf")
from pyscf import ao2mo, fci, gto, scf


def _pyscf_spin_orbital_integrals(xyz, basis="sto-3g"):
    mol = gto.M(atom=xyz, basis=basis, symmetry=False)
    mean_field = scf.RHF(mol).run(verbose=0)

    h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h1e = reduce(np.dot, (mean_field.mo_coeff.T, h1e_ao, mean_field.mo_coeff))

    h2e_ao = mol.intor("int2e_sph", aosym="1")
    h2e = ao2mo.incore.full(h2e_ao, mean_field.mo_coeff)

    # PySCF returns two-electron integrals in chemist notation. The fermion
    # transform consumes coefficients for adag_p adag_q a_r a_s.
    h2e = np.asarray(h2e.transpose(0, 2, 3, 1), order="C")

    num_spin_orbitals = 2 * h1e.shape[0]
    one_body = np.zeros((num_spin_orbitals, num_spin_orbitals),
                        dtype=np.complex128)
    two_body = np.zeros((num_spin_orbitals, num_spin_orbitals,
                         num_spin_orbitals, num_spin_orbitals),
                        dtype=np.complex128)

    for p in range(num_spin_orbitals // 2):
        for q in range(num_spin_orbitals // 2):
            one_body[2 * p, 2 * q] = h1e[p, q]
            one_body[2 * p + 1, 2 * q + 1] = h1e[p, q]

            for r in range(num_spin_orbitals // 2):
                for s in range(num_spin_orbitals // 2):
                    coefficient = 0.5 * h2e[p, q, r, s]
                    two_body[2 * p, 2 * q, 2 * r, 2 * s] = coefficient
                    two_body[2 * p + 1, 2 * q + 1, 2 * r + 1,
                             2 * s + 1] = coefficient
                    two_body[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = coefficient
                    two_body[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = coefficient

    return mol, mean_field, one_body, two_body, mean_field.energy_nuc()


def jw_molecule_test(xyz):
    mol, mean_field, one_body, two_body, nuclear_energy = \
        _pyscf_spin_orbital_integrals(xyz)

    fci_energy = fci.FCI(mean_field).kernel()[0]
    spin_op = algorithms.fermion.jordan_wigner(one_body,
                                               two_body,
                                               scalar_offset=nuclear_energy,
                                               tolerance=1e-12)
    jw_energy = np.min(np.linalg.eigvalsh(spin_op.to_matrix()))

    assert np.isclose(jw_energy, fci_energy, atol=1e-4), \
        f"{mol.atom}: JW energy {jw_energy} differs from FCI {fci_energy}"


def test_ground_state():
    jw_molecule_test([("H", (0., 0., 0.)), ("H", (0., 0., .7474))])
    jw_molecule_test([("H", (0., 0., 0.)), ("H", (0., 0., .7474)),
                      ("H", (1., 0., 0.)), ("H", (1., 0., .7474))])
    jw_molecule_test([("H", (0., 0., 0.)), ("H", (1.0, 0., 0.)),
                      ("H", (0.322, 2.592, 0.1)), ("H", (1.2825, 2.292, 0.1))])
    jw_molecule_test([("Li", (0., 0., 0.)), ("H", (0., 0., 1.1774))])
