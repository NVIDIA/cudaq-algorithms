# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FCIDUMP reader -> chemist-notation tensors -> qubit Hamiltonian.

The integral data is injected as an in-line string (the contents a caller
would read from a `.fcidump` file), so these tests need no external files.
The parsed tensors are pinned against the same literature H2/STO-3G values
`test_df_qsvt_bridge.py` uses, and the resulting qubit Hamiltonian is
cross-checked by whole-spectrum equality against those integrals fed
directly -- the convention-level check `the preprocessing guide`
recommends.
"""

import numpy as np
import pytest

from cudaq_algorithms import chemistry

# The qubit-Hamiltonian path needs fermion.jordan_wigner (imported lazily);
# gate those tests on the real dependency, mirroring test_df_qsvt_bridge.py.
# The pure-NumPy parsing/validation tests stay ungated.
try:
    from cudaq_algorithms.fermion import jordan_wigner as _jw  # noqa: F401
    _HAS_FERMION = True
except Exception:
    _HAS_FERMION = False

_needs_fermion = pytest.mark.skipif(
    not _HAS_FERMION,
    reason="qubit_hamiltonian requires the fermion subpackage")

# H2 / STO-3G at R = 0.7414 A: the MO-basis chemist-notation integrals from
# test_df_qsvt_bridge.py, the independent reference the parse must reproduce.
H1 = np.array([[-1.25246357, 0.0], [0.0, -0.47594871]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = 0.67449876
ERI[1, 1, 1, 1] = 0.69716349
ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.66347258
ERI[0, 1, 0, 1] = ERI[1, 0, 1, 0] = 0.18128881
ERI[0, 1, 1, 0] = ERI[1, 0, 0, 1] = 0.18128881
E_NUCLEAR = 0.71375697
FCI_ENERGY = -1.137270

# The same molecule as an FCIDUMP: only the symmetry-unique integral records
# (the parser fills the eight-fold partners). Two-electron records first, then
# the one-electron h_ij (k = l = 0), then the core energy (all indices zero).
H2_STO3G_FCIDUMP = """\
&FCI NORB=2,NELEC=2,MS2=0,
 ORBSYM=1,1,
 ISYM=1,
&END
  0.67449876   1  1  1  1
  0.69716349   2  2  2  2
  0.66347258   1  1  2  2
  0.18128881   2  1  2  1
 -1.25246357   1  1  0  0
 -0.47594871   2  2  0  0
  0.71375697   0  0  0  0
"""

# The same integrals through the parser's tolerant path: lowercase `&fci`, a
# `/` namelist terminator, no ORBSYM/ISYM, a Fortran `D` exponent, a lowercase
# `e` exponent, blank and comment lines, irregular spacing, and two optional
# orbital-energy records (`value i 0 0 0`) the parser must skip, not reject.
H2_STO3G_FCIDUMP_VARIANT = """\
&fci norb=2 nelec=2 ms2=0 /

  0.67449876D+00    1  1  1  1
  0.69716349e+00    2  2  2  2
! Coulomb and exchange
  0.66347258        1  1  2  2
  0.18128881        2  1  2  1

 -1.25246357        1  1  0  0
 -0.47594871        2  2  0  0
 -0.57855300        1  0  0  0
  0.66940115        2  0  0  0
  0.71375697        0  0  0  0
"""


def _spectrum(one_body, eri, core_energy):
    operator = chemistry.qubit_hamiltonian(one_body,
                                           eri,
                                           scalar_offset=core_energy)
    return np.linalg.eigvalsh(np.asarray(operator.to_matrix()))


def test_from_fcidump_reproduces_reference_integrals():
    one_body, eri, core_energy = chemistry.from_fcidump(H2_STO3G_FCIDUMP)
    assert one_body.shape == (2, 2)
    assert eri.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(one_body, H1, atol=1e-12)
    np.testing.assert_allclose(eri, ERI, atol=1e-12)
    assert core_energy == pytest.approx(E_NUCLEAR, abs=1e-12)


def test_from_fcidump_fills_eightfold_symmetry():
    # A single (21|21) record must populate all four elements of its orbit,
    # and spin_orbital_tensors' symmetry check must accept the result.
    _, eri, _ = chemistry.from_fcidump(H2_STO3G_FCIDUMP)
    for indices in ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)):
        assert eri[indices] == pytest.approx(0.18128881, abs=1e-12)
    # Does not raise: the parsed eri obeys the chemist permutation symmetry.
    chemistry.spin_orbital_tensors(np.zeros((2, 2)),
                                   eri,
                                   validate_symmetry=True)


def test_tolerant_parsing_matches_strict():
    strict = chemistry.from_fcidump(H2_STO3G_FCIDUMP)
    variant = chemistry.from_fcidump(H2_STO3G_FCIDUMP_VARIANT)
    for reference, parsed in zip(strict, variant):
        np.testing.assert_allclose(np.asarray(parsed),
                                   np.asarray(reference),
                                   atol=1e-12)


@_needs_fermion
def test_from_fcidump_spectrum_matches_direct_integrals():
    one_body, eri, core_energy = chemistry.from_fcidump(H2_STO3G_FCIDUMP)
    parsed_spectrum = _spectrum(one_body, eri, core_energy)
    direct_spectrum = _spectrum(H1, ERI, E_NUCLEAR)
    np.testing.assert_allclose(parsed_spectrum, direct_spectrum, atol=1e-10)
    assert float(parsed_spectrum.min()) == pytest.approx(FCI_ENERGY, abs=5e-5)


def test_from_fcidump_rejects_missing_header():
    with pytest.raises(ValueError, match="&FCI"):
        chemistry.from_fcidump("0.5 1 1 0 0\n")


def test_from_fcidump_requires_norb():
    with pytest.raises(ValueError, match="NORB"):
        chemistry.from_fcidump("&FCI NELEC=2 &END\n 0.5 1 1 0 0\n")


def test_from_fcidump_rejects_malformed_line():
    with pytest.raises(ValueError, match="5 fields"):
        chemistry.from_fcidump("&FCI NORB=2 &END\n 0.5 1 1 1\n")


def test_from_fcidump_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        chemistry.from_fcidump("&FCI NORB=2 &END\n 0.5 3 3 0 0\n")


def test_from_fcidump_skips_orbital_energy_records():
    # `value i 0 0 0` is a standard optional orbital-energy record; it must be
    # skipped, leaving the integral tensors identical to a file without them.
    one_body, eri, core_energy = chemistry.from_fcidump(
        H2_STO3G_FCIDUMP_VARIANT)
    strict = chemistry.from_fcidump(H2_STO3G_FCIDUMP)
    np.testing.assert_allclose(one_body, strict[0], atol=1e-12)
    np.testing.assert_allclose(eri, strict[1], atol=1e-12)
    assert core_energy == pytest.approx(strict[2], abs=1e-12)


def test_from_fcidump_rejects_unrestricted_iuhf():
    with pytest.raises(ValueError, match="unrestricted"):
        chemistry.from_fcidump(
            "&FCI NORB=2,NELEC=2,IUHF=1,\n&END\n 0.5 1 1 1 1\n")


def test_from_fcidump_rejects_psi4_uhf_true():
    with pytest.raises(ValueError, match="unrestricted"):
        chemistry.from_fcidump(
            "&FCI NORB=2,NELEC=2,UHF=.TRUE.,\n&END\n 0.5 1 1 1 1\n")


def test_from_fcidump_accepts_restricted_iuhf_zero():
    # IUHF=0 is restricted and must not trip the unrestricted guard.
    one_body, _, _ = chemistry.from_fcidump(
        "&FCI NORB=2,NELEC=2,IUHF=0,\n&END\n -1.25246357 1 1 0 0\n")
    assert one_body[0, 0] == pytest.approx(-1.25246357, abs=1e-12)


@_needs_fermion
def test_from_fcidump_interops_with_pyscf_writer(tmp_path):
    # Round-trip through the canonical writer: PySCF writes an FCIDUMP, we
    # read the file's text back and parse it, and the spectrum must match the
    # integrals fed directly. tmp_path is used only because PySCF's writer
    # takes a path; from_fcidump itself still consumes a string.
    pytest.importorskip("pyscf")
    from pyscf.tools import fcidump

    path = tmp_path / "h2.fcidump"
    fcidump.from_integrals(str(path), H1, ERI, H1.shape[0], 2, nuc=E_NUCLEAR)

    one_body, eri, core_energy = chemistry.from_fcidump(path.read_text())
    np.testing.assert_allclose(_spectrum(one_body, eri, core_energy),
                               _spectrum(H1, ERI, E_NUCLEAR),
                               atol=1e-10)
