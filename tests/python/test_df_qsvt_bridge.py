# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Double factorization -> qubit Hamiltonian -> quantum primitives.

Pins the chemistry bridge (spin expansion + compiled Jordan-Wigner) against
literature H2/STO-3G values, and the DF-compression path end to end:
factorize, reconstruct, bridge, and feed PauliLCU/Walk.
"""

import importlib.util

import numpy as np
import pytest

import cudaq_algorithms as algorithms
from cudaq_algorithms import PauliLCU, Walk, chemistry

df = algorithms.double_factorization

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cudaq_algorithms._pycudaq_algorithms") is None,
    reason="chemistry bridge requires the compiled extension "
    "(fermion.jordan_wigner)")

# H2 / STO-3G at R = 0.7414 A: standard MO-basis integrals (chemist
# notation, spatial orbitals); FCI total energy -1.137270 Ha.
H1 = np.array([[-1.25246357, 0.0], [0.0, -0.47594871]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = 0.67449876
ERI[1, 1, 1, 1] = 0.69716349
ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.66347258
ERI[0, 1, 0, 1] = ERI[1, 0, 1, 0] = 0.18128881
ERI[0, 1, 1, 0] = ERI[1, 0, 0, 1] = 0.18128881
E_NUCLEAR = 0.71375697
FCI_ENERGY = -1.137270


def test_bridge_reproduces_h2_ground_state():
    spin_op = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    assert spin_op.qubit_count == 4
    ground = float(np.min(np.linalg.eigvalsh(spin_op.to_matrix())))
    assert ground == pytest.approx(FCI_ENERGY, abs=5e-5)


def test_bridge_validates_shapes():
    with pytest.raises(ValueError, match="square"):
        chemistry.spin_orbital_tensors(np.zeros((2, 3)), ERI)
    with pytest.raises(ValueError, match="chemist-notation"):
        chemistry.spin_orbital_tensors(H1, np.zeros((3, 3, 3, 3)))


def test_full_rank_factorization_roundtrips_the_hamiltonian():
    factorization = df.explicit_double_factorization(ERI, threshold=0.0)
    assert df.factorization_error(ERI, factorization) < 1e-10

    direct = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    reconstructed = chemistry.qubit_hamiltonian(
        H1, df.reconstruct_eri(factorization), scalar_offset=E_NUCLEAR)

    direct_ground = float(np.min(np.linalg.eigvalsh(direct.to_matrix())))
    rebuilt_ground = float(
        np.min(np.linalg.eigvalsh(reconstructed.to_matrix())))
    assert rebuilt_ground == pytest.approx(direct_ground, abs=1e-8)

    assert PauliLCU(reconstructed).alpha == pytest.approx(
        PauliLCU(direct).alpha, abs=1e-8)


def test_compression_reduces_alpha_and_bounds_energy_error():
    exact_alpha = PauliLCU(
        chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)).alpha

    truncated = df.explicit_double_factorization(ERI, max_num_leaves=1)
    truncated_h = chemistry.qubit_hamiltonian(H1,
                                              df.reconstruct_eri(truncated),
                                              scalar_offset=E_NUCLEAR)
    truncated_alpha = PauliLCU(truncated_h).alpha

    # Truncation drops LCU weight and shifts the spectrum by an amount
    # controlled by the reconstruction error.
    assert truncated_alpha < exact_alpha
    ground = float(np.min(np.linalg.eigvalsh(truncated_h.to_matrix())))
    tensor_error = df.factorization_error(ERI, truncated)
    assert abs(ground - FCI_ENERGY) < 4.0 * tensor_error + 5e-5


def test_bridged_hamiltonian_feeds_the_walk():
    spin_op = chemistry.qubit_hamiltonian(H1, ERI, scalar_offset=E_NUCLEAR)
    encoding = PauliLCU(spin_op)
    walk = Walk(encoding)

    dense = np.asarray(spin_op.to_matrix())
    rng = np.random.default_rng(11)
    ket = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

    expected_t1 = float(np.real(ket.conj() @ (dense @ ket))) / encoding.alpha
    assert walk.moment(ket, 1) == pytest.approx(expected_t1, abs=1e-8)
    assert walk.moment(ket, 0) == pytest.approx(1.0, abs=1e-8)
