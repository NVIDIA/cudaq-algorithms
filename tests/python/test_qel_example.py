# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "qel_precomputed_h2.py"
DATA_PATH = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "data" / "h2_sto3g_jw.json"


@pytest.fixture(scope="module")
def qel_example_module():
    spec = importlib.util.spec_from_file_location("qel_precomputed_h2",
                                                  EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_precomputed_h2_fixture(qel_example_module):
    data = qel_example_module.load_qubit_hamiltonian(DATA_PATH)

    assert data.name == "H2 STO-3G"
    assert data.mapping == "Jordan-Wigner"
    assert data.num_qubits == 4
    assert data.num_electrons == 2
    assert data.occupied_qubits == (0, 1)
    assert data.constant == pytest.approx(-0.09706627)
    assert len(data.terms) == 14


def test_precomputed_h2_dense_reference_energy(qel_example_module):
    data = qel_example_module.load_qubit_hamiltonian(DATA_PATH)

    assert qel_example_module.exact_ground_energy(data) == pytest.approx(
        -1.137283840778, abs=1.0e-10)


def test_conditioned_generalized_eigenproblem_filters_small_overlap(
        qel_example_module):
    hamiltonian_matrix = np.diag([0.25, 2.0])
    overlap_matrix = np.diag([1.0, 1.0e-14])

    result = qel_example_module.solve_conditioned_generalized_eigenproblem(
        hamiltonian_matrix, overlap_matrix, 1.0e-10)

    assert result.kept_rank == 1
    assert result.condition_estimate == pytest.approx(1.0)
    assert result.eigenvalues == pytest.approx([0.25])
    assert result.overlap_eigenvalues == pytest.approx([1.0e-14, 1.0])
