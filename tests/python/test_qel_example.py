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
EXAMPLE_PATH = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "qel_precomputed_molecules.py"
DATA_DIR = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "data"
H2_DATA_PATH = DATA_DIR / "h2_sto3g_jw.json"


@pytest.fixture(scope="module")
def qel_example_module():
    spec = importlib.util.spec_from_file_location("qel_precomputed_molecules",
                                                  EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "filename,num_qubits,num_electrons,num_terms,reference_kind,krylov_dim",
    [
        ("h2_sto3g_jw.json", 4, 2, 14, "FCI", 5),
        ("lih_sto3g_jw.json", 12, 4, 630, "FCI", 8),
        ("n2_active_space_jw.json", 8, 4, 80, "CASCI", 8),
        ("benzene_active_space_jw.json", 8, 4, 160, "CASCI", 8),
    ],
)
def test_load_precomputed_molecule_fixtures(qel_example_module, filename,
                                            num_qubits, num_electrons,
                                            num_terms, reference_kind,
                                            krylov_dim):
    data = qel_example_module.load_qubit_hamiltonian(DATA_DIR / filename)

    assert data.mapping == "Jordan-Wigner"
    assert data.num_qubits == num_qubits
    assert data.num_electrons == num_electrons
    assert data.occupied_qubits == tuple(range(num_electrons))
    assert data.reference_energy is not None
    assert data.reference_energy_kind == reference_kind
    assert data.recommended_krylov_dimension == krylov_dim
    assert len(data.terms) == num_terms
    assert all(len(term.word) == num_qubits for term in data.terms)


def test_precomputed_h2_dense_reference_energy(qel_example_module):
    data = qel_example_module.load_qubit_hamiltonian(H2_DATA_PATH)

    assert qel_example_module.exact_ground_energy(data) == pytest.approx(
        -1.137283840778, abs=1.0e-10)


def test_large_fixture_uses_stored_reference_when_dense_exact_is_disabled(
        qel_example_module):
    data = qel_example_module.load_qubit_hamiltonian(DATA_DIR /
                                                     "lih_sto3g_jw.json")

    energy, label = qel_example_module.comparison_energy(data,
                                                         exact_max_qubits=8)

    assert label == "FCI"
    assert energy == pytest.approx(-7.882401932290)


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
