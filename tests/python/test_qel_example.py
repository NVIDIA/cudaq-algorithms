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
EXAMPLE_PATH = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "quantum_exact_lanczos_molecules.py"
H2_EXAMPLE_PATH = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "quantum_exact_lanczos_h2.py"
DATA_DIR = REPO_ROOT / "examples" / "quantum_exact_lanczos" / "data"
H2_DATA_PATH = DATA_DIR / "h2_sto3g_jw.json"


@pytest.fixture(scope="module")
def qel_example_module():
    spec = importlib.util.spec_from_file_location(
        "quantum_exact_lanczos_molecules", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qel_h2_example_module(qel_example_module):
    # The walkthrough imports quantum_exact_lanczos_molecules, which the
    # qel_example_module fixture already registered in sys.modules.
    spec = importlib.util.spec_from_file_location("quantum_exact_lanczos_h2",
                                                  H2_EXAMPLE_PATH)
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


@pytest.fixture
def qpp_cpu_target():
    import cudaq
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _exact_chebyshev_moments(qel, data, num_moments):
    """Exact <HF| T_k(H/alpha) |HF> using CUDA-Q's own matrix/state ordering."""
    import cudaq
    import cudaq_algorithms as algorithms

    spin_op = qel.spin_hamiltonian(data.terms)
    alpha = float(algorithms.PauliLCU(spin_op, data.num_qubits).normalization)
    scaled = np.array(spin_op.to_matrix(), dtype=np.complex128) / alpha

    num_qubits = data.num_qubits
    occupied = [int(q) for q in data.occupied_qubits]

    @cudaq.kernel
    def hartree_fock():
        system = cudaq.qvector(num_qubits)
        for qubit in occupied:
            x(system[qubit])

    psi = np.array(cudaq.get_state(hartree_fock), dtype=np.complex128)

    identity = np.eye(scaled.shape[0], dtype=np.complex128)
    chebyshev = [identity, scaled.copy()]
    while len(chebyshev) < num_moments:
        chebyshev.append(2.0 * scaled @ chebyshev[-1] - chebyshev[-2])
    return [
        float(np.real(psi.conj() @ chebyshev[k] @ psi))
        for k in range(num_moments)
    ]


# Test purpose: execute the QEL quantum workflow (previously untested) and check
# the measured Chebyshev moments against exact dense values.
def test_h2_measured_moments_match_exact_chebyshev(qel_example_module,
                                                   qpp_cpu_target):
    import cudaq_algorithms as algorithms

    qel = qel_example_module
    data = qel.load_qubit_hamiltonian(H2_DATA_PATH)
    encoding = algorithms.PauliLCU(qel.spin_hamiltonian(data.terms),
                                   data.num_qubits)
    dimension = 3
    num_moments = algorithms.krylov.required_chebyshev_moments(dimension)

    measured = qel.collect_chebyshev_moments(encoding, data.occupied_qubits,
                                             dimension, 0)
    exact = _exact_chebyshev_moments(qel, data, num_moments)

    assert np.allclose(measured, exact, atol=1e-6)


# Test purpose: exercise the 8-qubit quantum moment path (occupied qubits
# (0, 1, 2, 3)), which the H2 tests do not reach.
def test_n2_measured_low_order_moments_match_exact_chebyshev(
        qel_example_module, qpp_cpu_target):
    import cudaq_algorithms as algorithms

    qel = qel_example_module
    data = qel.load_qubit_hamiltonian(DATA_DIR / "n2_active_space_jw.json")
    assert data.num_qubits == 8
    assert data.occupied_qubits == (0, 1, 2, 3)

    encoding = algorithms.PauliLCU(qel.spin_hamiltonian(data.terms),
                                   data.num_qubits)
    # Keep this cheap: a dimension-2 basis only needs walk powers 0 and 1.
    dimension = 2
    num_moments = algorithms.krylov.required_chebyshev_moments(dimension)

    measured = qel.collect_chebyshev_moments(encoding, data.occupied_qubits,
                                             dimension, 0)
    exact = _exact_chebyshev_moments(qel, data, num_moments)

    assert np.allclose(measured, exact, atol=1e-8)


# Test purpose: run the recommended-starting-point H2 walkthrough end to end.
def test_h2_walkthrough_example_matches_dense_exact(qel_h2_example_module,
                                                    qpp_cpu_target,
                                                    monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["quantum_exact_lanczos_h2.py"])

    # main() raises RuntimeError if the QEL energy misses dense exact
    # diagonalization by more than the example tolerance.
    assert qel_h2_example_module.main() == 0

    output = capsys.readouterr().out
    assert "QEL energy" in output
    assert "Dense exact energy" in output


# Test purpose: run the full QEL workflow end to end and check it reproduces FCI.
def test_h2_qel_workflow_energy_matches_fci(qel_example_module,
                                            qpp_cpu_target):
    qel = qel_example_module
    data = qel.load_qubit_hamiltonian(H2_DATA_PATH)

    result = qel.run_qel_workflow(data,
                                  krylov_dimension=3,
                                  overlap_cutoff=1.0e-6,
                                  shots_count=0,
                                  exact_max_qubits=data.num_qubits)

    assert result["qel_energy"] == pytest.approx(qel.exact_ground_energy(data),
                                                 abs=1.0e-6)
