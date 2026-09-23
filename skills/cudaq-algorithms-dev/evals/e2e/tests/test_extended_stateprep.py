# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent contracts for the state-preparation completion cases."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))


@pytest.fixture(scope="module")
def cases():
    return importlib.import_module("extended_stateprep")


def test_literal_little_endian_pauli_and_fermion_signs(cases):
    np.testing.assert_allclose(
        cases._pauli_matrix("YI")[:, 0], [0.0, 1.0j, 0.0, 0.0])
    annihilators = cases._annihilators(2)
    np.testing.assert_allclose(annihilators[1][:, 3], [0.0, -1.0, 0.0, 0.0])


def test_reference_primitives_pin_half_angles_and_resource_formulas(cases):
    params = {
        "num_qubits": 4,
        "num_electrons": 2,
        "spin": 0,
        "explicit_occupation": [0, 2],
        "single": {
            "theta": 0.6,
            "occupied": [0],
            "p": 0,
            "q": 1
        },
        "double": {
            "theta": 0.6,
            "occupied": [0, 1],
            "p": 0,
            "q": 1,
            "r": 2,
            "s": 3,
        },
    }
    got = cases.expected("reference_excitation_primitives", params)

    canonical = np.zeros(16, complex)
    canonical[3] = 1.0
    explicit = np.zeros(16, complex)
    explicit[5] = 1.0
    single = np.zeros(16, complex)
    single[1], single[2] = np.cos(0.3), -np.sin(0.3)
    double = np.zeros(16, complex)
    double[3], double[12] = np.cos(0.3), np.sin(0.3)
    np.testing.assert_allclose(got["canonical_state"], canonical, atol=1e-14)
    np.testing.assert_allclose(got["explicit_state"], explicit, atol=1e-14)
    np.testing.assert_allclose(got["single_state"], single, atol=1e-14)
    np.testing.assert_allclose(got["double_state"], double, atol=1e-14)
    np.testing.assert_allclose([
        got[name]
        for name in ("canonical_particle_number", "explicit_particle_number",
                     "single_particle_number", "double_particle_number")
    ], [2.0, 2.0, 1.0, 2.0])
    assert got["canonical_num_qubits"] == 4
    assert got["canonical_num_electrons"] == 2
    assert got["canonical_num_x_gates"] == 2
    assert got["explicit_num_qubits"] == 4
    assert got["explicit_num_electrons"] == 2
    assert got["explicit_num_x_gates"] == 2


def test_single_excitation_fixture_rejects_parity_free_ladder_mutation(cases):
    params = cases.parameters("reference_excitation_primitives", 0)
    single = params["single"]
    assert single["occupied"] == [0, 1]
    assert (single["p"], single["q"]) == (0, 3)

    theta = single["theta"]
    expected = cases.expected("reference_excitation_primitives",
                              params)["single_state"]
    literal = np.zeros(16, complex)
    literal[3], literal[10] = np.cos(theta / 2), np.sin(theta / 2)
    np.testing.assert_allclose(expected, literal, atol=1e-14)

    from scipy.linalg import expm
    lowering = cases._annihilators(4, fermionic=False)
    transfer = lowering[3].conj().T @ lowering[0]
    parity_free_generator = -1j * (transfer - transfer.conj().T)
    mutant = expm(-0.5j * theta * parity_free_generator) @ cases._basis_state(
        4, single["occupied"])
    assert np.linalg.norm(expected - mutant) > 0.4


def test_grouped_kernels_have_distinct_signed_actions_and_exact_resources(
        cases):
    params = cases.parameters("grouped_ucc_device", 0)
    params["uccgsd_amplitudes"] = [0.0] * 9
    params["uccgsd_amplitudes"][1] = 0.4
    params["upccgsd_amplitudes"] = [0.4, 0.0, 0.0]
    params["ceo_amplitudes"] = [0.4, 0.0, 0.0, 0.0]
    params["fixed_amplitudes"] = [0.4]
    params["fixed_words"] = [["IYXI", "IXYI"]]
    params["fixed_coefficients"] = [[0.5, -0.5]]
    got = cases.expected("grouped_ucc_device", params)

    ucc = np.zeros(16, complex)
    ucc[3], ucc[6] = np.cos(0.4), -np.sin(0.4)
    up = np.zeros(16, complex)
    up[3], up[6] = np.cos(0.4), -np.sin(0.4)
    ceo = np.zeros(16, complex)
    ceo[3], ceo[6] = np.cos(0.4), np.sin(0.4)
    fixed = np.zeros(16, complex)
    fixed[3], fixed[5] = np.cos(0.4), np.sin(0.4)
    np.testing.assert_allclose(got["uccgsd_state"], ucc, atol=1e-14)
    np.testing.assert_allclose(got["upccgsd_state"], up, atol=1e-14)
    np.testing.assert_allclose(got["ceo_state"], ceo, atol=1e-14)
    np.testing.assert_allclose(got["fixed_state"], fixed, atol=1e-14)
    assert got["fixed_num_qubits"] == 4
    assert got["fixed_num_excitations"] == 1
    assert got["fixed_num_pauli_rotations"] == 2
    assert got["fixed_max_pauli_rotations_per_excitation"] == 2


def test_givens_analytic_map_slater_minors_and_resource_formulas(cases):
    theta, phase = 0.37, 0.73
    orbitals = np.array([[np.cos(theta)], [np.sin(theta)], [0.0]])
    params = {
        "num_qubits": 3,
        "rotation_theta": theta,
        "rotation_first": 0,
        "rotation_second": 1,
        "phase_theta": theta,
        "phase": phase,
        "phase_first": 0,
        "phase_second": 1,
        "orbitals": orbitals.tolist(),
    }
    got = cases.expected("givens_raw_device", params)
    real = np.zeros(8, complex)
    real[1], real[2] = np.cos(theta), np.sin(theta)
    phase_state = np.zeros(8, complex)
    # Raw rz contributes the displayed absolute phases; their ratio is exp(i phi).
    phase_state[1] = np.exp(-0.5j * phase) * np.cos(theta)
    phase_state[2] = np.exp(0.5j * phase) * np.sin(theta)
    np.testing.assert_allclose(got["givens_state"], real, atol=1e-14)
    np.testing.assert_allclose(got["phase_givens_state"],
                               phase_state,
                               atol=1e-14)
    np.testing.assert_allclose(got["slater_state"], real, atol=1e-14)
    assert got["slater_norm"] == pytest.approx(1.0)
    assert got["slater_particle_number"] == pytest.approx(1.0)
    assert got["num_spin_orbitals"] == 3
    assert got["num_electrons"] == 1
    assert got["num_givens_rotations"] == 1
    assert got["num_exp_pauli_calls"] == 2
    assert got["num_phase_rotations"] == 0
    assert got["two_qubit_gate_count_proxy"] == 2
    assert got["depth_proxy"] == 2


def test_case_schemas_are_numeric_finite_and_variants_are_distinct(cases):
    specs = cases.case_specs()
    assert [spec["id"] for spec in specs] == [
        "reference_excitation_primitives", "grouped_ucc_device",
        "givens_raw_device"
    ]
    for spec in specs:
        variants = [
            cases.parameters(spec["id"], variant) for variant in (0, 1)
        ]
        assert json.dumps(variants[0],
                          sort_keys=True) != json.dumps(variants[1],
                                                        sort_keys=True)
        for params in variants:
            json.dumps(params, allow_nan=False)
            expected = cases.expected(spec["id"], params)
            assert set(expected) == set(spec["outputs"])
            assert all(
                np.asarray(value).dtype != object and np.isfinite(value).all()
                for value in expected.values())
        compile(cases.reference_source(spec["id"]), "app.py", "exec")


def test_specs_map_exactly_the_fifteen_uncovered_stateprep_records(cases):
    expected = {
        "state-preparation-kernel-hartree-fock",
        "state-preparation-kernel-hartree-fock-occupation",
        "state-preparation-kernel-single-excitation",
        "state-preparation-kernel-double-excitation",
        "state-preparation-kernel-uccgsd",
        "state-preparation-kernel-upccgsd",
        "state-preparation-kernel-ceo",
        "state-preparation-kernel-fixed-parameter-ucc",
        "state-preparation-kernel-givens-rotation",
        "state-preparation-kernel-phase-givens-rotation",
        "state-preparation-kernel-slater-determinant",
        "state-preparation-resources-givens",
        "state-preparation-resources-hartree-fock",
        "state-preparation-resources-hartree-fock-occupation",
        "state-preparation-resources-fixed-parameter-ucc",
    }
    actual = {
        feature
        for spec in cases.case_specs()
        for feature in spec["feature_ids"]
    }
    assert actual == expected
    assert sum(len(spec["feature_ids"]) for spec in cases.case_specs()) == 15


def test_all_public_apis_are_disclosed_and_provenance_is_exact(cases):
    aliases = {
        "cudaq_algorithms.stateprep._hartree_fock.estimate_hartree_fock_resources":
        "cudaq_algorithms.stateprep.estimate_hartree_fock_resources",
        "cudaq_algorithms.stateprep._hartree_fock.estimate_hartree_fock_occupation_resources":
        "cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources",
        "cudaq_algorithms.stateprep._hartree_fock.estimate_fixed_parameter_ucc_resources":
        "cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources",
        "cudaq_algorithms.stateprep._givens.make_givens_rotation_schedule":
        "cudaq_algorithms.stateprep.make_givens_rotation_schedule",
        "cudaq_algorithms.stateprep._givens.estimate_givens_resources":
        "cudaq_algorithms.stateprep.estimate_givens_resources",
    }
    for spec in cases.case_specs():
        public = []
        for symbol in spec["required_symbols"] + spec["required_kernels"]:
            if symbol in aliases:
                public.append(aliases[symbol])
            elif symbol.startswith("cudaq_algorithms.stateprep."):
                public.append("cudaq_algorithms.stateprep." +
                              symbol.rsplit(".", 1)[1])
            else:
                public.append(symbol)
        assert spec["required_public_apis"] == public
        for api in public:
            assert api in spec["task"]


def test_oracles_do_not_import_the_evaluated_package(cases):
    before = {
        name
        for name in sys.modules if name.startswith("cudaq_algorithms")
    }
    for spec in cases.case_specs():
        cases.expected(spec["id"], cases.parameters(spec["id"], 0))
    after = {
        name
        for name in sys.modules if name.startswith("cudaq_algorithms")
    }
    assert after == before


def test_grading_rejects_wrong_numeric_output_and_missing_device_use(
        cases, tmp_path):
    from grading import compare_output, required_calls_seen

    spec = cases.case_specs()[0]
    expected = cases.expected(spec["id"], cases.parameters(spec["id"], 0))
    wrong = dict(expected)
    wrong["single_state"] = np.zeros_like(wrong["single_state"])
    path = tmp_path / "wrong.npz"
    np.savez(path, **wrong)
    assert not compare_output(path, expected, spec)["passed"]
    assert not required_calls_seen(spec["required_kernels"], [])["passed"]


@pytest.mark.skipif(not os.environ.get("E2E_RUNTIME_PYTHON"),
                    reason="supported runtime supplied by controller")
def test_real_gold_apps_run_in_isolation_on_both_variants(
        cases, monkeypatch, tmp_path):
    import run

    registry = {spec["id"]: (cases, spec) for spec in cases.case_specs()}
    monkeypatch.setattr(run, "CASES", registry)
    python = os.environ["E2E_RUNTIME_PYTHON"]
    for spec in cases.case_specs():
        destination = tmp_path / spec["id"]
        result = run.execute_app(REPO,
                                 spec,
                                 cases.reference_source(spec["id"]),
                                 python,
                                 destination,
                                 limit=180)
        assert result["passed"], result
