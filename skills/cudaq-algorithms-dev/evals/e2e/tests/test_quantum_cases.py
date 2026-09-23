# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent literal physics checks and optional real-runtime preflights."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class QuantumContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = ROOT / "quantum_cases.py"
        assert path.exists(
        ), "quantum_cases.py must implement the numerical contracts"
        spec = importlib.util.spec_from_file_location("quantum_cases", path)
        cls.q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.q)

    def test_y_sign_and_qubit_zero_is_low_bit(self):
        ket = np.array([1, 0, 0, 0], complex)
        np.testing.assert_allclose(
            self.q.pauli_matrix("YI") @ ket, [0, 1j, 0, 0])
        np.testing.assert_allclose(
            self.q.pauli_matrix("IY") @ ket, [0, 0, 1j, 0])
        np.testing.assert_allclose(
            self.q.pauli_matrix("YI") @ [0, 1, 0, 0], [-1j, 0, 0, 0])

    def test_lcu_preserves_signed_action_identity_energy_and_success_norm(
            self):
        p = {
            "terms": [[.5, "I"], [-1.5, "Y"]],
            "ket_real": [1., 0.],
            "ket_imag": [0., 0.]
        }
        got = self.q.expected("pauli_action", p)
        np.testing.assert_allclose(got["block"], [.25, -.75j])
        np.testing.assert_allclose(got["action"], [.5, -1.5j])
        self.assertAlmostEqual(got["alpha"], 2)
        self.assertAlmostEqual(got["energy"], .5)
        self.assertAlmostEqual(got["success_probability"], .625)
        self.assertGreater(
            np.linalg.norm(got["block"] -
                           got["block"] / np.linalg.norm(got["block"])), .1)

    def test_slater_minors_have_relative_complex_phase_and_fermion_sign(self):
        orbitals = np.array([[1, 0], [0, 1], [1j, 0], [0, 1]]) / np.sqrt(2)
        want = np.zeros(16, complex)
        want[[3, 9, 6, 12]] = [.5, .5, -.5j, .5j]
        actual = self.q.slater_state(orbitals)
        np.testing.assert_allclose(actual, want, atol=1e-14)
        self.assertGreater(np.linalg.norm(actual - want.conj()), 1)

    def test_uccsd_single_and_double_have_distinct_signs(self):
        gs = self.q.pool_generators("pool_uccsd")
        self.assertEqual(len(gs), 3)
        # |0011> -> |0110> crosses occupied orbital 1; double -> |1100>.
        self.assertAlmostEqual(gs[0][6, 3], 1j)
        self.assertAlmostEqual(gs[1][9, 3], -1j)
        self.assertAlmostEqual(gs[2][12, 3], 1j)

    def test_generalized_paired_and_ceo_generators_have_literal_actions(self):
        gs = self.q.pool_generators("pool_uccgsd")
        self.assertEqual(len(gs), 9)
        self.assertAlmostEqual(gs[0][2, 1], -1j)
        self.assertAlmostEqual(gs[6][12, 3], 1j)
        paired = self.q.pool_generators("pool_upccgsd")
        self.assertEqual(len(paired), 3)
        self.assertAlmostEqual(paired[2][12, 3], -1j)
        ceo = self.q.pool_generators("pool_ceo")
        self.assertEqual(len(ceo), 4)
        self.assertAlmostEqual(ceo[0][6, 3], -1j)  # no JW parity
        self.assertAlmostEqual(ceo[2][12, 3], 1j)
        self.assertAlmostEqual(ceo[3][12, 3], 1j)
        self.assertAlmostEqual(ceo[2][9, 6], 1j)
        self.assertAlmostEqual(ceo[3][9, 6], -1j)

    def test_direct_uccsd_half_angle_and_mixed_double_sign(self):
        p = self.q.parameters("direct_uccsd", 0)
        p["amplitudes"] = [0., 0., .6]
        want = np.zeros(16, complex)
        want[3], want[12] = np.cos(.3), -np.sin(.3)
        np.testing.assert_allclose(self.q.expected("direct_uccsd", p)["state"],
                                   want,
                                   atol=1e-13)
        fixed = self.q.expected("pool_uccsd", p)["state"]
        self.assertAlmostEqual(fixed[12], -np.sin(.6))
        self.assertGreater(np.linalg.norm(fixed - want), .2)

    def test_chebyshev_recurrence_includes_odd_order_sign(self):
        p = self.q.parameters("walk_spectrum", 0)
        p.update(terms=[[.25, "I"], [.75, "Z"]],
                 ket_real=[1 / np.sqrt(2)] * 2,
                 ket_imag=[0., 0.],
                 count=5,
                 power=3)
        got = self.q.expected("walk_spectrum", p)
        np.testing.assert_allclose(got["moments"], [1, .25, .25, 1, .25],
                                   atol=1e-13)
        np.testing.assert_allclose(got["block"], [-1 / np.sqrt(2)] * 2,
                                   atol=1e-13)

    def test_qsvt_projector_phase_and_direction_conventions(self):
        self.assertAlmostEqual(self.q.signal_response(.3, [.2]), np.exp(.2j))
        self.assertAlmostEqual(self.q.signal_response(.3, [.2, -.7]),
                               -.3 * np.exp(-.5j))
        self.assertAlmostEqual(self.q.signal_response(.3, [0, 0, 0]),
                               2 * .3**2 - 1)
        self.assertAlmostEqual(self.q.signal_response(.3, [0, 0, 0], [0, 1]),
                               1)
        self.assertAlmostEqual(
            self.q.signal_response(.3, [.2, -.7], convention="qsp"),
            -.3 * np.exp(-1j))

    def test_recovery_is_unitary_real_domain_evolution_with_nontrivial_phase(
            self):
        from scipy.linalg import expm
        for v in (0, 1):
            p = self.q.parameters("qsvt_recovery", v)
            h = np.array([[p["terms"][1][0], p["terms"][0][0]],
                          [p["terms"][0][0], -p["terms"][1][0]]])
            ket = np.array(p["ket_real"])
            want = expm(-1j * p["time"] * h) @ ket
            got = self.q.expected("qsvt_recovery", p)
            np.testing.assert_allclose(got["evolved"], want, atol=1e-13)
            cos = got["cos_block"] * np.exp(-1j * sum(p["cos_phases"]))
            sin = got["sin_block"] * np.exp(-1j * sum(p["sin_phases"]))
            np.testing.assert_allclose(2 * (cos.real + 1j * sin.imag),
                                       want,
                                       atol=1e-13)
            wrong = got["cos_block"].real + 1j * got["sin_block"].imag
            self.assertGreater(np.linalg.norm(wrong - want), .2)

    def test_all_contracts_are_finite_serializable_and_variants_differ(self):
        specs = self.q.case_specs()
        self.assertEqual(len(specs), 10)
        for spec in specs:
            variants = []
            for v in (0, 1):
                p = self.q.parameters(spec["id"], v)
                json.dumps(p, allow_nan=False)
                out = self.q.expected(spec["id"], p)
                self.assertEqual(set(out), set(spec["outputs"]))
                for value in out.values():
                    self.assertTrue(np.isfinite(value).all())
                    self.assertNotEqual(np.asarray(value).dtype, object)
                variants.append(p)
                compile(self.q.reference_source(spec["id"]), "app.py", "exec")
            self.assertNotEqual(variants[0], variants[1])

    def test_required_simulation_helpers_are_disclosed_in_task_prompt(self):
        for spec in self.q.case_specs():
            for symbol in spec["required_symbols"]:
                if symbol.startswith("cudaq_algorithms.sim_utils."):
                    with self.subTest(case=spec["id"], helper=symbol):
                        self.assertIn(symbol.removeprefix("cudaq_algorithms."),
                                      spec["task"])

    def test_every_traced_requirement_discloses_its_public_api(self):
        aliases = {
            "cudaq_algorithms.pauli_lcu.PauliLCU.__init__":
            "cudaq_algorithms.PauliLCU",
            "cudaq_algorithms.qubitization.Walk.moments":
            "cudaq_algorithms.Walk.moments",
            "cudaq_algorithms.qubitization.Walk.kernel":
            "cudaq_algorithms.Walk.kernel",
            "cudaq_algorithms.qsvt.PhaseSequence.__init__":
            "cudaq_algorithms.PhaseSequence",
            "cudaq_algorithms.qsvt.QSVT.kernel":
            "cudaq_algorithms.QSVT.kernel",
            "cudaq_algorithms.qsvt.recover_real_time_evolution":
            "cudaq_algorithms.recover_real_time_evolution",
        }
        for spec in self.q.case_specs():
            public = []
            for symbol in spec["required_symbols"] + spec.get(
                    "required_kernels", []):
                if symbol.startswith("cudaq_algorithms.stateprep."):
                    public.append("cudaq_algorithms.stateprep." +
                                  symbol.rsplit(".", 1)[1])
                else:
                    public.append(aliases.get(symbol, symbol))
            with self.subTest(case=spec["id"]):
                self.assertEqual(spec.get("required_public_apis"), public)

    @unittest.skipUnless(
        os.environ.get("CUDAQ_E2E_PYTHON"),
        "set supported runtime interpreter to execute real gold programs")
    def test_real_package_preflights_on_public_and_hidden_inputs(self):
        for spec in self.q.case_specs():
            for v in (0, 1):
                with self.subTest(
                        case=spec["id"],
                        variant=v), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    # Test fixture creation is local to the test, never package mocking.
                    (root / "app.py").write_text(
                        self.q.reference_source(spec["id"]))
                    p = self.q.parameters(spec["id"], v)
                    (root / "input.json").write_text(json.dumps(p))
                    run = subprocess.run([
                        os.environ["CUDAQ_E2E_PYTHON"],
                        str(root / "app.py"), "--input",
                        str(root / "input.json"), "--output",
                        str(root / "result.npz")
                    ],
                                         text=True,
                                         capture_output=True,
                                         timeout=120)
                    self.assertEqual(run.returncode, 0,
                                     run.stdout + run.stderr)
                    with np.load(root / "result.npz",
                                 allow_pickle=False) as got:
                        for field, want in self.q.expected(spec["id"],
                                                           p).items():
                            actual = got[field]
                            if field in spec.get("phase_invariant_fields", []):
                                overlap = np.vdot(want, actual)
                                self.assertGreater(abs(overlap), 1e-12)
                                actual = actual * np.exp(
                                    -1j * np.angle(overlap))
                            np.testing.assert_allclose(actual,
                                                       want,
                                                       atol=spec["atol"],
                                                       rtol=spec["rtol"])


if __name__ == "__main__":
    unittest.main()
