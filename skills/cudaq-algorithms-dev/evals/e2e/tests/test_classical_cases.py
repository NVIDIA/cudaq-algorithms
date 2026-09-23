# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent numerical contracts; never import cudaq_algorithms for oracles."""
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
sys.path.insert(0, str(ROOT))


class ClassicalContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spec = importlib.util.find_spec("classical_cases")

    def module(self):
        self.assertIsNotNone(self.spec,
                             "classical case implementation is missing")
        import classical_cases
        return classical_cases

    def test_pauli_y_sign_and_little_endian(self):
        c = self.module()
        np.testing.assert_allclose(c._pauli("YI")[:, 0], [0, 1j, 0, 0])
        np.testing.assert_allclose(c._pauli("IX")[:, 0], [0, 0, 1, 0])

    def test_fock_ladders_sign_and_quartic_order(self):
        c = self.module()
        a = c._annihilators(2)
        np.testing.assert_allclose(a[1][:, 3], [0, -1, 0, 0])
        h = np.zeros((2, 2), complex)
        v = np.zeros((2, 2, 2, 2), complex)
        v[0, 1, 1, 0] = 0.7
        np.testing.assert_allclose(c._fock(h, v, 0.2),
                                   np.diag([.2, .2, .2, .9]))

    def test_bk_occupation_permutation_is_fenwick_parity(self):
        c = self.module()
        self.assertEqual(
            c._bk_permutation(4).tolist(),
            [0, 11, 10, 1, 12, 7, 6, 13, 8, 3, 2, 9, 4, 15, 14, 5])

    def test_chemist_energy_has_half_factor_and_positive_repulsion(self):
        c = self.module()
        np.testing.assert_allclose(
            c._chemist_fock(np.array([[-1.]]), np.array([[[[.6]]]]), .3),
            np.diag([.3, -.7, -.7, -1.1]))

    def test_trotter_identity_phase_is_required_and_state_moves(self):
        c = self.module()
        p = c.parameters("trotter_dynamics", 0)
        out = c.expected("trotter_dynamics", p)
        initial = np.array(p["state_real"]) + 1j * np.array(p["state_imag"])
        self.assertGreater(np.linalg.norm(out["state"] - initial), .1)
        wrong = out["state"] * np.exp(1j * p["identity"] * p["time"])
        self.assertGreater(np.linalg.norm(wrong - out["state"]), .05)
        self.assertAlmostEqual(np.linalg.norm(out["state"]), 1.)
        self.assertEqual(int(out["pauli_rotations"]), 24)
        self.assertEqual(int(out["estimated_cx_count"]), 24)

    def test_compression_integrals_are_nontrivial_rank_two(self):
        c = self.module()
        for variant in (0, 1):
            p = c.parameters("molecular_compression", variant)
            out = c.expected("molecular_compression", p)
            eri = out["eri"]
            self.assertEqual(
                np.linalg.matrix_rank(eri.reshape(4, 4), tol=1e-10), 2)
            self.assertGreater(abs(eri[0, 0, 0, 1]), .01)
            np.testing.assert_allclose(out["compressed_eri"], eri)
            np.testing.assert_allclose(out["explicit_eri"], eri)
            self.assertEqual(int(out["compressed_leaves"]), 1)

    def test_compression_input_contains_no_analytic_factorization_answers(
            self):
        c = self.module()
        for variant in (0, 1):
            self.assertEqual(
                set(c.parameters("molecular_compression", variant)),
                {"fcidump", "num_electrons", "num_leaves"})

    def test_compression_reference_reads_fcidump_without_analytic_metadata(
            self):
        c = self.module()
        p = {
            "fcidump":
            "&FCI NORB=2,NELEC=2,MS2=0,\n&END\n"
            "0.6 1 1 1 1\n0.4 2 2 2 2\n"
            "-1.0 1 1 0 0\n-0.25 2 2 0 0\n0.3 0 0 0 0\n",
            "num_electrons":
            2,
            "num_leaves":
            1
        }
        out = c.expected("molecular_compression", p)
        np.testing.assert_allclose(out["one_body"], np.diag([-1., -.25]))
        eri = np.zeros((2, 2, 2, 2))
        eri[0, 0, 0, 0], eri[1, 1, 1, 1] = .6, .4
        np.testing.assert_allclose(out["eri"], eri)
        np.testing.assert_allclose(out["modified_one_body"],
                                   np.diag([-1.3, -.45]))
        self.assertAlmostEqual(out["ground_energy"], -1.1)
        p["fcidump"] = p["fcidump"].replace("0.3 0 0 0 0", "0.7 0 0 0 0")
        self.assertAlmostEqual(
            c.expected("molecular_compression", p)["ground_energy"], -.7)

    def test_required_simulation_helpers_are_disclosed_in_task_prompt(self):
        for spec in self.module().case_specs():
            for symbol in spec["required_symbols"]:
                if symbol.startswith("cudaq_algorithms.sim_utils."):
                    with self.subTest(case=spec["id"], helper=symbol):
                        self.assertIn(symbol.removeprefix("cudaq_algorithms."),
                                      spec["task"])

    def test_every_traced_requirement_discloses_its_public_api(self):
        aliases = {
            "cudaq_algorithms.trotter.Trotter.__init__":
            "cudaq_algorithms.trotter.Trotter",
            "cudaq_algorithms.fermion._compilers.jordan_wigner":
            "cudaq_algorithms.fermion.jordan_wigner",
            "cudaq_algorithms.fermion._compilers.bravyi_kitaev":
            "cudaq_algorithms.fermion.bravyi_kitaev",
        }
        for name in ("explicit_double_factorization",
                     "compressed_double_factorization", "reconstruct_eri",
                     "modified_one_body_integrals"):
            aliases["cudaq_algorithms.double_factorization._factorization." +
                    name] = ("cudaq_algorithms.double_factorization." + name)
        for spec in self.module().case_specs():
            with self.subTest(case=spec["id"]):
                self.assertEqual(spec.get("required_public_apis"), [
                    aliases.get(symbol, symbol)
                    for symbol in spec["required_symbols"]
                ])

    def test_schemas_and_hidden_inputs(self):
        c = self.module()
        self.assertEqual(len(c.case_specs()), 5)
        for spec in c.case_specs():
            p0, p1 = (c.parameters(spec["id"], v) for v in (0, 1))
            self.assertNotEqual(json.dumps(p0), json.dumps(p1))
            compile(c.reference_source(spec["id"]), spec["id"], "exec")
            for variant in (p0, p1):
                json.dumps(variant)
                if spec["id"] not in ("pyscf_energy", "psi4_energy"):
                    out = c.expected(spec["id"], variant)
                    self.assertEqual(set(out), set(spec["outputs"]))
                    self.assertTrue(
                        all(np.all(np.isfinite(v)) for v in out.values()))

    def test_oracles_do_not_load_evaluated_package(self):
        c = self.module()
        for case_id in ("trotter_dynamics", "fermion_transport",
                        "molecular_compression"):
            c.expected(case_id, c.parameters(case_id, 0))
        self.assertFalse(
            any(n.startswith("cudaq_algorithms") for n in sys.modules))

    @unittest.skipUnless(importlib.util.find_spec("pyscf"),
                         "independent PySCF oracle unavailable")
    def test_provider_reference_agrees_with_h2_physical_energy(self):
        c = self.module()
        out = c.expected("pyscf_energy", c.parameters("pyscf_energy", 0))
        self.assertAlmostEqual(out["nuclear_energy"], 1 / 1.4)
        self.assertAlmostEqual(out["ground_energy"], -1.1372759436, places=8)
        self.assertAlmostEqual(out["hf_energy"], -1.1167143251, places=8)
        self.assertLess(out["ground_energy"], out["hf_energy"] - .01)


@unittest.skipUnless(os.environ.get("E2E_RUNTIME_PYTHON"),
                     "supported runtime supplied by controller")
class GoldRuntimeContracts(unittest.TestCase):

    def test_real_gold_apps_on_public_and_hidden_inputs(self):
        import classical_cases as c
        runtime = os.environ["E2E_RUNTIME_PYTHON"]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT.parents[3] / "python")
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        for spec in c.case_specs():
            missing = subprocess.run([
                runtime, "-c",
                "import importlib.util,json; print(json.dumps([n for n in " +
                repr(spec["dependencies"]) +
                " if importlib.util.find_spec(n) is None]))"
            ],
                                     env=env,
                                     text=True,
                                     capture_output=True,
                                     check=True)
            if json.loads(missing.stdout.strip()):
                continue  # Controller records this task as infrastructure-blocked.
            for variant in (0, 1):
                with self.subTest(
                        case=spec["id"],
                        variant=variant), tempfile.TemporaryDirectory(
                            prefix="classical-gold-") as directory:
                    path = Path(directory)
                    (path / "app.py").write_text(c.reference_source(
                        spec["id"]))
                    (path / "profile_app.py").write_text(
                        "import sys,runpy,json\nseen=set()\n"
                        "def profile(frame,event,arg):\n"
                        "    if event == 'call':\n"
                        "        module=frame.f_globals.get('__name__','')\n"
                        "        if module.startswith('cudaq_algorithms'):\n"
                        "            seen.add(module+'.'+frame.f_code.co_qualname)\n"
                        "sys.setprofile(profile)\n"
                        "try:\n    runpy.run_path('app.py',run_name='__main__')\n"
                        "finally:\n    sys.setprofile(None)\n"
                        "    with open('profile.json','w') as handle:\n"
                        "        json.dump(sorted(seen),handle)\n")
                    params = c.parameters(spec["id"], variant)
                    (path / "input.json").write_text(json.dumps(params))
                    proc = subprocess.run([
                        runtime,
                        str(path / "profile_app.py"), "--input",
                        str(path / "input.json"), "--output",
                        str(path / "result.npz")
                    ],
                                          env=env,
                                          cwd=path,
                                          capture_output=True,
                                          text=True,
                                          timeout=120)
                    self.assertEqual(proc.returncode, 0,
                                     proc.stdout + proc.stderr)
                    seen = set(json.loads((path / "profile.json").read_text()))
                    self.assertTrue(
                        set(spec["required_symbols"]) <= seen,
                        sorted(set(spec["required_symbols"]) - seen))
                    expected = c.expected(spec["id"], params)
                    with np.load(path / "result.npz",
                                 allow_pickle=False) as actual:
                        self.assertEqual(set(actual.files), set(expected))
                        for field, want in expected.items():
                            np.testing.assert_allclose(actual[field],
                                                       want,
                                                       atol=spec["atol"],
                                                       rtol=spec["rtol"],
                                                       err_msg=field)


@unittest.skipUnless(os.environ.get("E2E_SANDBOX_PYTHON"),
                     "explicit restrictive sandbox runtime required")
class ProviderSandboxContracts(unittest.TestCase):

    def test_pyscf_bridge_runs_without_proc_memory_metadata(self):
        """Regresses the real PySCF /proc/<namespaced-pid>/statm failure."""
        import classical_cases as c
        import runtime
        interpreter = os.environ["E2E_SANDBOX_PYTHON"]
        with tempfile.TemporaryDirectory(
                prefix="classical-provider-sandbox-") as directory:
            for variant in (0, 1):
                with self.subTest(variant=variant):
                    work = Path(directory) / str(variant)
                    params = c.parameters("pyscf_energy", variant)
                    runtime.stage_workspace(ROOT.parents[3], work, "baseline",
                                            params)
                    (work / "app.py").write_text(
                        c.reference_source("pyscf_energy"))
                    command = runtime.sandbox_command(work, interpreter, [
                        work / "app.py", "--input", work / "input.json",
                        "--output", work / "result.npz"
                    ])
                    process = subprocess.run(command,
                                             cwd=work,
                                             env=runtime.environment(work),
                                             capture_output=True,
                                             text=True,
                                             timeout=60)
                    self.assertEqual(process.returncode, 0,
                                     process.stdout + process.stderr)
                    with np.load(work / "result.npz",
                                 allow_pickle=False) as actual:
                        for name, want in c.expected("pyscf_energy",
                                                     params).items():
                            np.testing.assert_allclose(actual[name],
                                                       want,
                                                       atol=2e-8,
                                                       rtol=2e-8)
                    isolation = runtime.isolation_probe(
                        work, interpreter, [
                            ROOT / "classical_cases.py",
                            runtime.SKILL / "SKILL.md"
                        ])
                    self.assertTrue(isolation["passed"], isolation)


if __name__ == "__main__":
    unittest.main()
