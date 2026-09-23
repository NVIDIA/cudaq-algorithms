# Runtime and host-case review package

Baseline: pre-follow-up filesystem snapshot; no Git mutations.
Requirements/reports: psi4-completion-task.md, psi4-completion-report.md, classical-completion-report.md, EVALUATION-COMPLETION.md.

## runtime.py

```diff
--- before/runtime.py
+++ e2e/runtime.py
@@ -1,16 +1,17 @@
 """Fair source staging, restrictive native profiles and runtime metadata."""
 from __future__ import annotations
 
 import hashlib
 import json
 import os
 from pathlib import Path
 import shutil
+import stat
 import subprocess
 
 CODEX = Path("/home/.codex/packages/standalone/releases/0.144.4-x86_64-unknown-linux-musl/bin/codex")
 REPO = Path(__file__).resolve().parents[4]
 SKILL = REPO / "skills/cudaq-algorithms"
 MODEL = "gpt-5.5"
 PROFILE = "cudaq_e2e"
 EXCLUDE = {".git", ".agents", ".codex", "__pycache__", ".pytest_cache", "_build", "build", "AGENTS.md"}
@@ -64,22 +65,63 @@
     search = Path(rg_path) if rg_path is not None else source/"tools/rg"
     if search.is_file():
         (workspace/"tools").mkdir()
         shutil.copy2(search, workspace/"tools/rg")
     (workspace/"input.json").write_text(json.dumps(public, indent=2))
     (workspace/".tmp").mkdir()
 
 
+def stage_runtime_artifacts(workspace, case_id):
+    """Contain fixed provider side files without allowing new root artifacts."""
+    if case_id != "psi4_energy":
+        return
+    workspace = Path(workspace)
+    scratch = workspace/".tmp"
+    try:
+        info = scratch.lstat()
+        workspace_root = workspace.resolve(strict=True)
+        scratch_root = scratch.resolve(strict=True)
+    except OSError as exc:
+        raise ValueError("Psi4 scratch must be an existing workspace directory") from exc
+    if scratch.is_symlink() or not stat.S_ISDIR(info.st_mode) or scratch_root.parent != workspace_root:
+        raise ValueError("Psi4 scratch must be a real directory inside the workspace")
+    timer = workspace/"timer.dat"
+    if os.path.lexists(timer):
+        raise FileExistsError(f"refusing to replace existing runtime artifact: {timer}")
+    timer.symlink_to(".tmp/timer.dat")
+
+
+def _is_psi4_conda_runtime(python):
+    """Detect Psi4 from inert conda records in the interpreter's allowed prefix."""
+    metadata = Path(python).absolute().parent.parent/"conda-meta"
+    if metadata.is_symlink() or not metadata.is_dir():
+        return False
+    for record in metadata.glob("psi4-*.json"):
+        if record.is_symlink() or not record.is_file():
+            continue
+        try:
+            payload = json.loads(record.read_text())
+        except (OSError, ValueError, TypeError):
+            continue
+        if isinstance(payload, dict) and payload.get("name") == "psi4":
+            return True
+    return False
+
+
 def profile_args(workspace, python):
     workspace = Path(workspace).resolve()
     environment = Path(python).absolute().parent.parent
     permissions = {":root": "deny", ":minimal": "read", ":workspace_roots": "write",
                    str(CODEX): "read", str(REPO): "deny", str(environment): "read",
                    "/home/.codex/skills": "deny", "/home/.codex/plugins": "deny"}
+    if _is_psi4_conda_runtime(python):
+        # oneMKL locates its loader through /proc/self/exe. This also exposes
+        # the already-sanitized environment of this worker, never other PIDs.
+        permissions["/proc/self"] = "read"
     for name in ("python", "docs", "tests", "skills", "tools", "input.json", "pyproject.toml", "README.md", "LICENSE", "LICENSE.txt", "_trace_app.py"):
         if (workspace/name).exists():
             permissions[str(workspace/name)] = "read"
     table = "{" + ",".join(json.dumps(k)+"="+json.dumps(v) for k, v in permissions.items()) + "}"
     return ["-c", f"permissions.{PROFILE}.filesystem="+table,
             "-c", f"permissions.{PROFILE}.network.enabled=false"]
 
 
```

## run.py

```diff
--- before/run.py
+++ e2e/run.py
@@ -17,17 +17,18 @@
 import time
 
 import classical_cases
 import quantum_cases
 from grading import compare_output, required_calls_seen, read_regular
 from comparison import distribution, paired_comparisons, scientific_api_passed
 from runtime import (CODEX, MODEL, REPO, configuration, environment, inventory,
                      isolation_probe, runtime_metadata, sandbox_command,
-                     scope_changes, search_tool_metadata, stage_workspace)
+                     scope_changes, search_tool_metadata, stage_runtime_artifacts,
+                     stage_workspace)
 from telemetry import TelemetryCollector
 
 HERE = Path(__file__).resolve().parent
 CASES = {s["id"]: (module, s) for module in (quantum_cases, classical_cases) for s in module.case_specs()}
 SUITE = "historical_api_informed"
 
 
 def save(path, value):
@@ -112,16 +113,17 @@
     checks = []
     for variant in (0, 1):
         evidence = destination/str(variant)
         evidence.mkdir()
         params = module.parameters(spec["id"], variant)
         reference = module.expected(spec["id"], params)
         workspace = Path(tempfile.mkdtemp(prefix="cudaq-e2e-check-"))/"workspace"
         stage_workspace(source, workspace, "baseline", params)
+        stage_runtime_artifacts(workspace, spec["id"])
         (workspace/"app.py").write_text(app_source)
         shutil.copy2(HERE/"grading.py", workspace/"_trace_app.py")
         initial = inventory(workspace)
         cmd = sandbox_command(workspace, python, [workspace/"_trace_app.py", "--app", workspace/"app.py",
             "--input", workspace/"input.json", "--output", workspace/"result.npz",
             "--trace", workspace/".tmp/trace.json", "--package-root", workspace/"python/cudaq_algorithms"])
         started = time.monotonic()
         try:
@@ -336,16 +338,17 @@
     # A partial attempt is retained and marked interrupted, never retried.
     if destination.exists():
         return blocked_result(run, output, "interrupted prior attempt; evidence preserved", attempted=True)
     destination.mkdir(parents=True)
     source = Path(manifest["source_snapshot"])
     workspace = Path(manifest["staging_root"])/"runs"/run["id"]/"workspace"
     module, spec = CASES[run["case"]]
     stage_workspace(arm_source(manifest, run["arm"]), workspace, run["arm"], module.parameters(run["case"], 0))
+    stage_runtime_artifacts(workspace, run["case"])
     before = inventory(workspace)
     python = manifest["interpreters"][run["case"]]
     prompt = prompt_for(spec, python, run["arm"])
     (destination/"prompt.txt").write_text(prompt)
     save(destination/"initial-inventory.json", before)
     print(json.dumps({"started": run["id"]}), flush=True)
     start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
     with TelemetryCollector() as collector:
```

## tests/test_psi4_runtime.py

```diff
--- before/tests/test_psi4_runtime.py
+++ e2e/tests/test_psi4_runtime.py
@@ -0,0 +1,192 @@
+"""Psi4-only strict-runtime permissions and task-local side artifacts."""
+from __future__ import annotations
+
+import json
+import os
+from pathlib import Path
+import sys
+
+import pytest
+
+ROOT = Path(__file__).resolve().parents[1]
+sys.path.insert(0, str(ROOT))
+
+import classical_cases
+import run
+import runtime
+
+
+def _runtime(tmp_path: Path, name: str, record: dict | None = None) -> Path:
+    prefix = tmp_path / name
+    python = prefix / "bin/python"
+    python.parent.mkdir(parents=True)
+    python.write_text("")
+    if record is not None:
+        metadata = prefix / "conda-meta"
+        metadata.mkdir()
+        (metadata / "psi4-1.10-test_0.json").write_text(json.dumps(record))
+    return python
+
+
+def _filesystem_config(workspace: Path, python: Path) -> str:
+    args = runtime.profile_args(workspace, python)
+    return next(value for value in args if value.startswith("permissions.cudaq_e2e.filesystem="))
+
+
+def test_only_a_detected_psi4_conda_runtime_gets_process_self_read(tmp_path):
+    workspace = tmp_path / "workspace"
+    workspace.mkdir()
+    core = _runtime(tmp_path, "core")
+    wrong_record = _runtime(tmp_path, "wrong", {"name": "not-psi4", "version": "1.10"})
+    psi4 = _runtime(tmp_path, "psi4", {"name": "psi4", "version": "1.10"})
+
+    assert '"/proc/self"="read"' not in _filesystem_config(workspace, core)
+    assert '"/proc/self"="read"' not in _filesystem_config(workspace, wrong_record)
+    psi4_config = _filesystem_config(workspace, psi4)
+    assert '"/proc/self"="read"' in psi4_config
+    assert '"/proc"=' not in psi4_config
+    assert '"/sys"=' not in psi4_config
+
+
+def test_non_psi4_staging_adds_no_runtime_artifact(tmp_path):
+    workspace = tmp_path / "workspace"
+    (workspace / ".tmp").mkdir(parents=True)
+
+    runtime.stage_runtime_artifacts(workspace, "pyscf_energy")
+
+    assert sorted(path.name for path in workspace.iterdir()) == [".tmp"]
+
+
+def test_psi4_timer_is_fixed_inside_scratch_and_link_integrity_is_checked(tmp_path):
+    workspace = tmp_path / "workspace"
+    (workspace / ".tmp").mkdir(parents=True)
+    runtime.stage_runtime_artifacts(workspace, "psi4_energy")
+    timer = workspace / "timer.dat"
+    before = runtime.inventory(workspace)
+
+    assert timer.is_symlink()
+    assert os.readlink(timer) == ".tmp/timer.dat"
+    timer.write_text("provider timer\n")
+    assert runtime.scope_changes(before, runtime.inventory(workspace)) == []
+
+    timer.unlink()
+    timer.symlink_to(".tmp/repointed.dat")
+    assert runtime.scope_changes(before, runtime.inventory(workspace)) == ["timer.dat"]
+
+
+@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
+def test_psi4_timer_staging_rejects_every_existing_timer_path(tmp_path, kind):
+    workspace = tmp_path / "workspace"
+    (workspace / ".tmp").mkdir(parents=True)
+    timer = workspace / "timer.dat"
+    if kind == "file":
+        timer.write_text("owned by caller")
+    else:
+        timer.symlink_to("missing-target")
+
+    with pytest.raises(FileExistsError):
+        runtime.stage_runtime_artifacts(workspace, "psi4_energy")
+
+
+def test_psi4_timer_staging_rejects_scratch_symlink_escape(tmp_path):
+    workspace = tmp_path / "workspace"
+    outside = tmp_path / "outside"
+    workspace.mkdir()
+    outside.mkdir()
+    (workspace / ".tmp").symlink_to(outside, target_is_directory=True)
+
+    with pytest.raises(ValueError, match="scratch"):
+        runtime.stage_runtime_artifacts(workspace, "psi4_energy")
+    assert not (workspace / "timer.dat").exists()
+
+
+def test_run_one_stages_psi4_timer_before_its_initial_inventory(tmp_path, monkeypatch):
+    source = tmp_path / "source"
+    package = source / "python/cudaq_algorithms/__init__.py"
+    package.parent.mkdir(parents=True)
+    package.write_text("")
+    output = tmp_path / "output"
+    staging = tmp_path / "staging"
+    manifest = {
+        "source_snapshot": str(source),
+        "staging_root": str(staging),
+        "timeout_s": 1,
+        "interpreters": {"psi4_energy": sys.executable},
+        "model": "test-model",
+    }
+    record = {"id": "psi4_energy--1--baseline", "case": "psi4_energy",
+              "arm": "baseline", "repetition": 1}
+
+    def fake_native(command, workspace, prompt, destination, timeout, collector):
+        (Path(workspace) / "timer.dat").write_text("provider timer\n")
+        (Path(workspace) / "app.py").write_text("print('application')\n")
+        return {"returncode": 0, "timed_out": False, "elapsed_s": 0.,
+                "commands": 0, "tool_output_bytes": 0}
+
+    class FakeTelemetry:
+        config_args = []
+        events = []
+
+        def __enter__(self):
+            return self
+
+        def __exit__(self, *args):
+            return None
+
+        def summary(self):
+            return {"totals": {"input_tokens": 0, "output_tokens": 0}}
+
+    monkeypatch.setattr(run, "run_native", fake_native)
+    monkeypatch.setattr(run, "TelemetryCollector", FakeTelemetry)
+    monkeypatch.setattr(run, "execute_app",
+                        lambda *args, **kwargs: {"passed": True, "checks": []})
+
+    result = run.run_one(record, manifest, output)
+
+    initial = json.loads((output / "runs" / record["id"] /
+                          "initial-inventory.json").read_text())
+    assert initial["timer.dat"] == "symlink:.tmp/timer.dat"
+    assert result["scope_changes"] == []
+
+
+@pytest.mark.skipif(not os.environ.get("E2E_SANDBOX_PSI4_PYTHON"),
+                    reason="explicit restrictive Psi4 runtime required")
+def test_real_psi4_gold_passes_both_variants_in_strict_sandbox(tmp_path):
+    interpreter = os.environ["E2E_SANDBOX_PSI4_PYTHON"]
+    spec = run.CASES["psi4_energy"][1]
+
+    result = run.execute_app(runtime.REPO, spec,
+        classical_cases.reference_source("psi4_energy"), interpreter,
+        tmp_path / "checks", limit=180)
+
+    assert result["passed"], result
+    assert len(result["checks"]) == 2
+    assert all(check["scope_changes"] == [] for check in result["checks"])
+
+
+@pytest.mark.skipif(not os.environ.get("E2E_SANDBOX_PSI4_PYTHON"),
+                    reason="explicit restrictive Psi4 runtime required")
+def test_psi4_process_self_read_cannot_escape_isolation(tmp_path):
+    interpreter = os.environ["E2E_SANDBOX_PSI4_PYTHON"]
+    primary = tmp_path / "primary"
+    sibling = tmp_path / "sibling"
+    runtime.stage_workspace(runtime.REPO, primary, "baseline", {"probe": 1})
+    runtime.stage_workspace(runtime.REPO, sibling, "baseline", {"probe": 2})
+    evaluator = ROOT / "classical_cases.py"
+    forbidden = [
+        evaluator,
+        runtime.SKILL / "SKILL.md",
+        sibling / "input.json",
+        "/proc/1/environ",
+        "/proc/2/statm",
+        "/proc/self/root" + str(evaluator),
+        "/proc/self/root" + str(sibling / "input.json"),
+        "/proc/self/cwd/../sibling/input.json",
+    ]
+
+    result = runtime.isolation_probe(primary, interpreter, forbidden)
+
+    assert result["passed"], result
+    observed = json.loads(result["stdout"])
+    assert all(observed[str(path)] for path in forbidden)
+    assert observed["network_blocked"]
```

## extended_classical.py

```diff
--- before/extended_classical.py
+++ e2e/extended_classical.py
@@ -0,0 +1,286 @@
+"""Scoped host, simulation and raw Trotter outcomes missing from the first suites.
+
+References import no evaluated package. Private gold sources use real public
+APIs and are never staged for task agents. All state phases are absolute.
+"""
+import textwrap
+
+import numpy as np
+from scipy.linalg import expm
+
+from classical_cases import _chemist_fock
+from quantum_cases import pauli_matrix
+
+
+def case_specs():
+    prefix = "cudaq_algorithms."
+    common = " Pauli words list q0 first; q0 is the least significant statevector bit. Preserve absolute complex phase. "
+    data = [
+        ("trotter_device_surfaces", ["trotter"],
+         "Plan and execute the same product-state evolution through three public circuit surfaces.",
+         "Use make_trotter_terms to extract/prune terms at tolerance, reporting its original-order coefficients/words and identity separately. "
+         "Construct Trotter using TrotterOrdering(ordering). Prepare q0 with Ry(angles[0]) then Rz(phase), q1 with Ry(angles[1]). "
+         "Execute Trotter.kernel with injected preparation, Trotter.state_kernel with that cudaq.State, and raw apply_trotter inside a live-register kernel. "
+         "All three implement the specified order/steps/time product formula without the identity phase. Return their full states and the ordered plan. "
+         "Also execute raw apply_trotter with zero steps and report its unchanged state. "
+         "Report raw estimate_trotter_resources fields in order [num_terms,steps,order,pauli_rotations,estimated_cx_count,identity_coefficient], "
+         "both for the planned lists and after appending coefficient 0 with word II; the raw estimator counts supplied words without pruning. "
+         "Encode every word numerically with I=0,X=1,Y=2,Z=3. Return three host-rejection flags for NaN time, zero steps and order 3 passed to the factory.",
+         {"raw_coefficients": "extracted coefficients before ordering", "raw_words": "numeric word rows before ordering",
+          "planned_coefficients": "ordered coefficients", "planned_words": "numeric ordered word rows",
+          "identity_coefficient": "separate scalar identity", "injected_state": "injected factory state",
+          "state_input_state": "state-taking factory state", "raw_state": "direct raw kernel state",
+          "invalid_raw_state": "raw zero-step no-op state", "resource_fields": "six estimator fields",
+          "zero_probe_resources": "six raw estimator fields including zero word", "host_rejections": "three rejection flags"},
+         ["trotter.make_trotter_terms", "TrotterOrdering", "trotter.Trotter.kernel", "trotter.Trotter.state_kernel", "trotter.apply_trotter", "trotter.estimate_trotter_resources"],
+         ["trotter.make_trotter_terms", "trotter.Trotter.__init__", "trotter.Trotter.kernel", "trotter.Trotter.state_kernel", "trotter.estimate_trotter_resources"],
+         ["trotter.apply_trotter"],
+         "Two two-qubit product-state inputs, second/fourth-order formulas and two orderings; direct device and factory outcomes, pruning/identity, raw no-op and formula-level resources. Not controlled evolution or measured hardware cost."),
+        ("df_spin_diagnostics", ["chemistry", "double-factorization"],
+         "DF reconstruction diagnostics through spin expansion and a two-electron energy.",
+         "Construct DoubleFactorization from the supplied real orthogonal rotations and symmetric cores (method='C-DF'); these are data, not an optimizer task. "
+         "Reconstruct its chemist ERI, call factorization_error against target_eri and expand one_body plus the reconstructed ERI using spin_orbital_tensors. "
+         "Obtain the Fock-like eigenvalues of one_body minus half the contraction sum_r ERI[p,r,q,r] and compute double_factorization_one_norm in both lcu and burg conventions. "
+         "Build the Jordan-Wigner qubit Hamiltonian of one_body/reconstructed ERI with no scalar offset; report its lowest energy in the exactly two-electron sector. "
+         "Return five diagnostics: reject rank-1 one_body, mismatched ERI size, an ERI with only [0,1,0,0] increased by .2, and norm convention='unknown'; "
+         "then report that a square nonsymmetric one_body (only [0,1] increased by .2) is accepted by spin_orbital_tensors. "
+         "Use validation defaults. Norms are named formula-level Hamiltonian normalizations, not gate/runtime bounds.",
+         {"reconstructed_eri": "reconstructed spatial ERI", "one_body_so": "interleaved-spin one-body tensor",
+          "two_body_so": "ladder coefficients including half factor", "residual": "absolute Frobenius residual",
+          "lcu_norm": "LCU formula norm", "burg_norm": "gauge-fixed eigenfactor formula norm",
+          "two_electron_energy": "lowest fixed-sector energy", "diagnostics": "five ordered boundary flags"},
+         ["chemistry.spin_orbital_tensors", "double_factorization.factorization_error", "double_factorization.double_factorization_one_norm"],
+         ["chemistry.spin_orbital_tensors", "double_factorization._factorization.factorization_error", "double_factorization._factorization.double_factorization_one_norm"], [],
+         "Two real two-orbital/two-leaf fixtures; spin-index/half-factor tensors, nonzero residual, both named norm formulas and fixed-sector energy. Explicit validation boundaries; no optimizer robustness, encoding or hardware-resource claim."),
+        ("good_subspace_layout", ["simulation", "block-encoding"],
+         "Extract a real encoded complex state and verify layout, copying and normalization boundaries.",
+         "Build PauliLCU from terms, prepare ket_real+i ket_imag, and execute its encode_kernel using an actual cudaq.State. "
+         "Call sim_utils.good_subspace directly on the full state. Return its unnormalized block, good_probability=norm(block)^2, "
+         "bad_probability from the complementary full-state amplitudes, and full_norm. Also extract from the full array multiplied by scale_real+i scale_imag; "
+         "return scaled_block and its block weight, which need not be a probability. "
+         "Verify that mutating a returned block does not mutate the full array (copy_independent), and reject a column array, a one-element-short array, and a one-element-long array (three shape_rejections).",
+         {"block": "unnormalized good block", "good_probability": "zero-ancilla probability",
+          "bad_probability": "complement probability", "full_norm": "full squared norm",
+          "scaled_block": "block from scaled full input", "scaled_block_weight": "scaled block squared norm",
+          "copy_independent": "copy/nonalias flag", "shape_rejections": "three rejection flags"},
+         ["sim_utils.good_subspace", "PauliLCU"],
+         ["sim_utils.good_subspace", "pauli_lcu.PauliLCU.__init__"], ["pauli_lcu.apply"],
+         "Two two-qubit complex-input LCU circuits; exact good-block layout/phase, complementary probability, nonunit-input block weight, copying and shape rejection. Statevector-only, not shots or QPU execution."),
+    ]
+    return [dict(id=cid, families=families, summary=summary, task=task+common, outputs=outputs,
+                 required_public_apis=[prefix+s for s in public], required_symbols=[prefix+s for s in host],
+                 required_kernels=[prefix+s for s in device], atol=2e-9, rtol=2e-9,
+                 dependencies=["cudaq", "cudaq_algorithms", "numpy", "scipy"], coverage_scope=scope)
+            for cid, families, summary, task, outputs, public, host, device, scope in data]
+
+
+def parameters(case_id, variant):
+    if variant not in (0, 1):
+        raise ValueError("variant must be public 0 or held-out 1")
+    if case_id == "trotter_device_surfaces":
+        return {"terms": [[.3-.38*variant, "II"], [.61-1.14*variant, "XI"],
+                          [-.44+1.21*variant, "ZZ"], [.13-.41*variant, "IY"],
+                          [1e-10, "YX"], [0., "XX"], [-.1, "II"]],
+                "tolerance": 1e-8, "ordering": ("preserve_input", "coefficient_magnitude_descending")[variant],
+                "time": .72+.21*variant, "steps": 2+variant, "order": (2, 4)[variant],
+                "angles": [.47+.16*variant, -.62+.27*variant], "phase": .33-.61*variant}
+    if case_id == "df_spin_diagnostics":
+        rotations = [np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
+                     for t in (.31+.12*variant, -.43+.19*variant)]
+        cores = [np.array([[1.2+.2*variant, -.24], [-.24, .6-.11*variant]]),
+                 np.array([[.27, .19+.05*variant], [.19+.05*variant, -.14]])]
+        p = {"rotations": [r.tolist() for r in rotations], "cores": [c.tolist() for c in cores],
+             "one_body": [[-.7-.1*variant, .16], [.16, -.22+.07*variant]]}
+        eri = reconstruct(p)
+        perturbation = np.array([[.7, -.2], [-.2, .4]])
+        p["target_eri"] = (eri + (.017+.009*variant)*np.einsum("pq,rs->pqrs", perturbation, perturbation)).tolist()
+        return p
+    if case_id == "good_subspace_layout":
+        ket = np.array([.4+.1j, -.2+.3j, .7-.05*variant, -.1-.2j*(1+variant)])
+        ket /= np.linalg.norm(ket)
+        return {"terms": [[.23, "II"], [-.65+.17*variant, "XI"], [.41, "YZ"], [.18+.06*variant, "IZ"]],
+                "ket_real": ket.real.tolist(), "ket_imag": ket.imag.tolist(),
+                "scale_real": 1.8+.2*variant, "scale_imag": -.4+.1*variant}
+    raise ValueError("unknown case: "+case_id)
+
+
+def reconstruct(p):
+    """Sum outer products of orbital projectors, independently of package einsum."""
+    n = len(p["one_body"])
+    eri = np.zeros((n,)*4)
+    for rotation, core in zip(p["rotations"], p["cores"]):
+        u = np.asarray(rotation)
+        for k in range(n):
+            left = np.outer(u[:, k], u[:, k])
+            for l in range(n):
+                right = np.outer(u[:, l], u[:, l])
+                eri += core[k][l]*left[:, :, None, None]*right[None, None, :, :]
+    return eri
+
+
+def expected(case_id, p):
+    if case_id == "trotter_device_surfaces":
+        pairs = [(c, w) for c, w in p["terms"] if c != 0 and abs(c) >= p["tolerance"]]
+        identity = sum(c for c, w in pairs if set(w) == {"I"})
+        raw = [(c, w) for c, w in pairs if set(w) != {"I"}]
+        planned = sorted(raw, key=lambda item: abs(item[0]), reverse=True) if p["ordering"] == "coefficient_magnitude_descending" else raw
+        a, b = p["angles"]; phase = p["phase"]
+        initial = np.kron([np.cos(b/2), np.sin(b/2)],
+                          [np.cos(a/2)*np.exp(-.5j*phase), np.sin(a/2)*np.exp(.5j*phase)])
+        state = initial.copy(); dt = p["time"]/p["steps"]
+        weight = 1/(2-2**(1/3))
+        for _ in range(p["steps"]):
+            if p["order"] == 1:
+                sequence = [(dt, planned)]
+            else:
+                fractions = [1.] if p["order"] == 2 else [weight, 1-2*weight, weight]
+                sequence = [(dt*f/2, terms) for f in fractions for terms in (planned, list(reversed(planned)))]
+            for t, terms in sequence:
+                for c, word in terms:
+                    state = expm(-1j*t*c*pauli_matrix(word)) @ state
+        scale = p["steps"]*{1: 1, 2: 2, 4: 6}[p["order"]]
+        cx = sum(2*max(0, sum(letter != "I" for letter in w)-1) for _, w in planned)*scale
+        fields = [len(planned), p["steps"], p["order"], len(planned)*scale, cx, identity]
+        zero = [len(planned)+1, p["steps"], p["order"], (len(planned)+1)*scale, cx, identity]
+        words = lambda terms: np.array([["IXYZ".index(ch) for ch in w] for _, w in terms], dtype=int)
+        return {"raw_coefficients": np.array([c for c, _ in raw]), "raw_words": words(raw),
+                "planned_coefficients": np.array([c for c, _ in planned]), "planned_words": words(planned),
+                "identity_coefficient": identity, "injected_state": state, "state_input_state": state,
+                "raw_state": state, "invalid_raw_state": initial, "resource_fields": np.array(fields),
+                "zero_probe_resources": np.array(zero), "host_rejections": np.ones(3, int)}
+    if case_id == "df_spin_diagnostics":
+        one = np.asarray(p["one_body"]); eri = reconstruct(p); n = len(one); m = 2*n
+        one_so = np.array([[one[a//2, b//2] if a % 2 == b % 2 else 0 for b in range(m)] for a in range(m)], complex)
+        two_so = np.zeros((m,)*4, complex)
+        for a, b, c, d in np.ndindex(two_so.shape):
+            if a % 2 == d % 2 and b % 2 == c % 2:
+                two_so[a, b, c, d] = .5*eri[a//2, d//2, b//2, c//2]
+        fock_eigs = np.linalg.eigvalsh(one - .5*np.einsum("prqr->pq", eri))
+        lcu = burg = float(sum(abs(fock_eigs)))
+        for core in p["cores"]:
+            z = np.asarray(core)
+            lcu += sum(abs(z[k, l]) for k in range(n) for l in range(k+1, n)) + .25*sum(abs(z.diagonal()))
+            eigenvalues, vectors = np.linalg.eigh(z)
+            burg += .25*sum(abs(v)*sum(abs(vectors[:, k]))**2 for k, v in enumerate(eigenvalues))
+        h = _chemist_fock(one, eri, 0.)
+        sector = [i for i in range(1 << m) if i.bit_count() == 2]
+        return {"reconstructed_eri": eri, "one_body_so": one_so, "two_body_so": two_so,
+                "residual": float(np.linalg.norm(np.asarray(p["target_eri"])-eri)),
+                "lcu_norm": lcu, "burg_norm": burg,
+                "two_electron_energy": float(np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0]),
+                "diagnostics": np.ones(5, int)}
+    if case_id == "good_subspace_layout":
+        ket = np.asarray(p["ket_real"])+1j*np.asarray(p["ket_imag"])
+        alpha = sum(abs(c) for c, _ in p["terms"])
+        h = sum(c*pauli_matrix(w) for c, w in p["terms"])
+        block = h @ ket / alpha
+        probability = float(np.vdot(block, block).real)
+        scaled = (p["scale_real"]+1j*p["scale_imag"])*block
+        return {"block": block, "good_probability": probability, "bad_probability": 1-probability,
+                "full_norm": 1., "scaled_block": scaled, "scaled_block_weight": float(np.vdot(scaled, scaled).real),
+                "copy_independent": 1, "shape_rejections": np.ones(3, int)}
+    raise ValueError("unknown case: "+case_id)
+
+
+def reference_source(case_id):
+    header = '''
+import argparse, json
+import numpy as np
+import cudaq
+cudaq.set_target("qpp-cpu")
+parser = argparse.ArgumentParser()
+parser.add_argument("--input", required=True)
+parser.add_argument("--output", required=True)
+args = parser.parse_args()
+with open(args.input) as handle:
+    p = json.load(handle)
+out = {}
+def rejected(call):
+    try:
+        call()
+    except (ValueError, TypeError):
+        return 1
+    return 0
+'''
+    bodies = {
+        "trotter_device_surfaces": '''
+from cudaq_algorithms import trotter, sim_utils, TrotterOrdering
+coefficients, words, identity, n = trotter.make_trotter_terms(p["terms"], p["tolerance"])
+evolution = trotter.Trotter(p["terms"], ordering=TrotterOrdering(p["ordering"]), coefficient_tolerance=p["tolerance"])
+a, b = map(float, p["angles"])
+phase = float(p["phase"])
+@cudaq.kernel
+def prep(q: cudaq.qview):
+    ry(a, q[0])
+    rz(phase, q[0])
+    ry(b, q[1])
+@cudaq.kernel
+def initial():
+    q = cudaq.qvector(2)
+    prep(q)
+@cudaq.kernel
+def raw(c: list[float], w: list[cudaq.pauli_word], t: float, steps: int, order: int):
+    q = cudaq.qvector(2)
+    prep(q)
+    trotter.apply_trotter(c, w, t, steps, order, q)
+t, steps, order = float(p["time"]), int(p["steps"]), int(p["order"])
+initial_state = cudaq.get_state(initial)
+out["injected_state"] = np.asarray(cudaq.get_state(evolution.kernel(t, steps, order, state_prep=prep)))
+out["state_input_state"] = np.asarray(cudaq.get_state(evolution.state_kernel(t, steps, order), initial_state))
+out["raw_state"] = np.asarray(cudaq.get_state(raw, evolution.coefficients, evolution.words, t, steps, order))
+out["invalid_raw_state"] = np.asarray(cudaq.get_state(raw, evolution.coefficients, evolution.words, t, 0, order))
+encode = lambda ws: np.asarray([["IXYZ".index(ch) for ch in w] for w in ws], dtype=int)
+out.update(raw_coefficients=np.asarray(coefficients), raw_words=encode(words),
+           planned_coefficients=np.asarray(evolution.coefficients), planned_words=encode(evolution.words), identity_coefficient=identity)
+fields = lambda r: np.asarray([r.num_terms, r.steps, r.order, r.pauli_rotations, r.estimated_cx_count, r.identity_coefficient])
+out["resource_fields"] = fields(trotter.estimate_trotter_resources(evolution.coefficients, evolution.words, steps, order, identity))
+out["zero_probe_resources"] = fields(trotter.estimate_trotter_resources(evolution.coefficients+[0.], evolution.words+["II"], steps, order, identity))
+out["host_rejections"] = np.asarray([rejected(lambda: evolution.kernel(float("nan"), steps, order)),
+    rejected(lambda: evolution.kernel(t, 0, order)), rejected(lambda: evolution.kernel(t, steps, 3))])
+''',
+        "df_spin_diagnostics": '''
+from cudaq_algorithms import chemistry
+from cudaq_algorithms import double_factorization as df
+one = np.asarray(p["one_body"])
+n = len(one)
+factor = df.DoubleFactorization(n, [np.asarray(r) for r in p["rotations"]], [np.asarray(z) for z in p["cores"]], "C-DF")
+eri = df.reconstruct_eri(factor)
+out["reconstructed_eri"] = eri
+out["one_body_so"], out["two_body_so"] = chemistry.spin_orbital_tensors(one, eri)
+out["residual"] = df.factorization_error(np.asarray(p["target_eri"]), factor)
+values = np.linalg.eigvalsh(df.modified_one_body_integrals(one, eri))
+out["lcu_norm"] = df.double_factorization_one_norm(factor, values, "lcu")
+out["burg_norm"] = df.double_factorization_one_norm(factor, values, "burg")
+h = np.asarray(chemistry.qubit_hamiltonian(one, eri).to_matrix())
+sector = [i for i in range(1 << (2*n)) if i.bit_count() == 2]
+out["two_electron_energy"] = np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0]
+bad = eri.copy(); bad[0, 1, 0, 0] += .2
+asymmetric = one.copy(); asymmetric[0, 1] += .2
+out["diagnostics"] = np.asarray([rejected(lambda: chemistry.spin_orbital_tensors(one.ravel(), eri)),
+    rejected(lambda: chemistry.spin_orbital_tensors(one, eri[:-1])),
+    rejected(lambda: chemistry.spin_orbital_tensors(one, bad)),
+    rejected(lambda: df.double_factorization_one_norm(factor, values, "unknown")),
+    1-rejected(lambda: chemistry.spin_orbital_tensors(asymmetric, eri))])
+''',
+        "good_subspace_layout": '''
+from cudaq_algorithms import PauliLCU, sim_utils
+enc = PauliLCU(p["terms"])
+ket = np.asarray(p["ket_real"])+1j*np.asarray(p["ket_imag"])
+full = np.asarray(cudaq.get_state(enc.encode_kernel(), sim_utils.state_from(ket)), dtype=np.complex128)
+block = sim_utils.good_subspace(enc, full)
+scaled = sim_utils.good_subspace(enc, full*(p["scale_real"]+1j*p["scale_imag"]))
+probe = sim_utils.good_subspace(enc, full)
+before = full.copy(); probe[0] += .3+.2j
+out.update(block=block, good_probability=np.vdot(block, block).real,
+    bad_probability=np.vdot(full[len(block):], full[len(block):]).real,
+    full_norm=np.vdot(full, full).real, scaled_block=scaled,
+    scaled_block_weight=np.vdot(scaled, scaled).real,
+    copy_independent=int(not np.shares_memory(probe, full) and np.array_equal(before, full)))
+out["shape_rejections"] = np.asarray([rejected(lambda: sim_utils.good_subspace(enc, full[:, None])),
+    rejected(lambda: sim_utils.good_subspace(enc, full[:-1])),
+    rejected(lambda: sim_utils.good_subspace(enc, np.append(full, 0.)))])
+''',
+    }
+    if case_id not in bodies:
+        raise ValueError("unknown case: "+case_id)
+    return textwrap.dedent(header+bodies[case_id]+"\nnp.savez(args.output, **out)\n")
```

## tests/test_extended_classical.py

```diff
--- before/tests/test_extended_classical.py
+++ e2e/tests/test_extended_classical.py
@@ -0,0 +1,95 @@
+"""Independent contracts for the remaining host/simulation/Trotter surfaces."""
+import importlib
+import json
+import os
+from pathlib import Path
+import sys
+
+import numpy as np
+import pytest
+
+sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
+
+
+def module():
+    return importlib.import_module("extended_classical")
+
+
+def test_trotter_oracle_pins_phase_pruning_and_raw_resource_semantics():
+    p = {"terms": [[0.25, "II"], [0.5, "XI"], [1e-10, "ZZ"]],
+         "tolerance": 1e-8, "ordering": "preserve_input", "time": 0.6,
+         "steps": 1, "order": 1, "angles": [0., 0.], "phase": 0.}
+    out = module().expected("trotter_device_surfaces", p)
+    np.testing.assert_allclose(out["raw_coefficients"], [0.5])
+    np.testing.assert_array_equal(out["raw_words"], [[1, 0]])
+    np.testing.assert_allclose(out["identity_coefficient"], 0.25)
+    for field in ("injected_state", "state_input_state", "raw_state"):
+        np.testing.assert_allclose(out[field], [np.cos(.3), -1j*np.sin(.3), 0, 0], atol=1e-14)
+    np.testing.assert_allclose(out["invalid_raw_state"], [1, 0, 0, 0])
+    np.testing.assert_allclose(out["resource_fields"], [1, 1, 1, 1, 0, .25])
+    # The raw estimator counts the appended zero/identity word too.
+    np.testing.assert_allclose(out["zero_probe_resources"], [2, 1, 1, 2, 0, .25])
+
+
+def test_df_norm_spin_and_residual_have_literal_independent_reference():
+    target = np.zeros((2, 2, 2, 2)); target[0, 0, 0, 0] = 2.4
+    p = {"rotations": [np.eye(2).tolist()], "cores": [[[2., 0.], [0., 0.]]],
+         "one_body": [[1.5, 0.], [0., .3]], "target_eri": target.tolist()}
+    out = module().expected("df_spin_diagnostics", p)
+    assert out["residual"] == pytest.approx(.4)
+    assert out["lcu_norm"] == pytest.approx(1.3)
+    assert out["burg_norm"] == pytest.approx(1.3)
+    assert out["two_electron_energy"] == pytest.approx(.6)
+    np.testing.assert_allclose(out["one_body_so"], np.diag([1.5, 1.5, .3, .3]))
+    assert out["two_body_so"][0, 1, 1, 0] == 1.
+    assert out["two_body_so"][0, 1, 0, 1] == 0.
+    assert np.count_nonzero(out["two_body_so"]) == 4
+
+
+def test_good_subspace_keeps_complex_phase_and_nonunit_block_weight():
+    p = {"terms": [[.5, "I"], [-1.5, "Y"]], "ket_real": [1., 0.],
+         "ket_imag": [0., 0.], "scale_real": 2., "scale_imag": 0.}
+    out = module().expected("good_subspace_layout", p)
+    np.testing.assert_allclose(out["block"], [.25, -.75j])
+    np.testing.assert_allclose(out["scaled_block"], [.5, -1.5j])
+    assert out["good_probability"] == pytest.approx(.625)
+    assert out["bad_probability"] == pytest.approx(.375)
+    assert out["scaled_block_weight"] == pytest.approx(2.5)
+
+
+def test_new_cases_have_distinct_inputs_and_wrong_outcomes_fail(tmp_path):
+    from grading import compare_output, required_calls_seen
+    for spec in module().case_specs():
+        assert spec["required_public_apis"] and spec["required_symbols"]
+        if spec["id"] != "df_spin_diagnostics":
+            assert spec["required_kernels"]
+            assert not required_calls_seen(spec["required_kernels"], [])["passed"]
+        results = []
+        for variant in (0, 1):
+            params = module().parameters(spec["id"], variant)
+            json.dumps(params, allow_nan=False)
+            expected = module().expected(spec["id"], params)
+            assert set(expected) == set(spec["outputs"])
+            assert all(np.isfinite(v).all() for v in expected.values())
+            results.append(expected)
+            path = tmp_path/"result.npz"
+            np.savez(path, **expected)
+            assert compare_output(path, expected, spec)["passed"]
+            for key in expected:
+                wrong = dict(expected); wrong[key] = np.asarray(expected[key]) + .2
+                np.savez(path, **wrong)
+                assert not compare_output(path, expected, spec)["passed"], key
+        assert any(not np.allclose(results[0][k], results[1][k]) for k in results[0])
+    with pytest.raises(ValueError):
+        module().parameters("df_spin_diagnostics", 2)
+
+
+@pytest.mark.skipif(not os.environ.get("CUDAQ_E2E_PYTHON"), reason="supported runtime required")
+@pytest.mark.parametrize("case_id", ["trotter_device_surfaces", "df_spin_diagnostics", "good_subspace_layout"])
+def test_real_extended_gold_on_both_inputs(tmp_path, monkeypatch, case_id):
+    import run
+    cases = {s["id"]: (module(), s) for s in module().case_specs()}
+    monkeypatch.setattr(run, "CASES", cases)
+    result = run.execute_app(run.REPO, cases[case_id][1], module().reference_source(case_id),
+                             os.environ["CUDAQ_E2E_PYTHON"], tmp_path/"grading")
+    assert result["passed"], result
```

## extended.py

```diff
--- before/extended.py
+++ e2e/extended.py
@@ -0,0 +1,33 @@
+"""Run the separately scored scientific coverage-completion campaign.
+
+Usage: extended.py prepare|preflight|run|report --output PATH.
+The historical and natural-language campaign registries remain unchanged.
+"""
+import classical_cases
+import extended_classical
+import extended_stateprep
+import run
+
+SUITE = "coverage_completion"
+
+
+def case_registry():
+    cases = {spec["id"]: (module, spec)
+             for module in (extended_stateprep, extended_classical)
+             for spec in module.case_specs()}
+    psi4 = next(spec for spec in classical_cases.case_specs() if spec["id"] == "psi4_energy")
+    cases[psi4["id"]] = (classical_cases, psi4)
+    return cases
+
+
+def main(argv=None):
+    original, label = run.CASES, run.SUITE
+    try:
+        run.CASES, run.SUITE = case_registry(), SUITE
+        run.main(argv)
+    finally:
+        run.CASES, run.SUITE = original, label
+
+
+if __name__ == "__main__":
+    main()
```

## tests/test_extended_suite.py

```diff
--- before/tests/test_extended_suite.py
+++ e2e/tests/test_extended_suite.py
@@ -0,0 +1,45 @@
+"""The coverage completion suite is isolated from historical experiments."""
+import importlib
+import json
+from pathlib import Path
+import sys
+
+import pytest
+
+sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
+
+
+def test_completion_cli_keeps_historical_registry_and_paired_staging(tmp_path, monkeypatch):
+    import run
+    from test_runner import fixture_source
+    suite = importlib.import_module("extended")
+    source = fixture_source(tmp_path/"source", "skill")
+    monkeypatch.setattr(run, "REPO", source)
+    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **kw: "revision\n")
+    monkeypatch.setattr(run, "runtime_metadata", lambda p: {"executable": p})
+    before, label = dict(run.CASES), run.SUITE
+    destination = tmp_path/"campaign"
+    suite.main(["prepare", "--output", str(destination), "--python", sys.executable,
+                "--repetitions", "1", "--cases", "df_spin_diagnostics,psi4_energy"])
+    manifest = json.loads((destination/"manifest.json").read_text())
+    assert manifest["suite"] == "coverage_completion"
+    assert len(manifest["runs"]) == 4
+    assert {r["arm"] for r in manifest["runs"]} == {"baseline", "skill"}
+    assert {c["id"] for c in manifest["cases"]} == {"df_spin_diagnostics", "psi4_energy"}
+    assert run.CASES == before and run.SUITE == label
+    with pytest.raises(RuntimeError, match="suite"):
+        run.main(["report", "--output", str(destination)])
+    suite.main(["report", "--output", str(destination)])
+    assert run.CASES == before and run.SUITE == label
+
+
+def test_every_uncovered_record_has_a_direct_scientific_contract():
+    suite = importlib.import_module("extended")
+    root = Path(__file__).resolve().parents[3]
+    registry = json.loads((root/"coverage/features.json").read_text())
+    required = {symbol for _, spec in suite.case_registry().values() for symbol in spec["required_public_apis"]}
+    for feature in registry["features"]:
+        has_old_current_evidence = any(e["campaign"].endswith("refactored") and e["status"] == "scientific_pass"
+                                       for e in feature["executions"])
+        if not has_old_current_evidence:
+            assert required.intersection(feature["symbols"]), feature["id"]
```
