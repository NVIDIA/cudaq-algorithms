# End-to-end paired skill benchmark implementation plan

> For agentic workers: use subagent-driven-development with scoped implementation and independent review. User approved the replacement experiment on 2026-09-11. This file also records execution progress; it is not operational skill guidance.

## Goal and approved design

Evaluate real agent-authored applications across all nine documented families, with and without the unchanged skill. Numerical execution, independent references, held-out inputs, and token measurements are mandatory. The previous fixture pilot is historical smoke evidence only.

## Global constraints

- Maintained code and artifacts stay under `skills/cudaq-algorithms/evals/`; no new YAML, `.agents/` migration, skill-text changes, commits, pushes, or changes to unrelated files.
- Work in the existing `algo-skill` checkout. The user requested this location; agent execution itself uses separate temporary sandboxes. Do not create another worktree or modify Git metadata.
- Python >=3.11; supported CUDA-Q >=0.15,<0.16; NumPy/SciPy; CPU `qpp-cpu` fp64 for deterministic checks. Install only into a new task-owned temporary environment. Optional providers must execute genuinely or be reported as infrastructure-blocked.
- No mocked `cudaq_algorithms`, fake runtime, reference answers computed by the tested library, or pass for an unexecuted application.
- Both arms receive identical package source, ordinary docs/examples, inputs, tools and limits. Only the operational skill is added to the treatment. Evaluator implementation, hidden inputs, reference results, prior runs and original repository remain inaccessible.
- Target 15 tasks x 2 arms x 5 fresh attempts = 150 attempts, with randomized order and no stop-on-pass. Record infrastructure-blocked tasks without scoring them as model failures. No full-coverage claim if any family lacks a passed runtime preflight.
- Fix seeds and tolerances before runs. Each generated application is tested on visible input and at least one hidden input. Do not repair the model's application outside its measured attempt.
- Cumulative, cached and output tokens, observed request-input peaks, compactions and elapsed time are distinct. Missing measurements are null with a reason, never zero or context capacity. Validate measurement capture before the campaign.
- No credentials read, copied or logged. Native Codex login stays with the parent process. No model transport interception. Worker network is disabled.

## Shared interfaces and file ownership

The evaluator is a small Python package in this directory. Modules are invoked with this directory on the controller's module search path, never staged for task agents.

Case modules implement:

```python
def case_specs() -> list[dict]: ...
def parameters(case_id: str, variant: int) -> dict: ...
def expected(case_id: str, params: dict) -> dict[str, object]: ...
def reference_source(case_id: str) -> str: ...
```

Each spec contains `id`, `families`, `summary`, `task` (scientific task, not solution code), `dependencies` (import module names), `outputs` (NPZ field -> human-readable meaning), `atol`, `rtol`, `required_symbols` (qualified package function names for execution diagnostics), and optional `phase_invariant_fields`. `parameters` returns JSON-serializable scientific data; use separate `<name>_real` / `<name>_imag` arrays for complex data. Variants 0 and 1 are public and held-out data, respectively. `expected` returns NumPy-compatible scalar/array values and must not import the evaluated package. Where a molecular provider is itself an independent oracle, make the independence explicit.

Every model application and gold preflight application accepts:

```text
python app.py --input input.json --output result.npz
```

They must execute actual packaged primitives, write a non-object NPZ file, and work for either input variant without code edits. The gold application returned by `reference_source` is only for evaluator preflight and never visible to benchmark agents. A common bootstrap may parse args/read JSON/set CUDA-Q target, but do not create a fake package.

Case modules own their numerical contract tests. The controller owns the common runner, file staging, independent execution process, measurement collector, common result comparison, and report. Do not change another worker's files.

## Task 1: quantum family applications

Files: `quantum_cases.py`, `tests/test_quantum_cases.py`, `quantum-report.md`.

- [x] Implement ten task IDs: `slater_energy`, `pool_uccsd`, `pool_uccgsd`, `pool_upccgsd`, `pool_ceo`, `direct_uccsd`, `pauli_action`, `walk_spectrum`, `qsvt_filter`, `qsvt_recovery`.
- [x] Write independent oracle tests first: literal basis/determinant checks, Pauli Y/sign/endianness tests, known Chebyshev recurrence, and convention-sensitive phase tests. Tests must reject a known wrong phase/sign/normalization.
- [x] Implement numerical expectations without package-derived pools or package phase-response helpers. Fixed-pool and direct UCCSD conventions are distinct. Define deterministic public and hidden inputs that exercise nontrivial amplitudes.
- [x] Produce real-package preflight applications, with real `.py` kernels. QSVT phases may be supplied scientific input; phase synthesis is not a packaged API. Recovery uses its documented real input/Hamiltonian domain.
- [x] Validate CPU runtime applications once the controller provides the supported interpreter. Report any unsupported runtime path rather than changing the library.
- [x] Independent spec and numerical-quality review before model runs.

## Task 2: classical, chemistry, evolution applications

Files: `classical_cases.py`, `tests/test_classical_cases.py`, `classical-report.md`.

- [x] Implement five task IDs: `trotter_dynamics`, `fermion_transport`, `molecular_compression`, `pyscf_energy`, `psi4_energy`.
- [x] Write tests first for independent Pauli/Fock-space operators, BK occupation permutation, integral reconstruction and energy references. Include a nonzero identity shift and non-eigenstate dynamics.
- [x] Cover real Trotter execution and analysis, JW/BK representations, FCIDUMP loading, actual explicit and compressed DF, and separate real PySCF/Psi4 provider jobs.
- [x] Do not infer QSVT execution from the title of `df_compression_to_qsvt.py`; the current example does not execute QSVT. Example-only DF encodings are not packaged public classes.
- [x] Implement the shared case interfaces and gold applications. Missing provider modules produce explicit infrastructure blockers, not passing substitutes.
- [x] Independent spec and numerical-quality review before model runs.

## Task 3: token and context instrumentation

Files: `telemetry.py`, `tests/test_telemetry.py`, `telemetry-report.md`.

- [x] Preserve native authentication and the existing `exec --ignore-user-config` isolation if possible. Investigate invocation-only native OTel collection or an equivalently safe native event route.
- [x] Write literal event tests before code: cumulative snapshots are not summed, cache/reasoning subsets are not double-counted, missing fields are not zero, duplicate events/compactions are handled.
- [x] Expose a `TelemetryCollector` context manager with `.config_args` (list of Codex `-c` arguments), `.events` (sanitized metric events), and `.summary()` (JSON-serializable totals/observed peak/completeness/missing reasons). Do not log prompts, environment variables, authentication, or tool text in the telemetry collector.
- [x] If using local HTTP export, bind only 127.0.0.1 on an ephemeral port; accept only its metrics endpoint; close after the attempt. This is native diagnostic telemetry, not interception of model traffic.
- [x] Use one authorized native Codex calibration run with a real tool call. Cross-check cumulative counters against the normal final JSONL usage event. Report exactly which request-level fields are observed, including transport limitations.
- [x] Escalate a measurement blocker instead of inventing a peak from total input or model capacity.

## Task 4: controller, supported runtime and evaluation

Files: `run.py`, `runtime.py`, `grading.py`, `tests/test_runner.py`, `README.md`; generated runs under `../results/`.

- [x] Provision a supported task-local environment and record exact distribution versions and CPU target. Compile/execute a small kernel before evaluating agents.
- [x] Test all case parameter/output schemas and gold programs on both variants; numerical mismatches or missing dependencies gate only the affected tasks. Do not spend model calls on a known-broken case.
- [x] Unit-test fair staging and grading: baseline cannot read skill; both cannot read evaluator/gold/hidden inputs; mutation checks catch package/evidence edits; wrong numerical output fails even if process exit is zero; unavailable output fails without becoming zero.
- [x] Export identical package source/docs/tests into each sandbox without Git history or eval data; permit only task app/output paths to be written. Pin model `gpt-5.5`, low reasoning effort, and fixed per-attempt time limits. Disable other skills, user project docs, network and subagents.
- [x] Capture event data without text-buffer timestamp artifacts; validate actual command access boundaries before the campaign. Include token information from Task 3 and CLI cumulative counters.
- [x] Run agent application in an evaluator-owned sandbox on public and hidden inputs. Preserve code, result NPZ, stdout/stderr, invocation, timing, scope hashes, and grading results.
- [x] Report each family and subfeature, success rates, attempt totals, successful-solution timing separately from failure timing, token distributions, peak-observation coverage and infrastructure failures. Preserve raw failed attempts and do not select only successful repeats.
- [x] Run independent review of runner/isolation and case modules, fix material findings, then launch the ready full campaign and generate a self-contained report.

## Progress and decisions

- 2026-09-11: approved design captured; original skill unchanged at start. Existing unrelated `.superpowers/` and `docs/superpowers/` preserved.
- Ruling: use this user-requested `skills/` plan/ledger location and existing feature checkout; do not use skill defaults that create `.superpowers/`, worktrees, or commits. Cost if wrong: no separate Git branch for implementation; mitigated by scoped files and preserved source hashes.
- Ruling: independent case modules and telemetry may be implemented concurrently because their file ownership does not overlap. Integration is gated by the shared interfaces above and independent review.
- Numerical module preflights: twenty quantum and ten classical/provider applications passed outside the final restrictive sandbox. Independent numerical review corrected undisclosed helper requirements and removed analytic compression answers from inputs; all public API requirements are now disclosed identically to both arms.
- Controller review reproduced and fixed corrupt/linked/nonregular artifact handling, EOF timeout enforcement, timeout denominator classification, affected-task gating, interrupted-attempt preservation and evaluator hash enforcement. Source staging excludes planning documents as well as evaluator data and Git history.
- Telemetry review fixed partial/final accounting, peak completeness, compaction aliases and listener shutdown. A fresh final-controller native calibration executed one shell call, reported no unrelated startup skills, and recorded authoritative cumulative counts plus an explicitly unverified observed input peak. Compaction count is unavailable in this exec transport, not zero.
- Runtime ruling: private supported CUDA-Q 0.15.1 environments only. Psi4 1.10 required LibXC 7.0.0 to import. The final restrictive sandbox still prevents Psi4's MKL loader from finding its process executable. Four legitimate loader settings failed; legacy Landlock refuses this read-isolating profile. Do not weaken isolation or mock Psi4. Preserve its ten planned model slots as infrastructure-blocked if final preflight reproduces this.
- PySCF sandbox compatibility uses documented in-core integrals for these tiny molecules, disclosed in both prompts and gold programs. Both restrictive-sandbox variants and isolation probes passed after this setting; no provider/library monkeypatch was used.
- Campaign ordering uses four parallel paired lanes, serial execution inside each randomized pair, five repetitions and a 480-second per-attempt deadline. Native calibration is excluded from scientific attempts. No model campaign attempts have started at this checkpoint.
- Final fixed-code gate passed: four live isolation probes and twenty-eight numerical gold executions across fourteen tasks/all nine families. Only the two Psi4 gold variants remain infrastructure-blocked. Launched `results/e2e-20260911-full` with 140 runnable attempts and ten explicitly blocked planned slots; operational skill and package source remain unchanged.
- Campaign completed with controller exit 0: 150 finalized records, 140 actual attempts, ten unattempted Psi4 slots, no interrupted attempts. Both arms passed all 70 scientific/API checks and 64/70 frozen overall contracts; all twelve overall failures were extra bytecode-cache files. Each application passed its public and held-out executions (280 grading executions total, reusing 28 distinct scientific inputs).
- Final verification: 55 evaluator tests and 20 subtests passed, three optional runtime tests skipped; actual-runtime evidence is the campaign preflight and grading records, not those skips. All ten evaluator Python hashes still match the gold-certified version, and all 158 staged original package/doc/test/skill files match the unchanged workspace. Final native token counters are complete for all 140 attempts; request peaks are observed but unverified, and compaction counts remain unavailable.

| Interface check | Finding |
| --- | --- |
| Task 1 -> controller | Four functions and common spec/NPZ contract defined above; no shared source edits. |
| Task 2 -> controller | Same four functions; provider blockers represented explicitly. |
| Task 3 -> controller | Context manager supplies invocation args and sanitized metrics; authentication remains native. |
| Tasks 1/2 -> grading | Independent expected arrays, explicit tolerance and phase-invariant fields. |
| Runtime -> all tasks | Supported Python interpreter shared read-only after preflight. |
