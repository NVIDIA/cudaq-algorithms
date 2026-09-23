# End-to-end skill benchmark

This evaluator asks fresh Codex sessions to write and run applications using
the actual CUDA-Q Algorithms package. It compares a candidate operational
skill against identical source, documentation, tests, runtime and task inputs
without that skill, optionally including a preserved previous skill as a third
arm. It is **not NVIDIA SkillEvaluator** and does not need an
API key beyond the existing native Codex login.

## Scope

Fifteen small-system applications cover the nine catalog families. Every
application receives one public input and is independently executed again on
a held-out input. Numerical references use independent NumPy/SciPy algebra or
PySCF FCI rather than the evaluated package. Gold applications validate the
runtime and reference contracts before any model attempts; they are never
shown to task agents.

Required public entry points are disclosed in every arm. This intentionally
tests API-informed implementation and scientific correctness, not unaided
keyword routing. Five fresh sessions per task/arm give 150 planned attempts
with two arms or 225 with three. The fixed seed randomizes case/repetition
blocks and the three-arm starting order, not model sampling. Rotating arm
order balances each position within one block across the campaign. Each
block runs its arms serially; `--parallel` limits concurrent blocks. Repetitions reuse
each task's public/held-out input pair; they are not new scientific instances.
This is not a claim of coverage of every leaf or large-system path.
For detailed boundaries, see [quantum-report.md](quantum-report.md) and
[classical-report.md](classical-report.md).

## Running

Use Python >=3.11, CUDA-Q >=0.15,<0.16, NumPy, SciPy, PySCF and pytest. The
Psi4 case additionally needs a real Psi4 installation. The tested temporary
runtime uses CUDA-Q 0.15.1 and CPU `qpp-cpu` fp64. Psi4 1.10 required LibXC
7.0.0; LibXC 7.1.2 broke provider import. Do not alter the repository package
or substitute a fake provider to make a preflight pass.

The current controller pins the installed native Codex CLI path in
`runtime.py` and model `gpt-5.5`, low reasoning. Adjust installation paths
before preparing a new campaign, never during an existing one.

From the repository root, with your supported interpreter replacing
`/path/to/python`:

```bash
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/run.py prepare \
  --output skills/cudaq-algorithms-dev/evals/results/my-campaign \
  --python /path/to/python --psi4-python /path/to/psi4/python \
  --previous-skill skills/cudaq-algorithms-dev/evals/results/refactor-20260911/previous-skill \
  --rg /path/to/real/rg \
  --repetitions 5 --parallel 2 --timeout 480
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/run.py preflight \
  --output skills/cudaq-algorithms-dev/evals/results/my-campaign
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/run.py run \
  --output skills/cudaq-algorithms-dev/evals/results/my-campaign
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/run.py report \
  --output skills/cudaq-algorithms-dev/evals/results/my-campaign
```

`--previous-skill` must name the preserved skill root containing `SKILL.md`.
It enables `baseline` (no skill), `previous` and `skill` (candidate). Both
with-skill arms use exactly the same prompt and `skills/cudaq-algorithms/`
path. Omit the option for the existing two-arm workflow. Stage application
content only: `SKILL.md`, `references/` and existing `assets/`; never stage
`authoring/`, `coverage/`, `scripts/`, `evals/` or previous result evidence.

`--rg` selects a real installed ripgrep executable; omitting it uses PATH
discovery and records null if none is available. The controller copies the
selected binary to read-only `tools/rg` in every workspace, prepends that
directory to PATH and records its original path, worker path, full version
and SHA-256. No global installation is performed. The native CLI bundles
ripgrep in its own release directory (`codex-path/rg` next to the `codex`
binary), and that copy is the one selected.

Native authentication and parent-to-model transport need ordinary process
access. In a restricted outer container, authorize the controller execution;
the controller still enforces its own worker sandbox with network disabled.
The tested worker sandbox lacks `/proc/<pid>/statm`, which PySCF's default
memory check queries. These tiny molecular cases use `mol.incore_anyway=True`
and quiet logging, disclosed identically in both prompts and gold programs.
No provider is monkeypatched. Preflight tests actual provider execution using
the worker permission profile. Earlier campaigns recorded a Psi4 MKL loader
blocker. The repaired profile allows `/proc/self` reads only for detected
Psi4 conda runtimes, enabling oneMKL's executable lookup. It still denies other
processes and external workspaces. The sanitized self environment is readable;
API credentials are not added. A checked `timer.dat -> .tmp/timer.dat` symlink
contains Psi4's fixed timing output without expanding the file-change
allowlist. Replacing or repointing that link remains a scope failure. See the
[repair evidence](../psi4-completion-report.md). Historical blocked results
are retained, not retroactively changed to passes.

Run unit tests with `python -m pytest -q -p no:cacheprovider
skills/cudaq-algorithms-dev/evals/e2e/tests`. For module-level real-runtime tests,
set `CUDAQ_E2E_PYTHON` (quantum, focused and extended host cases),
`E2E_RUNTIME_PYTHON` (classical and extended state preparation),
`E2E_SANDBOX_PYTHON` (PySCF isolation) and `E2E_SANDBOX_PSI4_PYTHON`
(Psi4 isolation) to their supported interpreters. Set `CUDAQ_E2E_RG` to real
ripgrep for the discovery-tool isolation check.
The campaign's preflight is the authoritative all-case gate;
unit test skips are not evidence of executed providers.

## Isolation and evidence

Source snapshots and fresh working directories are generated under task-owned
temporary directories. Source, ordinary docs/tests, input and treatment skill
are read-only to workers. Git history, evaluator code, hidden inputs, prior
results, other skills and sibling workspaces are denied. Only `app.py`,
`result.npz` and `.tmp/` may change. Gold and agent applications execute in
separate evaluator-owned sandboxes; the parent rejects symlink, nonregular,
oversized, corrupt, nonfinite and wrong-shaped output artifacts.
`PYTHONPYCACHEPREFIX` points to `.tmp/pycache` equally in every arm and grading
workspace, so explicit `python -m py_compile` also respects the file contract;
bytecode outside `.tmp/` remains a contract failure.

Durable results remain under `evals/results/`: manifest, exact versions and
source/evaluator hashes, per-arm snapshot paths and file inventories, prompts, raw native JSONL, sanitized telemetry,
application code, public/hidden numerical results, package-call traces and
grading decisions. No credentials or raw OTel payloads are recorded. No
global Codex configuration or authentication files are changed by this code.

Cases whose gold programs cannot execute are infrastructure-blocked; ready
cases still run. Timeouts are model failures even when final usage is missing.
Interrupted attempts retain evidence and are not automatically retried.
Re-running `run` resumes only pending attempts; a lock prevents concurrent
controllers. Evaluator changes require a new campaign after attempts exist.
Temporary source/runtime paths must survive to resume; the manifest makes
those dependencies explicit.
Preparation freezes each staged arm and checks that all non-skill files match.
Preflight checks those inventories before and after gold execution; run/resume
checks them again before launching attempts. Later edits to the original skill
directories cannot alter a prepared snapshot. Evaluator code is frozen by the
successful gold preflight; changing it invalidates the run gate.

## Metrics

- Scientific/API checks require both public and held-out execution, correct
  outputs and observed required API execution. The frozen overall pass also
  requires compliance with the workspace file rules. Report these endpoints
  separately: an extra bytecode-cache file can fail the overall contract even
  when the scientific/API checks pass. Merely writing code or printing a
  claimed answer cannot pass.
- Final native usage records supply cumulative input/output/cache/reasoning
  counters. Cached input and reasoning output are subsets, not additional
  tokens. Missing counters remain null. Observed partial totals are separate.
- Request input peaks are maxima of observed native completion counters, not
  cumulative input, model context capacity or guaranteed internal context
  occupancy. [Calibration](telemetry-report.md) records a native input-only
  event not accounted for by final usage; per-run cross-checks expose this.
- Attempt duration includes discovery, tool calls, coding and self-testing.
  Time to verified solution adds independent public/hidden execution time,
  and the stored metric is reported only for overall-contract successes.
  All-attempt duration comparisons retain scientific successes that failed
  workspace hygiene. Failed attempts retain time and available tokens; no
  stop-on-pass or cherry-picked repeat is used.
- Reports separate executed attempts, blocked slots, failures and measurement
  coverage. No dollar cost is inferred from a ChatGPT subscription.
- `summary.json` retains all existing per-arm distributions and adds strict
  and scientific/API counts, including separately reported family outcomes.
  `paired_comparisons.candidate_minus_previous` and
  `candidate_minus_baseline` match case/repetition records and retain raw
  pair differences plus distributions for token fields, observed request-input
  peaks (not complete context occupancy), attempt time, strict
  and scientific/API outcomes. Available failure telemetry contributes; missing
  values stay null, with observed-pair, missing-pair and blocked-pair counts.
  Missing-pair counts use all planned blocks, including blocks with no finalized
  records. Raw paired peak measurements retain each arm's completeness flag.
  Paired verified-solution timing includes only pairs where both arms passed
  the strict contract. These conditional timing figures do not replace
  all-attempt comparisons, and a negative difference means the candidate used
  less of that resource. Historical campaign wall times do not establish a
  causal comparison with a fresh campaign using different harness settings.

Package profiling establishes observed use, not adversarial proof that each
output depends on each call. CPU statevectors are not GPU/QPU benchmarks.
Small exact compressed-DF cases exercise genuine compression but may converge
at initialization; they do not establish iterative optimizer robustness.
Family coverage overlaps: one task can exercise several families. Family
denominators must not be summed or treated as independent experiments.

## Separate natural-language discovery suite

`focused.py` reuses the same worker isolation, grading, telemetry, three-arm
staging and reporting while selecting three additional cases:

- `focused_determinant_energy`: orthonormal occupied orbitals to a prepared
  electronic determinant, its Hamiltonian energy and particle number.
- `focused_signed_action`: signed Hamiltonian encoding, complex action and
  unnormalized postselection with the corresponding probability.
- `focused_phase_filter`: supplied-phase filtering with forward/adjoint steps,
  explicit phase conventions and preservation of the unnormalized complex block.

Every focused prompt withholds package callable names. Private API tracing
requires the common computational entries of helper and direct circuit paths:
the determinant schedule and compiled complex preparation, the signed encoding
constructor and encoding kernel, or phase-sequence construction and the spectral
transformation kernel. Convenience helpers are optional. Independent numerical
oracles remain those of the matching historical contract. This measures unaided API selection from scientific task
descriptions with ordinary source/docs/tests available, plus the applicable
skill arm; it is not discovery without package context. The filter task states
its mathematical phase and reflection conventions to make the requested output
unambiguous. The suite tests scientific phase/postselection boundaries, but
does not score refusals, unsupported-feature advice or authoring authorization.

Focused scientific inputs use new fixed coefficients, states/orbitals and
phases. Each case has distinct public and held-out variants. Repetitions reuse
those variants, so five repetitions do not constitute five new instances.
The three cases produce 45 planned attempts with five repetitions and three
arms. Their scores stay separate from the historical 15-case corpus.

```bash
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/focused.py prepare \
  --output skills/cudaq-algorithms-dev/evals/results/my-focused-campaign \
  --python /path/to/python \
  --previous-skill skills/cudaq-algorithms-dev/evals/results/refactor-20260911/previous-skill \
  --rg /path/to/real/rg --repetitions 5 --parallel 2 --timeout 480
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/focused.py preflight \
  --output skills/cudaq-algorithms-dev/evals/results/my-focused-campaign
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/focused.py run \
  --output skills/cudaq-algorithms-dev/evals/results/my-focused-campaign
/path/to/python skills/cudaq-algorithms-dev/evals/e2e/focused.py report \
  --output skills/cudaq-algorithms-dev/evals/results/my-focused-campaign
```

Use the same wrapper for every action on a focused campaign. Manifests and
reports are labeled `focused_natural_language`; historical campaigns are
`historical_api_informed`. The CLIs reject a mismatched campaign suite.
Invoking `run.py` directly retains the historical cases and API-informed prompts.

## Separate coverage-completion suite

`extended.py` selects seven API-informed application groups: three direct
state-preparation groups, separate Trotter call paths, spin-tensor/DF
diagnostics, good-subspace semantics and the existing real Psi4 energy case.
Together they target the 25 records without scoped passing evidence after
the refactor campaign. Gold tests establish evaluator readiness, not agent
success or full validation of those contracts.

Use `extended.py` for each `prepare`, `preflight`, `run` and `report` action,
with the same arguments shown above. Omitting `--previous-skill` and using
five repetitions creates 70 attempts: seven cases, two arms, five repetitions.
The manifest label is `coverage_completion`. Its denominators and KPI results
remain separate from the historical and natural-language suites. See the
[state-preparation contracts](../stateprep-completion-report.md) and
[host/simulation/Trotter contracts](../classical-completion-report.md) for
independent oracles, phase/sign distinctions and narrow resource-formula
boundaries. These CPU simulator checks are not GPU/QPU measurements.
