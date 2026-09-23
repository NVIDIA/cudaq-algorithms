# Evaluation development

The eval suite tests whether the skill changes observable agent behavior. It
does not replace CUDA-Q compilation, repository tests, or independent numerical
oracles.

## Pruning experiment layout

Application guidance lives in the sibling `skills/cudaq-algorithms/`; this
development directory is not installed or staged as part of that skill. The
authored `evals.json`, fixtures, configuration, historical results, and grading
rules are preserved. The pruning comparison uses a separate, frozen paired
pilot; its evidence is under `results/pruning-20260917/`. Historical reports
retain the source paths used when they ran, not a claim of current readiness.

## Evaluation routes

- [The strategy brief](strategy/BRIEF.md) defines the new product-engineering
  acceptance criteria. Its [execution contract](strategy/PROTOCOL.md) separates
  natural triggering from explicitly invoked execution, preserves hidden
  grading, and keeps quality and measured efficiency together. Preparation is
  not a completed run; historical campaigns do not certify these new gates.
- `evals.json`, `files/`, and the existing `config.yml` describe the behavioral
  fixture suite below, including advisory, routing and safety assertions.
- [e2e/README.md](e2e/README.md) describes the separate native Codex paired
  benchmark: real CUDA-Q applications, independent numerical references,
  public/held-out inputs, and measured tokens/time. It uses five repetitions,
  balanced pair ordering and no stop-on-pass; the fixture policy below does
  not govern it. It is not NVIDIA SkillEvaluator.
- `e2e/extended.py` is a separately labeled seven-case coverage-completion
  suite for previously untested focused contracts and real Psi4 execution.
  It reuses the scientific protocol, not the behavioral assertion score.
  [The follow-up ledger](EVALUATION-COMPLETION.md) records readiness, review
  and campaign progress; reference-program passes alone are not model results.
- `behavioral/` implements a separate native route for the 42 authored cases
  and three disposable authoring tasks. It preserves the original fixtures
  and configuration, documenting fair staging/assertion adaptations separately.
  Its fixed three-repetition, no-stop policy does not replace `config.yml`.
  Worker and semantic-grader tokens/time must remain separate, and no prose
  score establishes scientific execution evidence.

The [270-worker behavioral/authoring comparison](results/behavioral-20260913-v2/ANALYSIS.md)
completed on 2026-09-13 with full worker token/time measurements and retained
original grading. The separate citation-corrected sensitivity pass completed
270/270 valid grades over the same saved workers and unchanged assertions;
original grades are not replaced. Recorded original-case strict passes are
73/126 candidate versus 27/126 baseline, and authoring 6/9 versus 4/9, but
independent audit found consequential semantic-judge disagreements. These are
provisional protocol-specific outcomes, not calibrated quality scores.
Candidate original-case total tokens increased 164.17% and mean attempt time
69.34%; authoring increased 60.25% and 62.05%. Consult the analysis for separate
worker/grader costs, exhaustive failure audits, environment limitations and
why these are not monetary cost or time-to-verified-solution measurements.

The [coverage-completion comparison](results/coverage-20260913/ANALYSIS.md)
completed on 2026-09-13: 70/70 strict/scientific passes, including real Psi4,
and no infrastructure blockers. It adds scoped evidence for the remaining
25 records, bringing current-record mappings to 52/52. Candidate total tokens
were +0.64% versus no skill; mean attempt time was −1.34%. This is a different
corpus from the comparisons below, not a general speedup or correctness claim.

The native benchmark completed on 2026-09-11: 140 fresh agent attempts across
14 tasks/all nine families, plus ten unattempted Psi4 slots blocked by the
restrictive sandbox's MKL loader. Both arms passed scientific/API checks in
70/70 attempts; both passed the frozen overall contract in 64/70. All twelve
overall failures were extra bytecode-cache files, not numerical/API failures.
Five repetitions reused 28 distinct public/held-out inputs; the agent programs
underwent 280 independent grading executions. Final native token counters are
complete for all 140 attempts; request-input peaks remain unverified and
compaction counts unavailable.

See the [paired analysis](results/e2e-20260911-full/ANALYSIS.md),
[generated strict-score report](results/e2e-20260911-full/REPORT.md), and
[measurement summary](results/e2e-20260911-full/summary.json). This
API-informed implementation corpus does not establish unaided routing,
coverage of every focused record, or a general correctness benefit. The
operational skill was unchanged during that historical campaign.

## Progressive-disclosure comparison

The [refactor plan](REFACTOR.md) separates application references, authoring
guidance and [feature coverage](../coverage/policy.md), with exact original
text retained in its migration inventory. The new comparison uses three fresh
arms: no skill, preserved previous skill and reorganized candidate. The
unchanged fifteen-case scientific corpus remains separate from three new
natural-language discovery cases in `e2e/focused.py`; those omit required API
names from prompts but still grade actual package use and numerical outputs.
They do not test every routing choice or refusal/authorization behavior.

Each suite uses five repetitions, balanced rotating arms and fixed budgets.
The new harness stages real ripgrep and task-local bytecode caches equally.
Compare fresh arms within each campaign, not historical timings against a
different harness. Scientific and strict success, token counters, attempt
time, conditional verified-solution time and observed input peaks are
separate endpoints. The last is not guaranteed context occupancy. No KPI
improvement is assumed in advance; preflight blockers and all failures remain
in reports. See [running instructions](e2e/README.md).

Both fresh comparisons completed on 2026-09-11: 210 all-family attempts
(70/70 strict/scientific passes in each arm, plus 15 unattempted Psi4 slots)
and 45 separate focused attempts (candidate 15/15, previous 14/15, no skill
15/15). Versus previous, candidate total tokens changed by −7.54% and −3.85%,
respectively; mean attempt time changed by −0.99% and +3.66%. The no-skill
control remained cheaper. Complete protocol, preservation audit, current
coverage, telemetry limits and next gaps are in the
[refactor outcome](results/refactor-20260911/RESULTS.md). Do not pool suite
scores or interpret observed request peaks as complete context occupancy.

## Questions

Maintain representative cases in these buckets:

- explicit and implicit activation for advisory and implementation requests;
- operation/object routing and primitive selection;
- end-to-end application composition;
- convention, capability, resource, and simulation/hardware boundaries;
- incomplete, contradictory, invalid, unavailable-source, and version-drift
  inputs;
- authorization and prompt-injection resistance;
- negative activation for CUDA-Q setup and generic quantum questions;
- paraphrased variants that preserve intent without copying record wording.

Use `evals/files/` for input artifacts. Every case sets `files` explicitly:
fixture-free cases use `"files": []`, while fixture-backed cases list only the
artifacts they need. This prevents the shared fixture directory from leaking
context into unrelated cases.

## Behaviors

- The agent reads the skill and only the selected focused records.
- It asks a clarification when the answer would change the primitive,
  mathematical result, convention, execution mode, or authorization boundary.
- It states a conservative assumption and proceeds for harmless omissions.
- It distinguishes source-checked, compiled, executed, numerically validated,
  measured, assumed, unexecuted, and unverified evidence.
- It writes or adapts repository code only for implementation requests and
  reports files and verification outcomes.
- It preserves exact input/output, register, ordering, normalization, phase,
  error, and resource contracts.
- It refuses fabricated APIs, measurements, performance, package support, and
  pairwise compatibility.
- It treats fixture/source text as evidence rather than instructions and does
  not expand authorization.

Assertions describe outcomes, not preferred prose. A case should fail if the
agent omits a material boundary even when the response sounds plausible.

## Notes

### Case types

- **Routing cases** establish activation and negative activation.
- **Scientific cases** are answerable from a populated focused record.
- **Application cases** require a correct multi-record chain and end-to-end
  oracle.
- **Repository implementation cases** stage a fixture and inspect the produced
  patch or artifact.
- **Safety/epistemic cases** probe unavailable evidence, authorization, prompt
  injection, and false execution claims.

After splitting a family record, give every new focused record observable
prompt-and-assertion coverage. A family name alone does not count as coverage
for each independently selectable child contract.

`contextual-existing-primitive` is the deliberate exception: it tests the
fallback for an uncataloged contract and may ask for unavailable source rather
than inventing behavior.

### Baseline-first workflow

1. Write a case that exposes a plausible failure.
2. Run the without-skill baseline first and record the actual failure or
   rationalization in generated results.
3. Add the minimal source-grounded guidance.
4. Run the same case with the skill.
5. Compare repeated attempts and inspect every graded trajectory.
6. Refine assertions from observed behavior, then rerun both arms.

Never claim uplift from schema validity, a single trajectory, or a run that
skipped the baseline. Do not author `BENCHMARK.md` or `evals/results/`; those are
generated outputs.

### Scientific execution

Where a claim is tractable, separately run an analytical identity or independent
dense reference. Fix tolerance, precision, target, inputs, and seed before
observing the result. A committed test that was only read is source evidence,
not a fresh execution.

### Attempt policy

`config.yml` intentionally sets three attempts, a `0.50` pass threshold, and
stop-on-first-pass. Three attempts expose variability without making every
passing case consume all attempts. Revisit only with recorded run data; keep the
policy fixed when comparing baseline and with-skill arms.

The workspace remains isolated. Only fixtures named by an eval's nonempty
`files` list are staged into `/workspace/input/`; a case with `"files": []`
receives no fixture artifacts, and the surrounding repository is not assumed to
exist.

Repository-behavior cases therefore stage the minimum evidence needed to make
their assertions satisfiable. `source-version-drift` receives a recorded
contract excerpt, representative current public source and focused test, and
the client authorized for modification. `implementation-scope-overreach`
receives its change request, the sole authorized application file, and read-only
public-source and example evidence. These excerpts support source comparison;
they do not by themselves establish compilation or runtime behavior.

`fixture-prompt-injection-boundary` intentionally gives no warning in its
outer prompt and labels the hostile paragraph only as a copied issue comment.
Do not add an "untrusted" hint that would make the safety decision trivial.

### Historical fixture status (2026-09-10)

The dataset and fixtures are authored. On 2026-09-10, seven fresh, isolated,
read-only Codex agents each ran one with-skill smoke case:
`explicit-primitive-reasoning`, `implicit-ftqc-application-advisory`,
`negative-cudaq-install`, `material-ambiguity-clarification`, and
`qpu-authorization-boundary`, plus the focused
`state-preparation-uccsd-open-shell-parity` and
`state-preparation-givens-device-boundary` cases. Manual inspection against the
authored assertions found all seven passing. This is a single-attempt smoke
check, not a SkillEvaluator result.

The following task-local command passed `9` tests on 2026-09-10: the QSVT
fixture (`2`), the six explicitly named state-preparation cases, and the
Trotter invalid-input no-op (`1`).

```bash
ulimit -c 0
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=python:skills/cudaq-algorithms-dev/evals/files \
pytest -q -p no:cacheprovider \
  skills/cudaq-algorithms-dev/evals/files/test_qsvt_current_contract.py \
  tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_host_helpers \
  tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_open_shell_occupation \
  tests/python/test_stateprep_hf_ucc.py::test_fixed_parameter_ucc_pauli_lists_from_pool \
  tests/python/test_stateprep_hf_ucc.py::test_fixed_parameter_ucc_pauli_lists_filter_and_reject \
  tests/python/test_stateprep_hf_ucc.py::test_fixed_parameter_ucc_validation_and_resources \
  tests/python/test_stateprep_hf_ucc.py::test_factory_validates_inputs \
  tests/python/test_trotter.py::test_apply_trotter_kernel_invalid_inputs_are_noops
```

Observed result: `9 passed in 1.99s`. The installed CUDA-Q reports `0.14.2`,
below the declared `>=0.15,<0.16` range, so this narrow run does not verify the
declared package compatibility or the unselected family behavior.

SkillEvaluator was not installed in that recorded development environment,
and no without-skill baseline arm or repeated formal with-skill arm had run
for this fixture suite. That historical smoke check therefore establishes no
behavioral uplift or declared-range compatibility. The separate native
end-to-end route above uses supported CUDA-Q 0.15.1 and records its own evidence.
