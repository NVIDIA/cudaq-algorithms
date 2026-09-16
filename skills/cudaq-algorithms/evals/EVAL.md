# Evaluation development

The eval suite tests whether the skill changes observable agent behavior. It
does not replace CUDA-Q compilation, repository tests, or independent numerical
oracles.

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

### Current status

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
PYTHONPATH=python:skills/cudaq-algorithms/evals/files \
pytest -q -p no:cacheprovider \
  skills/cudaq-algorithms/evals/files/test_qsvt_current_contract.py \
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

SkillEvaluator is not installed in the recorded development environment, and
no without-skill baseline arm or repeated formal with-skill arm has run.
Therefore no behavioral uplift or declared-range compatibility claim is made.
