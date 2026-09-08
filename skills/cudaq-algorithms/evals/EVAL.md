# Evaluation development

The dataset has two kinds of case.

**Scaffold cases** test claims the skill's structure makes: activation,
ownership boundaries, operation-and-object routing, composition discipline, and
epistemic discipline about unavailable contracts. These do not assert any
scientific contract.

**Scientific cases** test populated family records. The
`state-preparation-*` cases and one positive-plus-boundary case for each other
populated family must be answerable from the skill package alone. Assertions
must remain supported by the linked record and must not silently broaden a
`derived`, `unverified`, or unsupported claim. These cases do not allow an
agent to pass by ignoring the skill and reconstructing the contract from
repository source, because that is the baseline behavior they exist to
distinguish.

`contextual-existing-primitive` is deliberately the exception: it tests the
documented fallback for a family that has *no* populated record, so
repository-derived reasoning is the correct behavior there.

For each scientific refinement:

1. create a task that exposes a baseline failure without the new guidance;
2. record repeated baseline behavior;
3. add the minimal source-grounded record;
4. rerun the task with the skill;
5. add executable or numerical oracles when tractable;
6. add a boundary, misconception, or negative case;
7. inspect correctness, discoverability, effectiveness, efficiency, and safety.

A scientific case must not be added before the record that backs it, and its
assertions must reference that record unconditionally once it exists.

Assertions must test observable behavior rather than preferred wording. Where a
record's status is `unverified` — as the state-preparation record's verified
version range is — the assertion asserts the honest status, not a value.

Where an assertion describes register ownership, it must describe the current
source-grounded injection behavior (the packaged consumer factories allocate
the system register), never a library-wide or future ownership policy: that
policy is an open question.

New datasets should use NVIDIA SkillEvaluator's current `skill_name` and
`evals` shape.

## Maintainer execution

Everything in this section requires write and execute tools. The skill itself
declares `allowed-tools: Read Glob Grep`, so a running agent can inspect and
describe evaluation evidence but cannot author fixtures, execute oracles, or
run the evaluator. These are maintainer actions performed outside the skill.

- **Executable and numerical oracles.** Where a claim is tractable, back it
  with an analytical identity or an independently constructed dense reference,
  fix the tolerance before observing results, and record the target and
  version the execution ran on. Until that execution happens, a committed
  repository assertion stays `derived` cited evidence — see
  `references/validation.md`.
- **SkillEvaluator runs.** Run NVIDIA SkillEvaluator with and without the
  skill. A benchmark establishes uplift only when repeated results show better
  correctness or efficiency than the baseline; schema validity alone proves
  nothing.
- **Offline gates first.** Validate the dataset and package before any live
  run, so a failure is a content failure rather than a staging failure.

Do not author `BENCHMARK.md` manually. SkillEvaluator generates it from
with-skill and without-skill runs. **No uplift has been measured for the
current dataset**, and no case in `evals.json` has been executed with or
without the skill.

## Harness configuration

`config.yml` pins the four keys the current NVIDIA SkillEvaluator guidance
establishes as commit-required:

- `schema_version: 1`;
- `harbor.task_source: evals_json`, so the cases come from `evals.json`;
- `grading.mode: default`;
- `skill_workspace.mode: isolated`, so only the target skill directory is
  staged.

`Deferred:` `n_attempts` and `pass_threshold`. The keys exist, but no captured
guidance fixes their spelling or default for this schema version, and unknown
keys are a hard load error rather than a warning — so they are left unset until
the first live run establishes them, rather than guessed.

`skill_workspace.mode: isolated` means `python/cudaq_algorithms/`,
`tests/python/`, and `docs/sphinx/` are **not** staged in the container.
Scientific cases are answerable from the skill package alone, which is what
the populated records are for. `contextual-existing-primitive` asserts that
the agent "inspects **or requests**" source for an uncataloged primitive, which
is satisfiable in isolation; if that case is tightened to require real
inspection, stage fixtures under `evals/files/` and select them per case
instead of widening the workspace mode.
