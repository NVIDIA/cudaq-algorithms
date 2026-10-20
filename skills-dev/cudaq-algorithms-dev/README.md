# CUDA-Q Algorithms skill development

This directory contains maintainer tooling for the
[CUDA-Q Algorithms skill](../../skills/cudaq-algorithms/SKILL.md). It lives outside
`skills/` so the skills catalog does not evaluate or publish it as an application
skill. Scientific guidance remains in the runtime skill and its references.

The optional [maintainer skill entrypoint](SKILL.md) selects the development
references and checks below. It stays outside the application skill catalog and
agent-facing deployments.

| Directory | Purpose | Start here |
| --- | --- | --- |
| `authoring/` | Record schemas, extension workflow, source-review policy and design decisions | [Architecture](authoring/architecture.md) and [extension workflow](authoring/extension-workflow.md) |
| `coverage/` | Public contract inventory and mappings from features to evaluation evidence | [Coverage policy](coverage/policy.md), [feature registry](coverage/features.json) and [API drift contract](coverage/api_contract.json) |
| `evals/` | Canonical development suite, pinned fixtures, case contracts, runner, grading and regression tests | [Evaluation protocol](evals/EVAL.md) and [runner instructions](evals/runner/README.md) |
| `scripts/` | Coverage and packaging checks, official SkillEvaluator adapter, campaign execution, monitoring and reporting | [Coverage checker](scripts/check_coverage.py), [packaging checker](scripts/check_packaging.py), [SkillEvaluator adapter](scripts/evaluate_skill.py), [API drift checker](scripts/check_api_drift.py) and [runner CLI](scripts/run_eval.py) |

Read the existing contract and current public source before changing a record.
Use the extension workflow to update coverage and evaluations together. Keep
fixture bytes consistent with `evals/manifest.json`; do not add headers or
formatting to pinned fixture copies. Historical campaign evidence belongs to
its original source and suite revision.

## Runtime catalogs and agent installations

The repository intentionally keeps the catalog-facing evaluation dataset under
`skills/cudaq-algorithms/evals/`, including negative controls, while this directory
owns the development harness and its canonical suite. Catalog tests check their
agreement. Moving the maintainer tooling outside `skills/` does not remove the
runtime catalog required by NVSkills.

For an agent-facing deployment, omit the entire runtime `evals/` directory,
including its answer keys (`assertions` and `expected_output`), and omit all of
`skills-dev/`. Copy the runtime `SKILL.md`, `references/`, `scripts/`, and any
required `assets/` into a clean installation directory. Do not copy development
coverage, checkers, grading rubrics, results, Python caches or credentials. If
updating an existing installation, ensure old excluded files are removed by the
deployment process; copying allowed files over an old tree is insufficient.
These exclusions apply to the deployed copy, not the repository catalog used by
CI. The skill still needs the CUDA-Q Algorithms source checkout described in its
prerequisites.

The development runner enforces this boundary for its own workers. Other
installers must apply the same exclusions explicitly; the presence of an
`evals/` directory in a repository does not make it safe to expose to a worker.
See the [runner instructions](evals/runner/README.md) for the separate controller
credentials, root/capability requirements and execution-isolation boundary.

## Local checks and CI

Run development commands from the repository root, or use absolute script paths.
The existing [offline workflow](../../.github/workflows/skill_evaluation.yaml)
runs development suites on relevant pull requests and supports manual and reusable
invocation. Packaging, API drift and SkillEvaluator checks can be run locally.
SkillEvaluator CI integration is deferred. The API drift checker requires the
library dependencies and imports the requested source checkout; update its
explicit absence claims when a formerly missing API becomes public.

Offline tests, real isolated execution, local SkillEvaluator results, full model
regressions and hosted NVSkills approval provide different evidence. Follow the
[recorded evaluation protocol](evals/EVAL.md) and preserve failures, missing usage
and unknown assessments when reporting results. A local pass does not establish
that hosted CI ran on the final published revision.
