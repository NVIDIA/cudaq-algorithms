# Runtime skill evaluations

This directory is a standalone SkillEvaluator dataset for `cudaq-algorithms`.
It contains the same 62 ordered cases as the repository's canonical development
suite: 42 regression cases and 20 scientific cases. There are 59 positive cases
and three negative routing cases:

- `negative-cudaq-install`
- `negative-generic-quantum-concept`
- `negative-generic-cudaq-kernel`

Each negative case has `expected_skill: null`; it must not activate this skill.
All ten fixtures referenced by the cases are included under `files/`. Paths are
relative to this directory, so the runtime skill can be copied or installed
without the development skill.

SkillEvaluator excludes `evals/` from the agent-visible skill copy, including
links to evaluation material. Assertions stay in the private verifier entry;
the task question and declared input fixtures are staged separately.

## Preserving the canonical questions

`evals.json` uses ASCII JSON escapes for non-ASCII characters. Its decoded object,
including every prompt, assertion, expected output, case order, and fixture path,
is identical to the canonical suite. The fixtures are byte-identical copies.
The repository's `test_catalog_evals.py` checks both properties and standalone
fixture resolution. Update this view only alongside a reviewed canonical change;
do not independently regenerate or refine its questions.

## Local validation

With NVIDIA SkillEvaluator and its static scanner dependencies installed, run
these provider-free checks from the repository root:

```sh
skillevaluator validate skills/cudaq-algorithms --tiers 1 --no-llm --no-dedup
skillevaluator tier3 validate skills/cudaq-algorithms --strict
```

For a separately installed skill, replace `skills/cudaq-algorithms` with its
directory. The first command checks the skill's static packaging, security, and
quality. The second validates the dataset, configuration, and fixture paths.
Neither command runs an agent. A missing scanner is an incomplete check, not a
pass. The configuration uses supported SkillEvaluator keys only.
Five-seed paired campaign metadata belongs to the development runner, not this
Harbor configuration.

Live evaluation is a separate operation requiring configured agent/provider and
runtime dependencies. The default SkillEvaluator assertion grader does not run
the development campaign's independent numerical checkers or reproduce its
published five-seed measurements.

The upstream [eval dataset contract](https://docs.nvidia.com/skills/skillevaluator/eval-datasets)
and [source loader](https://github.com/NVIDIA/SkillEvaluator/blob/efcd97ae7e0d2e2179e8a87c5b30c062626f69e7/src/skillevaluator/tier3/harbor/adapter.py)
define this packaging. The loader requires a real local dataset file; an external
development-suite link is not a supported substitute.
