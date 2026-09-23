# Feature coverage and evidence policy

This is reviewer/CI bookkeeping, not an application workflow. Application
agents use [scientific validation](../../cudaq-algorithms/references/validation.md); maintainers
start at [authoring architecture](../authoring/architecture.md).

## Registry

[features.json](features.json) inventories the 52 independently selectable
operation/object contracts. Stable IDs follow existing leaf filenames; shared
selectors, representations, conventions and workflows are `support_records`,
not extra primitives. Register additions and renames in the same change as the
record and its routing links.

- `record`, `family`, and `symbols` identify the actual public contract.
- `behavioral_evals` lists authored cases in `evals/evals.json`. Listing a case
  is not evidence that it ran or passed. Empty lists mean no explicit mapping.
- `executions` links a campaign, required public API, case, arm, repetition
  count and narrow scientific scope. A family label alone is insufficient.
- `scientific_pass` requires both public and held-out numerical inputs,
  independent oracles and actual required host/device API evidence for every
  listed repetition. Strict agent success additionally requires its complete
  workspace and execution contract; report the two outcomes separately.
- `executed` records attempts without asserting scientific success; `blocked`
  records unattempted infrastructure failures. Neither counts as a pass.
- Historical metadata indices refer to exact source-record fragments in the
  inventory linked by [history](history.md). These are preserved audit text,
  not current lifecycle labels or API authority.

The checker reports historical scientific evidence separately from evidence
whose **individual record bytes** still match the recorded skill inventory.
Matching one leaf does not validate the entire current skill, its references,
all parameters, target support or a newer dependency version. Campaigns retain
exact source/skill/evaluator hashes, dependencies, targets and commands. Five
agent repetitions over two inputs are not ten different scientific problems.

## Historical lifecycle and promotion criteria

The former record vocabulary was `draft | verified | deprecated | removed`;
capabilities separately used `candidate | provisional | stable taxonomy
contract`. Preserve those historical values in the audit history, not in
application records. They do not replace the concrete evidence states above.

The former `verified` promotion criterion required the contract, runnable
usage, relevant scientific tests and representative evals to actually pass on
a recorded, supported package/CUDA-Q version combination. That evidence bar
still applies to a claim of broad validation. One successful run or source
inspection alone does not establish it. Source inspection supports
`source-checked` only for the task performing it.

Deprecation documentation must name the replacement, first deprecated version
and behavioral differences. Prefer a new primitive name plus explicit
migration for incompatible contracts, rather than silently redefining one.
Keep scientifically meaningful unsupported, absent and unverified boundaries
in live references; they are not lifecycle clutter.

## Evaluation versus scientific validation

SkillEvaluator addresses activation, routing, usefulness, safety and answer
quality; that alone does not establish a circuit's numerical correctness.
Behavioral comparisons need no-skill and with-skill arms, while scientific
claims need repository tests or independent oracles. The native isolated
Codex experiments here are **not NVIDIA SkillEvaluator**. See
[evaluation methodology](../evals/EVAL.md) and
[executable evaluator](../evals/e2e/README.md) for protocols and limits.

## Deterministic CI entry point

From the repository root, with Python 3.9+ and no model or scientific runtime:

```bash
python skills/cudaq-algorithms-dev/scripts/check_coverage.py
python -m unittest discover -s skills/cudaq-algorithms-dev/scripts/tests
```

Use `--json` for machine-readable diagnostics or `--feature FEATURE_ID` to
inspect one registered contract. By default the checker treats sibling
`cudaq-algorithms` and `cudaq-algorithms-dev` directories as the runtime and
support roots. Pass both `--root` and `--support-root` for another split
layout; an explicit `--root` alone retains combined-layout compatibility.
The checker validates record inventory, local file links (not
heading-fragment existence), incremental runtime reachability, selectable
family-table entries, authored IDs and scoped run evidence. It does not rerun
science or judge prose correctness. External CI YAML wiring is intentionally
not changed under the skills-only constraint.
