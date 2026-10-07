# Feature coverage and evidence policy

This is maintainer bookkeeping. Application agents use
[scientific validation](../../cudaq-algorithms/references/validation.md);
maintainers start at [authoring architecture](../authoring/architecture.md).

## Live registry

[`features.json`](features.json) inventories 52 independently selectable
operation/object contracts. Shared selectors, representations, conventions and
workflows are `support_records`, not extra primitives. Register additions and
renames together with their records and routing links.

- `record`, `family` and `symbols` identify the public contract.
- `behavioral_evals` lists authored cases in the canonical
  [`evals/evals.json`](../evals/evals.json). A mapping is not evidence that a
  case ran or passed; an empty list means no explicit mapping.
- Historical campaign pointers, execution claims and old metadata indices
  were archived with the pre-delivery tree. The live registry has no bundled
  run evidence. Zero evidence counts mean none is attached here, not that
  scientific checks failed or that historical runs never occurred.

The checker retains optional campaign/evidence validation for future work.
Only attach a campaign when its referenced manifests and results are present
under the declared support root. Each execution must identify the required
public API, case, arm, repetition count and narrow scientific scope. Keep
campaign artifacts outside the delivered folders and attach them only in a
separate review workspace when using this capability.

## Evidence claims

Source inspection, compilation, execution and numerical validation are distinct.
Record the actual source/skill versions, dependencies, target, precision,
commands, independent oracle and tolerances with each result. Matching one
record's bytes does not validate the whole current skill or a newer runtime.
Historical evidence keeps its original scope and does not certify the 62-case
delivery suite. Do not restore stale success claims merely to increase coverage
counts after cleanup.

For optional historical campaign checks, `scientific_pass` requires the
recorded public and held-out numerical variants and host/device evidence for
every listed repetition. `executed` and `blocked` do not count as numerical
passes. Report scientific correctness separately from agent task success.

Behavioral comparisons require matched baseline and with-skill arms; scientific
claims require independent oracles or appropriate repository tests. Follow the
[delivery evaluation protocol](../evals/EVAL.md). No bundled model results are
implied by a successful static check.

## Deterministic checks

From the repository root:

```bash
python3 -B skills/cudaq-algorithms-dev/scripts/check_coverage.py
python3 -B -m unittest discover -s skills/cudaq-algorithms-dev/scripts/tests
```

Use `--json` for diagnostics or `--feature FEATURE_ID` for one contract. Default
roots are sibling `cudaq-algorithms` and `cudaq-algorithms-dev` directories;
pass `--root` and `--support-root` for another split layout. An explicit
`--root` alone retains combined-layout compatibility.

Checks cover inventory, local file links, runtime reachability, selectable
family-table entries, authored IDs and any explicitly attached run evidence.
They do not rerun science, grade model answers or validate Markdown anchors.
