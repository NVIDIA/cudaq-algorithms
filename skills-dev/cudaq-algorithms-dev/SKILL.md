---
name: cudaq-algorithms-dev
description: Use when maintaining the CUDA-Q Algorithms skill's records, coverage, packaging, or evaluation tooling; not for scientific API usage.
license: Apache-2.0
metadata:
  author: CUDA-Q Algorithms Team <cuda-quantum@nvidia.com>
---

# CUDA-Q Algorithms skill maintenance

This optional maintainer entrypoint belongs in `skills-dev/`, outside the
application skill catalog. For ordinary scientific API requests, use the
[runtime skill](../../skills/cudaq-algorithms/SKILL.md) and its scientific
references. Use this companion to maintain that skill and its development tools
in a source checkout.

## Select the maintenance reference

- For record boundaries and placement, read [architecture](authoring/architecture.md)
  and the relevant record/template links there.
- For additions or contract changes, follow the [extension workflow](authoring/extension-workflow.md).
  Inspect current public source and authoritative tests using [source review](authoring/source-review.md);
  update the record, routing, runnable usage, validation and evaluation coverage
  together. Roadmap names alone do not establish public APIs.
- For inventory and evidence mappings, read [coverage policy](coverage/policy.md)
  and [features.json](coverage/features.json). Source inspection, execution and
  independent scientific validation support different claims.
- For evaluation work, read the [evaluation protocol](evals/EVAL.md) and
  [runner instructions](evals/runner/README.md) before selecting commands or
  interpreting results. Loading this skill does not authorize paid endpoint
  probes, grading or campaigns; use the user's authorized scope and budget.

## Check the change

From the repository root:

```bash
python3 -B skills-dev/cudaq-algorithms-dev/scripts/check_coverage.py
python3 -B skills-dev/cudaq-algorithms-dev/scripts/check_packaging.py
```

Select additional source/API, runner or scientific checks from the
[development README](README.md) and the affected contract. Check newly added
links directly: the coverage checker does not scan this entrypoint or the
development README. Static checks do not establish behavioral uplift,
scientific correctness or hosted NVSkills approval. Report the commands,
observed results and remaining unknowns.

## Preserve the delivery and evidence boundaries

Keep pinned fixture bytes consistent with `evals/manifest.json`; intentional
suite changes need the recorded rebaseline workflow. Preserve failed attempts,
missing usage and unknown assessments. Historical results retain their original
source, skill and suite revisions and do not validate newly edited bytes. Keep
campaign artifacts and generated reports outside both skill trees.

Keep the runtime catalog dataset under `skills/cudaq-algorithms/evals/` for CI.
For agent deployments, omit that entire `evals/` directory and all of
`skills-dev/`, following the [deployment instructions](README.md#runtime-catalogs-and-agent-installations).
Workers must not receive answer keys, private checkers or provider credentials.
Credentials belong to the controller; retain the runner's isolation boundary
and fail-closed behavior when execution privileges are unavailable.
