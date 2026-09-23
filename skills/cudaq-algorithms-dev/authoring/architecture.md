# Architecture

## Design goal

Represent CUDA-Q Algorithms as a durable library of independently selectable
scientific contracts. Applications consume and validate those contracts; they
do not define the taxonomy.

This file is maintainer and reviewer policy. Scientific tasks normally begin at
[the catalog](../../cudaq-algorithms/references/catalog.md), not here.

## Organization

- `../../cudaq-algorithms/SKILL.md` is the application entry point.
- `../../cudaq-algorithms/references/catalog.md` selects nine scientific families; detailed
  operation/object rows live in their family selectors.
- `../../cudaq-algorithms/references/<family>/<record>.md` holds one independently selectable
  primitive per focused file. A front door is navigation, not a multi-contract record.
- Shared representations and capabilities live in a focused support record
  or family front door when multiple producers/consumers exchange them.
- `../../cudaq-algorithms/references/conventions.md` selects five convention records;
  `validation.md` and `source-provenance.md` own validation and current-source lookup.
- `../../cudaq-algorithms/references/application-composition.md` describes application chains.
- `../../cudaq-algorithms/references/workflow.md` shares advisory and implementation guidance.
- This `authoring/` directory holds schemas, maintenance policy, and open design decisions.
- `../coverage/` holds lifecycle/history and per-feature evaluation mappings;
  `../scripts/check_coverage.py` checks consistency.

This development directory is not an application skill and has no `SKILL.md`.
Evaluation harnesses, fixtures, and historical evidence remain under `../evals/`;
do not package them as application guidance. Historical reports retain the paths
and revisions used by their original runs.

Keep scientific family paths shallow, with no primitive subdirectories. Every
focused record must be linked from its family selector or directly from the root
catalog. Each record covers every applicable canonical schema field once;
short records may combine adjacent headings when boundaries remain explicit.

## Authoring routes

- [Record design](record-design.md): identity, granularity, metadata, composition, resources.
- [Capability design](capability-design.md): semantic boundaries and identifiers.
- [Source review](source-review.md): source authority and freshness.
- [Extension workflow](extension-workflow.md): lifecycle and coordinated changes.
- [Design decisions](design-decisions.md): open ownership and promotion questions.
- Templates: [primitive](templates/primitive-record-template.md),
  [representation](templates/representation-record-template.md),
  [capability](templates/capability-record-template.md),
  [convention](templates/convention-record-template.md).
- [Coverage history](../coverage/history.md) and [feature registry](../coverage/features.json).
