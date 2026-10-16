# Extension Workflow

Only add records for QROM, arithmetic, sparse-oracle, eigensolver, THC, or
additional resource-model concepts once current public source establishes a
contract. Roadmap names alone do not establish available primitives.

## Lifecycle and evidence

The following preserves the historical review vocabulary and promotion bar.
Store lifecycle history and executed evidence under
[coverage](../coverage/policy.md), not as status comments in operational
records. Current coverage uses scoped evidence, not blanket lifecycle stamps.

`draft | verified | deprecated | removed` is the only record lifecycle
vocabulary. A capability separately uses
`candidate | provisional | stable taxonomy contract`.

A record becomes `verified` only after its contract, runnable usage, relevant
scientific tests, and representative evals have actually passed on a recorded,
supported package/CUDA-Q version combination. Store exact source revisions,
dependencies, targets, and commands with that result. Source inspection alone
supports `source-checked` in the task that performs it, not `verified`.

Deprecation records name the replacement, first deprecated version, and
behavioral differences. Incompatible contract changes prefer a new primitive
name and explicit migration over silent redefinition.

## Growth rule

Add knowledge in this order:

1. shared representation or convention when demonstrated;
2. one independently selectable primitive contract;
3. a capability only when multiple producers or consumers justify it;
4. a composite protocol when its lower-level contracts are populated;
5. catalog routing, runnable usage, validation, and eval coverage in the same
   change.

Do not add records for roadmap concepts or speculative APIs. Split existing
records when real contracts have become independently selectable.

## Incremental development rule

Add or refine one independently selectable contract at a time. Update its
catalog entry, source provenance, complete contract, resource status,
independent validation method, runnable example or usage test, and evaluation
coverage together. Split a record whenever operation/object identity, return
type, execution layer, validation oracle, approximation behavior, resource
contract, or composition boundary can be selected independently.

`verified` is a record lifecycle state, not shorthand for reading source or for
one successful run. Promotion requires a recorded, supported package/CUDA-Q
version combination; exact revisions, dependencies, targets, and commands
belong with the validation or evaluation result.

## Skill evaluation versus scientific validation

SkillEvaluator checks activation, routing, usefulness, safety, and answer
quality. It does not establish that a quantum circuit or numerical transform is
scientifically correct. Run baseline and with-skill eval arms for behavioral
uplift, and run repository tests or independent numerical oracles separately
for scientific claims.
