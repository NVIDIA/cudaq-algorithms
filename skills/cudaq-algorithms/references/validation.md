# Validation and evidence

Correctness is a gate. Compilation, plausible output, or agreement between two
paths with shared assumptions is not sufficient scientific validation.

## Evidence hierarchy

Prefer the strongest tractable independent evidence:

1. exact analytical identity;
2. independently constructed dense operator or state;
3. trusted external implementation after convention alignment;
4. invariant or property test;
5. cross-check between mathematically independent algorithms;
6. convergence study with a predeclared criterion.

## Validation record

Every verified primitive record must identify:

- claim under test;
- oracle and why it is independent;
- input domain and representative cases;
- convention translation;
- precision and execution target;
- tolerance fixed before observing results;
- invariant or comparison;
- expected failure or adversarial case;
- result and evidence status.

Use only these evidence labels:

- **derived:** follows from a stated definition or proof, including from a
  source or test file read at a named commit;
- **measured:** produced by a recorded execution, with the target, version, and
  conditions of that execution stated;
- **assumed:** required but not established;
- **unverified:** plausible or documented elsewhere but not checked here.

**A committed repository test assertion is cited evidence, not a measurement.**
Reading `assert np.max(np.abs(actual - reference)) < 1e-12` establishes what the
repository asserts at that commit, which is `derived`. It becomes `measured`
only when the test is actually executed and the result recorded. This skill
operates read-only: it can inspect and describe evidence — assertions,
tolerances, oracles, provenance — and must never restate that evidence as a
fresh measurement, a runtime figure, or a performance claim.

## Required dimensions

- **Semantic:** operation, sign, normalization, ordering, and phase.
- **Interface:** type, shape, dtype, register, controls, ancillas, and boundary.
- **Numerical:** absolute/relative error, conditioning, precision, and tolerance.
- **Approximation:** controlling parameter and expected convergence behavior.
- **Resources:** metric, unit, abstraction level, assumptions, and evidence.
- **Boundary:** malformed, unsupported, non-Hermitian, or unavailable cases.

Do not infer an asymptotic law from one data point. Do not weaken a tolerance
after observing a mismatch without changing and justifying the scientific
contract.

## Evaluation coverage a record must declare

Each populated scientific record states which evaluation cases its sections
support, spanning:

- explicit, implicit, and contextual positive routing where relevant;
- adjacent and generic negative routing;
- primitive selection;
- exact input/output and convention handling;
- legal and illegal capability compositions;
- approximation or resource claims;
- invalid, unsupported, and simulation-only boundaries;
- refusal to fabricate measurements or undocumented behavior.

Declared coverage is not validated coverage. Authoring, staging, and running
those cases — including executable oracles and NVIDIA SkillEvaluator runs with
and without the skill — is maintainer work that needs write and execute tools
this skill does not have. `evals/EVAL.md` owns that procedure; do not restate
it here, and do not claim uplift from a mapping table.
