# Validation and evidence

## Choose the validation path

Use the strongest path available without overstating it:

| Path | Minimum evidence | Permitted claim |
| --- | --- | --- |
| Source path | Current public source, tests, and docs inspected; checkout or release captured with the result | `source-checked` or `derived` |
| Build path | Source path plus successful parse/import/compile/build | `compiled` for that artifact and environment |
| Execution path | Build path plus successful run on a named target | `executed` for that case |
| Scientific path | Execution path plus independent oracle and predeclared tolerance | `numerically validated` |
| Measurement path | Scientific protocol plus newly collected empirical quantity | `measured` |

## Scientific validation procedure

1. State the claim and mathematical oracle before execution.
2. Translate qubit order, tensor order, signs, phases, normalization, and units.
3. Fix target, precision, representative inputs, random seed when relevant, and
   tolerance before viewing results.
4. Run the smallest test that distinguishes the intended contract from likely
   convention errors.
5. Include an adversarial or expected-failure case.
6. Record the command, date, package/CUDA-Q version, target, precision, result,
   and limitations.

Prefer dense constructions, analytic identities, independently implemented
matrix actions, conserved quantities, or cross-representation checks. A test
that calls the same implementation through a second wrapper is not an
independent oracle.

## Evidence labels

- `derived`: mathematical consequence of stated assumptions or source.
- `source-checked`: current public source/tests/docs were inspected in the
  current task, with the checkout or release captured in the result.
- `compiled`: the artifact successfully parsed, imported, compiled, or built.
- `executed`: it ran successfully on the stated environment and case.
- `numerically validated`: execution agreed with an independent oracle within
  the predeclared tolerance.
- `measured`: an empirical property such as runtime or memory was collected in
  the stated protocol.
- `unexecuted`: an execution-state qualifier meaning no successful run was
  completed in the current task. Combine it with the strongest available
  evidence label, for example `source-checked; unexecuted`; it is not evidence
  of failure or correctness.
- `assumed`: used without verification; state why and its impact.
- `unverified`: evidence is absent or insufficient.

Keep scope explicit. A committed repository assertion that was not run in the
current task is `source-checked` or `derived`, not `executed`. One simulator
result does not establish hardware support. A logical-operation proxy is not a
transpiled gate count, runtime, or memory measurement.

## Unable to execute

When dependencies, hardware, credentials, source, or time prevent a run:

- label code and conclusions `unexecuted` or `unverified`;
- say exactly what blocked execution;
- give the command, fixture, oracle, target, precision, and tolerance needed;
- do not weaken the check or substitute a non-independent oracle merely to
  produce a passing result.
