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
   Independently validate constructed inputs against the user's definition
   before reusing them in a circuit and dense oracle; see
   [product-state amplitudes](conventions/ordering.md#product-state-amplitudes).
3. Fix target, precision, representative inputs, random seed when relevant, and
   tolerance before viewing results.
4. Run the smallest test that distinguishes the intended contract from likely
   convention errors.
5. For new or repaired implementations, include a relevant discriminating
   edge/failure case (for example, a complex state or near-zero success branch).
   Avoid adding unrelated tests to an already checked numerical calculation.
6. Record the command, date, package/CUDA-Q version, target, precision, result,
   and limitations.

Prefer dense constructions, analytic identities, independently implemented
matrix actions, conserved quantities, or cross-representation checks. A test
that calls the same implementation through a second wrapper is not an
independent oracle.

## Match the check to the scientific claim

Let `psi` be normalized, `b` an unnormalized success branch, and `O` Hermitian.
Declare units, sector, absolute/relative tolerance, and sampling uncertainty
where applicable. Dense references below are for tractable validation cases.

| Requested quantity | Independent checkpoint | What it does not establish |
| --- | --- | --- |
| Prepared physical state | Occupations/symmetries and `1 - abs(vdot(target, psi))**2`; normalize both states first | A global phase needed later by a controlled or interferometric operation |
| Raw block or phase-sensitive amplitude | Vector/matrix norm against an independent dense action, without renormalization or phase alignment | Conditional fidelity or heralding probability by itself |
| Energy or gap | Diagonalize the same physical sector, with every scalar included exactly once; specify which distinct levels define a gap | A sector ground state need not be the global ground state; a small energy error need not mean high state fidelity |
| Observable dynamics | `abs(vdot(psi, O @ psi) - reference)` at all requested times, plus the declared cost model | Conservation alone does not prove accurate dynamics; a finite grid is not a uniform-in-time bound |
| Return amplitude | Complex spectral sum or independent matrix exponential; preserve scalar phases | Its squared magnitude loses phase information |
| Conditional observable | `p = vdot(b, b).real`, then `vdot(b, O @ b).real / p` only for resolvably nonzero `p` | Normalizing first erases the success probability; conditioning is not tracing out a register |
| Discarded-register state | Partial trace of the actual dilation, or independently justified Kraus operators; check trace, Hermiticity, positivity | Knowing only the good block does not determine an arbitrary dilation's discarded channel |
| Model compression | Reconstruct the physical operator, then check the requested sector energy/gap/observable and cost | Tensor Frobenius residual, leaf count, or optimizer success alone does not certify physical accuracy |
| Resource or cost claim | Name the level: a logical-operation proxy (formula counts, exact Toffoli counts of a minted kernel) is distinct from a transpiled gate count for a named target, which is distinct from measured runtime/memory; compare candidates at one declared level | Hardware cost; a proxy at one level does not establish the count at another, and one simulator run is not a measurement of device cost |

For low-probability branches, finite shots, rank truncation, or near-degenerate
levels, expose sensitivity instead of hiding it with normalization or a loose
tolerance. Keep model discrepancy separate from numerical agreement. Use the
family guide's checkpoint to establish the selected boundary; an entire chain
requires its own end-to-end comparison. A candidate that fails a requested
accuracy target is a valid assessment when the evidence and searched scope
are reported accurately.

Record enough run provenance to reproduce the result in the artifact: package
versions, target/precision, inputs or their provenance, settings, reference,
and errors. Summarize routine checks in the answer, including the exact command
that reproduces the delivered result and its observed outcome; retain detailed
logs with the artifact. Offline execution against an
independent oracle can establish numerical correctness for the checked cases;
static link, coverage, and source checks alone cannot. Neither establishes
improved model pass rates, time, or token usage. Those claims require separate
paired model evaluations on the same tasks, grading, environments, and budgets,
with the skill revision and suite identity recorded.

## Requested task completion

Check the request's deliverables against the actual answer and artifact:

- **When a result is requested through a named library object:** trace it to
  that object's returned data or executed operation. Instantiating it only to
  report its size leaves the calculation unimplemented. An independent oracle
  validates this path; it does not replace it.

- **When state preparation is requested:** supply the preparation operation
  and compatible register/signature. An occupation list or basis index describes
  the input but does not implement preparation. If execution is requested and
  available, check the resulting state against that input.

- **For composition or interchangeability advice:** cover relevant representations,
  capabilities, register layout, normalization, phases, and execution layers.
  Distinguish host factories, device kernels, and simulation-only analysis;
  separate compatibility from target selection.

- **When claiming sufficient improvement:** compare against the user's criterion.
  If none is given, quantify the change and state what criterion would make it
  sufficient; a feasibility bound alone does not define usefulness.

Numerical predictions alone need not include a new circuit. Match the requested
execution level, preserving any explicit library or artifact requirements.

## Requested output artifacts

When the request requires machine-readable results, validate the exact delivery
interface as well as the scientific calculation:

1. Identify the requested entrypoint, output destination and format: a named
   file, a stdout record, or a function return value. Build one complete payload
   containing the required results and emit it at that destination. Additional
   summaries or sidecar files supplement that payload.
2. Execute the declared command or call and capture the result as the consumer
   receives it. For a last-stdout-line JSON contract, capture stdout and parse
   `json.loads(completed.stdout.splitlines()[-1])`; for a file contract, read
   that exact file after execution; for a return-value contract, inspect the
   actual returned object. Check the exit status or exception too.
3. Validate the captured payload against the requested schema and the checks
   below. Keep evidence of this check with the reproduction command. A valid
   payload at another destination does not establish this interface's validity.

- Check required fields, types, array shapes, admissible values, and units.
  Correct scientific numbers alone do not establish a valid output contract.
- Check field meanings as well as shapes: reconstruct the reported settings from
  the actual computation and express them in the consumer's specified convention.
  Internal substep coefficients, normalized values, or per-register counts may
  need conversion before they describe a full step, physical value, or total.
  Verify that the declared settings reproduce the reported result where feasible.
- Where IDs identify candidates, check uniqueness and that every selection or
  cross-reference resolves to an emitted ID. Derive stable IDs from the settings
  that distinguish candidates; equal realized sizes do not imply equal IDs.
- Record requested caps separately from realized rank or size, deriving the
  latter from returned data. For example, explicit double factorization's
  `max_num_leaves` is a cap; `factorization.num_leaves` is the returned count and
  may stop below it. See the [explicit-factorization record](double-factorization/double-factorization-explicit.md)
  and `python/cudaq_algorithms/double_factorization/_factorization.py`.
- If multiple settings produce the same candidate, preserve distinct requested
  settings or, when the output contract permits deduplication, retain their
  mapping to the shared result. Check table headings and summary claims against
  the emitted settings, realized sizes, and selected IDs.

Apply the [completion checkpoint](workflow.md#completion-checkpoint) to reconcile
the final explanation with this validated payload.

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
