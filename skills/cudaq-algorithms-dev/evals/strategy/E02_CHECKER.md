# E02 scale checker: first phase

Initial implementation status (retained history): isolated implementation for review. It is not registered with
`numerical.py`, `artifact_checks.py`, or any pilot. No production source,
runtime, BRIEF, skill, or existing E03 checker is changed. Runtime calibration
must finish before these tests support any scored acceptance decision.

## Calibration checkpoint — 2026-09-15

The [preserved independent calibration report](../results/strategy-20260915-e02-calibration/REPORT.md)
records **30 calibration tests passed, no skips**, in the pinned CUDA-Q 0.15.1
fp64 qpp-cpu image. Measured checker duration was 715.69 seconds. The result,
sanitized freeze/launcher records, log, and original/preserved hashes are
retained together; current checker-file hashes matched the recorded inputs
when preserved. The run recorded zero model and worker calls.

This validates the 30-test calibration stage only. The 60-test worker-targeted
suite has not been executed by that calibration. Result classification,
source binding integration, registry hookup, and E02 worker acceptance remain
unfinished. The initial inventory and design history below are retained;
calibration success is not a complete E02 certification or a primary NVIDIA
benchmark result.

## Positive-binding checkpoint — 2026-09-15

The [preserved end-to-end positive-control report](../results/strategy-20260915-e02-positive-controls/REPORT.md)
records the full `checks/test_scale_encoding.py` selection passing separately
against `scale_calibration.compact` and `scale_calibration.padded`: **60 passed
per binding, no skips or errors**, in the same pinned CUDA-Q 0.15.1 fp64
qpp-cpu image. Checker durations were 469.29 and 525.29 seconds. Both results
record unchanged inputs/source and zero worker/model calls; commands, source,
helper, checker, and explicit binding hashes were frozen before either run.

These are evaluator-only tomography implementations, not accepted worker
solutions or KPI improvements. The earlier 30-test checkpoint above remains
historical evidence. Positive end-to-end adapter operation is now checked;
worker source/consumer binding, end-to-end mutant fail/unknown classification,
missing-runtime/binding runtime diagnostics, and registry integration remain
unfinished. Neither checkpoint completes E02 certification.

## Captured-source integration checkpoint — 2026-09-15

The current local runner registers E02 separately from E03: 60 targeted tests
with no skips, plus the existing 311-case regression selection with at most
four documented skips. Supply `artifact_checks.py --attempt-id ... --e02-binding
...` for one E02 attempt. No entrypoint is guessed from the worker response;
missing or unreadable bindings leave targeted coverage unknown while retaining
independent regression evidence.

`numerical.py --case-id E02 --kind targeted --binding-path ...` stages only the
binding JSON in a read-only mount and requires `E02_ARTIFACT_ROOT` provenance.
The selected callable/module or bound method must originate in the captured
repository. Public PauliLCU, Walk, QSVT and PhaseSequence symbols must resolve
to their expected files; the six consumer/helper files in `E02_CONSUMERS` must
match frozen source. Changes to those files require independent review, not an
automatic scientific-failure verdict. The guard checks provenance; it is not a
security boundary against arbitrary Python code inside the isolated container.

Binding, captured inventory, checker inventory and consumer hashes are frozen
before execution and checked afterward. Artifact collection and posthoc grading
verify that evidence against the exact case and captured attempt, including the
input-freeze hash. A bare log reporting 60 passes is insufficient. Earlier
frozen helpers and results are unchanged.

Host integration tests now exercise these paths. Pinned-runtime validation of
the revised source-bound path, including end-to-end negative/unknown controls,
remains required before scored E02 acceptance. Earlier positive controls do
not by themselves validate these newer integration changes. No E02 worker
solution or quality/efficiency improvement is certified by this checkpoint.

## Explicit artifact adapter

Mount a trusted evaluator-authored JSON file and set `E02_SCALE_ADAPTER` to
its path inside the existing credential-free checker container. Select the
entrypoint from the captured public API/docs before running numerical probes.
Example declaration (placeholder module; not a required worker API):

```json
{
  "schema_version": 1,
  "kind": "callable",
  "module": "cudaq_algorithms.composition",
  "attribute": "scale",
  "args": ["encoding", "factor"],
  "kwargs": {},
  "zero_policy": "encode"
}
```

`callable` accepts a public function, class constructor, or nested public
callable selected by dotted attribute. Positional/keyword arguments map each
role exactly once. For a constructor with keyword-only arguments, use
`args: []`, `kwargs: {"base": "encoding", "multiplier": "factor"}`. `method`
omits `module`, selects a method on the input encoding, and maps only `factor`.
No automatic method insertion or inheritance is supplied to foreign encodings.

There are no constants, expressions, factories, result unwrappers, sign fixes,
normalization fixes, or fallback implementations in the adapter. It returns
the artifact's object unchanged. A return wrapper or another API shape needing
additional interpretation remains unverified in this first phase. A controller
must verify/hash the binding and worker module origin; this module executes
imports only inside the isolated checker and is not itself a security sandbox.
Never bind worker acceptance to the evaluator calibration module.

The binding preflight resolves and checks the signature without calling the
operation. Missing/mismapped entrypoints are setup errors (unknown coverage).
Explicit `zero_policy: "reject"` plus TypeError/ValueError/NotImplementedError
on zero produces an `E02_ZERO_UNSUPPORTED` setup error: no scientific failure
and no acceptance pass. A zero arithmetic error or wrong returned block fails
the test. Invalid nonreal/nonfinite factors must be rejected; real-valued
complex scalar acceptance is not required. Geometries exceeding seven total
qubits including the control produce `E02_PROBE_LIMIT` setup errors, not a
claim that larger valid encodings are wrong.

## Intended inventory (provisional until pinned-runtime calibration)

| Selection | Count | Required skips | Coverage |
|---|---:|---:|---|
| `checks/test_scale_encoding.py` | 60 | 0 | 16 blocks/coherent inputs, 8 controls/adjoints, 8 coherent walk-sign probes, 8 downstream checks (QSVT degrees one/two), 4 zero, 16 invalid factors |
| `checks/test_scale_calibration.py` | 30 | 0 | 20 positive combinations (two normalizations, two sizes, five factors), ten negative-control tests |

The NumPy/adapter tests under `tests/test_scale_*.py` are host unit tests,
not worker numerical acceptance. Collection verifies inventory only; it is
not runtime calibration. Freeze counts and test/binding hashes only after
both positive calibration policies pass in the pinned CUDA-Q 0.15.1 fp64
qpp-cpu image and every numerical mutant is rejected as intended.

Run the two runtime files separately using the existing container isolation
pattern, replacing only the explicit pytest selection in a new diagnostic
command. The current runner remains E03-only. Calibration needs no adapter
environment variable. Worker checks require it. Do not run worker-modified
tests or expose credentials. Map missing runtime, binding, or unsupported
coverage to unknown. A full numerical pass requires all 60 tests and no skips
or errors; existing regression passes cannot substitute for it.

References use literal matrices with Y terms, unequal coefficients, identity
offsets and the repository's qubit ordering. Full protocol-kernel matrices
are reconstructed by evaluator-owned register wrappers, independently of
`sim.action`, `encode_kernel`, artifact dense builders, and artifact alpha
computation. Both `B = c H / alpha_out` and `alpha_out B = c H` are checked
to avoid a huge normalization hiding a wrong block. Full controlled matrices
check coherent relative phases, not only probabilities or control-one states.
An evaluator-owned PREPARE → W → UNPREPARE wrapper checks the entire good
block against `-H_scaled / alpha_out`; slicing raw W would use the wrong
ancilla basis. This catches a coherent sign error shared by forward/adjoint
walks even when their inverses and controls remain internally consistent.
Walk moments 0/1/2 and QSVT degrees one/two use their public consumers. The
degree-two all-zero phase sequence is checked against `(2 signal² - I) ket`,
which detects inconsistent action on the bad subspace hidden at degree one.
Before scored
runs the controller must verify those consumer source files against the frozen
source or explicitly review changes, so a worker cannot weaken its own oracle.

Calibration `compact` reconstructs a small input through simulator tomography
and makes a PauliLCU; `padded` adds cancelling identity terms, changing alpha
and sometimes ancilla width without changing H. These are evaluator-only
positive controls, not proposed production scale implementations. Their
decomposition is separately tested against literal Pauli coefficients. The
expected oracle matrices do not come from that decomposition. Mutants cover
ignored factors, lost negative signs, wrong alpha, swapped system order,
relative control phase, wrong adjoint, invalid-factor acceptance, zero division,
and built-in-only dispatch. Additional numerical mutants negate both walk
directions with matching controlled-branch phases, and change `U` to `U D`
with `D=+1` on the good subspace and `-1` elsewhere, matching its controlled
variant while retaining the original walk kernels. Their calibration tests
verify the preceding inverse/control or degree-one checks pass, then require
the new numerical assertion to fail. The zero-division and built-in-only
tests assert control exceptions; before
freezing, also run the targeted test file against those bindings to verify
end-to-end fail/unknown classification and missing-runtime/binding behavior.

Remaining boundary: arbitrary result-container adapters, more general
normalization extremes, source binding enforcement, end-to-end mutant result
classification, and registry integration are later reviewable steps. This
first phase must not be reported as complete E02 certification.
