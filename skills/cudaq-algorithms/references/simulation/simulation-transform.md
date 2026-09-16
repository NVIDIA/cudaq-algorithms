# Simulated QSVT transform

Status: draft. Operation + object: **analyze** a **QSVT-transformed
good-subspace statevector**.

## Contract

`sim_utils.transform(transformer, ket, sequence, convention=None)` calls the
state-taking `QSVT.kernel`, runs it with `cudaq.get_state`, and returns the
unnormalized good-subspace block. On an eigenstate it represents
`p(lambda/alpha)|psi>` for the polynomial implemented by the sequence.

`ket` must be a one-dimensional statevector of exactly
`2**transformer.encoding.num_system` amplitudes. The helper does not check its
shape or normalization on the host before `state_from`; direct callers own
that check, and wrong-shape backend behavior is unverified.

## Classification and boundaries

- Public symbol/source: `cudaq_algorithms.sim_utils.transform`, `sim_utils.py`.
- Kind/role: simulation-only analysis, driver.
- Inputs: a concrete `QSVT`, statevector, and `PhaseSequence` or phase iterable.
- Output: unnormalized NumPy complex vector.
- Phase convention, walk directions, polynomial validity, and global phase are
  governed by [qsvt-sequence.md](../qsvt/qsvt-sequence.md).
- This is not a device kernel, QPU protocol, shot-based observable, or normalized
  conditional state.

Independent oracle: use the 2x2 signal response or dense eigendecomposition,
translate phase convention and global phase, then compare the entire good
block. Include a wrong-dimension ket and conflicting convention retag as
adversarial cases. Runnable pointers are the `sim.transform` cases in
`tests/python/test_qsvt.py` and the QSVT examples named in
[application-composition.md](../application-composition.md). Current public
source and tests are authoritative and must be rechecked at use time; this
record was not freshly executed.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context. No resource estimator exists; statevector simulation work is not
a logical-circuit estimate or measured performance.

Declared eval coverage: `simulation-analysis-hardware-boundary`,
`qsvt-phase-and-recovery-boundary`, and `qsvt-paraphrase-convention`.
