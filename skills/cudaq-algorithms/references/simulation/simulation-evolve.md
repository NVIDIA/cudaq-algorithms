# Simulated Trotter evolution

Status: draft. Operation + object: **evolve** a **statevector with a Trotter
simulation helper**.

## Contract

`sim_utils.evolve(evolution, ket, time, steps=1, order=2,
include_identity_phase=True)` validates a one-dimensional ket of dimension
`2**evolution.num_qubits`, runs `Trotter.state_kernel` with `cudaq.get_state`,
and returns a NumPy complex vector.

By default it multiplies the result by
`exp(-i * identity_coefficient * time)`, restoring the global identity phase
that device circuits cannot represent. Disable that only when comparison up to
global phase is intended.

## Classification and boundaries

- Public symbol/source: `cudaq_algorithms.sim_utils.evolve`, `sim_utils.py`.
- Kind/role: simulation-only analysis, driver.
- Approximation: inherits time, steps, order, term ordering, and pruning from
  [the state-input Trotter kernel](../trotter/trotter-state-kernel-factory.md).
- Output is a statevector, not a kernel. It cannot replace `Trotter.kernel` in
  a hardware or composable-kernel interface.
- It performs no measurement and has no shot count.

Independent oracle: compare with dense `exp(-iHt) @ ket`, including a nonzero
identity term, and test both identity-phase settings. Reject wrong ket shape and
invalid evolution parameters. Current public source and tests are authoritative
and must be rechecked at use time; this record was not freshly executed.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.

The runnable pointer is
`tests/python/test_trotter.py::test_sim_utils_evolve_includes_identity_phase`,
with validation cases in `test_sim_utils_evolve_validates_parameters`. No
separate resource estimator belongs to this simulation helper; use the focused
Trotter resource records for logical circuit proxies, and do not infer measured
simulator cost.

Eval coverage includes `simulation-analysis-hardware-boundary` and application
composition cases.
