# State-input Trotter kernel factory

Status: draft. Operation + object: **evolve** a **caller-supplied
`cudaq.State` under a stored Suzuki–Trotter plan**.

## Identity and classification

- Public entry point: `cudaq_algorithms.Trotter.state_kernel(time, steps=1,
  order=2)`.
- Source/tests: `python/cudaq_algorithms/trotter.py` and
  `tests/python/test_trotter.py`.
- Kind/role/layers: quantum operation, computational composite; host validation,
  kernel factory, device kernel.
- Input/output: stored `Trotter` plan and evolution controls -> kernel with one
  `cudaq.State` argument.
- Lifecycle/evidence: draft; current public source and tests are authoritative
  and must be rechecked at use time; unexecuted for this record.
- Source provenance: [source-provenance.md](../source-provenance.md) records
  historical last-review audit context.

## Scientific, input, and output contract

The factory validates finite `time`, positive integral `steps`, and `order` in
`{1, 2, 4}`, using the same stored non-identity terms and formula as
[`Trotter.kernel`](trotter-kernel-factory.md). It emits exactly:

```text
(state: cudaq.State) -> None
```

At invocation, `cudaq.qvector(state)` creates the live register and the product
formula evolves it. The supplied state must have dimension
`2**Trotter.num_qubits`. This factory does not validate that geometry; direct
callers are responsible, and the identity-only branch cannot detect a mismatch.
`sim_utils.evolve` performs the shape check before invoking this contract.

The stored plan's pruning and ordering are fixed at construction. Identity
terms are omitted from the circuit, leaving global factor
`exp(-i * identity_coefficient * time)`; an identity-only plan returns the input
state unchanged. Product-formula error depends on noncommutation, time, steps,
ordering, and pruning, with no packaged error bound or step selector.

## Composition, boundaries, and resources

This factory accepts no `state_prep` and provides no state-preparation
capability. It is the circuit boundary used by
[simulated evolution](../simulation/simulation-evolve.md), but it remains a kernel factory,
not a statevector-returning simulation helper. Use
[`Trotter.kernel`](trotter-kernel-factory.md) for a zero-argument, sampleable
circuit with optional injected preparation, or
[apply_trotter](trotter-apply-kernel.md) for a caller-owned live qview.

Resource claims belong to
[the planned estimator](trotter-resources-planned.md). They exclude state
loading, simulator work, transpiled gates, hardware depth, runtime, and memory.

## Validation and evaluation

Compare the output against a dense product-formula or `exp(-iHt)` oracle while
handling the identity phase explicitly. Cross-check it against
`Trotter.kernel(state_prep=...)` on the identical prepared ket, exercise the
identity-only state branch, and verify wrong geometry through
`sim_utils.evolve`'s host guard rather than claiming this factory rejects it.
Relevant runnable cases are
`test_kernel_factory_accepts_state_prep_injection`,
`test_sim_utils_evolve_includes_identity_phase`,
`test_identity_only_hamiltonian_is_a_global_phase`, and
`test_sim_utils_evolve_validates_parameters` in
`tests/python/test_trotter.py`; tolerances are case-specific there.

Declared eval coverage is `trotter-evolution-resource-boundary`, contradictory
output-boundary cases, and application composition cases. Neither runtime
compatibility nor behavioral uplift has been verified.
