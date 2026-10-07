# Good-subspace extraction

Operation + object: **extract** the **zero-ancilla amplitude
block from a simulated statevector**.

## Contract

`sim_utils.good_subspace(encoding, state)` checks a one-dimensional statevector
of size `2**(num_system + num_ancilla)` and returns a copy of its first
`2**num_system` amplitudes. This slice is correct because packaged factories
allocate system qubits first and CUDA-Q uses qubit 0 as the least-significant
bit.

The result is intentionally **unnormalized**. If the full input statevector is
normalized, the slice's squared norm is the postselection probability for the
all-zero ancilla block. The helper validates shape but not normalization; for
an arbitrary array the squared norm is only block weight, not a probability.
Normalize only when a downstream calculation explicitly asks for the
conditional state, and preserve the original norm separately.

## Classification and boundaries

- Public symbol/source: `cudaq_algorithms.sim_utils.good_subspace`,
  `sim_utils.py`.
- Kind/role: simulation-only analysis, computational leaf.
- Execution: pure host array validation/slicing once a full statevector exists.
- Input type annotation names `PauliLCU`, though the body uses only block
  geometry. Compatibility with arbitrary foreign encodings is unverified.
- No shots, measurement protocol, QPU execution, or amplitude amplification is
  performed.

Validate with basis states that put amplitude inside and outside the good block,
check the exact slice and norm, and reject wrong dimensions. Current public
source and tests are authoritative and must be rechecked at use time; this
record was not freshly executed.
[Source lookup](../source-provenance.md) gives shared current-source paths. The runnable pointers are the `good_subspace` cases in
`tests/python/test_pauli_lcu.py`. No resource estimator exists; array size and
slicing work are not measured runtime or memory claims.
