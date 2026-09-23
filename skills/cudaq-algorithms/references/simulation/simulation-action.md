# Simulated block-encoding action

Operation + object: **analyze** **`(H/alpha)|ket>` for a
Pauli LCU encoding**.

## Contract

`sim_utils.action(encoding, ket)` builds `encoding.encode_kernel()`, converts
the input with `state_from`, runs `cudaq.get_state`, and returns
`good_subspace(encoding, state)`. Multiply by `encoding.alpha` to recover
`H|ket>`.

`ket` must be a one-dimensional statevector of exactly
`2**encoding.num_system` amplitudes. Unlike `sim_utils.evolve`, this helper has
no host-side shape or normalization check before constructing `cudaq.State`;
direct callers must validate both, and backend failure behavior for a wrong
shape is not a library contract.

## Classification and boundaries

- Public symbol/source: `cudaq_algorithms.sim_utils.action`, `sim_utils.py`.
- Kind/role: simulation-only analysis, driver.
- Concrete input: `PauliLCU`. This helper calls `encode_kernel`, which is not a
  member of the shared `BlockEncoding` protocol; do not advertise it for every
  structurally conforming encoding.
- Output: unnormalized NumPy complex vector.
- No shots, QPU execution, postselection sampling, or success amplification.

Independent oracle: construct the dense Pauli matrix and compare the output to
`H @ ket / alpha` with a predeclared tolerance; separately check that the
squared output norm is the postselection probability for normalized `ket`.
Runnable pointers are the `sim.action` cases in
`tests/python/test_pauli_lcu.py` and
`docs/sphinx/examples/python/pauli_lcu_demo.py`. Current public source and tests
are authoritative and must be rechecked at use time; this record was not freshly
executed. [Source lookup](../source-provenance.md) gives shared current-source paths. No resource estimator exists; simulator statevector
cost is not a
quantum-resource or measured-performance result.
