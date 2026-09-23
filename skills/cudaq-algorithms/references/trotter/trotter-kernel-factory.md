# Zero-argument Trotter kernel factory

Operation + object: **evolve** a **newly allocated all-zero or
injected-preparation quantum state under a stored Suzuki–Trotter plan**.

## Identity and classification

- Public entry point: `cudaq_algorithms.Trotter.kernel(time, steps=1,
  order=2, state_prep=None)`.
- Source/tests: `python/cudaq_algorithms/trotter.py` and
  `tests/python/test_trotter.py`.
- Kind/role/layers: quantum operation, computational composite; host validation,
  kernel factory, device kernel.
- Input/output: stored `Trotter` plan plus evolution controls and optional
  one-register preparation -> zero-argument kernel.
- Source provenance: [Source lookup](../source-provenance.md) gives shared current-source paths.

## Scientific, input, and output contract

The factory validates finite `time`, positive integral `steps`, and `order` in
`{1, 2, 4}`. It allocates `Trotter.num_qubits` in `|0...0>`, optionally invokes
`state_prep(qubits)`, then applies the selected product formula over the stored
non-identity terms. The returned kernel always has signature `() -> None` and
is directly sampleable.

The stored plan comes from [Trotter planning](trotter-planning.md): coefficient
pruning and term ordering are fixed at object construction. Identity terms are
not device rotations; the omitted factor is
`exp(-i * identity_coefficient * time)`. For an identity-only plan, the factory
emits an identity circuit of the correct width, or exactly the supplied
preparation when `state_prep` is present, using a special branch that avoids
capturing empty lists.

First-, second-, and fourth-order formulas are distinct approximations. Error
depends on noncommutation, time, steps, ordering, and pruning; the API supplies
no a priori error bound or step selector.

## Capability, boundaries, and resources

Optional preparation consumes capability
`cudaq-algorithms.state-preparation.unitary.v1`. The consumer allocates the
fresh system register and runs preparation first. Exact width is required but
not generally checked at factory time; mismatch behavior can fail or silently
prepare a wrong state and is unverified. Preparation is not controlled, and no
controlled-Trotter factory is documented.

Use [Trotter.state_kernel](trotter-state-kernel-factory.md) for a caller-supplied
`cudaq.State`, [apply_trotter](trotter-apply-kernel.md) for a live qview, and
[simulation evolve](../simulation/simulation-evolve.md) for a returned statevector. Resource
claims belong to [the planned estimator](trotter-resources-planned.md); its
logical proxies are not hardware depth, runtime, or memory.

## Validation

Compare with dense `exp(-iHt)` while explicitly removing/accounting for the
identity global phase. Check zero-state evolution, injected-preparation
equivalence to the state-input factory, sampleability, identity-only branches,
invalid parameters, commuting/noncommuting cases, step/order scaling, and
many-term inputs. Runnable factory cases include
`test_kernel_factory_accepts_state_prep_injection`,
`test_kernel_factory_evolves_the_zero_state`, and
`test_identity_only_hamiltonian_kernel_factory_branches` in
`tests/python/test_trotter.py`; tolerances are case-specific there.
