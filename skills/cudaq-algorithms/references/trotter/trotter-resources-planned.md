# Planned Trotter resource estimate

Operation + object: **estimate** a **logical circuit description
for an already validated `Trotter` plan**.

## Identity and classification

- Public method: `cudaq_algorithms.Trotter.resources(steps, order)`.
- Result: root-exported `cudaq_algorithms.TrotterResourceEstimate`.
- Source/tests: `python/cudaq_algorithms/trotter.py`, `test_trotter.py`.
- Kind/role/layer: resource estimator, computational leaf, host.
- Resource status: exact formula-level term and rotation counts plus a
  documented logical CNOT decomposition proxy; not measured hardware cost.

## Contract

`Trotter` construction first validates and canonicalizes the Hamiltonian as
described in [term planning](trotter-planning.md): it prunes zero and
below-threshold coefficients, separates identity terms, validates common-width
Pauli words and real coefficients, and applies the selected term ordering.

`evolution.resources(steps, order)` passes those stored non-identity terms and
the separated identity coefficient to the raw estimator. Both arguments are
required. `steps` must be a positive integer and `order` must be one of
`{1,2,4}`.

For this wrapper, the result fields mean:

| Field | Meaning and status |
| --- | --- |
| `num_terms` | retained non-identity terms after construction-time pruning; exact |
| `steps` | validated positive step count; exact |
| `order` | validated formula order 1, 2, or 4; exact |
| `pauli_rotations` | `num_terms * steps * {1,2,6}[order]`; exact logical count |
| `estimated_cx_count` | two CNOTs per additional non-identity Pauli in each rotation; decomposition proxy |
| `identity_coefficient` | construction-time sum of retained identity coefficients; exact stored summary |

The identity coefficient is reported but is not a device rotation. Ordering
does not change these counts, although it can change numerical product-formula
error.

## Boundaries and composition

The estimate describes the same canonical terms held by this `Trotter` object,
but the caller must pass the same `steps` and `order` to the evolution factory;
the result is not attached to a previously built circuit. It does not inspect a
transpiled circuit or include target connectivity, routing, native gates,
optimization, state preparation, simulation cost, or an error bound. No
composition rule with other estimators is provided.

Use the [raw estimator](trotter-resources-raw.md) only when flattened lists are
already the owned representation and its weaker validation is acceptable.

## Oracle and runnable evidence

With retained Pauli weights `w_j` and multiplicity `m={1:1,2:2,4:6}`, verify
`pauli_rotations = T * steps * m` and
`estimated_cx_count = steps * m * sum_j(2 * max(w_j - 1, 0))`. Independently
check construction-time pruning and identity separation.

Run the authoritative wrapper cases from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_trotter.py \
  -k 'trotter_orders_terms_and_estimates_resources or zero_coefficient_terms_dropped or resources_requires_explicit_parameters'
```

Current public source and tests are authoritative and must be rechecked at use
time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
