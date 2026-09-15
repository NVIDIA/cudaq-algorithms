# Explicit-occupation Hartree-Fock resource estimate

Status: draft. Operation + object: **estimate** a **logical `X`-gate description
for an explicit Hartree-Fock occupation**.

## Identity and classification

- Public function:
  `cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources(
  num_qubits, occupied_orbitals)`.
- Result: frozen
  `cudaq_algorithms.stateprep.HartreeFockResourceEstimate`.
- Source/tests: `python/cudaq_algorithms/stateprep/_hartree_fock.py`,
  `tests/python/test_stateprep_hf_ucc.py`.
- Kind/role/layer: resource estimator, computational leaf, host.
- Lifecycle and implementation status: `draft`; the implementation was present
  and the record was historically source-reviewed; review provenance is recorded
  in [Source provenance](../source-provenance.md). Current public source/tests
  are authoritative and must be checked at use time; declared package/CUDA-Q
  compatibility remains unverified.

## Input and rejection contract

`num_qubits` must be a non-negative integer count. `occupied_orbitals` must be a
sized iterable because the estimator validates every entry and then calls
`len(occupied_orbitals)`. Each index must be a non-negative integer below
`num_qubits`, with no duplicates. Booleans, negative or fractional indices,
out-of-range indices, and duplicates raise `ValueError`. An empty occupation is
accepted, including with `num_qubits == 0`.

The function does not infer a canonical occupation, validate an electron-spin
relationship, sort the indices, or prepare a state. Index order does not affect
the result because only the validated length is counted.

## Result formulas and status

Let `N` be the validated `num_qubits` and `E=len(occupied_orbitals)`. The result
is exactly:

| Field | Formula and meaning |
| --- | --- |
| `num_qubits` | `N`; input echo |
| `num_electrons` | `E`; number of distinct validated occupied indices |
| `num_x_gates` | `E`; exact number of logical occupation-setting `X` operations |

These are logical pre-transpilation counts. The result does not retain the
indices, and it supplies no depth, routing, native-gate, runtime, memory,
measurement, or state-accuracy information. No bound or cross-estimator
composition rule is documented.

## Boundaries, validation, and evaluation

Use
[`estimate_hartree_fock_resources`](state-preparation-resources-hartree-fock.md)
when `(num_electrons, spin)` is the owned input and canonical occupation
generation is required. Exact integer equality with the formulas above is the
oracle; no numerical tolerance applies.

Run the focused source case from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_stateprep_hf_ucc.py \
  -k hartree_fock_host_helpers
```

The task-local combined run reported in
[the resource front door](state-preparation-resources.md) included this case.
`state-preparation-provider-selection` now requires routing explicit occupations
to this estimator and preserving the logical-estimate/hardware boundary; it does
not assert the field formulas or duplicate/out-of-range rejection. No
SkillEvaluator arm has been run, so this is authored rather than validated
coverage.
