# Fixed-parameter UCC resource estimate

Status: draft. Operation + object: **estimate** a **logical grouped
Pauli-rotation description for fixed-parameter UCC data**.

## Identity and classification

- Public function:
  `cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources(
  num_qubits, pauli_words)`.
- Result: frozen
  `cudaq_algorithms.stateprep.FixedParameterUccResourceEstimate`.
- Source/tests: `python/cudaq_algorithms/stateprep/_hartree_fock.py`,
  `tests/python/test_stateprep_hf_ucc.py`.
- Kind/role/layer: resource estimator, computational leaf, host.
- Lifecycle and implementation status: `draft`; the implementation was present
  and the record was historically source-reviewed; review provenance is recorded
  in [Source provenance](../source-provenance.md). Current public source/tests
  are authoritative and must be checked at use time; declared package/CUDA-Q
  compatibility remains unverified.

## Input and rejection contract

`num_qubits` must be a non-negative integer count; a boolean, negative value, or
fractional value raises `ValueError`. `pauli_words` is an outer iterable of
groups, and every group must support `len`. The function records only those
group lengths.

This estimator deliberately does **not** call `validate_fixed_parameter_ucc`.
It does not consume parameters or coefficients and does not validate Pauli-word
types, alphabet, width, coefficient reality, matching group lengths, or whether
a group represents a physical excitation. Empty outer input is accepted. Empty
groups are accepted and count toward `num_excitations` while contributing zero
rotations. Unsupported container shapes can raise ordinary Python `TypeError`;
no estimator-specific rejection wraps them.

For counts corresponding to an executable UCC product, validate the same
`num_qubits`, parameters, word groups, and coefficient groups with
`cudaq_algorithms.stateprep.validate_fixed_parameter_ucc` before interpreting
this structural estimate scientifically.

## Result formulas and status

Let `G=len(pauli_words)` after iterating it once into group sizes, and let
`n_g=len(pauli_words[g])`. The result is exactly:

| Field | Formula and meaning |
| --- | --- |
| `num_qubits` | validated `num_qubits`; input echo |
| `num_excitations` | `G`; number of outer groups, without semantic validation |
| `num_pauli_rotations` | `sum_g n_g`; one source-level `exp_pauli` call per listed term in the corresponding kernel path |
| `max_pauli_rotations_per_excitation` | `max_g n_g`, or `0` for no groups; maximum group size |

These are exact structural formulas over the supplied grouping. The maximum is
not circuit depth or a parallel schedule. The result does not depend on
parameters, coefficients, Pauli weights, term values, target decomposition, or
connectivity, and it supplies no CNOT/native-gate count, runtime, memory,
measurement, or ansatz-error estimate.

## Boundaries, validation, and evaluation

Compute group lengths independently and require exact integer equality. Probe
empty input, an empty group, malformed word width/alphabet, and mismatched
parameter/coefficient groups to keep this estimator's weak structural contract
separate from `validate_fixed_parameter_ucc`.

Run the authoritative source case from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_stateprep_hf_ucc.py \
  -k fixed_parameter_ucc_validation_and_resources
```

The task-local combined run reported in
[the resource front door](state-preparation-resources.md) included this case.
`state-preparation-provider-selection` now requires routing fixed-parameter UCC
inputs to this estimator and preserving the logical-estimate/hardware boundary;
`state-preparation-ucc-parameterization-boundary` separately checks grouping and
amplitude semantics. Neither asserts the result formulas or this estimator's
weak-validation boundary. No SkillEvaluator arm has been run, so this is
authored rather than validated coverage.
