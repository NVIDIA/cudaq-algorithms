# Givens schedule resource estimate

Status: draft. Operation + object: **estimate** a **logical circuit description
for a Givens rotation schedule**.

## Identity and classification

- Public function:
  `cudaq_algorithms.stateprep.estimate_givens_resources(schedule)`.
- Result: frozen
  `cudaq_algorithms.stateprep.GivensResourceEstimate`.
- Source/tests: `python/cudaq_algorithms/stateprep/_givens.py`,
  `tests/python/test_stateprep_givens.py`.
- Kind/role/layer: resource estimator, computational leaf, host.
- Lifecycle and implementation status: `draft`; the implementation was present
  and the record was historically source-reviewed; review provenance is recorded
  in [Source provenance](../source-provenance.md). Current public source/tests
  are authoritative and must be checked at use time; declared package/CUDA-Q
  compatibility remains unverified.

## Input and rejection contract

`schedule` is a
`cudaq_algorithms.stateprep.GivensRotationSchedule`. The function first calls
`validate_givens_rotation_schedule`; it does not construct or repair a schedule.
That validator raises `ValueError` when:

- `num_spin_orbitals <= 0`, `num_electrons <= 0`, or
  `num_electrons > num_spin_orbitals`;
- either orbital index of a rotation is outside
  `[0, num_spin_orbitals)`, or the two indices are not adjacent;
- a complex schedule does not have exactly one `final_phases` entry per
  electron; or
- a real schedule has a nonempty `final_phases` list whose length differs from
  `num_electrons`.

A real schedule with either no final phases or exactly one per electron passes
validation, but the estimator counts no phase rotations when `is_complex` is
false. The validator does not establish finiteness of rotation angles or phases,
nor does the estimator validate an orbital-coefficient matrix; use
`make_givens_rotation_schedule` for that separate preparation-planning contract.

## Result formulas and status

Let `N=schedule.num_spin_orbitals`, `E=schedule.num_electrons`, and
`R=len(schedule.rotations)`. Let `P=R+E` for a complex schedule and `P=0` for a
real schedule. The result is exactly:

| Field | Formula and meaning |
| --- | --- |
| `num_spin_orbitals` | `N`; input echo |
| `num_electrons` | `E`; input echo |
| `num_givens_rotations` | `R`; exact schedule length |
| `num_exp_pauli_calls` | `2 * R`; two source-level two-qubit `exp_pauli` calls per Givens rotation |
| `num_phase_rotations` | `P`; one `rz` per complex Givens rotation plus one per electron, or zero on the real path |
| `two_qubit_gate_count_proxy` | `2 * R`; equal to `num_exp_pauli_calls` |
| `depth_proxy` | `2 * R + P`; serial source-operation proxy |

These formulas are exact evaluations over the validated schedule. The result
does not count the initial `E` occupation-setting `X` gates. Its two proxy fields
are documented as decomposition-independent upper bounds, not transpiled gate
counts; `depth_proxy` is not scheduled, routed, or measured hardware depth.

## Boundaries, validation, and evaluation

The estimator neither inspects an emitted kernel nor includes native
decomposition, connectivity, compilation, runtime, memory, state-preparation
error, or a composition rule with another estimator. Integer equality against
the formulas above is the independent oracle; no numerical tolerance applies.

Run the focused source cases from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_stateprep_givens.py \
  -k 'basis_determinant_schedule_has_no_rotations or real_schedule_shape_and_resources or complex_schedule_shape_and_resources or validate_schedule_rejects_non_adjacent_rotation or validate_schedule_rejects_invalid_counts_and_indices'
```

The task-local combined run reported in
[the resource front door](state-preparation-resources.md) included all five
cases. `state-preparation-provider-selection` now requires routing the Givens
input to this estimator and preserving the logical-estimate/hardware boundary;
it does not assert the field formulas or schedule rejection matrix. No
SkillEvaluator arm has been run, so this is authored rather than validated
coverage.
