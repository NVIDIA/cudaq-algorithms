# Givens rotation schedule planning

Status: draft. Operation + object: **preprocess** an **orthonormal
orbital-coefficient matrix into a Givens rotation schedule**.

## Identity and classification

- Public symbols in `cudaq_algorithms.stateprep`:
  `make_givens_rotation_schedule`, `validate_givens_rotation_schedule`,
  `GivensRotation`, `GivensRotationSchedule`, and the three
  `get_givens_rotation_*` accessors.
- Source/tests: `python/cudaq_algorithms/stateprep/_givens.py` and
  `tests/python/test_stateprep_givens.py`. Documentation and runnable usage are
  in `docs/sphinx/guide/state_prep.rst` and
  `docs/sphinx/examples/python/05_state_prep_and_injection.py`.
- Kind/role/layer: classical transformation, computational leaf, host.
- Input/output: orbital-coefficient matrix -> `GivensRotationSchedule`.
- Lifecycle/evidence: draft; this record was historically source-reviewed, with
  review provenance recorded in [Source provenance](../source-provenance.md). Current
  public source/tests are authoritative and must be checked at use time;
  unexecuted for this record.

## Scientific and input contract

`make_givens_rotation_schedule(Q, tolerance=1.0e-12)` accepts a real or complex
rank-2 NumPy array or rectangular nested list of shape
`(num_spin_orbitals, num_electrons)`. Columns are occupied orbitals and must be
normalized and mutually orthogonal; rows use the package's interleaved
alpha-even/beta-odd ordering when spin is represented.

The host algorithm eliminates sub-diagonal entries bottom-up with adjacent-row
rotations, then reverses elimination order into circuit application order. A
complex NumPy dtype selects the complex route even if values are real; nested
Python or NumPy complex scalars also select it. Real schedules have zero phases;
complex schedules retain relative rotation phases and one final phase per
electron.

The factory raises `ValueError` for empty input, no occupied column, more
occupied columns than rows, ragged rows, a column residual greater than
`100 * tolerance`, or an overlap greater than `100 * tolerance`. Values at or
below `tolerance` are treated as zero during elimination and phase extraction,
so this threshold can change the represented determinant; source gives no
aggregate fidelity bound. Non-finite or negative tolerances have no supported
contract.

## Output, composition, and rejection

The frozen schedule stores positive spin-orbital/electron counts, `is_complex`,
adjacent `GivensRotation` entries in application order, and final phases.
`validate_givens_rotation_schedule` rejects nonpositive or inconsistent counts,
out-of-range/nonadjacent rotations, and invalid final-phase lengths. Factory
output is validated before return; hand-built schedules require an explicit
validator call. The accessors return plain flattened indices, angles, and
phases for kernel capture.

This record provides no capability. Its concrete consumers are
[the injectable kernel](state-preparation-slater-determinant-kernel.md) and
[the Givens resource estimator](state-preparation-resources-givens.md). Host
planning runtime and memory are neither estimated nor measured.

## Accuracy and validation

The schedule is deterministic and exact for its retained elimination, subject
to floating-point synthesis and the threshold behavior above. Interleaved row
ordering is an unchecked caller obligation.

Validate analytic two-orbital output; real/complex shape, phase, and flattened-
list invariants; and every rejection above. End-to-end schedule semantics use
the kernel consumer and independent dense-minor or second-quantized determinant
oracles. Runnable cases include `test_givens_schedule_two_orbital_statevector`,
`test_real_schedule_shape_and_resources`,
`test_complex_schedule_shape_and_resources`, both `test_validate_schedule_*`
cases, and `test_orbital_coefficient_validation_errors`. Statevector tolerances
are `1e-12` for fp64 or `5e-5` for fp32; structural assertions are exact.

Declared eval coverage is `state-preparation-provider-selection`; it must keep
schedule data distinct from the emitted kernel. Evals have not been run.
