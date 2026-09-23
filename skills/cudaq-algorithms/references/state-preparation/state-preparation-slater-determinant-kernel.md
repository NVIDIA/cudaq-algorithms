# Injectable Slater-determinant kernel

Operation + object: **prepare** a **Slater-determinant quantum
state from a Givens rotation schedule**.

## Identity and classification

- Public symbol: `cudaq_algorithms.stateprep.slater_determinant_kernel`.
  Related device kernels `slater_determinant` and
  `complex_slater_determinant` are implementation branches, not this factory.
- Source/tests: `python/cudaq_algorithms/stateprep/_givens.py`,
  `tests/python/test_stateprep_givens.py`, and
  `tests/python/test_state_prep_injection.py`.
- Kind/role/layers: quantum operation, computational leaf; host validation and
  flattening, kernel factory, device kernel.
- Input/output: `GivensRotationSchedule` -> `(qubits: cudaq.qview) -> None`.
## Scientific and input contract

`slater_determinant_kernel(schedule)` validates and flattens its schedule, then
returns a one-argument kernel. Applied to a fresh all-zero register of exactly
`schedule.num_spin_orbitals` qubits, it prepares the schedule's
`num_electrons`-particle determinant. For a schedule built from matrix `Q`, the
amplitude on occupied set `S` is `det(Q[S, :])`, up to global phase and any
schedule-level threshold approximation.

Invalid hand-built counts, indices, adjacency, or phase-list lengths raise
`ValueError` at factory time. Captured data are flattened into plain lists. The
factory has real/complex branches with and without rotations; rotation-free
branches avoid the source-documented CUDA-Q failure to capture empty lists.

Exact width is required but cannot generally be checked before a consumer hands
the register to the kernel. A mismatch may fail, partially no-op, or prepare a
plausible wrong state; behavior is provider/consumer-dependent and unverified.

## Output, capability, and limitations

The kernel allocates no ancilla or control, measures nothing, and has no status
channel. It provides capability
`cudaq-algorithms.state-preparation.unitary.v1`, owned by
[injection-contract.md](injection-contract.md). Match its exact signature, width,
ordering, and fresh-register precondition to every consumer. Controlled,
adjoint, dirty-register, measurement-assisted, width-mismatch, and foreign-
consumer behavior retain the family record's unsupported/unverified labels.

The circuit is deterministic and exact for the validated schedule, subject to
floating-point gate-angle synthesis; source states no error bound. Schedule
construction semantics belong to
[state-preparation-givens-schedule.md](state-preparation-givens-schedule.md).
Logical proxies belong to
[state-preparation-resources-givens.md](state-preparation-resources-givens.md),
not this record.

## Validation

Independent oracles construct determinant amplitudes from dense minors and,
separately, dense Jordan–Wigner creation operators. Compare up to global phase
and check particle number. Factory cases are
`test_factory_matches_real_kernel_path`,
`test_factory_matches_complex_kernel_path`,
`test_factory_rotation_free_shapes`, and
`test_factory_injects_as_state_prep_kernel`; the injection seam is also tested
in `tests/python/test_state_prep_injection.py`. Givens statevector tolerances are
`1e-12` for fp64 or `5e-5` for fp32; the injection suite has separate fixed
tolerances.
