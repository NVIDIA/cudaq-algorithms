# QSVT sequence application

Status: draft. Operation + object: **transform** an **encoded spectrum with a
phase sequence**.

## Identity and classification

- Public symbols: `cudaq_algorithms.PhaseSequence`, `cudaq_algorithms.QSVT`.
- Source: `python/cudaq_algorithms/qsvt.py` and signal kernels in
  `common_kernels.py`.
- Tests: `test_qsvt.py`, `test_walk_qsvt_orchestration.py`,
  `test_block_encoding_protocol.py`, `test_df_encoding.py`.
- Examples: `02_hamiltonian_simulation.py`, `07_matrix_inversion_qsvt.py`,
  `hamiltonian_simulation_qsvt.py`.
- Kind/role/abstraction: quantum operation, driver, composite protocol.
- Layers: host phase validation, kernel factory, device kernel.
- Exactness: exact for supplied phases; polynomial design error is caller-owned.

## `PhaseSequence` representation

`PhaseSequence(phases, walk_directions=None, convention="qsvt")` stores:

- raw finite phases, length `d+1`, in radians;
- exactly `d` directions over forward/adjoint (0/1 and documented aliases);
- convention `qsvt` or `qsp`.

`phases` always remains raw. Circuits use `projector_phases`: unchanged for
`qsvt`, doubled for `qsp`. The QSP conversion is equivalent up to global phase
`exp(i * sum(phases))`. Re-tagging an existing sequence with a conflicting
convention is rejected because it would reinterpret rather than convert angles.
A bare iterable defaults to `qsvt`, so never pass QSPPACK angles without an
explicit tag.

The library checks finiteness, nonempty input, convention, and direction count.
It does not generate phases, prove polynomial boundedness, choose a degree, or
estimate approximation error.

## QSVT contract

`QSVT(encoding)` rejects `num_ancilla == 0`, caches the encoding, and exposes:

- `kernel(sequence, convention=None, state_prep=None)`;
- `controlled_kernel(sequence, convention=None, control_state=1,
  state_prep=None)`.

Without `state_prep`, each emitted kernel takes one `cudaq.State`. With the
one-register preparation seam, it is zero-argument. System is allocated first;
signal or `[control, signal...]` follows. The preparation runs once,
uncontrolled. Exact preparation width is required, but mismatch behavior is
provider/consumer-dependent and unverified.

The sequence interleaves projector phases with the encoding's `apply_kernel`
or `controlled_apply_kernel`, using each direction code to choose operation
order. A degree-0 sequence is legal and applies only a signal phase.

## Composition and limitations

- Requires `cudaq-algorithms.block-encoding.zero-flagged.v1`.
- Optionally consumes `cudaq-algorithms.state-preparation.unitary.v1`.
- QSVT uses `apply_kernel`, not `walk_step_kernel`; a caller-side walk-sign
  correction is wrong.
- `BlockEncoding` conformance does not prove scientific semantics or
  `state_prep` support for an unchecked consumer.
- Controlled kernels initialize their own control basis value; behavior with an
  externally supplied control superposition is not this factory's contract.
- Success amplification, phase generation, QSVT resource estimation, and
  hardware validation are absent or unverified.

## Resources

For degree `d`, the plain register width is `num_system + num_ancilla` and the
controlled width adds one qubit. The circuit has `d` encoding invocations,
`d` zero-state reflections, and `d+1` projector-phase blocks. These are exact
logical subcircuit counts, not decomposed gates, depth, runtime, memory, or a
degree/error bound.

## Validation and evaluation

Independent oracles in the repository include a 2x2 signal-model product,
dense eigendecomposition of the transformed block, a countable mock circuit,
and QSPPACK-generated cosine/sine phases compared with exact diagonalization.
Translate QSP global phase before comparison and predeclare tolerance; cited
tests use values from `1e-12` to `1e-8` by case.

Expected adversarial cases include negative eigenvalues, conflicting retags,
zero-ancilla encodings, degree zero, and control 0. Current public source and
tests are authoritative and must be rechecked at use time; this record was not
freshly executed. [source-provenance.md](../source-provenance.md) records
historical last-review audit context. Runnable pointers are
`tests/python/test_qsvt.py` and
`docs/sphinx/examples/python/02_hamiltonian_simulation.py`. Eval coverage includes
`qsvt-phase-and-recovery-boundary` and the application composition cases.

## External alignment

QSPPACK is optional and example-only. Its raw phases use the QSP convention;
the class handles the factor-of-two projector conversion, while callers or
[recovery](qsvt-recovery.md) handle the global phase. No foundational-paper
attribution should be made on the library's behalf unless separately sourced.
