# Kernel Boundaries

## Kernel boundary: data erasure and register-only signatures

- **Convention.** Kernel factories return `@cudaq.kernel` objects whose
  signatures carry registers only; all problem data is captured inside the
  kernel at factory time. The state-preparation seam is the one-register form
  `(qubits: cudaq.qview)`.
- **Consequence.** A multi-argument device kernel is not itself injectable at
  a register-only seam, even when it implements the same physics. Wrapping it
  in a one-argument kernel is the caller's explicit act, not an equivalence.
- **Mismatch symptom.** A kernel that will not launch, or a consumer factory
  that silently keeps its `cudaq.State` argument mode.
- **Source.** `python/cudaq_algorithms/block_encoding.py` module docstring
  ("data erasure at the kernel boundary"); `docs/sphinx/guide/state_prep.rst`
  ("State preparation injection").

## Register ownership at the current state-preparation seam

- **Source-derived behavior (current implementation only).** Every packaged
  consumer factory that accepts `state_prep` allocates the system register
  itself, so the register arrives fresh in `|0...0>`; the preparation kernel
  receives only that register, runs before the consumer operation, and the
  consumer allocates ancilla, signal, or control registers afterwards. The
  preparation width must equal the consumer's system width exactly; this is
  not verifiable at factory time. Exact width remains required, but a mismatch
  may fail or silently act on an unintended register; runtime behavior is
  provider/consumer-dependent and unverified rather than a factory-time error.
- **Scope limit.** This describes the current injection seam only. Recheck the
  cited call sites; it is not a library-wide guarantee for future capabilities.
  General register ownership and input-state, width, ancilla, inverse, control,
  and failure semantics remain unresolved beyond the documented current boundary.
- **Unverified.** Dirty (non-zero) input registers, width-mismatch error type
  and message, and foreign-consumer injection are not characterized by source
  or tests.
- **Source.** `python/cudaq_algorithms/pauli_lcu.py`,
  `python/cudaq_algorithms/qubitization.py`, `python/cudaq_algorithms/qsvt.py`,
  `python/cudaq_algorithms/trotter.py` (the `state_prep` call sites);
  `docs/sphinx/guide/state_prep.rst`, at the quoted prose "Contract (shared
  with the other primitives)" rather than a section heading.

## Validation happens on the host, before device kernels

- **Convention.** Device kernels have no uniform host-validation or error
  contract, and a kernel `return` is ignored by the compiler. Specific positive
  guards can make malformed input a silent no-op; unchecked list shortages,
  indices, and geometry can instead fail at launch or produce a wrong unitary.
  Input validation therefore runs on the host before launch, and the library
  exposes callable `validate_*` helpers for supported hand-built inputs.
- **Rule.** An illegal argument must never silently produce a result. Where a
  device kernel is a silent no-op on invalid input, record it as a known
  implementation limitation, not as established semantics.
- **Mismatch symptom.** A circuit that runs and returns a plausible but wrong
  state instead of raising.
- **Source.** `python/cudaq_algorithms/stateprep/_givens.py` module docstring
  and kernel guards; `docs/sphinx/guide/state_prep.rst`, at the quoted prose
  "Validation runs at factory time" rather than a section heading; public
  `validate_givens_rotation_schedule`,
  `validate_hartree_fock_occupation`, `validate_fixed_parameter_ucc`.
