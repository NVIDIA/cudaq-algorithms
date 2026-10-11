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
  that silently keeps its `cudaq.State` argument mode: every consumer factory
  branches on `state_prep is None`, so omitting the `state_prep=` keyword
  yields the `cudaq.State`-taking kernel rather than an error.
- **Source.** `python/cudaq_algorithms/block_encoding.py` module docstring
  ("data erasure at the kernel boundary"); `docs/sphinx/guide/state_prep.rst`
  ("State preparation injection"); the `if state_prep is not None` branches
  in `pauli_lcu.py`, `qubitization.py`, `qsvt.py`, and `trotter.py`.

## Empty captured lists cannot cross the kernel boundary

- **Convention.** Every flattened list a factory captures into a kernel is
  non-empty. `PauliLCU` keeps `num_ancilla >= 1` even for a single term, pads
  an identity-only `term_ops` list to `[0, 0]`, and `QSVT.kernel` pads a
  degree-0 sequence's `walk_directions` with one unused forward entry
  (cuda-quantum#4847).
- **Consequence.** A caller composing the module-level kernels
  (`pauli_lcu.apply_phase_sequence`, `prepare`, `select`, ...) from
  `kernel_args` must pass the same non-empty lists; a hand-built encoding
  with `num_ancilla == 0` is rejected by `Walk`, `QSVT`, and
  `reflection_observable`.
- **Mismatch symptom.** A launch or marshalling failure on a degenerate
  input (one term, identity-only operator, degree-0 sequence) that works for
  every larger case.
- **Source.** `python/cudaq_algorithms/pauli_lcu.py` (`PauliLCU.__init__`
  comments and the `apply_phase_sequence` docstring);
  `python/cudaq_algorithms/qsvt.py` (`QSVT.kernel`, "empty list captures
  cannot be marshaled").

## Inverses are hand-written; `cudaq.adjoint` is not relied on

- **Convention.** Every packaged inverse is an explicit reversed kernel:
  `pauli_lcu.unprepare` reverses `prepare` gate by gate, `adjoint_walk`
  reorders two self-adjoint factors, and the primitives expose `kernel_adj` /
  `adjoint_kernel()` as literal gate reversals. `cudaq.adjoint` is not used
  because it fails at runtime on conditionally conjugated rotations
  (cuda-quantum#4898) and silently mis-replays loop-carried classical updates
  (cuda-quantum#4897).
- **Consequence.** A caller who needs the inverse of a packaged kernel calls
  the packaged inverse factory (`unprepare_kernel`, `adjoint_walk_step_kernel`,
  `Walk.adjoint_kernel`, `kernel_adj`), not `cudaq.adjoint` of the forward
  kernel. The inverse property is pinned by tests (`test_unprepare_inverts_prepare`,
  op-then-inverse identities in the primitives suites).
- **Mismatch symptom.** A runtime failure on an adjointed kernel, or an
  "inverse" that returns a wrong state without raising.
- **Source.** `python/cudaq_algorithms/pauli_lcu.py` (`unprepare`
  docstring); `python/cudaq_algorithms/primitives/_arithmetic.py`,
  `_unary_iteration.py`, `_alias_sampling.py` module docstrings.

## Controls live at qubit 0 of one combined register

- **Convention.** A CUDA-Q Python control set cannot mix a bare qubit with a
  separate register (qview). Every packaged controlled kernel therefore takes
  one contiguous `[control, ancilla...]` (or `[control, signal...]`) view
  whose qubit 0 is the external control, and every inner control set is a
  view of that same register. Consumers allocate that combined register
  themselves and initialize qubit 0 to `control_state`.
- **Consequence.** Encodings ship explicit `controlled_*` factories instead
  of relying on `cudaq.control` over the uncontrolled kernel. Kernels that
  call sub-kernels (for example the alias-sampling PREPARE) are not to be
  placed under `cudaq.control`: control-variant generation rejects nested
  kernel calls on CUDA-Q 0.15 (the package pins `cudaq >= 0.15.0, < 0.16`).
  The unary-iteration primitives take a separate one-qubit `control` view
  instead of a leading qubit, for the same reason.
- **Mismatch symptom.** A compile error on a mixed control set, or a
  controlled kernel that is not the identity at control `|0>`.
- **Source.** `python/cudaq_algorithms/block_encoding.py` module docstring
  ("Register conventions shared by all implementations");
  `python/cudaq_algorithms/qubitization.py` (`Walk.controlled_kernel`
  docstring); `python/cudaq_algorithms/pauli_lcu.py` (`controlled_select`);
  `python/cudaq_algorithms/primitives/_alias_sampling.py`,
  `_unary_iteration.py` module docstrings.

## Kernel language limits

- **Convention.** `@cudaq.kernel` functions must be defined in a real `.py`
  file; they do not compile from a `python -c` string or a shell heredoc. A
  ternary expression `x if c else y` inside a kernel is a `CompilerError`
  ("CUDA-Q does not currently support ternary IfExp expressions"); use an
  `if`/`else` block. A kernel `return` is silently ignored (cuda-quantum#4845),
  so guards are positive `if` blocks, never early returns.
- **Consequence.** Save every probe and checkpoint as a `.py` file before
  running it, and move data-dependent branching to the host or to explicit
  `if` statements.
- **Mismatch symptom.** A compile error that names the source line, or a
  guard that appears to work while the kernel keeps executing past it.
- **Source.** `docs/sphinx/examples/python/04_double_factorization_and_the_protocol.py`
  and `tests/python/test_df_encoding.py` ("CUDA-Q kernels need real .py
  files"); `python/cudaq_algorithms/pauli_lcu.py` and `common_kernels.py`
  (cuda-quantum#4845 comments); the ternary rejection is a CUDA-Q 0.15.1
  compiler diagnostic.

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
