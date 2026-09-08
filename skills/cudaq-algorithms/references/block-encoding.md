# Block encodings — family record

Status: draft. Operation + object: **encode** an **operator as a block of a
unitary**.

This file carries all three record types of
`assets/primitive-record-template.md`, kept visibly separate:

1. a **Representation Record** for the `BlockEncoding` protocol surface — the
   object `Walk`, `QSVT`, and the shared observables exchange;
2. a **Capability Record** for zero-flagged block-encoded operator access —
   the composition boundary, which here is additionally backed by a real
   `typing.Protocol` in source;
3. one **Primitive Record** for `PauliLCU`, the single packaged concrete
   encoding, instantiating the primitive-only headings once.

`Composite protocol` is omitted per its own instruction in the template:
`PauliLCU`'s abstraction level is `leaf operation`. The composite consumers
are documented in [qubitization.md](qubitization.md) and
[qsvt.md](qsvt.md); the constraints this family imposes on them remain under
"Composition with the packaged consumers" below.

`conventions.md` owns the cross-cutting qubit-ordering, Pauli-word, kernel-
boundary, and resource-abstraction rules this family depends on. Its "Records
still to develop" queue has been resolved with concise cross-cutting records
for block normalization, good-subspace geometry, and walk sign; this record
owns the full family-specific details.

## Family scope and source provenance

- Owner: CUDA-Q Algorithms Team.
- Family scope: represent a generally non-unitary operator `H` as the
  zero-ancilla-flagged block of a unitary `U_A`, and construct that unitary
  for a Pauli-sum input. Choosing the polynomial, phase sequence, moment
  order, or spectral algorithm that consumes the encoding is a consumer
  concern, out of this family.
- Public symbols and import paths:
  - `cudaq_algorithms.BlockEncoding` (re-export of
    `cudaq_algorithms.block_encoding.BlockEncoding`);
  - `cudaq_algorithms.PauliLCU`, `cudaq_algorithms.select_observable`
    (re-exports from `cudaq_algorithms.pauli_lcu`);
  - `cudaq_algorithms.block_encoding.Kernel`,
    `cudaq_algorithms.block_encoding.mint_cached_kernel`;
  - the module-namespaced composable device kernels
    `cudaq_algorithms.pauli_lcu.{prepare, unprepare, select,
    controlled_select, apply, walk, reflect_about_prepare, adjoint_walk,
    controlled_reflect_about_prepare, controlled_walk,
    controlled_adjoint_walk, apply_phase_sequence,
    apply_controlled_phase_sequence}` and the type aliases
    `HamiltonianLike`, `LCUKernelArgs`. `__init__.py` states these keep their
    module namespace deliberately: "their names are too generic to export from
    the package root".
- Source paths: `python/cudaq_algorithms/block_encoding.py`,
  `python/cudaq_algorithms/pauli_lcu.py`,
  `python/cudaq_algorithms/common_kernels.py` (the encoding-independent
  reflection and signal-phase kernels and the shared host validators);
  `python/cudaq_algorithms/qubitization.py` and
  `python/cudaq_algorithms/qsvt.py` for the consumer side;
  `python/cudaq_algorithms/sim_utils.py` for the simulation-only helpers.
- Authoritative tests: `tests/python/test_pauli_lcu.py`,
  `tests/python/test_block_encoding_protocol.py`,
  `tests/python/test_walk_qsvt_orchestration.py`,
  `tests/python/test_qubitization.py`, `tests/python/test_qsvt.py`,
  `tests/python/test_df_encoding.py`; shared oracles in
  `tests/python/dense_references.py`; simulation target pinned in
  `tests/python/conftest.py`.
- Authoritative documentation: `docs/sphinx/guide/block_encodings.rst`,
  `docs/sphinx/conventions.rst` ("Block encodings", "Qubitization walk",
  "QSVT", "Reflection *gate* vs reflection *observable*"),
  `docs/sphinx/guide/qubitization_qsvt.rst`,
  `docs/sphinx/examples_rst/block_encodings.rst`. Examples:
  `docs/sphinx/examples/python/01_quickstart_block_encoding.py`,
  `pauli_lcu_demo.py`, `06_bring_your_own_encoding.py`, `df_encoding.py`,
  `df_block_encoding.py`.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q
  pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`). No
  test, example, or kernel was executed for this record.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft.
- Replacement and migration notes: none; no symbol in this family is
  deprecated at the cited commit.
- **Evidence rule for this family.** Every test named below is *cited
  repository evidence* — a committed assertion read at the cited commit,
  labeled `derived` — never a fresh measurement. Nothing here was run.

---

## Representation Record — the `BlockEncoding` protocol surface

- Object name and canonical symbol: `BlockEncoding`
  (`cudaq_algorithms.block_encoding.BlockEncoding`, re-exported as
  `cudaq_algorithms.BlockEncoding`).
- Public type or structural form, and source path: a `@runtime_checkable`
  `typing.Protocol` with **twelve members** — three sizes, eight kernel
  factories, one observable hook — in
  `python/cudaq_algorithms/block_encoding.py`. Conformance is **structural**:
  implementations inherit from nothing.
  `test_block_encoding_protocol.py::test_pauli_lcu_satisfies_protocol` asserts
  both halves of that claim — `isinstance(PauliLCU(...), BlockEncoding)` holds
  while `BlockEncoding not in type(PauliLCU(...)).__mro__`.
- Mathematical meaning: an object that owns a unitary `U_A` on an ancilla
  (signal) register plus a system register with

  ```text
  (<0|_anc  x  I) U_A (|0>_anc  x  I) = H / alpha
  ```

  and that can emit that unitary, its PREPARE/UNPREPARE factors, its
  qubitization walk step, the controlled forms of both, and its odd-moment
  observable, as data-free device kernels.
- Members, exactly as declared in source:

| Member | Form | Meaning |
| --- | --- | --- |
| `num_system` | `int` | system qubits the encoded operator acts on |
| `num_ancilla` | `int` | ancilla/signal qubits flagging the block; consumers require `>= 1` |
| `alpha` | `float` | normalization; the flagged block is `H / alpha` |
| `prepare_kernel()` | `(ancilla: qview)` | PREPARE the ancilla superposition |
| `unprepare_kernel()` | `(ancilla: qview)` | PREPARE-dagger |
| `apply_kernel()` | `(ancilla: qview, system: qview)` | the full `U_A` |
| `controlled_apply_kernel()` | `(control_and_ancilla: qview, system: qview)` | `U_A` controlled by qubit 0 of the combined register |
| `walk_step_kernel()` | `(ancilla: qview, system: qview)` | one walk step `W`, block-encoding `-H/alpha` |
| `adjoint_walk_step_kernel()` | `(ancilla: qview, system: qview)` | one `W†` |
| `controlled_walk_step_kernel()` | `(control_and_ancilla: qview, system: qview)` | controlled `W` |
| `controlled_adjoint_walk_step_kernel()` | `(control_and_ancilla: qview, system: qview)` | controlled `W†` |
| `select_observable()` | returns `Any` (a `cudaq.SpinOperator` in the packaged provider) | odd-moment observable; see the optional-member boundary below |

- Shape, layout, ordering, dtype, and units: the three sizes are plain
  integers; the emitted objects are compiled CUDA-Q kernels. The source's
  return annotation is `Kernel = Any`, with the comment that CUDA-Q "exposes
  no stable public Python type" for a compiled kernel — so this boundary is
  **not** type-checkable, only structurally checkable. The operator carries the
  library's qubit-0-least-significant and position-equals-qubit-index Pauli
  conventions (`conventions.md`); `alpha` is dimensionless relative to the
  units of `H`.
- Normalization, sign, and phase convention:
  - the block of `U_A` is `+H/alpha`;
  - the block of the **walk step** is `-H/alpha`, the sign folded into the walk
    construction. The module docstring states `Walk.moment` and the QSVT
    response conventions rely on it.
- Required mathematical properties (applicability preconditions):
  `num_ancilla >= 1`; sizes and emitted kernels fixed and immutable after
  construction (consumers cache the minted kernels — see the Capability
  Record); kernel signatures exactly as tabulated, with all encoding data
  captured inside at factory time.
- **Signature argument order is not register allocation order.** The kernels
  take `(ancilla, system)`, while the packaged factory kernels *allocate* the
  system register first and the ancillas after it. Both statements are in
  source and they are about different things; conflating them mislocates the
  good subspace.
- Producers (>=2 required to justify this record):
  - `PauliLCU` — packaged, `python/cudaq_algorithms/pauli_lcu.py`;
  - `DoubleFactorizedEncoding` —
    `docs/sphinx/examples/python/df_encoding.py`, a full-scale worked example
    validated against a dense reference by `tests/python/test_df_encoding.py`;
    it is an **example, not a packaged provider**, and it omits one member
    (below);
  - `TwoTermLCU` — `docs/sphinx/examples/python/06_bring_your_own_encoding.py`,
    a from-scratch teaching implementation that leaves the controlled hooks and
    `select_observable` raising `NotImplementedError`;
  - `ForeignEncoding` and `MockBlockEncoding` — test doubles in
    `tests/python/test_block_encoding_protocol.py` and
    `tests/python/test_walk_qsvt_orchestration.py`.
- Consumers: `Walk` and `reflection_observable`
  (`python/cudaq_algorithms/qubitization.py`), `QSVT`
  (`python/cudaq_algorithms/qsvt.py`), `mint_cached_kernel`
  (`python/cudaq_algorithms/block_encoding.py`), and the simulation-only
  helpers `good_subspace` / `action` / `transform`
  (`python/cudaq_algorithms/sim_utils.py`).
- Invariants preserved across the boundary: *data erasure at the kernel
  boundary* (`block_encoding.py` module docstring) — every factory returns a
  `@cudaq.kernel` whose signature carries registers only, so a consumer can
  sequence the kernels without knowing anything about the encoding's internals
  and no status channel exists.
- Observable symptom of a misinterpretation:
  - reading the good subspace at the wrong offset (an ancilla-first allocation
    assumption) yields a wrong-normalized or scrambled block while total norm
    stays 1;
  - dropping the walk-step minus sign flips odd Chebyshev moments and shifts
    QSVT responses by the parity of the polynomial;
  - offering a multi-argument kernel where a protocol kernel is expected does
    not launch.
- Unsupported or ambiguous forms:
  - **The three sizes need not be Python properties.** The protocol declares
    them with `@property`, but `ForeignEncoding` and `MockBlockEncoding` set
    plain attributes and conform. The boundary requires attribute *access*.
  - **`isinstance` conformance does not mean the surface works.** It checks
    member presence, not signature or semantics. `DoubleFactorizedEncoding`
    passes `isinstance` (`test_df_encoding.py::
    test_satisfies_block_encoding_protocol`) while `select_observable()`
    raises `NotImplementedError`
    (`test_df_encoding.py::test_odd_moments_are_unavailable`, which also
    asserts that `Walk(encoding).moment(ket, 1)` propagates it). Treat
    `select_observable` as **optional in practice** and check it per
    implementation.
  - **`state_prep` is not on this boundary.** No protocol member mentions it;
    injection support comes from the concrete factories
    (`PauliLCU.encode_kernel`, `PauliLCU.walk_kernel`) or from the consumers
    (`Walk`, `QSVT`). `state-preparation.md` records the same non-implication
    from the other side.
  - **Non-zero flag subspaces are not representable.** The protocol is
    zero-flagged by definition and the consumers hard-code all-zero-state
    reflections and projector phases (`common_kernels.reflect_about_zero`,
    `signal_phase`).
- Source paths, tests, docs, and last verification: "Family scope and source
  provenance" above.

---

## Capability Record — zero-flagged block-encoded operator access

- Stable ID: `cudaq-algorithms.block-encoding.zero-flagged.v1`. Like
  `state-preparation.unitary.v1`, this only fits `architecture.md`'s
  `cudaq-algorithms.<capability-name>.v<major>` if `<capability-name>` is read
  as a dotted name. The final ID remains an **open** owner decision; nothing
  in source names a capability.
- Status: **provisional.** Several independent producers and consumers exist in
  source, tests, and CI-validated examples, so the boundary is past
  `candidate`. It is not verified: the only packaged provider is `PauliLCU`,
  and one member is already optional in practice.
- Contract type: **backed by a source-level protocol** —
  `cudaq_algorithms.block_encoding.BlockEncoding`, a `@runtime_checkable`
  `typing.Protocol`. This is the distinguishing feature of this family: the
  Python surface already exists, unlike the state-preparation seam. The
  *capability identifier* above is still documentation-only and is not a
  public symbol.
- Boundary representation and exact signature: the Representation Record above.
- Semantic invariants, `derived` from the protocol, both consumer modules, and
  the cited tests:
  1. **Zero-flagged block.** `(<0|_anc x I) U_A (|0>_anc x I) = H / alpha`.
  2. **`num_ancilla >= 1`.** `Walk.__init__`, `QSVT.__init__`, and
     `reflection_observable` each raise `ValueError` on a zero-ancilla
     encoding (`test_walk_qsvt_orchestration.py::
     test_consumers_reject_zero_ancilla_encodings`). The stated reason is
     physical, not defensive: the walk's sign comes from a reflection that is
     a no-op on an empty register.
  3. **Walk-step sign.** The walk step block-encodes `-H/alpha`.
  4. **Register-geometry contract, not register ownership.** Consumers
     allocate every register themselves: system first (from a `cudaq.State`,
     or in `|0...0>` when a `state_prep` kernel is supplied), then the ancilla
     register, or then a combined `[control, ancilla...]` register for the
     controlled forms. The encoding supplies only widths and kernels.
  5. **Immutability.** Consumers mint each factory once and cache it
     (`mint_cached_kernel`; `Walk.encoding` and `QSVT.encoding` are read-only
     properties, pinned by
     `test_qubitization.py::test_walk_encoding_is_read_only`). An encoding
     mutated after injection would serve stale circuits.
  6. **Control convention.** Controlled kernels take one combined register
     whose qubit 0 is the external control — CUDA-Q Python cannot mix a bare
     qubit with a separate register in one control set — and must reduce to
     the identity at control `|0>`.
  7. **Optional member.** `select_observable` is required only by odd-order
     `Walk.moment`; every kernel path and all even moments work without it.
- Register or shape geometry and ownership: invariant 4, with the exact
  statevector layouts under "Outputs" below.
- Convention requirements: `conventions.md` qubit ordering and Pauli words;
  the walk-sign and reflection-gate-versus-observable conventions of
  `docs/sphinx/conventions.rst`.
- Host/device/simulation boundary: providers validate and flatten data on the
  host at construction time; the emitted kernels are pure device code.
  `sim_utils` is simulation-only (it needs `cudaq.get_state`), and
  `docs/sphinx/conventions.rst` plus the `sim_utils` docstring both say the
  library classes themselves never call `get_state`.
- Providers, with source paths: the four producers in the Representation
  Record. Only `PauliLCU` is packaged; two are examples and two are test
  doubles.
- Consumers, with source paths:

| Consumer | Symbols | Protocol members it reaches for |
| --- | --- | --- |
| `python/cudaq_algorithms/qubitization.py` | `Walk.kernel`, `Walk.adjoint_kernel` | `num_system`, `num_ancilla`, `prepare_kernel`, `unprepare_kernel`, `walk_step_kernel` or `adjoint_walk_step_kernel` |
| | `Walk.roundtrip_kernel` | the above plus both step directions |
| | `Walk.controlled_kernel` | `prepare_kernel`, `unprepare_kernel`, `controlled_walk_step_kernel` |
| | `Walk.controlled_roundtrip_kernel` | the above plus `controlled_adjoint_walk_step_kernel` |
| | `Walk.moment` / `Walk.moments` | even order: geometry only, via `reflection_observable`; odd order: `select_observable` |
| | `reflection_observable` | `num_system`, `num_ancilla` only |
| `python/cudaq_algorithms/qsvt.py` | `QSVT.kernel` | `num_system`, `num_ancilla`, `apply_kernel` |
| | `QSVT.controlled_kernel` | `num_system`, `num_ancilla`, `controlled_apply_kernel` |
| `python/cudaq_algorithms/sim_utils.py` | `good_subspace`, `transform` | `num_system`, `num_ancilla` |
| | `action` | `num_system`, `num_ancilla`, **and `encode_kernel`** — not a protocol member |

  The factory-reach column is pinned by
  `test_walk_qsvt_orchestration.py::test_factory_call_accounting`, which
  asserts that `Walk(encoding).kernel(power=5)` mints `prepare`, `unprepare`,
  and `walk_step` exactly once each and that `QSVT(encoding).kernel(...)`
  mints `apply` exactly once. Two asymmetries, reported rather than smoothed
  over: `sim_utils.good_subspace` and `action` are *type-annotated*
  `encoding: PauliLCU` although `good_subspace` uses geometry only, and
  `action` additionally depends on `encode_kernel`, which the protocol does
  not declare — so `action` is not usable with an arbitrary conforming
  encoding.
- Unsupported and unverified conditions: the boundaries table near the end of
  this file.
- Promotion criteria to a public protocol or compiler IR operation, and the
  explicit decision still required: the Python protocol already exists, so the
  open promotion question is the opposite one — whether the *taxonomy*
  capability should be declared stable. That needs (a) a second **packaged**
  provider, (b) a decision on whether `select_observable` belongs in the
  protocol at all given that a CI-validated example cannot implement it,
  (c) a decision on whether `encode_kernel` / `walk_kernel` / `state_prep`
  support belong on the boundary or stay provider-specific, and (d) an
  explicit recorded team decision with an owner. None holds at the cited
  commit; no promotion is proposed.

---

## Primitive Record: `PauliLCU`

Everything from here to "External alignment" is the contract of the one
packaged concrete encoding.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths: `cudaq_algorithms.PauliLCU`;
  `cudaq_algorithms.select_observable`; the module-namespaced device kernels
  and type aliases listed under "Family scope and source provenance".
- Source paths: `python/cudaq_algorithms/pauli_lcu.py`, with the shared host
  validators (`_real_coefficient`, `_term_qubit_extent`, `_validate_power`,
  `_bit_projector`) and encoding-independent kernels (`reflect_about_zero`,
  `signal_phase`, and their controlled forms) in
  `python/cudaq_algorithms/common_kernels.py`.
- Authoritative tests: `tests/python/test_pauli_lcu.py` (the contract),
  `tests/python/test_qubitization.py` and `tests/python/test_qsvt.py`
  (behavior through the consumers),
  `tests/python/test_block_encoding_protocol.py` (protocol conformance).
- Authoritative documentation: `docs/sphinx/guide/block_encodings.rst`;
  `docs/sphinx/conventions.rst`; examples `01_quickstart_block_encoding.py`
  and `pauli_lcu_demo.py`.
- Package/CUDA-Q versions verified: **unverified**, as recorded above.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft.
- Replacement and migration notes: none.

## Classification

- Operation + mathematical object (primary identity): **encode** a **Pauli-sum
  operator** as a zero-flagged block of a unitary.
- Kind: quantum operation (kernel factories) with host classical
  decomposition, validation, and angle synthesis in the same class — the same
  provisional compound the catalog already records for state preparation.
- Routine role: **computational.** It performs one distinct, well-defined,
  independently usable task; `Walk` and `QSVT` are the drivers built on it.
  `mint_cached_kernel` is an **auxiliary** utility, documented here only
  because it explains the immutability requirement, and it gets no record.
- Abstraction level: leaf operation.
- Parameterization: construction-time. The Hamiltonian, width, ancilla count,
  and every flattened array are baked in at `__init__`; the emitted kernels
  take registers only. `walk_kernel(power=...)` is a factory argument, not a
  runtime kernel parameter.
- Execution layers: host preprocessing and validation, kernel factory, device
  kernel, and observable (`select_observable`).
- Input representations: `cudaq.SpinOperator`, `cudaq.SpinOperatorTerm`,
  `Mapping[str, complex]` of Pauli word to coefficient, or an iterable of
  `(coefficient, word)` pairs — the `HamiltonianLike` alias.
- Output representations: the `BlockEncoding` protocol surface (Representation
  Record above); plus two non-protocol kernel factories, a `cudaq.SpinOperator`
  observable, and the `LCUKernelArgs` flattened-array tuple.
- Domain: domain-independent. Chemistry use is a consumer of it, not part of
  its identity.
- Required dependencies: `cudaq`. The module itself imports only `math` and
  `cudaq`; NumPy enters through `common_kernels.state_from` and the
  simulation-only helpers.
- Optional dependencies: none.

Provisional metadata — record the value, do not route on it:

- Exactness: **exact** for the retained terms. The encoded block is exactly
  `H_kept / alpha_kept`; the only approximation is the deliberate dropping of
  terms below `coefficient_threshold` (see "Approximation controls").
- Uncertainty: deterministic construction. Randomness enters only through the
  consumer's measurement statistics, and `Walk.moment` uses `cudaq.observe`
  expectations rather than sampling.
- Method: direct.

## Scientific contract

- Purpose: turn a Hamiltonian written as a weighted sum of Pauli words into a
  unitary whose zero-ancilla block is that Hamiltonian divided by a known
  scalar, so that qubitization and QSVT can act on its spectrum.
- Mathematical definition. For `H = sum_i c_i P_i` with real `c_i` and Pauli
  words `P_i`:

  ```text
  alpha   = sum_i |c_i|                            (the LCU 1-norm)
  PREPARE : |0...0>_anc -> sum_i sqrt(|c_i| / alpha) |i>_anc
  SELECT  : |i>_anc x |psi> -> |i>_anc x sign(c_i) P_i |psi>
  U_A     = PREPARE† . SELECT . PREPARE
  (<0|_anc x I) U_A (|0>_anc x I) = sum_i (|c_i|/alpha) sign(c_i) P_i = H / alpha
  ```

  The magnitude of each coefficient rides in the PREPARE amplitude and the
  sign rides in SELECT — this split is why a single negative-coefficient term
  still encodes the negative operator
  (`test_pauli_lcu.py::test_single_term_negative_coefficient_keeps_sign`).
- Why and when to use: any Pauli-sum Hamiltonian, when the downstream
  algorithm needs coherent access to `H` (Chebyshev moments, QSVT/QSP
  polynomials, phase estimation on the walk) rather than a product-formula
  approximation of `e^{-iHt}`.
- When not to use: when the term count makes the LCU 1-norm or the SELECT cost
  unattractive and the operator has exploitable structure — the worked
  double-factorized example encoding exists for exactly that case and reaches
  the same consumers; when a product formula suffices
  (`cudaq_algorithms.trotter`); when the target operator is not a real-
  coefficient Pauli sum (rejected — see "Inputs").
- Approximation controls: `coefficient_threshold` (default `1e-12`) drops
  terms with `|c_i| < threshold` **before** `alpha` is computed, so the
  encoding is exact for the truncated operator rather than approximate for the
  original one. `include_identity=False` removes identity words from the
  encoded operator and from `alpha` while `constant_term` still reports their
  sum, so the encoded operator becomes `H - constant_term * I`; the caller
  must reinstate that shift classically. Both are exact, deliberate changes of
  which operator is encoded, not error-controlled approximations.

### PREPARE, SELECT, UNPREPARE, and the walk step

All four are real kernels in source, and their exact behavior matters for
composition:

- **PREPARE** (`pauli_lcu.prepare`) walks a binary amplitude tree over the
  ancilla register: one `ry(angles[0])` on `ancilla[0]`, then for each layer
  `1..num_ancilla-1` and each branch of that layer one `ry.ctrl` on
  `ancilla[layer]` controlled by `ancilla.front(layer)`, with `x` conjugation
  selecting the branch. Angles come from
  `_prepare_angles(probabilities)` with `probabilities[i] = |c_i| / alpha`,
  zero-padded to `2**num_ancilla`.
- **Term indexing.** `ancilla[0]` holds the **most significant** bit of the
  term index: bit `b` of the control pattern is bit `num_ancilla - 1 - b` of
  the term's position in retained-term order. This is consistent across
  `_prepare_angles`, `select`, and `select_observable`, and it runs opposite to
  the qubit-0-least-significant statevector convention — so the ancilla block
  index in a simulated statevector is the bit reversal of the term index.
  Irrelevant to the good subspace (all-zero either way); it matters only if
  you postselect a non-zero ancilla pattern.
- **SELECT** (`pauli_lcu.select`) loops over retained terms; for each it
  conjugates with `x` gates so the term's pattern becomes all-ones, applies
  each Pauli factor of the word as `x.ctrl` / `y.ctrl` / `z.ctrl` on the
  target system qubit controlled by the **whole** ancilla register, applies
  the sign as `z(ancilla[0])` when `num_ancilla == 1` or an
  `(num_ancilla-1)`-controlled `z` otherwise, then undoes the conjugation. It
  requires a non-empty ancilla register.
- **UNPREPARE** (`pauli_lcu.unprepare`) is a **hand-written** inverse, not
  `cudaq.adjoint(prepare, ...)`. The docstring records why: adjoint generation
  fails at runtime on `prepare`'s conditionally-conjugated rotations
  (cuda-quantum#4898) and silently mis-replays loop-carried classical updates
  (cuda-quantum#4897). The inverse property is pinned directly by
  `test_pauli_lcu.py::test_unprepare_inverts_prepare` at `atol=1e-12`.
- **Walk step** (`pauli_lcu.walk`) is SELECT, then a reflection about the
  PREPARE state: `unprepare`, `reflect_about_zero`, `prepare`. Since
  `reflect_about_zero` is `I - 2|0..0><0..0|`, conjugating the step by PREPARE
  gives `R_0 . U_A`, whose flagged block is `-<0|U_A|0> = -H/alpha` — which is
  why `Walk.kernel` sandwiches its `power` steps between PREPARE and
  UNPREPARE and why `docs/sphinx/conventions.rst` describes the step as
  `W = R U_A` with eigenphases `pi -/+ arccos(lambda/alpha)`. The two
  descriptions are the same operator up to that PREPARE conjugation; that
  equivalence is `derived` algebraically from the two kernel bodies, not
  asserted anywhere in source. What the tests do assert is the resulting
  block: `test_pauli_lcu.py::
  test_single_term_walk_keeps_minus_h_over_alpha_sign` pins the `-H/alpha`
  sign of the sandwiched circuit at `atol=1e-10`, and
  `test_qubitization.py::test_walk_kernel_options` pins that
  `Walk.kernel(power=2)` and `PauliLCU.walk_kernel(power=2)` produce the same
  state at `atol=1e-12`.
- **Guard style.** Every kernel uses positive `if n > 0:` blocks instead of
  early `return`, because a kernel `return` is silently ignored
  (cuda-quantum#4845); `test_pauli_lcu.py::
  test_kernels_tolerate_empty_register_views` pins that empty views are a
  no-op rather than a crash. This is the device-side face of the
  "validation happens on the host" convention in `conventions.md`.

## Inputs

- Arguments: `PauliLCU(hamiltonian, num_qubits=None, *,
  include_identity=True, coefficient_threshold=1e-12)`.
- Shapes/ranks: Pauli words are strings of exactly one character per qubit;
  all words in a mapping or pair-iterable input must share one length.
  `num_qubits` is a single optional integer.
- Dtypes/domains: coefficients must be real-valued. A `complex` whose
  imaginary part exceeds `1e-10` in absolute value is rejected; a real-valued
  complex such as `0.5 + 0j` is accepted. Words use the alphabet `IXYZ`.
- Units: none; `alpha` inherits the units of the coefficients.
- Ordering/layout: `word[0]` acts on qubit 0 (`conventions.md`). Retained-term
  order follows input iteration order and fixes the ancilla index assignment.
- Normalization: none required of the input. The class computes
  `alpha = sum |c_i|` over retained terms.
- Required mathematical properties: the input is a real-coefficient Pauli sum,
  hence automatically Hermitian. Duplicate words are **not** coalesced — two
  entries for the same word remain two LCU terms (each consuming ancilla index
  space and SELECT work) even though the encoded operator is unchanged.
- Validation and rejection behavior — all on the host at construction time,
  all `derived` from `_terms_from_input`, `_real_coefficient`, and
  `PauliLCU.__init__`:

| Condition | Behavior |
| --- | --- |
| unrecognized input type, including `str`/`bytes` | `TypeError`, message names `cudaq.SpinOperator`, the mapping, and the pair forms (`test_pauli_lcu.py::test_string_hamiltonian_rejected_with_type_error`) |
| no terms at all | `ValueError("hamiltonian has no terms")` |
| words of differing length | `ValueError("all Pauli words must have the same length")` |
| a character outside `IXYZ` | `ValueError("unsupported Pauli word: ...")` |
| coefficient with `abs(imag) > 1e-10` | `ValueError` naming complex coefficients, identically for every input form (`test_complex_coefficients_rejected_uniformly`) |
| `num_qubits != word length`, mapping or pair input | `ValueError` naming both values |
| `num_qubits < register extent`, `SpinOperator` input | `ValueError` naming the requested width and the required extent (`test_spin_operator_rejects_explicit_width_below_extent`) |
| every term dropped by the threshold | `ValueError("hamiltonian has no retained terms")` |

  Two input-path asymmetries worth stating before you rely on either: for a
  mapping or pair input `num_qubits` must **equal** the word length, whereas
  for a `SpinOperator` input any `num_qubits >= extent` is legal padding
  (`test_spin_operator_wider_explicit_width_pads`). And the register extent of
  a `SpinOperator` is the **largest targeted qubit + 1**, deliberately not
  CUDA-Q's `qubit_count`, which counts distinct targets and undercounts
  off-zero or gapped operators (`_term_qubit_extent`;
  `test_spin_operator_infers_register_extent`). A scalar identity term acts on
  no degrees and constrains no extent — CUDA-Q raises rather than returning a
  sentinel, and the helper catches that (`test_spin_operator_identity_term_extent`).

## Outputs

- Return type or emitted kernel signature:
  - the eight protocol kernel factories, with the signatures tabulated in the
    Representation Record;
  - two **non-protocol** factories: `encode_kernel(state_prep=None)` and
    `walk_kernel(power=1, state_prep=None)`, each returning a kernel taking
    one `cudaq.State` argument, or a zero-argument kernel when a
    `(qubits: cudaq.qview)` preparation kernel is supplied;
  - `select_observable()` returning a `cudaq.SpinOperator`;
  - inspection properties `num_system`, `num_ancilla`, `num_terms`, `alpha`,
    `constant_term`, `terms`, `kernel_args`, plus `__repr__`.
- Mathematical meaning:
  - `encode_kernel`: applies `U_A`, whose all-zero-ancilla block is
    `H / alpha`;
  - `walk_kernel(power=p)`: PREPARE, `p` walk steps, UNPREPARE — the
    all-zero-ancilla block is `T_p(-H/alpha)` applied to the input state;
  - `select_observable()`: `sum_i sign_i |i><i|_anc x P_i`, whose expectation
    after PREPARE and `p` walk steps *without* UNPREPARE is the odd Chebyshev
    moment `<T_{2p+1}(H/alpha)>`;
  - `terms` returns a copy of the retained `(coefficient, word)` list;
    `kernel_args` returns defensive copies of
    `(angles, term_controls, term_ops, term_lengths, term_signs)` for
    composing the module-level kernels inside a caller's own kernel. The
    internal `_kernel_data` is the uncopied twin used by the factories.
- Shape/register geometry:
  - `num_system` = the Pauli word length (or the validated `num_qubits`);
  - `num_ancilla = max(1, (num_retained_terms - 1).bit_length())`, i.e.
    `max(1, ceil(log2(num_terms)))`. **It is never 0**: a single-term or
    identity-only Hamiltonian gets one idle ancilla, both because the walk's
    sign needs a non-empty register to reflect and because an empty flattened
    list cannot cross the kernel boundary (cuda-quantum#4847).
  - Statevector layout, with qubit 0 least significant: the factory kernels
    allocate the **system register first**, so an uncontrolled circuit indexes
    as `sys + (anc << num_system)` and the **good subspace is the first
    `2**num_system` amplitudes** (`sim_utils.good_subspace`,
    `docs/sphinx/conventions.rst`). The controlled consumer kernels allocate
    system, then the combined `[control, ancilla...]` register, giving
    `sys + (ctrl << num_system) + (anc << (num_system + 1))` — the layout maps
    asserted in `test_qubitization.py::_controlled_layout_maps`.
- Normalization, sign, and phase: the block of `U_A` is `+H/alpha`; the block
  of the walk step is `-H/alpha`. `alpha` is the 1-norm of the retained
  coefficients, so `alpha >= ||H||` always holds and is generally loose. No
  packaged path tightens it.
- Observable or measurement interpretation: `select_observable` is measured on
  the combined register with the ancilla projectors at offsets
  `num_system + b` — the offset convention that matches system-first
  allocation. The even-moment counterpart `2|0..0><0..0| - I` needs only
  geometry, so `qubitization.reflection_observable` derives it without an
  encoding hook. Note the deliberate sign clash flagged in
  `docs/sphinx/conventions.rst`: the reflection **gate** is
  `I - 2|0..0><0..0|` while the reflection **observable** is
  `2|0..0><0..0| - I`.
- Error/status information: none at the device boundary. Everything is
  detected on the host at construction; the emitted kernels have no error
  channel.

## Capabilities and composition

Provided:

- Stable ID: `cudaq-algorithms.block-encoding.zero-flagged.v1`
- Direction: **provides**
- Owning family record: this file (Capability Record above)
- Boundary representation: the `BlockEncoding` protocol surface
- Semantic invariants: all seven capability invariants, including the optional
  status of `select_observable` — which `PauliLCU`, unlike the example
  encodings, does implement
- Shape/register geometry: `num_system` from the word length,
  `num_ancilla = max(1, ceil(log2(num_terms)))`, system-first allocation
- Normalization, sign, phase, and ordering: `alpha` = retained 1-norm; `U_A`
  block `+H/alpha`; walk-step block `-H/alpha`; `word[0]` on qubit 0;
  `ancilla[0]` the most significant term-index bit
- Convention requirements: `conventions.md` qubit ordering and Pauli words
- Host/device/simulation boundary: host construction and validation, device
  kernels, one host-built observable; no simulation-only dependency in the
  class itself
- Unsupported conditions: the boundaries table below

Required:

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`
- Direction: **requires**, and only optionally — the `state_prep` argument of
  `encode_kernel` and `walk_kernel`
- Owning family record: [state-preparation.md](state-preparation.md)
- Boundary representation: a one-argument `(qubits: cudaq.qview)` preparation
  kernel
- Semantic invariants and geometry: `PauliLCU` allocates a fresh system
  register of width `num_system` in `|0...0>`, calls `state_prep(system)`, then
  allocates the ancilla register and applies the encoding. The width must
  match exactly and this is not verifiable at factory time
- Unsupported conditions: those of the state-preparation record; note that
  `state_prep` is **not** a `BlockEncoding` member, so a conforming foreign
  encoding promises nothing about it

Compatibility requires matching IDs and compatible major versions. A
capability ID here is a documentation contract, not a Python protocol or
public symbol — even though this family happens to have a real protocol too.

## Accuracy and limitations

- Error behavior or bounds: exact for the retained-term operator. If terms are
  dropped by `coefficient_threshold`, the operator error is bounded by the
  dropped coefficient 1-norm, `||H - H_kept|| <= sum_{dropped} |c_i|`, because
  every Pauli word has unit spectral norm. That bound is `derived`
  analytically and is **not** stated in the source or tests; no packaged
  helper computes or reports it.
- Precision sensitivity: `_prepare_angles` fires its zero-rotation shortcut
  only on **exactly zero** subtree mass, precisely so that two genuinely tiny
  retained siblings still split correctly instead of one branch being silently
  zeroed (`test_prepare_angles_keep_tiny_sibling_terms`). Simulation precision
  dominates everything else: `docs/sphinx/conventions.rst` records that the
  default CUDA-Q target is fp32 with ~`1e-7` statevector error, while the
  library tests pin `qpp-cpu` (fp64) via `tests/python/conftest.py` — whose
  fixture honors `CUDAQ_DEFAULT_SIMULATOR` and falls back to `qpp-cpu` — and
  assert at `1e-10`..`1e-12`. A ~`1e-7` residual should be read as target
  precision before being read as a bug.
- Unsupported inputs: complex coefficients, non-`IXYZ` characters, ragged word
  lengths, `str`/`bytes`, an empty operator, an all-dropped operator, and a
  `num_qubits` inconsistent with the input — each with the specific exception
  in the Inputs table.
- Known implementation limitations, each recorded in source with an upstream
  issue:
  - a kernel `return` is silently ignored, forcing positive guards
    (cuda-quantum#4845);
  - empty flattened lists cannot cross the kernel boundary, forcing the
    minimum-one-ancilla normalization and the never-dereferenced `term_ops`
    padding for identity-only Hamiltonians (cuda-quantum#4847);
  - `cudaq.adjoint` cannot generate UNPREPARE (cuda-quantum#4898,
    cuda-quantum#4897), so the inverse is hand-written and separately tested;
  - a CUDA-Q Python control set cannot mix a bare qubit with a register, which
    is the whole reason for the combined `[control, ancilla...]` register.
- Unsupported versus unverified: see the explicit three-way labeling in the
  boundaries table below.

## Resources

No resource estimator exists for this family. `estimate_*_resources` helpers
exist only for state preparation and `TrotterResourceEstimate` for Trotter;
`pauli_lcu.py`, `block_encoding.py`, and `qubitization.py` contain no
estimator symbol.

*Deferred:* a packaged block-encoding resource estimator, and any depth, T/
Toffoli, transpiled-gate, runtime, or memory figure. Follow-up trigger: a
public estimator symbol appearing for this family, or a benchmark whose
execution conditions are recorded.

The counts below are `derived` by reading the kernel bodies at the cited
commit. Every one of them is a **count of pre-decomposition logical
operations, including multi-controlled gates**, at the abstraction level
`conventions.md` requires be kept distinct from transpiled gate counts.
Architecture and execution assumptions: none — these are circuit-structural
counts, independent of target, and they say nothing about depth or runtime.
Controlling parameters are `num_ancilla`, `num_terms`, and the Pauli weight of
each word.

| Quantity | Metric and unit | Value | Status |
| --- | --- | --- | --- |
| Qubits, uncontrolled path | logical qubits | `num_system + num_ancilla` | exact |
| Qubits, controlled path | logical qubits | `num_system + num_ancilla + 1` | exact |
| Ancilla width | logical qubits | `max(1, ceil(log2(num_terms)))` | exact |
| PREPARE angles | host floats | `2**num_ancilla - 1` | exact |
| PREPARE rotations | `ry` gates | `2**num_ancilla - 1`, of which `2**num_ancilla - 2` are controlled on `1..num_ancilla-1` ancillas | exact |
| PREPARE branch conjugation | `x` gates | `sum_{L=1}^{num_ancilla-1} L * 2**L` (one pair per zero bit of each branch index) | exact |
| UNPREPARE | as PREPARE | identical counts with negated angles | exact |
| SELECT controlled Paulis | `x/y/z.ctrl` gates, each with `num_ancilla` controls | `sum_i weight(P_i)`, the total non-identity character count over retained words | exact |
| SELECT sign phases | `z` or `(num_ancilla-1)`-controlled `z` | one per retained term with a negative coefficient | exact |
| SELECT pattern conjugation | `x` gates | `2 *` (zero bits in each term's ancilla pattern), summed over terms | exact |
| `reflect_about_zero` | `x` gates plus one `r1` | `2 * n` plus one `r1` (uncontrolled at `n == 1`, else `(n-1)`-controlled) | exact |
| `U_A` (`apply`) | composite | PREPARE + SELECT + UNPREPARE | exact |
| Walk step | composite | SELECT + UNPREPARE + `reflect_about_zero` + PREPARE | exact |
| `Walk.kernel(power=p)` | composite | PREPARE + `p` walk steps + UNPREPARE when `uncompute=True` | exact |
| `QSVT.kernel`, degree `d` | composite | `d` `U_A` applications, `d` zero reflections, `d + 1` signal phases | exact |

Composition rules that are known: ancilla width and PREPARE cost grow with the
**number of retained terms** (logarithmically in qubits, linearly in gates,
since `2**num_ancilla` is `num_terms` rounded up to a power of two), while
SELECT cost grows with the total Pauli weight. Duplicate words are not
coalesced, so they inflate both. Controlled variants add one control to every
SELECT gate and leave PREPARE/UNPREPARE uncontrolled — by design, so that the
PREPARE pair cancels at control `|0>`. `alpha` is the resource that matters
most to downstream query counts and it is *not* a circuit count: it is the
retained 1-norm, and reducing it requires a structurally different encoding,
which is the stated motivation for the double-factorized example.

## Validation

- Independent oracle: `tests/python/dense_references.py::dense_matrix` builds
  the dense Pauli-sum matrix directly in CUDA-Q's little-endian order by
  explicit bit manipulation, sharing no code path with the encoding; the
  Chebyshev references are built from the dense matrix by the recurrence in
  each test module; `tests/python/test_qsvt.py::reference_response` is the
  executable 2x2 signal model, which `docs/sphinx/conventions.rst` names the
  arbiter for QSVT disagreements. Example
  `01_quickstart_block_encoding.py` carries an independent dense oracle of its
  own, and `06_bring_your_own_encoding.py` a third.
- Invariants exercised by the committed suites:
  - encoded block equals `H/alpha` on random and basis input states;
  - single negative term keeps its sign; identity-only Hamiltonians encode a
    signed identity;
  - `walk_kernel(power=p)` block equals `T_p(-H/alpha)`;
  - `Walk.moments` equals dense `<T_k(H/alpha)>` for both parities;
  - UNPREPARE inverts PREPARE on an arbitrary ancilla state;
  - walk and adjoint walk round-trip to the identity, controlled and
    uncontrolled;
  - controlled circuits act as the identity at control `|0>` and reproduce the
    uncontrolled state at control `|1>`, half by half;
  - the module-level composable kernels agree with the `QSVT` factories;
  - a foreign encoding reaching the consumers only through the protocol
    reproduces `PauliLCU` results exactly;
  - empty register views are a no-op rather than a crash.
- Representative cases: 1-qubit asymmetric spectrum `{"I": 0.2, "X": 0.5,
  "Z": 0.3}`; the 2-qubit four-term `{"ZI": 0.70, "IZ": -0.43, "XX": 0.19,
  "YZ": 0.11}`; single-term `{"XZ": -0.5}`; identity-only `{"II": ±}`;
  off-zero `0.5 * spin.x(1)` and gapped `spin.x(0) + spin.z(3)` operators;
  walk powers 1..3 and moment orders 0..5; a mixed forward/adjoint QSVT
  sequence.
- Predeclared tolerances, as committed in the suites: `atol=1e-10` for
  dense-block and moment comparisons, `atol=1e-12` for circuit-versus-circuit
  and inverse-property comparisons, `pytest.approx` for scalar `alpha` and
  `constant_term`. Execution target is whatever `CUDAQ_DEFAULT_SIMULATOR`
  names, defaulting to `qpp-cpu` (fp64) via `tests/python/conftest.py`.
- Expected failure/adversarial cases: every row of the Inputs validation
  table; `Walk`/`QSVT`/`reflection_observable` rejecting a zero-ancilla
  encoding; negative and non-integral `power`, `order`, `count`, and an
  out-of-range `control_state`; `Walk.encoding` refusing assignment; the
  tiny-sibling angle regression; `DoubleFactorizedEncoding.select_observable`
  raising `NotImplementedError` and that propagating through odd
  `Walk.moment`.
- Reference results: the assertions themselves, at the paths and tolerances
  above.
- Evidence status per claim: every contract, geometry, convention, resource,
  and validation statement in this record is **`derived`** — from source or
  from a committed test assertion read at commit
  `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`. Nothing is `measured`: no test,
  example, or kernel was executed for this record. The version range is
  `unverified`. The truncation bound under "Accuracy and limitations" is
  `derived` analytically rather than from source. Anything in the boundaries
  table labeled `unverified` or `absent` is exactly that.

## Evaluation coverage

Declared coverage: `block-encoding-capability-boundary` in `evals/evals.json`
tests positive capability interpretation and the structural-conformance,
composition, and unmeasured-probability boundaries. It is backed by
Classification, Scientific contract, the Capability Record, and Accuracy and
limitations. The case has not been run with or without the skill, so no uplift
is claimed.

The following are additional coverage opportunities, not existing case IDs:

| Coverage opportunity | What this record must supply |
| --- | --- |
| Positive selection/application | Classification and Scientific contract: choosing `PauliLCU` for a real-coefficient Pauli sum, naming `alpha` as the retained 1-norm and `num_ancilla` as `max(1, ceil(log2(num_terms)))` |
| Convention or misconception | Outputs and the PREPARE/SELECT subsection: system-first allocation and the first-`2**num_system` good subspace; `U_A` block `+H/alpha` versus walk-step block `-H/alpha`; reflection gate versus observable sign |
| Capability composition | Capability Record plus "Composition with the packaged consumers": which factories each consumer entry point reaches, the combined-control-register convention, immutability, and the fact that `state_prep` is not a protocol member |
| Invalid/unsupported boundary | Inputs validation table and the boundaries table: complex coefficients, ragged words, zero-ancilla rejection, and refusing to confirm `isinstance` conformance as a working surface |
| Negative activation | Boundaries table: sparse-oracle encodings and sum/product/scale combinators have no API here, so the correct answer is that the contract is unavailable rather than a designed one |

## External alignment

- Literature conventions:
  - `alpha` is the *subnormalization* of the block encoding
    (`docs/sphinx/guide/block_encodings.rst`), and for an LCU it is the 1-norm
    of the coefficients, so `alpha >= ||H||` automatically;
  - the walk-step sign is the documented translation trap.
    `docs/sphinx/conventions.rst` states that this library's walk block-encodes
    `-H/alpha` with eigenphases `pi -/+ arccos(lambda/alpha)`, while much of
    the literature (Low–Chuang is named) uses the opposite reflection sign or
    the other operator order and quotes `+H/alpha` with eigenphases
    `+/- arccos(lambda/alpha)`. Translate before comparing; the library's
    consumers rely on the `-H/alpha` form;
  - the double-factorized example encoding cites von Burg et al., *PRX
    Quantum* **2**, 030305 (2021), arXiv:2007.14460, and
    `test_df_encoding.py::test_alpha_matches_published_lcu_one_norm` asserts
    its `alpha` reproduces `double_factorization_one_norm(..., "lcu")` up to
    the identity weight at `rel=1e-12`.
- External package translations: QSVT phase conventions are the adjacent
  translation surface — `PhaseSequence` accepts `qsvt` (projector) or `qsp`
  (QSPPACK) phases and doubles the latter, and
  `docs/sphinx/conventions.rst` flags that QSPPACK's native `W_x` rotation has
  imaginary off-diagonals while the circuit's response step is real. That
  contract belongs to a QSVT record, not this one; do not restate it as a
  block-encoding property.
- Known semantic differences: no external-package comparison of `PauliLCU`
  itself exists in the repository. Any comparison with OpenFermion,
  Qualtran, or similar LCU tooling is `unverified` here and must start from
  the Pauli-word and qubit-ordering translations in `conventions.md`.

---

## Composition with the packaged consumers

Source-backed only. [qubitization.md](qubitization.md) and
[qsvt.md](qsvt.md) own the consumer contracts; this section states only the
constraints this family places on those consumers.

- **Both consumers are generic over the protocol and reach nothing else.**
  `test_block_encoding_protocol.py` drives `Walk.moment` (orders 0..4),
  `Walk.controlled_roundtrip_kernel`, and `QSVT.kernel` through a
  `ForeignEncoding` that exposes only the protocol surface — no `kernel_args`,
  no `terms`, no `PauliLCU` inheritance — and asserts identical results to the
  `PauliLCU` it wraps at `atol=1e-12`/`1e-10`.
- **Sequencing.** `Walk.kernel(power=p)` emits PREPARE, `p` walk steps, and
  optionally UNPREPARE (`uncompute=True`, the default); the sandwich is what
  makes the composed operator the `R_0 . U_A` form of the walk.
  `Walk.roundtrip_kernel` appends `p` adjoint steps and is the identity.
  `QSVT.kernel` emits a signal phase, then per walk a direction-dependent pair
  of `U_A` and zero reflection, then a signal phase. A degree-0 sequence pads
  `walk_directions` with one unused entry, again because empty lists cannot
  cross the kernel boundary.
- **Which factories each entry point mints** is tabulated in the Capability
  Record and pinned by `test_factory_call_accounting`. Consequence for a
  partial implementation: an encoding missing only the controlled hooks still
  works for `Walk.kernel`, `Walk.moment` at even order, and nothing else that
  needs them — the failure surfaces at the factory call, not at construction.
- **Register geometry across the seam.** Uncontrolled: system, then ancilla.
  Controlled: system, then one combined `[control, ancilla...]` register, with
  `prep`/`unprep` applied to `control_and_ancilla.back(num_ancilla)` so the
  PREPARE pair stays uncontrolled and cancels at control `|0>`.
- **Moments.** `Walk.moment(ket, k)` returns `<T_k(H/alpha)>` with the sign
  convention handled internally and **no caller-side negation** — even `k`
  from the geometry-derived reflection observable after UNPREPARE, odd `k`
  from the encoding's `select_observable` without UNPREPARE. It requires
  exactly one of `ket` or `state_prep` and raises `ValueError` otherwise, and
  it uses `cudaq.observe`, so it is hardware-shaped rather than
  simulation-only.
- **QSVT response.** On an eigenstate of `H` with eigenvalue `lambda` the
  good-subspace block implements `p(lambda / alpha)` at the plain scaled
  eigenvalue, with the walk sign already folded in by the circuits
  (`qsvt.py` module docstring).
- **Immutability is a real requirement, not advice.** Both consumers mint each
  factory once through `mint_cached_kernel` and expose `encoding` as a
  read-only property, so an encoding mutated after injection would be served
  stale circuits.
- **Injection interplay.** `Walk`, `QSVT`, and `PauliLCU`'s own
  `encode_kernel`/`walk_kernel` all accept `state_prep`; the protocol does
  not. For a caller-supplied encoding, check the consumer, not the protocol.
- **Simulation-only postselection** lives in `sim_utils`
  (`good_subspace`, `action`, `transform`) and needs `cudaq.get_state`. The
  library classes never call it. `action` additionally requires
  `encode_kernel`, which is not a protocol member.

## Unsupported, absent, and unverified boundaries

Read each label literally: **unsupported** means source rejects or documents
it as unavailable; **absent** means no such API exists in the package;
**unverified** means no source or test characterizes it.

| Boundary | Label | Why the label |
| --- | --- | --- |
| Complex Hamiltonian coefficients | unsupported | `_real_coefficient` raises `ValueError` above `abs(imag) > 1e-10`, uniformly for every input form |
| Encoding a general non-Hermitian or non-Pauli operator | unsupported | the only input forms are real-coefficient Pauli sums, which are Hermitian by construction; there is no dense-matrix or general-operator input path |
| `num_ancilla == 0` | unsupported | `Walk`, `QSVT`, and `reflection_observable` each raise `ValueError`; `PauliLCU` never produces it |
| Non-zero flag subspaces | unsupported | the protocol is zero-flagged and the consumers hard-code all-zero reflections and projector phases |
| `select_observable` on every conforming encoding | unsupported for the worked example | `DoubleFactorizedEncoding` and `TwoTermLCU` raise `NotImplementedError`; odd `Walk.moment` propagates it. Even moments and all kernel paths are unaffected |
| Controlled hooks on the teaching example | unsupported there | `06_bring_your_own_encoding.py` raises `NotImplementedError` for the three controlled factories, by design |
| Sparse-oracle block encoding | absent | no such symbol, module, or documented contract exists in the package. It is a roadmap concept only; do not design an API for it |
| Encoding combinators for sum, product, or scale | absent | no combinator symbol exists. `alpha` and geometry are computed only inside a single encoding's constructor; roadmap concept only |
| Amplitude amplification or oblivious AA over an encoding | absent | no such symbol in the package |
| A packaged resource estimator for this family | absent | estimators exist only for state preparation and Trotter |
| Depth, T/Toffoli, transpiled-gate, runtime, or memory figures | absent | nothing in the family reports them, and `conventions.md` forbids presenting the logical counts above as any of them |
| `state_prep` as part of the protocol | absent | no protocol member mentions it; support is per concrete factory or consumer |
| Mutating an encoding after injecting it into `Walk`/`QSVT` | unsupported | kernels are cached against it and `encoding` is read-only |
| Signature or semantic checking at the protocol boundary | unverified as a safeguard | `isinstance` checks member presence only; the source's own `Kernel = Any` alias means the kernel types are not checkable |
| A tighter normalization than the 1-norm | absent | `alpha` is always the retained 1-norm; no packaged path minimizes it |
| An error bound for `coefficient_threshold` truncation | unverified in source | the dropped-1-norm bound in "Accuracy and limitations" is analytical; no helper computes or reports it |
| Dirty (non-zero) ancilla input, or ancilla reuse across encodings | unverified | every packaged path allocates its ancillas fresh in `|0...0>`; other input states are uncharacterized |
| Behavior of a width-mismatched injected `state_prep` | unverified | documented as not verifiable at factory time; no test exercises it and no error type is stated |
| Hardware execution of any circuit in this family | unverified | all cited evidence is simulator-shaped assertions in CI; nothing was executed for this record |

Do not hand-roll an alternative for an `absent` row and present it as a
library capability, and do not convert an `unverified` row into a contract.
