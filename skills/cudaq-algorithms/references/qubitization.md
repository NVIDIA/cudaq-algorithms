# Qubitization — family record

Status: draft. Two operation/object identities live in this file:

- **A. Walk kernels** — *evolve* a **quantum state** by the qubitization walk
  operator of a block-encoded operator (quantum operation, kernel factories).
- **B. Chebyshev moments** — *measure* a **block-encoded operator's spectral
  moments** and return classical floats (measurement/readout protocol).

They are one file because they share one object (`Walk`), one injected
encoding, and one sign convention — **not** because they are one contract. A
and B differ in kind, routine role, output representation, error surface, and
failure mode, and B additionally requires an encoding hook that A does not.
Every canonical heading below is instantiated once at family level, with `A.`
and `B.` subsections beneath it, per `assets/primitive-record-template.md`.

**Do not collapse A into B.** A returns circuits; the caller decides what to do
with them. B returns numbers through `cudaq.observe` and consumes A internally.
An answer that reports moments when the user asked for a walk circuit — or that
describes `Walk.moment` as "running the walk kernel" without naming the
observable, the parity split, and the classical `float` output — has lost the
distinction this record exists to preserve.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths:

| Symbol | Import path | What it is |
| --- | --- | --- |
| `Walk` | `cudaq_algorithms.Walk`, defined in `cudaq_algorithms.qubitization` | the family object; holds one encoding, emits A's kernels and B's numbers |
| `reflection_observable` | `cudaq_algorithms.reflection_observable`, defined in `cudaq_algorithms.qubitization` | host constructor for the even-moment observable `2\|0...0><0...0\| - I` |
| `select_observable` | `cudaq_algorithms.select_observable`, defined in **`cudaq_algorithms.pauli_lcu`** | the odd-moment observable, **LCU-specific**; `Walk` reaches it only through the encoding's `select_observable()` protocol hook, never by importing this symbol |

  `Walk`'s public methods are `kernel`, `adjoint_kernel`, `roundtrip_kernel`,
  `controlled_kernel`, `controlled_roundtrip_kernel` (contract A), `moment`,
  `moments` (contract B), and the read-only `encoding` property.
- Source paths: `python/cudaq_algorithms/qubitization.py` (the record's
  subject). Supporting sources it depends on, cited where used:
  `python/cudaq_algorithms/block_encoding.py` (the `BlockEncoding` protocol and
  `mint_cached_kernel`), `python/cudaq_algorithms/common_kernels.py`
  (`_validate_power`, `_validate_control_state`, `_bit_projector`,
  `state_from`, `reflect_about_zero`, `signal_phase`),
  `python/cudaq_algorithms/pauli_lcu.py` (the one packaged provider's walk
  kernels and its `select_observable`).
- Authoritative tests:

| Test file | What it pins for this record |
| --- | --- |
| `tests/python/test_qubitization.py` | moments against dense Chebyshev references; roundtrip identity; `power`/`order`/`count`/`control_state` rejection; controlled-layout semantics; read-only `encoding` |
| `tests/python/test_walk_qsvt_orchestration.py` | that `Walk` *sequences* injected kernels correctly (countable mock), factory-call accounting, zero-ancilla rejection |
| `tests/python/test_block_encoding_protocol.py` | that `Walk` reaches the encoding only through the protocol surface (foreign encoding equals `PauliLCU`) |
| `tests/python/test_state_prep_injection.py` | `state_prep` mode equals the `cudaq.State` mode for every A factory and for B; the one-of-two input rule |
| `tests/python/test_df_encoding.py` | even moments against a dense double-factorized reference; odd moments raising `NotImplementedError` |
| `tests/python/test_pauli_lcu.py` | the `-H/alpha` walk-block sign, including the single-term and identity-only regressions |

  Shared oracle helpers live in `tests/python/dense_references.py`
  (`dense_matrix`, `random_ket`).
- Authoritative documentation: `docs/sphinx/guide/qubitization_qsvt.rst`
  (primary); `docs/sphinx/guide/block_encodings.rst` (the protocol this record
  consumes); `docs/sphinx/conventions.rst`, sections "Qubitization walk" and
  the reflection gate-versus-observable warning; `docs/sphinx/api/python_api.rst`.
  Examples: `01_quickstart_block_encoding.py`, `03_chemistry_to_ground_state.py`,
  `05_state_prep_and_injection.py`, `06_bring_your_own_encoding.py`,
  `08_quantum_phase_estimation.py`, `pauli_lcu_demo.py`, `df_encoding.py`
  (all under `docs/sphinx/examples/python/`). Examples are evidence and
  recipes, never contract authority.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`), CUDA-Q
  pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`). No
  test in this family was executed while writing this record.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft.
- Replacement and migration notes: none. No symbol in this family is
  deprecated at the cited commit.
- **Evidence rule for this family.** Every test assertion named anywhere below
  is *cited repository evidence* read at the cited commit — labeled `derived`,
  never a fresh measurement, runtime, or performance figure.

## Classification

### A. Walk kernels

- Operation + mathematical object (primary identity): **evolve** a **quantum
  state** by the qubitization walk operator of a block-encoded operator.
- Kind: quantum operation (kernel factories).
- Routine role: **computational.** Each factory performs one distinct,
  well-defined, independently usable task — emit one walk circuit. It does not
  solve a complete user problem; the caller supplies the surrounding protocol
  (QPE, moment estimation, a Krylov solve).
- Abstraction level: **composite protocol.** Every factory sequences kernels
  obtained from a required lower-level capability rather than emitting gates of
  its own, so the "Composite protocol" heading below is mandatory.
- Parameterization: **construction-time.** The encoding is fixed at `Walk`
  construction; `power`, `uncompute`, `control_state`, and `state_prep` are
  fixed at factory-call time and baked into the emitted kernel. The emitted
  kernels take no algorithmic runtime parameters — at most one `cudaq.State`.
- Execution layers: host object and factory logic; kernel factory; device
  kernel. No simulation-only dependency: `qubitization.py` never calls
  `cudaq.get_state`.
- Input representations: a `BlockEncoding`-conforming object; optional
  one-argument preparation kernel; non-negative integers; `control_state` in
  `{0, 1}`.
- Output representations: a compiled CUDA-Q kernel, either `()` (with
  `state_prep`) or `(state: cudaq.State)` (without).
- Domain: domain-independent. The packaged provider `PauliLCU` and the
  double-factorized example encoding are chemistry-flavored inputs, but nothing
  in `qubitization.py` is chemistry-specific.
- Required dependencies: `cudaq`. `numpy` is reached indirectly through
  `common_kernels.state_from` when a `ket` is converted (contract B).
- Optional dependencies: none.

Provisional metadata — record the value, do not route on it:

- Exactness: **exact** as a circuit contract. The emitted circuit is exactly
  the sequenced composition; any inexactness comes from the injected encoding
  or from floating-point synthesis, not from this family.
- Uncertainty: deterministic.
- Method: direct.

### B. Chebyshev moments

- Operation + mathematical object (primary identity): **measure** the
  **Chebyshev moments of a block-encoded operator** in a given state.
- Kind: **measurement/readout** with classical output.
- Routine role: **driver.** It solves a complete user problem — "give me
  `<T_k(H/alpha)>`" — by sequencing a contract-A kernel, an observable, and
  `cudaq.observe`, and returning a number.
- Abstraction level: composite protocol (it consumes contract A plus an
  observable).
- Parameterization: construction-time for the encoding; per-call for `order` /
  `count`, the input state, and `state_prep`.
- Execution layers: host; kernel factory (via A); device kernel;
  observable/measurement through `cudaq.observe`; classical float output.
- Input representations: array-like `ket` or a built `cudaq.State`, **or** a
  one-argument preparation kernel — exactly one of the two; a non-negative
  integer `order` or `count`.
- Output representations: `float` (`moment`) and `list[float]` (`moments`).
- Domain: domain-independent.
- Required dependencies: `cudaq` (`cudaq.observe`), `numpy` (through
  `state_from`, for the array-like `ket` path).
- Optional dependencies: none.

Provisional metadata:

- Exactness: exact in the ideal-expectation limit; realized accuracy is set by
  the execution target's precision and by whatever estimator `cudaq.observe`
  uses on that target. The library passes no `shots_count`.
- Uncertainty: **deterministic on the statevector targets the tests use**;
  behavior on a shot-based target is **unverified** here — no source or test in
  this family characterizes sampling error for `Walk.moment`.
- Method: direct.

## Scientific contract

### Shared starting point: the encoding

A `BlockEncoding` supplies a unitary `U_A` on `num_system` system qubits plus
`num_ancilla >= 1` ancilla (signal) qubits with

```text
(<0|_anc (x) I) U_A (|0>_anc (x) I) = H / alpha
```

`alpha` is the subnormalization: the flagged block implements `H / alpha`, so
`alpha` must be at least the spectral norm of `H`
(`docs/sphinx/guide/block_encodings.rst`). For `PauliLCU`, `alpha` is the LCU
one-norm `sum_i |c_i|` over *retained* terms.

### A. Walk kernels

- **Purpose.** Emit circuits that apply powers of the qubitization walk
  operator over an injected encoding, in forward, adjoint, controlled, and
  roundtrip forms, in either input mode.
- **Mathematical definition.** The protocol requires
  `walk_step_kernel` to block-encode `-H/alpha`
  (`python/cudaq_algorithms/block_encoding.py`), and `Walk` sequences it as

```text
kernel(power=p, uncompute=True)   =  PREPARE, W^p, UNPREPARE
kernel(power=p, uncompute=False)  =  PREPARE, W^p
adjoint_kernel(power=p)           =  PREPARE, (W-dagger)^p, [UNPREPARE]
roundtrip_kernel(power=p)         =  PREPARE, W^p, (W-dagger)^p, UNPREPARE

controlled_kernel(power=p)        =  [x on control], PREPARE,
                                     controlled-W^p, [UNPREPARE]
controlled_roundtrip_kernel(p)    =  [x on control], PREPARE,
                                     controlled-W^p, controlled-(W-dagger)^p,
                                     UNPREPARE
```

  `[x on control]` is applied iff `control_state == 1`; `[UNPREPARE]` iff
  `uncompute`, and the two roundtrip factories have no `uncompute` parameter.
  In the controlled forms PREPARE and UNPREPARE act on the ancilla slice
  `control_and_ancilla.back(num_ancilla)` and stay **uncontrolled** — they
  cancel when the control is `|0>`, which is what makes the controlled circuit
  the exact identity there.

  With `uncompute=True` the good-subspace (all-zero-ancilla) block of the
  result is `T_p(-H/alpha)` applied to the input state — the Chebyshev
  polynomial of the walk block
  (`PauliLCU.walk_kernel` docstring; `docs/sphinx/conventions.rst`,
  "Qubitization walk"). `roundtrip_kernel` is the identity by construction and
  exists as a property test.

- **Two descriptions of one walk step, reconciled.** The module docstring says
  a step is "SELECT followed by a reflection about the PREPARE state";
  `docs/sphinx/conventions.rst` says `W = R U_A` with `R = I - 2|0><0|`. For
  the packaged provider these are the same operator in two frames, and the
  PREPARE/UNPREPARE sandwich makes them coincide. Composing the `pauli_lcu`
  kernel bodies (operators applied right to left):

```text
U_A  = PREPARE-dagger . SELECT . PREPARE                (apply)
W    = PREPARE . R0 . PREPARE-dagger . SELECT           (walk)
     = (reflection about the PREPARE state) . SELECT
PREPARE-dagger . W^p . PREPARE = (R0 . U_A)^p           (Walk.kernel body)
```

  so `Walk.kernel(power=p, uncompute=True)` realizes exactly `(R0 U_A)^p` with
  `R0 = I - 2|0...0><0...0|` on the ancillas. `derived` by composing
  `pauli_lcu.apply`, `pauli_lcu.walk`, and `Walk._factory` at the cited commit;
  the equality of the *frames* is provider-specific reasoning, whereas the
  general contract is only that the injected `walk_step_kernel` block-encodes
  `-H/alpha`.
- **Eigenphases.** `W`'s eigenphases are `pi -/+ arccos(lambda/alpha)` for an
  eigenvalue `lambda` of `H` (`docs/sphinx/conventions.rst`). Example 8 decodes
  a walk-QPE register with `E = -alpha * cos(theta)` after folding the
  conjugate mirror phases — consistent with that convention, and example-level
  evidence, not a library API.
- **Why and when to use.** When you need the walk circuit itself: QPE over `W`,
  a hand-built spectral protocol, a Chebyshev filter, or a hardware-shaped
  circuit to sample. Use `controlled_kernel` when an external control must
  select the walk (the QPE shape). Use `roundtrip_kernel` to validate an
  encoding or a target.
- **When not to use.** When you want the polynomial `p(H/alpha)` for an
  arbitrary polynomial — that is `QSVT`, a sibling consumer of the same
  encoding. When you want `exp(-iHt)` by a product formula — that is `Trotter`.
  When you only want the moments — use contract B rather than post-processing A
  yourself, because B owns the parity/observable convention.
- **Approximation controls.** None. `power` selects which exact operator is
  emitted; it is not an accuracy knob. Any approximation lives in the injected
  encoding (for example a truncated double factorization), not here.

### B. Chebyshev moments

- **Purpose.** Return `<psi| T_k(H/alpha) |psi>` as a Python float, through
  `cudaq.observe` — a hardware-legitimate measurement path, not statevector
  inspection (`docs/sphinx/guide/qubitization_qsvt.rst`).
- **Mathematical definition — the QEL even/odd convention.** With
  `p = order // 2`:

| Parity | Circuit | Observable | Source of the observable |
| --- | --- | --- | --- |
| even, `k = 2p` | `self.kernel(power=p, uncompute=True)` | `2\|0...0><0...0\| - I` on the ancillas | derived by `Walk` from register geometry alone — **no encoding hook** |
| odd, `k = 2p+1` | `self.kernel(power=p, uncompute=False)` | the encoding's `select_observable()` | delegated to the injected encoding — **encoding-specific hook** |

  Both return `<T_k(H/alpha)>` with **no caller-side negation**
  (`docs/sphinx/conventions.rst`). The even case is derivable: the flagged
  block after `p` steps is `T_p(-H/alpha)`, the reflection observable's
  expectation is `2 * P(ancillas = 0) - 1`, and
  `2 T_p(x)^2 - 1 = T_{2p}(x)` with `T_{2p}(-x) = T_{2p}(x)`, so the walk's
  minus sign cancels at even order. At **odd** order `T_{2p+1}(-x) =
  -T_{2p+1}(x)`, and the compensating sign lives inside the encoding-specific
  SELECT observable (for an LCU, `sum_i sign_i |i><i|_anc (x) P_i`). Treat the
  even derivation as `derived` and the odd sign compensation as the source's
  stated contract, pinned by the dense-reference tests — this record does not
  re-derive it.
- **Why and when to use.** Chebyshev moments are the input to spectral
  post-processing: example 3 feeds `walk.moments(reference, 2 * m)` into a
  classical Krylov solve and recovers a ground-state energy checked against
  FCI. Moments of `H/alpha` are dimensionless; multiply by `alpha` (and restore
  any offset excluded from the encoding, such as nuclear repulsion) to get
  physical energies.
- **When not to use.** When the encoding cannot supply `select_observable` and
  you need odd orders (see Accuracy and limitations). When you want the
  statevector block rather than an expectation — that is `sim_utils`,
  simulation-only. When you want an eigenvalue directly — moments are raw data,
  and the solver that turns them into an eigenvalue is **not** in this library.
- **Approximation controls.** None in the library. `count` sets how many
  moments you get, not their accuracy. Accuracy is governed by target precision
  and the `cudaq.observe` estimator.

### The classical layer is not in this library

`qubitization.py`'s module docstring names the "quantum exact Lanczos (QEL)"
even/odd convention, and example 3 cites Kirby, Motta and Mezzacapo,
*Quantum* **7**, 1018 (2023), arXiv:2208.00567. **No QEL, Lanczos, or quantum
Krylov public API exists in `cudaq_algorithms` at the cited commit.** The
`krylov_matrices` and `ground_eigenvalue` helpers used to turn moments into an
energy are defined inside
`docs/sphinx/examples/python/03_chemistry_to_ground_state.py` — example code,
not a package symbol. The roadmap lists "Quantum Exact Lanczos and quantum
Krylov methods" as a future item in the taxonomy design record; a roadmap card
is not a contract. Do not name, import, or invent such an API: the library's
output is the moment list, and the Krylov solve is the caller's.

## Inputs

### A. Walk kernels

| Argument | Where | Type/domain | Default | Notes |
| --- | --- | --- | --- | --- |
| `encoding` | `Walk(...)` | any object structurally satisfying `BlockEncoding` | — | must expose `num_ancilla >= 1`; conformance is structural (`typing.Protocol`, runtime-checkable), no inheritance |
| `power` | all A factories | non-negative integer | `1` | `power=0` is legal: PREPARE then UNPREPARE, the identity |
| `uncompute` | `kernel`, `adjoint_kernel`, `controlled_kernel` | bool | `True` | **not** a parameter of `roundtrip_kernel` or `controlled_roundtrip_kernel`, which always uncompute |
| `control_state` | `controlled_kernel`, `controlled_roundtrip_kernel` | exactly `0` or `1` | `1` | the value the control qubit is initialized to, via an `x` gate when `1` |
| `state_prep` | all A factories | one-argument kernel `(qubits: cudaq.qview)` or `None` | `None` | selects the input mode; see below |

- **Ordering/layout.** Registers are allocated **system first**, ancillas
  after, so with CUDA-Q's little-endian statevector order the good subspace is
  the first `2**num_system` amplitudes. Uncontrolled index
  `= sys + (anc << num_system)`; controlled index
  `= sys + (ctrl << num_system) + (anc << (num_system + 1))`, because the
  controlled variants take one combined `[control, ancilla...]` register whose
  qubit 0 is the external control (a CUDA-Q Python control set cannot mix a
  bare qubit with a separate register). Both maps are the ones the tests use.
- **Normalization.** None imposed here; the `ket`-mode `cudaq.State` is passed
  through to `cudaq.qvector(state)` unchanged.
- **Required mathematical properties.** `num_ancilla >= 1`. The encoding must
  be immutable after construction: `Walk` caches each minted kernel and each
  observable per instance, and the `encoding` property is read-only precisely
  so a swap cannot serve stale circuits.
- **Validation and rejection behavior (host-side, at construction or factory
  call).**

| Condition | Behavior |
| --- | --- |
| `encoding.num_ancilla == 0` | `Walk.__init__` raises `ValueError` naming `num_ancilla >= 1` |
| `power` negative or non-integral | `ValueError("power must be a non-negative integer")` from `_validate_power` |
| `control_state` not in `{0, 1}` | `ValueError("control_state must be 0 or 1")` from `_validate_control_state` |
| assigning `walk.encoding = ...` | `AttributeError` (read-only property) |
| `state_prep` width != `num_system` | **not checked.** Documented as not verifiable at factory time; fails at launch, with the error type and message **unverified** |

### B. Chebyshev moments

| Argument | Type/domain | Notes |
| --- | --- | --- |
| `ket` | array-like, a built `cudaq.State`, or `None` | array-like goes through `common_kernels.state_from`, which casts to `cudaq.complex()` so the dtype matches the active target's precision |
| `order` (`moment`) / `count` (`moments`) | non-negative integer | `count` is a *number of moments*: `moments(ket, n)` returns orders `0 .. n-1` |
| `state_prep` (keyword-only) | one-argument kernel or `None` | the hardware-shaped input mode |

- **Exactly one input mode.** `moment` and `moments` both raise
  `ValueError("provide exactly one of ket or state_prep")` when both or neither
  is supplied. The both-`None` guard is deliberate: a test records that the
  earlier fall-through into `state_from(None)` aborted the process in native
  code.
- **Rejection behavior.** `ValueError("order must be a non-negative integer")`;
  `ValueError("count must be a non-negative integer")`; plus every contract-A
  rejection above, since B builds an A kernel.
- **Not validated.** A `ket` whose dimension is not `2**num_system`; a
  non-normalized `ket`; a non-Hermitian encoded operator. None of these is
  checked in `qubitization.py`, and no test in this family exercises them —
  **unverified**, not "accepted".

### The two input modes (both contracts)

| Mode | How you pass the state | Emitted kernel signature | Character |
| --- | --- | --- | --- |
| data / `ket` | `ket` array-like or `cudaq.State`; the kernel does `cudaq.qvector(state)` | `(state: cudaq.State)` | simulation-friendly; the statevector crosses the API boundary |
| injected / `state_prep` | a `(qubits: cudaq.qview)` kernel | `()` — zero arguments, directly sampleable | hardware-shaped and fully synthesizable |

The two modes must agree numerically; `tests/python/test_state_prep_injection.py`
asserts exactly that for every A factory and for `moment`/`moments`. In
`state_prep` mode the consumer allocates the system register fresh in
`|0...0>`, runs the preparation on it, and only then allocates the ancilla (or
`[control, ancilla...]`) register — so in the **controlled** factories the
preparation runs once, **uncontrolled**, before the control qubit exists, which
is the QPE shape. The full seam contract, including the register-ownership
scope limit, is owned by
[state-preparation.md](state-preparation.md) and `conventions.md`; do not
restate it here as a library-wide policy.

## Outputs

### A. Walk kernels

- **Return type or emitted kernel signature.** A compiled CUDA-Q kernel;
  `Kernel` is aliased to `Any` in `block_encoding.py` because CUDA-Q exposes no
  stable public Python type. Signature is `()` in `state_prep` mode and
  `(state: cudaq.State)` otherwise.
- **Mathematical meaning.** As tabulated in the Scientific contract: with
  `uncompute=True` the all-zero-ancilla block is `T_power(-H/alpha)` acting on
  the input state; with `uncompute=False` the PREPARE superposition is left in
  place on the ancillas, which is what the odd-moment observable needs.
- **Shape/register geometry.** `num_system + num_ancilla` qubits uncontrolled,
  `num_system + 1 + num_ancilla` controlled, in the allocation order and index
  maps given under Inputs.
- **Normalization, sign, and phase.**
  - The walk step block-encodes `-H/alpha`, sign folded in, so no caller-side
    negation is ever correct.
  - **Reflection: same word, opposite sign.** The *gate*
    `common_kernels.reflect_about_zero` is `I - 2|0...0><0...0|` (the all-zero
    state acquires `-1`); the *observable* `qubitization.reflection_observable`
    is `2|0...0><0...0| - I` (expectation `+1` on the all-zero state).
    `docs/sphinx/conventions.rst` flags this as a standard source of sign bugs.
  - The zero reflection is implemented as `signal_phase(register, pi)` using
    `r1`, which carries **no global phase**, which is why the controlled
    variants reduce exactly to the identity at control `|0>` rather than to a
    phase.
  - Controlled contract: with control `|0>` the circuit is the identity up to
    the cancelling PREPARE/UNPREPARE pair; with control `|1>` it reproduces the
    uncontrolled result in the control-`1` half of the statevector. Both halves
    are asserted amplitude by amplitude in `test_controlled_walk_respects_control`.
- **Observable or measurement interpretation.** None — contract A emits
  circuits and measures nothing. `mz` appears nowhere in `qubitization.py`.
- **Error/status information.** None on the device: a kernel has no error
  channel, and a kernel `return` is silently ignored by the compiler
  (cuda-quantum#4845), so all validation is host-side and happens before the
  kernel exists.

### B. Chebyshev moments

- **Return type.** `moment` returns a Python `float`; `moments` returns
  `list[float]` of length `count`, in ascending order `T_0 .. T_{count-1}`.
  `moments(ket, 0)` returns `[]`.
- **Mathematical meaning.** `<psi| T_k(H/alpha) |psi>`, dimensionless. `T_0` is
  identically `1` for a normalized input, which is a cheap self-check
  (`test_single_term_encoding_walks_with_correct_sign` asserts
  `moment(ket, 0) == 1.0` to `1e-10`).
- **Shape/register geometry.** Both observables are built at ancilla offset
  `num_system` — `reflection_observable` from geometry alone, the LCU
  `select_observable` from the encoding's signed terms with the ancilla index
  bits laid out most-significant-first within the ancilla register.
- **Normalization, sign, and phase.** Returned with the `+<T_k(H/alpha)>`
  convention for both parities; see the Scientific contract. To reach physical
  units, multiply by `alpha` and restore any constant the encoding excluded.
- **Observable interpretation.** For the even branch the expectation is exactly
  `2 * P(ancillas measured all-zero) - 1`; `pauli_lcu_demo.py` and
  `test_walk_moments_match_chebyshev` compute the same quantity from
  postselection probabilities as a cross-check.
- **Error/status information.** No status object. Failures are Python
  exceptions: the `ValueError`s tabulated above, or `NotImplementedError`
  propagated from an encoding whose `select_observable` is unavailable.

## Capabilities and composition

### Required — encoded-operator access (a zero-flagged block encoding)

- Stable ID: `cudaq-algorithms.block-encoding.zero-flagged.v1`, the
  provisional documentation capability owned by
  [block-encoding.md](block-encoding.md).
- Direction: **requires** (both contracts).
- Owning family record: [block-encoding.md](block-encoding.md), backed by the
  source-level `python/cudaq_algorithms.block_encoding.BlockEncoding` protocol.
- Boundary representation and exact signatures. Three properties, eight kernel
  factories, and one observable hook. `Walk` uses the subset marked below;
  `QSVT` uses a different subset from the same protocol.

| Member | Returned kernel signature | Used by contract A | Used by contract B |
| --- | --- | --- | --- |
| `num_system` | `int` property | yes | yes |
| `num_ancilla` | `int` property | yes (guard) | yes |
| `alpha` | `float` property | not read by `Walk` | not read by `Walk` — it is the caller's rescaling factor |
| `prepare_kernel()` | `(ancilla: qview)` | every factory | via A |
| `unprepare_kernel()` | `(ancilla: qview)` | every factory | via A |
| `walk_step_kernel()` | `(ancilla: qview, system: qview)` | forward and roundtrip | via A |
| `adjoint_walk_step_kernel()` | `(ancilla: qview, system: qview)` | adjoint and roundtrip | — |
| `controlled_walk_step_kernel()` | `(control_and_ancilla: qview, system: qview)` | controlled forms | — |
| `controlled_adjoint_walk_step_kernel()` | `(control_and_ancilla: qview, system: qview)` | controlled roundtrip | — |
| `select_observable()` | `cudaq.SpinOperator` | — | **odd orders only** |
| `apply_kernel()` | `(ancilla: qview, system: qview)` | not used by `Walk` | — |
| `controlled_apply_kernel()` | `(control_and_ancilla: qview, system: qview)` | not used by `Walk` | — |

- Semantic invariants the provider must satisfy: `<0|_anc U_A |0>_anc =
  H/alpha`; the walk step block-encodes **`-H/alpha`**; controlled variants
  reduce to the identity at control `|0>`; `num_ancilla >= 1`; **data erasure at
  the kernel boundary** — every factory returns a kernel whose signature is
  registers only, with all encoding data captured inside at factory time;
  immutability after construction, since consumers cache the minted kernels.
- Shape/register geometry: system register allocated first, ancillas after;
  controlled variants take one combined `[control, ancilla...]` register with
  the control at qubit 0.
- Normalization, sign, phase, and ordering: `alpha` is the subnormalization;
  the `-H/alpha` walk sign is the provider's responsibility; qubit 0 is the
  least significant statevector bit, and Pauli-word position equals qubit index
  (`conventions.md`).
- Convention requirements: `conventions.md` qubit ordering and Pauli words; for
  chemistry inputs, the interleaved spin-orbital layout.
- Host/device/simulation boundary: the provider's host constructor validates
  and flattens; the returned kernels are pure device code; nothing in the
  boundary is simulation-only.
- Unsupported conditions: `num_ancilla == 0` (rejected by `Walk`, `QSVT`, and
  `reflection_observable`); a provider without `select_observable` (odd moments
  only — see below); a mutable provider (out of contract, behavior unverified).
- Known providers at the cited commit: `PauliLCU` (packaged, the only one in
  the library); `DoubleFactorizedEncoding` and `TwoTermLCU` (runnable examples,
  the former dense-validated in CI); `ForeignEncoding` and `MockBlockEncoding`
  (tests). Conformance is structural — `test_pauli_lcu_satisfies_protocol`
  asserts both that `PauliLCU` passes `isinstance(..., BlockEncoding)` and that
  `BlockEncoding` is deliberately **not** in its MRO.

### Required (optional at call time) — unitary state preparation

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`.
- Direction: requires, only when `state_prep` is passed.
- Owning family record: [state-preparation.md](state-preparation.md), which
  lists `Walk`'s seven `state_prep`-accepting entry points — the five A
  factories plus `moment` and `moments` — and owns the invariants. Read it
  rather than re-deriving the seam here.

### Provided

**None demonstrated.** No packaged consumer in `cudaq_algorithms` consumes
`Walk`'s output; contract A's kernels and contract B's floats are terminal
products handed to the caller. Even the walk-QPE example composes the
*encoding's* `controlled_walk_step_kernel` directly rather than going through
`Walk`. Declaring a provided capability ID here would freeze an interface that
no second party uses, which the architecture's extraction gate forbids. If a
future protocol consumes walk kernels, extract the capability then.

**No Capability Record section is instantiated in this file.** The reusable
block-encoding and state-preparation boundaries already have owning family
records. Recording either here would create a second home for another family's
contract.

### Composition boundaries to check before composing

- **`Walk` and `QSVT` are peers, not layers.** Both consume the same encoding
  and each keeps its own kernel cache; neither calls the other. `QSVT` uses
  `apply_kernel` / `controlled_apply_kernel`, which `Walk` never touches.
  Handing one encoding object to both is safe by construction — each consumer
  owns a separate cache and the encoding is immutable — but no test drives one
  encoding *instance* through both consumers at once, so treat that specific
  combination as `derived`, not test-pinned.
- **Observables belong to the uncontrolled layout only.** Both moment
  observables index ancillas starting at qubit `num_system`, which in the
  **controlled** layout is the *control* qubit. `Walk.moment` only ever builds
  uncontrolled kernels. Do not measure `reflection_observable` or
  `select_observable` against a `controlled_kernel` circuit — the indices do not
  line up, and no source or test supports that composition.
- **One `Walk` per encoding instance.** Kernels and observables are cached per
  `Walk`; `test_factory_call_accounting` asserts `Walk(encoding).kernel(power=5)`
  calls `prepare_kernel`, `unprepare_kernel`, and `walk_step_kernel` exactly
  once each, regardless of `power`.
- **`sim_utils` is a different boundary.** `good_subspace`, `action`, and
  `transform` require `cudaq.get_state` and are simulation-only.
  `Walk.moment`/`moments` deliberately stay in the library proper because
  `cudaq.observe` is a hardware-legitimate path. Do not present a `sim_utils`
  result as a hardware-reachable measurement.

## Composite protocol

Required because both contracts are composite protocols.

- **Required lower-level capabilities.** Encoded-operator access (mandatory);
  unitary state preparation (optional, per call). Contract B additionally
  requires the encoding's `select_observable` hook **for odd orders only**.
- **Canonical reference composition.**
  - A: allocate the system register (from `cudaq.State`, or fresh plus
    `state_prep`); allocate the ancilla register (or `[control, ancilla...]`,
    flipping the control when `control_state == 1`); `prepare`; `power`
    repetitions of the injected step kernel; optionally `unprepare`.
    `roundtrip_kernel` inserts `power` adjoint steps before the `unprepare` and
    always uncomputes.
  - B: `p = order // 2`; even order builds the `uncompute=True` kernel and pairs
    it with `reflection_observable(encoding)`; odd order builds the
    `uncompute=False` kernel and pairs it with `encoding.select_observable()`;
    then one `cudaq.observe(...).expectation()`, cast to `float`.
- **Default recipe and its applicability conditions.** `uncompute=True` and
  `power=1` are the defaults; `control_state=1` is the default for controlled
  forms. `uncompute=False` is applicable when a later stage needs the PREPARE
  superposition intact — which is exactly what the odd-moment observable
  requires and, in the source, its only use.
- **Materially different alternatives.** All are alternatives to A, not to B:

| Alternative | Source | Why it differs |
| --- | --- | --- |
| `PauliLCU.walk_kernel(power)` | `pauli_lcu.py` | LCU-only, always uncomputes, no adjoint/controlled/roundtrip forms. `test_walk_kernel_options` asserts it matches `Walk.kernel(power=…)` with `uncompute=True` to `1e-12`; that agreement is for this one option, not a general equivalence |
| module-level `pauli_lcu.walk`, `adjoint_walk`, `controlled_walk`, … composed inside your own `@cudaq.kernel`, with `enc.kernel_args` supplying the flattened arrays | `pauli_lcu.py` | full control, LCU-specific, no host validation, caller owns register layout |
| composing `encoding.controlled_walk_step_kernel()` directly | example 8 | needed when the surrounding protocol (a QPE counting register with a centered exponent schedule) does not match `Walk`'s register plan |
| `QSVT` | `qsvt.py` | a general polynomial response instead of Chebyshev powers |

- **Propagated conventions.** Qubit ordering, system-first allocation, the
  combined controlled register, and the `-H/alpha` sign all originate in the
  encoding/protocol layer and pass through unchanged. `Walk` adds only the
  parity convention of contract B.
- **Propagated errors.** `NotImplementedError` from an encoding's
  `select_observable` surfaces from `Walk.moment` at the first odd order and is
  not caught, wrapped, or downgraded. Encoding kernels are minted lazily and
  cached, so any exception from a provider factory surfaces on the first
  factory call that needs it, not at `Walk` construction.
- **Propagated resources and how they compose.** Qubit count is the encoding's
  `num_system + num_ancilla` (plus one control). The count of injected walk
  steps is exactly `power`; everything below a step is the encoding's cost and
  this record makes no claim about it.
- **Component substitution.** Supply any object satisfying the protocol; no
  inheritance or registration. A substitute must preserve every invariant above
  — in particular the `-H/alpha` walk sign and the controlled-identity property
  — or the moments will be silently wrong rather than raise.
  `tests/python/test_block_encoding_protocol.py` pins this by driving `Walk`
  with a foreign class that hides a `PauliLCU` behind the bare protocol surface
  and requiring identical results.

## Accuracy and limitations

- **Error behavior or bounds.** Contract A introduces no approximation.
  Contract B's realized error comes from the execution target and the
  `cudaq.observe` estimator, and from any approximation already inside the
  encoding (a compressed double factorization changes *which* `H` is encoded,
  and lowers `alpha`). No error bound is stated by the source and none is
  asserted here.
- **Precision sensitivity.** The guide states that the `nvidia` target is fp32
  and misses the suite's `1e-8 .. 1e-10` tolerances — *precision, not
  correctness* — and that an fp64 target should be used. `state_from` casts
  input data to `cudaq.complex()` so that fp32 simulators do not reject
  complex128 initial-state data.
- **Unsupported inputs** (rejected with an exception): zero-ancilla encodings;
  negative or non-integral `power`, `order`, `count`; `control_state` outside
  `{0, 1}`; both or neither of `ket` and `state_prep`; assigning to
  `walk.encoding`.
- **Known implementation limitations.**
  - **Odd moments require an encoding hook.** An encoding whose
    `select_observable` raises `NotImplementedError` supports even orders and
    every walk circuit, and nothing else. `DoubleFactorizedEncoding` is exactly
    this case — its SELECT terms are frame-rotated `Z` words, not
    computational-frame Pauli words — and `test_odd_moments_are_unavailable`
    pins both the direct call and `Walk(encoding).moment(ket, 1)`.
    `TwoTermLCU` in example 6 leaves the same hook (and the controlled
    factories) as an exercise.
  - **`moments` is a loop, not a batch.** `moments(ket, n)` calls `moment` once
    per order, so it builds `n` outer kernels and issues `n` separate
    `cudaq.observe` calls; only the encoding's sub-kernels and the two
    observables are cached. The `ket` is converted to a `cudaq.State` once and
    reused.
  - **`state_prep` width is unverifiable at factory time**, so a mismatch fails
    at launch.
  - **Cached kernels assume an immutable encoding**; the read-only `encoding`
    property blocks the obvious mistake but cannot stop in-place mutation of
    the encoding object.
  - **Upstream CUDA-Q constraints shape the stack this record sits on:** kernel
    `return` is silently ignored (cuda-quantum#4845), so guards are positive
    `if` blocks; empty lists cannot cross the kernel boundary
    (cuda-quantum#4847), which is why `PauliLCU` normalizes to at least one
    ancilla and pads its flattened arrays; adjoint autogeneration fails or
    mis-replays on the PREPARE kernel (cuda-quantum#4898, #4897), so `unprepare`
    is a hand-written inverse pinned by `test_unprepare_inverts_prepare`. The
    walk's correctness depends on that hand-written inverse.
- **Unsupported versus unverified.** Read each label literally.

| Behavior | Label | Basis |
| --- | --- | --- |
| Zero-ancilla encoding | **unsupported** | explicit `ValueError` in `Walk.__init__`, `QSVT.__init__`, and `reflection_observable`; pinned by `test_consumers_reject_zero_ancilla_encodings` |
| Odd moments without `select_observable` | **unsupported** | documented in the guide and the DF example; raises `NotImplementedError` |
| Negative/non-integral `power`, `order`, `count`; bad `control_state`; both-or-neither input mode | **unsupported** | explicit `ValueError`s, each pinned by a test |
| Measuring a moment observable against a **controlled** kernel | **unsupported by construction** | the ancilla offset collides with the control qubit; no source or test composes them |
| `ket` of the wrong dimension | **unverified** | not validated in `qubitization.py`; no test |
| Non-normalized `ket` | **unverified** | not validated; the moment's interpretation as an expectation assumes normalization |
| Non-Hermitian or complex-coefficient operator | **unsupported at the provider** for `PauliLCU` (`ValueError` on complex coefficients); **unverified** for a foreign encoding |
| Shot-based / noisy targets for `moment` | **unverified** | the library passes no `shots_count`; no test runs a shot-based or noisy target |
| `state_prep` width mismatch: error type and message | **unverified** | documented to fail at launch; nothing states how |
| Dirty (non-zero) system register in `state_prep` mode | **unverified** | the seam documents an all-zero register; see `state-preparation.md` |
| A mutable encoding | **unverified** | the protocol documents immutability; the failure mode (stale cached kernels) is described in source comments, not tested |
| Any runtime, depth, or hardware-resource figure | **absent** | no estimator, no benchmark, no measurement exists for this family |

## Resources

**There is no resource estimator for this family.** Unlike `Trotter`
(`TrotterResourceEstimate`) and the state-preparation providers, nothing in
`qubitization.py` estimates resources, and no benchmark for `Walk` exists in
the repository at the cited commit. The quantities below are the only ones the
source fixes; all are **logical composition counts**, `derived` by reading the
emitted sequences.

| Quantity | Metric and unit | Abstraction level | Value | Status | Controlling parameters |
| --- | --- | --- | --- | --- | --- |
| Qubits, uncontrolled A factories | qubits | logical register allocation | `num_system + num_ancilla` | exact | the encoding |
| Qubits, controlled A factories | qubits | logical register allocation | `num_system + 1 + num_ancilla` | exact | the encoding |
| Injected walk-step invocations | count of injected sub-kernel calls | logical composition | `power` (forward or adjoint); `2 * power` for either roundtrip form | exact | `power` |
| PREPARE / UNPREPARE invocations | count | logical composition | one PREPARE always; one UNPREPARE iff `uncompute` (roundtrips always) | exact | `uncompute` |
| Host-side factory mint calls | count of `encoding.<factory>()` calls | host | one per distinct factory per `Walk` instance, independent of `power` | exact | caching in `mint_cached_kernel`; pinned by `test_factory_call_accounting` |
| Walk steps per moment of order `k` | count | logical composition | `k // 2` | exact | `order` |
| `cudaq.observe` calls | count | host/measurement | `1` per `moment`; `count` per `moments` | exact | `count` |

Architecture and execution assumptions: these are counts of *logical
composition*, before any decomposition, transpilation, or routing. They say
nothing about gate counts, T or Toffoli counts, circuit depth, runtime, or
memory, and they must never be compared against quantities at those levels.
The per-step cost is the encoding's, and this record makes no claim about it.

`alpha` is the encoding's subnormalization and is the quantity that rescales
moments back to physical units. The literature's query-complexity scaling in
`alpha` is **not** claimed by any source in this repository and is not asserted
here.

## Validation

- **Independent oracles.**
  1. A dense matrix built independently from the Pauli terms
     (`tests/python/dense_references.py:dense_matrix`, little-endian by
     construction) combined with the Chebyshev recurrence
     `T_{k+1} = 2 x T_k - T_{k-1}` — `exact_chebyshev_moments` in
     `test_qubitization.py`.
  2. An analytic eigen-decomposition: `test_walk_moments_match_chebyshev`
     compares against `sum_j w_j cos(2 k arccos(E_j / alpha))` for a
     hand-diagonalized one-qubit spectrum — independent of the Chebyshev
     recurrence itself.
  3. Explicit 2x2/4x4 matrix products against a countable mock encoding
     (`test_walk_qsvt_orchestration.py`), which isolate *sequencing* from
     physics: the mock is not a block encoding of any Hamiltonian.
  4. A dense double-factorized reference (`test_df_encoding.py`) for even
     moments through a second, independently written provider.
  5. FCI, at application level, in example 3 — not a test, and requires PySCF.
- **Invariants.** `roundtrip_kernel` and `controlled_roundtrip_kernel` are the
  identity; `T_0 = 1`; controlled `|0>` is the identity and controlled `|1>`
  reproduces the uncontrolled state amplitude by amplitude;
  `Walk.kernel(uncompute=True)` matches `PauliLCU.walk_kernel`; `state_prep`
  mode matches the `cudaq.State` mode; a foreign encoding matches the
  `PauliLCU` it wraps; every `state_prep`-mode kernel is zero-argument and
  therefore directly sampleable.
- **Representative cases.** One qubit, three terms with an asymmetric spectrum
  (`{"I": 0.2, "X": 0.5, "Z": 0.3}`); two qubits, four terms including a
  negative coefficient (`{"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}`);
  a single negative-coefficient term (`{"XZ": -0.5}`), the regression that pins
  the sign for one-ancilla encodings; an identity-only Hamiltonian; a random
  seeded ket and a Hartree-Fock-style basis state; powers `1, 2, 3` and orders
  `0 .. 5`; four qubits with a truncated double factorization.
- **Predeclared tolerances as committed in the repository.** `atol=1e-12` for
  circuit-versus-circuit and prep-versus-data comparisons; `atol=1e-10` /
  `abs=1e-10` for moments against dense or analytic references; `1e-10`
  amplitude thresholds for the controlled-layout assertions.
- **Expected failure / adversarial cases.** A zero-ancilla mock must be
  rejected by all three consumers; a `DoubleFactorizedEncoding` must raise
  `NotImplementedError` on odd moments; every invalid argument above must raise
  `ValueError`; reassigning `encoding` must raise `AttributeError`; two kernels
  minted from one factory with different `state_prep` kernels must not
  cross-contaminate (`test_two_preps_from_one_factory_do_not_cross_contaminate`).
- **Reference results.** The moment values are not tabulated in the tests; each
  is compared against a reference computed in the same run. No numeric result
  is reproduced in this record.
- **Evidence status per claim.**

| Claim | Status |
| --- | --- |
| Kernel sequencing, signatures, register order, validation, caching, and the observable constructions | `derived` from source at the cited commit |
| The `(R0 U_A)^p` frame reconciliation | `derived` by composing `pauli_lcu` and `qubitization` kernel bodies; provider-specific |
| The even-parity moment identity | `derived` from the stated walk block plus Chebyshev identities |
| The odd-parity sign compensation inside `select_observable` | source-stated contract, `derived` from the docstrings and guide; pinned by tests, not re-derived here |
| Moments match dense/analytic references at the tolerances above | `derived` — committed assertions read, **not executed in this session** |
| Any runtime, gate count, depth, or accuracy-versus-cost claim | none made; would be `unverified` |

**Nothing in this family was executed while writing this record.** Every
tolerance above is a repository assertion, not a measurement.

## Evaluation coverage

Declared coverage: `qubitization-walk-moment-boundary` in `evals/evals.json`
tests positive selection between the walk-kernel and moment contracts, required
capability checks, odd-moment support, and the sign misconception. The case has
not been run with or without the skill, so no uplift is claimed.

The following are additional coverage opportunities, not existing case IDs:

| Coverage opportunity | What it would test | Sections that answer it |
| --- | --- | --- |
| `qubitization-walk-versus-moment-selection` (positive selection) | asking for "the walk operator applied `p` times" must yield a contract-A kernel, and asking for `<T_k>` must yield contract B with its observable and float output — not one collapsed into the other | Classification; Scientific contract A and B; Outputs |
| `qubitization-sign-convention` (convention/misconception) | the walk block is `-H/alpha` while `Walk.moment` returns `+<T_k(H/alpha)>`, and the reflection *gate* and *observable* have opposite signs; no caller-side negation | Scientific contract; Outputs → Normalization, sign, and phase |
| `qubitization-encoding-composition` (capability composition) | which protocol members `Walk` needs, that conformance is structural, that `num_ancilla >= 1`, and that an encoding without `select_observable` still supports every circuit and all even moments | Capabilities and composition; Composite protocol |
| `qubitization-odd-moment-boundary` (invalid/unsupported boundary) | odd moments on a double-factorized encoding raise `NotImplementedError`; the correct answer is the limitation, not a hand-rolled substitute observable | Accuracy and limitations |
| `qubitization-qel-api-refusal` (fabrication refusal) | there is no QEL/Lanczos/Krylov public API; the Krylov solve is example code; no runtime or resource figure exists for this family | Scientific contract → "The classical layer is not in this library"; Resources |
| `qubitization-input-mode-contract` (exact input/output handling) | `ket` versus `state_prep` are mutually exclusive; `state_prep` mode yields zero-argument kernels; widths are unchecked at factory time | Inputs → the two input modes; Capabilities → state preparation |

## External alignment

- **Literature conventions.** This library's walk block-encodes `-H/alpha`,
  with eigenphases `pi -/+ arccos(lambda/alpha)`. Much of the literature
  (for example Low–Chuang) uses the opposite reflection sign or the opposite
  operator order and quotes `+H/alpha` with eigenphases
  `+/- arccos(lambda/alpha)`. **Translate before comparing**; every consumer in
  this repository relies on the `-H/alpha` form
  (`docs/sphinx/conventions.rst`).
- **The QEL even/odd moment convention** is named in the `qubitization.py`
  module docstring; example 3 cites Kirby, Motta and Mezzacapo, *Quantum* **7**,
  1018 (2023), arXiv:2208.00567. Only the measurement convention is
  implemented — see the refusal note under the Scientific contract.
- **The walk-QPE exponent schedule** in example 8 follows Berry et al.,
  *PRX Quantum* **6**, 020327 (2025). Example-level, not a library API.
- **The double-factorized example encoding** follows von Burg et al.,
  *PRX Quantum* **2**, 030305 (2021), arXiv:2007.14460. It is a runnable
  example under `docs/sphinx/examples/python/`, dense-validated in CI, and not
  a packaged symbol.
- **External package translations.** None. No comparison against an external
  qubitization implementation exists in this repository, so any cross-package
  equivalence claim is **unverified**.
- **Known semantic differences.** Beyond the walk sign above: `alpha` here is
  the LCU one-norm of the *retained* terms, after `coefficient_threshold`
  filtering and subject to `include_identity`, so it can differ from the
  one-norm of the Hamiltonian a paper quotes; and moments are of `H/alpha`,
  dimensionless, whereas published Lanczos/Krylov tables usually report
  energies in physical units.
