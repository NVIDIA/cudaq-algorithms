# QSP / QSVT — polynomial transformation of an encoded spectrum

Status: draft. Operation + object: **transform** an **encoded spectrum by a
polynomial**, plus one **reconstruct** contract on good-subspace statevectors.

This is one family file covering **three providers** of that family. Following
`assets/primitive-record-template.md`, each canonical Primitive-Record heading
is instantiated **once at family level**, with the per-provider material in
visibly separate subsections beneath it:

| Provider | What it is | Where it runs |
| --- | --- | --- |
| **A. QSVT sequence application** — `QSVT.kernel`, `QSVT.controlled_kernel` | kernel factories generic over the `BlockEncoding` protocol | host factory emitting device code |
| **B. LCU-specialized sequence kernels** — `pauli_lcu.apply_phase_sequence`, `pauli_lcu.apply_controlled_phase_sequence` | the same projector-phase sequence as composable device kernels carrying LCU data explicitly | device |
| **C. Real-time-evolution reconstruction** — `recover_real_time_evolution` | classical recombination of a cosine and a sine run | host (NumPy) |

The **Representation Record for `PhaseSequence`** follows the primitive
headings, as the template orders it. `conventions.md` owns the cross-cutting
qubit-ordering, kernel-boundary, host-validation, resource-abstraction, and
QSP/QSVT phase-translation conventions; this record owns the complete
family-specific contract.

**Simulation-only helpers are not part of any provider above.**
`sim_utils.transform` and `sim_utils.good_subspace` read a simulated
statevector through `cudaq.get_state` and are treated as a separate boundary
throughout this file — see "Host/device/simulation boundary" under Capabilities
and composition. They are the way most of the cited evidence observes provider
A, which is exactly why they must not be described as part of its circuit.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths:
  - from the package root (`cudaq_algorithms/__init__.py:39-40`): `QSVT`,
    `PhaseSequence`, `FORWARD`, `ADJOINT`, `recover_real_time_evolution`;
  - module namespace only, deliberately not re-exported because the names are
    too generic (`__init__.py:44-47`): `apply_phase_sequence` and
    `apply_controlled_phase_sequence` in `cudaq_algorithms.pauli_lcu`, and the
    geometry-only `signal_phase`, `reflect_about_zero`,
    `controlled_signal_phase`, `controlled_reflect_about_zero` in
    `cudaq_algorithms.common_kernels`;
  - simulation-only companions, separate surface:
    `cudaq_algorithms.sim_utils.{transform, good_subspace}`.
- Source paths: `python/cudaq_algorithms/qsvt.py` (the family's own module);
  `python/cudaq_algorithms/common_kernels.py` (phase and reflection kernels,
  `_validate_control_state`); `python/cudaq_algorithms/pauli_lcu.py:280-337`
  (provider B); `python/cudaq_algorithms/block_encoding.py` (the consumed
  protocol and `mint_cached_kernel`); `python/cudaq_algorithms/sim_utils.py`
  (the simulation-only observation path).
- Authoritative tests: `tests/python/test_qsvt.py` (the family suite and the
  `reference_response` oracle); `tests/python/test_walk_qsvt_orchestration.py`
  (sequencing and factory-call accounting against a mock encoding);
  `tests/python/test_block_encoding_protocol.py` (encoding genericity);
  `tests/python/test_state_prep_injection.py` (the `state_prep` mode);
  `tests/python/test_pauli_lcu.py:137-176` (provider B against provider A);
  `tests/python/test_df_encoding.py:235-253` (a second, example-only encoding
  against the shared oracle); `tests/python/conftest.py` (fp64 target pinning);
  `tests/python/dense_references.py` (shared dense oracle helpers).
- Authoritative documentation: `docs/sphinx/guide/qubitization_qsvt.rst`;
  `docs/sphinx/conventions.rst` ("QSVT", "Qubitization walk", "Block
  encodings", "Reflection gate vs reflection observable", "Simulation targets
  and tolerances"); `docs/sphinx/guide/block_encodings.rst`;
  `docs/sphinx/api/python_api.rst`. Examples, all executed in CI by
  `tests/python/test_examples.py`:
  `docs/sphinx/examples/python/02_hamiltonian_simulation.py`,
  `07_matrix_inversion_qsvt.py`, `hamiltonian_simulation_qsvt.py`,
  `df_compression_to_qsvt.py`, `04_double_factorization_and_the_protocol.py`.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q
  pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`). No
  test was executed for this record.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft, for all three providers and the Representation Record.
- Replacement and migration notes: none. No symbol here is marked deprecated at
  the cited commit.
- **Evidence rule for this family.** Every test named below is *cited
  repository evidence* — a committed assertion read at the cited commit,
  labeled `derived` — never a fresh measurement in this session. No tolerance,
  error figure, or example output quoted here was produced by running anything.

## Classification

- Operation + mathematical object (primary identity):
  - A and B: **transform** + **encoded spectrum by a polynomial** (the object
    the phases define is a degree-`d` polynomial; the object transformed is the
    block-encoded operator's spectrum).
  - C: **reconstruct** + **good-subspace statevector pair**.
- Kind:
  - A: quantum operation (kernel factory) with host-side phase-sequence
    validation and marshaling.
  - B: quantum operation (device kernel), composable inside a caller's own
    `@cudaq.kernel`.
  - C: classical transformation (host post-processing).
  - No provider here is a measurement/readout protocol, a resource estimator,
    or a simulation-only analysis. The simulation-only analyses that observe
    this family live in `sim_utils` and are not providers.
- Routine role:
  - A: **driver** — it solves the complete "apply this polynomial to this
    encoded operator" problem by sequencing lower-level routines (the
    encoding's `apply_kernel` and the geometry-only phase/reflection kernels).
    Role follows problem completeness, not execution location
    (`architecture.md`).
  - B: computational — one distinct, independently usable operation at a
    different composition boundary.
  - C: computational — a complete, independently usable classical step.
- Abstraction level: A is a **composite protocol** (see the Composite protocol
  heading). B and C are leaf operations.
- Parameterization: construction-time for A and C's inputs in the sense that
  the phase list, the direction list, and the register widths are captured
  inside the emitted kernel at factory time (`conventions.md`, "Kernel boundary:
  data erasure"); B is runtime-parameterized — the phases, directions, and the
  LCU arrays are kernel *arguments*, which is precisely the difference between
  the two.
- Execution layers: host validation and marshaling (`_as_sequence`,
  `_direction_code`, `_validate_control_state`); kernel factory (`QSVT`
  methods); device kernel (B and everything A emits); host classical
  post-processing (C). No layer of this family calls `cudaq.get_state`
  (`qsvt.py` contains no such call; `sim_utils.py:3-11` states the library
  classes never execute it).
- Input representations: `PhaseSequence` or a plain iterable of phases (see the
  Representation Record); a `BlockEncoding`-conforming object; optionally a
  one-argument `(qubits: cudaq.qview)` preparation kernel; a `cudaq.State` in
  the non-injected mode. For C: two good-subspace statevectors plus the two raw
  phase lists that produced them.
- Output representations: a compiled CUDA-Q kernel — `(state: cudaq.State)`
  without `state_prep`, zero-argument with it (A); an in-place device operation
  on caller-allocated registers (B); a `numpy.ndarray` of `complex128` (C).
- Domain: domain-independent. Nothing in `qsvt.py` is chemistry-specific; the
  chemistry connection arrives only through whichever encoding is injected.
- Required dependencies: `cudaq` for A and B; NumPy for C (imported lazily
  inside `recover_real_time_evolution`) and for the `ArrayLike`/`NDArray`
  annotations, which are `TYPE_CHECKING`-only in `qsvt.py:35-38`. A simulator or
  hardware target is needed to *run* an emitted kernel, not to build one.
- Optional dependencies: **QSPPACK** and SciPy are used to *generate* phase
  angles in tests and examples only. They are explicitly not runtime
  dependencies of `cudaq_algorithms`
  (`docs/sphinx/guide/qubitization_qsvt.rst`, "Phase-angle **generation** is
  external to the library"); the test guards them with
  `pytest.importorskip` (`test_qsvt.py:176-178`) and the examples exit with a
  one-line install hint.

Provisional metadata — record the value, do not route on it:

- Exactness: the emitted circuit is **exact** for the phase sequence it is
  given. Every approximation lives in the caller's choice of polynomial, which
  this library neither computes nor bounds.
- Uncertainty: deterministic as a circuit. Postselection onto the good subspace
  (however performed) is probabilistic, and no library helper reports a success
  probability.
- Method: direct.

## Scientific contract

### A. QSVT sequence application

- Purpose: emit a circuit that applies a polynomial in the block-encoded
  operator to a system register, flagged by the all-zero state of the signal
  register.
- Mathematical definition. Let `U_A` be a zero-flagged block encoding with
  `<0|_anc U_A |0>_anc = H / alpha` (`block_encoding.py:52-57`;
  `docs/sphinx/conventions.rst`, "Block encodings"). A degree-`d` sequence has
  `d + 1` phases and `d` steps. The circuit is, in emission order
  (`qsvt.py:213-221`):

  1. `signal_phase(signal, phases[0])`;
  2. for `i = 1 .. d`: a **forward** step is `U_A` then
     `reflect_about_zero(signal)`; an **adjoint** step is
     `reflect_about_zero(signal)` then `U_A`; then
     `signal_phase(signal, phases[i])`.

  `signal_phase(register, phi)` is `exp(i * phi * |0...0><0...0|)`, i.e. the
  projector phase `diag(e^{i phi}, 1)` on the signal subspace
  (`common_kernels.py:94-106`); `reflect_about_zero` is `I - 2|0...0><0...0|`
  and is documented as exactly the `phi = pi` case of the same kernel
  (`common_kernels.py:109-116`).

  On an eigenstate of `H` with eigenvalue `lambda`, the good-subspace block
  acts as `p(lambda / alpha)` where `p` is the polynomial the phase sequence
  implements, evaluated at the **plain** scaled eigenvalue — the walk's
  `-H/alpha` sign is folded into the step and **no caller-side negation is
  correct** (`qsvt.py:15-19`; `docs/sphinx/guide/qubitization_qsvt.rst`, "Sign
  convention"). The exact 2x2 signal model is `reference_response` in
  `tests/python/test_qsvt.py:18-53`, which `docs/sphinx/conventions.rst`
  ("QSVT") names the *executable specification*.
- Why and when to use: when a block encoding of the operator already exists and
  the caller holds (or can generate) phase angles for the target polynomial —
  spectral transforms such as time evolution, matrix inversion, filtering, or
  amplitude shaping on the encoded spectrum.
- When not to use:
  - when the target is `exp(-iHt)` and a product formula is acceptable — that
    is `Trotter`, a mathematically unrelated route with different
    approximation controls, no ancillas, and no block encoding
    (`docs/sphinx/guide/trotter.rst` and `guide/qubitization_qsvt.rst` each
    describe the other as "the other route"); see [trotter.md](trotter.md);
  - when the goal is Chebyshev moments or walk powers rather than an arbitrary
    polynomial — that is `Walk`
    (`python/cudaq_algorithms/qubitization.py`), which uses **different**
    kernels (see the walk-step warning under "Capabilities and composition");
  - when the caller has no way to obtain phase angles: this library generates
    none.
- Approximation controls: the polynomial degree `d = len(phases) - 1` and the
  externally supplied angles. **There is no library-side approximation
  parameter, error bound, degree heuristic, or convergence criterion.** The
  primitives consume whatever sequence they are handed and never check that it
  corresponds to any polynomial, or that `|p| <= 1` on `[-1, 1]` as QSP
  requires (that requirement is stated in the caller-side prose of
  `docs/sphinx/examples/python/07_matrix_inversion_qsvt.py:31-35`, not enforced
  anywhere in the library).

### B. LCU-specialized sequence device kernels

- Purpose: run the same projector-phase sequence from inside a caller's own
  `@cudaq.kernel`, with the LCU data passed explicitly as arguments rather than
  captured by a factory.
- Mathematical definition: identical to A, restricted to the `PauliLCU`
  encoding — the step's `U_A` is the module-level `pauli_lcu.apply` (PREPARE,
  SELECT, PREPARE-dagger) instead of an injected `apply_kernel`
  (`pauli_lcu.py:280-307`). The controlled form wraps an *uncontrolled*
  PREPARE/UNPREPARE pair around a *controlled* SELECT so each step collapses to
  the identity at control `|0>` (`pauli_lcu.py:310-337`).
- Why and when to use: when the sequence must be embedded in a larger
  hand-written kernel, or when a genuine external control qubit is needed —
  unlike A, the controlled device kernel does **not** initialize its control
  (compare `qsvt.py:271-272` with the caller-applied `x(...)` in
  `test_pauli_lcu.py:165`).
- When not to use: with any encoding other than `PauliLCU`; these kernels take
  LCU-flattened arrays, not a protocol object.
- Approximation controls: as in A.

### C. Real-time-evolution reconstruction

- Purpose: combine the cosine and sine components of a QSP time-evolution run
  into `exp(-i H t)|psi>`.
- Mathematical definition (`qsvt.py:304-323`): given good-subspace vectors
  `cos_state` and `sin_state` produced by running two `qsp`-convention
  sequences through the QSVT circuit, and their **raw** phase lists, the
  function removes each sequence's global phase `exp(i * sum(phases))` and
  returns `2 * (cos_state.real + 1j * sin_state.imag)` after that removal.
- Why and when to use: `exp(-iHt)` is complex and is therefore not a single
  QSP polynomial; the route needs two circuit runs recombined classically
  (`docs/sphinx/guide/qubitization_qsvt.rst`, "Hamiltonian simulation").
- When not to use: **outside the documented validity restriction** — the
  docstring states the reconstruction is valid *for real Hamiltonians and real
  input states*, where the cosine and sine parts live in the real and imaginary
  components. Nothing in source characterizes any other case, and no test
  probes one.
- Approximation controls: none of its own; it inherits the polynomial degree of
  the two sequences it recombines.

## Inputs

### A. QSVT sequence application

```python
from cudaq_algorithms import PhaseSequence, QSVT

transformer = QSVT(encoding)                       # any BlockEncoding
kernel = transformer.kernel(sequence)              # (state: cudaq.State)
kernel = transformer.kernel(sequence, state_prep=prep)   # zero-argument
controlled = transformer.controlled_kernel(sequence, control_state=1)
```

- Arguments:
  - `QSVT(encoding)` (`qsvt.py:161-169`). `encoding` is any object satisfying
    the structural `BlockEncoding` protocol; conformance is `typing.Protocol`
    and `runtime_checkable`, with no inheritance
    (`block_encoding.py:52-58`).
  - `kernel(sequence, convention=None, state_prep=None)` (`qsvt.py:184-239`).
  - `controlled_kernel(sequence, convention=None, control_state=1,
    state_prep=None)` (`qsvt.py:241-301`).
- Shapes/ranks: `sequence` is a `PhaseSequence` or any iterable of floats;
  `d + 1` phases give `d` steps. `control_state` is a scalar `int`.
- Dtypes/domains: phases are coerced with `float(...)` and must be finite;
  `control_state` must be exactly `0` or `1`; `state_prep` is annotated
  `Kernel | None` where `Kernel = Any` (`block_encoding.py:37`), so the type
  system enforces nothing about it.
- Units: phases are **radians**. The polynomial argument `x = lambda / alpha`
  is dimensionless.
- Ordering/layout: the sequence is applied left-to-right in list order, phase
  `i` after step `i`; qubit 0 is least significant and the system register is
  allocated first (see "Outputs" for the full geometry).
- Normalization: **`QSVT` never reads `encoding.alpha`.** The subnormalization
  enters only through the caller's phase generation — for time evolution the
  examples and the test set `tau = encoding.alpha * time` before calling
  QSPPACK (`test_qsvt.py:207`;
  `docs/sphinx/examples/python/hamiltonian_simulation_qsvt.py:95`). A record
  reader must not assume the object rescales anything for them.
- Required mathematical properties: `encoding.num_ancilla >= 1`, enforced;
  the injected encoding must be a genuine zero-flagged block encoding, which is
  **not** checkable (a conforming mock that encodes no Hamiltonian compiles and
  runs — that is exactly what `MockBlockEncoding` in
  `test_walk_qsvt_orchestration.py` is); the phases should implement a
  polynomial bounded by 1 on `[-1, 1]`, which is not checked.
- Validation and rejection behavior, all host-side at factory or construction
  time, all `ValueError` (device kernels have no error channel —
  `conventions.md`, "Validation happens on the host"):

| # | Rejected condition | Raised by | Message fragment |
| --- | --- | --- | --- |
| 1 | `encoding.num_ancilla == 0` | `QSVT.__init__` (`qsvt.py:162-166`) | "QSVT requires an encoding with num_ancilla >= 1" |
| 2 | empty `phases` | `PhaseSequence` (`qsvt.py:94-95`) | "phases must contain at least one value" |
| 3 | a non-finite phase | `PhaseSequence` (`qsvt.py:96-97`) | "phases must be finite" |
| 4 | `convention` not `"qsvt"`/`"qsp"` | `PhaseSequence` (`qsvt.py:99-101`) | "convention must be 'qsvt' or 'qsp'" |
| 5 | unrecognized walk direction | `_direction_code` (`qsvt.py:54-60`) | "walk direction must be 'forward', 'adjoint', 0, or 1" |
| 6 | `len(walk_directions) != degree` | `PhaseSequence` (`qsvt.py:108-111`) | "walk_directions must contain len(phases) - 1 entries" |
| 7 | re-tagging a tagged sequence with a different convention | `_as_sequence` (`qsvt.py:135-144`) | "would reinterpret its phases" |
| 8 | `control_state` not in `{0, 1}` | `_validate_control_state` (`common_kernels.py:73-77`) | "control_state must be 0 or 1" |

  Conditions 2-6 are asserted at `test_qsvt.py:60-82`, condition 7 at
  `test_qsvt.py:165-172`, and condition 1 at
  `test_walk_qsvt_orchestration.py:266-274` (which also pins that `Walk` and
  `reflection_observable` reject the same degenerate encoding).

  **Not validated, by contrast:** the polynomial's boundedness; whether the
  sequence corresponds to any function at all; the injected encoding's physical
  correctness; and the width of `state_prep` (next paragraph). A wrong sequence
  produces a perfectly valid circuit with a meaningless response.

- `state_prep` composition (the injected-preparation mode):
  - signature `(qubits: cudaq.qview)` — the one-register seam whose contract is
    owned by [state-preparation.md](state-preparation.md);
  - given it, both factories return **zero-argument** kernels that allocate the
    system register in `|0...0>`, run the preparation on it, and only then
    allocate the signal (or `[control, signal]`) register
    (`qsvt.py:208-223`, `266-283`);
  - the preparation width must equal `encoding.num_system` exactly. This is
    documented as **not verifiable at factory time**, so a mismatch fails at
    launch (`docs/sphinx/guide/qubitization_qsvt.rst`, "Contract"); the error
    type and message are `unverified`;
  - in `controlled_kernel` the preparation runs **once, uncontrolled**, before
    the control register exists (`qsvt.py:268-270`) — a sequencing choice, not
    evidence of a controlled-preparation capability;
  - `test_state_prep_injection.py:85-101` pins the prep-mode circuits against
    their `cudaq.State`-taking twins fed the identical prepared state (atol
    `1e-12`), and `:132-157` pins that every prep-mode kernel is directly
    sampleable with no arguments.

### B. LCU-specialized sequence device kernels

- Arguments (`pauli_lcu.py:280-286`, `310-315`):
  `apply_phase_sequence(signal, system, phases, walk_directions, angles,
  term_controls, term_ops, term_lengths, term_signs)` and
  `apply_controlled_phase_sequence(control_and_signal, system, phases,
  walk_directions, angles, ...)`.
- Shapes/ranks: `phases` is `list[float]` of length `d + 1`;
  `walk_directions` is `list[int]` of length `d` and **must be non-empty** —
  captured empty lists cannot cross the kernel boundary, so a degree-0 sequence
  passes one unused entry, exactly as the factories do
  (`pauli_lcu.py:292-294`; the same workaround appears at `qsvt.py:199-201`).
- Ordering/layout: pass `sequence.projector_phases` and
  `list(sequence.walk_directions)`, never the raw `phases` of a `qsp`-tagged
  sequence; the five LCU arrays come from `PauliLCU.kernel_args`, which returns
  defensive copies (`pauli_lcu.py:571-580`).
- Required mathematical properties: **the signal register must start in
  `|0...0>`** (`pauli_lcu.py:288-289`). The caller owns that register here, so
  this precondition is the caller's to satisfy.
- Validation and rejection behavior: **none.** These are device kernels; an
  invalid argument is a silent no-op or worse, never an exception
  (`conventions.md`). All the host-side guards of provider A are bypassed.

### C. Real-time-evolution reconstruction

- Arguments: `recover_real_time_evolution(cos_state, sin_state, cos_phases,
  sin_phases)` (`qsvt.py:304-307`).
- Shapes/ranks: `cos_state` and `sin_state` are 1-D good-subspace statevectors
  of dimension `2**num_system`; `cos_phases` and `sin_phases` are the **raw**
  phase sequences that produced them.
- Dtypes/domains: both states are coerced with `np.asarray(..., dtype=
  np.complex128)`; the phase lists are summed with `np.sum`.
- Ordering/layout: `cos_*` first, `sin_*` second; swapping them silently
  returns a different vector.
- Normalization: the inputs are the *postselected, unrenormalized* good-subspace
  vectors; the factor `2` and the `.real`/`.imag` projections are part of the
  contract, not a normalization convenience.
- Required mathematical properties: a **real** Hamiltonian and a **real** input
  state, per the docstring.
- Validation and rejection behavior: **none.** There is no shape check, no
  dtype rejection, no length agreement check between a state and its phase
  list, and no check that the sequences were `qsp`-tagged. Mismatched or
  mis-ordered inputs produce a plausible wrong answer rather than an error.
  Contrast `sim_utils.evolve`, which does validate its dimensions
  (`sim_utils.py:96-102`) — the asymmetry is real and worth stating to a
  caller.

## Outputs

### A. QSVT sequence application

- Return type or emitted kernel signature:
  - without `state_prep`: a compiled kernel taking one `cudaq.State`;
  - with `state_prep`: a compiled kernel taking **no arguments**.
  Both factories return a fresh kernel per call; the *encoding's* data-free
  kernels are minted once per `QSVT` instance and reused
  (`mint_cached_kernel`, `block_encoding.py:40-48`).
- Mathematical meaning: the emitted circuit is unitary on the full register.
  Only its **good-subspace block** — the block flagged by the signal register
  in `|0...0>` — realizes the polynomial. The complementary blocks are not a
  specified part of the contract.
- Shape/register geometry, in allocation order (`qsvt.py:206-239`, `264-301`):

| Mode | Registers, in order | Total qubits | Good subspace |
| --- | --- | --- | --- |
| plain | system (`num_system`), signal (`num_ancilla`) | `ns + na` | leading `2**ns` amplitudes |
| controlled | system, then one combined `[control, signal]` of width `1 + na` | `ns + 1 + na` | amplitudes with control **and** signal bits zero; the test's index map is `sys + (ctrl << ns) + (anc << (ns + 1))` (`test_qsvt.py:236-237`) |

  The controlled form uses **one combined register** because a CUDA-Q Python
  control set cannot mix a bare qubit with a separate register
  (`qsvt.py:251-253`); the external control is qubit 0 of that register, which
  is global qubit index `ns`.

  **Consequence for postselection:** `sim_utils.good_subspace` requires a
  statevector of dimension exactly `2**(num_system + num_ancilla)`
  (`sim_utils.py:46-49`), so it applies to the plain kernel and **raises
  `ValueError` on a controlled-kernel state**. The controlled test slices the
  state by hand instead (`test_qsvt.py:239-258`).
- Normalization, sign, and phase:
  - the good-subspace vector is generally **subnormalized**: the block is
    non-unitary whenever `|p| < 1`, and neither the circuit nor
    `good_subspace` renormalizes (`sim_utils.py:50` copies a slice);
  - the response is evaluated at the **plain** `lambda / alpha` with no
    caller-side negation, guarded adversarially by
    `test_qsvt.py:102-104`, which asserts the negated-eigenvalue prediction
    must *not* match at odd degree;
  - a `qsvt`-convention sequence matches `reference_response` directly; a
    `qsp`-convention sequence runs **doubled** projector phases and differs
    from that model by the global phase `exp(i * sum(phases))`
    (`qsvt.py:118-126`, `test_qsvt.py:28-31`);
  - a degree-0 sequence is a pure signal phase: the good subspace comes back as
    `exp(i * phi_0)` times the input (`test_qsvt.py:146-151`);
  - in the controlled kernel the control qubit is **initialized inside the
    circuit** to `control_state` and is never uncomputed. This is a
    self-contained circuit with a fixed control value, not a controlled-`U`
    oracle a caller can drive with an external superposition; for that, compose
    provider B, which leaves the control alone.
- Observable or measurement interpretation: none. This family emits no
  observable and performs no measurement. `Walk.moment` / `Walk.moments` are
  the library's measurement path, and they belong to the qubitization family,
  not this one.
- Error/status information: none at the device boundary. No return value, no
  status channel, no success flag — which is why every guard is host-side.

### B. LCU-specialized sequence device kernels

- Return type: none; they act in place on the registers the caller allocated.
- Mathematical meaning, geometry, and phase behavior: identical to A for the
  same phases and directions, pinned at atol `1e-12` against the A factories
  for both the plain and controlled forms (`test_pauli_lcu.py:137-176`).
- Error/status information: none, and no host-side guard either.

### C. Real-time-evolution reconstruction

- Return type: `numpy.ndarray` of `complex128`, the same shape as the inputs.
- Mathematical meaning: an approximation of `exp(-i H t)|psi>` on the system
  register, subject to the polynomial degree and the real-Hamiltonian
  restriction.
- Normalization, sign, and phase: each sequence's global phase
  `exp(i * sum(phases))` is removed here; the factor `2` recombines the two
  half-amplitude components. The result is not renormalized.
- Observable or measurement interpretation: none — this is host arithmetic on
  an already-postselected vector.
- Error/status information: none; see the missing validation noted under
  Inputs.

## Capabilities and composition

### Required — the block encoding (consumed by A)

- Stable ID: `cudaq-algorithms.block-encoding.zero-flagged.v1`.
  [block-encoding.md](block-encoding.md) owns this provisional taxonomy
  capability and its source-level `BlockEncoding` protocol surface.
- Direction: requires.
- Owning family record: [block-encoding.md](block-encoding.md). The capability
  remains provisional and documentation-level; the Python protocol checks
  structural member presence, not scientific correctness.
- Boundary representation and exact members actually used. This matters,
  because the protocol is much wider than the consumer:

| Member | Used by | Not used |
| --- | --- | --- |
| `num_system` | both factories (register width) | |
| `num_ancilla` | both factories; `>= 1` enforced in `__init__` | |
| `apply_kernel()` → `(ancilla: qview, system: qview)` | `kernel` | |
| `controlled_apply_kernel()` → `(control_and_ancilla: qview, system: qview)` | `controlled_kernel` only | |
| `alpha` | — | **never read by `QSVT`** |
| `walk_step_kernel` and the adjoint/controlled walk factories | — | never read by `QSVT` |
| `select_observable` | — | never read by `QSVT` |

  `test_walk_qsvt_orchestration.py:252-263` asserts the exact minted set:
  `QSVT(encoding).kernel(PhaseSequence([...]))` mints `{"apply": 1}` and
  nothing else.
- Semantic invariants required of the provider: `<0|_anc U_A |0>_anc = H/alpha`;
  system register allocated first so the good subspace is the leading
  `2**num_system` amplitudes; the controlled variant takes a combined
  `[control, ancilla...]` register and reduces to the identity at control `|0>`
  (`block_encoding.py:19-30`; `docs/sphinx/conventions.rst`).
- **Two consequences a composer must not blur:**
  1. **The QSVT step is not the walk step.** QSVT builds its step from
     `apply_kernel` composed with `reflect_about_zero` — forward is `U_A` then
     reflect, adjoint is reflect then `U_A`. The qubitization `Walk` step is
     SELECT followed by a reflection about the *PREPARE* state, which is why
     `Walk` sandwiches its powers in an outer PREPARE/UNPREPARE pair
     (`qubitization.py:10-16`; `pauli_lcu.py:210-217`). Both flag `-H/alpha`
     and the two are conjugate by PREPARE, but they are different kernels and
     `walk_step_kernel` is not the operator QSVT applies.
  2. **Protocol conformance does not imply full downstream availability.** The
     example-only `DoubleFactorizedEncoding` satisfies the protocol except
     `select_observable`, which raises `NotImplementedError`; odd Chebyshev
     moments are therefore unavailable on it while **the whole QSVT surface is
     unaffected** (`docs/sphinx/examples/python/df_encoding.py:47-52`;
     `test_df_encoding.py:226-253`). Conformance also promises nothing about
     `state_prep`: no protocol member mentions it
     ([state-preparation.md](state-preparation.md)).
- Host/device/simulation boundary: the provider must supply data-free kernels
  (registers-only signatures, data captured at factory time) —
  `conventions.md`, "Kernel boundary: data erasure".
- Unsupported conditions: `num_ancilla == 0` is rejected outright.

### Required, optional — unitary state preparation (consumed by A)

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`.
- Direction: requires, **optionally** — omitting `state_prep` selects the
  `cudaq.State` input mode instead.
- Owning family record: [state-preparation.md](state-preparation.md), whose
  Capability Record carries the invariants, the four-module consumer table
  (which lists `QSVT.kernel` and `QSVT.controlled_kernel`), and the explicit
  scope limit on register ownership. Do not restate that ownership behavior as
  a settled library-wide policy; at the cited commit it is a `derived`
  description of the current seam only.
- Boundary representation: a kernel whose only parameter is
  `(qubits: cudaq.qview)`.
- Convention requirements and unsupported conditions: as in that record —
  including that controlled and adjoint preparation are `unverified`, and that
  a width mismatch is undetectable at factory time.

### Provided

- Stable ID: **none declared.** The architecture gate for extracting a
  capability record is *multiple independent producers or consumers of a
  reusable boundary*. At the cited commit this family has one producer of the
  polynomial-response boundary (`QSVT`, mirrored by provider B) and no
  in-library consumer of its output — the outputs are consumed by callers,
  examples, and the simulation-only helpers. `Deferred:` a provided-capability
  declaration until a second producer or an in-library consumer exists.
- What a downstream reader can rely on regardless: the response contract, the
  register geometry, and the good-subspace definition stated under Outputs.

### Host/device/simulation boundary (including the simulation-only helpers)

- **Library, hardware-shaped:** everything in `qsvt.py`, `common_kernels.py`,
  and `pauli_lcu.py`. None of these modules calls `cudaq.get_state`;
  `sim_utils.py:3-11` states the library classes never execute it, and
  `docs/sphinx/guide/qubitization_qsvt.rst` repeats the rule.
- **Simulation-only, and not part of any provider here:**
  - `sim_utils.good_subspace(encoding, state)` — validates that the statevector
    dimension is exactly `2**(num_system + num_ancilla)` and returns a copy of
    the leading `2**num_system` amplitudes. It postselects by slicing; it does
    not renormalize, and it does not apply to controlled-kernel states.
  - `sim_utils.transform(transformer, ket, sequence, convention=None)` —
    builds `transformer.kernel(sequence, convention)`, runs `cudaq.get_state`,
    and postselects. Note what it is *not*: it has **no `state_prep`
    parameter**, so it always exercises the `cudaq.State` input mode, and it
    never touches `controlled_kernel`. Most of the cited QSVT evidence observes
    the circuit through this function; that does not make it part of the
    circuit.
  - Annotation caveat: `good_subspace` is annotated `encoding: PauliLCU` and
    `transform` is annotated `transformer: QSVT`, but both bodies use only
    `num_system`, `num_ancilla`, and the kernel. They are structurally usable
    with any conforming encoding, yet `test_df_encoding.py` defines its own
    `encoded_block` helper rather than relying on that. Foreign-encoding use of
    these helpers is `unverified`.
- **Neither, despite living in `sim_utils`' namespace:** `state_from` is
  defined in `common_kernels.py:25-34` and re-exported; it builds a
  `cudaq.State` at the active target's precision and is used by the
  hardware-shaped `Walk.moment` path, so it is not a simulation-only helper.
- **Host classical, in the library proper:** `recover_real_time_evolution`
  (provider C) lives in `qsvt.py` and imports only NumPy. It is *fed* by
  simulation-only postselection today, but it is not itself simulator-bound —
  any source of good-subspace amplitudes, including a hardware postselection
  the caller implements, satisfies its input contract.

## Composite protocol

Provider A's abstraction level is `composite protocol`; B and C are leaf
operations and this heading does not describe them.

- Required lower-level capabilities: the block-encoding members tabulated above
  (`num_system`, `num_ancilla >= 1`, `apply_kernel`, and
  `controlled_apply_kernel` for the controlled form); the geometry-only
  projector-phase and zero-reflection kernels of `common_kernels.py`; and,
  optionally, unitary state preparation.
- Canonical reference composition: the emission order given under "Scientific
  contract", whose executable statement is `reference_response`
  (`test_qsvt.py:18-53`) and whose sequencing is pinned independently of any
  physics by the 4x4 mock-encoding matrix product
  (`test_walk_qsvt_orchestration.py:229-249`).
- Default recipe and its applicability conditions: all-forward steps
  (`walk_directions` defaults to `(FORWARD,) * degree`), projector convention,
  control initialized to `|1>` in the controlled form. Applicable whenever the
  caller's phases were generated in the projector convention for the plain
  scaled eigenvalue.
- Materially different alternatives, all present in source:
  - **mixed directions** — `walk_directions` may interleave adjoint steps;
    the adjoint step is the reverse product of the same two self-adjoint
    factors, and both the dense and mock oracles cover a mixed sequence
    (`test_qsvt.py:107-126`);
  - **provider B** — the same sequence as a composable device kernel with an
    uninitialized external control;
  - **the qubitization `Walk`** — a different construction for Chebyshev
    responses and moments, not a substitution for a general polynomial.
- Propagated conventions: qubit-0-least-significant ordering, system-first
  register allocation, the `[control, signal]` combined-register geometry, and
  the `-H/alpha` walk-block sign all propagate unchanged from the encoding
  layer; the phase convention is resolved *inside* this family by
  `projector_phases`.
- Propagated errors: the encoding's own construction-time validation runs
  before `QSVT` sees it. `QSVT` adds only the eight conditions tabulated under
  Inputs. Nothing propagates from the device layer, which has no error channel.
- Propagated resources and how they compose: the emitted circuit contains `d`
  invocations of the encoding's `apply_kernel` (or its controlled twin), so
  whatever that encoding costs multiplies by the degree. No quantity beyond
  that structural count is available — see Resources.
- Component substitution: supply any object satisfying the `BlockEncoding`
  protocol. The substitution must preserve the zero-flagged block semantics,
  `num_ancilla >= 1`, the system-first geometry, and identity-at-control-`|0>`
  for the controlled variant. `test_block_encoding_protocol.py:108-116` pins
  that a foreign encoding exposing only the protocol surface drives QSVT to
  results identical to the `PauliLCU` it wraps (atol `1e-12`), and
  `test_df_encoding.py:235-253` runs a genuinely different encoding against the
  same signal-model oracle.

## Accuracy and limitations

- Error behavior or bounds: **the library states none.** The circuit is exact
  for the sequence it is given; all approximation error is the distance between
  the caller's polynomial and their target function, which is computed
  externally. There is no degree heuristic, no error-versus-degree bound, and
  no convergence criterion in source. The prose `d ~ alpha * t` in
  `docs/sphinx/examples/python/df_compression_to_qsvt.py:11-12` is example
  commentary, `unverified` as a bound.
- Precision sensitivity: the emitted circuit is precision-agnostic; observed
  agreement depends on the active simulator. `docs/sphinx/conventions.rst`
  ("Simulation targets and tolerances") states the default CUDA-Q target is
  fp32 with ~1e-7 statevector error, which would miss this family's
  1e-8..1e-12 assertions; `tests/python/conftest.py` pins `qpp-cpu` (fp64) and
  honors `CUDAQ_DEFAULT_SIMULATOR`. `state_from` matches the input dtype to
  `cudaq.complex()` because fp32 simulators reject `complex128` initial-state
  data.
- Unsupported inputs: the eight rejected conditions under Inputs.
- Known implementation limitations, each traceable to an upstream CUDA-Q defect
  or an API constraint rather than to arbitrary design:
  - a degree-0 sequence's empty `walk_directions` is padded with one unused
    entry, because captured empty lists cannot cross the kernel boundary
    (`qsvt.py:199-201`; `pauli_lcu.py:292-294`);
  - the controlled variant needs one combined `[control, signal]` register
    because a CUDA-Q Python control set cannot mix a bare qubit with a separate
    register (`qsvt.py:251-253`);
  - guards elsewhere in the stack use positive `if n > 0:` blocks rather than
    early `return`, since a kernel `return` is silently ignored by the compiler
    (`common_kernels.py:14-16`);
  - `QSVT.encoding` is read-only, because kernels are cached against the
    encoding and swapping it would serve stale circuits (`qsvt.py:171-175`);
  - `PhaseSequence` enforces no immutability. Its attributes are ordinary
    instance attributes; the factories read `projector_phases` at call time and
    capture the resulting values, so mutating a sequence afterwards cannot
    change an already-minted kernel. Any other consequence of mutation is
    `unverified`.
- Unsupported versus unverified — read each label literally:

| Behavior | Label | Why the label |
| --- | --- | --- |
| Phase-angle generation | **absent** | no symbol in the library computes angles; the guide states generation is external and QSPPACK is example-only |
| Polynomial boundedness / validity check | **absent** | nothing inspects the sequence beyond finiteness and length |
| Error bound or degree selection | **absent** | no estimator, bound, or heuristic exists in source |
| Success probability or amplitude amplification for the postselected block | **absent** | no library construct reports or boosts it |
| Resource estimate for QSVT | **absent** | the stack's only estimator belongs to `Trotter` |
| `recover_real_time_evolution` on a complex Hamiltonian or complex input state | **unverified** | the docstring restricts it to real cases; no test probes any other case |
| `sim_utils.transform` / `good_subspace` with a foreign (non-`PauliLCU`) encoding | **unverified** | annotations say `PauliLCU`/`QSVT`; the bodies are structural; the DF suite avoids the question with its own helper |
| `state_prep` width mismatch: error type and message | **unverified** | documented to fail at launch; nothing states how |
| Controlled kernel driven by an externally prepared control superposition | **unverified** | the factory initializes its own control to a basis value and never uncomputes it |
| Combining `controlled_kernel` output with `sim_utils.good_subspace` | **unsupported** | the dimension check rejects it; postselect by hand |
| Hardware execution of any circuit in this family | **unverified** | no target beyond simulators appears in tests or examples |

- Do not hand-roll an alternative for any "absent" row and then present it as a
  library capability.

## Resources

**This family has no resource estimator, and its absence is the contract.**
`docs/sphinx/api/python_api.rst` exposes no estimator for `QSVT`, and the only
estimator in the simulation stack belongs to `Trotter`. Never infer from the
absence that the circuit is cheap.

What *is* available is exact structural information about the emitted circuit.
For a degree-`d` sequence on an encoding with `ns` system and `na` signal
qubits:

| Quantity | Value | Metric and unit | Abstraction level | Status | Controlling parameter |
| --- | --- | --- | --- | --- | --- |
| register width, plain kernel | `ns + na` | qubits | logical register geometry | exact for the emitted circuit | the encoding |
| register width, controlled kernel | `ns + 1 + na` | qubits | logical register geometry | exact | the encoding |
| block-encoding invocations | `d` | count of `apply_kernel` (or controlled) calls | logical sub-circuit invocation | exact | `degree` |
| zero-state reflections | `d` | count | logical sub-circuit invocation | exact | `degree` |
| projector-phase blocks | `d + 1` | count | logical sub-circuit invocation | exact | `degree` |
| host kernel mints per `QSVT` instance | 1 per encoding factory used | count | host | exact | — |

Architecture and execution assumptions: these are counts of *emitted logical
sub-circuits before any decomposition or transpilation*. They are **not** gate
counts, T or Toffoli counts, depth, runtime, or memory, and they must never be
compared against a quantity at a different abstraction level
(`conventions.md`, "Resource abstraction levels must not be conflated"). Even
one projector-phase block is not a fixed gate count: `signal_phase` emits
`2 * na` `X` gates around a single `r1` that is multi-controlled on `na - 1`
qubits (`common_kernels.py:94-106`), whose cost is decomposition- and
target-dependent.

Composition rule: the per-step cost is whatever the injected encoding's
`apply_kernel` costs, multiplied by `d`. Since no encoding in this repository
publishes a resource contract, that product cannot be evaluated here.

Adjacent quantities that are cost-relevant but are **not** resource estimates:
`encoding.alpha` (the LCU one-norm and a spectral bound `||H|| <= alpha`, which
governs the degree needed for a given accuracy in the literature) and
`encoding.num_ancilla`. Presenting either as a resource estimate would violate
the rule above.

Efficiency behavior that is a design contract rather than a benchmark:
`mint_cached_kernel` plus the per-instance `_kernel_cache` exist so repeated
factory calls do not recompile identical sub-kernels (`qsvt.py:168-179`;
`block_encoding.py:40-48`). No timing was measured, and none may be claimed.

## Validation

- Independent oracles — five, mathematically independent of the code under
  test, all **cited** repository assertions rather than executed here:
  1. **`reference_response`** (`test_qsvt.py:18-53`) — an explicit 2x2
     signal-model matrix product on the invariant subspace, built from
     `step_forward = [[-x, -s], [s, -x]]`, `step_adjoint = step_forward.T`, and
     the convention-dependent phase matrix. `docs/sphinx/conventions.rst`
     names it the executable specification, and `test_df_encoding.py:237`
     imports the same function for a different encoding.
  2. **Dense eigendecomposition** — the full device block reconstructed as
     `eigenvectors @ diag(response) @ eigenvectors.conj().T` for a
     mixed-direction sequence (`test_qsvt.py:107-126`, atol `1e-9`), using the
     `dense_matrix` helper in `tests/python/dense_references.py`, which builds
     the Pauli sum by bit manipulation and depends on no CUDA-Q code.
  3. **External package plus exact diagonalization** — QSPPACK-generated
     degree-12 Jacobi-Anger cosine/sine phases run through both sequences and
     recombined, required to satisfy
     `norm(evolved - exact) < 1e-8` against eigendecomposition-based
     `exp(-i t H)` (`test_qsvt.py:201-225`).
  4. **Countable mock circuit** — `MockBlockEncoding` isolates sequencing from
     physics; the QSVT interleaving is checked against an explicit 4x4 matrix
     product (`test_walk_qsvt_orchestration.py:229-249`, atol `1e-12`), and the
     factory-call accounting asserts exactly `{"apply": 1}`.
  5. **Genericity oracle** — `ForeignEncoding` exposes only the protocol
     surface and must drive QSVT to the wrapped `PauliLCU`'s results
     (`test_block_encoding_protocol.py:108-116`, atol `1e-12`).
- Invariants asserted at the cited commit:
  - eigenstate response equals `reference_response(seq, lambda / alpha)` times
    the eigenvector (`test_qsvt.py:85-104`, atol `1e-10`);
  - a `qsp` sequence executes exactly as the doubled projector sequence
    (`test_qsvt.py:129-143`, atol `1e-12`) **and** the two conventions
    genuinely differ in the signal model;
  - a degree-0 sequence is `exp(i phi_0)` on the good subspace
    (`test_qsvt.py:146-151`, atol `1e-12`);
  - a single-term encoding with zero phases at degree 1 implements `-H/alpha`
    (`test_qsvt.py:154-162`, atol `1e-10`);
  - the controlled kernel reproduces the uncontrolled state in the control-`|1>`
    block and is the identity at control `|0>`, amplitude by amplitude
    (`test_qsvt.py:228-258`, abs `1e-10`);
  - provider B matches provider A for both plain and controlled forms
    (`test_pauli_lcu.py:137-176`, atol `1e-12`);
  - prep-mode kernels match their `cudaq.State` twins and are zero-argument
    sampleable (`test_state_prep_injection.py:85-101`, `132-157`).
- Representative cases: 1-qubit two-term and 2-qubit four-term Pauli
  Hamiltonians; a single-term negative-coefficient encoding; a 4-qubit
  eight-term Hamiltonian in the example; a 3-qubit padded 5x5 SPD matrix in the
  matrix-inversion example; a 16-dimensional double-factorized encoding.
  Degrees run from 0 through the degree-12 and degree-16 Jacobi-Anger
  sequences of the test and examples; the matrix-inversion example derives its
  own degree from the conditioning rather than fixing one.
- Convention translation performed before each comparison: `projector_phases`
  for the qsp doubling; the global phase `exp(i sum(phases))` removed by
  provider C (and by hand at
  `docs/sphinx/examples/python/07_matrix_inversion_qsvt.py:192`); little-endian
  index maps for every hand-written slice.
- Precision and execution target: `qpp-cpu` fp64, pinned by an autouse fixture
  in `tests/python/conftest.py` and honoring `CUDAQ_DEFAULT_SIMULATOR`.
- Predeclared tolerances: as tabulated above, `1e-12` down to `1e-8`
  depending on the oracle; the examples self-check their own outputs at
  `< 1e-6` (`02_hamiltonian_simulation.py:126`) and exit nonzero above `1e-10`
  (`hamiltonian_simulation_qsvt.py:127-128`).
- Expected failure and adversarial cases: the negated-eigenvalue guard
  (`test_qsvt.py:102-104`, requiring a discrepancy `> 1e-6`); the conflicting
  convention re-tag (`test_qsvt.py:165-172`); `ZeroAncillaMock` rejected by
  `QSVT`, `Walk`, and `reflection_observable`
  (`test_walk_qsvt_orchestration.py:266-274`); the two-preps-one-factory
  contamination check in the injection suite.
- Reference results: none stored; every oracle is constructed inside the cited
  test or example.
- **Validation gaps to declare, not fill:** no test exercises
  `recover_real_time_evolution` on a complex Hamiltonian or a complex input
  state (its restriction is `derived` from the docstring alone); no test passes
  a foreign encoding to the `sim_utils` helpers; no test covers
  `controlled_kernel` with `control_state=0` *and* `state_prep` together; no
  test covers an unbounded (`|p| > 1`) sequence; no hardware target is
  exercised anywhere; and no resource claim exists to test.
- Evidence status per claim: `derived` from the cited source or committed test
  assertion, unless labeled `assumed` or `unverified` in place. **Nothing in
  this record is `measured`.**

## Evaluation coverage

Declared coverage: `qsvt-phase-and-recovery-boundary` in `evals/evals.json`
tests positive sequence application, the QSP/QSVT phase translation, required
block-encoding composition, two-state reconstruction, and the unsupported
complex-data assumption. It is backed by Scientific contract, Inputs,
Capabilities and composition, and Accuracy and limitations. The case has not
been run with or without the skill, so no uplift is claimed.

Two other existing cases use this record as background without owning it:

| Existing case id | How this record contributes |
| --- | --- |
| `implicit-composition-boundary` | the required-member table, the walk-step-versus-QSVT-step warning, and the host/device/simulation boundary give the concrete composability check for "an encoded operator plus a spectral transformation" |
| `state-preparation-injection-composition` | owned by [state-preparation.md](state-preparation.md); this record supplies the QSVT-side detail of the injected mode (zero-argument kernel, signal allocated after the system register, prep uncontrolled in the controlled form) |

Additional coverage opportunities include:

- positive routing: choosing QSVT over Trotter for a stated accuracy and
  ancilla budget, and naming `PhaseSequence` plus a conforming encoding as the
  exact inputs;
- convention misconception: refusing a caller-side negation of the eigenvalue,
  and refusing to re-tag a `qsp` sequence as `qsvt`;
- capability composition: refusing to treat `walk_step_kernel` as the operator
  QSVT applies, and refusing to infer `state_prep` support from protocol
  conformance;
- resource boundary: refusing to state a T-count, depth, or degree bound for
  QSVT, and refusing to present the structural counts above as gate counts;
- unsupported boundary: declining to generate phase angles from inside the
  library, and reporting `recover_real_time_evolution` on a complex Hamiltonian
  as `unverified`.

Declared coverage is not validated coverage. Nothing here has been run with or
without the skill.

## External alignment

- Literature conventions, all `unverified` here — this record neither
  re-derives them nor compares against the papers:
  - **Walk sign.** `docs/sphinx/conventions.rst` warns that much of the
    literature (naming Low-Chuang) uses the opposite reflection sign or
    operator order and quotes `+H/alpha` with eigenphases
    `+/- arccos(lambda/alpha)`, whereas this library's walk block is
    `-H/alpha` with eigenphases `pi -/+ arccos(lambda/alpha)`. Translate before
    comparing; consumers here rely on the `-H/alpha` form.
  - **Reflection gate versus reflection observable.** Same word, opposite sign:
    the gate `common_kernels.reflect_about_zero` is `I - 2|0..0><0..0|`, while
    `qubitization.reflection_observable` is `2|0..0><0..0| - I`
    (`docs/sphinx/conventions.rst`). Only the gate appears in this family, but
    the trap catches readers comparing against the moment machinery.
  - **Childs-Kothari-Somma** closed form `(1 - (1-x^2)^b)/x` for the scaled
    inverse, used in the matrix-inversion example
    (`07_matrix_inversion_qsvt.py:25-30`); named there without a full citation.
  - **Jacobi-Anger** expansion in Bessel functions for the cosine and sine
    components of `exp(-i tau x)`, implemented in test and example helpers, not
    in the library.
  - No QSVT or QSP foundational paper is cited anywhere in the source read for
    this record. Do not attribute one on the library's behalf.
- External package translation — **QSPPACK**:
  - QSPPACK's convention is the `qsp` one, Z-rotation phases
    `diag(e^{i phi}, e^{-i phi})`; `PhaseSequence(phases, convention="qsp")`
    accepts them raw and doubles them into projector phases wherever a circuit
    is built (`qsvt.py:78-82`, `117-126`);
  - the resulting circuit differs from the 2x2 model by
    `exp(i * sum(phases))`, which `recover_real_time_evolution` removes and
    which the matrix-inversion example removes by hand;
  - the solver options used in tests and examples are
    `{"criteria": 1e-12, "method": "Newton", "typePhi": "full", "useReal":
    True}` with `targetPre=True` for the cosine (even parity, `0`) and
    `targetPre=False` for the sine (odd parity, `1`)
    (`test_qsvt.py:186-198`);
  - QSPPACK is optional and example-only, **not** a runtime dependency.
- Known semantic differences: `docs/sphinx/conventions.rst` records that
  QSPPACK's native `W_x` rotation has **imaginary** off-diagonals while this
  circuit's response step is real; the bridge lives in the phase-generation
  options, and `reference_response` is named the executable arbiter whenever
  the two appear to disagree. Treat any other cross-package comparison as
  unresolved until translated and checked.

---

# Optional schema: Representation Record — `PhaseSequence`

Justified under the architecture gate: multiple primitives exchange and
interpret this object — `QSVT.kernel`, `QSVT.controlled_kernel`,
`sim_utils.transform`, and (through `projector_phases` plus
`walk_directions`) the LCU device kernels of provider B.

- Object name and canonical symbol: a validated QSP/QSVT phase sequence; `phi`
  for an individual angle, `d` for the degree.
- Public type or structural form, and source path:
  `cudaq_algorithms.PhaseSequence`, defined at
  `python/cudaq_algorithms/qsvt.py:68-130`, with the module-level direction
  codes `FORWARD = 0` and `ADJOINT = 1` (`qsvt.py:41-42`). A plain iterable of
  floats is accepted anywhere a `PhaseSequence` is, and is normalized by
  `_as_sequence` (`qsvt.py:133-145`).
- Mathematical meaning: `d + 1` signal-processing angles defining a degree-`d`
  polynomial, together with a per-step direction list of length `d` and a tag
  saying which convention the raw angles are in.
- Shape, layout, ordering, dtype, and units: `phases` is a `tuple[float, ...]`
  of length `d + 1` in application order; `walk_directions` is a
  `tuple[int, ...]` of length `d` over `{FORWARD, ADJOINT}`; `convention` is
  `"qsvt"` or `"qsp"`. Angles are radians. Direction inputs accept the strings
  `"forward"`, `"adjoint"`, `"backward"`, `"reverse"` (case-insensitive) and
  the integers `0` and `1`; `"backward"` and `"reverse"` are aliases of
  `ADJOINT` (`qsvt.py:44-51`).
- Normalization, sign, and phase convention — the heart of this record:
  - `phases` **always stays raw**, exactly as the caller supplied it
    (`qsvt.py:82`);
  - `projector_phases` is what the circuits implement: identity for `"qsvt"`,
    and **doubled** for `"qsp"` (`qsvt.py:117-126`);
  - the doubling is an equivalence only up to the global phase
    `exp(i * sum(phases))`, which is why provider C exists;
  - `degree` is `len(phases) - 1`.
- Required mathematical properties (applicability preconditions): at least one
  phase; all phases finite; a recognized convention; `walk_directions` either
  absent or exactly `degree` entries long. Boundedness of the implied
  polynomial is a caller obligation with **no detection**.
- Producers (>= 2 required to justify this record): callers constructing the
  object directly; QSPPACK-generated angle lists in `tests/python/test_qsvt.py`
  and in the three examples `02_hamiltonian_simulation.py`,
  `07_matrix_inversion_qsvt.py`, and `hamiltonian_simulation_qsvt.py` under
  `docs/sphinx/examples/python/`; `_as_sequence`, which promotes a bare
  iterable into one.
- Consumers: `QSVT.kernel` and `QSVT.controlled_kernel` (through
  `projector_phases` and `walk_directions`); `sim_utils.transform`;
  `pauli_lcu.apply_phase_sequence` and `apply_controlled_phase_sequence`, which
  take the two lists as explicit kernel arguments;
  `reference_response` in the test suite, which reads `phases`, `convention`,
  and `walk_directions` directly.
- Invariants preserved across the boundary: raw angles are never mutated by a
  consumer; the convention tag travels with the angles, so a sequence cannot be
  silently reinterpreted; the direction list length always matches the degree.
- Observable symptom of a misinterpretation: a **factor-of-two phase error** if
  qsp angles are read as projector angles (or vice versa) — the exact error the
  re-tag rejection at `qsvt.py:135-144` exists to prevent, with the message
  "would reinterpret its phases"; a missing or extra global phase
  `exp(i sum(phases))` if the qsp global phase is not removed before comparing
  against the 2x2 model; a response that is the correct polynomial evaluated at
  the wrong sign if a caller negates the eigenvalue themselves.
- Unsupported or ambiguous forms: a bare iterable carries no convention tag and
  defaults to `"qsvt"` — silently, so a qsp angle list passed as a plain list
  is a real hazard; an empty sequence is rejected; a degree-0 sequence is legal
  and means a pure signal phase; mutation of the tuples after construction is
  `unverified`; the class has no `__eq__`, no serialization, and no
  round-trip format, so equality is identity.
- Source paths, tests, docs, and last verification: "Identity and provenance"
  above. The validation and conversion behavior is asserted at
  `tests/python/test_qsvt.py:60-82` and `:165-172`.

---

# Optional schema: Capability Record

`Deferred:` none is extracted. The architecture gate requires multiple
independent producers or consumers of a reusable boundary. The boundaries this
family touches are owned elsewhere — unitary state preparation by
[state-preparation.md](state-preparation.md), and the block-encoding members by
[block-encoding.md](block-encoding.md) — while the polynomial-response boundary
this family provides has one producer and no in-library consumer. Revisit when
a second producer or in-library consumer of that response boundary appears.
