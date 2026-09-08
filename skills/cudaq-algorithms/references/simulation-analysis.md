# Simulation-only analysis and evolution helpers

Status: draft. Operation + object: **analyze** a **simulated statevector** —
one file, four public helpers, three of which additionally *run* a circuit on a
host simulator to obtain that statevector.

This is a Primitive Record family file. Each heading of
`assets/primitive-record-template.md` appears **once** below, with one
subsection per helper beneath it. One optional **Representation Record** — the
postselected good-subspace amplitude block — follows at the end, because three
of the four helpers produce it and a library function outside this family
consumes it. No Capability Record is created here; see
"Capabilities and composition" for why.

## The simulation-only boundary — read this before quoting anything below

Nothing in this family is a hardware capability, and nothing in it is the
library's time-evolution primitive.

1. **These helpers exist because `cudaq.get_state` exists only on
   simulators.** The module docstring
   (`python/cudaq_algorithms/sim_utils.py:3-11`) states that everything in it
   "depends on statevector access (`cudaq.get_state` / postselection
   slicing), which only exists on simulators", that the module "ships with the
   package as a clearly-labeled companion", and that it "is not part of the
   hardware-shaped API: the library classes (encodings, kernel factories,
   observables, `Walk.moment` via `cudaq.observe`) never execute
   `get_state`." `README.md:28-29` and
   `docs/sphinx/guide/qubitization_qsvt.rst:152-169` repeat the rule.
2. **`sim_utils.evolve` is not a hardware-shaped time-evolution primitive.**
   It is a host statevector experiment that loads a ket as data, runs one
   product-formula circuit under a simulator, reads the amplitudes back, and
   optionally multiplies in a phase **on the host**. The hardware-shaped
   product-formula surface is `Trotter.kernel` / `Trotter.state_kernel` /
   `trotter.apply_trotter`, whose class docstring says the `Trotter` object
   "is hardware-shaped: nothing here executes a simulator-only API (see
   `sim_utils.evolve` for statevector-based evolution)"
   (`python/cudaq_algorithms/trotter.py:334-336`). The hardware-shaped contracts
   are owned by [trotter.md](trotter.md), [qsvt.md](qsvt.md), and
   [qubitization.md](qubitization.md); do not answer them from this file.
3. **Postselection here is array slicing, not measurement.** `good_subspace`
   selects the all-zero-ancilla block deterministically from a full
   statevector. There is no shot sampling, no measured outcome, no
   success-probability protocol, no repetition or amplitude-amplification
   contract, and no classical feed-forward. A hardware realization of the same
   mathematics would need measurement and repetition, and this family says
   nothing about their cost.
4. **The returned block is not renormalized.** Its squared norm *is* the
   postselection success probability; nothing in the library reports that
   number for you.
5. **Cost grows exponentially in qubit count.** The simulator holds
   `2**(num_system + num_ancilla)` amplitudes. No source states a size limit.
6. **Never cite these helpers as evidence that an operation runs on hardware,
   and never present a `get_state` result as a measured hardware quantity.**
   Per `validation.md`, a repository assertion read at a commit is `derived`
   evidence; this skill executed nothing.

`Walk.moment` / `Walk.moments` are the contrasting case: they measure through
`cudaq.observe`, which the guide calls "a hardware-legitimate path"
(`docs/sphinx/guide/qubitization_qsvt.rst:167-169`), so they stay in the
library proper and are **not** part of this family.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths: `sim_utils.__all__` is
  `["state_from", "good_subspace", "action", "transform", "evolve"]`
  (`python/cudaq_algorithms/sim_utils.py:31`). The four helpers owned here are
  `good_subspace`, `action`, `transform`, `evolve`. They are **not** exported
  from the package root: `python/cudaq_algorithms/__init__.py:33-37` imports the
  `sim_utils` module and separately re-exports only `state_from` (from
  `common_kernels`). The reachable paths are therefore
  `from cudaq_algorithms import sim_utils` or the alias used throughout the
  tests and docs, `from cudaq_algorithms import sim_utils as sim`, then
  `sim.good_subspace(...)`, `sim.action(...)`, `sim.transform(...)`,
  `sim_utils.evolve(...)`.
- **`state_from` is in the module's `__all__` but is not a member of this
  family.** It is defined in `python/cudaq_algorithms/common_kernels.py`
  (`state_from`), re-exported here as a convenience
  (`python/cudaq_algorithms/sim_utils.py:20-21`, comment "Re-exported:
  precision-aware initial-state construction"), and is called by the
  observable-based path `Walk.moment` / `Walk.moments`
  (`python/cudaq_algorithms/qubitization.py:412,430`), which is not
  simulation-only. Its own contract — matching input dtype to `cudaq.complex()`
  because fp32 simulators reject `complex128` initial-state data — belongs to a
  future shared-kernels or measurement record.
  `Deferred:` populate `state_from` when a record owns `common_kernels`
  or the moment-measurement protocol; trigger is either record being written.
  Separately note that source calls the `cudaq.State`-argument kernel mode "the
  simulation-friendly form" (`python/cudaq_algorithms/qubitization.py:379-380`,
  `docs/sphinx/guide/qubitization_qsvt.rst:129-136`); whether loading arbitrary
  state data is available on a hardware target is **unverified** — no source
  statement covers it.
- Source paths: `python/cudaq_algorithms/sim_utils.py` (module docstring
  `:3-11`; `__all__` `:31`; `good_subspace` `:34-50`; `action` `:53-59`;
  `transform` `:62-74`; `evolve` `:77-109`). Consumed contracts:
  `python/cudaq_algorithms/pauli_lcu.py` (`PauliLCU`, `encode_kernel`, `alpha`),
  `python/cudaq_algorithms/qsvt.py` (`QSVT.kernel`, `QSVT.encoding`,
  `PhaseSequence`, `recover_real_time_evolution`),
  `python/cudaq_algorithms/trotter.py` (`Trotter.state_kernel`,
  `identity_coefficient`, `num_qubits`, the `_validate_*` guards),
  `python/cudaq_algorithms/block_encoding.py` (register geometry and the
  `BlockEncoding` protocol), `python/cudaq_algorithms/common_kernels.py`
  (`state_from`).
- Authoritative tests: `tests/python/test_trotter.py`
  (`test_sim_utils_evolve_includes_identity_phase`,
  `test_identity_only_hamiltonian_is_a_global_phase`,
  `test_sim_utils_evolve_validates_parameters`);
  `tests/python/test_pauli_lcu.py` (`action` and `good_subspace` call sites at
  `:35,45-47,59,68-71,110,206-216`); `tests/python/test_qsvt.py` (`transform`
  call sites at `:96,120,136-137,150,160,214-217`; `reference_response` as the
  executable oracle);
  `tests/python/test_block_encoding_protocol.py:108-116` (a foreign encoding
  drives `QSVT`, but the good subspace is sliced by hand rather than through
  `good_subspace`); `tests/python/dense_references.py` (the shared dense
  oracle); `tests/python/conftest.py` (target pinned to `qpp-cpu`, fp64, honoring
  `CUDAQ_DEFAULT_SIMULATOR`); `tests/python/test_examples.py` (every example
  script must run and self-verify in CI).
- Authoritative documentation: `docs/sphinx/guide/qubitization_qsvt.rst`
  ("Hardware-shaped vs. simulation-only", `:152-169`; "Simulator selection",
  `:178-187`); `docs/sphinx/guide/trotter.rst` ("Simulation-only helper",
  `:128-149`); `docs/sphinx/conventions.rst` (good-subspace geometry `:90-108`;
  "Simulation targets and tolerances" `:149-155`);
  `docs/sphinx/api/python_api.rst:65-71` (`automodule cudaq_algorithms.sim_utils`);
  `README.md:28-29`. Examples: `docs/sphinx/examples/python/pauli_lcu_demo.py`
  (`action`, `good_subspace`), `02_hamiltonian_simulation.py` and
  `hamiltonian_simulation_qsvt.py` (`transform`, `evolve`),
  `07_matrix_inversion_qsvt.py` (`transform`), `trotter_chemistry.py`
  (`evolve`).
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml:11,17`) and
  CUDA-Q pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af`
  (`.cudaq_version`). No test, example, or interpreter was executed for this
  family in this session.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03), `git describe` `0.1.0-6-g61ac072`.
- Lifecycle: draft.
- Replacement and migration notes: none. No symbol here is deprecated, and no
  predecessor is recorded in source.
- **Evidence rule for this family.** Every test and example assertion named
  below is *cited repository evidence* read at the commit above — labeled
  `derived`, never a fresh measurement, never a runtime or performance claim.

## Classification

Shared by all four helpers:

- Operation + mathematical object (primary identity): **analyze** a **simulated
  statevector**. `evolve` additionally performs an **evolve** operation on a
  **statevector**, but only inside a simulator, so it is routed here and not as
  a time-evolution primitive.
- Kind: **simulation-only analysis** (the `architecture.md` vocabulary value),
  for all four.
- Execution layers: **simulation-only analysis** throughout; `action`,
  `transform`, and `evolve` additionally drive a kernel factory and a simulator
  execution, and all four finish with host array work. None emits a device
  kernel of its own.
- Domain: domain-independent.
- Required dependencies: `cudaq` with a **statevector simulator target**, and
  NumPy (imported lazily inside `good_subspace` and `evolve`). Importing
  `sim_utils` also imports `.common_kernels` and `.trotter`
  (`python/cudaq_algorithms/sim_utils.py:20-22`).
- Optional dependencies: none.

Per helper:

| Helper | Routine role | Abstraction level | Parameterization | Input representations | Output representations |
| --- | --- | --- | --- | --- | --- |
| `good_subspace` | computational | leaf operation | runtime | a block encoding's geometry (`num_system`, `num_ancilla`) plus a full statevector array | good-subspace amplitude block (Representation Record below) |
| `action` | driver | composite protocol | runtime | `PauliLCU` encoding plus a system ket | good-subspace amplitude block, equal to `(H/alpha)|ket>` |
| `transform` | driver | composite protocol | runtime | `QSVT` transformer, a system ket, a `PhaseSequence` or raw phase list | good-subspace amplitude block, equal to the polynomial response applied to `ket` |
| `evolve` | driver | composite protocol | runtime | `Trotter` object plus a full system ket and evolution parameters | full statevector of the evolved register — **not** a postselected block |

Routine role follows `architecture.md`'s completeness criterion, not execution
location: `good_subspace` performs one distinct, independently usable task,
while the other three sequence a kernel factory, a simulator run, and host
post-processing to answer a complete user question.

Provisional metadata — record the value, do not route on it:

- Exactness: `good_subspace` **exact** (an index-range copy). `action` and
  `transform` are exact for the circuit that was run, subject to simulator
  floating-point precision; the *scientific* accuracy of `transform` is
  whatever the caller's phase sequence implements. `evolve` is **approximate**:
  a product-formula approximation controlled by `(steps, order)`, plus simulator
  precision.
- Uncertainty: deterministic. No sampling, no shot noise, no random number
  generation anywhere in the module.
- Method: direct.

## Scientific contract

Shared conventions, all `derived` from `docs/sphinx/conventions.rst:90-108` and
`python/cudaq_algorithms/block_encoding.py:19-29`: qubit 0 is the least
significant statevector index bit; the kernel factories allocate the **system
register first** and the ancilla/signal register after it; therefore the
all-zero-ancilla block is the **leading** `2**num_system` amplitudes. Pauli
words index left-to-right by qubit (`conventions.md`).

### `good_subspace(encoding, state)`

- Purpose: postselect the all-zero-ancilla block of a simulated statevector.
- Mathematical definition: for a state `|Psi>` on `system (x) ancilla` with
  `n_s = num_system`, `n_a = num_ancilla`, the little-endian index factorizes as
  `index = i_sys + 2**n_s * i_anc`, so `i_anc = 0` is exactly the contiguous
  range `index < 2**n_s`. The helper returns that leading block, i.e. the
  unnormalized vector `(<0|_anc (x) I) |Psi>`. This index derivation is the
  reason the docstring calls it "the first contiguous block of `2**num_system`
  amplitudes" (`python/cudaq_algorithms/sim_utils.py:37-42`).
- Why and when to use: to read the flagged block of any zero-flagged
  block-encoding circuit you ran yourself on a simulator, including circuits you
  composed from the module-level kernels.
- When not to use: on hardware; when the register allocation order is not
  system-first; when you need a normalized state or a measured probability.
- Approximation controls: none. The operation is exact.

### `action(encoding, ket)`

- Purpose: obtain `(H/alpha)|ket>` by simulating a `PauliLCU` block encoding
  and postselecting.
- Mathematical definition: the encoding satisfies
  `(<0|_anc (x) I) U_A (|0>_anc (x) I) = H / alpha` with
  `alpha = sum_i |c_i|` over the **retained** terms
  (`python/cudaq_algorithms/pauli_lcu.py:21-23,552-554`). Hence
  `action(encoding, ket) = (H/alpha) |ket>`, and the docstring's instruction
  "Multiply by `encoding.alpha` to recover `H|ket>`" is exactly that identity
  rearranged (`python/cudaq_algorithms/sim_utils.py:54-57`).
- Why and when to use: to check an encoding against a dense reference, or to
  apply `H` once to a state during development.
- When not to use: as an operator-application primitive for a real workflow —
  it materializes a statevector; and never to claim `H` was applied on
  hardware.
- Approximation controls: none of its own. Note that `alpha` depends on which
  terms were retained: `include_identity=False` or a raised
  `coefficient_threshold` changes both the encoded operator and `alpha`
  (`python/cudaq_algorithms/pauli_lcu.py:479-502`).

### `transform(transformer, ket, sequence, convention=None)`

- Purpose: obtain the good-subspace state after a QSVT phase sequence.
- Mathematical definition: for an eigenstate of `H` with eigenvalue `lambda`,
  the good-subspace block acts as multiplication by `p(lambda / alpha)`, where
  `p` is the degree-`d` polynomial implemented by the `d + 1` projector phases
  interleaved with `d` walk steps
  (`python/cudaq_algorithms/sim_utils.py:66-71`;
  `python/cudaq_algorithms/qsvt.py:15-19`). The argument is the **plain**
  scaled eigenvalue: the walk block encodes `-H/alpha` and the circuits fold that
  sign in, so **no caller-side negation is correct**. On a general input the
  result is `p(H/alpha)|ket>`, which is why the block reconstructed
  column-by-column equals `V diag(p(lambda_j)) V^dagger` in the eigenbasis
  (`tests/python/test_qsvt.py:107-126`).
- Why and when to use: to inspect the polynomial response of an encoded
  spectrum, or to produce the cosine/sine components that
  `recover_real_time_evolution` recombines.
- When not to use: to claim a hardware QSVT result; to obtain `exp(-iHt)` from
  one call (it is complex, so two sequences plus classical recombination are
  needed — `docs/sphinx/guide/qubitization_qsvt.rst`,
  `python/cudaq_algorithms/qsvt.py:304-323`); or as a phase-generation tool —
  the library generates no phases.
- Approximation controls: entirely the caller's phase sequence — its degree and
  the approximation it encodes. The library validates the sequence's structure,
  never its approximation quality.

### `evolve(evolution, ket, time, steps=1, order=2, include_identity_phase=True)`

- Purpose: simulate a Suzuki-Trotter evolution of `ket` and return the evolved
  statevector, with the circuit-unrealizable identity phase optionally restored
  on the host.
- Mathematical definition: write `H = c I + H'` with
  `c = evolution.identity_coefficient` and `H'` the retained non-identity terms.
  The circuit applies the order-`p` product formula for `H'` only — identity
  terms cannot be realized as a circuit on the evolved register
  (`python/cudaq_algorithms/trotter.py:20-27`,
  `docs/sphinx/guide/trotter.rst:119-127`). The helper returns
  `S_p(t/steps)^steps |ket>`, multiplied by `exp(-i c t)` when
  `include_identity_phase` is true **and** `c != 0`, so that the result
  approximates the full `exp(-i H t)|ket>`
  (`python/cudaq_algorithms/sim_utils.py:83-88,106-109`). The formulas
  themselves: order 1 is one sweep of `exp(-i c_i dt P_i)`; order 2 is the
  symmetric half-angle forward-then-reverse (Strang) sweep; order 4 is
  Forest-Ruth — three symmetric second-order sub-steps with time fractions
  `w1, w0, w1`, `w1 = 1/(2 - 2**(1/3))`, `w0 = 1 - 2*w1`
  (`python/cudaq_algorithms/trotter.py:46-54,67-119`).
- Why and when to use: to study product-formula convergence, or to compare a
  Trotter circuit against an exact reference during development.
- When not to use: as the time-evolution primitive of a workflow, as evidence of
  a hardware evolution, or to obtain a controlled evolution — `Trotter` exposes
  no controlled factory, and the identity phase that is a harmless global phase
  here becomes a **real relative phase** under control or interference
  (`python/cudaq_algorithms/trotter.py:20-27`).
- Approximation controls: `steps` and `order` (`order in {1, 2, 4}`). Increasing
  either reduces product-formula error; see "Accuracy and limitations".
  Term ordering is fixed earlier, at `Trotter` construction, and is not a
  parameter of this helper.

## Inputs

### `good_subspace`

- Arguments: `encoding` (annotated `PauliLCU`), `state` (`ArrayLike`).
- Shapes/ranks: `state` must have shape exactly `(2**(num_system +
  num_ancilla),)` — one-dimensional, exact length.
- Dtypes/domains: coerced by `np.asarray(state, dtype=np.complex128)`, so any
  array-like convertible to `complex128` is accepted, including a `cudaq.State`
  (this is how every call site passes it).
- Units: none.
- Ordering/layout: little-endian, system register first. The helper cannot check
  this and assumes it.
- Normalization: not required and not checked.
- Required mathematical properties: only the dimension. **The helper does not
  verify that `state` came from a circuit built with this encoding**, nor that
  the ancillas are the trailing register.
- Validation and rejection behavior: exactly one check —
  `ValueError(f"expected a statevector of dimension {expected}, got shape
  {vector.shape}")` when the shape differs, with
  `expected = 1 << (num_system + num_ancilla)`
  (`python/cudaq_algorithms/sim_utils.py:45-49`). No other input is rejected.

### `action`

- Arguments: `encoding` (annotated `PauliLCU`), `ket` (`ArrayLike`).
- Shapes/ranks: `ket` is intended to be a `2**num_system` statevector.
- Dtypes/domains: passed to `state_from(ket)`, which applies
  `np.asarray(ket, dtype=cudaq.complex())` — i.e. the dtype the **active
  target** expects (`python/cudaq_algorithms/common_kernels.py`, `state_from`).
- Ordering/layout, normalization: the encoding's system-register convention; a
  normalized ket is assumed, not enforced.
- Required mathematical properties: `ket` must match the encoding's system
  width.
- Validation and rejection behavior: **`action` performs no host validation of
  its own.** A wrong-width `ket` is not rejected before the simulation; the only
  possible rejection is `good_subspace`'s post-hoc dimension `ValueError`, or an
  error raised inside CUDA-Q while loading or allocating from the state. The
  precise behavior for a non-power-of-two length, a two-dimensional array, or an
  unnormalized ket is **unverified** — no source statement and no test covers
  it.

### `transform`

- Arguments: `transformer` (annotated `QSVT`), `ket` (`ArrayLike`), `sequence`
  (`PhaseSequence | Iterable[float]`), `convention` (`str | None`, default
  `None`).
- Shapes/ranks: `ket` as in `action`, against `transformer.encoding.num_system`.
- Dtypes/domains: `ket` via `state_from`. `sequence` is either a validated
  `PhaseSequence` or a raw iterable of `d + 1` floats.
- Ordering/layout: the signal register of `num_ancilla` qubits is allocated
  **after** the system register and starts in `|0...0>`
  (`python/cudaq_algorithms/qsvt.py:10-13,193-196`).
- Required mathematical properties: `transformer.encoding.num_ancilla >= 1`,
  enforced by `QSVT.__init__` before this helper is ever reached
  (`python/cudaq_algorithms/qsvt.py:161-166`).
- Validation and rejection behavior, all raised by the QSVT layer, not by
  `transform`:
  - `ValueError` when `sequence` is already a tagged `PhaseSequence` and a
    *different* `convention` is passed — re-tagging would reinterpret rather
    than convert the raw phases, a factor-of-two phase error
    (`python/cudaq_algorithms/qsvt.py:133-146`);
  - `PhaseSequence` construction errors for empty phases, non-finite phases, a
    convention outside `{"qsvt", "qsp"}`, a `walk_directions` length other than
    `len(phases) - 1`, or an unrecognized direction
    (`python/cudaq_algorithms/qsvt.py:89-111,54-60`).
  - No validation of `ket` width, as in `action`.

### `evolve`

- Arguments: `evolution` (`Trotter`), `ket` (`ArrayLike`), `time` (`float`),
  `steps` (`int`, default `1`), `order` (`int`, default
  `SECOND_ORDER_TROTTER == 2`), `include_identity_phase` (`bool`, default
  `True`).
- Shapes/ranks: `ket` must be one-dimensional with exactly
  `2**evolution.num_qubits` entries — this helper is the one member of the
  family that checks its own ket.
- Dtypes/domains: `np.asarray(ket, dtype=np.complex128)` first, then
  `state_from` for target precision. `time` must be finite; `steps` a positive
  integer with no silent truncation; `order in {1, 2, 4}`.
- Units: `time` is in the inverse units of the Hamiltonian coefficients; the
  library fixes no unit system.
- Normalization: a normalized `ket` is assumed, not enforced.
- Validation and rejection behavior, in the order the checks run:
  1. `ValueError("ket must be a 1-D statevector of dimension {2**n} for {n}
     qubit(s); got shape {shape}")` from `evolve` itself
     (`python/cudaq_algorithms/sim_utils.py:96-101`);
  2. then `Trotter.state_kernel` -> `_prepared_args`, which raises
     `ValueError("time must be a finite number")`,
     `ValueError("steps must be a positive integer")`, and
     `ValueError("order must be one of {1, 2, 4}")`
     (`python/cudaq_algorithms/trotter.py:249-269,390-394`).
- **Why this validation exists at all.** `trotter.apply_trotter` turns invalid
  runtime inputs into a **silent no-op** — zero steps, mismatched
  coefficient/word lengths, or an unsupported order leave the register unchanged
  — because `return` inside a Python `@cudaq.kernel` is silently ignored
  (cuda-quantum#4845), so every guard is a positive `if` block
  (`python/cudaq_algorithms/trotter.py:70-84`). Without host validation
  `evolve` would return the **unevolved** state as though it were the
  evolution result. `tests/python/test_trotter.py:627-644` documents exactly
  that as a fixed regression. This is the `conventions.md` rule "validation
  happens on the host, before device kernels" in its sharpest form.

## Outputs

### `good_subspace`

- Return type: `NDArray[np.complex128]` of length `2**num_system`, produced by
  `vector[:1 << num_system].copy()` — an independent copy, never a view into
  the caller's array (`python/cudaq_algorithms/sim_utils.py:50`).
- Mathematical meaning: the unnormalized amplitudes of the all-zero-ancilla
  block, `(<0|_anc (x) I)|Psi>`.
- Shape/register geometry: the system register only; the ancilla register is
  projected out, not returned.
- Normalization, sign, and phase: **not renormalized.** `norm**2` is the
  probability of the all-zero-ancilla outcome for a normalized input, which
  `docs/sphinx/examples/python/pauli_lcu_demo.py:45-47` prints as "success
  probability" — an example-level interpretation of the Born rule, not a
  library-returned quantity. Global phase is whatever the circuit produced.
- Observable or measurement interpretation: none. This is exact postselection
  in a simulator, not a measurement outcome.
- Error/status information: none beyond the `ValueError`. There is no success
  flag, no probability field, and no warning when the block norm is tiny.

### `action`

- Return type: `NDArray[np.complex128]` of length `2**num_system`.
- Mathematical meaning: `(H/alpha)|ket>`; multiply by `encoding.alpha` for
  `H|ket>`.
- Normalization, sign, and phase: unnormalized. Since `||H|| <= alpha`, the
  norm is at most `||ket||`; it equals `1` exactly for a single Pauli-word
  encoding, which the demo states and prints
  (`docs/sphinx/examples/python/pauli_lcu_demo.py:89-91`). Signs are the
  encoding's: a negative single-term coefficient produces the opposite state
  (`tests/python/test_pauli_lcu.py:50-59`).
- Error/status information: none of its own.

### `transform`

- Return type: `NDArray[np.complex128]` of length
  `2**transformer.encoding.num_system`.
- Mathematical meaning: the good-subspace state after the sequence, equal to
  `p(H/alpha)|ket>` in the sense above.
- Normalization, sign, and phase: unnormalized, and **phase-sensitive to the
  convention**. `PhaseSequence` keeps `phases` raw and exposes
  `projector_phases`, which **doubles** `qsp` phases; the circuit executing the
  doubled phases differs from the `qsp` signal model by the global phase
  `exp(i * sum(phases))`
  (`python/cudaq_algorithms/qsvt.py:118-126,304-316`). `transform` does **not**
  remove that phase. Callers remove it either with
  `recover_real_time_evolution` (which strips it per sequence) or by hand, as in
  `docs/sphinx/examples/python/07_matrix_inversion_qsvt.py:190-192`
  (`raw * np.exp(-1j * np.sum(phases))`).
- Error/status information: none of its own.

### `evolve`

- Return type: `NDArray[np.complex128]` of length `2**evolution.num_qubits`.
- Mathematical meaning: the **full** evolved statevector. This helper is the
  family's asymmetry: there is no ancilla register and no postselection, so
  nothing is projected out.
- Shape/register geometry: the whole evolved system register.
- Normalization, sign, and phase: norm-preserving up to simulator precision —
  the circuit is unitary and the identity factor `exp(-i c t)` has unit
  modulus. With `include_identity_phase=True` (the default) the result is
  directly comparable to `exp(-iHt)|ket>` **without phase alignment**; with
  `False` it differs from it by exactly `exp(-i c t)`. The test comments make
  this explicit ("Direct comparison, NOT phase-aligned",
  `tests/python/test_trotter.py:484-495`).
- Error/status information: none. No error estimate, no achieved-accuracy
  field, and no indication of how far the product formula is from exact.

## Capabilities and composition

**No capability ID is minted or required by name in this record.** Per
`architecture.md`, a capability is extracted only when multiple independent
producers or consumers demonstrate a reusable boundary. These helpers consume
concrete objects documented by [block-encoding.md](block-encoding.md),
[qsvt.md](qsvt.md), and [trotter.md](trotter.md); only `good_subspace` uses a
small structural geometry subset. Treat that as representation-level
composition, not a new simulation capability.

The requirements are therefore stated **structurally**, from source:

- Direction: **requires** — the geometry and normalization surface of a
  zero-flagged block encoding.
  - Boundary representation: an object exposing `num_system: int` and
    `num_ancilla: int` (`good_subspace`, and through it `action` and
    `transform`), plus `alpha: float` for the caller's own rescaling and
    `encode_kernel()` for `action`.
  - Semantic invariants: `(<0|_anc (x) I) U_A (|0>_anc (x) I) = H/alpha`;
    `num_ancilla >= 1`; system register allocated first, ancillas after, so the
    good subspace is the leading `2**num_system` amplitudes.
  - Host/device/simulation boundary: the encoding's own factories are
    hardware-shaped; only the postselection and the `get_state` call are
    simulation-only.
  - **Annotation versus structural use — an honest divergence.**
    `good_subspace` and `action` are annotated `encoding: PauliLCU` and
    `transform` is annotated `transformer: QSVT`, but the bodies touch only
    `num_system`, `num_ancilla`, `encode_kernel`, `kernel`, and `encoding`.
    They are therefore *structurally* usable with any object satisfying the
    `BlockEncoding` protocol. Whether that works is **unverified**: no test
    passes a foreign encoding to any of these helpers, and
    `tests/python/test_block_encoding_protocol.py:108-116` slices
    `state[:len(psi)]` by hand for its `ForeignEncoding` instead of calling
    `good_subspace`.
- Direction: **requires** — a validated QSVT phase sequence (`transform`):
  `PhaseSequence` or a raw phase iterable, with the projector/`qsp` doubling
  and retag rules above.
- Direction: **requires** — a constructed `Trotter` object (`evolve`),
  supplying `num_qubits`, `identity_coefficient`, and `state_kernel`. Note
  `evolve` does **not** accept `state_prep`: `Trotter.state_kernel` takes its
  register from a `cudaq.State` by construction
  (`python/cudaq_algorithms/trotter.py:456-471`). The one-argument
  preparation-kernel seam is a different input mode, documented in
  [state-preparation.md](state-preparation.md).
- Direction: **provides** — nothing composable on the device, by design. These
  helpers terminate a workflow on the host and return NumPy arrays.
- **Candidate boundary, not a capability.** The good-subspace amplitude block
  produced by `good_subspace`/`action`/`transform` is consumed by
  `recover_real_time_evolution` (`python/cudaq_algorithms/qsvt.py:304-323`), a
  host classical-reconstruction function outside this family, and by the
  examples. Three in-family producers and one library consumer justify the
  **Representation Record** at the end of this file, but not a capability ID:
  one consumer does not demonstrate a reusable multi-consumer boundary.

Composition rules that must not be inferred away:

- Simulation-only helpers cannot appear in a hardware-shaped composition at
  all. Substituting `sim_utils.evolve` for a `Trotter` kernel, or `action` for
  an applied operator, changes the execution layer, not just the spelling.
- `transform` output normally needs at least one more classical step (global
  phase removal, and for real-time evolution a second sequence plus
  `recover_real_time_evolution`). Treating a single `transform` call as
  `exp(-iHt)` is wrong.

## Composite protocol

Applies to `action`, `transform`, and `evolve` (abstraction level: composite
protocol). Omitted for `good_subspace`, a leaf operation.

- Required lower-level capabilities: the structural requirements listed above,
  plus a statevector simulator target and `state_from` for precision-matched
  state construction.
- Canonical reference composition — the source itself, which is short enough
  that the whole protocol is visible:
  - `action`: `cudaq.get_state(encoding.encode_kernel(), state_from(ket))`,
    then `good_subspace(encoding, state)`
    (`python/cudaq_algorithms/sim_utils.py:58-59`).
  - `transform`:
    `cudaq.get_state(transformer.kernel(sequence, convention), state_from(ket))`,
    then `good_subspace(transformer.encoding, state)`
    (`python/cudaq_algorithms/sim_utils.py:72-74`).
  - `evolve`: validate the ket, `kernel = evolution.state_kernel(time, steps,
    order)`, `np.asarray(cudaq.get_state(kernel, state_from(ket_array)))`, then
    the optional host phase multiplication
    (`python/cudaq_algorithms/sim_utils.py:95-109`).
- Default recipe and its applicability conditions: exactly one circuit
  execution per call, on the currently active target, with the ket loaded as
  data through the `cudaq.State` kernel mode. Applicable only when that target
  is a statevector simulator and the register fits in memory.
- Materially different alternatives, all outside this family: compose the
  module-level kernels inside your own kernel and slice the result yourself
  (`docs/sphinx/examples/python/pauli_lcu_demo.py:65-79`, pinned against
  `action` at `atol=1e-12`); use the `state_prep` injection mode to get a
  zero-argument, directly sampleable circuit and measure it; for moments, use
  `Walk.moment` / `Walk.moments` through `cudaq.observe`. The last two are
  hardware-shaped; these helpers are not.
- Propagated conventions: qubit ordering, Pauli-word indexing, system-first
  register geometry, the walk's `-H/alpha` sign folded into the QSVT response,
  and the projector-versus-`qsp` phase convention.
- Propagated errors: every `ValueError` listed under Inputs originates in the
  encoding, QSVT, or Trotter layer except `good_subspace`'s dimension check and
  `evolve`'s ket check. The silent-no-op semantics of `apply_trotter` are
  *neutralized* for `evolve` by host validation, but remain live for a caller
  who composes `apply_trotter` directly.
- Propagated resources and how they compose: see "Resources" — the circuit cost
  is the underlying primitive's, and the simulator cost is exponential in
  register width.
- Component substitution: `action`/`transform` accept any object with the
  structural surface above (**unverified** for anything other than `PauliLCU`
  and `QSVT`); a substitute must preserve the system-first register geometry and
  the `num_ancilla` count, or the postselection silently selects the wrong
  amplitudes. `evolve` requires a `Trotter` object specifically — it calls
  `state_kernel`, `num_qubits`, and `identity_coefficient`.

## Accuracy and limitations

### Error behavior or bounds

- `good_subspace`: exact, up to the precision of the amplitudes handed to it.
- `action`, `transform`: exact for the circuit that ran, limited by simulator
  precision. `transform`'s scientific accuracy is entirely the caller's phase
  sequence; **the library states no polynomial-approximation error bound and
  generates no phases.**
- `evolve`: order-`p` product-formula error, documented as
  `O(t^2/steps)` for order 1, `O(t^3/steps^2)` for order 2, and
  `O(t^5/steps^4)` for order 4 (`docs/sphinx/guide/trotter.rst:55-64`).
  **No rigorous commutator bound exists anywhere in the source** — the scaling
  is documented asymptotics plus the fitted-slope study described under
  Validation. Do not convert it into a guaranteed error for a specific
  Hamiltonian.

### Precision sensitivity

The default CUDA-Q target is fp32 and carries roughly `1e-7` statevector error;
the library tests pin `qpp-cpu` (fp64) and assert at `1e-10` to `1e-12`
(`docs/sphinx/conventions.rst:149-155`,
`docs/sphinx/guide/qubitization_qsvt.rst:178-187`,
`tests/python/conftest.py`). When one of these helpers shows a `~1e-7`
residual, suspect the target precision before the code. The returned arrays are
`complex128` regardless of target precision, so a `float32` simulator's error is
silently widened into a `complex128` container.

### Unsupported inputs

- Hardware and any non-statevector target: `cudaq.get_state` does not exist
  there (module docstring; both guides). This is documented as unsupported;
  the *failure mode* on such a target is **unverified** — no source names an
  error type or message.
- `good_subspace` with a state whose dimension is not
  `2**(num_system + num_ancilla)`: rejected with the `ValueError` above.
- `evolve` with a non-1-D ket, wrong dimension, non-finite time, non-positive
  or fractional steps, or an order outside `{1, 2, 4}`: rejected.

### Known implementation limitations

1. **No renormalization and no success probability.** Three of four helpers
   return an unnormalized block and no probability. The caller must compute
   `norm**2` and decide what it means.
2. **`good_subspace` validates the dimension only.** It cannot detect a
   different register allocation order, a state from a different encoding, or an
   unnormalized state; the result is then silently wrong rather than rejected.
3. **`action` and `transform` do no host ket validation.** A width mismatch
   surfaces late, from `good_subspace` or from CUDA-Q, and the exact behavior is
   unverified.
4. **`transform` leaves the `qsp` global phase in place.** A caller who forgets
   `exp(-i * sum(phases))` sees a rotated state, not an error.
5. **`evolve` returns no accuracy information**, and its default
   `include_identity_phase=True` differs from the circuit primitives, which
   cannot include that phase. Comparing a circuit result with an `evolve`
   result without accounting for `exp(-i c t)` is the predictable mistake; the
   example does it correctly by multiplying the kernel path by the phase
   (`docs/sphinx/examples/python/trotter_chemistry.py:133-136`).
6. **Exponential memory in register width**, with no documented limit.
7. **`sim_utils.__all__` includes `state_from`,** which is not simulation-only;
   membership in this module is not proof that a symbol is simulator-bound.
8. **The helpers are not exported from the package root** — only the module is.

### Unsupported versus unverified

| Behavior | Label | Why the label |
| --- | --- | --- |
| Execution on a hardware/QPU target | unsupported (documented) | module docstring and both guides state statevector access exists only on simulators |
| Error raised on a hardware target | unverified | no source or test names the failure |
| Foreign (non-`PauliLCU`/`QSVT`) encodings passed to these helpers | unverified | structurally plausible from the bodies; no test does it, and the protocol test slices by hand instead |
| `action`/`transform` with a wrong-width, non-power-of-two, or 2-D ket | unverified | no host check, no test, no documented message |
| Unnormalized input ket in any helper | unverified | never checked, never tested |
| `good_subspace`'s dimension `ValueError` message and type | derived, untested | present in source; no test asserts it |
| Density matrices, noisy simulators, mixed states | absent | the module has no such code path |
| Sampling, expectation values, fidelity, overlap, renormalization, amplitude amplification, success-probability reporting | absent | no such helper exists in the module; `Walk.moment`/`moments` own expectation values, outside this family |
| Batched or multi-ket input | absent | every helper takes one ket |

Read the labels literally: `unsupported`, `absent`, and `unverified` are three
different statements, and none of them is "safe to try and report as a library
capability".

## Resources

Every quantity below is a **simulator-side or host-side** quantity, or a
circuit-level count owned by another primitive. None is a transpiled gate
count, a depth, a runtime, or a hardware memory figure, and none may be
presented as one.

| Quantity | Metric and unit | Abstraction level | Assumptions | Status | Controlling parameters | Limitations / composition |
| --- | --- | --- | --- | --- | --- | --- |
| Full statevector held during `action`/`transform` | amplitudes, count `2**(num_system + num_ancilla)` | simulator state | statevector simulator; system-first allocation | exact (derived from the geometry) | encoding width and ancilla count | bytes depend on target precision (fp32/fp64) and were not measured |
| Statevector held during `evolve` | amplitudes, count `2**num_qubits` | simulator state | statevector simulator | exact | Hamiltonian register width | no ancillas are allocated by this path |
| Returned array | amplitudes, `2**num_system` (`good_subspace`, `action`, `transform`) or `2**num_qubits` (`evolve`) | host memory | NumPy `complex128`, i.e. 16 bytes per amplitude | exact | as above | `good_subspace` returns a copy, so the peak holds both the full state and the block |
| Circuit executions per call | count | simulator invocations | one `cudaq.get_state` per call | exact | none — it is always 1 for `action`/`transform`/`evolve`, and 0 for `good_subspace` | a real-time-evolution QSVT workflow needs **two** `transform` calls (cosine and sine) plus classical recombination |
| Product-formula circuit cost behind `evolve` | `pauli_rotations` (count of `exp_pauli` operations) | logical operation | the emitted product-formula circuit | exact for that circuit | `num_terms`, `steps`, `order` | from `Trotter.resources(steps, order)`; both parameters are required so an estimate can never describe a different circuit than the one built |
| `estimated_cx_count` behind `evolve` | count | **decomposition proxy** | two CNOTs per additional non-identity Pauli per rotation; **unmerged** circuit | estimated | same as above | never a transpiled count; the guide records that merging back-to-back half-rotations at sweep/step boundaries is deliberately deferred |
| State loading (`cudaq.qvector(state)`) | — | — | — | **not modeled** | — | `Trotter.resources` does not account for it |

**Absent by design, and their absence must be reported rather than filled:**
there is **no** resource estimator for `PauliLCU`, `BlockEncoding`, `Walk`, or
`QSVT`, so no T-count, Toffoli count, depth, query count, runtime, or memory
figure may be quoted for `action` or `transform`. There is likewise no
success-probability, repetition-count, or amplitude-amplification budget for
any postselected path in this family, and no shot count anywhere, because
nothing here samples.

Adjacent cost-relevant quantities that are **not** resource estimates:
`PauliLCU.alpha` (the coefficient 1-norm, a spectral bound `||H|| <= alpha`)
and `PauliLCU.num_ancilla = max(1, (num_terms - 1).bit_length())` (a qubit-count
fact).

## Validation

- Independent oracle: `tests/python/dense_references.py` — `dense_matrix(terms,
  num_qubits)` constructs the Pauli-sum matrix directly by bit manipulation in
  little-endian order, and `random_ket(num_qubits, seed)` gives seeded
  normalized states. It uses NumPy only, so it shares no code path with the
  primitives under test. Additional independent oracles:
  `test_trotter.py::_exact_evolve` (eigendecomposition of the dense matrix),
  `test_trotter.py::_simulate_trotter` (an explicit Pauli-rotation statevector
  simulator),   `test_qsvt.py::reference_response` (an explicit 2x2 signal-model
  matrix product, which `docs/sphinx/conventions.rst:127-137` calls "the
  executable specification" and "the executable arbiter" for the QSVT response),
  and the analytic Chebyshev identity
  `sum_j w_j cos(2 k arccos(E_j/alpha))`.
- Invariants exercised through these helpers: the encoded block equals
  `dense_matrix @ ket / alpha`; sign retention for a negative single-term
  encoding; identity-only encodings acting as `+/- I`; the QSVT response at the
  **plain** scaled eigenvalue with an explicit guard that the negated-eigenvalue
  prediction must *not* match at odd degree (`test_qsvt.py:101-104`); `qsp`
  phases executing as doubled projector phases; a degree-0 sequence acting as a
  pure signal phase; the
  identity phase appearing in `evolve` and disappearing when switched off; and
  parameter validation raising instead of silently returning an unevolved
  state.
- Representative cases and predeclared tolerances, all read from committed
  assertions:

| Claim under test | Call site | Oracle | Tolerance |
| --- | --- | --- | --- |
| `action` returns `(H/alpha)ket` for a four-term 2-qubit `H` | `test_pauli_lcu.py:35` | `dense_matrix @ ket / alpha` | `atol=1e-10` |
| `action` agrees across `SpinOperator` and `(coeff, word)` input forms | `test_pauli_lcu.py:45-47` | the other input path | `atol=1e-10` |
| `action` keeps the sign of a negative single-term encoding | `test_pauli_lcu.py:59` | `dense_matrix` | `atol=1e-10` |
| `action` on identity-only encodings returns `+ket` / `-ket` | `test_pauli_lcu.py:206-209` | analytic | `atol=1e-12` |
| `good_subspace` of `walk_kernel(power=1)` equals `T_1(-H/alpha)ket` | `test_pauli_lcu.py:68-71` | `-dense_matrix @ ket` | `atol=1e-10` |
| `good_subspace` norm-squared gives the even Chebyshev moment `2 p_0 - 1` | `test_pauli_lcu.py:107-115` | analytic `sum_j w_j cos(2 k arccos(E_j/alpha))` | `abs=1e-10` |
| `transform` matches the signal model on an eigenstate, and the negated prediction does not | `test_qsvt.py:85-104` | `reference_response` | `atol=1e-10`; guard `abs(response - flipped) > 1e-6` |
| `transform` reconstructs the full 4x4 good-subspace block | `test_qsvt.py:107-126` | `V diag(response) V^dagger` | `atol=1e-9` |
| `qsp` phases execute as doubled projector phases | `test_qsvt.py:129-143` | the doubled-`qsvt` sequence | `atol=1e-12`, plus a guard that the two conventions genuinely differ |
| degree-0 sequence is a pure signal phase | `test_qsvt.py:146-151` | `exp(0.7i) ket` | `atol=1e-12` |
| `transform` + `recover_real_time_evolution` reproduce `exp(-iHt)ket` from QSPPACK phases | `test_qsvt.py:201-225` | eigendecomposition; external QSPPACK phase generation (`importorskip`) | `norm < 1e-8` |
| `evolve` includes the identity phase and needs no phase alignment | `test_trotter.py:476-495` | `_exact_evolve` | `norm < 1e-3` at `steps=64, order=2`; and `> 0.1` with the phase switched off |
| `evolve` on an identity-only Hamiltonian is exactly a global phase | `test_trotter.py:498-511` | `exp(0.1i) ket` | `atol=1e-12` |
| `evolve` rejects bad steps, order, time, ket rank, and ket dimension | `test_trotter.py:627-644` | `pytest.raises(ValueError, match=...)` | exact match on `steps`, `order`, `finite`, `dimension`, `1-D` |
| `evolve` agrees with a hand-composed `apply_trotter` kernel once the identity phase is reintroduced, and with an exact reference | `docs/sphinx/examples/python/trotter_chemistry.py:154-155` | `_exact_evolve`-style eigendecomposition | predeclared in the script: fails if `l2_error > 5e-3` or `kernel_vs_evolve > 1e-12` |
| a hand-composed encoding kernel plus `good_subspace` matches `action` | `docs/sphinx/examples/python/pauli_lcu_demo.py:65-79` | `action` itself (an internal consistency check, not an independent oracle) | `atol=1e-12` |
| `transform`-based QSVT evolution reproduces `exp(-iHt)` | `docs/sphinx/examples/python/02_hamiltonian_simulation.py:106-126` | dense exact evolution | `max amplitude error < 1e-6` |
| `transform`-based matrix inversion solves `Ax = b` | `docs/sphinx/examples/python/07_matrix_inversion_qsvt.py:190-215` | `numpy.linalg.solve` | `fidelity > 0.999`, `residual < 1e-2` |

  Examples are executed in CI on the fp64 CPU simulator and must exit zero with
  their own self-checks passing (`tests/python/test_examples.py`).

- Expected failure/adversarial case: `test_trotter.py:627-644` is the family's
  adversarial test — it exists because `evolve` once coerced parameters
  silently, letting the `apply_trotter` no-op return the **unevolved** state as
  the result. The `test_qsvt.py:101-104` guard is the second: it asserts that a
  plausible-but-wrong sign convention does *not* reproduce the data.
- Reference results: as tabulated. Nothing was recomputed here.
- Evidence status per claim:
  - the contracts, formulas, register geometry, error messages, and validation
    order stated in this record: **`derived`** from the cited source at commit
    `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`;
  - every tolerance and oracle in the table above: **`derived`** — cited
    committed assertions, **not** measurements. No test or example was executed
    in this session;
  - product-formula asymptotic scaling for `evolve`: **`derived`** from
    `docs/sphinx/guide/trotter.rst`, with the fitted-slope study
    (`test_trotter.py:334-363`, slope `== -order` within `abs=0.4`) noted as
    evidence for `apply_trotter`, the same formula `evolve` runs, rather than for
    `evolve` itself;
  - CUDA-Q/package version compatibility for this family: **`unverified`**;
  - foreign-encoding support, wrong-width ket behavior, hardware-target failure
    mode, and unnormalized-input behavior: **`unverified`**;
  - any statement about wall-clock time, memory in bytes, or hardware behavior:
    **absent by policy** — none is made.

### Validation gaps to declare, not fill

No test covers: `good_subspace`'s `ValueError` path; any of these helpers with a
foreign `BlockEncoding` implementation; `action` or `transform` with a
mismatched ket; an unnormalized input state; `evolve` against the fitted-slope
convergence study (that study drives `apply_trotter` directly); any fp32
target; or any hardware target.

## Evaluation coverage

Declared coverage: `simulation-analysis-hardware-boundary` in
`evals/evals.json` tests the simulator-only execution boundary, unnormalized
good-subspace representation, absence of a shot protocol, and invalid
substitution for hardware-shaped Trotter kernels. It has not been run with or
without the skill, so no uplift is claimed.

Additional coverage opportunities include:

- Positive selection/application: a request to obtain `(H/alpha)|psi>` or a
  QSVT-transformed state for a small Hamiltonian while developing on a
  simulator — answered from "Scientific contract", "Inputs", and "Outputs",
  with the simulation-only boundary stated in the answer rather than omitted.
- Convention or misconception: (a) the returned good-subspace block is
  **unnormalized** and its squared norm is the success probability; (b) the
  QSVT response uses the **plain** scaled eigenvalue with no caller-side
  negation; (c) `transform` output under `convention="qsp"` still carries the
  `exp(i * sum(phases))` global phase; (d) `evolve` includes the identity phase
  by default while the circuit primitives cannot.
- Capability composition: `sim_utils.evolve` may **not** be substituted for a
  hardware-shaped evolution, and a `transform` result is not `exp(-iHt)` from
  one call — answered from "Capabilities and composition" and "Composite
  protocol".
- Invalid/unsupported boundary: a request to run these helpers on a QPU target,
  to get a success probability or shot count from them, to quote a T-count or
  depth for `action`/`transform`, or to pass a foreign encoding — answered from
  "Accuracy and limitations" and "Resources", including the refusal to invent
  the unverified failure modes.
- Negative activation: a general CUDA-Q simulator-installation or
  target-selection question routes to `cudaq-guide`, not here
  (`SKILL.md` ownership boundary); requests for Trotter, QSVT, or qubitization
  primitive contracts route to their populated family records rather than this
  simulation-only companion.

## External alignment

- Literature conventions: postselecting the flagged block of a block encoding
  to realize `H/alpha` and reading `norm**2` as a success probability is the
  standard LCU/qubitization presentation; the QSVT reading of a polynomial
  response `p(lambda/alpha)` on the encoded spectrum is standard QSP/QSVT.
  Suzuki-Trotter orders 1, 2, and 4 with the Forest-Ruth splitting are standard
  product formulas. This record adds nothing to that literature; it records
  only what the helpers compute.
- External package translations: `qsp`-convention phases (the QSPPACK
  convention, `diag(e^{i phi}, e^{-i phi})`) are **doubled** into the library's
  projector convention `diag(e^{i phi}, 1)`, and the executed circuit differs
  from the `qsp` signal model by `exp(i * sum(phases))`. `PhaseSequence` keeps
  `phases` raw and refuses a conflicting re-tag rather than silently
  reinterpreting them. QSPPACK itself is optional and test/example-only, never
  a runtime dependency. One further trap recorded in
  `docs/sphinx/conventions.rst:134-137`: QSPPACK's native `W_x` rotation has
  *imaginary* off-diagonals while the circuit's response step is real, so when
  a QSPPACK prediction and a `transform` result seem to disagree,
  `reference_response` is the arbiter — not either side's intuition.
- Known semantic differences: qubit 0 is the **least** significant statevector
  bit and Pauli-word position equals qubit index, both reversed relative to
  common left-to-right tensor-product notation (`conventions.md`); one
  qubitization walk step block-encodes **`-H/alpha`**, with the sign folded into
  the circuits, so the QSVT response and the moment conventions require no
  caller-side negation. Comparing a `transform` or `action` result against an
  external implementation requires these translations first.

---

# Optional schema: Representation Record

## The postselected good-subspace amplitude block

Included because three helpers in this family produce this object and a library
function outside the family consumes it. It is **not** a public Python type,
and no capability ID is minted for it.

- Object name and canonical symbol: the good-subspace amplitude block,
  `(<0|_anc (x) I)|Psi>`; written `good` in the docs and examples.
- Public type or structural form, and source path: a NumPy
  `NDArray[np.complex128]` of length `2**num_system`, returned by
  `python/cudaq_algorithms/sim_utils.py:50` (`good_subspace`) and passed through
  unchanged by `action` (`:59`) and `transform` (`:74`). There is no named type.
- Mathematical meaning: the system-register amplitudes conditioned on all
  ancilla/signal qubits being `|0>`, i.e. the image of the encoded operator or
  polynomial response applied to the input state, before renormalization.
- Shape, layout, ordering, dtype, and units: one-dimensional, length
  `2**num_system`, little-endian with qubit 0 least significant, `complex128`
  irrespective of the simulator's own precision, unitless.
- Normalization, sign, and phase convention: **unnormalized.** `norm**2` is the
  postselection success probability for a normalized input. Signs come from the
  encoding's signed terms; a `qsp`-convention `transform` result additionally
  carries the global phase `exp(i * sum(phases))`.
- Required mathematical properties (applicability preconditions): the producing
  circuit must allocate the system register first and the ancilla/signal
  register after it, and must flag its good subspace on the ancilla all-zero
  state. Neither condition is checked at runtime.
- Producers (>=2 required to justify this record): `good_subspace`, `action`,
  and `transform` (`python/cudaq_algorithms/sim_utils.py:34-74`), plus
  hand-written slicing in tests and examples
  (`tests/python/test_block_encoding_protocol.py:116`,
  `docs/sphinx/conventions.rst:102-108`).
- Consumers: `recover_real_time_evolution`
  (`python/cudaq_algorithms/qsvt.py:304-323`), which consumes a **pair** of such
  blocks — a cosine and a sine sequence — removes each sequence's global phase,
  and returns `2 * (cos.real + 1j * sin.imag)`, valid **only for real
  Hamiltonians and real input states**; and the examples, which renormalize,
  compute success probabilities, or fit a classical constant
  (`pauli_lcu_demo.py:45-47`, `02_hamiltonian_simulation.py:106-113`,
  `07_matrix_inversion_qsvt.py:190-206`).
- Invariants preserved across the boundary: length `2**num_system`; `complex128`
  dtype; no renormalization; an independent copy, so a consumer may mutate it
  safely.
- Observable symptom of a misinterpretation: amplitudes that look bit-reversed
  (qubit-order mismatch); a state with the right direction but the wrong length
  (missing renormalization, or `norm**2` mistaken for a normalization factor); a
  uniformly rotated state (an unremoved `qsp` global phase); or a
  plausible-but-wrong vector when the producing circuit allocated its ancillas
  before the system register.
- Unsupported or ambiguous forms: a block from a circuit with a different
  register order (silently wrong, not rejected); a density matrix or mixed state
  (absent from the module); `evolve`'s return value, which is a **full**
  statevector and not a member of this representation at all.
- Source paths, tests, docs, and last verification: as in "Identity and
  provenance" above; commit `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).

---

# Optional schema: Capability Record

`Deferred:` not created. `architecture.md` extracts a capability record only
when multiple independent producers **and** consumers demonstrate a reusable
boundary. The good-subspace block has one library consumer, and the contracts
this family requires are now owned by [block-encoding.md](block-encoding.md),
[qsvt.md](qsvt.md), and [trotter.md](trotter.md). Trigger for revisiting: a
second independent consumer of the good-subspace block inside the package.
