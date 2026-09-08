# Suzuki-Trotter product-formula evolution

Status: draft. Operation + object: **evolve** a **quantum state** under a
**Pauli-sum Hamiltonian**.

This is one **family record** covering the five contracts of
`python/cudaq_algorithms/trotter.py` plus its simulation-only driver. It
instantiates each canonical Primitive-Record heading of
`assets/primitive-record-template.md` **once at family level**, with the
per-contract material in visibly separate subsections beneath it. Headings are
never renamed or dropped; where one does not apply it carries a one-line
`Deferred:` or not-applicable statement.

The state-preparation seam this family consumes is owned by
[state-preparation.md](state-preparation.md) and is not restated here.
Cross-cutting qubit ordering, Pauli-word, host-validation, and
resource-abstraction rules are owned by `conventions.md`. Evidence labels are
defined in `validation.md`.

## The five contracts in this family

| # | Contract | Public symbols |
| --- | --- | --- |
| **A** | Host extraction and planning of a Pauli sum into ordered real terms, an identity coefficient, and a register width | `trotter.make_trotter_terms`, `trotter.Trotter.__init__` and its read-only properties, `trotter.TrotterOrdering` |
| **B** | The product-formula device kernel, composable inside a caller's own kernel | `trotter.apply_trotter` |
| **C** | Kernel factories returning a ready circuit, with or without injected state preparation | `trotter.Trotter.kernel`, `trotter.Trotter.state_kernel` |
| **D** | Logical-operation resource estimate for a chosen `(steps, order)` | `trotter.estimate_trotter_resources`, `trotter.Trotter.resources`, `trotter.TrotterResourceEstimate` |
| **E** | Simulation-only statevector evolution driver, identity phase reintroduced | `sim_utils.evolve` |

These are five independently selectable contracts, not five spellings of one:
A is a host transform that raises, B is device code that is a **silent no-op**
on invalid input, C is a factory that revalidates and raises, D returns numbers
and no circuit, and E requires a statevector simulator. Orders 1, 2, and 4 are
**not** separate contracts — `order` is a runtime argument of one formula
family, so they share every heading below and are separated only where their
mathematics, resources, or error behavior differ.

---

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths:
  - from `cudaq_algorithms` (package root, `__init__.py:42`) — `Trotter`,
    `TrotterOrdering`, `TrotterResourceEstimate` **only**;
  - from `cudaq_algorithms.trotter` (module namespace, `trotter.py:532-543`) —
    additionally `apply_trotter`, `make_trotter_terms`,
    `estimate_trotter_resources`, `FIRST_ORDER_TROTTER`,
    `SECOND_ORDER_TROTTER`, `FOURTH_ORDER_TROTTER`, `HamiltonianLike`;
  - from `cudaq_algorithms.sim_utils` — `evolve`, `state_from`.
  The split is deliberate: `__init__.py:44-47` records that the composable
  device kernels keep their module namespaces because their names are too
  generic to export from the root. `from cudaq_algorithms import apply_trotter`
  is therefore **not** an available import path.
- Source paths: `python/cudaq_algorithms/trotter.py`;
  `python/cudaq_algorithms/sim_utils.py` (contract E);
  `python/cudaq_algorithms/common_kernels.py` for the three shared host
  helpers `_real_coefficient` (line 60), `_term_qubit_extent` (line 42), and
  `state_from` (line 25).
- Authoritative tests: `tests/python/test_trotter.py` (the whole family);
  `tests/python/dense_references.py` for the shared dense oracle;
  `tests/python/test_examples.py` (runs the three Trotter-using examples as
  self-verifying scripts); `tests/python/test_pauli_lcu.py:273-277` for the
  cross-primitive string-rejection parity claim.
- Authoritative documentation: `docs/sphinx/guide/trotter.rst`;
  `docs/sphinx/conventions.rst`; `docs/sphinx/api/python_api.rst:41-45`;
  `docs/sphinx/guide/state_prep.rst` for the injection seam;
  `docs/sphinx/guide/qubitization_qsvt.rst` for the contrasting route.
  Examples: `docs/sphinx/examples/python/trotter_chemistry.py`,
  `02_hamiltonian_simulation.py`, `08_quantum_phase_estimation.py`.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml:11,17`) and
  CUDA-Q pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af`
  (`.cudaq_version`). No test or example was executed for this record, and no
  version was confirmed by execution.
- Commit/date last verified: source read at commit
  `61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (committed 2026-09-03), read
  2026-09-08.
- Lifecycle: draft.
- Replacement and migration notes: none. No symbol in this family is marked
  deprecated at the cited commit.
- **Evidence rule for this family.** Every test and example named below is
  *cited repository evidence* — a committed assertion read at the cited commit,
  labeled `derived` — never a fresh measurement. No runtime, speedup, or
  hardware figure appears anywhere in this record, because none was measured.

---

## Classification

Family-level values, with per-contract deviations in the table that follows.

- Operation + mathematical object (primary identity): **evolve** + **quantum
  state under a Pauli-sum Hamiltonian**. The object acted on is the register
  state; the object parameterizing the operation is the Hamiltonian.
- Domain: domain-independent. Every packaged path consumes an abstract Pauli
  sum. The chemistry framing of `trotter_chemistry.py` and
  `08_quantum_phase_estimation.py` is application evidence, not part of the
  identity.
- Required dependencies: `cudaq >= 0.15.0, < 0.16` (`pyproject.toml:17`).
  Contracts A–D need **only** `cudaq` — `trotter.rst:7` states the module
  "Requires only the `cudaq` Python package", and `trotter.py` imports no
  NumPy. Contract E imports NumPy locally (`sim_utils.py:94`) and requires a
  statevector **simulator** target, because it calls `cudaq.get_state`.
- Optional dependencies: none.

| | A extraction/planning | B `apply_trotter` | C kernel factories | D resource estimate | E `sim_utils.evolve` |
| --- | --- | --- | --- | --- | --- |
| Kind | classical transformation | quantum operation (device kernel) | quantum operation (kernel factory) | resource estimator | simulation-only analysis |
| Routine role | computational | computational | driver | computational | driver |
| Abstraction level | leaf operation | leaf operation | composite protocol | leaf operation | composite protocol |
| Parameterization | construction-time | runtime (all six arguments) | construction-time terms, per-call `time`/`steps`/`order` | per-call | per-call |
| Execution layer | host preprocessing and validation | device kernel | kernel factory (host) emitting device code | host | simulation-only host driver |
| Input representations | `HamiltonianLike` (four accepted forms) | flattened `list[float]` + `list[cudaq.pauli_word]` + a live `cudaq.qview` | the constructed `Trotter`, plus an optional preparation kernel or `cudaq.State` | flattened terms, or the `Trotter` object | the `Trotter` object plus a 1-D statevector |
| Output representations | parallel `(coefficients, words)`, identity coefficient, width | in-place unitary on the register | a compiled CUDA-Q kernel | `TrotterResourceEstimate` | a NumPy `complex128` statevector |

Provisional metadata — record the value, do not route on it:

- Exactness: **approximate**, and this is the family's defining property.
  Approximation controls are `order` and `steps` (below). Two secondary,
  easily-missed approximations also live here: `coefficient_tolerance`
  truncates the Hamiltonian itself, and the omitted identity phase makes the
  circuit exact only up to a global phase. Contract D is exact-or-proxy per
  quantity (see "Resources"); contract A is exact apart from the tolerance
  filter.
- Uncertainty: deterministic. Every path is a fixed circuit or a fixed host
  computation. There is no randomized product formula (no qDRIFT, no random
  term ordering) in this family, so no sampling distribution over circuits
  exists to report. Shot noise from `cudaq.sample` belongs to the caller's
  measurement, not to this contract.
- Method: direct.

---

## Scientific contract

### Purpose

Realize an approximation to \(e^{-iHt}\) for a Hamiltonian given as a real
linear combination of Pauli strings, using **no ancilla qubits and no block
encoding**, by splitting the exponential into a product of individually
implementable Pauli rotations.

### Mathematical definition

Write the input Hamiltonian as

\[ H = c_I I + H', \qquad H' = \sum_{i=0}^{n-1} c_i P_i, \]

where each \(P_i\) is a non-identity Pauli word of the common width
\(N =\) `num_qubits`, each \(c_i \in \mathbb{R}\), and \(c_I\) is the summed
identity coefficient. Hermiticity of \(H\) is exactly the requirement that all
\(c_i\) and \(c_I\) are real, which contract A enforces
(`common_kernels.py:60-70`).

**The circuit implements \(H'\) only.** With \(dt = t/\)`steps`, the emitted
unitary is \(\big[S_{\text{order}}(dt)\big]^{\text{steps}}\) where, applying
factors to the state in the order the kernel emits them:

- **order 1** (`trotter.py:87-88`), one forward sweep:
  \[ S_1(dt) = \prod_{i=n-1}^{0} e^{-i\,c_i\,dt\,P_i} \]
- **order 2**, the default (`trotter.py:114-119`), a symmetric (Strang)
  half-angle forward sweep followed by a half-angle reverse sweep:
  \[ S_2(dt) = \Big(\prod_{i=0}^{n-1} e^{-i c_i dt P_i/2}\Big)\Big(\prod_{i=n-1}^{0} e^{-i c_i dt P_i/2}\Big) \]
  \(S_2\) is palindromic, hence time-symmetric: \(S_2(dt)^{-1} = S_2(-dt)\).
- **order 4** (`trotter.py:90-113`), the triple-jump composition of three
  symmetric second-order sub-steps at time fractions \(w_1, w_0, w_1\):
  \[ S_4(dt) = S_2(w_1 dt)\,S_2(w_0 dt)\,S_2(w_1 dt), \qquad w_1 = \frac{1}{2 - 2^{1/3}},\quad w_0 = 1 - 2w_1 \]
  The source calls these the Forest-Ruth weights and precomputes them as the
  private constants `_FOREST_RUTH_W1 = 1.3512071919596578` and
  `_FOREST_RUTH_W0 = -1.7024143839193153` (`trotter.py:46-54`), because a
  CUDA-Q kernel cannot evaluate a host-only cube root.

Each factor is emitted as `exp_pauli(-θ, qubits, word)` with the appropriate
\(θ\). The realized per-factor operator is \(e^{-i\,θ\,P}\); see "Sign and
phase conventions" below for how that is established.

**Two exact invariants of all three orders**, both `derived`:

1. **Angle consistency.** The total angle accumulated per term per step is
   \(c_i\,dt\) for every order: order 1 applies it once; order 2 applies
   \(c_i dt/2\) twice; order 4 applies \(c_i w_1 dt/2\) twice,
   \(c_i w_0 dt/2\) twice, and \(c_i w_1 dt/2\) twice, summing to
   \((2w_1 + w_0)\,c_i\,dt = c_i\,dt\). All three therefore agree with
   \(e^{-iH't}\) at first order in \(dt\).
2. **The order-4 condition holds exactly.** Because
   \(w_0 = 1 - 2w_1 = -2^{1/3} w_1\), we have \(w_0^3 = -2 w_1^3\) and hence
   \(2w_1^3 + w_0^3 = 0\). Together with the consistency condition
   \(2w_1 + w_0 = 1\) and the time-symmetry of \(S_2\), this is precisely the
   pair of conditions making the triple-jump composition fourth order. The
   record therefore states the fourth-order property as `derived` from the
   constants in source, not merely as a documented claim.

**Exactness on a commuting Hamiltonian.** If all \(P_i\) mutually commute, the
splitting is exact for every `order` and every `steps`, since
\(e^{-i(A+B)} = e^{-iA}e^{-iB}\) when \([A,B] = 0\). This is the family's
sharpest analytical validation case and is the one the repository asserts to
better than `1e-9` (`test_trotter.py:374-393`).

### Approximation controls

| Control | Where | Effect | Evidence |
| --- | --- | --- | --- |
| `order` ∈ {1, 2, 4} | per kernel/estimate call | total-error scaling \(O(t^{p+1}/\text{steps}^{p})\) for order \(p\) | documented `trotter.rst:56-64`; scaling asserted `test_trotter.py:334-371`; fourth-order condition `derived` above |
| `steps` ≥ 1, integral | per kernel/estimate call | \(dt = t/\)`steps`; error falls as \(\text{steps}^{-p}\), cost rises linearly | `trotter.py:256-261`, `test_trotter.py:334-371` |
| `coefficient_tolerance` (default `1e-12`) | construction only | **truncates the Hamiltonian**: terms with \(|c| <\) tolerance, and all exactly-zero terms, are dropped | `trotter.py:207-226`, `test_trotter.py:588-607` |
| term `ordering` | construction only | changes which product formula is realized, hence the error constant, but not the asymptotic order and not the resource counts | `trotter.py:236-245, 282-295` |

`coefficient_tolerance` is documented as a filter that avoids zero-angle
rotations and inflated resource estimates (`trotter.py:201-206`). Treat it as a
**Hamiltonian-truncation control** as well: raising it changes \(H'\), and that
systematic error is *not* covered by the order-\(p\) Trotter scaling above. Two
consequences are `derived` from `trotter.py:216-226` and covered by no test
(`unverified` as tested):

- the tolerance filter runs **before** identity accumulation, so an identity
  term with \(|c_I| <\) tolerance is dropped from `identity_coefficient` too
  rather than being reported;
- if the filter removes *every* non-identity term, `num_terms` becomes 0 and
  the factories silently return an identity circuit — a requested evolution
  becomes a no-op with no warning.

### Sign and phase conventions

- The realized per-factor operator is \(e^{-i θ P}\) for a call
  `exp_pauli(-θ, ...)`. This is `derived` from two independent places: the
  repository's dense oracle applies \(\cos α - i \sin α\, P = e^{-iαP}\) with
  \(α = c_i dt\) and the kernel is asserted to match it
  (`test_trotter.py:52-54, 66-92, 312-331`), and the locally installed CUDA-Q
  documents `exp_pauli(theta, qubits, pauliWord)` as `exp(i theta P)`. The
  second source is **weaker evidence and is not the basis of this record**: the
  locally installed distribution is `cuda-quantum-cu12 0.14.2`, which is
  *outside* the declared `>= 0.15.0, < 0.16` range. Rest the sign contract on
  the repository's own dense oracle, which is version-independent.
- Qubit ordering and Pauli-word position follow `conventions.md`: `word[0]`
  acts on qubit 0, and qubit 0 is the least significant statevector index.
  `tests/python/dense_references.py:8-26` builds its reference in that same
  little-endian order.
- **The identity phase \(e^{-i c_I t}\) is omitted from every circuit** and
  cannot be added to one, because a global phase on the evolved register is not
  a realizable gate. `identity_coefficient` is reported so the caller can
  account for it (`trotter.py:20-27`, `trotter.rst:119-127`). It is
  unobservable for a single unconditioned evolution and **observable** whenever
  the evolution is placed under control or interfered — see "Composite
  protocol" for the worked controlled case.

### Why and when to use

- The target is \(e^{-iHt}\) for a Pauli sum and **no ancilla qubits are
  available or wanted**; the register width is exactly `num_qubits`.
- A tunable accuracy/cost knob is wanted with no polynomial-approximation
  machinery, phase-factor generation, or block encoding.
- The Hamiltonian is Hermitian with real Pauli coefficients and is not
  time-dependent.
- The caller wants a directly sampleable, hardware-shaped circuit
  (`Trotter.kernel`), or a composable device kernel to place inside their own
  kernel (`apply_trotter`).

### When not to use

- **Exactness within a polynomial approximation matters more than ancilla
  count.** `docs/sphinx/guide/trotter.rst:9-12` and
  `qubitization_qsvt.rst:8-12` frame Trotter and qubitization/QSVT as the two
  time-evolution routes in the library; QSVT is "exact within a polynomial
  approximation, at the cost of an ancilla register"
  (`PauliLCU.num_ancilla` is at least 1, `pauli_lcu.py:473`). That is a
  different contract with a different error model, not a variant of this one,
  and this record makes no accuracy or cost comparison between the two —
  `02_hamiltonian_simulation.py` contrasts them, and its numbers were not
  reproduced here.
- **A controlled or adjoint evolution factory is needed.** `Walk` and `QSVT`
  ship controlled variants; `Trotter` ships none (see "Accuracy and
  limitations").
- **The Hamiltonian is time-dependent, non-Hermitian, or has complex Pauli
  coefficients.** All are rejected or absent.
- **The Hamiltonian is not available as a Pauli sum.** Route it through the
  package's preprocessing first (`preprocessing.rst:10-13`); see
  [chemistry-bridges.md](chemistry-bridges.md) and
  [fermion-transforms.md](fermion-transforms.md).
- **Amplitudes or parameters still have to be chosen or optimized.** That is a
  consumer workflow, out of this skill's scope (`SKILL.md`).

---

## Inputs

### A — Hamiltonian extraction and planning

```python
from cudaq_algorithms import trotter

evolution = trotter.Trotter(
    hamiltonian,                                   # HamiltonianLike
    ordering=trotter.TrotterOrdering.PRESERVE_INPUT,
    coefficient_tolerance=1e-12)                   # keyword-only
```

- Arguments: `Trotter(hamiltonian, ordering=PRESERVE_INPUT, *,
  coefficient_tolerance=1e-12)` (`trotter.py:339-347`). The flattening step is
  independently callable as `make_trotter_terms(hamiltonian,
  coefficient_tolerance=1e-12)` (`trotter.py:189-192`), which returns
  `(coefficients, words, identity_coefficient, num_qubits)` and performs no
  ordering.
- Accepted `HamiltonianLike` forms — **four**, with materially different
  padding behavior (`trotter.py:56-60, 127-186`):

| Form | Width rule | Test |
| --- | --- | --- |
| `cudaq.SpinOperator` | **padded** to the widest term's extent | `test_trotter.py:553-557` |
| a single `cudaq.SpinOperatorTerm` | wrapped in `cudaq.SpinOperator`, then as above | `test_trotter.py:155-161` |
| `{word: coefficient}` mapping | **not padded**; all words must already share one length | `test_trotter.py:164-172` |
| iterable of `(coefficient, word)` pairs, generators included | as above | `test_trotter.py:164-172, 546-550` |

  Spin-operator width uses `_term_qubit_extent` (largest targeted index + 1),
  **not** CUDA-Q's `qubit_count`, which undercounts gapped or off-zero targets
  (`common_kernels.py:42-57`).
- Shapes/ranks: `words` are equal-length strings over `IXYZ` of length
  `num_qubits`; `coefficients` is a parallel real sequence.
- Dtypes/domains: coefficients are coerced with `complex()` and must be real to
  within a **fixed, package-wide** imaginary tolerance of `1e-10`
  (`common_kernels.py:60-70`). `coefficient_tolerance` does **not** loosen this
  (`trotter.py:204-206`). A coefficient such as `0.5 + 1e-11j` is accepted and
  its imaginary part silently discarded — the same tolerance `PauliLCU` uses,
  which `test_trotter.py:560-567` pins as a cross-primitive parity claim.
- Units: `coefficients` carry the Hamiltonian's energy unit; `time` is
  conjugate to it, so \(c\cdot t\) must be dimensionless. Nothing in source
  fixes or checks a unit; `08_quantum_phase_estimation.py` uses Hartree and
  inverse Hartree. `tolerance` values are in coefficient units.
- Ordering/layout: `conventions.md` — `word[0]` on qubit 0, qubit 0 least
  significant. When the Pauli sum came from a fermionic transform, the
  interleaved alpha-even / beta-odd spin-orbital layout applies and is a caller
  obligation with no detection here.
- Normalization: none. Unlike `PauliLCU`, this family computes **no**
  \(\alpha = \sum|c_i|\) and performs no rescaling; coefficients enter the
  rotation angles as given.
- Required mathematical properties: Hermiticity, expressed as real
  coefficients; at least one term after extraction; at least one qubit.
- Validation and rejection behavior — host-side, at construction, **eight**
  conditions (`trotter.py:172-186, 207-209, 272-279`; all asserted at
  `test_trotter.py:175-189, 570-585`):

| # | Rejected condition | Exception and match |
| --- | --- | --- |
| 1 | input is not a spin operator/term, mapping, or iterable of pairs | `TypeError`, "must be a cudaq spin operator or term…" |
| 2 | input is `str`, `bytes`, `bytearray`, or `memoryview` | `TypeError`, same message (explicitly excluded from the iterable branch) |
| 3 | a coefficient with `|imag| > 1e-10` | `ValueError`, "complex Hamiltonian coefficients are not supported" |
| 4 | no terms at all (`{}`, `[]`) | `ValueError`, "hamiltonian has no terms" |
| 5 | words of differing length (mapping/pairs) | `ValueError`, "all Pauli words must have the same length" |
| 6 | a character outside `IXYZ` | `ValueError`, "unsupported Pauli word: …" |
| 7 | zero register width (e.g. `{"": 0.5}`) | `ValueError`, "hamiltonian must act on at least one qubit" |
| 8 | an unrecognized `ordering` value | `ValueError`, "unsupported Trotter ordering: …" |

- `TrotterOrdering` accepts either the enum or its string value
  (`"preserve_input"`, `"coefficient_magnitude_descending"`),
  `trotter.py:272-279`. `COEFFICIENT_MAGNITUDE_DESCENDING` sorts by
  `abs(coefficient)` descending with Python's stable `sorted`, so ties keep
  extraction order (`trotter.py:288-292`, `derived`).

### B — `apply_trotter` device-kernel arguments

```python
@cudaq.kernel
def my_kernel(coeffs: list[float], words: list[cudaq.pauli_word],
              t: float, steps: int, order: int):
    q = cudaq.qvector(4)
    # ... state preparation ...
    trotter.apply_trotter(coeffs, words, t, steps, order, q)
```

- Arguments, in exact positional order (`trotter.py:68-69`):
  `apply_trotter(coefficients: list[float], words: list[cudaq.pauli_word],
  time: float, steps: int, order: int, qubits: cudaq.qview)`. The register is
  **last**, which is the opposite of the one-argument state-preparation seam
  and of `common_kernels` kernels that take the register first.
- Shapes: `len(coefficients)` must equal `len(words)`; every word must have the
  width of the register handed in.
- `evolution.coefficients` and `evolution.words` supply the flattened arrays
  (`trotter.rst:100-116`). `words` are plain `str`, which are accepted directly
  as `list[cudaq.pauli_word]` **kernel arguments**; only words *captured* by a
  kernel need explicit `cudaq.pauli_word` conversion, which `Trotter.kernel`
  performs internally (`trotter.py:193-199, 390-394`). Both spellings appear in
  the repository: `trotter_chemistry.py:121-122` passes `str`,
  `08_quantum_phase_estimation.py:150` converts explicitly.
- **Validation and rejection behavior: there is none, by construction.** A
  device kernel has no error channel, and a `return` inside a
  `@cudaq.kernel` is silently ignored by the compiler
  (NVIDIA/cuda-quantum#4845), so the whole body is one positively-guarded
  `if` block (`trotter.py:77-83`). Invalid runtime input is a **silent no-op**
  that leaves the register unchanged: `steps <= 0`, a
  coefficient/word length mismatch, or `order` outside {1, 2, 4}. Asserted at
  `test_trotter.py:291-309`. This is the `conventions.md` rule "validation
  happens on the host, before device kernels" in its most consequential form
  here: a caller who reaches `apply_trotter` directly and passes `order=3`
  receives an unevolved state that looks like a result. Validate on the host
  first, or use contract C, which raises.

### C — kernel-factory arguments

- `Trotter.kernel(time, steps=1, order=SECOND_ORDER_TROTTER,
  state_prep=None)` (`trotter.py:396-400`).
- `Trotter.state_kernel(time, steps=1, order=SECOND_ORDER_TROTTER)`
  (`trotter.py:456-459`) — **no `state_prep` parameter**.
- Shared validation via `_prepared_args` (`trotter.py:390-394`), all
  `ValueError`, all asserted (`test_trotter.py:245-254, 610-616`):
  `time` must be finite (`nan`/`inf` rejected); `steps` must be a positive
  integer with **no silent truncation** (`2.9` is rejected, stated as parity
  with `Walk.kernel(power=...)`); `order` must be in {1, 2, 4}.
  `time` may be negative or zero — only finiteness is checked
  (`trotter.py:264-269`, `derived`); no test exercises either
  (`unverified` as tested).
- `state_prep` is a compiled CUDA-Q kernel with the single parameter
  `(qubits: cudaq.qview)`. Its width must equal `num_qubits` exactly and this
  is **not verifiable at factory time** (`trotter.py:406-415`,
  `state_prep.rst:238-244`). Its parameter is annotated `Any | None` here while
  the encoding consumers use `Kernel | None`, an asymmetry
  [state-preparation.md](state-preparation.md) already reports.
- `state_kernel`'s `cudaq.State` argument must have dimension
  `2**num_qubits`. `sim_utils.evolve` checks this; **direct callers of
  `state_kernel` are responsible for it themselves, and the identity-only
  variant cannot detect a mismatch** (`trotter.py:466-470`).

### D — resource-estimate arguments

- `Trotter.resources(steps, order)` (`trotter.py:490-494`): both are
  **required positional-or-keyword parameters with no defaults**, deliberately,
  "so the estimate can never silently describe a different circuit than the
  kernel you built". Calling `resources()` raises `TypeError`
  (`test_trotter.py:619-624`).
- `estimate_trotter_resources(coefficients, words, steps, order,
  identity_coefficient=0.0)` (`trotter.py:503-508`) accepts flattened terms
  directly and additionally raises `ValueError("coefficients and words must
  have equal length")` (`trotter.py:515-516`). It revalidates `steps` and
  `order` with the same helpers as the factories.

### E — `sim_utils.evolve` arguments

- `evolve(evolution, ket, time, steps=1, order=SECOND_ORDER_TROTTER,
  include_identity_phase=True)` (`sim_utils.py:77-82`).
- `ket` must be 1-D with exactly `2**num_qubits` entries; both failures raise
  `ValueError` with distinct messages ("1-D", "dimension"), asserted at
  `test_trotter.py:627-644`. That test is a recorded regression: `evolve`
  previously coerced silently and the device guard turned invalid parameters
  into a no-op, returning the *unevolved* state as if it were the result.
- `ket` is cast to the active target's precision by `state_from`
  (`common_kernels.py:25-33`), because fp32 simulators reject `complex128`
  input.

---

## Outputs

### A — extraction and planning outputs

- `make_trotter_terms` returns the 4-tuple `(coefficients: list[float],
  words: list[str], identity_coefficient: float, num_qubits: int)`. `words` are
  padded plain strings, deliberately readable and comparable
  (`trotter.py:193-199`).
- The `Trotter` object exposes read-only copies (`trotter.py:355-383`):
  `coefficients` and `words` (parallel, in **application order**),
  `identity_coefficient`, `num_qubits`, `num_terms`, `ordering`.
  `coefficients` and `words` return fresh `list` copies, so mutating them
  cannot corrupt the object.
- `__repr__` reports terms, qubits, and the identity coefficient to 6
  significant figures (`trotter.py:385-388`).
- Mathematical meaning: `coefficients`/`words` are the retained non-identity
  \(c_i, P_i\); `identity_coefficient` is \(c_I\), explicitly documented as
  "not realizable in circuit"; `num_qubits` is the register width the
  evolution kernels will allocate.
- Normalization, sign, and phase: coefficients are passed through unscaled and
  sign-preserved. No \(\alpha\) normalization exists in this family.
- Error/status information: exceptions only; there is no status object.

### B — `apply_trotter` output

- Returns nothing. It applies the in-place unitary
  \(\big[S_{\text{order}}(dt)\big]^{\text{steps}}\) to the `cudaq.qview` it is
  handed, and allocates nothing: **no ancilla, no signal, and no control
  register**.
- Register geometry: exactly the register passed in; the kernel never resizes
  or reallocates.
- Observable or measurement interpretation: none. It performs no measurement
  and no reset, so it is safely composable inside a larger coherent circuit.
- Error/status information: none — the silent-no-op semantics above.

### C — kernel-factory outputs

`Trotter.kernel` and `Trotter.state_kernel` return a compiled CUDA-Q kernel,
typed `Any` because "CUDA-Q exposes no stable public Python type for compiled
kernel objects" (`trotter.py:402-404`). **Six distinct emitted circuits**
(`trotter.py:420-488`):

| Factory call | Emitted signature | Body |
| --- | --- | --- |
| `kernel(...)`, terms present | zero-argument | allocate `num_qubits` in \|0…0⟩, then `apply_trotter` |
| `kernel(..., state_prep=p)`, terms present | zero-argument | allocate, run `p(qubits)`, then `apply_trotter` |
| `kernel(...)`, no terms | zero-argument | allocate only — the identity circuit |
| `kernel(..., state_prep=p)`, no terms | zero-argument | allocate, run `p(qubits)` — exactly the preparation |
| `state_kernel(...)`, terms present | `(state: cudaq.State)` | allocate from `state`, then `apply_trotter` |
| `state_kernel(...)`, no terms | `(state: cudaq.State)` | allocate from `state` only |

- **`Trotter.kernel` is zero-argument in both modes.** This is a real
  asymmetry against the encoding consumers, which return a
  `(state: cudaq.State)` kernel when `state_prep` is omitted; `Trotter`'s
  `cudaq.State` twin is the separate `state_kernel`
  ([state-preparation.md](state-preparation.md), Capability Record invariant
  3). The no-preparation `kernel` evolves \|0…0⟩.
- The two no-terms branches exist as a workaround: a captured **empty** list
  cannot be marshaled across the kernel boundary
  (NVIDIA/cuda-quantum#4847, `trotter.py:438-442`).
  `test_trotter.py:514-538` executes both branches specifically so a refactor
  cannot silently reintroduce the capture.
- Mathematical meaning: the emitted state is
  \(\big[S_{\text{order}}(t/\text{steps})\big]^{\text{steps}}\,|\psi_0\rangle\)
  where \(|\psi_0\rangle\) is \|0…0⟩, the prepared state, or the supplied
  `cudaq.State`. It approximates \(e^{-iH't}|\psi_0\rangle\), i.e.
  \(e^{-iHt}|\psi_0\rangle\) **up to the omitted global factor**
  \(e^{-i c_I t}\).
- Register geometry: exactly `num_qubits` qubits. Ancilla-free.
- Observable or measurement interpretation: none is emitted. The zero-argument
  forms are directly `cudaq.sample`-able (`test_trotter.py:455-459`); the
  caller owns any `mz`/`cudaq.observe`.
- Error/status information: exceptions at factory time only. Once a kernel is
  emitted, its arguments have already been validated — which is exactly why
  contract C is the safe path and contract B is the sharp one.

### D — `TrotterResourceEstimate`

A frozen dataclass (`trotter.py:308-321`) with six fields: `num_terms`,
`steps`, `order`, `pauli_rotations`, `estimated_cx_count`,
`identity_coefficient`. Meanings, status, and limits are in "Resources".
`identity_coefficient` is carried here purely so a resource summary and its
phase bookkeeping travel together; it is not a resource quantity.

### E — `sim_utils.evolve`

Returns a NumPy `complex128` statevector of length `2**num_qubits`. Unlike
every circuit path, it **reintroduces the identity phase by default**
(`include_identity_phase=True`), multiplying by
\(e^{-i c_I t}\) on the host (`sim_utils.py:106-108`), so the result
approximates the full \(e^{-iHt}|\psi\rangle\) with **no phase alignment
needed**. With `include_identity_phase=False` it matches the circuit
convention. `test_trotter.py:476-495` asserts both directions — the
phase-included result is compared to the exact evolution **without** phase
alignment and the phase-excluded one is asserted to be *far* from it
(`> 0.1`), which is what makes the two conventions distinguishable rather than
cosmetic.

---

## Capabilities and composition

### Required — unitary state preparation

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`
- Direction: **requires** (contract C, `Trotter.kernel`, only)
- Owning family record: [state-preparation.md](state-preparation.md). Its
  Capability Record owns the five invariants, the four-module consumer table
  (in which `Trotter.kernel` is one row), the register-ownership scope limit,
  and the shared unsupported/unverified boundaries. None of that is restated
  here.
- Boundary representation: a compiled kernel whose only parameter is
  `(qubits: cudaq.qview)`.
- Semantic invariants this consumer imposes: the factory allocates the register
  itself, so the preparation receives a **fresh \|0…0⟩ register of width
  `num_qubits`**, runs **before** any evolution gate, and must act only on that
  register. No ancilla, signal, or control register exists at that point,
  because this family allocates none at all.
- Shape/register geometry: exact width match with `num_qubits`, unenforced at
  factory time — a mismatch fails at launch, and the failure detail is
  `unverified`.
- Normalization, sign, phase, and ordering: the family imposes none beyond
  `conventions.md`. A preparation correct only up to a global phase (the Givens
  provider's contract) is fine for an unconditioned evolution and **not** fine
  once the composed circuit is placed under control or interfered — the same
  caveat as the identity phase.
- Convention requirements: the qubit-ordering and Pauli-word conventions of
  `conventions.md`, plus the interleaved alpha-even / beta-odd spin-orbital
  layout whenever the Hamiltonian and the preparation both derive from a
  fermionic transform. The preparation and the Hamiltonian must have been built
  at the same width and the same layout; nothing checks that they were.
- Host/device/simulation boundary: the preparation must be pure device code
  with no simulation-only dependency; the packaged providers satisfy this.
- Unsupported conditions: a multi-argument device kernel is a **different
  representation**, not a variant — wrapping one in a one-argument kernel is
  the caller's explicit act, not an equivalence (`conventions.md`).
  `state_kernel` accepts no `state_prep` at all.

### Not declared — a general time-evolution capability

**No provided capability ID is declared by this record.** `architecture.md`
permits extracting a capability record only when multiple independent producers
or consumers demonstrate a reusable boundary, and the taxonomy design record
classifies a generic `time-evolution` capability as a **candidate**, with
first- and higher-order Trotterization as concrete providers.

Source at the cited commit does not demonstrate the boundary:

- the library's two time-evolution routes have **incompatible interfaces** —
  `apply_trotter` takes flattened terms plus a bare register and needs no
  ancilla, while the QSVT route needs a block encoding, an ancilla register of
  width ≥ 1, a phase sequence, and, for real-time evolution, host-side
  recombination of two runs through
  `qsvt.recover_real_time_evolution` (`qsvt.py:304-323`);
- their normalization conventions differ: QSVT evolves \(H/\alpha\), this
  family evolves \(H'\) unscaled;
- their error models differ (product-formula splitting versus polynomial
  approximation of \(e^{-ix}\));
- no shared Python protocol, ABC, base class, or common signature exists, and
  no consumer in the package is written against both.

The single consumer written against this family's shape,
`sim_utils.evolve`, accepts a `Trotter` object concretely
(`sim_utils.py:77`). One concrete consumer is not a demonstrated boundary.

Promotion criteria, and the decision still required: (a) at least one consumer
written against an evolution *boundary* rather than against `Trotter`;
(b) a resolved normalization and register-ownership convention shared with the
QSVT route; (c) a decision on whether controlled evolution and the identity
phase are part of the boundary or of each provider — currently they are handled
per call site; (d) an explicit team decision recorded with an owner. None holds
at the cited commit; no promotion is proposed, and no capability ID should be
cited for this family until one is.

`apply_trotter`'s own six-argument device signature is a **one-off interface**
that stays inside this record by the same rule.

---

## Composite protocol

Applies to contracts **C** and **E** (Abstraction level `composite protocol`).
Contracts A, B, and D are leaf operations and this heading does not apply to
them.

### C — `Trotter.kernel` as prepare-then-evolve

- Required lower-level capabilities:
  `cudaq-algorithms.state-preparation.unitary.v1` (optional at call time —
  omitting it is a legal mode that evolves \|0…0⟩), plus this family's own
  contracts A and B.
- Canonical reference composition: extract and order terms once on the host
  (A) → allocate `num_qubits` in \|0…0⟩ → run `state_prep` on that register →
  apply `apply_trotter` with the validated arguments (B). This is literally
  `trotter.py:430-436`.
- Default recipe and applicability conditions: `order=2`, `steps=1`
  (`trotter.py:398-399`). The symmetric second-order formula is the default
  because it is strictly better than order 1 at twice the rotation count and
  needs no auxiliary constants. Applicability: any accepted Pauli sum. Choose
  `steps` from a **predeclared** accuracy target, not from an observed result
  — the repository has no automatic `steps` selection, and none should be
  invented.
- Materially different alternatives, all present in the repository:
  1. **Compose `apply_trotter` inside your own kernel** — the documented
     escape hatch when preparation *and* measurement must live in one kernel
     (`trotter.rst:99-116`). You then own all validation.
  2. **`state_kernel` + a `cudaq.State`** — loads the input state as data
     instead of preparing it. `test_trotter.py:438-453` pins the two paths
     against each other to `atol=1e-12` on the same prepared state, which is
     the repository's evidence that injection and state-loading agree.
  3. **`sim_utils.evolve`** — contract E, simulation-only, identity phase
     included.
  4. **Controlled evolution, hand-composed** — see below.
- Propagated conventions: qubit ordering, Pauli-word position, and any
  spin-orbital layout inherited from the Hamiltonian's producer; the
  \|0…0⟩-input and exact-width requirements of the preparation seam.
- Propagated errors: all host-side, at factory time. The preparation provider
  has already validated its own inputs when it was minted; this factory
  validates `time`, `steps`, `order`; and the resulting kernel has **no error
  channel**. A width mismatch between preparation and evolution escapes every
  host check and fails at launch (detail `unverified`).
- Propagated resources and how they compose: additively and only within one
  abstraction level. The preparation's own estimator (for the packaged Givens
  and UCC providers) counts *its* logical operations; `TrotterResourceEstimate`
  counts *this* family's. Adding them yields a logical-operation total for the
  composed circuit **only** if both are at that level and neither is a
  transpiled figure — `conventions.md` forbids conflating levels, and no
  source-level composition rule between the two estimators exists (`assumed`
  absent).
- Component substitution: any kernel with the exact `(qubits: cudaq.qview)`
  signature and the right width may be supplied, including a caller-written one
  (`_product_prep`, `test_trotter.py:432-434`;
  `my_prep`, `trotter.rst:84-90`). A substitution must preserve: the
  one-argument signature, the exact width, unitarity with no measurement or
  reset, and no allocation of its own registers. It need **not** be a packaged
  provider.

### Controlled evolution — a hand-composed pattern, not a packaged contract

`Trotter` ships **no** controlled or adjoint factory, unlike `Walk` (which has
`controlled_kernel`, `adjoint_kernel`, `roundtrip_kernel`,
`controlled_roundtrip_kernel`) and `QSVT` (`controlled_kernel`). The repository
does show controlled evolution built by hand, in
`08_quantum_phase_estimation.py:246-269`:

```python
cudaq.control(trotter.apply_trotter, counting[bit],
              coefficients, words, t, steps, order, system)
r1(-t * identity, counting[bit])   # restore the omitted identity phase
```

Two things must be said about it precisely, because they pull in opposite
directions:

- It is **executed evidence in CI** — `tests/python/test_examples.py:47-59`
  runs every example as a subprocess and requires its self-checks to pass, and
  that example asserts each modal energy estimate lands within one phase cell
  of exact diagonalization. So the pattern is exercised, not merely sketched.
  Under this skill's evidence rule the assertion is still cited `derived`
  evidence, because nothing was executed in this session.
- It is nonetheless **not a library contract**: there is no public controlled
  factory, no test in `test_trotter.py` covering `cudaq.control` over
  `apply_trotter`, and the identity-phase restoration is the caller's
  responsibility. The `r1(-t * identity, control)` correction is required, not
  optional — the identity coefficient is a global phase for free evolution and
  a **measurable relative phase** under control
  (`docs/sphinx/examples_rst/getting_started.rst:135-138`). Do not report
  controlled Trotter evolution
  as a packaged capability, and do not omit the phase correction when
  reproducing the pattern.

Adjoint evolution has weaker standing still: `cudaq.adjoint` never appears in
`trotter.py`, and although a negative `time` passes validation and is the
analytic inverse of the same product formula (`derived`, since
\(S_p(-dt)\) is what the formula yields), no test or example exercises it
(`unverified`).

### E — `sim_utils.evolve` as a simulation-only driver

- Required lower-level capabilities: contract C's `state_kernel`, plus
  statevector access. It is **simulation-only** by construction: `sim_utils`
  exists precisely because `cudaq.get_state` does not exist on hardware
  targets (`sim_utils.py:3-11`).
- Canonical reference composition: validate `ket` shape and dimension on the
  host → `state_from(ket)` at the target's precision → `state_kernel(time,
  steps, order)` → `cudaq.get_state` → multiply by \(e^{-i c_I t}\) if
  `include_identity_phase` (`sim_utils.py:94-109`).
- Default recipe and applicability conditions: `steps=1`, `order=2`,
  `include_identity_phase=True`. Applicable only on a statevector simulator,
  and only when a full statevector is an acceptable output.
- Materially different alternatives: contract C plus the caller's own
  `cudaq.get_state`, which then omits the identity phase and both validations.
- Propagated conventions: the circuit's, **except** the phase convention, which
  it deliberately inverts by reintroducing \(e^{-i c_I t}\). This is the one
  place in the family where a result can be compared to \(e^{-iHt}\) directly
  rather than up to a global phase.
- Propagated errors: it raises `ValueError` for a non-1-D or wrong-dimension
  `ket` and delegates the `time`/`steps`/`order` checks to `state_kernel`. Its
  docstring records why: without those checks the device guard silently
  returned an unevolved state.
- Propagated resources: the circuit's are unchanged; simulator memory and time
  are the caller's and are **not** modeled by any estimator in this family.
- Component substitution: none. It accepts a `Trotter` object concretely and
  offers no injection point.

---

## Accuracy and limitations

### Error behavior and bounds

- Documented total-error scaling (`trotter.rst:56-64`): order 1
  \(O(t^2/\text{steps})\), order 2 \(O(t^3/\text{steps}^2)\), order 4
  \(O(t^5/\text{steps}^4)\) — i.e. \(O(t^{p+1}/\text{steps}^p)\), consistent
  with `steps` applications of a local error \(O(dt^{p+1})\).
- The **order** in `steps` is asserted by a predeclared convergence study:
  `test_trotter.py:334-371` fits \(\log(\text{error})\) against
  \(\log(\text{steps})\) over `steps` ∈ {1, 2, 4} and requires the slope within
  `0.4` of \(-p\) for each of \(p\) = 1, 2, 4, plus strict improvement with
  order and fixed thresholds at `steps=4`. That is `derived` cited evidence.
- **No prefactor or commutator-norm bound is stated anywhere in source or
  docs.** There is no \(\sum_{i<j}\|[P_i,P_j]\|\)-style bound, no a-priori
  `steps` selector, and no tightness claim. Treat any specific error *value*
  for a new Hamiltonian as `unverified` until computed; treat the exponent as
  `derived`.
- Term ordering changes the error constant — `COEFFICIENT_MAGNITUDE_DESCENDING`
  is documented only as "a common heuristic for reducing Trotter error"
  (`trotter.py:239-241`). **No test or benchmark in the repository shows that
  it reduces error for any Hamiltonian.** Do not claim it does.
- Truncation by `coefficient_tolerance` adds a systematic error that the
  order-\(p\) scaling does not cover (see "Approximation controls").

### Precision sensitivity

- The emitted circuit is precision-agnostic; observed agreement depends on the
  active simulator precision. `conftest.py` sets the target from
  `CUDAQ_DEFAULT_SIMULATOR`, defaulting to `qpp-cpu` (fp64).
- **The Trotter suite's tolerances are not precision-gated.** Several
  assertions use a fixed `atol=1e-12` (`test_trotter.py:291-309, 438-453,
  498-538`) and others `1e-6`. Unlike `test_stateprep_givens.py`, which selects
  its tolerance from `np.dtype(cudaq.complex())`, nothing here adapts to an
  fp32 target. So these tolerances are implicitly fp64 claims;
  their behavior on an fp32 simulator such as the default `nvidia` target is
  `unverified`. `test_examples.py:15-17` states the matching constraint for
  examples: they run on the fp64 CPU simulator because "fp32 defaults would
  fail their tolerances".
- `state_from` casts input to `cudaq.complex()`, so on an fp32 target a
  `complex128` `ket` is downcast before evolution (`common_kernels.py:25-33`).

### Unsupported inputs

The eight rejected conditions of contract A, plus the three factory
validations of contract C, plus the length check in
`estimate_trotter_resources`. All are host-side and all raise.

### Known implementation limitations

1. **`apply_trotter` is a silent no-op on invalid runtime input** — the single
   most important limitation in this family. It is a documented consequence of
   the compiler ignoring `return` in a kernel (NVIDIA/cuda-quantum#4845), not
   established semantics.
2. **Sweep-boundary rotation merging is documented and deliberately not
   implemented** (`trotter.rst:69-73`): the back-to-back half-rotations at
   sweep and step boundaries of the order-2 and order-4 formulas — about
   \(1/n\) of all rotations, each a CX ladder on hardware — are emitted
   separately. No effect on simulator results; a real cost on hardware, and the
   resource estimate counts the unmerged circuit.
3. **The identity phase cannot be realized in a circuit** and is silently
   absent from every emitted kernel.
4. **Sub-tolerance identity terms vanish from `identity_coefficient`**, and a
   Hamiltonian whose non-identity terms are all filtered out yields a silent
   identity circuit (both `derived`, both untested).
5. **Input-form asymmetry for an identity-only Hamiltonian.** The mapping form
   `{"II": -0.2}` is accepted, giving `num_qubits == 2` and `num_terms == 0`
   (`test_trotter.py:498-538`). An identity-only `cudaq.SpinOperator` instead
   yields extent 0 from `_term_qubit_extent` (which returns 0 for an identity
   term, `common_kernels.py:51-57`) and is therefore **rejected** by a
   `ValueError` from the width checks. Which of the two messages fires depends
   on what `get_pauli_word(0)` returns, which no test pins — the rejection is
   `derived`, the exact message `unverified`.
6. **`PRESERVE_INPUT` on a spin-operator input preserves CUDA-Q's term
   iteration order**, which this package neither specifies nor pins. The
   repository's own spin-operator test compares `sorted(words)`
   (`test_trotter.py:553-557`), i.e. it does not rely on that order. Treat the
   realized term order for a `SpinOperator` input as `unverified`; use a
   mapping, a pair list, or `COEFFICIENT_MAGNITUDE_DESCENDING` when the order
   must be reproducible.
7. **`state_kernel` cannot detect a state-dimension mismatch in its
   identity-only branch**, and does not check dimension in either branch —
   only `sim_utils.evolve` does.
8. **`estimate_trotter_resources` does not validate word contents.** It
   accepts any strings and counts non-`I` characters as weight
   (`trotter.py:298-300`), so a word containing an illegal character yields a
   number rather than an error. Reach it through `Trotter.resources`, whose
   terms were already validated at construction.

### Unsupported versus unverified

Read each label literally — these are four different statements.

| Boundary | Label | Why the label |
| --- | --- | --- |
| Identity phase inside a circuit | **unsupported** | not realizable on the evolved register; source states it cannot be included and reports `identity_coefficient` instead |
| Complex Pauli coefficients / non-Hermitian \(H\) | **unsupported** | `_real_coefficient` raises above `1e-10` imaginary; explicit message |
| Orders other than 1, 2, 4 | **unsupported** | `_validate_order` raises on the host; `apply_trotter` is a no-op on the device |
| Zero, fractional, or negative `steps`; non-finite `time` | **unsupported** | validated and rejected |
| A packaged controlled or adjoint evolution factory | **absent** | no such method exists on `Trotter`; the hand-composed control pattern in example 08 is a caller composition, and no `test_trotter.py` case covers it |
| Randomized product formulas (qDRIFT), higher or arbitrary orders, adaptive `steps` | **absent** | nothing in source; do not present them as available |
| Commuting-set grouping or any ordering beyond the two `TrotterOrdering` values | **absent** | only two strategies exist |
| Time-dependent Hamiltonians | **absent** | no interface takes a time-dependent generator |
| Explicit error bound with a prefactor, and any tightness claim | **absent** | only the asymptotic order is documented |
| Circuit depth, transpiled gate counts, runtime, memory | **absent** | `TrotterResourceEstimate` has no such field; never substitute one of its proxies |
| Negative or zero `time` semantics | **unverified** | accepted by validation and analytically the inverse/identity, but exercised by no test or example |
| Behavior on an fp32 target at the suite's fixed tolerances | **unverified** | tolerances are not precision-gated |
| Injected-preparation width mismatch: error type and message | **unverified** | documented as failing at launch; no source or test states how |
| Dirty (non-\|0…0⟩) register at the preparation seam | **unverified** | shared seam boundary; see [state-preparation.md](state-preparation.md) |
| Realized term order for a `SpinOperator` input under `PRESERVE_INPUT` | **unverified** | depends on unpinned CUDA-Q iteration order |
| Exact `ValueError` message for an identity-only `SpinOperator` | **unverified** | depends on `get_pauli_word(0)` |
| Whether `COEFFICIENT_MAGNITUDE_DESCENDING` reduces error in practice | **unverified** | documented as a heuristic; no repository evidence |
| Any accuracy or cost comparison against the QSVT route | **unverified** | `02_hamiltonian_simulation.py` contrasts them; nothing was reproduced or measured here |

Do not hand-roll a substitute for an `absent` item and present it as library
behavior, and do not upgrade an `unverified` item to `derived` without citing
the source or test that establishes it.

---

## Resources

Contract D is the family's resource contract. It is a **host-side counting
helper, not a measurement**: `trotter.rst` and `state_prep.rst:148-149` both
place these dataclasses in the same category — "frozen dataclasses counting
logical operations before transpilation, not hardware gate counts". The
`conventions.md` rule against conflating abstraction levels governs everything
below.

Let \(n\) = `num_terms`, \(s\) = `steps`, and
\(r(\text{order}) = \{1{:}1,\, 2{:}2,\, 4{:}6\}\) rotations per term per step
(`trotter.py:303-305`), which matches the emitted sweep count exactly: 1 sweep
for order 1, 2 for order 2, and 6 for order 4 (three symmetric sub-steps of two
sweeps each).

| Quantity | `pauli_rotations` | `estimated_cx_count` | register width | `identity_coefficient` |
| --- | --- | --- | --- | --- |
| Formula | \(n \cdot s \cdot r\) | \(s \cdot r \cdot \sum_i 2\max(0,\,w_i - 1)\), \(w_i\) = non-identity weight of \(P_i\) | `num_qubits` | \(c_I\) |
| Metric and unit | count of `exp_pauli` logical rotations (dimensionless) | count of CNOTs (dimensionless) | count of qubits | Hamiltonian coefficient units |
| Abstraction level | **logical operation count, pre-transpilation** | **decomposition proxy** — the docstring's own word (`trotter.py:311-313`): "two CNOTs per additional non-identity Pauli in each rotation" | logical qubits | not a resource |
| Architecture/execution assumptions | none; it counts the emitted loop structure | each weight-\(w\) `exp_pauli` becomes a CX ladder of \(2(w-1)\) CNOTs; **no** connectivity, routing, SWAP insertion, basis-change gates, or transpiler optimization is modeled | no ancilla, signal, or control register is ever allocated by this family | — |
| Status | **exact** for the emitted circuit as written | **estimated** | **exact** | exact (a host sum) |
| Controlling parameters | \(n\), `steps`, `order` | Pauli weights, `steps`, `order` | the Hamiltonian's widest term | the identity terms above tolerance |
| Confidence / limitations | exact only for the **unmerged** circuit; the deferred sweep-boundary merging (~\(1/n\) of rotations) would reduce it. Not a depth, not a runtime | everything above, plus: it is not a transpiled gate count, not post-synthesis depth, not runtime, and not memory | contrast: the QSVT/qubitization route needs `num_ancilla` ≥ 1 (`pauli_lcu.py:473`) | — |
| Composition rule | additive across sequential Trotter kernels over the same term list; **no** source-stated rule for combining with another family's estimator (`assumed` absent) | as above, and only within the decomposition-proxy level | additive only in the sense that a composed circuit still uses `num_qubits` — preparation adds none | additive over identity terms |

Two properties worth stating because they are easy to get wrong:

- **Both counts are independent of term ordering** — each is a sum over all
  retained terms, so `PRESERVE_INPUT` and `COEFFICIENT_MAGNITUDE_DESCENDING`
  give identical estimates. Ordering is an accuracy knob, not a cost knob.
- **Both counts scale linearly in `steps` and in \(r(\text{order})\)**, so
  order 4 costs 6× order 1 and 3× order 2 at equal `steps`. Comparing accuracy
  across orders therefore requires an equal-cost comparison, which the
  repository does not perform; `test_trotter.py:334-371` compares at equal
  `steps`, i.e. at unequal cost. Report that distinction rather than presenting
  the test's order ranking as a cost-normalized result.

Arithmetic checks in the repository, `derived`: three terms of weights
{1, 2, 1} at `steps=3, order=4` give `pauli_rotations = 3·3·6 = 54` and
`estimated_cx_count = 2·3·6 = 36` (`test_trotter.py:225-242`); two terms of
weights {2, 1} at `steps=2, order=2` give `8` and `8`
(`test_trotter.py:257-264`).

`estimate_trotter_resources` is a reusable public helper, but it is not given
its own primitive record: it has no independent scientific semantics beyond the
formulas above and is not used outside this family at the cited commit.
`Deferred:` an independent estimator record, until a second family consumes it
or a resource-metric vocabulary is agreed.

---

## Validation

### Independent oracles

The suite carries **three** oracles, and their independence is the point:

1. **Exact dense diagonalization** — `_exact_evolve`
   (`test_trotter.py:95-103`) builds the dense Pauli-sum matrix with the
   shared `dense_references.dense_matrix` helper, adds \(c_I I\), and applies
   \(e^{-iHt}\) via `numpy.linalg.eigh`. Independent of any product formula:
   it never splits the exponential.
2. **An explicit Pauli-rotation statevector simulator** — `_simulate_trotter`
   (`test_trotter.py:29-92`) applies \(\cos α - i \sin α P\) per factor in
   pure NumPy, reimplementing the three orders independently of CUDA-Q. This
   is a *same-assumptions* oracle for the splitting structure and an
   *independent* one for the compiler and simulator, so it validates the kernel
   against the intended formula, not the formula against physics. Its value is
   catching a compilation or marshaling error; oracle 1 is what catches a wrong
   formula.
3. **The analytical identities in this record** — angle consistency and
   \(2w_1^3 + w_0^3 = 0\) — which are `derived` here from the source constants
   and need no execution at all. These are the strongest evidence in the
   family under the `validation.md` hierarchy, and they are what makes the
   order-4 claim more than a docstring.

The examples add a fourth, weaker layer: `trotter_chemistry.py` and
`02_hamiltonian_simulation.py` self-verify against dense references, and
`test_examples.py` runs them.

### Invariants

- **Exact on a commuting Hamiltonian** for every order and step count —
  all-`Z` terms, error `< 1e-9` (`test_trotter.py:374-393`). The sharpest
  invariant available, because it has an analytical reason.
- **Monotone improvement with order** at fixed `steps`
  (`test_trotter.py:197-217` on the reference alone, `334-371` on the kernel).
- **Order-\(p\) convergence in `steps`**, slope within `0.4` of \(-p\)
  (`test_trotter.py:334-371`).
- **Injection and state-loading agree**: `kernel(state_prep=p)` matches
  `state_kernel` fed the same prepared state to `atol=1e-12`
  (`test_trotter.py:438-453`).
- **Identity-only Hamiltonian is a pure global phase**:
  `evolve(...)` returns \(e^{-i c_I t}\)`ket` to `atol=1e-12`, and returns
  `ket` unchanged with the phase disabled (`test_trotter.py:498-511`).
- **Invalid device input leaves the register in \|0…0⟩** to `atol=1e-12`
  (`test_trotter.py:291-309`) — an invariant that *documents a limitation*
  rather than a correctness property.

### Representative cases

All four input forms including a generator; single-qubit through four-qubit
Hamiltonians with up to ten terms and weight up to four
(`test_trotter.py:396-424`); a chemistry-style 14-term four-qubit Hamiltonian
in `trotter_chemistry.py`; orders 1, 2, 4 at `steps` ∈ {1, 2, 3, 4, 64};
zero-coefficient, sub-tolerance-imaginary, identity-only, empty, zero-width,
and string inputs.

### Predeclared tolerances

`atol=1e-6` for kernel-versus-reference statevector comparisons; `1e-12` for
exactness and agreement claims; `< 1e-9` for the commuting case; `< 1e-3` for
`evolve` at 64 steps and `> 0.1` for its phase-disabled counterpart;
`2.0e-2 / 6.0e-4 / 5.0e-6` at `steps=4` for orders 1/2/4; slope tolerance
`0.4`. These are committed in the test file, i.e. fixed before any result was
observed in this session. As noted above they are **not** precision-gated, so
they are fp64 claims.

Comparisons against \(e^{-iHt}\) must either phase-align (`_phase_align_error`,
`test_trotter.py:127-131`) or reintroduce \(e^{-i c_I t}\); the suite does both
deliberately, and `test_trotter.py:485-487` marks the non-phase-aligned
comparison explicitly. Choose one before judging a result, per
`conventions.md`'s comparison discipline.

### Expected failure and adversarial cases

The eight construction rejections and three factory rejections above; the
device-level silent no-op for `steps=0`, `order=3`, and an empty word list; the
`evolve` dimension and rank failures; `resources()` with no arguments; and both
identity-only kernel branches, executed specifically to prevent a refactor from
reintroducing the empty-capture bug.

### Reference results

None are reproduced here. Every oracle is constructed inside the cited test or
example.

### Evidence status per claim

`derived` from the cited source line, committed test assertion, or the
analytical identities stated above, unless labeled `assumed` or `unverified` in
place. **Nothing in this record is `measured`.** No test, example, or kernel was
executed in this session; no runtime, throughput, or hardware figure appears
anywhere; and the CUDA-Q version range remains `unverified`.

---

## Evaluation coverage

Declared coverage: `trotter-evolution-resource-boundary` in `evals/evals.json`
tests fourth-order product-formula selection, approximation controls,
hardware-shaped versus simulation-only output, identity-phase handling, and
the logical-resource boundary. Nothing here has been run with or without the
skill, so no uplift is claimed.

The following are additional coverage opportunities, not existing case IDs:

| Coverage opportunity | Kind | What it must test, and the sections that support it |
| --- | --- | --- |
| `trotter-order-and-steps-selection` | positive selection | Choosing `order` and `steps` for an accuracy target, and reporting the error *exponent* while refusing to state an error *value* or a prefactor. Scientific contract (orders and controls); Accuracy and limitations (no bound in source) |
| `trotter-identity-phase-under-control` | convention / misconception | A user placing the evolution under control and expecting the library to handle the phase. Scientific contract (sign and phase); Composite protocol (the controlled pattern and its required `r1(-t*c_I)` correction); Outputs (contract E's inverted convention) |
| `trotter-state-prep-composition` | capability composition | Injecting a packaged preparation into `Trotter.kernel`, including the zero-argument asymmetry against the encoding consumers and the unenforced width match. Capabilities and composition; Composite protocol → C; plus [state-preparation.md](state-preparation.md) |
| `trotter-invalid-order-silent-noop` | invalid/unsupported boundary | A user calling `apply_trotter` directly with `order=3` and expecting an exception. Inputs → B; Accuracy and limitations (limitation 1); Unsupported versus unverified |
| `trotter-resource-metric-boundary` | resource claim / negative | A request for transpiled gate counts, depth, or runtime from `TrotterResourceEstimate`, and a cost-normalized accuracy comparison across orders. Resources (both notes); Unsupported versus unverified |
| `trotter-controlled-adjoint-factory-negative` | unsupported boundary | A request for `Trotter.controlled_kernel` / `adjoint_kernel` by analogy with `Walk` and `QSVT`. Composite protocol (controlled pattern); Unsupported versus unverified |
| `negative-generic-trotter-theory` | negative activation | A generic "explain Trotterization / product formulas" question with no reference to this library, which must be answered from general knowledge **without** activating this skill. Supported by no section here — its point is that the record is not loaded |

Each is answerable from this file plus, for the composition case, the one
record it links directly — no third hop. When these are authored, each
assertion must cite a section above unconditionally, and any assertion about
the CUDA-Q version must assert the honest `unverified` status rather than a
value, per `EVAL.md`.

---

## External alignment

- **Literature conventions.** The source names the order-4 weights
  "Forest-Ruth" (`trotter.py:46-52`, `trotter.rst:61-67`) and gives no
  citation. The constant \(w_1 = 1/(2 - 2^{1/3}) = 1.3512071919596578\) is the
  one appearing both in the Forest-Ruth symplectic integrator and in the
  standard Suzuki/Yoshida triple-jump composition of a symmetric second-order
  scheme, and this record's derivation above uses only the triple-jump order
  conditions, which the constants satisfy exactly. The *attribution* is
  therefore `unverified` — no paper was consulted — while the fourth-order
  *property* is `derived`. Order 2 is the symmetric Strang splitting, named as
  such in `trotter.rst:58-60`.
- **No commutator-bound literature is cited or used.** The well-known
  tighter product-formula bounds are not referenced in source and are not
  imported into this record.
- **External package translations.** `Deferred:` until a task needs one and it
  can be checked against that package's own documentation. Nothing in the
  repository translates this family's terms, orderings, or resource fields to
  an external Hamiltonian-simulation package, and the `order` argument's
  meaning must not be assumed to match another library's `order` or
  `trotter_order` without checking that library's own formula.
- **Known semantic differences within this repository**, all `derived` and all
  easy to trip over:
  - `Trotter.kernel` is zero-argument in **both** `state_prep` modes, while
    the encoding consumers switch to a `(state: cudaq.State)` kernel when
    `state_prep` is omitted;
  - `Trotter.kernel`'s `state_prep` is annotated `Any | None`, the encodings'
    `Kernel | None`;
  - this family computes **no** \(\alpha\) normalization, while `PauliLCU`
    normalizes by \(\alpha = \sum|c_i|\) and `QSVT` acts on \(H/\alpha\), so
    an "evolution time" is not directly comparable between the two routes;
  - string-like inputs raise `TypeError` here and in `PauliLCU`, deliberately
    and with matching intent (`test_trotter.py:581-585`,
    `test_pauli_lcu.py:273-277`), but the messages differ ("cudaq spin
    operator or term" versus "SpinOperator");
  - `make_trotter_terms` splits identity terms out and rejects a zero-width
    Hamiltonian, which `trotter.py:145-148` records as deliberate divergences
    from `pauli_lcu._terms_from_input`.
- **Representation record.** `Deferred:` the Pauli-sum Hamiltonian plausibly
  warrants one — it has multiple producers (`fermion`, `chemistry`,
  `double_factorization`) and multiple consumers (`PauliLCU`, this family) —
  but it is a **cross-family** object whose record does not belong inside this
  file. Follow-up trigger: the first record for a producer family, or the
  first task that must reconcile a producer's layout with a consumer's.
