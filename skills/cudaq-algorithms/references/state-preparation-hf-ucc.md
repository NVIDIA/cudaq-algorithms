# Hartree-Fock reference plus fixed-parameter UCC preparation

Status: draft. Operation + object: **prepare** a **quantum state**.

This is a **concrete primitive record**: it instantiates each applicable
canonical Primitive-Record heading of `assets/primitive-record-template.md`
exactly once, for one contract — a Hartree-Fock reference occupation optionally
followed by a UCC product at amplitudes the caller already knows. The shared
seam it plugs into (kernel representation, unitary capability, consumer table,
common boundaries) is [state-preparation.md](state-preparation.md);
cross-cutting layout, ownership, and validation conventions are
`conventions.md`. Neither is repeated here.

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | from `cudaq_algorithms.stateprep`: `hartree_fock_ucc_kernel`, `make_hartree_fock_occupation`, `validate_hartree_fock_occupation`, `get_fixed_parameter_ucc_pauli_lists`, `validate_fixed_parameter_ucc`, `estimate_hartree_fock_resources`, `estimate_hartree_fock_occupation_resources`, `estimate_fixed_parameter_ucc_resources`. Related **device** kernels in the same namespace, none of which is this seam: `hartree_fock`, `hartree_fock_occupation`, `fixed_parameter_ucc`, `uccsd` |
| Source paths | `python/cudaq_algorithms/stateprep/_hartree_fock.py`; `python/cudaq_algorithms/stateprep/_kernels.py` for the emitted gates; `python/cudaq_algorithms/stateprep/_pools.py` for operator pools |
| Authoritative tests | `tests/python/test_stateprep_hf_ucc.py`, `tests/python/test_stateprep_kernels.py`, `tests/python/test_operator_pools.py`, `tests/python/test_stateprep.py`, and `tests/python/test_state_prep_injection.py` for the seam |
| Authoritative documentation | `docs/sphinx/guide/state_prep.rst`, `docs/sphinx/conventions.rst`; example `docs/sphinx/examples/python/05_state_prep_and_injection.py` |
| Package/CUDA-Q versions verified | **unverified.** The declared environment is Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`), but no test was executed for this record |
| Commit/date last verified | `61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03) |
| Lifecycle | draft |
| Replacement and migration notes | none; no symbol here is marked deprecated at the cited commit |

**Evidence rule.** Every test named below is *cited repository evidence* — a
committed assertion read at that commit, labeled `derived` — never a fresh
measurement in this session.

## Classification

| Dimension | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **prepare** + **quantum state**. Register ownership is deliberately not part of the identity; the seam's source-derived behavior is in the family Capability Record |
| Kind | quantum operation (kernel factory) with host classical validation, plus three resource estimators covered under "Resources" instead of receiving their own records. `Deferred:` a single-token template value for this compound kind, until a second family needs the same compound |
| Routine role | computational — one distinct, independently usable task. Role follows problem completeness, not execution location, so host-side validation and flattening are computational too |
| Abstraction level | leaf operation. The two-stage circuit (reference determinant, then UCC product) is fixed inside this one contract |
| Parameterization | construction-time: amplitudes, words, coefficients, and register width are baked in when the kernel is minted. The runtime-parameterized device kernels named above are **not** this seam |
| Execution layers | host validation and flattening, kernel factory, device kernel |
| Input representations | Hartree-Fock occupation; grouped fixed-parameter UCC Pauli words and coefficients; a flat amplitude list |
| Output representations | the one-argument `(qubits: cudaq.qview)` preparation kernel of the family Representation Record |
| Domain | `quantum-chemistry` — a tag, not a parallel taxonomy; the capability it provides is domain-independent |
| Required dependencies | `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`); the host path uses `cudaq` only. A simulator or hardware target is needed to *run* an emitted kernel, not to build one |
| Optional dependencies | none |
| Exactness | exact for the emitted circuit contract, subject to host input validation and floating-point synthesis of the rotation angles |
| Uncertainty | deterministic |
| Method | a fixed-parameter ansatz product. `ansatz` is outside the provisional `direct \| variational \| heuristic` vocabulary of `architecture.md`, so it is a provisional extension here |

The last three rows are provisional metadata: record the value, never route on
it.

## Scientific contract

- Purpose: emit a one-argument device kernel that prepares a Hartree-Fock
  reference determinant and then applies a UCC product at caller-supplied
  amplitudes, on a register in `|0...0>`.

Notation, all `derived` from `_hartree_fock.py`:

| Symbol | Meaning |
| --- | --- |
| `num_qubits` | the register width in spin orbitals |
| `G`, `g` | the operator pool is an ordered list of `G` operators; `g = 0, ..., G-1` runs over them in **pool order** |
| `n_g`, `j` | group `g` contributes `n_g` Pauli terms indexed `j = 1, ..., n_g` in the order they appear inside `pauli_words[g]` |
| `P_gj` | the Pauli word `pauli_words[g][j]`: a string over `IXYZ` of length `num_qubits` whose character at position `k` acts on qubit `k` (`conventions.md`), or an equivalent `cudaq.pauli_word` |
| `c_gj` | the real coefficient `coefficients[g][j]` |
| `theta_g` | the single real amplitude `parameters[g]` |

**Argument grouping.** `pauli_words` and `coefficients` are lists of lists — one
group per pool operator, in pool order — and `parameters` is flat, holding
**exactly one amplitude per group**. `parameters[g]` pairs with `pauli_words[g]`
and `coefficients[g]`; the factory requires
`len(parameters) == len(pauli_words) == len(coefficients)` and
`len(pauli_words[g]) == len(coefficients[g])` for every `g`
(`_hartree_fock.py:196-212`, `:283-287`).

**Mathematical definition.** The emitted kernel first prepares the reference
determinant `|HF>` by applying `X` to each occupied spin orbital
(`_kernels.py:47-56`), then applies

```text
prod_{g=0}^{G-1} prod_{j=1}^{n_g} exp(+i * theta_g * c_gj * P_gj)
```

strictly in the listed order — groups in pool order, terms in list order within
a group — so the prepared state is that ordered product applied to `|HF>`. Each
factor is emitted as one `exp_pauli(theta_g * c_gj, qubits, P_gj)` call
(`_hartree_fock.py:43-49`, `:281-287`), and `exp_pauli(angle, qubits, P)`
implements `exp(+i * angle * P)`, as stated by the dense reference at
`tests/python/test_stateprep_kernels.py:105-112`. The exponent sign is `+i`;
there is **no** factor of `1/2` and no `theta/2` anywhere in this contract.

**Consequence for caller amplitudes.** Amplitudes expressed in a different UCC
convention — for example `exp(theta * (T - T^dagger))`, or the
`exp(-i * (theta / 2) * c * P)` form the `uccsd` device kernel realizes (see
"Accuracy and limitations") — prepare a different state here, and nothing
raises. Convert before calling.

**Scope of the statement.** It is `derived` from the cited source lines and
pinned by `test_fixed_parameter_ucc_matches_dense_pool_exponential`
(`tests/python/test_stateprep_hf_ucc.py:239-264`), which compares the kernel
against `expm(1.0j * theta * generator)` accumulated in pool order at
`(num_qubits, num_electrons, spin)` of `(4,2,0)`, `(6,3,1)`, and `(8,4,2)`. It
says nothing about whether the factors commute, and no source or test in the
repository establishes that a reordering, a resummation, or any other UCC form
is equivalent to this product.

- Why and when to use: when the target is a Hartree-Fock reference determinant,
  optionally followed by a UCC product at amplitudes the caller already knows.
- When not to use: when the target is a single Slater determinant from an
  orbital-coefficient matrix — that is
  [state-preparation-givens.md](state-preparation-givens.md), a different
  contract, not a variant of this one; when the amplitudes still have to be
  chosen or optimized (a consumer workflow, out of scope); or when a
  multi-argument device kernel is wanted, a different representation.
- Approximation controls: none. The circuit is exact for the data supplied.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | `hartree_fock_ucc_kernel(num_qubits, parameters, pauli_words, coefficients, *, num_electrons=None, spin=0, occupied_orbitals=None)` (`_hartree_fock.py:240-247`); `validate_fixed_parameter_ucc` and `validate_hartree_fock_occupation` check hand-built inputs |
| Shapes/ranks | `parameters` is a flat list of `G` real amplitudes; `pauli_words` and `coefficients` are `G` groups as defined above; `occupied_orbitals`, when given, is a list of distinct spin-orbital indices below `num_qubits` |
| Dtypes/domains | `parameters` and `coefficients` are real, since they become `exp_pauli` angles; words are `str` over `IXYZ` or `cudaq.pauli_word`; counts and indices must be non-negative integers and are validated without coercion |
| Units | dimensionless amplitudes; the products `theta_g * c_gj` are angles in radians as consumed by `exp_pauli` |
| Ordering/layout | pool order across groups, list order within a group; interleaved spin orbitals; `spin` is `2 * S_z`, so 4 electrons at `spin = 2` in 8 qubits occupy `{0, 1, 2, 4}`, not `{0, 1, 2, 3}` (`_hartree_fock.py:81-122`; cross-checked against `get_uccsd_excitations` at `test_stateprep_hf_ucc.py:119-135`) |
| Normalization | none required of the inputs |
| Required mathematical properties | each `P_gj` spans the full register width |

Validation and rejection behavior: `validate_fixed_parameter_ucc` runs first,
then the occupation guards; every rejection is a `ValueError`
(`_hartree_fock.py:94-118`, `:196-221`, `:258-279`).

| # | Rejected condition |
| --- | --- |
| 1 | unequal outer lengths of `parameters`, `pauli_words`, `coefficients` |
| 2 | a group whose word count differs from its coefficient count |
| 3 | a `str` word whose width is not `num_qubits`, or containing a character outside `IXYZ`. A `cudaq.pauli_word` exposes no accessor and is trusted (`assumed`) |
| 4 | not exactly one of `num_electrons` / `occupied_orbitals` — `ValueError("provide exactly one of num_electrons or occupied_orbitals")` |
| 5 | `spin != 0` supplied together with `occupied_orbitals`; encode an open-shell reference in `occupied_orbitals` directly |
| 6 | an occupied index that is non-integral or negative (validated before any integer coercion, so `2.5` is rejected rather than truncated to `2`), `>= num_qubits`, or duplicated |
| 7 | from `make_hartree_fock_occupation`, which runs inside the factory: `num_electrons > num_qubits`; odd `num_qubits` when `spin > 0`; `spin > num_electrons`; **`(num_electrons - spin)` odd when `spin > 0`**; an alpha count exceeding the spatial-orbital count `num_qubits // 2` |

The parity guard in row 7 is the easiest of these to miss, because `spin` reads
like a multiplicity: it is `2 * S_z`, so `(num_electrons - spin)` must be even.
The source states that without the guard the beta floor
`(num_electrons - spin) // 2` would silently realize `spin + 1` — `(8, 4, 1)`
would return the same occupation as `(8, 4, 2)` — and the rejection is asserted
at `test_stateprep_hf_ucc.py:149-150` (`_hartree_fock.py:104-111`).

### Building the grouped inputs

`get_fixed_parameter_ucc_pauli_lists(operator_pool, num_qubits,
coefficient_tolerance=1.0e-12)` is the packaged route from any operator pool to
the grouped form above (`_hartree_fock.py:161-193`; asserted at
`test_stateprep_hf_ucc.py:158-195`):

- it returns `(pauli_words, coefficients)` with **one group per pool operator,
  in pool order**, so group `g` corresponds to `operator_pool[g]` and pairs with
  `parameters[g]`;
- words are built at full `num_qubits` width through
  `term.get_pauli_word(num_qubits)`, so identity padding is explicit;
- a term with `|Re(c)| <= coefficient_tolerance` is **silently dropped** — it
  would only add an identity rotation — so a group may be shorter than its
  operator's term count, and may even be empty;
- a term with `|Im(c)| > coefficient_tolerance` is **rejected** with
  `ValueError`, because `exp_pauli` angles are real;
- a negative `coefficient_tolerance` is rejected.

```python
from cudaq_algorithms import stateprep

pool = stateprep.make_uccsd_operator_pool(num_qubits, num_electrons, spin)
words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(pool, num_qubits)
# thetas: exactly one amplitude per pool operator, in pool order
prep = stateprep.hartree_fock_ucc_kernel(num_qubits, thetas, words, coeffs,
                                         num_electrons=num_electrons,
                                         spin=spin)   # (qubits: cudaq.qview)
```

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | a compiled CUDA-Q kernel whose only parameter is `(qubits: cudaq.qview)` |
| Mathematical meaning | applied to an all-zero register of the baked-in width, the kernel produces the state defined under "Scientific contract" |
| Shape/register geometry | width is fixed at factory time to `num_qubits`; no ancilla, signal, or control register is allocated |
| Normalization, sign, and phase | normalized, and following the `+i` exponent convention above. The four factory shapes — reference plus product, reference only, product only, and an empty `pass` kernel when there is neither an occupation nor a term — differ in content, not in convention (`_hartree_fock.py:289-320`) |
| Observable or measurement interpretation | none; the kernel prepares a state, returns nothing, and performs no measurement |
| Error/status information | none at the device boundary — no return value, no status channel, which is why all validation is host-side (`conventions.md`) |

## Capabilities and composition

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`
- Direction: provides
- Owning family record: [state-preparation.md](state-preparation.md), whose
  Capability Record carries the invariants, the four-module consumer table, and
  the register-ownership scope limit. Match the capability ID, the
  `(qubits: cudaq.qview)` boundary representation, the exact register width, and
  the layout conventions before injecting; support for a consumer that record
  omits must be checked against its source, never inferred.
- Requires: nothing. This provider consumes no other family's capability.

## Composite protocol

Not applicable: `Abstraction level` is `leaf operation`; the template omits it.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no approximation is introduced; deviations come from floating-point synthesis of the rotation angles, and no error bound is stated in source |
| Precision sensitivity | the emitted circuit is precision-agnostic; observed agreement with a dense reference depends on the active simulator precision (see "Validation") |
| Unsupported inputs | the seven rejected condition groups tabulated above |
| Known implementation limitation 1 | the device kernel `fixed_parameter_ucc` requires the register to already hold a reference determinant; applied to an all-zero register the source calls the result "physically meaningless" (`_kernels.py:576-579`). `hartree_fock_ucc_kernel` is the packaged form that guarantees the ordering |
| Known implementation limitation 2 | the `hf_only` factory shape emits only `X` gates at fixed occupation indices (`_hartree_fock.py:300-306`), so a width mismatch that leaves those indices in range has no stated detection mechanism. Per `conventions.md` a silent wrong-state path must be recorded rather than smoothed over, so it is flagged as a **candidate** limitation: whether such a launch succeeds with a wrong-width determinant, fails, or no-ops is `unverified`, and no source or test covers it |
| Unsupported versus unverified | the shared boundaries — controlled, adjoint, measurement-assisted, dirty input register, width-mismatch failure detail, foreign consumer injection, and global phase under control — are stated once with their labels under "Shared unsupported and unverified boundaries" in [state-preparation.md](state-preparation.md). This provider adds none of its own |

### Parameterization boundary and non-interchangeability

A parameterized kernel *is* a state-preparation primitive: the operation is
`parameters + register -> prepared state`. Choosing or optimizing the parameters
is a consumer workflow, outside this library and this skill. But parameterization
is not interchangeability:

- `uccsd(qubits, thetas, num_electrons, spin)` and
  `fixed_parameter_ucc(qubits, thetas, words, coeffs)` are multi-argument device
  kernels, so **neither is directly injectable** as `state_prep`. Only this
  factory, the Givens factory, or an explicitly caller-written one-argument
  wrapper kernel is. `uccsd` performs no input validation, so `thetas` must hold
  exactly `get_num_uccsd_parameters(num_qubits, num_electrons, spin)` entries in
  the `get_uccsd_excitations` order.
- Substituting `make_uccsd_operator_pool` plus a group-consuming kernel for
  `uccsd` at the same `thetas` does not reproduce the same state. Three distinct
  committed assertions establish the parts of that statement, and they are
  **not** the same test or the same cases. In the table, `scale` is the
  multiplier `s` in the dense reference `exp(i * s * theta * c * P)` used at
  `tests/python/test_stateprep_kernels.py:105-112`, with `theta` the
  per-excitation amplitude, `c` the real pool coefficient of a term, and `P` its
  Pauli word.

| Claim | Cited assertion | Cases |
| --- | --- | --- |
| the group-consuming `fixed_parameter_ucc` path realizes `scale = +1` against the pool generators | `test_fixed_parameter_ucc_matches_dense_pool_exponential`, `tests/python/test_stateprep_hf_ucc.py:239-264` | `(4,2,0)`, `(6,3,1)`, `(8,4,2)` |
| the other group-consuming kernels also realize `scale = +1` | `tests/python/test_stateprep_kernels.py:211-255` | `uccgsd` at `num_qubits` 4, 6; `upccgsd` at 4, 8; `ceo` at `num_orbitals` 2, 3, 4 |
| the `uccsd` CNOT-ladder circuit instead comes out as `exp(-i * (theta / 2) * c * P)`, i.e. `scale = -1/2`, and negates theta for the double-excitation index patterns `(p < q and r > s)` and `(p > q and r < s)` while the pool operators carry no such sign | `test_uccsd_kernel_matches_dense_exponential` with `_uccsd_circuit_signs`, `tests/python/test_stateprep_kernels.py:139-182` | `(4,2,0)`, `(6,3,1)`, `(8,4,0)`, `(10,5,1)`, `(8,4,2)` |

`fixed_parameter_ucc` is **not** exercised in `test_stateprep_kernels.py`; its
evidence is the `test_stateprep_hf_ucc.py` case in the first row. That is tested
evidence of **non-interchangeability for the listed cases only**: do not promote
it into a general amplitude-conversion rule, and do not supply substitution code
that silently changes the prepared state.

**Parameterized-UCC scope decision.** A runtime-parameterized UCC construction
stays in scope as a state-preparation primitive (`SKILL.md`) and receives no
separate record here, because at the cited commit every packaged
runtime-parameterized form — `uccsd`, `fixed_parameter_ucc` — is a
multi-argument device kernel that does not meet this seam's representation.
Revisit if a one-argument runtime-parameterized factory appears.

## Resources

`estimate_hartree_fock_resources`, `estimate_hartree_fock_occupation_resources`,
and `estimate_fixed_parameter_ucc_resources` return the frozen dataclasses
`HartreeFockResourceEstimate` and `FixedParameterUccResourceEstimate`.

Interpretation, with attribution rather than a blanket claim: these three
docstrings carry **no** bounding wording of their own (`_hartree_fock.py:141`,
`:148`, `:226`). The "counts of logical operations before transpilation" reading
comes from `docs/sphinx/guide/state_prep.rst` (resource-estimate paragraphs), as
`conventions.md` records — unlike `GivensResourceEstimate`, whose own docstring
states "decomposition-independent upper bounds". Under either source these
quantities are not transpiled-gate figures, not post-synthesis depth, not
runtime, and not memory. No composition rule across estimators is stated
(`assumed` absent), and no timing or scaling evidence exists in this family.

`Deferred:` the per-quantity resource contract (metric, unit, abstraction level,
execution assumptions, exact/bounded/estimated status, controlling parameters,
confidence, composition rule), until the resource-metric vocabulary is agreed.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracles | two, both committed in the repository and cited rather than executed: a dense pool exponential built with NumPy/SciPy `expm` in pool order (`tests/python/test_stateprep_hf_ucc.py:65-76`, applied at `:239-264`), and an independent fermionic-generator bijection (`tests/python/test_operator_pools.py:172-231`). The seam itself is covered by `tests/python/test_state_prep_injection.py` — injected kernel versus `cudaq.State`-fed twin agreement, zero-argument sampleability, a no-op preparation on an all-zero register, and non-cross-contamination of two kernels minted by one factory |
| Invariants | the open-shell Hartree-Fock occupation equals the determinant implied by `get_uccsd_excitations` (`test_stateprep_hf_ucc.py:125-135`) |
| Representative cases | the `(num_qubits, num_electrons, spin)` triples in the non-interchangeability table, and `(8, 4, spin=2)` occupying `{0, 1, 2, 4}` |
| Predeclared tolerances | they differ by suite, so read the suite. `test_stateprep_hf_ucc.py:59-63` and `test_stateprep_kernels.py:127-131` select `1e-12` (fp64) or `5e-5` (fp32) from the active simulator precision, gating on `np.dtype(cudaq.complex()) == np.complex64`. `tests/python/test_state_prep_injection.py` does **not** gate on precision: it hard-codes `atol=1e-12` for the statevector comparisons and uses `abs=1e-10` / `atol=1e-10` for the `Walk.moment` / `Walk.moments` path, and `tests/python/conftest.py` honors `CUDAQ_DEFAULT_SIMULATOR` with a `qpp-cpu` fallback rather than forcing fp64. So do not tell a caller that "the repository's tolerances adapt to simulator precision" — that holds for the provider suites, not for the injection suite |
| Expected failure/adversarial cases | the factory rejections — seven `pytest.raises` blocks over six distinct messages (`test_stateprep_hf_ucc.py:335-379`) — and the occupation guards, including the `(num_electrons - spin)` parity rejection (`test_stateprep_hf_ucc.py:140-150`) |
| Reference results | none; each oracle is constructed inside the cited test |
| Evidence status per claim | `derived` from the cited source or committed test assertion, unless labeled `assumed` or `unverified` in place. Nothing here is `measured` |

## Evaluation coverage

| Case id in `evals/evals.json` | Sections that support it |
| --- | --- |
| `state-preparation-provider-selection` | Scientific contract (the HF-plus-UCC target); Inputs (grouped arguments, ordering, rejection table); Outputs; Identity and provenance (unverified versions) |
| `state-preparation-ucc-parameterization-boundary` | Scientific contract (grouping, `exp(+i theta c P)`, no factor of `1/2`); Accuracy and limitations → Parameterization boundary and non-interchangeability |

The other three `state-preparation-*` cases are answered from
[state-preparation.md](state-preparation.md). No case has been run with or
without the skill, so this is declared coverage, not validated coverage;
`evals/EVAL.md` owns the procedure.

## External alignment

- Literature conventions: no literature citation appears in `_hartree_fock.py`.
  The UCC form realized here is fixed by the source and the cited test, not by a
  referenced paper.
- External package translations: `Deferred:` until a task needs one and it can
  be checked against that package's own documentation. Nothing in the
  repository translates these inputs to an external chemistry package.
- Known semantic differences: inside this repository, the `uccsd` device
  kernel's amplitude convention differs from the group-consuming kernels exactly
  as tabulated under "Accuracy and limitations". No external-package difference
  is established.
