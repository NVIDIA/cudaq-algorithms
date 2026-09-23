# Hartree-Fock reference plus fixed-parameter UCC preparation

Operation + object: **prepare** a **quantum state**.

The factory prepares a Hartree-Fock reference and an optional UCC product
at known amplitudes. The shared [injection contract](injection-contract.md) owns
the one-register signature, consumer table, and common boundaries.

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | from `cudaq_algorithms.stateprep`: `hartree_fock_ucc_kernel`, `make_hartree_fock_occupation`, `validate_hartree_fock_occupation`, `get_fixed_parameter_ucc_pauli_lists`, `validate_fixed_parameter_ucc`. Related **device** kernels are separate contracts routed by the [state-preparation family selector](state-preparation.md) |
| Source paths | `python/cudaq_algorithms/stateprep/_hartree_fock.py`; `python/cudaq_algorithms/stateprep/_kernels.py` for the emitted gates; `python/cudaq_algorithms/stateprep/_pools.py` for operator pools |
| Authoritative tests | `tests/python/test_stateprep_hf_ucc.py`, `tests/python/test_stateprep_kernels.py`, `tests/python/test_operator_pools.py`, `tests/python/test_stateprep.py`, and `tests/python/test_state_prep_injection.py` for the seam |
| Authoritative documentation | `docs/sphinx/guide/state_prep.rst`, `docs/sphinx/conventions.rst`; example `docs/sphinx/examples/python/05_state_prep_and_injection.py` |

Cited repository assertions provide derived evidence until executed in the current task.

## Classification

| Dimension | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **prepare** + **quantum state**. Register ownership is deliberately not part of the identity; the seam's source-derived behavior is in the shared injection Capability Record |
| Kind | quantum operation. The independently selectable host resource helpers are routed from the [state-preparation family selector](state-preparation.md) and are not part of the emitted-kernel kind |
| Routine role | computational — one distinct, independently usable task. Role follows problem completeness, not execution location, so host-side validation and flattening are computational too |
| Abstraction level | leaf operation. The two-stage circuit (reference determinant, then UCC product) is fixed inside this one contract |
| Parameterization | construction-time: amplitudes, words, coefficients, and register width are baked in when the kernel is minted. The runtime-parameterized device kernels named above are **not** this seam |
| Execution layers | host validation and flattening, kernel factory, device kernel |
| Input representations | Hartree-Fock occupation; grouped fixed-parameter UCC Pauli words and coefficients; a flat amplitude list |
| Output representations | the one-argument `(qubits: cudaq.qview)` preparation kernel of the shared injection Representation Record |
| Domain | `quantum-chemistry` — a tag, not a parallel taxonomy; the capability it provides is domain-independent |
| Required dependencies | `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`); the host path uses `cudaq` only. A simulator or hardware target is needed to *run* an emitted kernel, not to build one |
| Optional dependencies | none |
| Exactness | exact for the emitted circuit contract, subject to host input validation and floating-point synthesis of the rotation angles |
| Uncertainty | deterministic |
| Method | a fixed-parameter ansatz product |

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
| `P_gj` | the Pauli word `pauli_words[g][j]`: a string over `IXYZ` of length `num_qubits` whose character at position `k` acts on qubit `k` (`../conventions.md`), or an equivalent `cudaq.pauli_word` |
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
  orbital-coefficient matrix — that is the
  [Givens schedule](state-preparation-givens-schedule.md) followed by the
  [injectable Slater-determinant kernel](state-preparation-slater-determinant-kernel.md),
  different contracts rather than a variant of this one; when the amplitudes
  still have to be chosen or optimized (a consumer workflow, out of scope); or when a
  multi-argument device kernel is wanted — route that different representation
  by exact symbol through
  [state-preparation family selector](state-preparation.md).
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

Validation runs on the host and raises `ValueError` for rejected inputs.
Read the [seven rejection groups and occupation parity guard](hf-ucc-validation.md)
before supplying hand-built inputs.

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
| Error/status information | none at the device boundary — no return value, no status channel, which is why all validation is host-side (`../conventions.md`) |

## Capabilities and composition

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`
- Direction: provides
- Owning family record: [injection-contract.md](injection-contract.md), whose
  Capability Record carries the invariants, the four-module consumer table, and
  the register-ownership scope limit. Match the capability ID, the
  `(qubits: cudaq.qview)` boundary representation, the exact register width, and
  the layout conventions before injecting; support for a consumer that record
  omits must be checked against its source, never inferred.
- Requires: nothing. This provider consumes no other family's capability.

## Composite protocol

Not applicable: `Abstraction level` is `leaf operation`.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no approximation is introduced; deviations come from floating-point synthesis of the rotation angles, and no error bound is stated in source |
| Precision sensitivity | the emitted circuit is precision-agnostic; observed agreement with a dense reference depends on the active simulator precision (see "Validation") |
| Unsupported inputs | the seven rejected condition groups in [HF/UCC validation](hf-ucc-validation.md) |
| Known implementation limitation 1 | the device kernel `fixed_parameter_ucc` requires the register to already hold a reference determinant; applied to an all-zero register the source calls the result "physically meaningless" (`_kernels.py:576-579`). `hartree_fock_ucc_kernel` is the packaged form that guarantees the ordering |
| Known implementation limitation 2 | the `hf_only` factory shape emits only `X` gates at fixed occupation indices (`_hartree_fock.py:300-306`), so a width mismatch that leaves those indices in range has no stated detection mechanism. Per `../conventions.md` a silent wrong-state path must be recorded rather than smoothed over, so it is flagged as a **candidate** limitation: whether such a launch succeeds with a wrong-width determinant, fails, or no-ops is `unverified`, and no source or test covers it |
| Unsupported versus unverified | the shared boundaries — controlled, adjoint, measurement-assisted, dirty input register, width-mismatch behavior, foreign consumer injection, and global phase under control — are stated once with their labels under "Shared unsupported and unverified boundaries" in [injection-contract.md](injection-contract.md). This provider adds none of its own |

### Parameterization boundary

Runtime UCC kernels and this injectable factory have distinct signatures and amplitude
conventions. Read [UCC parameterization](ucc-parameterization.md) before substituting them.

## Resources

This preparation record does not own the three independently selectable
estimators. Route by the exact input object through the
[state-preparation family selector](state-preparation.md) to the
canonical Hartree-Fock, explicit-occupation Hartree-Fock, or fixed-parameter UCC
resource record.

## Validation

Read [HF/UCC validation](hf-ucc-validation.md) for independent oracles,
representative cases, rejection evidence, and suite-specific precision/tolerance behavior.

## External alignment

- Literature conventions: no literature citation appears in `_hartree_fock.py`.
  The UCC form realized here is fixed by the source and the cited test, not by a
  referenced paper.
- External package translations: `Deferred:` until a task needs one and it can
  be checked against that package's own documentation. Nothing in the
  repository translates these inputs to an external chemistry package.
- Known semantic differences: inside this repository, the `uccsd` device
  kernel's amplitude convention differs from the group-consuming kernels exactly
  as tabulated in [UCC parameterization](ucc-parameterization.md). No external-package difference
  is established.
