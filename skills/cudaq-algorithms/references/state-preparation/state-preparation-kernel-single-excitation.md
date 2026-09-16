# UCCSD single-excitation device kernel

Status: draft. Operation + object: **apply** one **Jordan-Wigner UCCSD single
excitation**.

This is one runtime device-kernel contract. It does not prepare a reference
determinant and is not the full UCCSD product documented in
[the UCCSD device-kernel record](state-preparation-kernel-uccsd.md).

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.single_excitation`. The public package initializer imports the symbol, so it is addressable there, but omits it from `cudaq_algorithms.stateprep.__all__` |
| Contract-specific source paths | Device circuit: `python/cudaq_algorithms/stateprep/_kernels.py:59-91`; matching pool generator: `python/cudaq_algorithms/stateprep/_pools.py:150-154`; import surface: `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | **No direct standalone test exists.** Indirect composition evidence is `tests/python/test_stateprep_kernels.py::test_uccsd_kernel_matches_dense_exponential` (`:158-182`), which calls the full `uccsd` kernel and compares its ordered product against a dense oracle |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:24-72` documents the full UCCSD kernel and its host-generated endpoints; no standalone `single_excitation` example is documented |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | None for this record. The declared target is Python `>=3.11` and `cudaq >=0.15.0,<0.16`; execution in that range is unverified |
| Lifecycle | draft |
| Implementation status | documented and implemented; source was inspected during the historical last review; not compiled, directly executed, or directly numerically validated during this record expansion |
| Replacement and migration notes | Not applicable: no deprecation or replacement was identified during the historical last review; check the current public API/source at use time |

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **apply** + **one Jordan-Wigner UCCSD single excitation** |
| Kind | quantum operation |
| Routine role | auxiliary — a lower-level building block of `uccsd` |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | device kernel called from a caller-owned CUDA-Q kernel |
| Input representations | a live `cudaq.qview`, one real amplitude, and an ordered occupied-to-virtual endpoint pair |
| Output representations | in-place unitary state transformation; no host return |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16` as declared by `pyproject.toml` |
| Optional dependencies | Not applicable: the kernel has no optional dependency |
| Exactness | exact for the source-defined two-term Pauli-rotation circuit on valid endpoints, subject to floating-point gate realization |
| Uncertainty | deterministic |
| Method | Jordan-Wigner parity-string rotation synthesized with basis changes, CNOT ladders, and `Rz` rotations |

## Scientific contract

Let `Z_(p,q) = product_(k=p+1)^(q-1) Z_k`, with the empty product equal to
identity, and define

```text
G_1(p,q) = 0.5 * (Y_p Z_(p,q) X_q - X_p Z_(p,q) Y_q).
```

| Field | Contract |
| --- | --- |
| Purpose | Apply one occupied-to-virtual UCCSD single-excitation factor to an already prepared live register |
| Mathematical definition | For valid `0 <= p_occ < q_virt < qubits.size()`, the source circuit realizes `U_1(theta;p,q) = exp(-i * (theta/2) * G_1(p,q))`, equivalently `exp(-0.25i * theta * (Y_p Z_(p,q) X_q - X_p Z_(p,q) Y_q))`. This is the `uccsd` device-kernel amplitude convention pinned indirectly by the dense composition test |
| Why and when to use | Use only as a low-level call inside a caller-owned kernel when applying one endpoint pair produced under the package UCCSD enumeration convention |
| When not to use | Do not use it to prepare Hartree-Fock, as a directly injectable one-register provider, with a reversed/equal pair, or as an amplitude-compatible substitute for grouped `fixed_parameter_ucc` |
| Approximation controls | Not applicable: no approximation parameter exists; `theta` is the physical runtime parameter, not an error tolerance |

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact signature: `single_excitation(qubits: cudaq.qview, theta: float, p_occ: int, q_virt: int)` |
| Shapes/ranks | One live register view and three scalar runtime arguments |
| Dtypes/domains | `theta` is real; `p_occ` and `q_virt` are integer qubit indices satisfying `0 <= p_occ < q_virt < qubits.size()` |
| Units | `theta` is dimensionless/radians in the rotation convention above; indices are dimensionless |
| Ordering/layout | The pair is ordered occupied first, virtual second, with `p_occ < q_virt`; `Z_(p,q)` acts strictly between endpoints. Qubit 0 is least significant and spin orbitals are interleaved alpha-even/beta-odd; see [conventions.md](../conventions.md) |
| Normalization | No state normalization is checked; the source-defined operation is unitary and therefore preserves the norm of a normalized input |
| Required mathematical properties | Endpoints are distinct and ascending, in range, and should come from one of the single-excitation lists returned by `get_uccsd_excitations(num_qubits, num_electrons, spin)` when UCCSD occupation semantics matter |
| Validation and rejection behavior | The device body performs no bounds, distinctness, ordering, finiteness, or occupied/virtual validation. `get_uccsd_excitations` constructs supported ascending pairs after validating system counts; it is not a validator for an arbitrary pair. A reversed pair is unsupported: the CNOT ranges become empty while local basis and `Rz` gates still execute, so it is neither a supported reverse convention nor a promised rejection. Equal or out-of-range endpoints likewise have no uniform error contract |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | `None`; this is already a four-argument `@cudaq.kernel`, not a factory and not an emitted kernel |
| Mathematical meaning | In-place application of `U_1(theta;p,q)` to the state currently held by the caller's register |
| Shape/register geometry | The caller owns the register; the kernel allocates no system or ancilla qubit and touches the two endpoints plus the parity interval between them |
| Normalization, sign, and phase | Norm-preserving; exponent sign is `-i` and the amplitude scale is `theta/2` relative to `G_1`. No separate global-phase correction is documented |
| Observable or measurement interpretation | Not applicable: the kernel performs no measurement and returns no observable |
| Error/status information | Not applicable: device execution returns neither a validation result nor a status channel |

## Capabilities and composition

Not applicable: this lower-level multi-argument device kernel neither provides
nor requires a documented reusable capability. Its concrete composition
boundary is the exact call
`stateprep.single_excitation(qubits, theta, p_occ, q_virt)` from inside a
caller-owned kernel. It does **not** satisfy the one-argument
`cudaq-algorithms.state-preparation.unitary.v1` seam; a caller would have to
prepare the reference and explicitly wrap/capture every non-register argument.

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | No algorithmic approximation or source error bound is stated; deviations from the exact source-defined unitary depend on target floating-point and gate realization |
| Precision sensitivity | The continuous rotations can be precision-sensitive. The only repository tolerance is inherited from indirect full-UCCSD comparison: `5e-5` for an fp32 CUDA-Q complex type and `1e-12` otherwise |
| Unsupported inputs | Reversed or equal endpoints, out-of-range or non-integral endpoints, non-real/non-finite amplitudes, and calls that assume the kernel also prepares a reference determinant |
| Known implementation limitations | There is no device guard. In particular, a reversed pair does not canonicalize and does not reject: its parity ladders are empty while other gates still execute |
| Unsupported, absent, and unverified behavior | A direct standalone statevector oracle test is absent; exception behavior for invalid endpoints and non-finite amplitudes is unverified; controlled/adjoint use and behavior as an injected preparation are unverified |

## Resources

No public resource estimator exists. For valid ascending endpoints with
`d = q_virt - p_occ > 0`, exact source-level structural facts are four `H`
calls, four `Rx(+-pi/2)` calls, two `Rz` calls with angles `+-theta/2`, and
`4d` controlled-`X` calls; no qubit is allocated and no measurement occurs.
These are pre-transpilation logical source calls for this leaf only, not a
depth, native-gate, runtime, or memory estimate. The count is `derived` and
exact under the valid-order assumption, controlled by endpoint distance `d`,
with additive operation-count composition; it is not measured and makes no
architecture-level claim.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | For a direct validation, construct `G_1` from dense Pauli matrices with the repository's little-endian convention and compare a wrapper-kernel state to `scipy.linalg.expm(-0.5j * theta * G_1) @ ket`. The committed suite implements this dense-matrix method for the full `uccsd` product rather than this symbol alone |
| Invariants | Unit norm and fermion-number/parity preservation for valid endpoints; `theta=0` is the identity mathematically. These are derived from `G_1`, not fresh direct executions |
| Representative cases | Indirect full-product cases `(num_qubits,num_electrons,spin) = (4,2,0), (6,3,1), (8,4,0), (10,5,1), (8,4,2)`. No standalone single-excitation case exists in the repository |
| Predeclared tolerances | For the indirect dense suite, maximum absolute statevector error `<5e-5` when `cudaq.complex()` is `complex64`, otherwise `<1e-12` (`tests/python/test_stateprep_kernels.py:127-131`) |
| Expected failure/adversarial case | A reversed pair `p_occ > q_virt` must be refused or host-checked by the caller; do not expect canonicalization or a device exception |
| Runnable example or usage test | Indirect composition command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python pytest -q -p no:cacheprovider tests/python/test_stateprep_kernels.py::test_uccsd_kernel_matches_dense_exponential` |
| Execution record | Command above; date 2026-09-10; package/CUDA-Q version, target, and precision: unverified; result: **unexecuted** during this record expansion. It is not a direct standalone test |
| Evidence status per claim | Signature, mathematical circuit, ordering, invalid-path behavior, and structural resources: `derived` from source inspected during the historical last review; dense agreement and tolerance: `derived` from an indirect committed test inspected during that review; direct execution, direct numerical validation, and measurement: `unexecuted` / `unverified` |

## Evaluation coverage

| Field | Coverage |
| --- | --- |
| Positive selection/application | Authored case `state-preparation-reference-excitation-device-boundary` selects this symbol for one UCCSD single factor inside a caller-owned kernel |
| Convention or misconception | The case requires the intended ascending occupied-to-virtual order and separates this factor from reference preparation |
| Capability composition | The case checks that the multi-argument device kernel is not itself the one-argument injectable-provider seam |
| Invalid/unsupported boundary | The case forbids promising device rejection for reversed, duplicate, or out-of-range endpoints; reversed order is unsupported here rather than canonicalized |
| Negative activation | Not applicable: no dedicated negative-activation eval targets this symbol |
| Eval status | authored; baseline run, with-skill run, and comparison not run |

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | No literature citation in the cited source defines this factor. Use the exact repository generator, Jordan-Wigner ordering, exponent sign, and `theta/2` scale rather than assuming a paper's UCC amplitude convention |
| External package translations | Not applicable: no external-package amplitude or orbital-order translation is established by the cited source, tests, or documentation |
| Known semantic differences | Within this repository, grouped Pauli-product kernels are tested with `exp(+i * theta * coefficient * P)` factors, while the full `uccsd` circuit containing this gadget is tested against `exp(-i * (theta/2) * coefficient * P)`; do not treat equal numeric amplitudes as interchangeable or infer a general conversion beyond the tested source contract |
