# Parameterized UpCCGSD device kernel

Status: draft. Operation + object: **apply** a **spin-preserving singles and
paired-doubles UpCCGSD product** to a caller-owned quantum register.

This is one independently selectable primitive contract. Its grouped runtime
shape resembles other kernels, but its paired provider semantics are part of
the contract and cannot be inferred from the Python container types.

## Identity and provenance

| Field | Contract |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.upccgsd`; matching host provider `get_upccgsd_pauli_lists` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py` (`upccgsd`); `python/cudaq_algorithms/stateprep/_pools.py` (`make_upccgsd_operator_pool`, `get_upccgsd_pauli_lists`); generic validator in `python/cudaq_algorithms/stateprep/_hartree_fock.py`; exports in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | `tests/python/test_stateprep_kernels.py` (`test_upccgsd_kernel_matches_dense_exponential`); `tests/python/test_operator_pools.py` (independent paired-pool oracle); shapes and smoke use in `tests/python/test_stateprep.py` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst`, especially “Ansatz kernels and operator pools”; runnable use in `tests/python/test_stateprep_kernels.py` |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | **unverified.** Declared Python `>=3.11`, `cudaq >=0.15.0,<0.16`; no compatible execution was completed for this record |
| Lifecycle | draft |
| Implementation status | documented; source and committed tests were inspected during the historical last review; `unexecuted` and not freshly numerically validated |
| Replacement and migration notes | Not applicable: no deprecation or removal was identified during the historical last review; check the current public API/source at use time |

## Classification

| Field | Contract |
| --- | --- |
| Operation + mathematical object (primary identity) | apply + spin-preserving singles and paired-doubles UpCCGSD product |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | host paired-pool construction/validation, then device-kernel execution |
| Input representations | caller-owned `cudaq.qview`, flat amplitudes, grouped full-width Pauli words, grouped real coefficients |
| Output representations | in-place transformed live register; no host return value |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16`; a CUDA-Q target to execute the enclosing kernel |
| Optional dependencies | Not applicable: none in the public contract |
| Exactness | exact ordered product for supplied grouped data, subject to target floating-point synthesis |
| Uncertainty | deterministic |
| Method | paired generalized unitary coupled-cluster ansatz product |

## Scientific contract

- **Purpose:** apply the matching UpCCGSD provider's spin-preserving single and
  pair-excitation groups with one runtime amplitude per group.
- **Mathematical definition:** for `G = len(pauli_words_list)` and group size
  `n_g`, apply in listed group and term order

  ```text
  product_{g=0}^{G-1} product_{j=0}^{n_g-1}
      exp(+i * thetas[g] * coefficients_list[g][j] * P_gj).
  ```

  One amplitude is shared across all Pauli terms in a group.
- **Why and when to use:** use within a caller-written kernel when interleaved
  spin-orbital layout, same-parity singles, and paired alpha/beta doubles from
  `get_upccgsd_pauli_lists` are intended; `only_doubles=True` is a host-side
  provider selection.
- **When not to use:** do not use for occupancy-restricted UCCSD, arbitrary
  generalized doubles, CEO coupled-exchange groups, or an arbitrary operator
  pool. Do not pass this four-argument kernel directly to a one-argument
  `state_prep` seam.
- **Approximation controls:** Not applicable: there is no runtime cutoff or
  product-order control. The provider's `only_doubles` switch selects a
  different exact subset.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | exact signature `upccgsd(qubits: cudaq.qview, thetas: list[float], pauli_words_list: list[list[cudaq.pauli_word]], coefficients_list: list[list[float]])` |
| Shapes/ranks | `thetas` is flat; words and coefficients are nested groups. Equal outer lengths and equal word/coefficient lengths within each group are required |
| Dtypes/domains | amplitudes and coefficients are real CUDA-Q kernel floats; words are `cudaq.pauli_word`; provider `num_qubits` is a non-negative integral, non-`bool`, even spin-orbital count |
| Units | amplitudes and coefficients are dimensionless; products are radians for `exp_pauli` |
| Ordering/layout | interleaved alpha-even/beta-odd spin orbitals; provider emits spin-preserving singles first and paired doubles second, in its documented order; Pauli character position is qubit index |
| Normalization | no classical normalization requirement; the incoming quantum state should be normalized for physical-state interpretation |
| Required mathematical properties | even register width; each word spans that full width; all parallel lists align; groups retain UpCCGSD provenance; the register holds the intended reference determinant before applying the product |
| Validation and rejection behavior | the device body checks none of these. `len(pauli_words_list)` alone drives the outer loop, so extra theta/coefficient groups are ignored, missing parallel groups may fail, and extra coefficients within a group are ignored. `get_upccgsd_pauli_lists` rejects an odd width with `ValueError("...expects an even number of spin orbitals")`; `validate_fixed_parameter_ucc` rejects outer/inner mismatches and malformed string words, but trusts `cudaq.pauli_word` widths and does not check finite numeric values. Uniform device errors are unverified |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | returns nothing; mutates the supplied `cudaq.qview` |
| Mathematical meaning | the ordered `+i` paired UpCCGSD product applied to the incoming state |
| Shape/register geometry | width unchanged; no ancilla, control, or measurement register allocated |
| Normalization, sign, and phase | norm-preserving unitary; exponent sign `+i`, no factor of one half; phases follow the ordered product |
| Observable or measurement interpretation | Not applicable: no measurement or observable is produced |
| Error/status information | Not applicable: no return/status channel; malformed runtime inputs have no uniform characterized rejection mode |

## Capabilities and composition

| Field | Contract |
| --- | --- |
| Stable ID | Not applicable: no capability ID is assigned to this multi-argument kernel |
| Capability status | Not applicable |
| Direction | Not applicable |
| Owning record | this record owns the direct boundary; [state-preparation.md](state-preparation.md) owns a different one-register capability |
| Boundary representation and exact signature | direct device composition through the four-argument signature under Inputs |
| Semantic invariants | retain matching provider provenance, one amplitude per group, spin-preserving/paired content, and exact group/term order |
| Shape/register geometry | one caller-owned even-width register matching every full-width word |
| Normalization, sign, phase, and ordering | the `+i` ordered unitary contract under Outputs |
| Convention requirements | [conventions.md](../conventions.md), especially interleaved spin orbitals, Pauli-word position, and host validation |
| Host/device/simulation boundary | construct and validate on host, marshal runtime groups to device; simulation is optional validation only |
| Unsupported conditions | direct one-register injection and silent reinterpretation of UCCGSD, CEO, or arbitrary pool groups as UpCCGSD |

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Contract |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation bound; source gives no target floating-point synthesis bound |
| Precision sensitivity | dense tests predeclare maximum amplitude error `<1e-12` at double precision or `<5e-5` at single precision |
| Unsupported inputs | odd/wrong register width, misaligned groups, wrong-width/invalid words, non-real angles, or data without UpCCGSD meaning |
| Known implementation limitations | no device validation; extra list entries may be ignored and missing entries may fail; generic validation cannot inspect a `cudaq.pauli_word` width |
| Unsupported, absent, and unverified behavior | uniform malformed-input errors, non-finite values, hardware execution, controlled/adjoint wrapping, and equivalence after reordering are unverified; reference preparation and amplitude optimization are absent |

## Resources

No UpCCGSD-kernel-specific estimator exists. For group sizes `n_g`, the exact
source-level quantity is `R = sum_g n_g` logical `exp_pauli` invocations. It is
an exact, target-independent, pre-transpilation structural metric controlled by
the supplied groups and additive under sequential composition; the kernel
consumes `G` amplitudes and allocates no ancilla. For matching provider data at
even width `n`, `G = 3*C(n/2,2)` by default and `G = C(n/2,2)` with
`only_doubles=True`; single groups have two terms and paired-double groups have
eight. These are not depth, native-gate count, runtime, memory, or measurement.

## Validation

| Field | Contract |
| --- | --- |
| Independent oracle | at `n=4`, dense Jordan-Wigner ladder matrices independently establish the matching spin-preserving-single/paired-double pool; a separate NumPy/SciPy matrix-exponential product checks the device action |
| Invariants | unit norm, unchanged width, even-width provider domain, one amplitude per group, `+i` sign, and group/term order |
| Representative cases | dense device comparison at `n=4` and `n=8`; provider shape at `n=4` is two two-term single groups then one eight-term paired-double group |
| Predeclared tolerances | maximum amplitude error `<1e-12` for double precision or `<5e-5` for single precision; independent pool matrices use `atol=1e-10` |
| Expected failure/adversarial case | `get_upccgsd_pauli_lists(7)` rejects odd width; malformed parallel lists should be rejected with `validate_fixed_parameter_ucc`, not entrusted to the device body |
| Runnable example or usage test | `pytest -q tests/python/test_stateprep_kernels.py -k upccgsd`; provider oracle in `pytest -q tests/python/test_operator_pools.py -k upccgsd` |
| Execution record | unexecuted for this record in the declared package/CUDA-Q range; no result, target, or precision is claimed |
| Evidence status per claim | signatures, loops, formulas, and cited assertions are `derived` from source and committed tests inspected during the historical last review; fresh compilation, execution, numerical validation, and measurement are `unverified` with `unexecuted` qualifier |

## Evaluation coverage

| Field | Contract |
| --- | --- |
| Positive selection/application | authored `state-preparation-grouped-ucc-device-boundary` selects this record and the paired provider |
| Convention or misconception | the case preserves UpCCGSD provenance and rejects interchangeability based on similar signatures |
| Capability composition | the case keeps this runtime signature distinct from one-register `state_prep` |
| Invalid/unsupported boundary | the case covers even/full-width and parallel-list host validation plus ignored-extra/missing-entry behavior |
| Negative activation | Not applicable: no dedicated non-activation case targets this record |
| Eval status | authored in `../../evals/evals.json`; baseline and with-skill arms have not run, so no comparison is claimed |

## External alignment

| Field | Contract |
| --- | --- |
| Literature conventions | the authoritative test calls this `k`-UpCCGSD content, but no paper was reviewed; repository provider content and order control |
| External package translations | unverified; align interleaved spin layout, paired-excitation definition, pool order, coefficient signs, and exponent convention |
| Known semantic differences | unlike UCCGSD, singles preserve spin parity and doubles move complete alpha/beta pairs; unlike occupancy-restricted UCCSD, no electron count partitions occupied and virtual orbitals |
