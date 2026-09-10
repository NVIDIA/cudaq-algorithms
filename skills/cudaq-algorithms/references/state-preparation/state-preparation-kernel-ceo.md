# Parameterized CEO device kernel

Status: draft. Operation + object: **apply** a **coupled-exchange-operator
product** to a caller-owned quantum register.

This is one independently selectable primitive contract. CEO's spatial-orbital
units and non-Jordan-Wigner generator content remain part of the contract even
though its runtime containers resemble the grouped UCC kernels.

## Identity and provenance

| Field | Contract |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.ceo`; matching host provider `get_ceo_pauli_lists` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py` (`ceo`); `python/cudaq_algorithms/stateprep/_pools.py` (CEO construction and `get_ceo_pauli_lists`); generic validator in `python/cudaq_algorithms/stateprep/_hartree_fock.py`; exports in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | `tests/python/test_stateprep_kernels.py` (`test_ceo_kernel_matches_dense_exponential`); absolute provider oracle in `tests/python/test_operator_pools.py`; shapes and smoke use in `tests/python/test_stateprep.py` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst`, especially “Ansatz kernels and operator pools”; runnable use in `tests/python/test_stateprep_kernels.py` |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | **unverified.** Declared Python `>=3.11`, `cudaq >=0.15.0,<0.16`; no compatible execution was completed for this record |
| Lifecycle | draft |
| Implementation status | documented; source and committed tests were inspected during the historical last review; `unexecuted` and not freshly numerically validated |
| Replacement and migration notes | Not applicable: no deprecation or removal was identified during the historical last review; check the current public API/source at use time |

## Classification

| Field | Contract |
| --- | --- |
| Operation + mathematical object (primary identity) | apply + coupled-exchange-operator product |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | host CEO-pool construction/validation, then device-kernel execution |
| Input representations | caller-owned `cudaq.qview`, flat amplitudes, grouped full-width Pauli words, grouped real coefficients |
| Output representations | in-place transformed live register; no host return value |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16`; a CUDA-Q target to execute the enclosing kernel |
| Optional dependencies | Not applicable: none in the public contract |
| Exactness | exact ordered product for supplied grouped data, subject to target floating-point synthesis |
| Uncertainty | deterministic |
| Method | coupled-exchange-operator ansatz product |

## Scientific contract

- **Purpose:** apply the matching CEO provider's same-spin singles and
  coupled-exchange double groups at one runtime amplitude per group.
- **Mathematical definition:** for `G = len(pauli_words_list)` and group size
  `n_g`, apply in listed group and term order

  ```text
  product_{g=0}^{G-1} product_{j=0}^{n_g-1}
      exp(+i * thetas[g] * coefficients_list[g][j] * P_gj).
  ```

  One amplitude is shared by every term in a group. The matching provider emits
  alpha singles, beta singles, alpha same-spin doubles, beta same-spin doubles,
  then mixed doubles. CEO generators deliberately omit Jordan-Wigner `Z`
  parity strings.
- **Why and when to use:** use within a caller-written kernel when the
  coupled-exchange construction produced by `get_ceo_pauli_lists(M)` is the
  intended ansatz on `2*M` spin-orbital qubits.
- **When not to use:** do not use for a fermionic UCCSD/UCCGSD/UpCCGSD pool or
  reinterpret a qubit count as the provider's spatial-orbital count. Do not
  pass this four-argument kernel directly to one-argument `state_prep`.
- **Approximation controls:** Not applicable: there is no cutoff or product-
  order control.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | exact signature `ceo(qubits: cudaq.qview, thetas: list[float], pauli_words_list: list[list[cudaq.pauli_word]], coefficients_list: list[list[float]])` |
| Shapes/ranks | `thetas` is flat; words and coefficients are nested groups. Equal outer lengths and equal word/coefficient lengths per group are required |
| Dtypes/domains | amplitudes and coefficients are real CUDA-Q kernel floats; words are `cudaq.pauli_word`; provider `num_orbitals` is a non-negative integral, non-`bool` spatial-orbital count |
| Units | `num_orbitals` is spatial orbitals, so provider output width is `2*num_orbitals` qubits; amplitudes/coefficients are dimensionless and their products are radians |
| Ordering/layout | preserve CEO provider group and term order; full-width word character position equals qubit index; interleaved alpha-even/beta-odd spin orbitals |
| Normalization | no classical normalization requirement; the incoming quantum state should be normalized for physical-state interpretation |
| Required mathematical properties | register width is exactly twice the provider's spatial-orbital count; each word spans that full width; parallel lists align; data retains CEO provenance; the intended reference state is prepared first |
| Validation and rejection behavior | the device body checks none of these. `len(pauli_words_list)` controls iteration: extra theta/coefficient groups and extra within-group coefficients are ignored; missing entries may fail. `get_ceo_pauli_lists` rejects negative, fractional, or boolean `num_orbitals` with `ValueError("...must be a non-negative integer")` but permits sizes `0` and `1`, yielding empty data. `validate_fixed_parameter_ucc` rejects outer/inner mismatches and malformed string words, trusts `cudaq.pauli_word` width, and does not check finite numeric values. Device failure details are unverified |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | returns nothing; mutates the supplied `cudaq.qview` |
| Mathematical meaning | the ordered `+i` CEO Pauli product applied to the incoming state |
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
| Owning record | this record owns the direct boundary; [state-preparation.md](state-preparation.md) owns the different one-register capability |
| Boundary representation and exact signature | direct device composition through the four-argument signature under Inputs |
| Semantic invariants | retain CEO provenance, spatial-to-spin width conversion, non-parity-string construction, one amplitude per group, and group/term order |
| Shape/register geometry | one caller-owned register of width `2*num_orbitals`, matching every full-width word |
| Normalization, sign, phase, and ordering | the `+i` ordered unitary contract under Outputs |
| Convention requirements | [conventions.md](../conventions.md), especially Pauli-word position and host validation; [operator-pool-ceo.md](operator-pool-ceo.md) owns provider enumeration |
| Host/device/simulation boundary | construct and validate on host, marshal runtime groups to device; simulation is optional validation only |
| Unsupported conditions | direct one-register injection; treating CEO groups as fermionic/Jordan-Wigner UCC groups merely because the signature matches |

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Contract |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation bound; source gives no target floating-point synthesis bound |
| Precision sensitivity | dense tests predeclare maximum amplitude error `<1e-12` at double precision or `<5e-5` at single precision |
| Unsupported inputs | wrong spatial-to-qubit width, misaligned groups, wrong-width/invalid words, non-real angles, or data without CEO meaning |
| Known implementation limitations | no device validation; extra list entries may be ignored and missing entries may fail; generic validation cannot inspect `cudaq.pauli_word` width |
| Unsupported, absent, and unverified behavior | uniform malformed-input errors, non-finite values, hardware execution, controlled/adjoint wrapping, literature equivalence, and equivalence after reordering are unverified; reference preparation and amplitude optimization are absent |

## Resources

No CEO-kernel-specific estimator exists. For group sizes `n_g`, the exact
source-level quantity is `R = sum_g n_g` logical `exp_pauli` invocations, an
exact pre-transpilation, target-independent metric additive under sequential
composition; `G` amplitudes are consumed and no ancilla is allocated. For
matching provider data with `M` spatial orbitals,
`G = 2*C(M,2) + 12*C(M,4) + 2*C(M,2)^2` and
`R = 4*C(M,2) + 48*C(M,4) + 8*C(M,2)^2`; single groups have two terms and
double groups four. These counts are not native gates, depth, runtime, memory,
or measured cost.

## Validation

| Field | Contract |
| --- | --- |
| Independent oracle | an absolute known answer at `M=2` pins all CEO provider words, coefficients, order, and signs; a separate NumPy/SciPy matrix-exponential product checks device action |
| Invariants | unit norm, unchanged `2*M` width, no parity strings in provider generators, one amplitude per group, `+i` sign, and exact order |
| Representative cases | dense device comparisons at `M=2,3,4`; `M=2` has four groups with term counts `[2,2,4,4]`, and `M=4` exercises same-spin doubles |
| Predeclared tolerances | maximum amplitude error `<1e-12` for double precision or `<5e-5` for single precision; absolute provider coefficients are exact dyadic values |
| Expected failure/adversarial case | invalid provider counts reject; passing `4` as `num_orbitals` when four qubits were intended silently constructs eight-qubit words; malformed parallel data must be rejected on host |
| Runnable example or usage test | `pytest -q tests/python/test_stateprep_kernels.py -k ceo`; provider oracle in `pytest -q tests/python/test_operator_pools.py -k ceo` |
| Execution record | unexecuted for this record in the declared package/CUDA-Q range; no result, target, or precision is claimed |
| Evidence status per claim | signatures, loops, formulas, and cited assertions are `derived` from source and committed tests inspected during the historical last review; fresh compilation, execution, numerical validation, literature alignment, and measurement are `unverified` with `unexecuted` qualifier |

## Evaluation coverage

| Field | Contract |
| --- | --- |
| Positive selection/application | authored `state-preparation-grouped-ucc-device-boundary` selects this record and its matching provider |
| Convention or misconception | authored `operator-pool-ceo-units` covers spatial-orbital units and the non-Jordan-Wigner construction; the grouped-boundary case rejects provider interchangeability |
| Capability composition | the grouped-boundary case keeps this runtime signature distinct from one-register `state_prep` |
| Invalid/unsupported boundary | the two cases cover width conversion, provider provenance, host validation, and ignored-extra/missing-entry behavior |
| Negative activation | Not applicable: no dedicated non-activation case targets this record |
| Eval status | authored in `../../evals/evals.json`; baseline and with-skill arms have not run, so no comparison is claimed |

## External alignment

| Field | Contract |
| --- | --- |
| Literature conventions | source labels the provider “arXiv:2407.08696 conventions,” but that paper was not consulted; alignment is unverified |
| External package translations | unverified; align spatial-orbital units, interleaved qubit mapping, generator content, group order, coefficient signs, and exponent convention |
| Known semantic differences | CEO uses coupled-exchange Pauli products without Jordan-Wigner parity strings; it is not a fermionic UCC pool even when a grouped device signature looks identical |
