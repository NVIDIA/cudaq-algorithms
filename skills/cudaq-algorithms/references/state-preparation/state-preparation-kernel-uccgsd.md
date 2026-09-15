# Parameterized UCCGSD device kernel

Status: draft. Operation + object: **apply** a **generalized UCC singles-and-
doubles product** to a caller-owned quantum register.

This is one independently selectable primitive contract. The shared nested
runtime shape does not make it interchangeable with UpCCGSD, CEO, arbitrary
fixed-parameter UCC, or the one-register injection seam.

## Identity and provenance

| Field | Contract |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.uccgsd`; matching host provider `get_uccgsd_pauli_lists` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py` (`uccgsd`); `python/cudaq_algorithms/stateprep/_pools.py` (`make_uccgsd_operator_pool`, `get_uccgsd_pauli_lists`); generic validator in `python/cudaq_algorithms/stateprep/_hartree_fock.py`; exports in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | `tests/python/test_stateprep_kernels.py` (`test_uccgsd_kernel_matches_dense_exponential`); `tests/python/test_operator_pools.py` (independent generalized-pool oracle); shape and smoke checks in `tests/python/test_stateprep.py` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst`, especially “Ansatz kernels and operator pools”; runnable use in `tests/python/test_stateprep_kernels.py` |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | **unverified.** Declared Python `>=3.11`, `cudaq >=0.15.0,<0.16`; no compatible execution was completed for this record |
| Lifecycle | draft |
| Implementation status | documented; source and committed tests were inspected during the historical last review; `unexecuted` and not freshly numerically validated |
| Replacement and migration notes | Not applicable: no deprecation or removal was identified during the historical last review; check the current public API/source at use time |

## Classification

| Field | Contract |
| --- | --- |
| Operation + mathematical object (primary identity) | apply + generalized UCC singles-and-doubles product |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | host pool construction/validation followed by device-kernel execution |
| Input representations | caller-owned `cudaq.qview`, flat amplitudes, grouped full-width Pauli words, grouped real coefficients |
| Output representations | in-place transformed live register; no host return value |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16`; a CUDA-Q target to execute the enclosing kernel |
| Optional dependencies | Not applicable: none in the public contract |
| Exactness | exact ordered product for the supplied grouped data, subject to target floating-point synthesis |
| Uncertainty | deterministic |
| Method | generalized unitary coupled-cluster singles-and-doubles ansatz product |

## Scientific contract

- **Purpose:** apply the generalized UCCGSD pool representation at one runtime
  amplitude per ordered group to a caller-prepared register.
- **Mathematical definition:** for `G = len(pauli_words_list)` and
  `n_g = len(pauli_words_list[g])`, the kernel applies, strictly in group then
  term order,

  ```text
  product_{g=0}^{G-1} product_{j=0}^{n_g-1}
      exp(+i * thetas[g] * coefficients_list[g][j] * P_gj).
  ```

  Each factor is one `exp_pauli` call. One `theta_g` is shared by every Pauli
  term in group `g`.
- **Why and when to use:** use inside a caller-written kernel when the desired
  generator family is the generalized all-qubit UCCGSD pool produced by
  `get_uccgsd_pauli_lists`, including its requested singles/doubles subset.
- **When not to use:** do not use when spin-preserving singles and paired
  doubles (UpCCGSD), coupled-exchange operators (CEO), an occupancy-restricted
  UCCSD enumeration, or arbitrary-pool semantics are intended. Do not inject
  this four-argument kernel directly as one-argument `state_prep`.
- **Approximation controls:** Not applicable: the kernel has no cutoff or
  product-order parameter. Provider subset switches are host construction
  choices, not runtime approximation controls.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | exact signature `uccgsd(qubits: cudaq.qview, thetas: list[float], pauli_words_list: list[list[cudaq.pauli_word]], coefficients_list: list[list[float]])` |
| Shapes/ranks | `thetas` is flat; words and coefficients are nested groups. Valid use requires equal outer length `G` and, for every group, equal inner word/coefficient length |
| Dtypes/domains | `thetas` and coefficients are real CUDA-Q kernel floats; words are `cudaq.pauli_word`; the matching provider accepts a non-negative integral, non-`bool` qubit count |
| Units | amplitudes and coefficients are dimensionless; their products are radians for `exp_pauli` |
| Ordering/layout | groups remain in `get_uccgsd_pauli_lists` pool order and terms in provider order; Pauli character position is qubit index, with qubit 0 least significant |
| Normalization | no classical normalization requirement; the live input state should be normalized for physical-state interpretation |
| Required mathematical properties | each Pauli word spans exactly `qubits.size()` positions; parallel lists align positionally; the register holds the caller's intended reference state before the product |
| Validation and rejection behavior | no check runs on device. The outer loop is controlled only by `len(pauli_words_list)`: extra theta/coefficient groups are ignored, missing parallel groups may fail, and extra coefficients inside a group are ignored. Use the matching host provider and `validate_fixed_parameter_ucc(qubits_width, thetas, words, coeffs)`; that validator raises `ValueError` for outer/inner length mismatch and invalid string word width/alphabet, but trusts `cudaq.pauli_word` width and does not validate finite numeric values. Uniform device failure behavior is unverified |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | returns nothing; mutates the supplied `cudaq.qview` |
| Mathematical meaning | the ordered `+i` Pauli product defined above applied to the incoming state |
| Shape/register geometry | register width is unchanged; no ancilla, control, or measurement register is allocated |
| Normalization, sign, and phase | norm-preserving unitary; exponent sign is `+i` with no factor of one half; global and relative phases are those of the ordered product |
| Observable or measurement interpretation | Not applicable: no measurement or observable is emitted |
| Error/status information | Not applicable: no return/status channel; malformed runtime inputs have no single characterized rejection mode |

## Capabilities and composition

| Field | Contract |
| --- | --- |
| Stable ID | Not applicable: no capability ID is assigned to this multi-argument kernel |
| Capability status | Not applicable |
| Direction | Not applicable |
| Owning record | this primitive owns the direct boundary; [state-preparation.md](state-preparation.md) owns the different one-register capability |
| Boundary representation and exact signature | direct device composition through the exact four-argument signature under Inputs |
| Semantic invariants | preserve UCCGSD provider provenance, one amplitude per group, and group/term order |
| Shape/register geometry | one caller-owned register whose width equals every Pauli word's full width |
| Normalization, sign, phase, and ordering | `+i` ordered unitary contract under Outputs |
| Convention requirements | [conventions.md](../conventions.md), especially Pauli-word position and host-before-device validation |
| Host/device/simulation boundary | construct and validate grouped data on the host; marshal it into the enclosing device kernel; simulation is only a validation route |
| Unsupported conditions | direct injection as `(qubits: cudaq.qview)`; interpreting another pool's same-shaped groups as UCCGSD without an explicit scientific translation |

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Contract |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation bound; source states no target floating-point synthesis bound |
| Precision sensitivity | dense tests predeclare maximum state-amplitude error `<1e-12` at double precision or `<5e-5` at single precision |
| Unsupported inputs | misaligned groups, wrong-width/invalid words, non-real angles, and use without the intended provider/reference semantics |
| Known implementation limitations | no device validation; loop bounds permit ignored extra values and possible failure for missing values; a `cudaq.pauli_word` width cannot be inspected by the generic host validator |
| Unsupported, absent, and unverified behavior | uniform malformed-input errors, non-finite values, hardware execution, controlled/adjoint wrapping, and equivalence after group reordering are unverified; reference preparation and amplitude optimization are absent |

## Resources

No UCCGSD-kernel-specific estimator exists. For group sizes `n_g`, the exact
source-level count is `R = sum_g n_g` logical `exp_pauli` invocations, one per
word. This metric is exact, pre-transpilation, target-independent, controlled
only by the supplied group sizes, and additive under sequential composition.
The kernel consumes `G` amplitudes and carries no ancilla. For the matching
default provider, `G = C(n,2) + 3*C(n,4)` and terms are two per single group and
eight per double group; subset switches change those host counts. None of these
figures is depth, native-gate count, runtime, memory, or measured hardware cost.

## Validation

| Field | Contract |
| --- | --- |
| Independent oracle | at `n=4`, a dense Jordan-Wigner ladder-operator construction independently checks the provider pool; a separate NumPy/SciPy matrix-exponential product checks the device action against the ordered groups |
| Invariants | unit norm; unchanged width; one amplitude per group; `+i` sign; group/term order; provider and kernel group counts agree |
| Representative cases | dense device comparison at `n=4` and `n=6`; provider shape pin at `n=4` gives six two-term groups followed by three eight-term groups |
| Predeclared tolerances | maximum amplitude error `<1e-12` for double precision or `<5e-5` for single precision; independent pool matrices use `atol=1e-10` |
| Expected failure/adversarial case | host construction rejects negative/fractional/boolean `num_qubits`; malformed parallel groups should be rejected by `validate_fixed_parameter_ucc`, not entrusted to the device kernel |
| Runnable example or usage test | `pytest -q tests/python/test_stateprep_kernels.py -k uccgsd`; host-provider oracle in `pytest -q tests/python/test_operator_pools.py -k uccgsd` |
| Execution record | unexecuted for this record in the declared package/CUDA-Q range; no result, target, or precision is claimed |
| Evidence status per claim | signatures, loops, resource formula, and cited assertions are `derived` from source and committed tests inspected during the historical last review; fresh compilation, execution, numerical validation, and measurement are `unverified` with qualifier `unexecuted` |

## Evaluation coverage

| Field | Contract |
| --- | --- |
| Positive selection/application | authored `state-preparation-grouped-ucc-device-boundary` selects this record and its matching provider |
| Convention or misconception | the case preserves UCCGSD provenance and the one-theta-per-group `+i` convention instead of treating similar signatures as equivalent |
| Capability composition | the case keeps this runtime signature distinct from the one-register injection seam |
| Invalid/unsupported boundary | the case covers width and parallel-list non-validation plus ignored-extra/missing-entry behavior |
| Negative activation | Not applicable: no dedicated non-activation case targets this record |
| Eval status | authored in `../../evals/evals.json`; baseline and with-skill arms have not run, so no comparison is claimed |

## External alignment

| Field | Contract |
| --- | --- |
| Literature conventions | no paper was reviewed; the repository's generalized all-qubit pool, ordered product, and `+i` exponent are controlling |
| External package translations | unverified; align qubit/Pauli ordering, pool membership, group order, coefficient signs, and amplitude/exponent convention before comparison |
| Known semantic differences | unlike occupied-to-virtual UCCSD, this provider ignores electron count and occupancy; unlike UpCCGSD it includes mixed-parity singles and arbitrary generalized doubles; identical nested Python shapes do not remove those differences |
