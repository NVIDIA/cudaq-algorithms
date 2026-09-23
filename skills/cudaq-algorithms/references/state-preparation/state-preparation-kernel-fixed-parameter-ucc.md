# Arbitrary fixed-parameter UCC device kernel

Operation + object: **apply** an **arbitrary grouped fixed-
parameter UCC product** to a caller-owned quantum register.

This is one independently selectable primitive contract. The caller chooses
the operator-pool meaning and supplies fixed amplitudes; this record does not
turn that choice into UCCSD, UCCGSD, UpCCGSD, or CEO provenance.

## Identity and provenance

| Field | Contract |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.fixed_parameter_ucc`; host companions `get_fixed_parameter_ucc_pauli_lists`, `validate_fixed_parameter_ucc`, and `estimate_fixed_parameter_ucc_resources` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py` (`fixed_parameter_ucc`); `python/cudaq_algorithms/stateprep/_hartree_fock.py` (converter, validator, estimator, and injectable factory); exports in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | `tests/python/test_stateprep_hf_ucc.py` (converter, validation, resource fields, grouped-argument launch, and dense product); supporting grouped-kernel comparisons in `tests/python/test_stateprep_kernels.py` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst`, especially “Hartree-Fock references and fixed-parameter UCC”; runnable example `docs/sphinx/examples/python/05_state_prep_and_injection.py` and tests above |

## Classification

| Field | Contract |
| --- | --- |
| Operation + mathematical object (primary identity) | apply + arbitrary grouped fixed-parameter UCC product |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime, although amplitudes are fixed/non-variational in the calling workflow |
| Execution layers | host pool conversion/validation and optional resource estimation, then device-kernel execution |
| Input representations | caller-owned `cudaq.qview`, flat amplitudes, grouped full-width Pauli words, grouped real coefficients |
| Output representations | in-place transformed live register; no host return value |
| Domain | domain-independent at the execution boundary; a caller-selected pool may impose a domain such as quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16`; a CUDA-Q target to execute the enclosing kernel |
| Optional dependencies | Not applicable: none in the public contract |
| Exactness | exact ordered product for the supplied grouped data after converter pruning, subject to target floating-point synthesis |
| Uncertainty | deterministic |
| Method | fixed-amplitude unitary coupled-cluster-style Pauli product over a caller-selected pool |

## Scientific contract

- **Purpose:** apply a caller-selected grouped operator pool at one supplied,
  fixed amplitude per group after the caller prepares the intended reference
  determinant.
- **Mathematical definition:** for `G = len(pauli_words_list)` and group size
  `n_g`, apply strictly in listed group and term order

  ```text
  product_{g=0}^{G-1} product_{j=0}^{n_g-1}
      exp(+i * thetas[g] * coefficients_list[g][j] * P_gj).
  ```

  One `theta_g` is shared across the terms of group `g`; `exp_pauli` fixes the
  `+i` sign and there is no factor of one half.
- **Why and when to use:** use inside a caller-written kernel for fixed,
  already-known amplitudes over grouped Pauli data from any operator pool.
- **When not to use:** do not use on an all-zero register when a determinant
  reference is scientifically required—the kernel source calls that result
  physically meaningless. Do not use it as a direct one-register injectable
  provider; use [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md) for
  the factory that captures validated data and prepares the reference.
- **Approximation controls:** `coefficient_tolerance` belongs only to
  `get_fixed_parameter_ucc_pauli_lists`; it drops terms with
  `abs(real(coefficient)) <= tolerance` and therefore changes the represented
  product. The runtime kernel itself has no approximation control.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | exact signature `fixed_parameter_ucc(qubits: cudaq.qview, thetas: list[float], pauli_words_list: list[list[cudaq.pauli_word]], coefficients_list: list[list[float]])` |
| Shapes/ranks | `thetas` is flat; words and coefficients are nested groups. Validated use requires equal outer lengths and equal inner word/coefficient lengths |
| Dtypes/domains | amplitudes and coefficients are real CUDA-Q kernel floats; words are `cudaq.pauli_word`; host count and tolerance are real/integral as documented by their helpers |
| Units | amplitudes and coefficients are dimensionless; their products are radians consumed by `exp_pauli` |
| Ordering/layout | retain caller pool order across groups and source term order within each group; every full-width word uses character position as qubit index, qubit 0 least significant |
| Normalization | no classical normalization condition; incoming quantum state should be normalized for physical interpretation |
| Required mathematical properties | every word spans `qubits.size()`; lists align positionally; coefficients/angles are real; pool meaning and group order are retained; intended reference determinant is prepared before this call |
| Validation and rejection behavior | the kernel validates nothing. `len(pauli_words_list)` drives iteration: extra theta/coefficient groups and extra within-group coefficients are ignored; missing parallel entries may fail; empty word groups emit no rotations. Before launch, `validate_fixed_parameter_ucc` raises `ValueError` for unequal outer lengths, unequal inner lengths, and string words with wrong width or characters outside `IXYZ`; it trusts `cudaq.pauli_word` width and does not validate numeric finiteness. `get_fixed_parameter_ucc_pauli_lists` rejects negative tolerance and material imaginary coefficients, while silently pruning negligible real terms. Uniform device errors remain unverified |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | returns nothing; mutates the supplied `cudaq.qview` |
| Mathematical meaning | the ordered `+i` Pauli product above applied to the incoming, caller-prepared state |
| Shape/register geometry | width unchanged; no ancilla, control, or measurement register allocated |
| Normalization, sign, and phase | norm-preserving unitary; exponent sign `+i`, no `1/2`; phases follow strict product order |
| Observable or measurement interpretation | Not applicable: no measurement or observable is produced |
| Error/status information | Not applicable: no return/status channel; malformed runtime inputs have no uniform characterized rejection mode |

## Capabilities and composition

| Field | Contract |
| --- | --- |
| Stable ID | Not applicable: no capability ID is assigned to this multi-argument kernel |
| Capability status | Not applicable |
| Direction | Not applicable |
| Owning record | this record owns the direct boundary; [injection-contract.md](injection-contract.md) owns the different one-register capability |
| Boundary representation and exact signature | direct device composition through the exact four-argument signature under Inputs |
| Semantic invariants | one amplitude per group, caller pool provenance, full-width words, `+i` exponent, strict group/term order, and prior reference preparation |
| Shape/register geometry | one caller-owned register matching every Pauli word; no ancilla |
| Normalization, sign, phase, and ordering | the ordered unitary contract under Outputs |
| Convention requirements | [conventions.md](../conventions.md), especially Pauli-word position and host-before-device validation |
| Host/device/simulation boundary | convert, validate, and optionally estimate on host; pass nested groups at runtime to device; simulation is optional validation only |
| Unsupported conditions | direct one-register injection; treating arbitrary pool data as a named ansatz without preserving and checking its provenance |

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Contract |
| --- | --- |
| Error behavior or bounds | no runtime algorithmic approximation except any host converter pruning; source provides no target floating-point error bound |
| Precision sensitivity | dense tests predeclare maximum amplitude error `<1e-12` at double precision or `<5e-5` at single precision; pruning depends on `coefficient_tolerance` |
| Unsupported inputs | misaligned groups, wrong-width/invalid words, material complex coefficients, non-real angles, or missing intended reference semantics |
| Known implementation limitations | no device validation; extra entries may be ignored, missing entries may fail, empty groups no-op, and generic validation cannot inspect a `cudaq.pauli_word` width |
| Unsupported, absent, and unverified behavior | uniform malformed-input errors, non-finite values, hardware execution, controlled/adjoint wrapping, and equivalence after reordering are unverified; reference preparation and amplitude optimization are absent from this kernel |

## Resources

`estimate_fixed_parameter_ucc_resources(num_qubits, pauli_words)` returns a
`FixedParameterUccResourceEstimate` with exact structural fields
`num_qubits`, `num_excitations = G`, `num_pauli_rotations = sum_g n_g`, and
`max_pauli_rotations_per_excitation = max_g n_g` (zero for no groups). The
metric is logical `exp_pauli` calls before transpilation, under no target-
architecture assumption; it is exact for the supplied word groups, controlled
by group sizes, and additive in `num_pauli_rotations` under sequential
composition. The estimator does not validate coefficients, amplitudes, or word
widths; use the validator separately. See
[state-preparation-resources-fixed-parameter-ucc.md](state-preparation-resources-fixed-parameter-ucc.md).
These fields are not native-gate count, depth, runtime, memory, or measured
hardware cost, and no such estimate is provided.

## Validation

| Field | Contract |
| --- | --- |
| Independent oracle | `test_fixed_parameter_ucc_matches_dense_pool_exponential` constructs each pool generator as an independent dense matrix and applies SciPy `expm(+i*theta_g*G_g)` to a dense Hartree-Fock ket |
| Invariants | unit norm, unchanged width, one amplitude per group, `+i` sign, group/term order, and agreement between grouped runtime kernel and injectable factory |
| Representative cases | UCCSD-derived pools at `(num_qubits,num_electrons,spin) = (4,2,0), (6,3,1), (8,4,2)`; grouped-argument launch with UpCCGSD data at width 4 |
| Predeclared tolerances | maximum amplitude error `<1e-12` at double precision or `<5e-5` at single precision; factory/manual comparison uses `atol=1e-12` |
| Expected failure/adversarial case | validator rejects outer/inner mismatch, over- and under-width string words, and invalid Pauli characters; converter rejects complex coefficients and negative tolerance; raw-device malformed combinations remain uncharacterized |
| Runnable example or usage test | `pytest -q tests/python/test_stateprep_hf_ucc.py -k fixed_parameter_ucc` |
| Evidence status per claim | signatures, loops, validation, estimates, and cited assertions are `derived` from source and committed tests cited in the repository; fresh compilation, execution, numerical validation, and measurement are `unverified` with `unexecuted` qualifier |

## External alignment

| Field | Contract |
| --- | --- |
| Literature conventions | no paper was reviewed; “fixed-parameter UCC” here means the exact grouped ordered `+i` Pauli product above, not a universal UCC exponential convention |
| External package translations | unverified; align pool definition, qubit/Pauli ordering, coefficient sign, amplitude scale, pruning, and product order before conversion |
| Known semantic differences | same numeric amplitudes are not interchangeable with `uccsd`, whose committed oracle uses a local `-i*theta/2` pool-term relationship plus double-order signs; UCCGSD, UpCCGSD, and CEO groups retain distinct provider meanings even though this kernel can execute their shapes |
