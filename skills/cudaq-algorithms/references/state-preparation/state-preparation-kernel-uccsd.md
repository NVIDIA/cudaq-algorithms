# Parameterized UCCSD device kernel

Operation + object: **apply** an **occupied-to-virtual UCCSD
product** to a caller-owned quantum register.

This is one independently selectable primitive contract. It documents the raw
runtime device kernel, not the host UCCSD pool, the arbitrary grouped
fixed-parameter kernel, or the one-register injectable state-preparation seam.

## Identity and provenance

| Field | Contract |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.uccsd`; host companions `get_uccsd_excitations` and `get_num_uccsd_parameters` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py` (`uccsd`, `single_excitation`, `double_excitation`); `python/cudaq_algorithms/stateprep/_pools.py` (`get_uccsd_excitations`, `get_num_uccsd_parameters`, `make_uccsd_operator_pool`); exports in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | `tests/python/test_stateprep_kernels.py` (`test_uccsd_kernel_matches_dense_exponential`, `test_uccsd_interleaved_mixed_double_matches_dense_exponential`, host-invalid-input cases); `tests/python/test_operator_pools.py` (absolute and fermionic pool oracles); smoke/regression coverage in `tests/python/test_stateprep.py` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst`, especially "Ansatz kernels and operator pools"; runnable repository usage in `tests/python/test_stateprep_kernels.py` |

## Classification

| Field | Contract |
| --- | --- |
| Operation + mathematical object (primary identity) | apply + occupied-to-virtual UCCSD product |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | device kernel, with required host enumeration and validation before launch |
| Input representations | caller-owned `cudaq.qview`, flat real amplitudes, electron count, and `spin = 2*S_z` |
| Output representations | in-place transformed live quantum register; no host return value |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.16.0,<0.17`; a CUDA-Q target to execute the enclosing kernel |
| Optional dependencies | Not applicable: none in the public kernel contract |
| Exactness | exact ordered circuit for the supplied amplitudes, subject to target floating-point synthesis; it is not a Trotter approximation of a summed generator |
| Uncertainty | deterministic |
| Method | unitary coupled-cluster singles-and-doubles ansatz product |

## Scientific contract

- **Purpose:** apply the packaged UCCSD excitation sequence to a live register
  whose intended reference determinant the caller has already prepared.
- **Mathematical definition:** let `E` be the five concatenated excitation
  groups returned by `get_uccsd_excitations`: alpha singles, beta singles,
  mixed doubles, alpha doubles, beta doubles. The kernel applies their gadgets
  in that order. Relative to the ordered Pauli terms `(c_ej, P_ej)` of
  `make_uccsd_operator_pool`, the committed dense oracle represents the action
  as

  ```text
  product_e product_j exp(-i * (theta_e / 2) * c_ej * P_ej).
  ```

  The pool coefficients already contain the sign for a double with exactly
  one descending pair, matching the device gadget. A grouped `+i` consumer
  therefore needs `theta_pool = -theta_uccsd/2` uniformly, with the same
  reference, width, and excitation order; no extra per-double sign applies.
  This is the local circuit/pool relationship, not a translation to arbitrary
  external UCC parameterizations.
- **Why and when to use:** use it inside a caller-written CUDA-Q kernel when
  occupied-to-virtual UCCSD enumeration and runtime amplitudes are required.
- **When not to use:** do not use it as a drop-in grouped-pool consumer or pass
  it directly as `state_prep`; its four-argument signature is not the
  one-register injection seam. For a packaged injectable Hartree-Fock plus UCC
  provider, use [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md).
- **Approximation controls:** Not applicable: there is no cutoff, tolerance,
  order, or step-count control in this device kernel.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | exact signature `uccsd(qubits: cudaq.qview, thetas: list[float], num_electrons: int, spin: int)` |
| Shapes/ranks | `qubits` has even width `n`; `thetas` is flat with exactly `N = get_num_uccsd_parameters(n, num_electrons, spin)` entries; the two counts are scalars |
| Dtypes/domains | amplitudes are real CUDA-Q kernel floats; `num_electrons` and `spin` are non-negative integers; `spin` means `2*S_z`, not multiplicity |
| Units | `thetas` are dimensionless rotation parameters; emitted gate angles are radians |
| Ordering/layout | qubit 0 is least significant; spin orbitals are interleaved alpha-even/beta-odd. Amplitudes follow the exact five-group and nested-loop order above |
| Normalization | no classical normalization condition; the live quantum state is acted on unitarily and should be normalized when interpreted as a physical state |
| Required mathematical properties | `n` is even; `num_electrons <= n`; for `spin == 0`, `num_electrons` is even; for `spin > 0`, `spin <= num_electrons`, **`(num_electrons - spin) % 2 == 0`**, and `n_occ_alpha = num_electrons - (num_electrons-spin)//2 <= n//2`; the register should hold the caller's intended reference state |
| Validation and rejection behavior | the device kernel performs no validation and exposes no error channel. Validate counts through `get_uccsd_excitations` or `get_num_uccsd_parameters` before calling; both reject open-shell parity mismatches with `ValueError("(num_electrons - spin) must be even when spin > 0 (spin is 2*S_z)")`. `make_uccsd_operator_pool` shares that rejection. The raw kernel still floors `n_occ_beta = (num_electrons - spin)//2`, so bypassing host validation with `num_electrons=4, spin=1` uses the `spin=2` partition (three alpha, one beta occupied). `make_hartree_fock_occupation(n, num_electrons, spin)` enforces the same open-shell parity and builds the matching reference occupation. Too few amplitudes can index out of range; extras are ignored. Non-finite amplitudes, out-of-range derived indices, and launch error types are otherwise unverified |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | returns nothing; mutates the supplied `cudaq.qview` in place |
| Mathematical meaning | the ordered UCCSD product under "Scientific contract" applied to the incoming register state |
| Shape/register geometry | register width is unchanged; no qubit, ancilla, control, or measurement register is allocated |
| Normalization, sign, and phase | norm-preserving unitary with the tested uniform `-i/2` scale relative to the signed pool terms; grouped kernels use `+i` and need the amplitude conversion above |
| Observable or measurement interpretation | Not applicable: no measurement or observable is produced |
| Error/status information | Not applicable: a device kernel has no host return or status channel; malformed inputs do not have one uniform characterized failure mode |

## Capabilities and composition

| Field | Contract |
| --- | --- |
| Stable ID | Not applicable: no reusable capability is assigned to this multi-argument runtime kernel |
| Capability status | Not applicable |
| Direction | Not applicable |
| Owning record | this primitive record owns the direct concrete boundary; [injection-contract.md](injection-contract.md) owns a different one-register capability |
| Boundary representation and exact signature | direct device composition through `uccsd(qubits, thetas, num_electrons, spin)` |
| Semantic invariants | preserve the five-group excitation and amplitude order, local UCCSD scale/sign convention, and caller-prepared input state |
| Shape/register geometry | one caller-owned even-width register; no allocated ancilla |
| Normalization, sign, phase, and ordering | as specified under Inputs and Outputs |
| Convention requirements | little-endian qubit indexing, interleaved spin orbitals, and `spin = 2*S_z` from [conventions.md](../conventions.md) |
| Host/device/simulation boundary | enumerate, count, and validate on the host; pass data at runtime to a device kernel; simulation is optional validation, not part of the operation |
| Unsupported conditions | direct use as the `(qubits: cudaq.qview)` state-preparation capability; unchecked invalid count or amplitude combinations |

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Contract |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation bound is needed; source states no target floating-point error bound |
| Precision sensitivity | dense tests predeclare `1e-12` at double precision and `5e-5` when `cudaq.complex()` is single precision |
| Unsupported inputs | inputs violating the mathematical properties in the Inputs table; a direct injection-seam use |
| Known implementation limitations | no device validation; extra amplitudes are ignored; missing amplitudes may fail by indexing; derived-index failures are uncharacterized. The kernel source preserves inline virtual offsets to avoid a historically observed CUDA-Q 0.15 miscompile; occurrence of that miscompile on 0.16 is unverified |
| Unsupported, absent, and unverified behavior | parity-invalid open-shell input is **rejected by the host helpers but unchecked by the raw device kernel**; uniform runtime failures, dirty-reference semantics as state preparation, controlled/adjoint use, and hardware behavior are unverified; host amplitude optimization is absent |

## Resources

No UCCSD-kernel resource estimator exists. Exact structural facts only:

| Quantity | Contract |
| --- | --- |
| excitation-gadget invocations | metric/unit: calls to `single_excitation` or `double_excitation`; source-level logical abstraction before transpilation; target-independent and exact; controlled by `(n, num_electrons, spin)`; count `N = get_num_uccsd_parameters(...)`; limited to valid host inputs; sequential composition adds counts |
| runtime amplitudes consumed | metric/unit: scalar amplitudes; API-boundary abstraction; exact; controlled by the same `N`; exactly the first `N` entries are consumed and extras are ignored |

These quantities are not transpiled gate count, circuit depth, runtime, memory,
or measured hardware cost. Underlying CNOT and rotation counts depend on
excitation endpoints and target lowering; no bound or measurement is claimed.

## Validation

| Field | Contract |
| --- | --- |
| Independent oracle | a NumPy/SciPy dense ordered product of Pauli-matrix exponentials in `test_stateprep_kernels.py`, combined with absolute UCCSD pool pins at `(4,2,0)` and independent dense Jordan-Wigner ordered generator equality, including absolute signs, at `(8,4,2)` in `test_operator_pools.py::test_uccsd_spin_two_pool_matches_fermionic_generators` |
| Invariants | unit norm; unchanged register width; exact five-group parameter count/order; agreement with the dense state; explicit mixed-double amplitude at `(8,4,2)` |
| Representative cases | `(n,e,spin) = (4,2,0), (6,2,0), (6,3,1), (8,4,0), (10,5,1), (8,4,2)`, with random and arithmetic amplitudes; `(6,2,0)` includes swapped and sorted mixed doubles, and a separate `(8,4,2)` test isolates an interleaved mixed double |
| Predeclared tolerances | maximum amplitude error `<1e-12` for double precision or `<5e-5` for single precision |
| Expected failure/adversarial case | host helpers raise `ValueError` for odd `n`, odd electrons at `spin=0`, excessive counts, invalid scalar counts, and parity-invalid open-shell inputs; `test_uccsd_rejects_open_shell_spin_with_odd_electron_difference` checks enumeration `(8,4,1)`, pool construction `(6,3,2)`, and parameter counting `(10,5,2)` |
| Runnable example or usage test | `pytest -q tests/python/test_stateprep_kernels.py -k uccsd`; smoke usage also lives in `tests/python/test_stateprep.py` |
| Evidence status per claim | signatures, loops, limitations, and cited assertions are `derived` from source and committed tests cited in the repository; fresh compilation, execution, numerical validation, and measurement are `unverified` with qualifier `unexecuted` |

## External alignment

| Field | Contract |
| --- | --- |
| Literature conventions | "UCCSD" names singles and doubles, but no paper was reviewed for this record; the precise local exponent, ordering, and spin conventions above control |
| External package translations | unverified; align orbital layout, generator sign, amplitude scale, excitation order, and double-endpoint sign before comparing or converting |
| Known semantic differences | unlike the grouped UCCGSD/UpCCGSD/CEO/fixed-parameter kernels, this kernel enumerates excitations internally and uses the uniform `-i*theta/2` scale relative to its signed UCCSD pool; equal numeric amplitude lists do not establish equivalent states |
