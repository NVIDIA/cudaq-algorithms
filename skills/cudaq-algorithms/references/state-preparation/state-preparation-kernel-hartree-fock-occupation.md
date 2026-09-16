# Explicit-occupation Hartree-Fock device kernel

Status: draft. Operation + object: **prepare** an **explicit-occupation
Hartree-Fock determinant**.

This record owns only the runtime device kernel. Host occupation construction,
the injectable factory, and the resource estimators are routed separately by
[the state-preparation front door](state-preparation.md).

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.hartree_fock_occupation`; exported in `cudaq_algorithms.stateprep.__all__` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py:47-56`; host validator and builder in `python/cudaq_algorithms/stateprep/_hartree_fock.py:74-137`; export in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | Direct assertions: `tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_explicit_occupation_statevector` and `::test_hartree_fock_open_shell_occupation` (`:114-150`). Validator assertions at `:84-106` are host-boundary evidence |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:74-108`; `docs/sphinx/examples/python/hartree_fock_ucc.py` constructs and consumes an open-shell occupation |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | None for this record. The declared target is Python `>=3.11` and `cudaq >=0.15.0,<0.16`; execution in that range is unverified |
| Lifecycle | draft |
| Implementation status | documented and implemented; source was inspected during the historical last review; not compiled, executed, or numerically validated during this record expansion |
| Replacement and migration notes | Not applicable: no deprecation or replacement was identified during the historical last review; check the current public API/source at use time |

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **prepare** + **explicit-occupation Hartree-Fock determinant** |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | host construction/validation when used safely, followed by a caller-owned device-kernel call |
| Input representations | a live `cudaq.qview` and a runtime list of occupied spin-orbital indices |
| Output representations | in-place state transformation; no host return |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16` as declared by `pyproject.toml` |
| Optional dependencies | Not applicable: the kernel has no optional dependency |
| Exactness | exact for the emitted sequence of logical `X` calls on valid inputs |
| Uncertainty | deterministic |
| Method | explicit computational-basis preparation by indexed bit flips |

## Scientific contract

| Field | Contract |
| --- | --- |
| Purpose | Prepare a reference determinant with a caller-specified occupied spin-orbital set, including open-shell interleaved occupations |
| Mathematical definition | For valid distinct indices `O=(o_0,...,o_(m-1))` and all-zero input, apply `U_O = product_(j=0)^(m-1) X_(o_j)`. The result is the basis state with occupied set `O` and little-endian index `sum_(o in O) 2**o`. The source emits gates in list order; for valid distinct indices the `X` factors commute |
| Why and when to use | Use inside a caller's kernel when the exact occupied-orbital list is already available, especially an open-shell list produced by `make_hartree_fock_occupation` |
| When not to use | Do not pass unvalidated hand-built indices or an empty runtime list. Use [`hartree_fock`](state-preparation-kernel-hartree-fock.md) for a simple contiguous prefix, or `hartree_fock_ucc_kernel` for a factory-validated one-register injectable provider |
| Approximation controls | Not applicable: no approximation parameter exists |

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact signature: `hartree_fock_occupation(qubits: cudaq.qview, occupied_orbitals: list[int])` |
| Shapes/ranks | `qubits` is a one-dimensional live register view; `occupied_orbitals` is a flat runtime list |
| Dtypes/domains | Every valid entry is a non-boolean, non-negative integer strictly below `qubits.size()`; entries are unique; the raw runtime list must be non-empty because of the cited CUDA-Q empty-list marshaling limitation |
| Units | Not applicable: spin-orbital indices are dimensionless |
| Ordering/layout | Each index is a spin-orbital/qubit index; qubit 0 is least significant, spatial orbital `p` maps to alpha `2p` and beta `2p+1`, and `spin` means `2*S_z`; see [conventions.md](../conventions.md) |
| Normalization | The input register must be a normalized all-zero state for the documented preparation meaning; the unitary preserves norm |
| Required mathematical properties | The indices must describe a set: valid, in range, and pairwise distinct. A safe open-shell list is produced by `make_hartree_fock_occupation(num_qubits, num_electrons, spin)` |
| Validation and rejection behavior | The device body checks nothing. Call `validate_hartree_fock_occupation(qubits_width, occupied_orbitals)` on the host: it rejects non-integral/negative, out-of-range, and duplicate entries with `ValueError`. Without it, duplicates toggle a qubit repeatedly, out-of-range behavior is unverified, and no uniform device exception is promised. The host validator can accept an empty list, but that does not remove the raw kernel's empty-list marshaling limitation |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | `None`; this is already a multi-argument `@cudaq.kernel`, not a factory and not an emitted kernel |
| Mathematical meaning | In-place application of `U_O`; from all-zero input and valid distinct indices, the determinant occupying exactly `O` |
| Shape/register geometry | The caller owns and allocates the register; the kernel allocates no system or ancilla qubit |
| Normalization, sign, and phase | Norm-preserving with computational-basis amplitude `+1` for the direct all-zero preparation; input list order does not change that valid distinct-index result |
| Observable or measurement interpretation | Not applicable: the kernel performs no measurement and returns no observable |
| Error/status information | Not applicable: device execution returns neither a validation result nor a status channel |

## Capabilities and composition

Not applicable: this multi-argument raw device kernel neither provides nor
requires a documented reusable capability. Its concrete composition boundary
is the exact call
`stateprep.hartree_fock_occupation(qubits, occupied_orbitals)` from inside a
caller-owned kernel. It does **not** satisfy the one-argument
`cudaq-algorithms.state-preparation.unitary.v1` injection seam unless the
caller explicitly wraps and captures a non-empty validated occupation.

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | The source introduces no approximation and states no numerical error bound; target gate realization is outside this source-level contract |
| Precision sensitivity | Not applicable at the logical source level because this kernel has no continuous parameter |
| Unsupported inputs | Empty runtime lists, duplicate or non-integral indices, negative or out-of-range indices, and a nonzero input used as though it were a freshly prepared reference |
| Known implementation limitations | Duplicate indices are not rejected and apply `X` repeatedly, so an even duplicate count can silently erase the intended occupation. Empty lists cannot cross this kernel boundary according to `_kernels.py:51-53` |
| Unsupported, absent, and unverified behavior | Out-of-range launch behavior is unverified; host validation inside the raw device kernel is absent; controlled, adjoint, dirty-input, and width-mismatch semantics beyond the literal gate sequence are unverified |

## Resources

For a valid non-empty list of length `m`, the exact source-level structural
cost is `m` logical `X` calls, no ancilla allocation, and no measurement. The
metric is pre-transpilation logical operation calls, the abstraction is the raw
device body, the execution assumption is a caller-owned register containing
every listed index, and the controlling parameter is
`len(occupied_orbitals)`. The count is exact and derived from historically
reviewed source, not measured; it composes additively with surrounding logical
operations. The separate public `estimate_hartree_fock_occupation_resources`
contract is routed through
[state-preparation-resources.md](state-preparation-resources.md). This claim is
not circuit depth, transpiled gates, hardware runtime, or memory.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | Build a length-`2**n` zero vector and place amplitude one at `sum_(o in O) 2**o`. `test_hartree_fock_explicit_occupation_statevector` uses this direct basis-vector oracle for `(n,O)=(4,[0,2])`; the open-shell test repeats it for `(8,[0,1,2,4])` |
| Invariants | Unit norm; every valid listed bit and no unlisted bit is set; permutations of a valid distinct index list produce the same basis state |
| Representative cases | `[0,2] -> 0b0101`; open-shell `(num_qubits,num_electrons,spin)=(8,4,2)` produces `[0,1,2,4] -> 0b00010111` and is cross-checked against the occupation implied by `get_uccsd_excitations` |
| Predeclared tolerances | `atol=1.0e-12` in `_assert_basis_state` at `tests/python/test_stateprep_hf_ucc.py:53-56`; no relative tolerance is declared there |
| Expected failure/adversarial case | The host validator rejects `[0,2,2]`, `[0,-1]`, and an index equal to the register width; the raw kernel must not be described as raising for them |
| Runnable example or usage test | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python pytest -q -p no:cacheprovider tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_explicit_occupation_statevector tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_open_shell_occupation` |
| Execution record | Command above; date 2026-09-10; package/CUDA-Q version, target, and precision: unverified; result: **unexecuted** during this record expansion |
| Evidence status per claim | Signature, gate sequence, list-order behavior, and structural cost: `derived` from source inspected during the historical last review; cited test results and tolerance: `derived` from committed evidence inspected during that review; fresh execution, compilation, numerical validation, and measurement: `unexecuted` / `unverified` |

## Evaluation coverage

| Field | Coverage |
| --- | --- |
| Positive selection/application | Authored case `state-preparation-reference-excitation-device-boundary` selects `make_hartree_fock_occupation` plus this kernel for its requested open-shell reference |
| Convention or misconception | The case checks explicit alpha-even/beta-odd open-shell occupation instead of contiguous filling |
| Capability composition | The case checks the caller-owned multi-argument device boundary rather than the one-argument injectable-provider seam |
| Invalid/unsupported boundary | The case requires host validation and forbids promising raw-device rejection of duplicate or out-of-range indices |
| Negative activation | Not applicable: no dedicated negative-activation eval targets this symbol |
| Eval status | authored; baseline run, with-skill run, and comparison not run |

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | No literature citation defines this source contract. Interpret indices using the repository's little-endian, interleaved alpha-even/beta-odd convention |
| External package translations | Not applicable: no external-package translation is established by the cited source, tests, or documentation |
| Known semantic differences | Blocked-spin external occupations must be permuted to the package's interleaved indexing before use. Duplicate list entries are repeated `X` operations, not set semantics, unless rejected on the host |
