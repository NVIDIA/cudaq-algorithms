# Canonical Hartree-Fock device kernel

Status: draft. Operation + object: **prepare** a **contiguous Hartree-Fock
occupation**.

This is one runtime device-kernel contract. The explicit-occupation kernel,
the injectable factory, and the resource estimators are separate records
routed by [the device-kernel front door](state-preparation-device-kernels.md)
and [the state-preparation front door](state-preparation.md).

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.hartree_fock`; exported in `cudaq_algorithms.stateprep.__all__` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_kernels.py:35-44`; export in `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | Direct device-kernel assertion: `tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_canonical_statevector` (`:109-111`). Host-helper assertions at `:84-106` are related evidence, not direct kernel validation |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:74-103`; `docs/sphinx/examples/python/08_quantum_phase_estimation.py` wraps this kernel for a two-electron reference |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | None for this record. The declared target is Python `>=3.11` and `cudaq >=0.15.0,<0.16`; execution in that range is unverified |
| Lifecycle | draft |
| Implementation status | documented and implemented; source was inspected during the historical last review; not compiled, executed, or numerically validated during this record expansion |
| Replacement and migration notes | Not applicable: no deprecation or replacement was identified during the historical last review; check the current public API/source at use time |

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **prepare** + **contiguous Hartree-Fock occupation** |
| Kind | quantum operation |
| Routine role | computational |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | device kernel called from a caller-owned CUDA-Q kernel |
| Input representations | a live `cudaq.qview` and an electron count |
| Output representations | in-place state transformation; no host return |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16` as declared by `pyproject.toml` |
| Optional dependencies | Not applicable: the kernel has no optional dependency |
| Exactness | exact for the emitted sequence of logical `X` calls on valid inputs |
| Uncertainty | deterministic |
| Method | computational-basis preparation by bit flips |

## Scientific contract

| Field | Contract |
| --- | --- |
| Purpose | Prepare the canonical contiguous reference determinant inside a caller's CUDA-Q kernel |
| Mathematical definition | For register width `n`, valid `0 <= N <= n`, and all-zero input, apply `U_HF(N) = product_(i=0)^(N-1) X_i`, producing the basis state with occupied set `{0, ..., N-1}` and little-endian index `2**N - 1`. On a nonzero input the same operator toggles those qubits; that is not preparation of the documented reference |
| Why and when to use | Use for a contiguous closed-shell-style reference whose occupied spin-orbital indices are exactly the first `N` qubits |
| When not to use | Do not use for an open-shell interleaved reference. Build its indices with `make_hartree_fock_occupation` and call [`hartree_fock_occupation`](state-preparation-kernel-hartree-fock-occupation.md), or use `hartree_fock_ucc_kernel` when a validated one-register injectable provider is required |
| Approximation controls | Not applicable: no approximation parameter exists |

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact signature: `hartree_fock(qubits: cudaq.qview, num_electrons: int)` |
| Shapes/ranks | `qubits` is a one-dimensional live register view; `num_electrons` is scalar |
| Dtypes/domains | Valid contract: integral `num_electrons` with `0 <= num_electrons <= qubits.size()` |
| Units | Not applicable: the count and qubit indices are dimensionless |
| Ordering/layout | Qubit 0 is the least-significant index bit; the kernel fills the contiguous prefix. It does not accept `spin` and does not construct the alpha-even/beta-odd open-shell layout; see [conventions.md](../conventions.md) |
| Normalization | The input register must be a normalized all-zero state for the documented preparation meaning; the unitary preserves norm |
| Required mathematical properties | The live register must contain at least `num_electrons` qubits and be in `|0...0>` if the desired result is the canonical reference |
| Validation and rejection behavior | The device body performs no validation and has no error channel. `num_electrons <= 0` makes the Python `range` body empty; negative input is therefore a silent no-op, not a supported reference. An excessive count indexes beyond the view and has unverified launch behavior. For host-validated counts and open-shell construction use `make_hartree_fock_occupation` |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | `None`; this is already a multi-argument `@cudaq.kernel`, not a factory and not an emitted kernel |
| Mathematical meaning | In-place application of `U_HF(N)`; from all-zero input, the determinant occupying the first `N` spin orbitals |
| Shape/register geometry | The caller owns and allocates the register; the kernel allocates no system or ancilla qubit |
| Normalization, sign, and phase | Norm-preserving with the computational-basis amplitude `+1` for the direct all-zero preparation; no measurement or postselection phase convention applies |
| Observable or measurement interpretation | Not applicable: the kernel performs no measurement and returns no observable |
| Error/status information | Not applicable: device execution returns neither a validation result nor a status channel |

## Capabilities and composition

Not applicable: this multi-argument raw device kernel neither provides nor
requires a documented reusable capability. Its concrete composition boundary
is the exact call `stateprep.hartree_fock(qubits, num_electrons)` from inside a
caller-owned kernel. It does **not** satisfy the one-argument
`cudaq-algorithms.state-preparation.unitary.v1` injection seam unless the
caller explicitly wraps and captures `num_electrons`.

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | The source introduces no approximation and states no numerical error bound; target gate realization is outside this source-level contract |
| Precision sensitivity | Not applicable at the logical source level because this kernel has no continuous parameter |
| Unsupported inputs | Negative or non-integral counts, counts larger than the register, an open-shell occupation requested through an absent `spin` argument, and use of a nonzero input as though it were a freshly prepared reference |
| Known implementation limitations | Invalid counts are not rejected in the device body. In particular, a negative count silently emits no `X`, while an out-of-range access has no characterized uniform failure mode |
| Unsupported, absent, and unverified behavior | Host/device exception behavior for an excessive count is unverified; host validation inside this raw kernel is absent; controlled, adjoint, dirty-input, and width-mismatch semantics beyond applying the literal gate sequence are unverified |

## Resources

For valid `0 <= N <= n`, the exact source-level structural cost is `N`
logical `X` calls, no ancilla allocation, and no measurement. The metric is
pre-transpilation logical operation calls, the abstraction is the raw device
body, the execution assumption is a valid caller-owned `n`-qubit register, and
the controlling parameter is `N = num_electrons`. The count is exact and
derived from historically reviewed source, not measured; it composes additively
with surrounding logical operations. No direct public estimator belongs to
this record, and this claim is not circuit depth, transpiled gates, hardware
runtime, or memory.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | Construct a length-`2**n` zero vector and place amplitude one at index `2**N - 1`; this uses the computational-basis definition rather than the device circuit. `test_hartree_fock_canonical_statevector` does exactly this for `(n,N)=(4,3)` |
| Invariants | Unit norm; exactly the prefix `{0,...,N-1}` is occupied from all-zero input; `N=0` leaves the vacuum state |
| Representative cases | Direct committed case `(4,3) -> 0b0111`; related host evidence checks `(6,4) -> [0,1,2,3]` |
| Predeclared tolerances | `atol=1.0e-12` in `_assert_basis_state` at `tests/python/test_stateprep_hf_ucc.py:53-56`; no relative tolerance is declared there |
| Expected failure/adversarial case | `num_electrons > qubits.size()` must be rejected on the host before invoking this raw kernel; the device failure mode is intentionally not promised |
| Runnable example or usage test | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python pytest -q -p no:cacheprovider tests/python/test_stateprep_hf_ucc.py::test_hartree_fock_canonical_statevector` |
| Execution record | Command above; date 2026-09-10; package/CUDA-Q version, target, and precision: unverified; result: **unexecuted** during this record expansion |
| Evidence status per claim | Signature, gate sequence, and structural cost: `derived` from source inspected during the historical last review; cited test result and tolerance: `derived` from committed evidence inspected during that review; fresh execution, compilation, numerical validation, and measurement: `unexecuted` / `unverified` |

## Evaluation coverage

| Field | Coverage |
| --- | --- |
| Positive selection/application | Authored case `state-preparation-reference-excitation-device-boundary` selects this device-kernel family but correctly rejects it for an open-shell reference |
| Convention or misconception | The case checks that this signature has no `spin` argument and that contiguous filling is not open-shell interleaving |
| Capability composition | The case checks the caller-owned multi-argument device boundary rather than the one-argument injectable-provider seam |
| Invalid/unsupported boundary | The case rejects reliance on the raw kernel for duplicate, reversed, or out-of-range endpoint validation; for this record the relevant boundary is the unchecked count/register extent |
| Negative activation | Not applicable: no dedicated negative-activation eval targets this symbol |
| Eval status | authored; baseline run, with-skill run, and comparison not run |

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | No literature citation defines this source contract. Interpret the result using the repository's little-endian qubit and interleaved-spin-orbital conventions |
| External package translations | Not applicable: no external-package translation is established by the cited source, tests, or documentation |
| Known semantic differences | This kernel always fills a contiguous prefix. Many open-shell chemistry references require noncontiguous alpha-even/beta-odd occupation and must use the explicit-occupation path |
