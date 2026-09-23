# UCCSD double-excitation device kernel

Operation + object: **apply** one **Jordan-Wigner UCCSD double
excitation**.

This is one runtime device-kernel contract. It does not prepare a reference
determinant and is not the full UCCSD product documented in
[the UCCSD device-kernel record](state-preparation-kernel-uccsd.md).

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.double_excitation`. The public package initializer imports the symbol, so it is addressable there, but omits it from `cudaq_algorithms.stateprep.__all__` |
| Contract-specific source paths | Public circuit and its two private realizations: `python/cudaq_algorithms/stateprep/_kernels.py:94-442`; matching pool generator: `python/cudaq_algorithms/stateprep/_pools.py:157-183`; import surface: `python/cudaq_algorithms/stateprep/__init__.py` |
| Authoritative tests | **No direct standalone test exists.** Indirect composition evidence is `tests/python/test_stateprep_kernels.py::test_uccsd_kernel_matches_dense_exponential` (`:158-182`). `::test_uccsd_interleaved_mixed_double_matches_dense_exponential` (`:185-208`) sets every other amplitude to zero and isolates the overlapping-interval tuple `[4,1,3,6]`, but still calls the full `uccsd` entry kernel |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:24-72` documents the full UCCSD kernel and its host-generated endpoints; no standalone `double_excitation` example is documented |

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **apply** + **one Jordan-Wigner UCCSD double excitation** |
| Kind | quantum operation |
| Routine role | auxiliary — a lower-level building block of `uccsd` |
| Abstraction level | leaf operation |
| Parameterization | runtime |
| Execution layers | device kernel called from a caller-owned CUDA-Q kernel |
| Input representations | a live `cudaq.qview`, one real amplitude, one occupied pair, and one virtual pair |
| Output representations | in-place unitary state transformation; no host return |
| Domain | quantum-chemistry |
| Required dependencies | `cudaq >=0.15.0,<0.16` as declared by `pyproject.toml` |
| Optional dependencies | Not applicable: the kernel has no optional dependency |
| Exactness | exact for the source-defined eight-term Pauli-rotation circuit on four valid distinct endpoints, subject to floating-point gate realization |
| Uncertainty | deterministic |
| Method | Jordan-Wigner double-excitation synthesis, using an optimized ladder when the canonical occupied pair precedes the canonical virtual pair and sparse Pauli rotations when parity intervals overlap |

## Scientific contract

For each supplied pair define canonical endpoints
`i=min(p_occ,q_occ)`, `j=max(p_occ,q_occ)`,
`a=min(r_virt,s_virt)`, and `b=max(r_virt,s_virt)`. Let
`Z_(u,v)=product_(k=u+1)^(v-1) Z_k`, with the empty product equal to identity,
and let products below retain the displayed source multiplication order:

```text
8 G_2(i,j,a,b) =
  + X_i Z_(i,j) X_j X_a Z_(a,b) Y_b
  + X_i Z_(i,j) X_j Y_a Z_(a,b) X_b
  + X_i Z_(i,j) Y_j Y_a Z_(a,b) Y_b
  + Y_i Z_(i,j) X_j Y_a Z_(a,b) Y_b
  - X_i Z_(i,j) Y_j X_a Z_(a,b) X_b
  - Y_i Z_(i,j) X_j X_a Z_(a,b) X_b
  - Y_i Z_(i,j) Y_j X_a Z_(a,b) Y_b
  - Y_i Z_(i,j) Y_j Y_a Z_(a,b) X_b.
```

Define `t=-theta` when exactly one of the supplied pairs is descending, and
`t=theta` when neither or both pairs are descending.

| Field | Contract |
| --- | --- |
| Purpose | Apply one occupied-pair-to-virtual-pair UCCSD double-excitation factor to an already prepared live register |
| Mathematical definition | For four distinct in-range indices, the source **canonicalizes both pairs** as above and realizes `U_2(theta;p,q,r,s) = exp(-i * (t/2) * G_2(i,j,a,b))`. Thus swapping both pairs preserves effective `theta`; swapping exactly one pair flips it. This sign is additional to pair canonicalization and must not be dropped |
| Why and when to use | Use as a low-level call inside a caller-owned kernel for a four-endpoint tuple produced under the package UCCSD enumeration convention; both separated and overlapping Jordan-Wigner parity intervals are supported |
| When not to use | Do not use it to prepare Hartree-Fock, as a directly injectable one-register provider, with repeated/out-of-range indices, or as an amplitude-compatible substitute for a grouped Pauli-product kernel |
| Approximation controls | Not applicable: no approximation parameter exists; `theta` is the physical runtime parameter, not an error tolerance |

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact signature: `double_excitation(qubits: cudaq.qview, theta: float, p_occ: int, q_occ: int, r_virt: int, s_virt: int)` |
| Shapes/ranks | One live register view and five scalar runtime arguments |
| Dtypes/domains | `theta` is real; all four endpoints are integer qubit indices in `[0, qubits.size())` and are pairwise distinct |
| Units | `theta` is dimensionless/radians in the rotation convention above; indices are dimensionless |
| Ordering/layout | The first two endpoints form the occupied pair and the last two the virtual pair. Either within-pair ordering is supported through canonicalization, with the exactly-one-descending sign rule. Qubit 0 is least significant and spin orbitals are interleaved alpha-even/beta-odd; see [conventions.md](../conventions.md) |
| Normalization | No state normalization is checked; the source-defined operation is unitary and therefore preserves the norm of a normalized input |
| Required mathematical properties | All four indices are distinct and in range. When UCCSD occupation semantics matter, take a tuple from the mixed-, alpha-, or beta-double lists returned by `get_uccsd_excitations(num_qubits, num_electrons, spin)` |
| Validation and rejection behavior | The device body performs no bounds, distinctness, finiteness, or occupied/virtual validation. It only branches on within-pair comparisons. Equal endpoints leave canonical locals at their zero initialization in some paths; other repeated or out-of-range indices can fail or silently realize a wrong unitary. `get_uccsd_excitations` constructs supported tuples after validating system counts; it is not a validator for an arbitrary tuple. No uniform device exception is promised |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | `None`; this is already a six-argument `@cudaq.kernel`, not a factory and not an emitted kernel |
| Mathematical meaning | In-place application of `U_2(theta;p,q,r,s)` to the state currently held by the caller's register |
| Shape/register geometry | The caller owns the register; the kernel allocates no system or ancilla qubit. It touches the four endpoints and the symmetric-difference support of the two Jordan-Wigner parity intervals |
| Normalization, sign, and phase | Norm-preserving; exponent sign is `-i`, scale is `t/2` relative to `G_2`, and `t` follows the exactly-one-descending rule. No separate global-phase correction is documented |
| Observable or measurement interpretation | Not applicable: the kernel performs no measurement and returns no observable |
| Error/status information | Not applicable: device execution returns neither a validation result nor a status channel |

## Capabilities and composition

Not applicable: this lower-level multi-argument device kernel neither provides
nor requires a documented reusable capability. Its concrete composition
boundary is the exact call
`stateprep.double_excitation(qubits, theta, p_occ, q_occ, r_virt, s_virt)`
from inside a caller-owned kernel. It does **not** satisfy the one-argument
`cudaq-algorithms.state-preparation.unitary.v1` seam; a caller would have to
prepare the reference and explicitly wrap/capture every non-register argument.

## Composite protocol

Not applicable: leaf operation.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | No algorithmic approximation or source error bound is stated; deviations from the exact source-defined unitary depend on target floating-point and gate realization |
| Precision sensitivity | The continuous rotations can be precision-sensitive. The only repository tolerance is inherited from indirect full-UCCSD comparison: `5e-5` for an fp32 CUDA-Q complex type and `1e-12` otherwise |
| Unsupported inputs | Any repeated endpoints, out-of-range or non-integral endpoints, non-real/non-finite amplitudes, and calls that assume the kernel also prepares a reference determinant |
| Known implementation limitations | There is no device guard. Pair equality can leave the canonical local indices at zero rather than reject; cross-pair repeats and out-of-range indices likewise have no defined safe result |
| Unsupported, absent, and unverified behavior | A direct standalone test is absent; exception behavior for invalid endpoints and non-finite amplitudes is unverified; controlled/adjoint use and behavior as an injected preparation are unverified |

## Resources

No public resource estimator exists. For four valid distinct endpoints, both
source branches implement eight Pauli-term gadgets and emit exactly eight `Rz`
calls; they allocate no qubit and perform no measurement. If canonical
`j < a`, `_ordered_double_excitation` uses the optimized ladder; otherwise the
kernel invokes `_double_pauli_rotation` eight times so overlapping parity
intervals are represented on their sparse symmetric-difference support.
Other one- and two-qubit logical-call counts depend on endpoint geometry and
the selected branch. These are exact pre-transpilation structural facts,
controlled by pair order and interval overlap, not a depth, native-gate,
runtime, or memory estimate. They compose additively only at the logical-call
level, are derived from source rather than measured, and
make no architecture-level claim.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | Canonicalize both pairs, compute `t` with the exactly-one-descending rule, construct the displayed `G_2` from dense little-endian Pauli matrices, and compare a wrapper-kernel state to `scipy.linalg.expm(-0.5j * t * G_2) @ ket`. The committed suite implements this dense-matrix method through the full `uccsd` kernel rather than calling this symbol alone |
| Invariants | Unit norm and fermion-number/parity preservation for valid endpoints; swapping neither or both pairs keeps `t`, while swapping exactly one pair negates `t`. These are source-derived, not fresh direct executions |
| Representative cases | Indirect full-product cases `(4,2,0)`, `(6,3,1)`, `(8,4,0)`, `(10,5,1)`, and `(8,4,2)`. The composed `(8,4,2)` case isolates `[4,1,3,6]`, whose occupied pair descends and virtual pair ascends, so effective `theta` flips and the parity intervals overlap |
| Predeclared tolerances | For the indirect dense suite, maximum absolute statevector error `<5e-5` when `cudaq.complex()` is `complex64`, otherwise `<1e-12` (`tests/python/test_stateprep_kernels.py:127-131`) |
| Expected failure/adversarial case | Repeat an endpoint within or across the two pairs: the caller must reject it before launch; do not expect a device exception or a meaningful canonicalization |
| Runnable example or usage test | Indirect composition command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python pytest -q -p no:cacheprovider tests/python/test_stateprep_kernels.py::test_uccsd_kernel_matches_dense_exponential tests/python/test_stateprep_kernels.py::test_uccsd_interleaved_mixed_double_matches_dense_exponential` |
| Evidence status per claim | Signature, canonicalization, sign rule, mathematical circuit, invalid-path behavior, and structural resources: `derived` from source cited in the repository; dense agreement and tolerance: `derived` from indirect committed tests cited in the repository; direct execution, direct numerical validation, and measurement: `unexecuted` / `unverified` |

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | No literature citation in the cited source defines this factor. Use the displayed repository generator, multiplication order, Jordan-Wigner indexing, exponent sign, and `t/2` scale rather than assuming a paper's UCC amplitude convention |
| External package translations | Not applicable: no external-package amplitude, endpoint-order, or orbital-layout translation is established by the cited source, tests, or documentation |
| Known semantic differences | The pool generator canonicalizes pairs but carries no exactly-one-descending amplitude sign; the device circuit applies that sign adjustment. Grouped Pauli-product kernels also use a different tested exponent convention, so equal numeric amplitudes are not interchangeable |
