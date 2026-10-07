# Real Givens-rotation device kernel

Operation + object: **apply** an **adjacent, real fermionic
two-mode Givens rotation**.

This is a concrete primitive record. It covers the runtime device-kernel
contract of `givens_rotation` only. The phase-aware rotation and the two
flattened Slater-determinant kernels have separate records linked from
[the state-preparation family selector](state-preparation.md).

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.givens_rotation`; it is imported by `cudaq_algorithms.stateprep` and listed in `stateprep.__all__` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_givens.py:67-85`; export surface in `python/cudaq_algorithms/stateprep/__init__.py:25-32,79` |
| Authoritative tests | No repository test calls this public symbol directly. `tests/python/test_stateprep_givens.py` exercises it transitively through `_real_entry` -> `stateprep.slater_determinant` -> `givens_rotation`, notably the analytic two-orbital case at `:153-167`, random real determinants at `:170-197`, and the sign case at `:179-188` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:151-195` names the raw call and its adjacent-pair restriction. There is no standalone repository example for this symbol; the runnable usage below is derived from the test entry-kernel pattern. `docs/sphinx/examples/python/givens_slater_determinant.py` exercises it only through `slater_determinant` |

All committed tests named here are `derived` repository evidence, not fresh
execution or measurement.

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **apply** + **adjacent, real fermionic two-mode Givens rotation** |
| Kind | quantum operation |
| Routine role | auxiliary: it is a public building block used by the flattened real Slater-determinant kernel |
| Abstraction level | leaf operation |
| Parameterization | runtime: angle and orbital indices cross the device-kernel boundary |
| Execution layers | device kernel only; any validation or schedule construction is a separate host operation |
| Input representations | caller-owned `cudaq.qview`, one real angle, and an ordered pair of integer orbital indices |
| Output representations | in-place unitary mutation of the supplied register; no host return value |
| Domain | quantum-chemistry / fermionic state preparation |
| Required dependencies | CUDA-Q kernel language and its `exp_pauli(angle, register, word)` operation |
| Optional dependencies | Not applicable: none |
| Exactness | exact for the two source-level Pauli rotations on a supported adjacent pair, subject to floating-point angle synthesis |
| Uncertainty | deterministic |
| Method | Jordan-Wigner two-mode rotation expressed as two two-qubit Pauli exponentials |

## Scientific contract

- Purpose: apply one oriented, number-preserving real Givens rotation between
  two adjacent spin orbitals inside a caller-owned CUDA-Q kernel.
- Mathematical definition: write `f = first_orbital` and
  `s = second_orbital`, and label occupations as `|n_f n_s>`. For supported
  adjacent indices, the state-preparation convention is

  ```text
  |10> ->  cos(theta) |10> + sin(theta) |01>
  |01> -> -sin(theta) |10> + cos(theta) |01>
  ```

  while `|00>` and `|11>` are unchanged. When `f + 1 == s`, the source applies
  `exp_pauli(+theta/2, qubits[f:f+2], "YX")` followed by
  `exp_pauli(-theta/2, qubits[f:f+2], "XY")`. When `s + 1 == f`, it slices
  from `s` and reverses both signs, preserving the same oriented map with
  `f` and `s` exchanged. Pauli-word position is local-qubit order, consistent
  with [the package conventions](../conventions.md).
- Why and when to use: use it when a caller-written device kernel already owns
  a register and must apply one runtime-angle adjacent rotation. It is also the
  lower-level operation used by `slater_determinant`.
- When not to use: do not use it for a non-adjacent orbital pair, for a complex
  rotation that needs a relative phase, or as a one-argument injectable
  `state_prep` provider. For validated determinant preparation, build a
  `GivensRotationSchedule` and use
  [slater_determinant_kernel](state-preparation-slater-determinant-kernel.md).
- Approximation controls: Not applicable: the kernel has no tolerance,
  truncation, step count, or other approximation parameter.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact source signature: `givens_rotation(qubits: cudaq.qview, theta: float, first_orbital: int, second_orbital: int)`; the source has no return annotation and the device operation yields no value |
| Shapes/ranks | `qubits` is a rank-1 quantum view; the other arguments are scalars. The operation addresses a contiguous two-qubit slice only when the ordered indices differ by one |
| Dtypes/domains | Supported scientific inputs use a finite real `theta` and integer indices `f,s` satisfying `0 <= f,s < qubits.size()` and `abs(f-s) == 1`. CUDA-Q performs kernel-argument marshaling; this routine performs no host coercion |
| Units | `theta` is a dimensionless angle in radians; indices are zero-based spin-orbital labels |
| Ordering/layout | both adjacent orientations are supported. Qubit 0 is least significant, Pauli-word position equals local qubit index, and spin-orbital layout is interleaved when spin is represented; see `../conventions.md` |
| Normalization | no parameter normalization applies. The operation is unitary and therefore preserves the norm of any valid input state |
| Required mathematical properties | indices must be distinct, adjacent, and in range for the stated rotation contract; the register must be live and caller-owned |
| Validation and rejection behavior | the only guard is adjacency. An in-range non-adjacent or equal pair takes neither branch and is a silent no-op. The kernel raises no host exception and returns no status. Bounds, integrality beyond the kernel annotation, and angle finiteness are not checked. An adjacent but out-of-range pair enters a slice branch and has unverified compile/launch behavior |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | no return value; this symbol is itself a multi-argument `@cudaq.kernel`, not a kernel factory |
| Mathematical meaning | the in-place unitary map stated above for a supported adjacent pair; identity for an in-range pair that fails the adjacency guard |
| Shape/register geometry | register width is unchanged; the supported branch touches exactly the contiguous two-qubit slice spanning `f` and `s`; no ancilla is allocated |
| Normalization, sign, and phase | norm and particle number are preserved. The plus sign in the `|10> -> |01>` amplitude is the package state-preparation convention and equals CUDA-Q's built-in Givens convention evaluated at `-theta`, according to the source docstring |
| Observable or measurement interpretation | Not applicable: no observable is returned and no measurement occurs |
| Error/status information | Not applicable: the device kernel has no error or status channel |

## Capabilities and composition

Not applicable: this raw multi-argument kernel neither provides nor requires a
documented capability ID. Its direct concrete composition boundary is a call
from another CUDA-Q kernel with a live `cudaq.qview` plus the runtime angle and
indices. Its full signature does not satisfy the one-register
`cudaq-algorithms.state-preparation.unitary.v1` injection seam.

## Composite protocol

Not applicable: leaf operation. The two Pauli exponentials are the fixed
implementation of this one rotation, not independently substitutable
capabilities.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation is introduced. Source gives no floating-point error bound for synthesized Pauli rotations |
| Precision sensitivity | statevector agreement depends on active simulator precision; use the predeclared fp64/fp32 tolerances under Validation |
| Unsupported inputs | non-adjacent pairs do not implement a Givens rotation; non-finite angles, non-integral values that do not marshal as annotated, and indices outside the register have no supported scientific contract |
| Known implementation limitations | invalid adjacency is a silent no-op rather than a host error. Adjacent out-of-range indexing and the operation's action on a general superposition are not characterized by a direct test, although the source-level unitary defines the latter |
| Unsupported, absent, and unverified behavior | non-adjacent rotation support is absent; host-side validation and status reporting are absent; adjacent out-of-range launch behavior, non-finite-angle behavior, and direct standalone execution of this public symbol are unverified |

## Resources

| Quantity | Contract |
| --- | --- |
| Two-qubit Pauli-exponential calls | metric/unit: source-level logical `exp_pauli` calls; abstraction: pre-transpilation device-kernel operations; assumptions: supported adjacent in-range pair; status: exact structural count of **2**; controls: the adjacency branch; limitations: not a native two-qubit-gate count, transpiled gate count, depth, runtime, or hardware measurement |
| Ancillas, measurements, and classical storage | metric/unit: allocated qubits and measurement operations; abstraction: source kernel; assumptions: all inputs; status: exact structural count of **0** for each; controls: none |
| Invalid-pair branch | metric/unit: applied logical operations; abstraction: source kernel; assumptions: an in-range pair with `abs(f-s) != 1`; status: exact structural count of **0**; controls: the two positive adjacency guards; limitations: does not characterize out-of-range pairs that satisfy an adjacency expression |

No resource estimator exists for this symbol. Counts above are structural facts
derived from source under the stated assumptions.
Sequential composition adds the `exp_pauli` call counts; no supported rule
converts them into transpiled depth, target cost, runtime, or memory.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | construct the analytic two-dimensional rotation above in the one-particle subspace, or compare the resulting one-electron amplitudes with the dense-minor determinant oracle in `tests/python/test_stateprep_givens.py:65-80`. No repository test invokes this symbol directly; the committed analytic two-orbital Slater case exercises it transitively |
| Invariants | unit norm, particle number, identity on `|00>` and `|11>`, and orientation consistency when the adjacent indices are reversed |
| Representative cases | proposed direct cases: `theta=0`, `theta=0.37` for `(0,1)` and `(1,0)`, and a two-qubit one-electron input. Committed transitive cases include the two-orbital analytic state, random real `4 x 2` and `5 x 3` determinants, and the real sign-convention case |
| Predeclared tolerances | compare statevectors with `rtol=0` and `atol=1e-12` on fp64 or `5e-5` on fp32, selected at call time by `np.dtype(cudaq.complex())`; structural counts and guard predicates are exact |
| Expected failure/adversarial case | an in-range non-adjacent pair must leave the register unchanged; an adjacent out-of-range pair must be reported as unsupported/unverified rather than promised to raise |
| Runnable example or usage test | the source-derived standalone example below prepares one electron and compares indices 1 and 2 with the analytic rotation. It was authored here but not run. The repository's transitive runnable suite is `tests/python/test_stateprep_givens.py` |
| Evidence status per claim | signatures, guards, call counts, and conventions are `derived` from source cited in the repository; the cited tests and docs are `derived` evidence from the cited assertions; this record is `unexecuted`, with compilation, numerical validation, and measurement `unverified` |

```python
import numpy as np
import cudaq
from cudaq_algorithms import stateprep

@cudaq.kernel
def one_rotation(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    stateprep.givens_rotation(q, theta, 0, 1)

theta = 0.37
actual = np.asarray(cudaq.get_state(one_rotation, theta))
expected = np.array([0.0, np.cos(theta), np.sin(theta), 0.0], complex)
atol = 5.0e-5 if np.dtype(cudaq.complex()) == np.dtype(np.complex64) else 1.0e-12
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
```

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | `_givens.py` places this operation in the adjacent Givens networks of Jiang et al., *Phys. Rev. Applied* 9, 044036 (2018), Sec. III, and Kivlichan et al., *Phys. Rev. Lett.* 120, 110501 (2018). Those citations are source-reported and were not independently revalidated in this task |
| External package translations | relative to the CUDA-Q built-in Givens convention described in the source docstring, this kernel implements the built-in operation at `-theta`. For another package, translate both its occupation ordering and its sign convention before comparing amplitudes |
| Known semantic differences | the package uses little-endian basis indexing and Pauli strings whose position is qubit index. Unlike a host validator, this device kernel silently ignores an in-range non-adjacent request |
