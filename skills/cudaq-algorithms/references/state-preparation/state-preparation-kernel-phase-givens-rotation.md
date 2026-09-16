# Phase-aware Givens-rotation device kernel

Status: draft. Operation + object: **apply** an **adjacent, phase-aware
fermionic two-mode Givens rotation**.

This is a concrete primitive record. It covers the runtime device-kernel
contract of `phase_givens_rotation` only. Its real-rotation component and the
flattened complex Slater-determinant preparation that consumes it remain
separate records linked from
[the Givens front door](state-preparation-givens.md).

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.phase_givens_rotation`; it is imported by `cudaq_algorithms.stateprep` and listed in `stateprep.__all__` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_givens.py:88-94`; delegated real rotation at `:67-85`; export surface in `python/cudaq_algorithms/stateprep/__init__.py:25-32,80` |
| Authoritative tests | No repository test calls this public symbol directly. `tests/python/test_stateprep_givens.py` exercises it transitively through `_complex_entry` -> `stateprep.complex_slater_determinant` -> `phase_givens_rotation`, notably the analytic one-electron case at `:200-212`, the relative-phase/sign case at `:215-226`, random complex determinants at `:229-246`, and the zero-phase equivalence at `:288-307` |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:151-195` names the raw call. There is no standalone repository example for this symbol; the runnable usage below is derived from the test entry-kernel pattern. `docs/sphinx/examples/python/givens_slater_determinant.py` exercises it only through `complex_slater_determinant` |
| Source provenance | Current public source/tests are authoritative and must be checked at use time; this record's historical source review is recorded in [Source provenance](../source-provenance.md) |
| Package/CUDA-Q versions executed | **unverified.** The declared environment is Python `>=3.11` and `cudaq >=0.15.0,<0.16`; this record was not executed in that environment |
| Lifecycle | draft |
| Implementation status | documented and implemented; source was inspected during the historical last review; compilation, execution, and numerical validation are unverified in this task |
| Replacement and migration notes | Not applicable: no deprecation or replacement was identified during the historical last review; check the current public API/source at use time |

All committed tests named here are `derived` repository evidence, not fresh
execution or measurement.

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **apply** + **adjacent, phase-aware fermionic two-mode Givens rotation** |
| Kind | quantum operation |
| Routine role | auxiliary: it is a public building block used by the flattened complex Slater-determinant kernel |
| Abstraction level | leaf operation |
| Parameterization | runtime: rotation angle, relative phase, and orbital indices cross the device-kernel boundary |
| Execution layers | device kernel only; schedule construction and validation are separate host operations |
| Input representations | caller-owned `cudaq.qview`, two real scalar angles, and an ordered pair of integer orbital indices |
| Output representations | in-place unitary mutation of the supplied register; no host return value |
| Domain | quantum-chemistry / complex fermionic state preparation |
| Required dependencies | CUDA-Q kernel language, [the real Givens kernel](state-preparation-kernel-givens-rotation.md), and CUDA-Q `rz` |
| Optional dependencies | Not applicable: none |
| Exactness | exact for the delegated Pauli rotations plus one source-level `rz` on supported inputs, subject to floating-point angle synthesis |
| Uncertainty | deterministic |
| Method | real Jordan-Wigner Givens rotation followed by a relative number phase on the designated second orbital |

## Scientific contract

- Purpose: apply one oriented, number-preserving complex Givens rotation
  inside a caller-written CUDA-Q kernel.
- Mathematical definition: write `f = first_orbital`,
  `s = second_orbital`, and label occupations as `|n_f n_s>`. The kernel
  first applies the real rotation `G(theta; f,s)` from
  [its focused record](state-preparation-kernel-givens-rotation.md), then
  applies `rz(phase)` to orbital `s`. Since
  `rz(phi) = exp(i * phi * n_s)` up to the state-independent global factor
  `exp(-i*phi/2)`, the supported one-particle map is, up to that global phase,

  ```text
  |10> ->  cos(theta) |10> + exp(i*phase) sin(theta) |01>
  |01> -> -sin(theta) |10> + exp(i*phase) cos(theta) |01>.
  ```

  The operation preserves occupation number. The relative phase is applied
  **after** the real rotation and always targets `second_orbital`.
- Why and when to use: use it when a caller-owned device kernel needs one
  adjacent complex orbital rotation with runtime angle and phase. It is the
  per-rotation operation used by `complex_slater_determinant`.
- When not to use: do not use it for a real-only rotation, a non-adjacent
  pair, or direct one-register `state_prep` injection. For validated complex
  determinant preparation, construct a `GivensRotationSchedule` and call
  [slater_determinant_kernel](state-preparation-slater-determinant-kernel.md).
- Approximation controls: Not applicable: the kernel has no tolerance,
  truncation, step count, or other approximation parameter.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact source signature: `phase_givens_rotation(qubits: cudaq.qview, theta: float, phase: float, first_orbital: int, second_orbital: int)`; the source has no return annotation and the device operation yields no value |
| Shapes/ranks | `qubits` is a rank-1 quantum view; the other arguments are scalars. A supported rotation addresses the contiguous pair plus the single `second_orbital` target |
| Dtypes/domains | Supported scientific inputs use finite real `theta` and `phase`, with integer indices `f,s` satisfying `0 <= f,s < qubits.size()` and `abs(f-s) == 1`. CUDA-Q performs marshaling; this routine performs no host coercion |
| Units | `theta` and `phase` are dimensionless angles in radians; indices are zero-based spin-orbital labels |
| Ordering/layout | both adjacent orientations are accepted by the delegated real rotation; `phase` always targets the ordered argument `second_orbital`. Qubit 0 is least significant and spin-orbital layout is interleaved when spin is represented; see `../conventions.md` |
| Normalization | no parameter normalization applies. The operation is unitary and preserves the norm of a valid input state |
| Required mathematical properties | indices must be distinct, adjacent, and in range for the stated phase-aware Givens contract; the register must be live and caller-owned |
| Validation and rejection behavior | there is **no outer guard**. `givens_rotation` silently skips its Pauli rotations when the pair is non-adjacent, but `rz(phase, qubits[second_orbital])` still executes. Bounds and angle finiteness are unchecked. A non-adjacent, in-range pair therefore receives a phase-only operation; an out-of-range `second_orbital` has unverified compile/launch behavior. No host exception or status is promised |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | no return value; this symbol is itself a multi-argument `@cudaq.kernel`, not a kernel factory |
| Mathematical meaning | the in-place phase-aware Givens map stated above on a supported adjacent pair; for an in-range non-adjacent pair, only `rz(phase)` on `second_orbital` is applied |
| Shape/register geometry | register width is unchanged; no ancilla is allocated. The valid path touches the adjacent two-qubit slice and then the second orbital |
| Normalization, sign, and phase | norm and particle number are preserved. Relative occupation phase is `exp(+i*phase)` on the second orbital; the physical `rz` contributes the global factor `exp(-i*phase/2)`, which must be removed for raw-statevector comparison |
| Observable or measurement interpretation | Not applicable: no observable is returned and no measurement occurs |
| Error/status information | Not applicable: the device kernel has no error or status channel |

## Capabilities and composition

Not applicable: this raw multi-argument kernel neither provides nor requires a
documented capability ID. It directly calls the real Givens kernel and is
directly callable from another CUDA-Q kernel with a live `cudaq.qview` and all
runtime arguments. Its full signature does not satisfy the one-register
`cudaq-algorithms.state-preparation.unitary.v1` injection seam.

## Composite protocol

Not applicable: leaf operation. The delegated real rotation plus the fixed
phase gate are one inseparable public operation, not a component-substitution
protocol.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation is introduced. Source gives no floating-point error bound for the Pauli or phase rotations |
| Precision sensitivity | statevector agreement depends on active simulator precision and must be compared up to global phase using the predeclared fp64/fp32 tolerances |
| Unsupported inputs | non-adjacent pairs do not implement a phase-aware Givens rotation; non-finite angles, non-integral values that do not marshal as annotated, and out-of-range indices have no supported scientific contract |
| Known implementation limitations | the absent outer adjacency guard makes an invalid in-range pair a phase-only operation, not a no-op. `first_orbital` may be irrelevant on that path, but `second_orbital` is always indexed |
| Unsupported, absent, and unverified behavior | non-adjacent phase-aware rotation support, host validation, and status reporting are absent; out-of-range launch behavior, non-finite-angle behavior, and direct standalone execution of this symbol are unverified |

## Resources

| Quantity | Contract |
| --- | --- |
| Valid adjacent operation | metrics/units: source-level logical calls; abstraction: pre-transpilation device-kernel operations; assumptions: finite angles and an adjacent in-range pair; status: exact structural counts of **2** two-qubit `exp_pauli` calls plus **1** single-qubit `rz`; controls: adjacency and the unconditional final phase; limitations: not transpiled gates, depth, runtime, or hardware cost |
| In-range non-adjacent operation | metrics/units: source-level logical calls; abstraction: source kernel; assumptions: valid `second_orbital` and `abs(f-s) != 1`; status: exact structural count of **0** `exp_pauli` and **1** `rz`; controls: delegated adjacency guards plus unconditional `rz` |
| Ancillas, measurements, and classical storage | metrics/units: allocated qubits and measurement operations; abstraction: source kernel; assumptions: all inputs; status: exact structural count of **0** for each; controls: none |

No resource estimator exists for this symbol. Counts above are structural facts
derived from historically reviewed source under the stated assumptions.
Sequential composition adds logical call counts; no supported rule converts
them into transpiled depth, target cost, runtime, or memory.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | use the analytic one-particle map above and remove one global phase before comparison. Independently, the dense-minor oracle in `tests/python/test_stateprep_givens.py:65-80` gives amplitudes `[cos(theta), exp(i*phase) sin(theta)]` for the two-orbital, one-electron coefficient matrix at `:200-212`. No repository test calls this public symbol directly |
| Invariants | unit norm, particle number, relative phase on second-orbital occupation, real-kernel equivalence at `phase=0`, and orientation consistency for reversed adjacent indices |
| Representative cases | proposed direct cases: `(theta,phase)=(0.37,0.73)`, `phase=0`, and both adjacent orientations. Committed transitive cases include the analytic complex one-electron state, the relative-phase/sign state, random complex `4 x 2` and `5 x 3` determinants, and zero-phase agreement with the real flattened kernel |
| Predeclared tolerances | after global-phase alignment, compare statevectors with `rtol=0` and `atol=1e-12` on fp64 or `5e-5` on fp32, selected at call time by `np.dtype(cudaq.complex())`; structural counts and guard predicates are exact |
| Expected failure/adversarial case | an in-range non-adjacent pair must still change a superposition through `rz(phase)`; it must not be described as a whole-kernel no-op. An out-of-range second index must be reported unsupported/unverified rather than promised to raise |
| Runnable example or usage test | the source-derived standalone example below prepares one electron, aligns global phase, and compares against the analytic amplitudes. It was authored here but not run. The repository's transitive runnable suite is `tests/python/test_stateprep_givens.py` |
| Execution record | **Deferred:** no successful command was run for this record in the declared dependency range. Resolution requires recording command, date, package/CUDA-Q version, target, precision, and result after running the standalone case and the cited suite |
| Evidence status per claim | signatures, gate order, weak guard, and call counts are `derived` from source inspected during the historical last review; the cited tests and docs are `derived` evidence from that review; this record is `unexecuted`, with compilation, numerical validation, and measurement `unverified` |

```python
import numpy as np
import cudaq
from cudaq_algorithms import stateprep

@cudaq.kernel
def one_phase_rotation(theta: float, phase: float):
    q = cudaq.qvector(2)
    x(q[0])
    stateprep.phase_givens_rotation(q, theta, phase, 0, 1)

theta, phase = 0.37, 0.73
actual = np.asarray(cudaq.get_state(one_phase_rotation, theta, phase))
expected = np.array(
    [0.0, np.cos(theta), np.exp(1.0j * phase) * np.sin(theta), 0.0],
    complex,
)
global_phase = actual[1] / expected[1]
global_phase /= abs(global_phase)
atol = 5.0e-5 if np.dtype(cudaq.complex()) == np.dtype(np.complex64) else 1.0e-12
np.testing.assert_allclose(actual, global_phase * expected, rtol=0.0, atol=atol)
```

## Evaluation coverage

| Field | Mapping |
| --- | --- |
| Positive selection/application | `state-preparation-givens-device-boundary` selects this raw kernel for a caller-owned register and distinguishes it from the real kernel and validated factory |
| Convention or misconception | the same case requires the post-rotation `exp(+i*phase*n_second)` convention and the fact that the phase still executes when the real rotation no-ops |
| Capability composition | Not applicable: the eval must not claim that this multi-argument signature provides the one-register injection capability |
| Invalid/unsupported boundary | the same case covers the non-adjacent phase-only path and the absence of bounds or finite-angle validation |
| Negative activation | Not applicable: no dedicated negative-activation case targets this symbol |
| Eval status | **authored**; one manual with-skill smoke attempt passed for `state-preparation-givens-device-boundary` on 2026-09-10. No baseline or formal repeated arm has run |

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | `_givens.py` places the surrounding complex determinant construction in the adjacent Givens networks of Jiang et al., *Phys. Rev. Applied* 9, 044036 (2018), Sec. III, and Kivlichan et al., *Phys. Rev. Lett.* 120, 110501 (2018). Those citations are source-reported and were not independently revalidated in this task |
| External package translations | translate the real Givens sign as described in [the real-kernel record](state-preparation-kernel-givens-rotation.md), then translate a package's phase-gate convention to `exp(+i*phase*n_second)` up to global phase. No other external-package mapping was checked |
| Known semantic differences | CUDA-Q `rz(phase)` carries a state-independent global phase relative to the number-phase operator. Unlike the real kernel, this wrapper has no whole-operation adjacency guard, so a non-adjacent request remains phase-active |
