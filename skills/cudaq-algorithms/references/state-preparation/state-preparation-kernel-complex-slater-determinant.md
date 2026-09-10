# Flattened complex Slater-determinant device kernel

Status: draft. Operation + object: **prepare** a **complex Slater-determinant
quantum state from flattened phase-aware Givens data**.

This is a concrete primitive record for the runtime device kernel
`complex_slater_determinant`. It is distinct from host-side
[schedule planning](state-preparation-givens-schedule.md) and from the
[injectable factory](state-preparation-slater-determinant-kernel.md), which
validates and captures a `GivensRotationSchedule`.

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.complex_slater_determinant`; it is imported by `cudaq_algorithms.stateprep` and listed in `stateprep.__all__` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_givens.py:114-128`; delegated phase-aware rotation at `:88-94`; flattening helpers at `:341-359`; export surface in `python/cudaq_algorithms/stateprep/__init__.py:25-32,82` |
| Authoritative tests | `tests/python/test_stateprep_givens.py`: `_complex_entry` calls this symbol directly at `:50-57`; statevector cases are `:200-246`; the independent second-quantized spot check is `:262-274`; zero-phase real equivalence is `:288-307`; full filling is `:341-351`; factory equivalence and rotation-free shape are `:583-621`. No committed test directly covers the three raw guards or extra-final-phase behavior |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:151-200`; runnable `docs/sphinx/examples/python/givens_slater_determinant.py`, especially `prepare_complex` at `:32-38` and `run_case` at `:68-96` |
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
| Operation + mathematical object (primary identity) | **prepare** + **complex Slater-determinant quantum state from flattened phase-aware Givens data** |
| Kind | quantum operation |
| Routine role | computational: it prepares a complete determinant state when supplied valid flattened complex schedule data |
| Abstraction level | leaf operation; its fixed calls to `phase_givens_rotation` are implementation details, not capability substitution points |
| Parameterization | runtime: indices, rotation angles, relative phases, final phases, and electron count cross the device-kernel boundary |
| Execution layers | device kernel only; construction and validation of a schedule are separate host operations |
| Input representations | caller-owned `cudaq.qview`; flat index, angle, per-rotation phase, and final-phase lists; scalar electron count |
| Output representations | in-place preparation on the supplied register; no host return value |
| Domain | quantum-chemistry / complex fermionic state preparation |
| Required dependencies | CUDA-Q kernel language and [the phase-aware Givens device kernel](state-preparation-kernel-phase-givens-rotation.md) |
| Optional dependencies | Not applicable: none |
| Exactness | exact for the ordered circuit represented by supported flattened data, subject to floating-point angle synthesis; any schedule-construction threshold is outside this raw kernel |
| Uncertainty | deterministic |
| Method | prepare the contiguous occupation reference, apply one final number phase per electron, then apply an ordered phase-aware adjacent Givens network |

## Scientific contract

- Purpose: prepare a complex Slater determinant within a caller-owned CUDA-Q
  kernel when the caller already has device-marshallable flattened schedule
  arrays.
- Mathematical definition: let `N = qubits.size()`,
  `E = num_electrons`, `R = len(angles)`,
  `(f_i,s_i) = (orbital_indices[2*i], orbital_indices[2*i+1])`, and let
  `phi_i = phases[i]`. When all three outer conditions hold,

  ```text
  len(orbital_indices) == 2R
  len(phases) == R
  len(final_phases) >= E,
  ```

  the kernel applies `X` to qubits `0, ..., E-1`, then applies
  `rz(final_phases[i])` to occupied qubit `i` for `i = 0, ..., E-1`, then
  applies `phase_givens_rotation(theta_i, phi_i, f_i, s_i)` strictly in list
  order. Up to one state-independent global factor per `rz`, this is the
  ordered phase-aware Givens network acting on a contiguous `E`-particle
  reference with final number phases.
- Guard meaning: if **any** of the three conditions is false, the whole body is
  skipped — no reference occupation, no final phases, and no rotations. Extra
  `final_phases` beyond the first `E` are ignored. If an individual pair is
  non-adjacent after the outer guard passes, its real Givens component no-ops
  but its `rz(phi_i)` on `s_i` still executes.
- Determinant meaning: when the lists come from a valid complex
  `GivensRotationSchedule` built from an orthonormal `N x E` orbital matrix
  `Q`, the amplitude on basis occupation set `S` is `det(Q[S,:])`, up to
  global phase. For arbitrary matching lists, the supported statement is only
  the ordered circuit above; the kernel does not establish orthonormality or
  determinant provenance.
- Why and when to use: use it from a caller-written kernel that already owns
  the register and intentionally supplies complex flattened schedule data at
  runtime.
- When not to use: do not use raw lists when host validation, data capture, or
  the one-register injection signature is required. Use
  `make_givens_rotation_schedule` followed by `slater_determinant_kernel`.
  Use `slater_determinant` when all relative and final phases are absent.
- Approximation controls: Not applicable: this kernel exposes no tolerance or
  truncation control. `make_givens_rotation_schedule(..., tolerance=...)` can
  approximate the upstream matrix representation, but that is a separate
  contract.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact source signature: `complex_slater_determinant(qubits: cudaq.qview, orbital_indices: list[int], angles: list[float], phases: list[float], final_phases: list[float], num_electrons: int)`; the source has no return annotation and the device operation yields no value |
| Shapes/ranks | `qubits` is rank 1; `angles` and `phases` have equal length `R >= 0`; `orbital_indices` has exactly `2R` entries; `final_phases` has at least `E` entries; `num_electrons` is scalar |
| Dtypes/domains | The supported schedule-derived domain has integer indices, finite real angles/phases, `N > 0`, `1 <= E <= N`, and every pair in range and adjacent. Lists cross as CUDA-Q kernel arguments without host coercion in this routine |
| Units | all angles and phases are dimensionless radians; `num_electrons` is a particle count; indices are zero-based spin-orbital labels |
| Ordering/layout | all per-rotation arrays are in application order; pair `i`, `angles[i]`, and `phases[i]` form one record. Only `final_phases[0:E]` are used, before the rotations. Qubit 0 is least significant and spin-resolved rows are interleaved alpha-even/beta-odd; see `../conventions.md` |
| Normalization | the flattened lists have no normalization. The determinant claim requires an upstream orthonormal coefficient matrix or otherwise valid schedule. The circuit itself is unitary and norm-preserving |
| Required mathematical properties | all three list-shape guards; supported electron count; finite angles and phases; every pair distinct, adjacent, and in range; an all-zero input register for the state-preparation interpretation |
| Validation and rejection behavior | the **only outer validation** is the conjunction of the three list-length predicates. A false predicate makes the whole kernel a silent no-op. With a true guard, electron count, bounds, adjacency, and finiteness are unchecked; extra final phases are accepted and ignored. A non-adjacent pair still applies its phase to `second_orbital`; out-of-range indexing has unverified compile/launch behavior. No host exception or status is promised |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | no return value; this symbol is itself a six-argument `@cudaq.kernel`, not a kernel factory |
| Mathematical meaning | on `|0...0>` with supported schedule-derived data, the complex Slater determinant stated above; with arbitrary supported flat data, the listed ordered unitary applied to the final-phased contiguous reference; with a false outer guard, the input register is unchanged |
| Shape/register geometry | register width remains `N`; the reference/final phases touch qubits `0:E`, and each valid rotation touches one adjacent pair plus its second-orbital phase target; no ancilla is allocated |
| Normalization, sign, and phase | norm and particle number are preserved on supported data. Relative phases use `exp(+i*phase*n_second)` up to global phase, and final phases use the same number-phase convention on the initially occupied qubits. Raw statevectors must be compared up to global phase |
| Observable or measurement interpretation | Not applicable: no observable is returned and no measurement occurs |
| Error/status information | Not applicable: the device kernel has no error or status channel |

## Capabilities and composition

Not applicable: this raw six-argument kernel neither provides nor requires a
documented capability ID. Its concrete boundary is a direct call from another
CUDA-Q kernel with the live register and runtime flattened data. It is not the
one-register `cudaq-algorithms.state-preparation.unitary.v1` representation;
the separately documented factory supplies that boundary.

## Composite protocol

Not applicable: leaf operation. Reference occupation, final phases, and the
ordered phase-aware rotation loop are fixed parts of this one public kernel,
with no lower-level capability substitution contract.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation occurs inside this kernel. Source gives no accumulated floating-point bound for the `E + R` phase gates and `R` two-mode rotations. Upstream schedule threshold error is not accepted or reported here |
| Precision sensitivity | statevector agreement depends on active simulator precision and global-phase alignment; use the fp64/fp32 tolerances under Validation |
| Unsupported inputs | any false list-shape predicate, `E` outside the supported schedule domain, non-adjacent or out-of-range pairs, non-finite angles/phases, and a dirty input register for the preparation claim |
| Known implementation limitations | the length conjunction is a silent whole-kernel no-op; extra final phases are silently ignored. With matching lengths, a non-adjacent rotation is only partially skipped because its second-orbital phase still applies, so malformed data can produce a plausible unintended state after partial work |
| Unsupported, absent, and unverified behavior | host rejection, device status, non-adjacent synthesis, and automatic determinant-validity checks are absent; out-of-range launch behavior, non-finite-angle behavior, dirty-register preparation semantics, and direct guard/extra-phase tests are unverified |

## Resources

| Quantity | Contract |
| --- | --- |
| Reference bit flips | metric/unit: source-level logical `X` calls; abstraction: pre-transpilation device operations; assumptions: true outer guard and supported `1 <= E <= N`; status: exact structural count `E`; controls: `num_electrons`; limitations: not transpiled gate count or depth |
| Phase rotations | metric/unit: source-level logical `rz` calls; abstraction: pre-transpilation device operations; assumptions: true guard and in-range final/second targets; status: exact structural count `E + R`, including `E` final phases and one phase per requested rotation, even if a pair is non-adjacent; controls: `E` and `R`; limitations: not transpiled gates, depth, runtime, or hardware cost |
| Givens rotations and Pauli exponentials | metrics/units: requested phase-Givens calls and applied logical `exp_pauli` calls; abstraction: pre-transpilation device operations; assumptions: true guard, with `A` adjacent in-range pairs among `R`; status: exact requested-call count `R` and applied Pauli count `2A` (`2R` for fully supported data); controls: pair adjacency; limitations: out-of-range behavior is unverified |
| Ancillas, measurements, and classical storage | metrics/units: allocated qubits and source operations; abstraction: source kernel; assumptions: supported inputs; status: exact structural count of **0** for each |
| False-guard path | metric/unit: applied logical operations; abstraction: source kernel; assumptions: any one of the three list predicates is false; status: exact structural count of **0** for all operations |

No estimator accepts this raw signature. The related
[`estimate_givens_resources`](state-preparation-resources-givens.md) consumes a
validated schedule and reports rotation/phase proxies; it does not validate
these runtime lists or report the reference `X` count. For repeated valid
calls, source-level operation counts add. No supported rule converts them into
transpiled depth, target cost, runtime, or memory.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | dense minors: for every `E`-element occupation set `S`, set amplitude `det(Q[S,:])` (`tests/python/test_stateprep_givens.py:65-80`). A structurally independent spot check builds dense Jordan-Wigner creation operators and applies the occupied modes to vacuum (`:83-109`, compared at `:262-274`) |
| Invariants | unit norm, particle number, phase-aware sign convention, paired-list application order, final phases before rotations, and exact reduction to the real flattened kernel when all phases are zero |
| Representative cases | analytic complex `2 x 1`; relative-phase/sign `3 x 2`; random complex `4 x 2` and `5 x 3`; full filling `3 x 3`; rotation-free complex basis state; zero-phase agreement with the real path. These are committed cases in `tests/python/test_stateprep_givens.py:200-246,288-307,341-351,614-621` |
| Predeclared tolerances | global-phase-aligned statevectors use `rtol=0` and `atol=1e-12` on fp64 or `5e-5` on fp32, selected at call time by `np.dtype(cudaq.complex())`; same-simulator real/complex and factory/raw comparisons use `atol=1e-12`; the complex64 list case uses `max(active_tolerance,1e-7)`; structural predicates are exact |
| Expected failure/adversarial case | independently falsify each of the three outer predicates and require identity; supply extra final phases and require them to be ignored; then supply one in-range non-adjacent pair and require its phase, but not its real rotation, to execute. These are source-derived and represented in the authored eval, but no committed direct test pins them |
| Runnable example or usage test | `python3 docs/sphinx/examples/python/givens_slater_determinant.py` runs both raw paths against dense minors. The smaller complex-only usage below follows the same entry-kernel pattern; neither was run for this record |
| Execution record | **Deferred:** no successful command was run for this record in the declared dependency range. Resolution requires recording command, date, package/CUDA-Q version, target, precision, and result for the example and `pytest -q tests/python/test_stateprep_givens.py` |
| Evidence status per claim | signature, ordering, guards, extra-phase behavior, and structural resources are `derived` from source inspected during the historical last review; cited tests/docs are `derived` evidence from that review; this record is `unexecuted`, with compilation, numerical validation, and measurement `unverified` |

```python
import numpy as np
import cudaq
from cudaq_algorithms import stateprep

@cudaq.kernel
def prepare_complex(num_orbitals: int, indices: list[int],
                    angles: list[float], phases: list[float],
                    final_phases: list[float], num_electrons: int):
    q = cudaq.qvector(num_orbitals)
    stateprep.complex_slater_determinant(
        q, indices, angles, phases, final_phases, num_electrons
    )

theta, phase = 0.37, 0.73
q_matrix = np.array([[np.cos(theta)],
                     [np.exp(1.0j * phase) * np.sin(theta)]])
schedule = stateprep.make_givens_rotation_schedule(q_matrix)
actual = np.asarray(cudaq.get_state(
    prepare_complex,
    schedule.num_spin_orbitals,
    stateprep.get_givens_rotation_indices(schedule),
    stateprep.get_givens_rotation_angles(schedule),
    stateprep.get_givens_rotation_phases(schedule),
    list(schedule.final_phases),
    schedule.num_electrons,
))
expected = np.array(
    [0.0, np.cos(theta), np.exp(1.0j * phase) * np.sin(theta), 0.0],
    complex,
)
pivot = int(np.argmax(np.abs(expected)))
global_phase = actual[pivot] / expected[pivot]
global_phase /= abs(global_phase)
atol = 5.0e-5 if np.dtype(cudaq.complex()) == np.dtype(np.complex64) else 1.0e-12
np.testing.assert_allclose(actual, global_phase * expected, rtol=0.0, atol=atol)
```

## Evaluation coverage

| Field | Mapping |
| --- | --- |
| Positive selection/application | `state-preparation-givens-device-boundary` selects this raw flattened kernel for a caller-owned register and distinguishes it from real flattening, schedule data, and the injectable factory |
| Convention or misconception | the same case requires final phases before the ordered rotations, one relative phase per angle, and the phase-active behavior of a non-adjacent pair |
| Capability composition | Not applicable: the eval must not claim that this six-argument signature provides the one-register injection capability; it should route validated injection to `make_givens_rotation_schedule` plus `slater_determinant_kernel` |
| Invalid/unsupported boundary | the same case covers all three outer predicates, silent whole-kernel no-op, ignored extra final phases, and absent count/bounds/adjacency/finiteness validation |
| Negative activation | Not applicable: no dedicated negative-activation case targets this symbol |
| Eval status | **authored**; one manual with-skill smoke attempt passed for `state-preparation-givens-device-boundary` on 2026-09-10. No baseline or formal repeated arm has run |

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | `_givens.py` states that schedule inversion follows Jiang et al., *Phys. Rev. Applied* 9, 044036 (2018), Sec. III, with the adjacent linear-depth scheduling of Kivlichan et al., *Phys. Rev. Lett.* 120, 110501 (2018). The citations were not independently revalidated in this task |
| External package translations | **Deferred:** no checked external-package flattened complex-schedule mapping exists in the cited source. Before translating, align orbital-row ordering, pair orientation, real-rotation sign, relative-phase sign/target, final-phase placement, application order, and basis indexing |
| Known semantic differences | this package uses one relative phase per requested rotation and one final phase per initially occupied orbital, implements number phases through `rz` up to global phase, starts from contiguous occupation of qubits `0:E`, and uses qubit 0 as the least-significant basis bit. Extra final phases are ignored only by this raw guard, not by the validated complex schedule contract |
