# Flattened real Slater-determinant device kernel

Operation + object: **prepare** a **real Slater-determinant
quantum state from flattened Givens data**.

This is a concrete primitive record for the runtime device kernel
`slater_determinant`. It is distinct from host-side
[schedule planning](state-preparation-givens-schedule.md) and from the
[injectable factory](state-preparation-slater-determinant-kernel.md), which
validates and captures a `GivensRotationSchedule`.

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | `cudaq_algorithms.stateprep.slater_determinant`; it is imported by `cudaq_algorithms.stateprep` and listed in `stateprep.__all__` |
| Contract-specific source paths | `python/cudaq_algorithms/stateprep/_givens.py:97-111`; delegated rotation at `:67-85`; flattening helpers at `:341-359`; export surface in `python/cudaq_algorithms/stateprep/__init__.py:25-32,81` |
| Authoritative tests | `tests/python/test_stateprep_givens.py`: `_real_entry` calls this symbol directly at `:42-47`; statevector cases are `:153-198`, particle-number evidence is `:249-259`, the basis/empty-rotation and one-orbital cases are `:315-338`, and factory equivalence is `:569-580`. No committed test directly covers the raw length-mismatch guard |
| Authoritative documentation and runnable examples | `docs/sphinx/guide/state_prep.rst:151-195`; runnable `docs/sphinx/examples/python/givens_slater_determinant.py`, especially `prepare_real` at `:24-29` and `run_case` at `:68-96` |

All committed tests named here are `derived` repository evidence, not fresh
execution or measurement.

## Classification

| Field | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **prepare** + **real Slater-determinant quantum state from flattened Givens data** |
| Kind | quantum operation |
| Routine role | computational: it prepares a complete determinant state when supplied valid flattened schedule data |
| Abstraction level | leaf operation; its fixed calls to `givens_rotation` are implementation details, not capability substitution points |
| Parameterization | runtime: indices, angles, and electron count cross the device-kernel boundary |
| Execution layers | device kernel only; construction and validation of a schedule are separate host operations |
| Input representations | caller-owned `cudaq.qview`; flat index and angle lists; scalar electron count |
| Output representations | in-place preparation on the supplied register; no host return value |
| Domain | quantum-chemistry / fermionic state preparation |
| Required dependencies | CUDA-Q kernel language and [the real Givens device kernel](state-preparation-kernel-givens-rotation.md) |
| Optional dependencies | Not applicable: none |
| Exactness | exact for the ordered circuit represented by supported flattened data, subject to floating-point angle synthesis; any schedule-construction threshold is outside this raw kernel |
| Uncertainty | deterministic |
| Method | prepare the contiguous occupation reference, then apply an ordered adjacent real Givens network |

## Scientific contract

- Purpose: prepare a real Slater determinant within a caller-owned CUDA-Q
  kernel when the caller already has device-marshallable flattened schedule
  arrays.
- Mathematical definition: let `N = qubits.size()`,
  `E = num_electrons`, `R = len(angles)`, and
  `(f_i,s_i) = (orbital_indices[2*i], orbital_indices[2*i+1])`. If the sole
  outer guard `len(orbital_indices) == 2*R` is true, the kernel applies `X` to
  qubits `0, ..., E-1`, producing the contiguous reference
  `|1...1 0...0>`, and then applies

  ```text
  G(theta_0; f_0,s_0), ..., G(theta_(R-1); f_(R-1),s_(R-1))
  ```

  strictly in list order. `G` is the oriented operation in
  [the real Givens record](state-preparation-kernel-givens-rotation.md). If the
  flat-length guard is false, the entire body is skipped: no reference `X`
  gates and no rotations are applied.
- Determinant meaning: when the lists come from a valid real
  `GivensRotationSchedule` built from an orthonormal `N x E` orbital matrix
  `Q`, the amplitude on basis occupation set `S` is `det(Q[S,:])`, up to
  global phase. For arbitrary matching lists, the supported statement is only
  the ordered circuit above; the kernel does not prove that the data encode an
  orthonormal-orbital determinant.
- Why and when to use: use it from a caller-written kernel that already owns
  the register and intentionally supplies all schedule data at runtime.
- When not to use: do not use raw lists when host validation, data capture, or
  the one-register injection signature is required. Use
  `make_givens_rotation_schedule` followed by `slater_determinant_kernel`.
  Use `complex_slater_determinant` for relative and final phases.
- Approximation controls: Not applicable: this kernel exposes no tolerance or
  truncation control. `make_givens_rotation_schedule(..., tolerance=...)` can
  approximate the upstream matrix representation, but that is a separate
  contract.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | Exact source signature: `slater_determinant(qubits: cudaq.qview, orbital_indices: list[int], angles: list[float], num_electrons: int)`; the source has no return annotation and the device operation yields no value |
| Shapes/ranks | `qubits` is rank 1; `angles` has length `R >= 0`; `orbital_indices` must have exactly `2R` entries, two consecutive entries per rotation; `num_electrons` is scalar |
| Dtypes/domains | The supported schedule-derived domain has integer indices, finite real angles, `N > 0`, `1 <= E <= N`, and every pair in range and adjacent. Lists cross as CUDA-Q kernel arguments without host coercion in this routine |
| Units | angles are dimensionless radians; `num_electrons` is a particle count; indices are zero-based spin-orbital labels |
| Ordering/layout | indices and angles are in application order. Pair `i` is exactly `(orbital_indices[2*i], orbital_indices[2*i+1])`. Qubit 0 is least significant; a spin-resolved coefficient matrix must use interleaved alpha-even/beta-odd rows; see `../conventions.md` |
| Normalization | the flattened lists have no normalization. The determinant claim requires an upstream orthonormal coefficient matrix or otherwise valid schedule. The circuit itself is unitary and norm-preserving |
| Required mathematical properties | exact `2:1` index/angle length relation; supported electron count; finite angles; every pair distinct, adjacent, and in range; an all-zero input register for the state-preparation interpretation |
| Validation and rejection behavior | the **only** device guard is `len(orbital_indices) == 2*len(angles)`. If false, the whole operation is a silent no-op. If true, electron count, bounds, adjacency, and finiteness are unchecked. An in-range non-adjacent pair silently skips that rotation **after** the reference and other rotations may already have executed. Out-of-range indexing has unverified compile/launch behavior. No host exception or status is promised |

## Outputs

| Field | Contract |
| --- | --- |
| Return type or emitted kernel signature | no return value; this symbol is itself a four-argument `@cudaq.kernel`, not a kernel factory |
| Mathematical meaning | on `|0...0>` with supported schedule-derived data, the real Slater determinant stated above; with arbitrary supported flat data, the listed ordered unitary applied to the contiguous `E`-particle reference; with a false outer guard, the input register is unchanged |
| Shape/register geometry | register width remains `N`; the reference touches qubits `0:E`, and each valid rotation touches one adjacent two-qubit slice; no ancilla is allocated |
| Normalization, sign, and phase | norm and particle number are preserved on supported data. Determinant amplitudes follow the package's real Givens sign convention and are compared up to global phase |
| Observable or measurement interpretation | Not applicable: no observable is returned and no measurement occurs |
| Error/status information | Not applicable: the device kernel has no error or status channel |

## Capabilities and composition

Not applicable: this raw four-argument kernel neither provides nor requires a
documented capability ID. Its concrete boundary is a direct call from another
CUDA-Q kernel with the live register and runtime flattened data. It is not the
one-register `cudaq-algorithms.state-preparation.unitary.v1` representation;
the separately documented factory supplies that boundary.

## Composite protocol

Not applicable: leaf operation. Reference occupation and the ordered rotation
loop are fixed parts of this one public kernel, with no lower-level capability
substitution contract.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no algorithmic approximation occurs inside this kernel. Source gives no accumulated floating-point bound for `R` synthesized rotations. Upstream schedule threshold error is not accepted or reported here |
| Precision sensitivity | statevector agreement depends on active simulator precision; use the fp64/fp32 tolerances declared under Validation |
| Unsupported inputs | mismatched flat-list lengths, `E` outside the supported schedule domain, non-adjacent or out-of-range pairs, non-finite angles, and a dirty input register for the preparation claim |
| Known implementation limitations | the one length guard is a silent whole-kernel no-op. With matching lengths, lower-level non-adjacent pairs skip individually and malformed data can therefore produce a plausible but unintended state after partial work |
| Unsupported, absent, and unverified behavior | host rejection, a device status channel, non-adjacent synthesis, and automatic determinant-validity checks are absent; out-of-range launch behavior, non-finite-angle behavior, dirty-register preparation semantics, and direct malformed-list tests are unverified |

## Resources

| Quantity | Contract |
| --- | --- |
| Reference bit flips | metric/unit: source-level logical `X` calls; abstraction: pre-transpilation device operations; assumptions: true outer guard and supported `1 <= E <= N`; status: exact structural count `E`; controls: `num_electrons`; limitations: not a transpiled gate or depth measurement |
| Givens rotations and Pauli exponentials | metrics/units: logical Givens calls and source-level `exp_pauli` calls; abstraction: pre-transpilation device operations; assumptions: true guard and `R` supported adjacent in-range pairs; status: exact counts `R` and `2R`; controls: `len(angles)`; limitations: non-adjacent pairs reduce the actually applied Pauli count and out-of-range behavior is unverified |
| Ancillas, phase rotations, measurements, and classical storage | metrics/units: allocated qubits and source operations; abstraction: source kernel; assumptions: supported inputs; status: exact structural count of **0** for each |
| False-guard path | metric/unit: applied logical operations; abstraction: source kernel; assumptions: `len(orbital_indices) != 2*len(angles)`; status: exact structural count of **0** for all operations |

No estimator accepts this raw signature. The related
[`estimate_givens_resources`](state-preparation-resources-givens.md) consumes a
validated schedule and reports rotation proxies; it does not validate these
runtime lists or report the reference `X` count. For repeated valid calls,
source-level operation counts add. No supported rule converts them into
transpiled depth, target cost, runtime, or memory.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracle | dense minors: for every `E`-element occupation set `S`, set amplitude `det(Q[S,:])` (`tests/python/test_stateprep_givens.py:65-80`). A structurally independent spot check builds dense Jordan-Wigner creation operators and applies the occupied modes to vacuum (`:83-109`, compared at `:262-274`) |
| Invariants | unit norm, exactly `E` occupied qubits on every supported basis component, real-Givens sign convention, application order, and the basis-determinant result when `R=0` |
| Representative cases | analytic `2 x 1`; random real `4 x 2` and `5 x 3`; signed localized `4 x 2`; basis determinant with zero rotations; one orbital/one electron. These are committed cases in `tests/python/test_stateprep_givens.py:153-198,249-285,315-338` |
| Predeclared tolerances | global-phase-aligned statevectors use `rtol=0` and `atol=1e-12` on fp64 or `5e-5` on fp32, selected at call time by `np.dtype(cudaq.complex())`; the same-simulator real/complex comparison and pure NumPy oracle agreement use `atol=1e-12`; structural predicates are exact |
| Expected failure/adversarial case | a `2R` length mismatch must leave the entire register unchanged; matching lengths with one in-range non-adjacent pair must still prepare the reference and skip only that rotation. These behaviors are source-derived, but no committed direct test pins them |
| Runnable example or usage test | `python3 docs/sphinx/examples/python/givens_slater_determinant.py` runs real and complex raw-kernel paths against dense minors. The smaller real-only usage below follows the same entry-kernel pattern; neither was run for this record |
| Evidence status per claim | signature, ordering, guard, and structural resources are `derived` from source cited in the repository; cited tests/docs are `derived` evidence from the cited assertions; this record is `unexecuted`, with compilation, numerical validation, and measurement `unverified` |

```python
import numpy as np
import cudaq
from cudaq_algorithms import stateprep

@cudaq.kernel
def prepare_real(num_orbitals: int, indices: list[int], angles: list[float],
                 num_electrons: int):
    q = cudaq.qvector(num_orbitals)
    stateprep.slater_determinant(q, indices, angles, num_electrons)

theta = 0.37
q_matrix = np.array([[np.cos(theta)], [np.sin(theta)]])
schedule = stateprep.make_givens_rotation_schedule(q_matrix)
actual = np.asarray(cudaq.get_state(
    prepare_real,
    schedule.num_spin_orbitals,
    stateprep.get_givens_rotation_indices(schedule),
    stateprep.get_givens_rotation_angles(schedule),
    schedule.num_electrons,
))
expected = np.array([0.0, np.cos(theta), np.sin(theta), 0.0], complex)
atol = 5.0e-5 if np.dtype(cudaq.complex()) == np.dtype(np.complex64) else 1.0e-12
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
```

## External alignment

| Field | Alignment |
| --- | --- |
| Literature conventions | `_givens.py` states that schedule inversion follows Jiang et al., *Phys. Rev. Applied* 9, 044036 (2018), Sec. III, with the adjacent linear-depth scheduling of Kivlichan et al., *Phys. Rev. Lett.* 120, 110501 (2018). The citations were not independently revalidated in this task |
| External package translations | **Deferred:** no checked external-package flattened-schedule mapping exists in the cited source. Before translating, align orbital-row ordering, ordered pair orientation, angle sign, circuit application order, and basis indexing |
| Known semantic differences | this package expects two flat indices per angle in **application order**, starts from contiguous occupation of qubits `0:E`, and uses qubit 0 as the least-significant basis bit. A producer using elimination order, reversed Givens sign, or blocked spin ordering requires explicit conversion |
