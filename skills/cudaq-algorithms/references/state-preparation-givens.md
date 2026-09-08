# Slater determinant preparation by Givens rotations

Status: draft. Operation + object: **prepare** a **quantum state**.

This is a **concrete primitive record**: it instantiates each applicable
canonical Primitive-Record heading of `assets/primitive-record-template.md`
exactly once, for one contract — a Slater determinant given by an orthonormal
orbital-coefficient matrix. The shared seam it plugs into (kernel
representation, unitary capability, consumer table, common boundaries) is
[state-preparation.md](state-preparation.md); cross-cutting layout, ownership,
and validation conventions are `conventions.md`. Neither is repeated here.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths: from `cudaq_algorithms.stateprep` —
  `make_givens_rotation_schedule`, `validate_givens_rotation_schedule`,
  `slater_determinant_kernel`, `GivensRotationSchedule`,
  `estimate_givens_resources`.
- Source paths: `python/cudaq_algorithms/stateprep/_givens.py`;
  `python/cudaq_algorithms/stateprep/_kernels.py` for the emitted gates.
- Authoritative tests: `tests/python/test_stateprep_givens.py`;
  `tests/python/test_state_prep_injection.py` for the seam;
  `tests/python/test_stateprep.py`.
- Authoritative documentation: `docs/sphinx/guide/state_prep.rst`,
  `docs/sphinx/conventions.rst`; example
  `docs/sphinx/examples/python/05_state_prep_and_injection.py`.
- Package/CUDA-Q versions verified: **unverified.** Declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q
  pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`); no
  test was executed for this record.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft.
- Replacement and migration notes: none. No symbol here is marked deprecated at
  the cited commit.
- **Evidence rule.** Every test named below is *cited repository evidence* — a
  committed assertion read at the cited commit, labeled `derived` — never a
  fresh measurement in this session.

## Classification

- Operation + mathematical object (primary identity): **prepare** + **quantum
  state**. Register ownership is deliberately not part of the identity; the
  source-derived behavior of the seam is in the family Capability Record.
- Kind: quantum operation (kernel factory) with host classical planning, plus a
  resource estimator covered under "Resources" rather than in its own record.
  `Deferred:` a single-token template value for this compound kind, until a
  second family needs the same compound.
- Routine role: computational — one distinct, independently usable task. Role
  follows problem completeness, not execution location, so the host-side
  scheduling and validation steps are computational too.
- Abstraction level: leaf operation.
- Parameterization: construction-time. The schedule and the register width are
  baked in when the kernel is minted.
- Execution layers: host preprocessing and validation, kernel factory, device
  kernel.
- Input representations: orbital-coefficient matrix; `GivensRotationSchedule`.
- Output representations: the one-argument `(qubits: cudaq.qview)` preparation
  kernel of the family Representation Record.
- Domain: `quantum-chemistry` (a tag, not a parallel taxonomy). The capability
  it provides is domain-independent.
- Required dependencies: `cudaq >= 0.15.0, < 0.16` and `numpy`
  (`pyproject.toml`), both used on the host path. A simulator or hardware target
  is needed to *run* an emitted kernel, not to build one.
- Optional dependencies: none.

Provisional metadata — record the value, do not route on it:

- Exactness: exact for the emitted circuit contract, subject to host input
  tolerances and floating-point synthesis of the rotation angles.
- Uncertainty: deterministic.
- Method: direct.

## Scientific contract

- Purpose: emit a one-argument device kernel that prepares a single Slater
  determinant on a register in `|0...0>`.
- Mathematical definition: for an orthonormal orbital-coefficient matrix `Q` of
  shape `(num_spin_orbitals, num_electrons)`, the prepared state is the Slater
  determinant whose amplitude on the computational-basis state `|S>` with
  occupied set `S` (`|S| = num_electrons`) is `det(Q[S, :])`, up to one global
  phase (`_givens.py:3-9`).
- Why and when to use: when the target state is a single Slater determinant and
  the caller holds its orbital-coefficient matrix.
- When not to use: when the target is a Hartree-Fock reference plus a UCC
  product at known amplitudes — that is
  [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md), a different
  contract, not a variant of this one; when the amplitudes still have to be
  chosen or optimized (a consumer workflow, out of scope); or when a
  multi-argument device kernel is wanted, a different representation.
- Approximation controls: none. The emitted circuit is exact for the supplied
  matrix; `tolerance` is a host validation knob, not an approximation control.

## Inputs

Two steps — the matrix does not go into the kernel factory:

```python
from cudaq_algorithms import stateprep
schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients,
                                                   tolerance=1.0e-12)
prep = stateprep.slater_determinant_kernel(schedule)   # (qubits: cudaq.qview)
```

- Arguments: `make_givens_rotation_schedule(orbital_coefficients,
  tolerance=1.0e-12)` (`_givens.py:224-225`), then
  `slater_determinant_kernel(schedule)` (`_givens.py:367`).
  `validate_givens_rotation_schedule` checks a hand-built schedule.
- Shapes/ranks: `orbital_coefficients` is a rank-2
  `(num_spin_orbitals, num_electrons)` matrix — a NumPy array or nested lists
  — with `num_electrons >= 1` and `num_electrons <= num_spin_orbitals`.
- Dtypes/domains: real or complex. An object exposing a dtype dispatches on it,
  so a complex dtype routes the complex path even when every entry is real. A
  nested Python list routes the complex path when any entry is a Python
  `complex` **or** an `np.complexfloating` — `np.complex64` is not a `complex`
  subclass, and the source calls that case out explicitly
  (`_givens.py:167-183`; asserted at `test_stateprep_givens.py:373-420`).
- Units: dimensionless amplitudes; `tolerance` is dimensionless.
- Ordering/layout: rows are spin orbitals in the package's interleaved
  alpha-even / beta-odd order, with qubit 0 least significant
  (`conventions.md`).
- Normalization: the columns are the occupied orbitals and must be orthonormal.
- Required mathematical properties: column orthonormality within the tolerances
  below.
- Validation and rejection behavior: host-side at factory time, **six**
  conditions, all `ValueError` (`_givens.py:186-215`; all six asserted at
  `test_stateprep_givens.py:535-550`):

| # | Rejected condition |
| --- | --- |
| 1 | empty input |
| 2 | no occupied column |
| 3 | `num_electrons > num_spin_orbitals` |
| 4 | non-rectangular rows |
| 5 | a column not normalized to within `100 * tolerance` |
| 6 | a pair of columns not orthogonal to within `100 * tolerance` |

  Caller obligation with no detection (`assumed`): the interleaved row ordering,
  so the determinant composes with occupations and pools built at the same spin.

## Outputs

- Return type or emitted kernel signature: `slater_determinant_kernel` returns
  a compiled CUDA-Q kernel whose only parameter is `(qubits: cudaq.qview)`.
  `make_givens_rotation_schedule` returns the intermediate
  `GivensRotationSchedule` dataclass, **not** a kernel.
- Mathematical meaning: applied to a `|0...0>` register of the baked-in width,
  the kernel produces the determinant defined above.
- Shape/register geometry: width is fixed at factory time to
  `schedule.num_spin_orbitals`. No ancilla, signal, or control register is
  allocated.
- Normalization, sign, and phase: the emitted state is normalized, and the
  determinant is reproduced **up to one global phase**.
- Observable or measurement interpretation: none. The kernel prepares a state,
  returns nothing, and performs no measurement.
- Error/status information: none at the device boundary — no return value, no
  status channel, which is why all validation is host-side (`conventions.md`).

## Capabilities and composition

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`
- Direction: provides
- Owning family record: [state-preparation.md](state-preparation.md), whose
  Capability Record carries the invariants, the four-module consumer table, and
  the register-ownership scope limit. Match the capability ID, the
  `(qubits: cudaq.qview)` boundary representation, the exact register width, and
  the layout conventions before injecting; support for a consumer that record
  omits must be checked against its source, never inferred.
- Requires: nothing. This provider consumes no other family's capability.

## Composite protocol

Not applicable: `Abstraction level` is `leaf operation`; the template omits it.

## Accuracy and limitations

- Error behavior or bounds: no approximation is introduced. Deviations come from
  host input tolerances and floating-point synthesis of the rotation angles. No
  error bound is stated in source.
- Precision sensitivity: the emitted circuit is precision-agnostic; observed
  agreement with a dense reference depends on the active simulator precision
  (see "Validation"). Matrix checks scale as `100 * tolerance` from a `1.0e-12`
  default, so a stricter check needs a smaller `tolerance`.
- Unsupported inputs: the six rejected conditions tabulated under "Inputs".
- Known implementation limitations: the interleaved row ordering is a caller
  obligation with no detection, so a blocked-ordering matrix silently prepares a
  determinant that no longer matches a pool built at the same spin.
- Unsupported versus unverified: the shared boundaries — controlled, adjoint,
  measurement-assisted, dirty input register, width-mismatch failure detail,
  foreign consumer injection, and global phase under control — are stated once
  with their labels under "Shared unsupported and unverified boundaries" in
  [state-preparation.md](state-preparation.md). This provider adds no
  unsupported-versus-unverified boundary of its own, but the global-phase item
  bites here specifically: this contract holds only up to a global phase, and no
  source characterizes such a preparation later placed under control.

## Resources

`estimate_givens_resources` returns the frozen dataclass
`GivensResourceEstimate`. Its docstring is the strictest statement in the
package: the proxies are "decomposition-independent **upper bounds**, not
transpiled gate counts" (`_givens.py:439-440`). They are therefore not
transpiled-gate figures, not post-synthesis depth, not runtime, and not memory.
No composition rule with another estimator is stated (`assumed` absent), and no
timing or scaling evidence exists — the linear-depth scheduling citation under
"External alignment" is literature, not a measurement.

`Deferred:` the per-quantity resource contract (metric, unit, abstraction level,
execution assumptions, exact/bounded/estimated status, controlling parameters,
confidence, composition rule), until the resource-metric vocabulary is agreed.

## Validation

- Independent oracles — two, mathematically independent of each other, both
  committed in the repository and cited rather than executed: dense minor
  expansion (`det(Q[S, :])`) **and** a second-quantized construction applying
  dense Jordan-Wigner creation operators to the vacuum, using no determinant
  identity. Both are in `test_stateprep_givens.py:65-109`, compared up to global
  phase. The seam itself is covered by `test_state_prep_injection.py`.
- Invariants: every amplitude above `1e-12` has popcount `num_electrons`
  (`test_stateprep_givens.py:249-259`).
- Representative cases: real and complex orbital-coefficient matrices, including
  the nested-list and `np.complex64` dispatch cases.
- Predeclared tolerances: `test_stateprep_givens.py:23-34` selects `1e-12`
  (fp64) or `5e-5` (fp32) from the active simulator precision, gating on
  `np.dtype(cudaq.complex()) == np.complex64`. Do not generalize that to the
  injection suite, which does not gate on precision — see the tolerance
  note in [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md).
- Expected failure/adversarial cases: a non-adjacent Givens rotation, and the
  six orbital-coefficient validation failures
  (`test_stateprep_givens.py:535-550`).
- Reference results: none; each oracle is constructed inside the cited test.
- Evidence status per claim: `derived` from the cited source or committed test
  assertion, unless labeled `assumed` or `unverified` in place. Nothing here is
  `measured`.

## Evaluation coverage

| Case id in `evals/evals.json` | Sections that support it |
| --- | --- |
| `state-preparation-provider-selection` | Scientific contract (the determinant target); Inputs (two-step flow, ordering, six validations); Outputs; Identity and provenance (unverified versions) |

The other four `state-preparation-*` cases are answered from
[state-preparation.md](state-preparation.md) and, for the UCC amplitude
boundary, [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md). No case
has been run with or without the skill, so this is declared coverage, not
validated coverage; `evals/EVAL.md` owns the procedure.

## External alignment

- Literature conventions: the source cites Jiang et al., Phys. Rev. Applied 9,
  044036 (2018) for the Givens network and Kivlichan et al., Phys. Rev. Lett.
  120, 110501 (2018) for the nearest-neighbour scheduling (`_givens.py:11-17`).
  Both literature claims are `unverified` here: this record neither re-derives
  them nor compares against the papers.
- External package translations: `Deferred:` until a task needs one and it can
  be checked against that package's own documentation. Nothing in the
  repository translates this input to an external chemistry package.
- Known semantic differences: none established against an external package.
