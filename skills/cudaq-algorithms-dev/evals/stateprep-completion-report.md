# State-preparation scientific-coverage completion

Implemented the three independently graded state-preparation groups requested
by `stateprep-completion-task.md`. The implementation is confined to
`e2e/extended_stateprep.py` and its focused test file. No package source,
operational skill, shared evaluator, registry, historical result, YAML, CI, or
Git state was changed.

## Cases and narrow evidence scopes

| Case | Exact uncovered records | Independent outcome checks |
| --- | --- | --- |
| `reference_excitation_primitives` | `state-preparation-kernel-hartree-fock`, `state-preparation-kernel-hartree-fock-occupation`, `state-preparation-kernel-single-excitation`, `state-preparation-kernel-double-excitation`, `state-preparation-resources-hartree-fock`, `state-preparation-resources-hartree-fock-occupation` | Four direct compiled device paths return full statevectors and particle numbers. Independent occupation-basis ladder matrices pin the single/double half-angle, Jordan-Wigner parity, pair canonicalization, and exactly-one-descending sign. Public variant 0 places an occupied spectator between the single-excitation endpoints (`occupied=[0,1]`, `p=0`, `q=3`), so removing the Jordan-Wigner parity string flips a transported amplitude and fails the oracle. Exact estimator fields are checked as source-level `X`-call formulas. The held-out input uses the supported open-shell occupation `[0,2]` for `(n,e,spin)=(4,2,2)`; this is not described as direct open-shell UCCSD. |
| `grouped_ucc_device` | `state-preparation-kernel-uccgsd`, `state-preparation-kernel-upccgsd`, `state-preparation-kernel-ceo`, `state-preparation-kernel-fixed-parameter-ucc`, `state-preparation-resources-fixed-parameter-ucc` | Four separate direct device calls start from the same two-electron reference and return absolute-phase statevectors and particle numbers. NumPy/SciPy references construct fermionic UCCGSD/UpCCGSD generators, qubit-ladder CEO generators, and a custom two-group number-preserving fixed-UCC product independently, then apply the strict ordered `exp(+i theta cP)` factors. Exact group-length formulas pin the estimator fields. |
| `givens_raw_device` | `state-preparation-kernel-givens-rotation`, `state-preparation-kernel-phase-givens-rotation`, `state-preparation-kernel-slater-determinant`, `state-preparation-resources-givens` | Direct compiled real and phase-aware rotations are checked against analytic one-particle maps. The phase-aware field retains the raw `rz` absolute phase. Dense determinant minors check the raw real Slater kernel, with global-phase equivalence allowed only for `slater_state`; norm and particle number remain separately exact. An independent real elimination count drives all schedule-resource formulas. |

Each spec records its exact `feature_ids`, required public paths, defining-module
host calls, and compiled device-kernel names. The union is exactly the 15
previously uncovered state-preparation records, with no duplicates.

## Inputs, grading, and provenance

Each case has a fixed public and held-out input with distinct signed angles or
orbital data. All outputs are finite numeric scalars/arrays in non-object NPZ
form. State comparisons use fixed `atol=2e-9, rtol=2e-9` on CPU `qpp-cpu`
fp64. No tolerance was relaxed during implementation. The controller-side
expectations import only NumPy/SciPy; no CUDA-Q Algorithms code participates in
the oracle.

API disclosure is symmetric: every traced public entry is named in the task
prompt. Host profiling requires the defining-module helper calls, while device
profiling requires compilation of every raw kernel. Tests also demonstrate
that a wrong statevector fails numerical comparison and an empty device trace
fails required-kernel grading.

Resource results are deliberately described only as exact structural formulas:
logical `X` calls, grouped `exp_pauli` calls, or documented Givens proxies.
They are not transpiled gates, target-aware depth, runtime, memory, or hardware
measurements.

## TDD and verification evidence

Initial red command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspaces/cudaq-algorithms/python \
  /tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python -m pytest -q \
  -p no:cacheprovider \
  skills/cudaq-algorithms/evals/e2e/tests/test_extended_stateprep.py
```

Result before implementation: eight errors and one skipped runtime check,
all caused by the intended `ModuleNotFoundError: extended_stateprep`.

The later exact-record-mapping test was also observed red before adding
metadata: one failure at missing `spec["feature_ids"]`. Host green after that
addition was `9 passed, 1 skipped in 0.18s`.

The first strict-child run from the already sandboxed outer process produced
no application output and hit the first 180-second limit. Per the implementation
brief, the identical test was rerun with normal outer execution approval; the
child permission profile and scientific contracts were unchanged. That focused
run passed all six applications (three cases times two variants), including
numeric, host-call, compiled-kernel, isolation, and footprint checks:

```text
1 passed, 8 deselected in 47.21s
```

The complete pre-review focused file with the supported runtime then reported
the corrected total (the earlier draft report incorrectly said nine):

```text
10 passed in 47.06s
```

Independent review then demonstrated that both original raw-single fixtures
were insensitive to replacing fermionic annihilators with parity-free qubit
lowering: the statevector difference was exactly zero for both variants. A new
literal/mutation test was observed red against that fixture (`1 failed`), then
variant 0 gained the occupied spectator described above. The literal expected
state is
`cos(theta/2)|0011> + sin(theta/2)|1010>`; the parity-free mutant reverses the
transported sign and differs in norm by more than `0.4`. The focused host-only
run then reported `10 passed, 1 skipped`, and the fresh full real-gold run after
the correction reported:

```text
11 passed in 47.35s
```

No scored model attempts or campaigns were run.

## Boundaries and follow-up

- The raw device kernels have no general device error channel. Evidence uses
  valid, in-range, correctly shaped data and does not claim uniform rejection
  for malformed lists, invalid indices, or non-finite parameters.
- The small four-qubit grouped cases do not establish optimization behavior,
  larger-system scaling, or all possible pools and excitation orderings.
- The real Givens case does not cover the separate complex-Slater kernel.
- Statevectors are simulator evidence. Compiled device provenance proves that
  the public device entries participated, but it is not a physical-QPU run.
- Registry updates must be linked only after a frozen scored campaign passes;
  this implementation and gold preflight alone do not turn the records into
  campaign `scientific_pass` evidence.
- The broader gap audit found that `features.json` named nonexistent
  `cudaq_algorithms.make_trotter_terms`; the registry owner corrected it to
  the implemented public path,
  `cudaq_algorithms.trotter.make_trotter_terms`. All state-preparation public
  paths used here resolve in the supported runtime.
