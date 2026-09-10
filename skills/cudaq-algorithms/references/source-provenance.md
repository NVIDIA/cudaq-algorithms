# Source provenance and freshness

## Authority and last review

- **Operational authority:** current public source and authoritative tests in the
  checked-out repository control API behavior.
- **Historical last-review anchor:**
  `61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03). This is audit
  metadata, not the active API contract or a compatibility guarantee.
- Declared Python: `>=3.11`.
- Declared CUDA-Q dependency: `cudaq >=0.15.0,<0.16`.
- Historical repository CUDA-Q development pin at the last review:
  `b28cf0f6f2d9387f12ca81b6824b8833a38530af`. Runtime behavior must be
  established against an installed, supported package version instead.
- Package/CUDA-Q combination executed for these records: **unverified**.
- SkillEvaluator baseline/with-skill runs: **not run**.
- Manual single-attempt activation/device-boundary smoke checks: see
  [evals/EVAL.md](../evals/EVAL.md); these are not uplift evidence.

The records were inspected during the historical review anchored above. Before
presenting a selected record's API or behavior as current, compare its named
public symbols and tests with current source when those sources are available.
A cited test is repository evidence, not a fresh execution result. If current
source is unavailable, report that freshness cannot be established instead of
treating the historical anchor as current behavior.

## Cross-cutting sources

- Package metadata: `pyproject.toml`, `.cudaq_version`.
- Shared kernels and validation: `python/cudaq_algorithms/common_kernels.py`.
- Scientific conventions: `docs/sphinx/conventions.rst`.
- Default simulator fixture: `tests/python/conftest.py`.
- Dense independent helpers: `tests/python/dense_references.py`.
- Public API index: `docs/sphinx/api/python_api.rst`.

## Family source map

| Family | Public source | Authoritative tests | Runnable examples/docs |
| --- | --- | --- | --- |
| State preparation | `python/cudaq_algorithms/stateprep/` | `test_stateprep_givens.py`, `test_stateprep_hf_ucc.py`, `test_state_prep_injection.py`, `test_stateprep.py`, `test_stateprep_kernels.py` | `guide/state_prep.rst`, `05_state_prep_and_injection.py` |
| Operator pools | `python/cudaq_algorithms/stateprep/_pools.py` | `test_operator_pools.py` | `guide/state_prep.rst` |
| Block encoding / Pauli LCU | `block_encoding.py`, `pauli_lcu.py` | `test_block_encoding_protocol.py`, `test_pauli_lcu.py`, `test_walk_qsvt_orchestration.py` | `guide/block_encodings.rst`, `01_quickstart_block_encoding.py`, `pauli_lcu_demo.py`, `06_bring_your_own_encoding.py` |
| Qubitization | `qubitization.py` | `test_qubitization.py`, `test_walk_qsvt_orchestration.py` | `guide/qubitization_qsvt.rst` |
| QSVT | `qsvt.py` | `test_qsvt.py`, `test_walk_qsvt_orchestration.py`, `test_df_encoding.py` | `02_hamiltonian_simulation.py`, `07_matrix_inversion_qsvt.py`, `hamiltonian_simulation_qsvt.py` |
| Trotter | `trotter.py` | `test_trotter.py` | API docs and tests are the runnable specification |
| Fermion transforms | `python/cudaq_algorithms/fermion/_compilers.py` | `test_fermion.py`, `test_fermion_compilers.py`, `test_jordan_wigner.py` | API docs |
| Chemistry bridges | `chemistry.py` | `test_fcidump.py`, `test_psi4_conversion.py`, `test_df_qsvt_bridge.py` | `guide/preprocessing.rst`, `03_chemistry_to_ground_state.py`, `trotter_chemistry.py` |
| Double factorization | `python/cudaq_algorithms/double_factorization/` | `test_double_factorization.py`, `test_df_encoding.py`, `test_df_qsvt_bridge.py` | `04_double_factorization_and_the_protocol.py`, `double_factorization.py`, `df_compression_to_qsvt.py`, `df_encoding.py`, `df_block_encoding.py`, `benchmarks/double_factorization/` |
| Simulation analysis | `sim_utils.py` | `test_pauli_lcu.py`, `test_qsvt.py`, `test_trotter.py`, `test_df_encoding.py` | examples above that inspect statevectors |

Paths under `docs/sphinx/examples/python/` are examples, not installed public
providers. In particular, double-factorized block encodings are example-only.

## Freshness check

Before implementing from a maintained record:

```bash
git rev-parse HEAD
git diff 61ac072d -- \
  python/cudaq_algorithms tests/python docs/sphinx \
  pyproject.toml .cudaq_version
```

Inspect only the selected symbols and their tests. If public signatures or
scientific assertions changed, follow current source for generated code, report
the drift, and update the maintained record only when that work is in scope. Do
not silently upgrade the lifecycle from `draft` or claim a newly verified
version range.

## Evidence wording

- `historically source-reviewed`: provenance wording for the maintained record;
  it is not a current-task evidence label.
- `source-checked`: reserve for current source/tests/docs inspected in the
  current task, with the checkout or release identified in the result.
- `executed`: use only after a fresh command completes successfully.
- `numerically validated`: use only after agreement with the named independent
  oracle within the declared tolerance.
- `measured`: use only for a quantity collected in the current protocol.

Source comments about speed, old benchmark notes, and committed test assertions
remain derived evidence until rerun. A validation or evaluation result records
the exact source revision, dependency versions, targets, and commands actually
used; it does not inherit the historical anchor unless that was the executed
source.
