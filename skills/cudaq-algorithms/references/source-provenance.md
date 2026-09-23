# Source lookup and freshness

## Current authority

Current public source and authoritative tests in the checked-out repository
control API behavior. Compare the selected record's named public symbols and
tests with current source before presenting its API or behavior as current.
A cited test is repository evidence, not a fresh execution result. If current
source is unavailable, say that freshness cannot be established.

Declared requirements in `pyproject.toml` are Python `>=3.11` and
`cudaq >=0.15.0,<0.16`; recheck the current file. Runtime compatibility requires
execution against an installed supported version; declarations alone do not
establish a tested combination.

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

Inspect the selected public symbols and authoritative tests in the current
checkout. Follow current source for generated code, report any drift, and
update maintained records only when that work is in scope. Record the exact
revision, dependencies, target, precision, and commands for a fresh run.

## Evidence wording

Use the [validation evidence labels](validation.md#evidence-labels) to distinguish
source inspection, execution, numerical agreement, and measurement.

Source comments about speed, old benchmark notes, and committed test assertions
remain derived evidence until rerun. A validation or evaluation result records
the exact source revision, dependency versions, targets, and commands actually
used; it does not inherit the historical anchor unless that was the executed
source.
