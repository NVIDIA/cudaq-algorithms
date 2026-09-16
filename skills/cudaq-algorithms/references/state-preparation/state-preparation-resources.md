# State-preparation resource estimation — family front door

Status: draft.

Four independently selectable host interfaces describe different input objects
and validation guarantees. Route by the exact public symbol; do not transfer
validation from a preparation factory or one estimator to another.

| Operation + object | Public entry point | Focused record |
| --- | --- | --- |
| estimate / logical operations for a validated Givens schedule | `cudaq_algorithms.stateprep.estimate_givens_resources` | [Givens schedule resources](state-preparation-resources-givens.md) |
| estimate / logical `X` operations for a canonical Hartree-Fock occupation | `cudaq_algorithms.stateprep.estimate_hartree_fock_resources` | [canonical Hartree-Fock resources](state-preparation-resources-hartree-fock.md) |
| estimate / logical `X` operations for an explicit Hartree-Fock occupation | `cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources` | [explicit-occupation Hartree-Fock resources](state-preparation-resources-hartree-fock-occupation.md) |
| estimate / grouped Pauli rotations for fixed-parameter UCC data | `cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources` | [fixed-parameter UCC resources](state-preparation-resources-fixed-parameter-ucc.md) |

All four functions and their frozen result dataclasses live in the
`cudaq_algorithms.stateprep` namespace; none is exported from the package root.
They execute on the host and do not build, compile, transpile, or run a quantum
circuit.

The guide classifies the returned quantities as logical operations before
transpilation, not hardware gate counts. The Givens result is additionally
documented by its source as a set of decomposition-independent upper-bound
proxies. The three Hartree-Fock/UCC estimator docstrings do not make that bound
claim. None of the four reports target-aware decomposition, routed depth,
runtime, memory, T/Toffoli count, measurement, or scientific approximation
error, and no cross-estimator composition rule is documented.

Shared source is `python/cudaq_algorithms/stateprep/_givens.py` and
`python/cudaq_algorithms/stateprep/_hartree_fock.py`; authoritative tests are
`tests/python/test_stateprep_givens.py` and
`tests/python/test_stateprep_hf_ucc.py`. Current public source/tests are
authoritative and must be checked at use time. All records remain lifecycle
`draft` and were historically source-reviewed; review provenance is recorded in
[Source provenance](../source-provenance.md). No compatible-range verification
was completed for all four estimators. The exact narrower task-local smoke
command, selection, environment limitation, and result are recorded centrally in
[EVAL.md](../../evals/EVAL.md); they do not verify this whole resource family
or hardware behavior.
