# Resource Levels

## Resource abstraction levels must not be conflated

- **Convention.** Every packaged resource figure is a count of logical
  operations before transpilation; none is a transpiled gate count, depth,
  runtime, or memory figure. The rule is library-wide. The producing APIs
  and their own wording are:

  | Producer | Figure | Source wording |
  | --- | --- | --- |
  | `stateprep.estimate_givens_resources` -> `GivensResourceEstimate` | `num_exp_pauli_calls`, `num_phase_rotations`, `two_qubit_gate_count_proxy`, `depth_proxy` | "decomposition-independent upper bounds, not transpiled gate counts" |
  | `stateprep.estimate_hartree_fock_resources`, `estimate_hartree_fock_occupation_resources`, `estimate_fixed_parameter_ucc_resources` | `num_x_gates`; `num_pauli_rotations`, `max_pauli_rotations_per_excitation` | docstrings carry no level statement; the Sphinx guide states "logical operations before transpilation, not hardware gate counts" |
  | `Trotter.resources`, `trotter.estimate_trotter_resources` -> `TrotterResourceEstimate` | `pauli_rotations` (exact), `estimated_cx_count` | "`estimated_cx_count` is a decomposition proxy: two CNOTs per additional non-identity Pauli in each rotation" |
  | `primitives.QROM.toffoli_count`, `UnaryIterationKernels.toffoli_count`, `AliasSamplingPrepare.qrom.toffoli_count`; the arithmetic kernels' documented `2n - 2` / `2n - 1` Toffoli formulas | exact logical two-controlled-`x` counts of the strictly unitary emitted circuit, pinned by the repo tests against `cudaq.estimate_resources(...).count_controls("x", 2)` (`test_primitives_resources.py`, `test_primitives_arithmetic.py`, `test_primitives_alias_sampling.py`) | counted logical gates: not a T count (the unary-iteration docstring gives the separate T-count caveat), not a transpiled gate count; see [primitives](../primitives/primitives.md) |
  | `double_factorization.double_factorization_one_norm` | one-norm `lambda` in the `"lcu"` or `"burg"` convention | a normalization used "to assess factorization quality", not a circuit cost |
  | `PauliLCU.alpha`, `Walk`/`QSVT` structural counts | normalization; register width, PREPARE/UNPREPARE pairs, walk or encoding invocations | no estimator exists; the family records state the logical structure only |

- **Rule.** Never present a proxy as a transpiled gate count, circuit depth
  after synthesis, runtime, or memory figure, and never compare quantities
  from different abstraction levels as though they were the same metric: a
  Toffoli count, a CNOT decomposition proxy, a Pauli-rotation count, and a
  one-norm are four different metrics.
- **Mismatch symptom.** A resource comparison that changes conclusion when
  the transpiler or target changes, or a one-norm reported as a gate count.
- **Source.** `python/cudaq_algorithms/stateprep/_givens.py`
  (`GivensResourceEstimate` docstring);
  `python/cudaq_algorithms/stateprep/_hartree_fock.py`;
  `docs/sphinx/guide/state_prep.rst` ("Resource estimation");
  `python/cudaq_algorithms/trotter.py` (`TrotterResourceEstimate`
  docstring); `python/cudaq_algorithms/primitives/_unary_iteration.py`,
  `_qrom.py`, `_alias_sampling.py`;
  `python/cudaq_algorithms/double_factorization/_factorization.py`
  (`double_factorization_one_norm`); `tests/python/test_primitives_resources.py`;
  the Trotter, state-preparation, double-factorization, and
  [primitives](../primitives/primitives.md) family records
  ([QROM](../primitives/primitives-qrom.md),
  [unary iteration](../primitives/primitives-unary-iteration.md),
  [arithmetic](../primitives/primitives-arithmetic.md),
  [alias sampling](../primitives/primitives-alias-sampling.md)).
