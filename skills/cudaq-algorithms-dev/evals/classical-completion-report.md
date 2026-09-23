# Host/simulator/Trotter coverage implementation

This task implements three of the seven scientific coverage groups described
in [the completion plan](EVALUATION-COMPLETION.md); it does not modify the
scientific package, original cases, numerical grader, telemetry or skill prose.

Files: `e2e/extended_classical.py`, `e2e/extended.py`,
`e2e/tests/test_extended_classical.py`, `e2e/tests/test_extended_suite.py`.

## Outcome contracts

- `trotter_device_surfaces` pins extracted/pruned and ordered terms, identity
  separation, three directly executed circuit surfaces, the raw zero-step
  no-op, host invalid-input rejection, and the raw count proxy including an
  explicitly appended zero word. Public/held-out inputs use different signs,
  preparation phases, orderings, time/steps and second/fourth-order formulas.
  Expected states are independent ordered Pauli-matrix exponentials, with
  absolute phase and no omitted-identity phase reintroduced. Resources are
  formula fields, not observed transpilation or hardware measurements.
- `df_spin_diagnostics` uses two real two-orbital/two-leaf inputs with symmetric
  cores and orthogonal rotations. It checks independent projector-sum ERI
  reconstruction, explicit spin indices and the half factor, nonzero residuals,
  both named core norm formulas and two-electron energy from independent
  chemist/Fock algebra. It also pins shape/symmetry/convention rejection and
  the square-asymmetric-one-body acceptance boundary. No compression optimizer
  performance or quantum encoding is asserted.
- `good_subspace_layout` runs actual signed LCU circuits on two complex states,
  directly extracts their blocks, and checks phase, unnormalized probabilities,
  complement weight, scaled nonunit-input weight, nonalias copying and invalid
  shapes. This remains simulator-only, not shots or QPU execution.

Each spec declares required public APIs, actual host trace names and compiled
device entries where applicable. The normal grader still verifies both inputs,
field schemas, finite values, numerical tolerances, traces and workspace scope.
New `extended.py` temporarily installs the new case registry plus the original
Psi4 case; historical and focused CLI registries are restored on exit.

## Test-first and live evidence

1. Before `extended_classical.py` existed, the three hand-derived reference
   tests failed with `ModuleNotFoundError: extended_classical`:
   `python -B -m pytest -q -p no:cacheprovider e2e/tests/test_extended_classical.py -k 'literal or oracle or nonunit'`.
2. After implementation, the same module suite gave `4 passed, 3 skipped`.
   It contains literal X-rotation/identity, spin/DF norm/residual/sector-energy,
   and signed-Y/unnormalized-weight fixtures. Every output field is corrupted
   independently and the numerical checker must reject it; missing compiled
   evidence is also rejected. Both variants change scientific results.
3. Real runtime run, from repository root:

   ```bash
   PYTHONPATH=/workspaces/cudaq-algorithms/python \
   CUDAQ_E2E_PYTHON=/tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python \
   /tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python -B -m pytest -q -p no:cacheprovider \
     skills/cudaq-algorithms/evals/e2e/tests/test_extended_classical.py
   ```

   Result: **7 passed in 44.81 seconds**. All three gold applications passed
   both variants under the strict child sandbox, including numeric/API/scope.
4. Before `extended.py`, its two integration tests failed with missing module;
   after implementation, **2 passed in 0.15 seconds**. They exercise actual
   prepare/report, two-arm staging, suite mismatch rejection and restoration,
   and require a direct new case mapping for every previously uncovered record.

Independent review and scored model attempts belong to the root task. Gold
success does not establish that a model will write the correct application.
