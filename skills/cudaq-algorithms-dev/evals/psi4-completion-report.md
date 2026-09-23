# Psi4 strict-runtime completion report

Date: 2026-09-13

## Outcome

The real Psi4 gold application now runs on both fixed geometries under the
native restrictive worker profile. The change is limited to the evaluator:
no package source, skill guidance, CI, global configuration, prior result, or
scientific input/oracle/tolerance was changed. No scored model campaign was
run; fresh campaigns remain a separate post-review step.

The repair has two bounded parts:

- Only an interpreter prefix with a regular, valid
  `conda-meta/psi4-*.json` record whose package name is `psi4` receives
  read access to `/proc/self`. Detection reads inert metadata and neither
  imports nor executes provider code. A core or unrecognized runtime retains
  the previous profile, and no `/proc` or `/sys` tree grant is added.
- Only `psi4_energy` receives the pre-inventory relative link
  `timer.dat -> .tmp/timer.dat`. Staging rejects every preexisting timer path,
  including a dangling link, and rejects a missing, non-directory, or symlinked
  `.tmp`. The existing inventory records the fixed link, ignores only writes
  to its scratch target, and detects replacement or repointing of the link.

Both authoring and independent grading workspaces stage the provider artifact;
common source snapshots remain unchanged. `timer.dat` was not added to the
scope allowlist.

## Root cause and observed runtime

oneMKL in the provider runtime calls `readlink("/proc/self/exe")` to locate its
loader. The original profile made that path appear absent with `ENOENT`, after
which `import psi4.core` exited 2 with:

```text
Intel oneMKL FATAL ERROR: Cannot load <mkl-loader>.
```

An exact `/proc/self/exe` permission was ineffective. Read-only `/proc/self`
made the call succeed without granting other process trees. Once loading was
repaired, Psi4 1.10's fixed cwd `timer.dat` became the only remaining workspace
scope change. The task-local scratch link contains that provider side effect.

Observed components:

- Psi4 interpreter: Python 3.12.14
- Psi4: 1.10, conda build `py312hf13b23c_3`
- oneMKL: 2025.3.1, conda build `h0e700b2_12`
- Core orchestration interpreter: Python 3.12.3
- Native Codex CLI: 0.144.4

## Test-first evidence

Before implementation, the new focused unit suite produced the expected red
result: `6 failed, 2 skipped`. The profile test showed the missing Psi4-only
self permission; five artifact tests failed because
`stage_runtime_artifacts` did not yet exist.

The real integration test was also observed red before implementation:
`1 failed, 7 deselected`. Both internal gold checks exited 2 at the oneMKL
loader and produced no result artifact.

After the minimal implementation:

```text
6 passed, 2 skipped in 0.14s
9 passed in 15.80s
45 passed, 1 skipped, 8 subtests passed in 22.46s
```

Direct `run_one` call-site coverage was added by temporarily removing that
integration line, observing the focused test fail on the absent initial timer
link, and restoring the minimal call; it then passed. The second command above
enabled the real Psi4 tests. The combined command covered
the existing runner contracts, existing PySCF provider isolation, and all new
Psi4 tests with the real core and provider interpreters. The one skipped test
was an existing optional runtime-gated contract, not a Psi4 failure.

Both Psi4 variants returned zero, executed the required
`cudaq_algorithms.chemistry.from_psi4` and `qubit_hamiltonian` APIs, produced no
out-of-scope changes, and agreed with the independent references to maximum
absolute error `2.65e-14`.

## Isolation and limitations

The real isolation probe continued to deny the evaluator, repository, skill,
an existing sibling workspace, `/proc/1/environ`, and `/proc/2/statm`. Paths
through `/proc/self/root` and `/proc/self/cwd` could not escape to the evaluator
or sibling. The network probe remained blocked.

Read-only `/proc/self` necessarily exposes the worker's own process metadata,
including its environment. That environment is constructed by the evaluator
from the existing sanitized allowlist and contains no API or authentication
variables. This exception is enabled only for detected Psi4 conda runtimes.
A non-conda Psi4 installation fails closed and would require separate review
rather than automatically receiving the exception.

The timer containment is intentionally not generic. If a future Psi4 build
changes or adds fixed cwd artifacts, they remain scope failures until reviewed.
