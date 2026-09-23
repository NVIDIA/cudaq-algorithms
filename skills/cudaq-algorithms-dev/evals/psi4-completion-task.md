# Psi4 strict-runtime repair brief

Implement the diagnosed repair for Task 1 in
[EVALUATION-COMPLETION.md](EVALUATION-COMPLETION.md).
Own `e2e/runtime.py`, the two minimal integration call sites in `e2e/run.py`,
new `e2e/tests/test_psi4_runtime.py`, and `psi4-completion-report.md` only.
No package source, numerical cases/oracles, telemetry, skill, CI, YAML, Git,
global configuration, credential or old-result edits. No subagents or scored
model campaigns. Use `apply_patch`, test first and independent review.

## Accepted diagnosis and bounded design

oneMKL in the real Psi4 interpreter calls `readlink("/proc/self/exe")`.
The strict profile hides it; allowing only `/proc/self` read makes provider
execution work. An exact symlink-file grant is ineffective. Keep this grant
limited to a detected Psi4 runtime: inspect the allowed interpreter prefix's
`conda-meta/psi4-*.json` records without importing or executing package code.
Ordinary core runtimes retain their old profile. This is support for the
process's own runtime introspection, not access to other processes or data.
Document that its sanitized self environment becomes readable.

Provider `timer.dat` is a fixed cwd side file. Add
`stage_runtime_artifacts(workspace, case_id)` to runtime.py. It does nothing
for other cases; for `psi4_energy` it creates exactly the relative symlink
`timer.dat -> .tmp/timer.dat` before initial inventory. Reject an existing
path (including dangling symlinks) rather than replacing anything. The fixed
`.tmp/` target must remain inside that workspace; do not follow a preexisting
symlink directory to another location. No generic arbitrary path allowance.

Call that helper after staging and before inventory in both `execute_app`
and `run_one`. Do not put the timer symlink into common source snapshots or
allow `timer.dat` in `scope_changes`. Existing inventory must continue to catch
link replacement/repointing; only actual `.tmp/` target writes are permitted.

## Red/green verification

Write regressions before implementing the repair:

- A core runtime retains the old profile; a real/detected Psi4 runtime gets
  self-process read but no broad `/proc` or `/sys` grant.
- Non-Psi4 artifact staging is empty. Psi4's fixed link routes timer writes
  into scratch with no scope changes; overwriting/repointing it is detected.
  Existing-file and escaping scratch-directory cases fail safely.
- With `E2E_SANDBOX_PSI4_PYTHON` set, real `run.execute_app` passes both
  `psi4_energy` gold variants (numeric, API and workspace checks).
- A real isolation probe rejects evaluator/repository and actual sibling paths,
  other-process `/proc`, and `/proc/self/root` or `/proc/self/cwd` escape paths;
  network remains blocked. Do not read actual credentials for any probe.

Use `/tmp/cudaq-e2e-runtime.zY0RJF/psi4/bin/python` as the real provider and
the existing core interpreter for test orchestration. Keep original scientific
tolerances and all gold inputs unchanged. Run covering regressions and the
existing runner/isolation tests; record exact red/green evidence, observed
provider versions, runtime behavior and limitations in the report. Root will
freeze fresh campaigns only after all concurrent evaluator work and reviews end.
