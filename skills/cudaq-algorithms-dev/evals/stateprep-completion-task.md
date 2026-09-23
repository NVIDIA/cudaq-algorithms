# State-preparation coverage implementation brief

Implement the three state-preparation groups in
[EVALUATION-COMPLETION.md](EVALUATION-COMPLETION.md). You own only
`e2e/extended_stateprep.py`, `e2e/tests/test_extended_stateprep.py`, and
`stateprep-completion-report.md`. No other edits, model campaigns, subagents,
CI work, YAML, Git mutations or package-source changes. Use `apply_patch`.
Existing skill content and historical results are immutable.

## Interface

Match `quantum_cases.py`: expose `case_specs()`,
`parameters(case_id, variant)` for variants 0 and 1,
`expected(case_id, params)` returning numeric scalar/array fields, and
`reference_source(case_id)` returning an executable gold application's source.
Gold accepts `--input` JSON and writes `--output` non-object NPZ. It imports
real CUDA-Q Algorithms and runs on CPU qpp-cpu fp64. Expected values import no
evaluated package. Use independent Fock/Pauli algebra, analytic rotations and
determinant minors. Fix tolerances before comparing gold.

Every spec includes exact `required_public_apis`, host `required_symbols`,
compiled `required_kernels` for device entries, output field descriptions,
families and narrow scope. Task prompts name required APIs equally to both
arms and describe mathematical outputs without exposing oracle code. Resource
proxies are formula-level estimates, never measured physical gate counts.

## Three independently graded groups

1. `reference_excitation_primitives`: raw `hartree_fock`,
   `hartree_fock_occupation`, `single_excitation`, `double_excitation`, plus
   `estimate_hartree_fock_resources` and
   `estimate_hartree_fock_occupation_resources`. Return individually observable
   prepared/evolved states, particle-number checks and named integer resource
   fields. Use valid indices/spin and nontrivial signed angles; verify the
   actual half-angle/parity convention independently. Include an open-shell
   occupation input when supported; do not claim open-shell direct UCCSD.
2. `grouped_ucc_device`: direct `uccgsd`, `upccgsd`, `ceo`,
   `fixed_parameter_ucc`, and `estimate_fixed_parameter_ucc_resources`.
   Execute each direct device entry on a prepared state, with distinct outputs
   compared to independent ordered exponentials. Preserve the different
   parameter/group conventions; fixed UCC also uses a custom non-pool group.
   Resource outputs pin the estimator's actual formula, not gate compilation.
3. `givens_raw_device`: direct `givens_rotation`, `phase_givens_rotation`,
   `slater_determinant`, and `estimate_givens_resources`. Compare individual
   raw rotations with analytic single-particle maps; compare real Slater
   preparation with determinant minors, norm and particle number. Use distinct
   supported public/held-out angles/orbitals and explicitly scoped phase
   equivalence; retain absolute phase wherever the contract requires it.

## Tests and evidence

Use TDD: first tests must fail because contracts/module are absent, then
implement. Expectations must include literal hand-checked examples independent
of package helpers. Test both changed scientific inputs, exact output schema,
finite values, and rejection of wrong numeric output or missing device use.
Demonstrate positive gold execution on both variants using `run.execute_app`
with a temporary case registry (pytest monkeypatch restores it).

Supported interpreter: `/tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python`.
Set `PYTHONPATH=/workspaces/cudaq-algorithms/python`. The real bundled ripgrep
is `/home/.codex/packages/standalone/releases/0.144.4-x86_64-unknown-linux-musl/codex-path/rg`.
Request normal outer execution approval if strict nested sandbox tests need
it; do not alter the child profile. Root owns the new suite wrapper and reviews.

Before reporting, self-review source paths, output dependence, all 15 listed
record mappings, and preserved scientific boundaries. Write red/green commands
and results, gold checks and limitations to `stateprep-completion-report.md`.
Return concise status and concerns. Do not run or repair scored model attempts.
