# Active evaluation questions

`evals.json` contains seven reviewable questions for the current lean evaluation:
state preparation through Walk, structural block-encoding simulation compatibility,
QSVT real-time recovery, phase-sensitive Trotter evolution, a catalyst-cluster
comparison with one mandatory clarification, double-factorization diagnostics,
and missing researcher feedback.

The 42 boundary cases live in `regression-evals.json`. Their expected outputs
and assertions were revised during the September 2026 evaluation campaign, and
`tests/test_active_evals.py` pins the file's SHA-256.
The seven questions have schema and input-fixture checks, but they are not all
executable artifact graders and no model-run result is implied. Historical
evaluation documentation remains in `archive/EVAL-previous.md`.
