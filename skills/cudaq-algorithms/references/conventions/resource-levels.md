# Resource Levels

## Resource abstraction levels must not be conflated

- **Convention.** The library's `estimate_*_resources` helpers return counts of
  logical operations before transpilation. `GivensResourceEstimate`'s docstring
  is the strictest statement in source: its proxies are
  decomposition-independent **upper bounds**, explicitly *not* transpiled gate
  counts. The Hartree-Fock and fixed-parameter UCC estimator docstrings carry
  no such wording, so for those the bound comes from the Sphinx guide.
- **Rule.** Never present a proxy as a transpiled gate count, circuit depth
  after synthesis, runtime, or memory figure, and never compare quantities
  from different abstraction levels as though they were the same metric.
- **Mismatch symptom.** A resource comparison that changes conclusion when the
  transpiler or target changes.
- **Source.** `python/cudaq_algorithms/stateprep/_givens.py`
  (`estimate_givens_resources` docstring); `docs/sphinx/guide/state_prep.rst`
  (resource-estimate paragraphs).
