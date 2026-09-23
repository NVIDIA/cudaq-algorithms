# Raw Trotter resource estimate

Operation + object: **estimate** a **logical circuit description
from caller-supplied flattened coefficient and Pauli-word lists**.

## Identity and classification

- Public function:
  `cudaq_algorithms.trotter.estimate_trotter_resources(coefficients, words,
  steps, order, identity_coefficient=0.0)`.
- Result: root-exported `cudaq_algorithms.TrotterResourceEstimate`.
- Source/tests: `python/cudaq_algorithms/trotter.py`, `test_trotter.py`.
- Kind/role/layer: resource estimator, computational leaf, host.
- Resource status: exact formulas over the supplied word list plus a logical
  CNOT decomposition proxy; not a validated Hamiltonian or measured cost.

## Contract

The function copies `coefficients` and `words` to lists, requires equal list
lengths, validates positive integral `steps`, validates `order` in `{1,2,4}`,
and converts `identity_coefficient` to `float` in the result. Empty parallel
lists are accepted and produce zero terms, rotations, and estimated CNOTs.

It does **not** canonicalize or order terms, prune zero or small coefficients,
separate identity words, validate coefficient values, enforce common word
widths, or validate `I/X/Y/Z` characters. In particular:

- `num_terms` is exactly `len(words)`, including supplied identity words and
  words paired with zero coefficients;
- coefficient values do not affect either count after the equal-length check;
- `identity_coefficient` is echoed from the caller, not derived from `words`;
- Pauli weight is computed as the number of characters whose string form is not
  `I`, so unsupported characters also inflate the proxy rather than raising.

Call [term planning](trotter-planning.md) first, or use the
[planned-resource wrapper](trotter-resources-planned.md), when canonical Pauli
and identity semantics are required.

## Result formulas and boundaries

Let `T=len(words)`, let `w_j` be the raw helper's character-count weight, and
let `m={1:1,2:2,4:6}`. The result uses:

```text
pauli_rotations    = T * steps * m
estimated_cx_count = steps * m * sum_j(2 * max(w_j - 1, 0))
```

Those are exact evaluations of the helper's formulas. The CNOT value is only a
simple Pauli-rotation decomposition proxy. Neither value is hardware depth,
runtime, a measured gate count, target-aware cost, memory, or a numerical error
estimate. The helper does not inspect a circuit.

## Oracle and runnable evidence

Compute both formulas independently from the exact supplied strings and verify
that `num_terms` remains `len(words)`; do not silently preprocess the oracle.
Also probe unequal lengths, invalid steps/order, identity and zero-coefficient
words, unsupported characters, and empty parallel lists to distinguish this
contract from the planned wrapper.

Run the authoritative flattened-list happy path from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_trotter.py \
  -k estimate_trotter_resources_accepts_flattened_terms
```

The cited test supplies output from `make_trotter_terms`; it does not directly
pin the weaker raw-input boundaries above. Current public source and tests are
authoritative and must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
