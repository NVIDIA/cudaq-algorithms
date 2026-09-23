# Trotter term planning

Operation + object: **preprocess** a **Pauli-sum Hamiltonian
into product-formula terms**.

## Identity and classification

- Public symbols: `cudaq_algorithms.trotter.make_trotter_terms`,
  `cudaq_algorithms.TrotterOrdering`, and `cudaq_algorithms.Trotter`.
- Source/tests: `python/cudaq_algorithms/trotter.py`, `test_trotter.py`.
- Kind/role: classical transformation, computational leaf, host.
- Exactness: exact extraction except explicit coefficient pruning.

## Contract

`make_trotter_terms(hamiltonian, coefficient_tolerance=1e-12)` accepts a
`cudaq.SpinOperator` or term, `{word: coefficient}` mapping, or iterable of
`(coefficient, word)` pairs. It returns:

```text
(coefficients, words, identity_coefficient, num_qubits)
```

Words are common-width `I/X/Y/Z` strings with position equal to qubit index.
Spin operators are padded to the widest targeted extent. Coefficients must be
real within the shared fixed noise tolerance. Empty and zero-width Hamiltonians,
string-like inputs, mismatched word widths, invalid characters, and negative
pruning tolerances are rejected.

Terms with exact zero or magnitude below the threshold are removed. Identity
terms are accumulated separately because their circuit action is a global
phase. This pruning changes the approximated Hamiltonian and must be included
in the error budget.

`Trotter` applies either `PRESERVE_INPUT` or
`COEFFICIENT_MAGNITUDE_DESCENDING` ordering at construction. Ordering is a
heuristic policy, not a proof of lower error.

## Output boundaries and validation

The flattened lists are accepted by
`cudaq_algorithms.trotter.apply_trotter` and the
[raw resource estimator](trotter-resources-raw.md). They do not themselves
evolve a state or select time, steps, or formula order. Constructing a
`cudaq_algorithms.Trotter` retains these validated, pruned terms for its kernel
factories and [planned-resource wrapper](trotter-resources-planned.md).

Independent checks: compare all supported input forms after canonicalization,
verify identity separation and widest-term padding, and assert all rejection
conditions. The cited test suite also checks that zero coefficients do not
inflate resources. The runnable pointer is `tests/python/test_trotter.py`, in
particular its term-extraction, ordering, and zero-coefficient cases.
Current public source and tests are authoritative and must be rechecked at use
time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
