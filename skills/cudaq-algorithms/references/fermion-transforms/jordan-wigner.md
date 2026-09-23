# Jordan–Wigner fermion transform

Operation + object: **transform** **fermionic ladder-coefficient
tensors into a Jordan–Wigner Pauli operator**.

## Identity and classification

- Public symbol: `cudaq_algorithms.fermion.jordan_wigner`.
- Source: `python/cudaq_algorithms/fermion/_compilers.py`.
- Tests: `test_jordan_wigner.py`, `test_fermion_compilers.py`,
  `test_fermion.py`.
- Kind/role: classical transformation, computational leaf, host.
- Exactness: exact algebra for retained coefficients; deterministic.

## Scientific contract

```python
jordan_wigner(
    one_body_or_two_body,
    two_body=None,
    scalar_offset=0.0,
    tolerance=1.0e-15,
)
```

Mode `j` maps to qubit `j`; computational-basis occupation and the packaged
Hartree–Fock/state-preparation ordering therefore align with this transform.
Jordan–Wigner parity strings can have linear weight in the mode index.

Inputs use the shared literal ladder-coefficient representation in
[fermion-transforms.md](fermion-transforms.md). Magnitudes below `tolerance`
are dropped on input and canonical output; this changes the returned operator.

## Outputs, composition, and limits

Returns `cudaq.SpinOperator`. The result may feed `PauliLCU`, `Trotter`, or
other Pauli consumers only after their scientific and geometry preconditions
are checked. Chemistry's `qubit_hamiltonian` calls this transform after spin
expansion.

Output width tracks the highest qubit actually touched after tolerance pruning
and cancellation, not the tensor's `n` modes. If high modes are untouched the
operator is narrower; fully pruned/all-zero input returns an empty operator,
and a surviving scalar alone has zero non-identity extent. Preserve the
intended `n` separately. For example, a scalar-only `cudaq.SpinOperator`
passed to `PauliLCU` needs an explicit positive `num_qubits`, while consumers
without a padding seam need their own documented treatment.

Complex or non-Hermitian input is compiled literally. Before passing the
result to a real-Hamiltonian consumer such as `PauliLCU` or `Trotter`, verify
Hermiticity and real canonical Pauli coefficients; the transform does not
establish those downstream conditions.

Do not swap in Bravyi–Kitaev while retaining the same computational-basis state
preparation or expected Pauli words. The two encodings represent occupations
differently even though they accept the same tensor form.

No resource estimator is provided. Term count and Pauli weight depend on tensor
sparsity, cancellations, tolerance, and mode count; do not invent asymptotic or
measured cost for a concrete input.

## Validation

Build independent dense creation/annihilation matrices with Jordan–Wigner
parity strings and compare the returned Pauli matrix. Include one-body,
two-body, scalar-offset, complex coefficient, tolerance-pruning, invalid-rank,
and mismatched-dimension cases. The runnable pointers are
`tests/python/test_jordan_wigner.py` and the width/pruning cases in
`tests/python/test_fermion_compilers.py`. Current public source and tests are
authoritative and must be rechecked at use time; this record was not freshly
executed.
[Source lookup](../source-provenance.md) gives shared current-source paths.
