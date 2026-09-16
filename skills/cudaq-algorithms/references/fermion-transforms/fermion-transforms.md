# Fermion-to-qubit transforms — family front door

Status: draft.

| Operation + object | Record |
| --- | --- |
| transform / ladder-coefficient tensors with Jordan–Wigner encoding | [jordan-wigner.md](jordan-wigner.md) |
| transform / ladder-coefficient tensors with Bravyi–Kitaev encoding | [bravyi-kitaev.md](bravyi-kitaev.md) |

## Shared input representation

Both public transforms accept:

- a rank-2 `(n,n)` tensor of coefficients for `a†_i a_j`, optionally followed
  by a rank-4 `(n,n,n,n)` tensor for `a†_i a†_j a_k a_l`; or
- the rank-4 tensor alone;
- `scalar_offset` for the identity term;
- `tolerance` for magnitude pruning before and after compilation.

Dimensions must be square and agree. Inputs are compiled literally; no hidden
chemist-to-physicist reorder or antisymmetrization is performed. Output is a
canonicalized `cudaq.SpinOperator` with qubit positions matching package Pauli
word conventions.

The returned operator does **not** retain the input mode count as independent
metadata. Its extent is set by non-identity qubits actually touched after
pruning and cancellation; it can be narrower than `n`, and a fully pruned or
all-zero Hamiltonian is empty. Identity-only output has zero non-identity
extent. A downstream fixed-width application must restore/check the intended
width rather than infer it from the operator.

Both transforms accept complex tensors and can return complex or
non-Hermitian operators. Consumers such as `PauliLCU` and `Trotter` require a
real Hermitian Hamiltonian contract; transform output is not automatically
eligible merely because its type is `cudaq.SpinOperator`.

Shared source is `python/cudaq_algorithms/fermion/_compilers.py`; tests are
`test_fermion.py`, `test_fermion_compilers.py`, and `test_jordan_wigner.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [source-provenance.md](../source-provenance.md) records historical
last-review audit context.
