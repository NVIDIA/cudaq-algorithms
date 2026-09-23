# Bravyi–Kitaev fermion transform

Operation + object: **transform** **fermionic ladder-coefficient
tensors into a Bravyi–Kitaev Pauli operator**.

## Identity and classification

- Public symbol: `cudaq_algorithms.fermion.bravyi_kitaev`.
- Source: `python/cudaq_algorithms/fermion/_compilers.py`.
- Tests: `test_fermion.py`, `test_fermion_compilers.py`.
- Kind/role: classical transformation, computational leaf, host.
- Exactness: exact algebra for retained coefficients; deterministic.

## Scientific contract

The signature and literal tensor convention match
[Jordan–Wigner](jordan-wigner.md), but qubits store Fenwick-tree partial sums
of occupations. Resulting update/parity sets and Pauli words depend on the
total number of modes; do not reuse words across mode counts.

The current implementation compiles every two-body entry literally as
`V[i,j,k,l] a†_i a†_j a_k a_l`, matching the Jordan–Wigner path. A retired
binding antisymmetrized internally; callers that relied on that behavior must
antisymmetrize before calling the current API.

## Outputs, composition, and limits

Returns `cudaq.SpinOperator`. It may feed Pauli consumers after its basis
contract, coefficient contract, and geometry are preserved. Packaged
Hartree–Fock preparation is not documented as a Bravyi–Kitaev basis-state
provider, so do not compose them as if only Pauli words changed.

Output width tracks transformed Pauli support after pruning/cancellation rather
than the tensor's mode count. The result may be narrower when high support is
absent, but Fenwick update sets can touch a higher qubit than the largest mode
index appearing in a coefficient; do not infer width from active tensor indices.
Fully pruned/all-zero input is empty, and a scalar-only result has zero
non-identity extent. Preserve the intended `n` separately and satisfy each
consumer's explicit-width or padding contract.

Complex or non-Hermitian tensors are compiled literally. Verify Hermiticity
and real canonical Pauli coefficients before using a real-Hamiltonian consumer
such as `PauliLCU` or `Trotter`.

Chemistry's `qubit_hamiltonian` has no transform selector and does not call this
routine. To use BK, call spin-orbital expansion and this transform explicitly,
then validate the chosen input-state representation.

No resource estimator is provided. The documented logarithmic-weight intent of
the encoding is not a per-input gate, depth, or runtime measurement.

## Validation

Use an independent Fenwick encoding matrix to map occupation-basis fermionic
operators into qubit space and compare dense matrices. Include several total
mode counts, because the words are mode-count dependent, plus scalar,
tolerance, literal two-body, and invalid-shape cases. Cross-check only operator
equivalence after the correct basis transform; raw JW/BK Pauli words are not
expected to match. The runnable pointer is
`tests/python/test_fermion_compilers.py`, including its width/pruning cases.
Current public source and tests are authoritative and must be rechecked at use
time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
