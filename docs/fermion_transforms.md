# Fermion-to-Qubit Transforms (Jordan-Wigner and Bravyi-Kitaev)

`cudaq_algorithms.fermion` compiles fermionic integrals to qubit
operators, in pure Python — no compiled extension required.

```python
from cudaq_algorithms.fermion import jordan_wigner, bravyi_kitaev

h = jordan_wigner(one_body, two_body, scalar_offset=e_nuc)   # SpinOperator
h = bravyi_kitaev(one_body, two_body, scalar_offset=e_nuc)
```

Both accept an `(n, n)` one-body tensor, optionally with an
`(n, n, n, n)` two-body tensor (or a two-body tensor alone); entries are
the coefficients of `adag_i a_j` and `adag_i adag_j a_k a_l` over `n`
spin orbitals:

```
H = scalar_offset * I + sum h[i, j] adag_i a_j
                      + sum V[i, j, k, l] adag_i adag_j a_k a_l
```

`scalar_offset` is added as an identity term (e.g. nuclear repulsion);
input entries and compiled terms below `tolerance` (default `1e-15`)
are dropped. The result is a `cudaq.SpinOperator`, ready for
`PauliLCU`/`Walk`/`QSVT` or `Trotter`.

## One construction, two transforms

A fermion-to-qubit encoding is treated as a linear map over GF(2): an
invertible binary matrix `A` stores the occupation vector `n` as qubit
bits `x = A n (mod 2)`. Jordan-Wigner is `A = I`; Bravyi-Kitaev is the
Fenwick (binary-indexed-tree) partial-sum matrix. Everything a ladder
operator needs is derived from `A` and its GF(2) inverse as three qubit
masks per mode — update (the X part), parity (the Z string carrying the
anticommutation sign), and flip (the number-operator Z word) — and the
Hamiltonian is compiled term by term in a symplectic (bitmask) Pauli
algebra. The Bravyi-Kitaev words touch `O(log n)` qubits where
Jordan-Wigner strings touch `O(n)`.

## Semantics: tensors are compiled exactly as given

Both transforms compile arbitrary coefficient tensors — no hermiticity
or index-ordering symmetry is assumed, and the two transforms are
guaranteed to encode the same fermionic operator (isospectral, pinned
by tests). This is a deliberate difference from the compiled C++
`bravyi_kitaev` this module replaces, which enumerated restricted index
patterns and silently assumed a chemistry-canonical tensor ordering;
tensors outside that form produced a wrong operator. The pure
transforms have Jordan-Wigner's generic semantics for every input.

## Validation

The test suite pins: Jordan-Wigner against dense ladder-operator
matrices (including non-hermitian tensors); Bravyi-Kitaev against the
exact per-pair operators and the H2 operator from the C++ unit tests;
isospectrality between the two encodings on unstructured tensors; the
`O(log n)` word-weight advantage; and, when the compiled extension is
present, term-by-term parity with the C++ transforms on their supported
inputs.

## Notes

- Cost scales with the number of nonzero tensor entries (16 Pauli-word
  products each for two-body entries); a fully dense 14-spin-orbital
  tensor compiles in seconds. Vectorizing the inner accumulation is
  documented future work if larger dense tensors become routine.
- The state-preparation utilities remain in the compiled extension for
  now; migrating them is planned separately.
