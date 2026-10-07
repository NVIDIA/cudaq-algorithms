# Fermion-to-qubit transforms - family front door

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| transform / ladder tensors to Jordan-Wigner Pauli operator | `fermion.jordan_wigner` | classical transformation; host | none | [Jordan-Wigner](jordan-wigner.md) |
| transform / ladder tensors to Bravyi-Kitaev Pauli operator | `fermion.bravyi_kitaev` | classical transformation; host | none | [Bravyi-Kitaev](bravyi-kitaev.md) |

## Shared input representation

Both public transforms accept:

- a rank-2 `(n,n)` tensor of coefficients for `a_i^dagger a_j`, optionally followed
  by a rank-4 `(n,n,n,n)` tensor for `a_i^dagger a_j^dagger a_k a_l`; or
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

## Workflow

Write the operator convention before compiling: mode order, the scalar, and
whether the supplied two-body tensor already includes its physical prefactor.
For spatial chemistry integrals use the
[spin expansion](../chemistry/chemistry-spin-orbital-tensors.md), rather than
passing a chemist tensor directly as ladder coefficients. Choose the conserved
sector and target observable alongside the Hamiltonian. JW and BK represent the
same physics; select between them using the requested downstream cost (such as
Pauli weight, synthesized gate count or measurement groups) at equal accuracy.
Neither transform is a universally cheaper algorithm.

For static energies, compare corresponding physical sectors. For dynamics,
transform the initial state and every observable as well as the Hamiltonian.
With occupation bits `n`, the current BK convention stores Fenwick parities
`x=A*n mod 2`. Consequently BK Hamming weight is not electron number. Build
sector indices from occupation configurations and then encode them; transform
number/spin observables too. A diagonal one-body occupation state is a BK basis
state, but generally at a different index from its JW representation.

Keep compilation pruning separate from time-discretization and sampling errors.
A dense calculation is a useful reference for small systems; execute a circuit
when that is the requested prediction or resource being studied, with the same
state, observable, time units and sign of evolution.

## Verification

This small-system oracle assembles literal ladder coefficients without using
either compiler. Supply `h` of shape `(m,m)`, `V` of shape `(m,m,m,m)` (zeros for
a one-body model), `offset`, `prune_tol`, and the numerical-check budget `atol`.
Mode zero is the least significant occupation bit. Dense storage grows as
`4**m`; use a selected-sector/sparse reference for larger problems.

```python
import itertools
import numpy as np
from cudaq_algorithms import fermion

def occupation_matrix(h, V, offset=0.0):
    m = len(h)
    out = offset * np.eye(1 << m, dtype=complex)
    terms = [(h[i, j], ((j, False), (i, True)))
             for i, j in itertools.product(range(m), repeat=2)]
    terms += [(V[i, j, k, l], ((l, False), (k, False),
                              (j, True), (i, True)))
              for i, j, k, l in itertools.product(range(m), repeat=4)]
    for coefficient, actions in terms:
        if coefficient == 0:
            continue
        for column in range(1 << m):
            row, amplitude = column, coefficient
            for mode, create in actions:  # rightmost ladder acts first
                if ((row >> mode) & 1) == create:
                    break
                amplitude *= (-1)**(row & ((1 << mode) - 1)).bit_count()
                row ^= 1 << mode
            else:
                out[row, column] += amplitude
    return out

def fixed_width_matrix(operator, m):
    matrix = np.asarray(operator.to_matrix())
    if matrix.size == 0:
        return np.zeros((1 << m, 1 << m), dtype=complex)
    assert len(matrix) <= 1 << m
    return np.kron(np.eye((1 << m) // len(matrix)), matrix)

m = len(h)
H_occ = occupation_matrix(h, V, offset)
H_jw = fixed_width_matrix(fermion.jordan_wigner(
    h, V, scalar_offset=offset, tolerance=prune_tol), m)
H_bk = fixed_width_matrix(fermion.bravyi_kitaev(
    h, V, scalar_offset=offset, tolerance=prune_tol), m)
perm = np.zeros(1 << m, dtype=int)  # perm[b] is the BK index of occupation b
for b in range(1 << m):
    for node in range(1, m + 1):
        start = node - (node & -node)
        parity = sum((b >> j) & 1 for j in range(start, node)) % 2
        perm[b] |= parity << (node - 1)
np.testing.assert_allclose(H_jw, H_occ, atol=atol, rtol=0)
np.testing.assert_allclose(H_bk[np.ix_(perm, perm)], H_occ, atol=atol, rtol=0)
```

For an occupation-basis state use `psi_bk[perm] = psi_occ`; for an observable use
`O_bk[np.ix_(perm, perm)] = O_occ` into a newly allocated matrix. Given physical
sector indices `sector`, the BK indices are `perm[sector]`. Verify Hermiticity
and conservation before using `eigvalsh` on either restricted Hamiltonian.
Spectra alone cannot detect an incorrectly mapped state or observable: compare
expectations or a time trace against independent `exp(-1j*t*H_occ)` evolution.
Align global phase only when the requested state metric permits it; preserve
phase for controlled/interferometric use, complex amplitudes, or raw vector errors.
For a physical particle-number check, compare the full number distribution or
sector weight; its mean alone can hide leakage between sectors.

Compiler tests (`tests/python/test_fermion_compilers.py`) contain
independent dense ladders, Fenwick checks and complex-coefficient examples;
compiler source (`python/cudaq_algorithms/fermion/_compilers.py`)
documents the binary encoding and literal tensor convention.

Shared source is `python/cudaq_algorithms/fermion/_compilers.py`; tests are
`test_fermion.py`, `test_fermion_compilers.py`, and `test_jordan_wigner.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [Source lookup](../source-provenance.md) gives shared current-source paths.
