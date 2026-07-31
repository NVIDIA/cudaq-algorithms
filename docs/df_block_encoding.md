# Double-Factorized Block Encoding (bring-your-own-encoding example)

`DoubleFactorizedEncoding` block-encodes the
electronic-structure Hamiltonian directly from its double-factorized
integrals (von Burg et al., *PRX Quantum* **2**, 030305 (2021);
[arXiv:2007.14460](https://arxiv.org/abs/2007.14460)), instead of first
expanding it into Pauli words. It satisfies the same `BlockEncoding`
protocol as `PauliLCU`, so `Walk` and `QSVT` consume it unchanged.

It is intentionally **not part of the package**: it lives at
[`examples/bring_your_own_encoding/df_encoding.py`](../examples/bring_your_own_encoding/df_encoding.py)
as the worked, full-scale demonstration that the `BlockEncoding` protocol
is structural — a complete encoding implemented outside the library,
against the public API only, plugs into every consumer. Its
dense-reference test suite (`tests/python/test_df_encoding.py`) stays in
CI, so the example is held to library-grade correctness.

```python
from df_encoding import DoubleFactorizedEncoding   # the example module
from cudaq_algorithms import Walk, QSVT
from cudaq_algorithms import double_factorization as df

factorization = df.compressed_double_factorization(eri, num_leaves=T)
encoding = DoubleFactorizedEncoding(one_body, factorization,
                                    scalar_offset=nuclear_repulsion)

walk = Walk(encoding)                       # same consumers as PauliLCU
kernel = QSVT(encoding).kernel(sequence)
```

`one_body` is the `(n, n)` symmetric core-Hamiltonian matrix and the
second argument is either a `DoubleFactorization` (truncation happens
there — `explicit_double_factorization` / `compressed_double_factorization`)
or a raw chemist-notation `(pq|rs)` tensor, which is factorized exactly.
Conventions (spatial orbitals, interleaved spins `2p` up / `2p + 1` down,
Jordan-Wigner) match `cudaq_algorithms.chemistry`.

## Construction

The factorized Hamiltonian is regrouped so that every term is *diagonal
in some rotated orbital basis*:

```
H = const + sum_k F_k N_k  +  1/2 sum_t sum_kl Z^t_kl (N^t_k - 1)(N^t_l - 1)
```

- **Frame 0** — the eigenbasis of the corrected one-body matrix `kappa`
  (raw integrals + the exchange correction `-1/2 sum_r (pr|rq)` + the
  one-body remainder from centering the leaf number operators, all
  evaluated on the *factorized* tensor, so a truncated factorization
  encodes exactly its truncated Hamiltonian). Terms: one Z per spin
  orbital, coefficient `-F_k / 2`.
- **Frames 1..T** — one per factorization leaf, in the leaf's eigenbasis
  `U^t`. Centering makes each leaf *pure ZZ*: coefficient `Z_kl / 4` per
  spin pair for `k < l`, plus one cross-spin ZZ of `Z_kk / 4` per
  diagonal.

SELECT walks through the frames: an **uncontrolled** Givens network
rotates the system into the frame's basis, the frame's Z words execute
**ancilla-controlled**, and the next segment rotates onward — with no
control active the segments telescope to the identity, which is what
makes the controlled variants cheap (only Z words and sign phases carry
the control). Each spatial Givens rotation lifts to two three-qubit
`exp_pauli` pairs (`XZY`/`YZX` on contiguous slices), one per spin.
PREPARE and the walk/QSVT composites are the same machinery `PauliLCU`
uses.

## alpha and the published one-norm

The subnormalization is the 1-norm of the encoded coefficients, and by
construction it reproduces the LCU one-norm of
`double_factorization.double_factorization_one_norm(..., "lcu")`
(arXiv:2212.07957, Eq. 13) exactly, up to the identity term:

```
alpha = |const| + sum_k |F_k| + sum_t ( sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk| )
```

Compressing the factorization (fewer leaves) lowers `alpha` and the term
count together — the knob a flat Pauli expansion does not have. Since
QSVT circuit depth for time evolution scales like `alpha * t`, the
compression translates directly into shallower circuits, at the price of
a spectrum shift bounded by the tensor reconstruction error.

## Inspection

`num_frames`, `num_givens_rotations`, `num_terms`, `constant_term`,
`factorization`, and `terms` (as `(coefficient, z_qubits, frame_index)`,
where the qubits are Z positions *in that frame's rotated basis*).

## Limitations

- `select_observable` raises `NotImplementedError`: the odd-Chebyshev-
  moment observable is LCU-specific (it needs computational-frame Pauli
  words). Even moments (`Walk.moment` with even order) and every kernel
  factory work unchanged.
- The Givens networks are emitted sequentially (one rotation at a time);
  merging adjacent exit/entry networks into a single relative rotation,
  and parallel-scheduling commuting rotations, are documented future
  circuit-level optimizations.

## Example

[`examples/bring_your_own_encoding/df_block_encoding.py`](../examples/bring_your_own_encoding/df_block_encoding.py)
builds a small system, compares `DoubleFactorizedEncoding` with a
`PauliLCU` of the same Hamiltonian (alpha, term count, structure), sweeps
factorization truncation, and measures Chebyshev moments through the
shared `Walk` consumer.
