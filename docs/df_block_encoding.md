# Double-Factorized Block Encoding (bring-your-own-encoding example)

`DoubleFactorizedEncoding` block-encodes the
electronic-structure Hamiltonian directly from its double-factorized
integrals (von Burg et al., *PRX Quantum* **2**, 030305 (2021);
[arXiv:2007.14460](https://arxiv.org/abs/2007.14460)), instead of first
expanding it into Pauli words. It satisfies the same `BlockEncoding`
protocol as `PauliLCU`, so `Walk` and `QSVT` consume it unchanged.

It ships as a runnable example —
[`examples/bring_your_own_encoding/df_encoding.py`](../examples/bring_your_own_encoding/df_encoding.py)
— and doubles as the worked, full-scale demonstration that the
`BlockEncoding` protocol is structural: implement the protocol's surface
against the public API and every consumer accepts the encoding. Its
dense-reference test suite (`tests/python/test_df_encoding.py`) runs in
CI, so the example is held to library-grade correctness.

```python
import sys
sys.path.insert(0, "examples/bring_your_own_encoding")  # from the repo root, or copy the file
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
runs the encoding on real molecules (integrals from PySCF —
`pip install pyscf`), across a menu of configurations:

```
python3 df_block_encoding.py [config]

h2         H2 / STO-3G           2 orbitals ->  4 system qubits  (default)
h2o-cas44  H2O / STO-3G CAS(4,4) 4 orbitals ->  8 system qubits
h4         linear H4 / STO-3G    4 orbitals ->  8 system qubits
lih        LiH / STO-3G          6 orbitals -> 12 system qubits
h2o        H2O / STO-3G          7 orbitals -> 14 system qubits
h2o-631g   H2O / 6-31G          13 orbitals -> 26 system qubits
```

Every configuration compares `DoubleFactorizedEncoding` with a `PauliLCU`
of the same Hamiltonian (alpha, term count, structure) and sweeps the
factorization-truncation dial. Small configurations additionally verify
the encoded block against a sparse Jordan-Wigner reference and measure
Chebyshev moments through the shared `Walk` consumer; large ones report
the statevector cost and tell the classical scaling story instead (at
H2O/6-31G the DF alpha is ~32% below the flat expansion's).
