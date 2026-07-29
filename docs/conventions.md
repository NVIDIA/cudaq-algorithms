# Conventions

The physics and numerical conventions used across the library. Most
subtle bugs are convention mismatches, not logic errors; when validating
numerics, check against this page first. Each convention below states
how to verify it in a few lines.

## Qubit ordering: qubit 0 is least significant

In statevectors and `SpinOperator.to_matrix()`, qubit 0 is the *least
significant* bit of the computational-basis index — the rightmost factor
in a Kronecker product. Basis state `|q_{n-1} ... q_1 q_0>` has index
`sum q_k 2^k`.

Building a dense reference for an operator on qubit `j` of `m` qubits:

```python
ops = [I] * m
ops[j] = Z
dense = functools.reduce(np.kron, ops[::-1])   # note the reversal
```

Verify: `np.asarray(cudaq.spin.z(0).to_matrix())` on 1 qubit is
`diag(1, -1)`; on 2 qubits `spin.z(0)` acts on index bit 0.

## Pauli words

Pauli words are strings over `IXYZ` with position = qubit index
(`word[0]` acts on qubit 0), so `"ZI"` means `Z` on qubit 0 — reversed
relative to most papers' left-to-right tensor-product notation. Always
translate at the boundary. Extract canonical term dictionaries with
`term.get_pauli_word(width)` / `term.evaluate_coefficient()`, passing
the intended qubit count as `width` so identity padding is explicit.

## Spin orbitals: interleaved

Spatial orbital `p` maps to spin orbitals `2p` (up) and `2p + 1` (down).
All fermionic modules (the fermion transforms, the chemistry-facing
helpers, the double-factorized encoding) use this convention.

## Fermionic integral tensors

`n` spin orbitals; tensors are coefficient arrays for normal-ordered
ladder products, with **no implicit symmetry factors**:

```
H = scalar_offset * I
    + sum_pq   h[p, q]       adag_p a_q
    + sum_pqrs V[p, q, r, s] adag_p adag_q a_r a_s
```

Any 1/2 factors from a chemist-notation source must already be folded
into `V`. From a chemist `(pq|rs)` spatial tensor, the standard
spin-orbital expansion is: reorder with `eri.transpose(0, 2, 3, 1)`
(chemist -> coefficients of `adag adag a a`), scale by `1/2`, and
distribute over the four spin patterns
`(2p, 2q, 2r, 2s)`, `(2p+1, 2q+1, 2r+1, 2s+1)`, `(2p, 2q+1, 2r+1, 2s)`,
`(2p+1, 2q, 2r, 2s+1)` — see `tests/python/test_jordan_wigner.py` for
the reference implementation.

Note that many inequivalent tensor layouts encode the *same* physical
operator (indices can be traded against fermionic antisymmetry). The
transforms compile tensors exactly as given, so any valid layout works —
but when comparing against external data (OpenFermion, literature
tables), normalize the layout first or compare spectra rather than
tensors.

## Jordan-Wigner ladder operators

```
adag_j = 1/2 * Z_0 ... Z_{j-1} (X_j - i Y_j)
a_j    = 1/2 * Z_0 ... Z_{j-1} (X_j + i Y_j)
```

Dense reference (with the qubit-0-least-significant kron ordering):
annihilator `a_j` = `Z^{⊗j} ⊗ [[0, 1], [0, 0]] ⊗ I^{⊗(m-j-1)}`,
Kronecker-multiplied right-to-left.

## Block encodings

For any `BlockEncoding` (e.g. `PauliLCU`, `DoubleFactorizedEncoding`):

- The encoded block is `<0|_anc U_A |0>_anc = H / alpha`, with `alpha`
  the 1-norm of the encoded coefficients.
- `encode_kernel()` circuits allocate the **system register first**
  (from the input `cudaq.State`), ancillas after — so the good subspace
  is the **first `2**num_system` amplitudes** of the output state.
- `num_ancilla >= 1` always (single-term encodings get one idle
  ancilla).
- Controlled variants take a combined `[control, ancilla...]` register
  with the external control at **qubit 0**, and reduce to the identity
  at control `|0>`.

Verify (the standard encode-block test):

```python
out = np.array(cudaq.get_state(enc.encode_kernel(), state_from(ket)))
np.testing.assert_allclose(out[:2**enc.num_system],
                           (H_dense @ ket) / enc.alpha, atol=1e-12)
```

## Qubitization walk

- One walk step `W = R U_A` (encoding, then reflection `R = I - 2|0><0|`)
  block-encodes **`-H/alpha`**; its eigenphases are
  `pi -/+ arccos(lambda/alpha)`. Much of the literature (e.g. Low-Chuang)
  uses the opposite reflection sign or the other operator order and quotes
  `+H/alpha` with eigenphases `+/- arccos(lambda/alpha)` — translate before
  comparing. Consumers of this library rely on the `-H/alpha` form.
- `walk_kernel(power)` / `Walk` powers apply Chebyshev polynomials:
  the good-subspace block after `p` steps is `T_p(-H/alpha)`.
- `Walk.moment(ket, k)` returns `<T_k(H/alpha)>` (the sign convention
  is handled internally; no caller-side negation).

## QSVT

Projector-phase convention; the exact 2x2 signal model of the circuit
built by `QSVT.kernel` is implemented as `reference_response` in
`tests/python/test_qsvt.py` — treat that function as the executable
specification. A forward step is `reflect_about_zero` then the block
encoding; `qsp`-convention sequences run doubled projector phases and
differ from the model by `exp(i * sum(phases))`
(`recover_real_time_evolution` accounts for it). One more
literature trap: QSPPACK's native `W_x` rotation has *imaginary*
off-diagonals, while the circuit's response step is real — the bridge
lives in the phase-generation options, and `reference_response` is the
executable arbiter whenever they seem to disagree.

## Reflection *gate* vs reflection *observable*

Same word, opposite sign — a standard source of off-by-a-sign bugs:

- the **gate** `common_kernels.reflect_about_zero` is `I - 2|0..0><0..0|`
  (the all-zero state acquires `-1`);
- the **observable** `qubitization.reflection_observable` is
  `2|0..0><0..0| - I` (expectation `+1` on the all-zero state).

## Simulation targets and tolerances

The default CUDA-Q target is fp32: statevector results carry ~1e-7
error. All library tests pin `qpp-cpu` (fp64, via `conftest.py`) and
assert at 1e-10..1e-12. When a probe shows ~1e-7 residuals, suspect the
target precision before suspecting the code.

## Hermiticity

`eigvalsh`, `.real`, and spectrum-sorting shortcuts are valid only for
Hermitian operators. Generic coefficient tensors produce non-Hermitian
operators; use `np.linalg.eigvals` and compare sorted complex spectra,
or hermitize the input deliberately and say so.
