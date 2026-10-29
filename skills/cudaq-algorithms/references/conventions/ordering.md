# Ordering

## Qubit ordering: qubit 0 is least significant

- **Convention.** In statevectors and `SpinOperator.to_matrix()`, qubit 0 is
  the least significant bit of the computational-basis index. Basis state
  `|q_{n-1} ... q_1 q_0>` has index `sum_k q_k 2^k`.
- **Alternative.** Most papers write the leftmost tensor factor as qubit 0.
- **Translation.** Build dense references by reversing the factor order:
  `functools.reduce(np.kron, ops[::-1])`.
- **Mismatch symptom.** Amplitudes appear bit-reversed; single-qubit operators
  act on the wrong wire while norms stay correct.
- **Verification.** `cudaq.spin.z(0).to_matrix()` on one qubit is
  `diag(1, -1)`; on two qubits `spin.z(0)` acts on index bit 0.
- **Source.** `docs/sphinx/conventions.rst` ("Qubit ordering").

## Product-state amplitudes

Fix the requested site-to-qubit mapping before constructing a state. For
normalized single-qubit factors `u[k] = [a_k, b_k]` assigned to qubit `k`,
the amplitude at index `i` is `product_k u[k][(i >> k) & 1]`.
The equivalent tensor product is `u[n-1] tensor ... tensor u[0]`:

```python
import numpy as np

psi = np.array([1.0 + 0j])
for factor in reversed(u):
    psi = np.kron(psi, factor)  # Append each descending-index factor.
```

Independently check the constructed amplitudes against the user's site/bit
definition before passing `psi` to both the circuit and its dense reference.
For a separate diagnostic fixture, use unequal complex local factors to expose
reversed sites and lost phases; preserve the user's factors in the actual task.
Norm preservation alone cannot detect either error. Agreement between two
calculations sharing the same incorrectly constructed state validates their
action on that state, not the requested input. For an entangled input, check
the supplied basis-amplitude specification instead of a product-state formula.
This recipe follows the qubit-index convention in
`docs/sphinx/conventions.rst`; preparation readback is a separate check.

## Pauli words: string position equals qubit index

- **Convention.** Pauli words are strings over `IXYZ` with `word[0]` acting on
  qubit 0, so `"ZI"` means `Z` on qubit 0.
- **Alternative.** Left-to-right tensor-product notation, i.e. the reverse.
- **Translation.** Reverse the string at the boundary. Extract canonical terms
  with `term.get_pauli_word(width)` and `term.evaluate_coefficient()`, passing
  the intended qubit count as `width` so identity padding is explicit.
- **Mismatch symptom.** Correct spectra with wrong per-qubit locality; gapped
  operators silently shift.
- **Source.** `docs/sphinx/conventions.rst` ("Pauli words").

## Spin orbitals: interleaved, and `spin` is `2 * S_z`

- **Convention.** Spatial orbital `p` maps to spin orbitals `2p` (alpha, even)
  and `2p + 1` (beta, odd). All fermionic modules, pools, excitation
  enumerations, and Hartree-Fock occupations share this layout.
- **`spin` semantics.** `spin` is the alpha-minus-beta electron-count
  difference, i.e. twice the total `S_z`, with
  `n_beta = (num_electrons - spin) // 2` and `n_alpha = num_electrons -
  n_beta`. `spin == 0` selects closed-shell forms; `spin > 0` requires an even
  qubit count and produces interleaved open-shell occupations.
- **Alternative.** Blocked ordering, all alpha orbitals before all beta.
- **Translation.** Permute rows/indices between blocked and interleaved before
  comparing with external data.
- **Mismatch symptom.** An open-shell reference occupies contiguous orbitals
  (for example `{0,1,2,3}` instead of the interleaved `{0,1,2,4}`), so the
  determinant no longer matches an excitation pool built at the same spin.
  A second symptom is a parity rejection: because `spin` is `2 * S_z`,
  `(num_electrons - spin)` must be even when `spin > 0`, and
  `make_hartree_fock_occupation` raises `ValueError` ("(num_electrons - spin)
  must be even when spin > 0 (spin is 2*S_z)") for inputs such as
  `(8, 4, 1)`. `get_uccsd_excitations`, `get_num_uccsd_parameters`, and
  `make_uccsd_operator_pool` share that parity rejection. The raw `uccsd`
  kernel still floors the beta count without validation, so the host check
  must precede launch. The UCCSD host helpers additionally raise
  `ValueError` for an odd electron count at `spin == 0` (for example
  `(8, 3, 0)`, which the occupation builder accepts as `[0, 1, 2]`).
- **Source.** `docs/sphinx/conventions.rst` ("Spin orbitals: interleaved");
  `docs/sphinx/guide/state_prep.rst` ("Package conventions");
  `python/cudaq_algorithms/stateprep/_hartree_fock.py`
  (`make_hartree_fock_occupation` guards);
  `python/cudaq_algorithms/stateprep/_pools.py` (`get_uccsd_excitations`).

## Register width is extent, not population

- **Convention.** For a Pauli operator, register width is one plus the largest
  touched target index, with explicit handling for identity-only operators. It
  is not the number of distinct touched qubits: an operator on qubits 0 and 3
  needs width 4, not 2.
- **Mismatch symptom.** Gapped Pauli terms are truncated or applied to the
  wrong-size register.
- **Source.** `python/cudaq_algorithms/common_kernels.py`
  (`_term_qubit_extent`); the block-encoding and Trotter family records.

## Dense-operator self-test

Build every dense Hamiltonian, operator or statevector reference with a routine
whose ordering is explicit, and test that routine once before trusting it.
`word[j]` acts on qubit `j`, and qubit 0 is the least-significant index bit, so
it is the rightmost Kronecker factor. Norm preservation, Hermiticity and
agreement between two objects built with the same reversed ordering do not
detect a reversal; the diagonal test below does.

```python
import numpy as np

PAULI = {"I": np.eye(2, dtype=complex),
         "X": np.array([[0, 1], [1, 0]], dtype=complex),
         "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
         "Z": np.diag([1.0 + 0j, -1.0 + 0j])}


def dense_pauli_matrix(terms, num_qubits):
    """Dense sum of c_k P_k; word[j] acts on qubit j, qubit 0 is the least-significant bit."""
    dim = 2 ** num_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for coefficient, word in terms:
        if len(word) != num_qubits or any(c not in PAULI for c in word):
            raise ValueError(f"word {word!r} must have {num_qubits} characters from IXYZ")
        factor = np.array([[1.0 + 0j]])
        for character in reversed(word):  # qubit 0 becomes the fastest index
            factor = np.kron(factor, PAULI[character])
        matrix += coefficient * factor
    return matrix


def check_dense_ordering(num_qubits):
    """Z on qubit 0 alternates sign with the least-significant index bit."""
    z0 = dense_pauli_matrix([(1.0, "Z" + "I" * (num_qubits - 1))], num_qubits)
    assert np.allclose(np.diag(z0), np.tile([1.0, -1.0], 2 ** (num_qubits - 1)))
    x0 = dense_pauli_matrix([(1.0, "X" + "I" * (num_qubits - 1))], num_qubits)
    assert x0[1, 0] == 1 and x0[0, 1] == 1  # X on qubit 0 swaps indices 0 and 1


check_dense_ordering(3)
```

Use `dense_pauli_matrix` with the same word strings the library receives. For a
site-specified state, index `i` holds the amplitude of `|q_{n-1} ... q_0>` with
`q_k = (i >> k) & 1`; check one asymmetric amplitude against the user's
definition before comparing dynamics. `cudaq.spin.z(0).to_matrix()` follows the
same convention and is an independent cross-check.
