# Occupied-to-virtual UCCSD operator pool

Operation + object: **preprocess** an **occupied-to-virtual
UCCSD excitation pool**.

## Identity and classification

- Public symbols, all from `cudaq_algorithms.stateprep`:
  `get_uccsd_excitations(num_qubits, num_electrons, spin=0)`,
  `get_num_uccsd_parameters(num_qubits, num_electrons, spin=0)`, and
  `make_uccsd_operator_pool(num_qubits, num_electrons, spin=0)`.
- Source: `python/cudaq_algorithms/stateprep/_pools.py:69-201`; exports:
  `python/cudaq_algorithms/stateprep/__init__.py:43-47,57-66`.
- Tests: `tests/python/test_operator_pools.py:21-57,92-127,276-291`,
  `tests/python/test_stateprep.py:8-38`, and
  `tests/python/test_stateprep_kernels.py:263-291`.
- Runnable usage: `docs/sphinx/examples/python/hartree_fock_ucc.py:58` and
  `pytest -q tests/python/test_operator_pools.py -k uccsd`.
- Source provenance: current public source/tests are authoritative and must be
  checked at use time.
- Kind/role/layer: exact deterministic classical transformation,
  computational leaf, host only; dependency `cudaq`.

## Scientific contract

The constructor emits one Hermitian Pauli generator per occupied-to-virtual
single or double. `get_uccsd_excitations` first returns the five ordered groups
`(singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta)`;
`get_num_uccsd_parameters` sums their lengths; the pool preserves that group
and within-group order exactly.

For `spin > 0`, `n_occ_beta=(num_electrons-spin)//2` and
`n_occ_alpha=num_electrons-n_occ_beta`. Alpha orbitals are even and beta
orbitals odd. Occupied lists precede virtual lists in each spin sector. For
`spin=0`, electron count must be even and each spin has
`num_electrons/2` occupied spatial orbitals. Within groups, occupied loops are
outer and virtual loops inner; same-spin doubles use ascending occupied and
ascending virtual pairs. A mixed double is specifically
`[alpha_occ, beta_occ, beta_virt, alpha_virt]`, not an alpha-then-beta virtual
pair.

A single `[p,q]` maps to
`0.5 Y_p Z(p,q) X_q - 0.5 X_p Z(p,q) Y_q`. A double is the eight-term
`0.125` `XXXY`-family expression in `_uccsd_double`; its four index-pattern
branches canonicalize the occupied and virtual pairs before inserting
`Z(i_occ,j_occ)` and `Z(a_virt,b_virt)`. The pool adds no branch-dependent sign
flip. Here `Z(l,h)` is the product of `Z` strictly between the endpoints.

Use this provider only when an occupied/virtual partition fixed by electron
count and `spin=2*S_z` is intended. It does not choose pool elements or
amplitudes, prepare Hartree–Fock, or establish equivalence with the separately
parameterized `uccsd` device kernel.

## Inputs, rejection, and silent gap

All three counts must be integral, non-negative, and not `bool`; fractional,
negative, or boolean values raise `ValueError` with "must be a non-negative
integer". Guards then apply in this order:

| Rejected condition | Exception |
| --- | --- |
| odd `num_qubits` | historical `RuntimeError`, "should be even" |
| `num_electrons > num_qubits` | `ValueError` |
| `spin > num_electrons` | `ValueError` |
| open-shell alpha occupancy exceeds `num_qubits/2` | `ValueError`, "does not fit" |
| odd `num_electrons` with `spin == 0` | historical `RuntimeError`, "spin multiplicity" |

The implementation has no parity guard for `(num_electrons-spin)` in the
open-shell branch. It silently floors the beta occupation, so `(8,4,1)` creates
the same partition and pool as `(8,4,2)`. The Hartree–Fock occupation builder
rejects the former. Report this mismatch; do not claim that `spin=1` is a
supported physical UCCSD/HF composition merely because pool construction
succeeds.

## Outputs and composition

The excitation helper returns a 5-tuple of `list[list[int]]`; the count helper
returns `int`; the builder returns the ordered raw object defined by the
[operator-pool Representation Record](operator-pools.md#representation-record--operator-pool).
Every operator acts on at most `num_qubits`; a smaller `op.qubit_count` is
normal and does not recover the construction width. Singles have two terms with
coefficients `+/-0.5`; doubles have eight terms with `+/-0.125`.

No UCCSD-specific `get_*_pauli_lists` helper exists. Use the generic
`get_fixed_parameter_ucc_pauli_lists` only under the validation, coefficient,
ordering, and exponent conventions in
[state-preparation-hf-ucc.md](state-preparation-hf-ucc.md). Consumers require
the same width, one parameter per pool element in order, and a prepared
reference determinant. The focused [`uccsd` device-kernel
contract](state-preparation-kernel-uccsd.md) re-enumerates the five groups
instead of consuming this pool and has a different amplitude convention. No
capability ID applies; this is direct object composition. Composite protocol:
not applicable, leaf operation.

## Accuracy, resources, and limitations

Enumeration and dyadic coefficients are exact. There is no pool resource
estimator. The exact host-output count is

```text
n_occ_a*n_virt_a + n_occ_b*n_virt_b
+ n_occ_a*n_occ_b*n_virt_b*n_virt_a
+ C(n_occ_a,2)*C(n_virt_a,2) + C(n_occ_b,2)*C(n_virt_b,2).
```

Committed assertions pin `3` elements at `(4,2,0)`, `8` at `(6,3,1)`, and
`875` at `(20,10,0)`. These are combinatorial host counts, not gate count,
depth, runtime, or memory. Selection, screening, and amplitude optimization are
absent. Cross-implementation agreement with the C++ paths named by the module
docstring is unverified because those paths were absent from the cited source; recheck current public source at use time.

## Validation

- Independent absolute oracle: exact words, coefficients, order, and signs at
  `(4,2,0)` (`test_operator_pools.py:92-127`).
- Independent fermionic oracle: dense Jordan–Wigner ladder matrices establish a
  full pool/generator bijection at `(8,4,2)`, including mixed doubles, within
  `atol=1e-10` (`:141-202,276-291`). It is intentionally agnostic to the
  overall `+/-i` sign.
- Structural checks: exact excitation lists at `(8,4,0)`, open-shell counts at
  `(6,3,1)`, width bounds, and all rejection rows above. The pool-derived dense
  circuit test is supporting but not independent because both sides reuse the
  pool.
- Evidence: the scientific assertions above are derived from cited source/tests;
  they require fresh execution before claiming numerical validation.
