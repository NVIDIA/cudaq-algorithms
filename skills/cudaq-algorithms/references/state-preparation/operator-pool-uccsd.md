# Occupied-to-virtual UCCSD operator pool

Operation + object: **preprocess** an **occupied-to-virtual
UCCSD excitation pool**.

## Identity and classification

- Public symbols, all from `cudaq_algorithms.stateprep`:
  `get_uccsd_excitations(num_qubits, num_electrons, spin=0)`,
  `get_num_uccsd_parameters(num_qubits, num_electrons, spin=0)`, and
  `make_uccsd_operator_pool(num_qubits, num_electrons, spin=0)`.
- Source: `python/cudaq_algorithms/stateprep/_pools.py:71-216`; exports:
  `python/cudaq_algorithms/stateprep/__init__.py:44-48,60-69`.
- Tests: `tests/python/test_operator_pools.py:21-57,92-127,276-297,327-344`,
  `tests/python/test_stateprep.py:8-38`, and
  `tests/python/test_stateprep_kernels.py:254-301`.
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
`Z(i_occ,j_occ)` and `Z(a_virt,b_virt)`. It negates the generator when exactly
one supplied pair is descending, matching `double_excitation`. Here `Z(l,h)`
is the product of `Z` strictly between the endpoints. Relative to these signed
pool generators, the ladder uses the uniform scale `-1/2`: a grouped `+i`
consumer needs `theta_pool = -theta_uccsd/2` with identical excitation order
and reference state. No additional per-double sign compensation is needed;
equal numeric amplitudes still describe different products.

Use this provider only when an occupied/virtual partition fixed by electron
count and `spin=2*S_z` is intended. It does not choose pool elements or
amplitudes, prepare Hartree-Fock, or establish equivalence with the separately
parameterized `uccsd` device kernel.

## Inputs and rejection

All three counts must be integral, non-negative, and not `bool`; fractional,
negative, or boolean values raise `ValueError` with "must be a non-negative
integer". Guards then apply in this order:

| Rejected condition | Exception |
| --- | --- |
| odd `num_qubits` | `ValueError`, "should be even" |
| `num_electrons > num_qubits` | `ValueError` |
| `spin > num_electrons` | `ValueError` |
| odd `(num_electrons-spin)` with `spin > 0` | `ValueError`, "must be even when spin > 0" |
| open-shell alpha occupancy exceeds `num_qubits/2` | `ValueError`, "does not fit" |
| odd `num_electrons` with `spin == 0` | `ValueError`, "spin multiplicity" |

All three host entry points share these guards through
`get_uccsd_excitations`. In particular, `(8,4,1)` raises
`ValueError("(num_electrons - spin) must be even when spin > 0 (spin is
2*S_z)")`, matching the Hartree-Fock occupation builder. The raw `uccsd`
device kernel still computes the beta occupation by floor division without
validation: bypassing the host check would use the `spin=2` partition for
`num_electrons=4, spin=1`. Validate counts with the host helper before launch;
the host rejection does not establish a device error contract.

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
- Independent fermionic oracle: dense Jordan-Wigner ladder matrices pin
  ordered equality and absolute signs at `(8,4,2)`, including mixed doubles,
  within `atol=1e-10` (`:152-187,276-297`). Singles are
  `-i*(a_q^dagger a_p - h.c.)`; doubles are
  `+i*(a_r^dagger a_s^dagger a_q a_p - h.c.)`. A separate absolute check at
  width 4 pins the sorted and one-pair-swapped double signs (`:327-344`).
- Structural checks: exact excitation lists at `(8,4,0)`, open-shell counts at
  `(6,3,1)`, width bounds, and all rejection rows above. The pool-derived dense
  circuit test is supporting but not independent because both sides reuse the
  pool.
- Evidence: the scientific assertions above are derived from cited source/tests;
  they require fresh execution before claiming numerical validation.

Previously fitted pool amplitudes from before the pair-swap sign correction
need negation for doubles with exactly one descending pair to preserve the
same prepared state when rebuilding the pool. Other entries and their order
are unchanged; the direct ladder amplitudes do not need that migration.
