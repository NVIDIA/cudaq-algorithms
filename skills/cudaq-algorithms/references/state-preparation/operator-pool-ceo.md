# Coupled-exchange-operator pool

Status: draft. Operation + object: **preprocess** a **coupled-exchange operator
pool**.

## Identity and classification

- Public symbols from `cudaq_algorithms.stateprep`:
  `make_ceo_operator_pool(num_orbitals)` and
  `get_ceo_pauli_lists(num_orbitals)`.
- Source: `python/cudaq_algorithms/stateprep/_pools.py:296-356,364-395`;
  exports: `python/cudaq_algorithms/stateprep/__init__.py:21-22,43-47,61,66`.
- Tests: `tests/python/test_operator_pools.py:82-89,255-273`,
  `tests/python/test_stateprep.py:63-69`, and
  `tests/python/test_stateprep_kernels.py:241-255,263-291`.
- Runnable usage: `docs/sphinx/guide/state_prep.rst:58,69,265-267` and
  `pytest -q tests/python/test_operator_pools.py -k ceo`.
- Source provenance: current public source/tests are authoritative and must be
  checked at use time. This record's historical source review is recorded in
  [Source provenance](../source-provenance.md). Lifecycle is
  draft; execution remains unverified for this record.
- Kind/role/layer: exact deterministic classical transformation,
  computational leaf, host only; dependency `cudaq`.

## Scientific contract

`num_orbitals` counts **spatial orbitals**, so this pool acts on
`2*num_orbitals` qubits. This differs from the spin-orbital/qubit units of the
UCC providers. CEO is deliberately non-fermionic: its single is
`0.5 Y_q X_p - 0.5 X_q Y_p`, and its doubles are plain four-Pauli products.
There is no Jordan–Wigner `Z` parity string, so do not characterize this as a
fermionic-excitation pool or apply the UCC fermionic-generator oracle.

Emission order is alpha singles, beta singles, alpha same-spin doubles, beta
same-spin doubles, then mixed doubles. Singles are `(2*i+offset,2*j+offset)`
for `j<i`, with `i` outer and `j` inner; offsets 0 and 1 select alpha and beta.
For each descending spatial 4-combination `i>j>k>l`, same-spin enumeration
emits `(p,q,r,s)`, `(p,r,q,s)`, then `(q,p,r,s)`. Mixed tuples are
`(2*i,2*j+1,2*k,2*l+1)` with `k<i` and `l<j`, in that nested-loop order.

Every double tuple contributes **two distinct operators**, `0.25*op_a` then
`0.25*op_b`, and therefore two consumer parameters. Use CEO only when this
coupled-exchange construction is intended. Pool selection and amplitude
optimization are outside the primitive.

## Inputs, rejection, and silent behavior

The sole count must be integral, non-negative, and not `bool`; negative,
fractional, or boolean input raises `ValueError` with "must be a non-negative
integer". There is no minimum-size rejection. `num_orbitals <= 1` returns an
empty pool, same-spin doubles do not appear before `num_orbitals=4`, and these
degenerate sizes produce no warning. Passing a qubit count as `num_orbitals`
silently doubles the intended register.

## Outputs, converter, and composition

The builder returns the ordered raw object defined by the
[operator-pool Representation Record](operator-pools.md#representation-record--operator-pool).
Singles have two `+/-0.5` terms. Each of the two operators produced by a double
has four `+/-0.25` terms. The raw list carries neither spatial count nor its
derived `2*num_orbitals` width.

`get_ceo_pauli_lists` returns grouped full-width Pauli words and floats, one
group per pool operator, and explicitly passes width `2*num_orbitals` to the
shared converter. Word/coefficient lengths match. The converter retains all
terms and silently takes `coefficient.real`; packaged CEO coefficients are
real dyadic values in the absolute oracle, while reachability of an imaginary
coefficient elsewhere is unverified. The generic converter has different
imaginary/tolerance behavior.

Composition requires an explicit spatial-to-spin width conversion, preservation
of order, one parameter per pool element, and a prepared reference determinant
for device use. The packaged [`ceo` kernel](state-preparation-kernel-ceo.md)
consumes grouped words and coefficients, not the raw list. No capability ID
applies. Composite protocol: not applicable, leaf operation.

## Accuracy, resources, and limitations

For `M=num_orbitals`, the exact count is
`2*C(M,2) + 12*C(M,4) + 2*C(M,2)^2`. Assertions pin `4` at `M=2` and `96` at
`M=4`; at `M=2`, group sizes are `[2,2,4,4]`. These are exact combinatorial
host-output counts, not gate count, depth, runtime, or memory. There is no pool
resource estimator, truncation, screening, or physical-reference validation.

The source cites arXiv:2407.08696, but the paper was not consulted for this
record; literature agreement is unverified. The dense kernel test derives its
reference from the same pool and cannot independently catch a count-preserving
word or sign error.

## Validation and evaluation coverage

- Independent oracle: an absolute known answer at `M=2` pins all four
  operators, Pauli words, coefficients, order, and signs exactly
  (`test_operator_pools.py:255-273`). There is intentionally no fermionic
  Jordan–Wigner oracle.
- Structural checks: exact counts at `M=2,4`, width bounds, converter grouping,
  invalid-count rejection, and kernel dense-exponential agreement at `M=2,3,4`.
  The latter is supporting, not independent, because it reuses the same pool.
- Evidence: source and committed assertions inspected during the historical
  last review support `derived` claims; package/CUDA-Q execution and numerical
  validation are `unexecuted` here.
- Eval coverage: authored `operator-pool-ceo-units` covers the spatial-orbital
  unit, doubled qubit width, and rejection of the false Jordan–Wigner/parity
  characterization. Baseline and with-skill arms have not run. Empty-pool
  behavior has no dedicated eval.
