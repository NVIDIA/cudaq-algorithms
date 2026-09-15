# Generalized UCCGSD operator pool

Status: draft. Operation + object: **preprocess** a **generalized UCC singles
and doubles pool**.

## Identity and classification

- Public symbols from `cudaq_algorithms.stateprep`:
  `make_uccgsd_operator_pool(num_qubits, only_singles=False,
  only_doubles=False)` and `get_uccgsd_pauli_lists(num_qubits,
  only_singles=False, only_doubles=False)`.
- Source: `python/cudaq_algorithms/stateprep/_pools.py:209-265,364-382`;
  exports: `python/cudaq_algorithms/stateprep/__init__.py:43-47,59,64`.
- Tests: `tests/python/test_operator_pools.py:60-69,141-228`,
  `tests/python/test_stateprep.py:41-52`, and
  `tests/python/test_stateprep_kernels.py:211-223,263-291`.
- Runnable usage: `docs/sphinx/guide/state_prep.rst:54,66-67` and
  `pytest -q tests/python/test_operator_pools.py -k uccgsd`.
- Source provenance: current public source/tests are authoritative and must be
  checked at use time. This record's historical source review is recorded in
  [Source provenance](../source-provenance.md). Lifecycle is
  draft; execution remains unverified for this record.
- Kind/role/layer: exact deterministic classical transformation,
  computational leaf, host only; dependency `cudaq`.

## Scientific contract

This provider ignores electron count and occupancy. Singles include every
qubit pair `(p,q)` with `p>q`, emitted in ascending `p` then ascending `q`, and
map to `0.5 Y_q Z(q,p) X_p - 0.5 X_q Z(q,p) Y_p`. If the caller interprets
qubits as interleaved spin orbitals, mixed-parity alpha/beta singles are still
present: this pool does not preserve separate alpha/beta particle counts.

For each `a<b<c<d`, doubles generate the three pairings
`((a,b),(c,d))`, `((a,c),(b,d))`, and `((a,d),(b,c))`. Each pair is normalized
to `(high,low)`, the pair-of-pairs to `(min,max)`, inserted into a set, and the
entire set returned in Python sorted-tuple order. That sorted normalized order,
not the 4-combination loop order, is the contract. Each double becomes an
eight-term `0.125` generator with parity strings `Z(q,p)` and `Z(s,r)`.

Use this provider for a generalized all-qubit pool or a singles-/doubles-only
subset. Do not use it when occupancy restrictions, spin-preserving singles, or
paired-only doubles are required; select UCCSD or UpCCGSD instead. Choosing or
optimizing amplitudes is outside the primitive.

## Inputs, rejection, and silent behavior

`num_qubits` is a spin-orbital/qubit count but need not be even. It must be an
integral non-negative non-`bool`; invalid counts raise `ValueError` with "must
be a non-negative integer". There are no other count or spin-layout guards.

The two switches are documented as booleans but are not type-validated; source
uses ordinary truthiness. `only_singles=True` suppresses doubles,
`only_doubles=True` suppresses singles, and both truthy returns `[]` without an
exception. Sizes too small for the requested family also return an empty or
partial pool normally.

## Outputs, converter, and composition

The builder returns the ordered raw object defined by the
[operator-pool Representation Record](operator-pools.md#representation-record--operator-pool).
Singles have two `+/-0.5` terms and doubles eight `+/-0.125` terms. Each
operator touches at most `num_qubits`, but the object does not carry that width.

`get_uccgsd_pauli_lists` returns
`(list[list[cudaq.pauli_word]], list[list[float]])`, one group per operator in
pool order. It pads every word to `num_qubits`; word and coefficient group
lengths match. The shared converter takes
`float(term.evaluate_coefficient().real)`, silently discarding any imaginary
part, and does not prune negligible terms. Packaged operators are supported by
fermionic Hermiticity evidence and have real dyadic coefficients; reachability
of an imaginary coefficient from this builder is unverified. The generic
fixed-parameter converter instead rejects material imaginary coefficients and
can prune negligible real terms.

Composition is positional: retain width and provide one parameter for every
group. Pool-derived device use additionally needs a reference determinant.
The packaged [`uccgsd` kernel](state-preparation-kernel-uccgsd.md) accepts
grouped words and coefficients, not the raw pool. No capability ID applies;
this is direct object conversion and consumer composition. Composite protocol:
not applicable, leaf operation.

## Accuracy, resources, and limitations

The exact pool count is `C(n,2)+3*C(n,4)`; singles-only is `C(n,2)` and
doubles-only is `3*C(n,4)`. Committed assertions pin `9/6/3` at `n=4` and
`238` total at `n=8`; the `n=4` converter group sizes are six groups of two and
three groups of eight. The set removes no scientifically distinct pairing; it
serves as the ordering device. These exact combinatorial host-output counts are
not gates, depth, runtime, or memory, and no pool resource estimator exists.

No truncation, screening, occupancy constraint, or spin adaptation is applied.
Per-operator overall signs outside source arithmetic are not independently
pinned. Cross-implementation agreement with the absent C++ sources named by the
module docstring remains unverified.

## Validation and evaluation coverage

- Independent oracle: at `n=4`, dense Jordan–Wigner ladder matrices establish a
  full bijection for the combined, singles-only, and doubles-only pools within
  `atol=1e-10` (`test_operator_pools.py:141-228`). It permits either overall
  `+i` or `-i`, so it pins content but not absolute generator sign.
- Structural checks: exact counts through `n=8`, `op.qubit_count <= n`, and
  converter shape/order at `n=4`. The kernel dense-exponential comparison is
  supporting but self-derived because it converts the same pool on both sides.
- Expected boundary: invalid counts reject; both subset switches truthy and
  undersized families return empty without warning.
- Evidence: source and committed assertions inspected during the historical
  last review support `derived` claims; package/CUDA-Q execution and numerical
  validation are `unexecuted` here.
- Eval coverage: authored `operator-pool-selection-boundary` covers selection
  among UCCSD, UCCGSD, and UpCCGSD, including generalized enumeration, output
  order, and the excluded optimization workflow. Baseline and with-skill arms
  have not run. There is no dedicated eval for both switches truthy or odd
  register width.
