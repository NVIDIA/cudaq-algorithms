# Paired UpCCGSD operator pool

Status: draft. Operation + object: **preprocess** a **spin-preserving singles
and paired-doubles UpCCGSD pool**.

## Identity and classification

- Public symbols from `cudaq_algorithms.stateprep`:
  `make_upccgsd_operator_pool(num_qubits, only_doubles=False)` and
  `get_upccgsd_pauli_lists(num_qubits, only_doubles=False)`.
- Source: `python/cudaq_algorithms/stateprep/_pools.py:209-250,273-288,364-388`;
  exports: `python/cudaq_algorithms/stateprep/__init__.py:43-47,60,65`.
- Tests: `tests/python/test_operator_pools.py:72-79,141-202,231-252`,
  `tests/python/test_stateprep.py:54-61,72-82`, and
  `tests/python/test_stateprep_kernels.py:226-238,263-291`.
- Runnable usage: `docs/sphinx/guide/state_prep.rst:57,68` and
  `pytest -q tests/python/test_operator_pools.py -k upccgsd`.
- Source provenance: current public source/tests are authoritative and must be
  checked at use time. This record's historical source review is recorded in
  [Source provenance](../source-provenance.md). Lifecycle is
  draft; execution remains unverified for this record.
- Kind/role/layer: exact deterministic classical transformation,
  computational leaf, host only; dependency `cudaq`.

## Scientific contract

For an interleaved alpha-even/beta-odd register, singles start from every pair
`p>q` in ascending `p` then `q` order and retain only `p%2 == q%2`. Thus alpha
mixes only with alpha and beta only with beta. Singles precede all doubles and
map to the same two-term Jordan–Wigner generator as UCCGSD.

For each spatial pair `p<q`, with `p` outer and `q` inner in ascending order,
one paired double is emitted as
`_uccgsd_double(2*q+1, 2*q, 2*p+1, 2*p)`. It moves a complete alpha/beta pair
between spatial orbitals and maps to the eight-term `0.125` generalized-double
generator. This is the `k`-UpCCGSD content named by the authoritative test.

Use this provider when spin-preserving singles and pair excitations are the
desired pool. It is not an occupancy-partitioned UCCSD pool, and it does not
offer arbitrary generalized doubles or a singles-only mode. Selection and
amplitude optimization remain outside the primitive.

## Inputs, rejection, and silent behavior

`num_qubits` counts interleaved spin orbitals, must be an integral non-negative
non-`bool`, and must be even. Invalid count types/ranges raise `ValueError` with
"must be a non-negative integer"; odd widths raise `ValueError` with "expects
an even number of spin orbitals".

`only_doubles` is documented as boolean but is not type-validated; ordinary
truthiness suppresses the singles. Paired doubles are always emitted, so there
is no `only_singles` switch and no combination of switches that suppresses
doubles. Widths with fewer than two spatial orbitals return an empty pool
without warning.

## Outputs, converter, and composition

The builder returns the ordered raw object defined by the
[operator-pool Representation Record](operator-pools.md#representation-record--operator-pool).
Each single has two `+/-0.5` Pauli terms and each paired double eight
`+/-0.125` terms. Operators touch at most `num_qubits`, but no width metadata is
carried.

`get_upccgsd_pauli_lists` returns grouped full-width Pauli words and floats,
one group per operator in pool order, with matching group lengths. Its shared
converter retains all terms and takes only `coefficient.real` without checking
the imaginary part. Packaged-pool coefficients are real under the cited
Hermiticity evidence; production of an imaginary coefficient is unverified.
The generic fixed-parameter converter has stricter imaginary/tolerance rules.

Composition requires the same register width, positional one-parameter-per-
element matching, and a prepared reference determinant for device use. The
packaged [`upccgsd` kernel](state-preparation-kernel-upccgsd.md) consumes
grouped words and coefficients, not a raw pool. No capability ID applies.
Composite protocol: not applicable, leaf operation.

## Accuracy, resources, and limitations

For even `n`, the exact count is `3*C(n/2,2)` overall and `C(n/2,2)` for
`only_doubles=True`. Assertions pin `3/1` at `n=4` and `135/45` at `n=20`;
the `n=4` converter groups contain `[2,2,8]` terms. These are exact
combinatorial host-output counts, not gate count, depth, runtime, or memory.
There is no pool resource estimator.

There is no screening, truncation, occupancy validation, or singles-only
subset. Absolute overall signs are derived from source rather than independently
pinned. Cross-implementation agreement with absent C++ source remains
unverified.

## Validation and evaluation coverage

- Independent oracle: at `n=4`, dense Jordan–Wigner ladder matrices establish a
  full bijection for the complete and doubles-only pools within `atol=1e-10`
  (`test_operator_pools.py:141-202,231-252`). It permits either global `+i` or
  `-i`, so it does not pin absolute generator sign.
- Structural checks: counts through `n=20`, width bounds, converter shape/order,
  and rejection of odd width. The kernel dense-exponential test at `n=4,8` is
  supporting but self-derived from the same pool.
- Evidence: source and committed assertions inspected during the historical
  last review support `derived` claims; package/CUDA-Q execution and numerical
  validation are `unexecuted` here.
- Eval coverage: authored `operator-pool-selection-boundary` explicitly covers
  selection among UCCSD, UCCGSD, and UpCCGSD, including this provider's paired
  semantics, ordered output, and the excluded optimization boundary. Baseline
  and with-skill arms have not run; even-width rejection and doubles-only
  selection remain explicit eval gaps.
