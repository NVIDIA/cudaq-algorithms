# Spin-orbital tensor expansion

Operation + object: **transform** **chemist-notation spatial
integrals into spin-orbital ladder-coefficient tensors**.

## Identity and classification

- Public symbol: `cudaq_algorithms.chemistry.spin_orbital_tensors`.
- Source: `python/cudaq_algorithms/chemistry.py`.
- Tests: `tests/python/test_df_qsvt_bridge.py` and
  `tests/python/test_fcidump.py`.
- Kind/role: classical transformation, computational leaf, host NumPy.
- Input: square `(n,n)` one-body matrix and `(n,n,n,n)` chemist ERI.
- Output: `(2n,2n)` one-body and `(2n,2n,2n,2n)` two-body tensors.
- Exactness: exact index transformation apart from floating point.

## Scientific contract

The spin-orbital order is interleaved: `2p` is alpha/up and `2p+1` is
beta/down. The returned two-body tensor is the coefficient of
`a†_p a†_q a_r a_s` expected by the packaged fermion compilers. The spatial ERI
is reordered and multiplied by `1/2` before populating the four same/mixed-spin
blocks.

By default, the function validates only the ERI against the three generators
of real-orbital eightfold chemist symmetry. Each comparison is
`np.allclose(eri, eri.transpose(axes), atol=1e-8)`, which retains NumPy's
default relative tolerance (`rtol=1e-5`); this is not an absolute-only test.
The function checks `one_body` only for square rank-2 shape and does not test
its symmetry or Hermiticity. Set `validate_symmetry=False` only when the caller
owns a different ERI symmetry contract; this skips ERI validation and does not
establish support for complex integrals.

## Rejection and composition

Reject a non-square one-body matrix or an ERI whose shape does not match
`(n,n,n,n)`. With validation enabled, reject ERI symmetry violations rather
than silently producing a potentially non-Hermitian operator. A square but
nonsymmetric or non-Hermitian `one_body` is accepted by this helper; callers
that require a Hermitian Hamiltonian must validate that property separately.

Consumes `cudaq-algorithms.chemistry-integrals.v1`. Its output may be passed to
an explicit fermion-to-qubit transform. It performs no pruning and adds no
scalar offset.

## Resources and validation

No resource estimator or benchmark is part of the contract. Dense two-body
output stores `(2n)^4` complex entries; treat that as a shape fact, not a fresh
memory measurement.

Validate index-by-index on a small tensor, check spin selection rules and the
`1/2` factor, then compare a downstream dense fermionic Hamiltonian with the
compiled qubit matrix. Include an ERI symmetry-breaking failure and a square
but asymmetric one-body case so the actual validation boundary stays visible.
Runnable usage is in `docs/sphinx/guide/preprocessing.rst`. Current public
source and tests are authoritative and must be rechecked at use time; this
record was not freshly executed.
