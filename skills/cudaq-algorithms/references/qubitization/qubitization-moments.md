# Qubitization Chebyshev moments

Operation + object: **measure** a **Chebyshev spectral moment**.

## Identity and classification

- Public symbols: `Walk.moment`, `Walk.moments`,
  `cudaq_algorithms.reflection_observable`.
- Source: `python/cudaq_algorithms/qubitization.py`.
- Tests: `test_qubitization.py`, `test_walk_qsvt_orchestration.py`,
  `test_pauli_lcu.py`.
- Kind/role: measurement/readout, computational leaf over a `Walk`.
- Layers: kernel construction, observable construction, `cudaq.observe`,
  classical float/list return.
- Uncertainty: exact expectation on analytic targets or shot/statistics
  dependent on the selected execution mode.

## Scientific contract

`moment(ket, order, *, state_prep=None)` returns
`<T_order(H/alpha)>`, despite the underlying walk-step block being
`-H/alpha`. Do not apply a second caller-side negation.

Even order `2p` measures the geometry-derived reflection observable after `p`
walks and uncomputation. Odd order `2p+1` measures the encoding-specific
`select_observable` after PREPARE and `p` walks without UNPREPARE. An encoding
that lacks a working `select_observable` can support even moments and still
fail for odd moments.

## Inputs and outputs

- Supply exactly one of `ket` and `state_prep`; both or neither raise
  `ValueError`.
- `order` and `count` must be non-negative integers.
- `ket` may be array-like or `cudaq.State`.
- `state_prep` uses the one-register preparation seam; exact width is required,
  but mismatch behavior is unverified rather than guaranteed to fail.
- `moment` returns `float`; `moments(..., count)` returns
  `[<T_0>, ..., <T_{count-1}>]`.

## Composition and boundaries

Requires the zero-flagged block-encoding capability. Odd moments additionally
require a semantically correct `select_observable`; structural protocol
conformance is insufficient. The observable is measured, not extracted from a
statevector, so the interface can be hardware-shaped even though repository
tests use simulators.

`reflection_observable` is `2|0...0><0...0| - I`, while the walk's reflection
gate uses the opposite sign. Keep gate and observable conventions distinct.

## Resources and validation

No independent resource estimator exists. Report `order // 2` walk calls and
one observable evaluation as logical structure only. Shot cost and variance
depend on execution configuration and are not fixed by this API.

Independent oracle: recursively build dense Chebyshev polynomials of
`H/alpha` and compare `<psi|T_k(H/alpha)|psi>`. Include odd/even orders, a
negative eigenvalue, and an encoding without `select_observable`. The cited
tests cover 1- and 2-qubit Hamiltonians. Runnable pointers are the moment cases
in `tests/python/test_qubitization.py` and
`tests/python/test_state_prep_injection.py`. Current public source and tests are
authoritative and must be rechecked at use time; this record was not freshly
executed. [Source lookup](../source-provenance.md) gives shared current-source paths.
