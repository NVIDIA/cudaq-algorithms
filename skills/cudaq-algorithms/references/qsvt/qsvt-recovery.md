# QSVT real-time-evolution recovery

Status: draft. Operation + object: **reconstruct** a **real-time-evolved
statevector from cosine and sine good-subspace components**.

## Identity and classification

- Public symbol: `cudaq_algorithms.recover_real_time_evolution`.
- Source: `python/cudaq_algorithms/qsvt.py`.
- Tests: `test_qsvt.py`; examples named in
  [qsvt-sequence.md](qsvt-sequence.md).
- Kind/role/abstraction: classical transformation, computational leaf.
- Execution: host post-processing of simulated good-subspace arrays.
- Dependencies: NumPy; QSPPACK is optional and only supplies example phases.

## Scientific contract

```python
recover_real_time_evolution(cos_state, sin_state, cos_phases, sin_phases)
```

The inputs are two good-subspace statevectors produced by QSP-convention QSVT
sequences for cosine and sine components. The helper removes each circuit's
global phase `exp(i * sum(phases))` and returns:

```text
2 * (real(corrected_cos_state) + i * imag(corrected_sin_state))
```

Both components and both raw phase lists are required. One transformed
statevector is insufficient.

The documented applicability is real Hamiltonians and real input states, where
the desired parts occupy those real/imaginary components. Complex Hamiltonians
or complex inputs are unverified and must not be silently generalized.

## Inputs, outputs, and composition

- Arrays are converted to `numpy.complex128`; source does not add an explicit
  matching-shape check beyond NumPy broadcasting behavior.
- Phase lists are interpreted in the raw QSP convention used to build the two
  circuits.
- Output is a complex NumPy array with the broadcast result shape.
- Upstream chain: two [QSVT sequences](qsvt-sequence.md), each followed by
  [simulation transform](../simulation/simulation-transform.md) or equivalent correctly
  sliced good-subspace extraction.
- This helper is classical and does not emit a kernel, run a QPU, or perform
  amplitude amplification.

## Accuracy, resources, and validation

Approximation error belongs to the cosine/sine polynomial designs and their
phase generation. The helper adds ordinary floating-point error but states no
bound. It has no quantum resource contract; host array work is not benchmarked.

Independent validation: compare the reconstructed vector against dense
eigendecomposition-based `exp(-i t H)|psi>` after using independently generated
phase sequences. The repository case requires norm error below `1e-8`. Include
a negative case with one missing component and do not claim unsupported complex
input behavior. Runnable pointers are the recovery cases in
`tests/python/test_qsvt.py` and
`docs/sphinx/examples/python/02_hamiltonian_simulation.py`. Current public source
and tests are authoritative and must be rechecked at use time; this record was
not freshly executed. [source-provenance.md](../source-provenance.md) records
historical last-review audit context.

Declared eval coverage: `qsvt-phase-and-recovery-boundary`,
`qsvt-paraphrase-convention`, and the QSVT application cases.
