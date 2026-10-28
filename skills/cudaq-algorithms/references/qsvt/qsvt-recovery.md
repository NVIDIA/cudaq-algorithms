# QSVT real-time-evolution recovery

Operation + object: **reconstruct** a **real-time-evolved
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

The helper's documented domain is real Hamiltonians and real input states,
where the desired parts occupy those real/imaginary components. Direct use on
blocks from complex inputs is outside that domain; use the caller composition
below for a real Hamiltonian. Complex Hamiltonians remain unverified.

## Complex-input composition for a real Hamiltonian

The helper's real/imaginary extraction is not complex-linear: applying it
directly to blocks from a complex ket can fail even for correctly calibrated
phases. For a verified real Hamiltonian, split a normalized ket into real and
imaginary components, normalize each nonzero component before circuit execution,
restore its weight afterward, and recombine. Put this caller-side recipe in a
`.py` file; it does not extend the helper's domain to complex Hamiltonians.

```python
import numpy as np
from cudaq_algorithms import PhaseSequence, recover_real_time_evolution
from cudaq_algorithms import sim_utils as sim

def evolve_complex_input(transformer, psi, cos_phases, sin_phases):
    """Caller composition for a real Hamiltonian and normalized complex ket."""
    psi = np.asarray(psi, dtype=np.complex128)
    cos_seq = PhaseSequence(cos_phases, convention="qsp")
    sin_seq = PhaseSequence(sin_phases, convention="qsp")

    def evolve_real(part):
        weight = np.linalg.norm(part)
        if weight == 0:
            return np.zeros_like(psi)
        unit = np.ascontiguousarray(part / weight, dtype=np.complex128)
        cos_state = sim.transform(transformer, unit, cos_seq)
        sin_state = sim.transform(transformer, unit, sin_seq)
        recovered = recover_real_time_evolution(
            cos_state, sin_state, cos_phases, sin_phases)
        return weight * recovered

    return evolve_real(psi.real) + 1j * evolve_real(psi.imag)
```

Materialize contiguous input arrays before handing slices or eigensolver column
views to CUDA-Q; `state_from` casts precision but does not ensure contiguity.
Keep the supplied phase lists unchanged. They must approximate cosine and sine
for the intended Hamiltonian, normalization, and time; splitting the input does
not establish that approximation. Check the raw complex result against
`scipy.linalg.expm(-1j * H_dense * time) @ psi`, with an independently assembled
real `H_dense` and a tolerance chosen before execution. Do not renormalize or
align global phase for a phase-sensitive vector check.

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
not freshly executed. [Source lookup](../source-provenance.md) gives shared current-source paths.
