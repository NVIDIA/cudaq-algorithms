# Numerical Comparison

## Precision, Hermiticity, and tolerances

- **Convention.** Establish dtype, target precision, and tolerance before a
  numerical comparison. Coefficient-pruning tolerances change the represented
  operator; comparison tolerances only judge outputs. A routine accepting
  non-Hermitian data must not be described as enforcing Hermiticity.
- **Rule.** Report global-phase-insensitive comparisons explicitly, and use
  invariant checks such as reconstructed operators, spectra, or expectation
  values when raw representations differ. For a raw block or phase-sensitive
  amplitude, compare against an independent dense action without
  renormalization or phase alignment ([validation](../validation.md), "Match
  the check to the scientific claim").
- **Source.** `docs/sphinx/conventions.rst` ("Simulation targets and
  tolerances"); fermion-transform, chemistry, QSVT, and validation family
  evidence linked from the catalog.

## Simulation target and achievable precision

| Target | Precision | Use |
| --- | --- | --- |
| `qpp-cpu` | fp64 | Default in `tests/python/conftest.py` unless overridden; the small reference checks commonly use 1e-10..1e-12 tolerances |
| `nvidia` (GPU) | fp32 by default | Select `nvidia-fp64` when fp64 is required and supported in the environment |
| default target | environment-dependent | Inspect or explicitly select the target before constructing states or kernels |

Machine epsilon is approximately `1.2e-7` for fp32 and `2.2e-16` for fp64;
these are arithmetic scales, not bounds on statevector or observable error.
Residuals depend on circuit depth, conditioning, accumulation, and the chosen
metric. Neither fp32 failure at a given tolerance nor fp64 agreement near
machine epsilon is universal. Establish a tolerance against an independent
reference for the actual calculation.

- **Convention.** Tests and examples default to `qpp-cpu` and honor
  `CUDAQ_DEFAULT_SIMULATOR`; `conftest.py` reads that variable and otherwise
  pins `qpp-cpu`. The checkpoints in this skill call
  `cudaq.set_target("qpp-cpu", precision="fp64")` before constructing any
  encoding or kernel.
- **Translation.** `state_from(ket)` builds the `cudaq.State` at the active
  target's precision (`cudaq.complex()`); fp32 simulators reject complex128
  initial-state data with "[sim-state] invalid data precision".
- **Mismatch symptom.** Residuals near 1e-7 on a check expected to agree to
  1e-10: inspect target precision alongside algorithmic error and conditioning.
- **Source.** `docs/sphinx/conventions.rst` ("Simulation targets and
  tolerances"); `docs/sphinx/guide/qubitization_qsvt.rst` (target and
  `CUDAQ_DEFAULT_SIMULATOR` paragraph); `tests/python/conftest.py`;
  `python/cudaq_algorithms/common_kernels.py` (`state_from`).

## Statevector feasibility

Count every live system, ancilla, work, and counting qubit. A dense statevector
on `n` total qubits needs at least `itemsize * 2**n` bytes: complex64 uses 8
bytes per amplitude and complex128 uses 16. For example, 30 total qubits
require 16 GiB for one complex128 vector, before simulator workspace, copies,
or density matrices. A dense Hamiltonian needs `itemsize * 4**n` bytes.
Check the actual representation and available memory; there is no universal
qubit cutoff. Use sparse/sector references or reduced validation cases when
the full calculation does not fit, and state which requested size remains
unverified.

## Comparison discipline

Before comparing two implementations or sources:

1. identify both convention sets;
2. write the explicit translation;
3. select an invariant comparison when possible;
4. fix precision and tolerance;
5. label unresolved differences as unknown.

Prefer spectra, expectation values, reconstructed operators, known identities,
or other invariants over raw arrays when equivalent representations exist.
