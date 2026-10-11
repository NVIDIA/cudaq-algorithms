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

| Target | Precision | Statevector residual | Use |
| --- | --- | --- | --- |
| `qpp-cpu` | fp64 | ~1e-15 (library tests assert at 1e-10..1e-12) | what `tests/python/conftest.py` pins; the checkpoints in these records |
| `nvidia` (GPU) | fp32 by default | ~1e-7 | fails the 1e-8..1e-10 test tolerances; select `nvidia-fp64` for fp64 |
| default target | machine-dependent (fp32 where the GPU target is the default) | ~1e-7 on an fp32 default | never assume; call `cudaq.set_target(...)` before building anything |

- **Convention.** Tests and examples default to `qpp-cpu` and honor
  `CUDAQ_DEFAULT_SIMULATOR`; `conftest.py` reads that variable and otherwise
  pins `qpp-cpu`. The checkpoints in this skill call
  `cudaq.set_target("qpp-cpu", precision="fp64")` before constructing any
  encoding or kernel.
- **Translation.** `state_from(ket)` builds the `cudaq.State` at the active
  target's precision (`cudaq.complex()`); fp32 simulators reject complex128
  initial-state data with "[sim-state] invalid data precision".
- **Mismatch symptom.** Residuals near 1e-7 on a check expected to agree to
  1e-10: suspect target precision before suspecting the code.
- **Source.** `docs/sphinx/conventions.rst` ("Simulation targets and
  tolerances"); `docs/sphinx/guide/qubitization_qsvt.rst` (target and
  `CUDAQ_DEFAULT_SIMULATOR` paragraph); `tests/python/conftest.py`;
  `python/cudaq_algorithms/common_kernels.py` (`state_from`).

## Comparison discipline

Before comparing two implementations or sources:

1. identify both convention sets;
2. write the explicit translation;
3. select an invariant comparison when possible;
4. fix precision and tolerance;
5. label unresolved differences as unknown.

Prefer spectra, expectation values, reconstructed operators, known identities,
or other invariants over raw arrays when equivalent representations exist.
