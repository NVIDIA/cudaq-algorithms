# QSP and QSVT — family front door

Quantum signal processing (QSP) and quantum singular value transformation
(QSVT): route supplied phase angles, QSPPACK conventions, and polynomial
transformations of encoded operators to the sequence record; route cosine/sine
good-block reconstruction of real-time evolution to recovery.

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| transform / encoded spectrum with a phase sequence | `PhaseSequence`, `QSVT.kernel`, `QSVT.controlled_kernel` | quantum driver; host validation + kernel factory/device kernel | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; optional state preparation | [QSVT sequence](qsvt-sequence.md) |
| reconstruct / real-time state from cosine and sine QSVT blocks | `recover_real_time_evolution` | classical transformation; host | concrete QSVT/good-block inputs | [QSVT recovery](qsvt-recovery.md) |

Simulation-only execution and good-subspace extraction are separate records
under [simulation analysis](../simulation/simulation-analysis.md). Phase generation and
polynomial-degree selection are absent from the installed library.

Shared source is `python/cudaq_algorithms/qsvt.py`; shared tests are
`test_qsvt.py`, `test_walk_qsvt_orchestration.py`, and `test_df_encoding.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [Source lookup](../source-provenance.md) gives shared current-source paths.

## Workflow

1. Specify the physical output: a filtered state's spectral weight, an
   observable, or an evolved state. Define the error metric, phase treatment,
   target domain and any minimum success probability before comparing choices.
2. Construct the encoding and retain its actual `alpha`. Interpret supplied
   angles with their declared convention and directions using the
   [sequence contract](qsvt-sequence.md); metadata validation does not prove
   that the angles approximate the requested function.
3. Evaluate each response at the actual scaled spectrum `lambda/alpha`.
   A changed Hamiltonian, normalization or evolution time can invalidate a
   calibration. Agreement at selected eigenvalues is not a uniform bound.
4. For filtering, preserve the raw block `b`: success is `p = b†b` and a
   band's conditional weight is `b† P_band b / p`. Compare with the unfiltered
   weight; a high conditional weight can accompany a poor successful yield.
   The conditional state is undefined at zero success probability.
5. For evolution, respect the [recovery domain](qsvt-recovery.md): the packaged
   helper uses real Hamiltonians and real inputs. For a real Hamiltonian and
   complex input, a justified linear extension can evolve normalized real and
   imaginary components separately and restore their weights. This is caller
   composition, not a general complex-Hamiltonian recovery contract.

A supplied-filter prediction can use an equivalent signal/operator calculation
when no circuit is requested. A circuit claim additionally needs evidence for
that circuit; a successful approximation should not be inferred from its name.

## Verification

Separate circuit/model agreement from approximation error against the desired
physical operation. For a phase-sensitive vector comparison, include the known
QSP circuit phase and preserve the raw norm; for fidelity or a phase-insensitive
metric, state that choice explicitly. Use precision adequate for the requested
error and interpret gains smaller than numerical uncertainty cautiously.

This independent two-dimensional signal product checks a nonconstant supplied
sequence on a complex superposition. It does not synthesize phases or certify a
particular filter design. Save the block as a `.py` file and run it with the
package importable; the checkpoint selects fp64 and a raw-vector tolerance.

```python
import cudaq
import numpy as np
from cudaq_algorithms import PauliLCU, PhaseSequence, QSVT, sim_utils as sim

cudaq.set_target("qpp-cpu", precision="fp64")
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1., -1.])
H = 0.23*np.eye(2) + 0.41*X - 0.28*Z
enc = PauliLCU([(0.23, "I"), (0.41, "X"), (-0.28, "Z")])
psi = np.array([1., 2.j])/np.sqrt(5.)
phases = [0.13, -0.27, 0.08, 0.19]  # raw QSP, all forward
seq = PhaseSequence(phases, convention="qsp")

def response(x):
    s = np.sqrt(max(0., 1. - x*x))
    step = np.array([[-x, -s], [s, -x]], complex)
    def phase(a):
        return np.diag([np.exp(1j*a), np.exp(-1j*a)])
    matrix = phase(phases[0])
    for a in phases[1:]:
        matrix = phase(a) @ step @ matrix
    return matrix[0, 0]

eigenvalues, V = np.linalg.eigh(H)
a = V.conj().T @ psi
r = np.array([response(x/enc.alpha) for x in eigenvalues])
reference = np.exp(1j*sum(phases)) * (V @ (r*a))
b = sim.transform(QSVT(enc), psi, seq)
error = np.linalg.norm(b - reference)
assert error < 1e-10
p = float(np.vdot(b, b).real)
assert 0. < p <= 1. + 1e-10
band = eigenvalues < 0.
initial = np.sum(np.abs(a[band])**2)
conditional = np.sum(np.abs((V.conj().T @ b)[band])**2)/p
print("raw error, success, initial/conditional band weight:",
      error, p, initial, conditional)
```

Source and further checkpoints: `python/cudaq_algorithms/qsvt.py`,
`tests/python/test_qsvt.py` (signal response and mixed directions), and
`docs/sphinx/examples/python/02_hamiltonian_simulation.py` (its phase generator
uses optional QSPPACK; the checkpoint above does not).
