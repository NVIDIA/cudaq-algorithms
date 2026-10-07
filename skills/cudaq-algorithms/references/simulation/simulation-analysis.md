# Simulation analysis - family front door

These helpers are packaged but statevector-oriented; none is a
QPU substitute for the hardware-shaped kernel or observable contracts.

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| extract / zero-ancilla amplitude block | `sim_utils.good_subspace` | simulation analysis; host array slicing | concrete geometry | [good subspace](simulation-good-subspace.md) |
| analyze / `(H/alpha)|ket>` | `sim_utils.action` | simulation analysis; `cudaq.get_state` | concrete `PauliLCU.encode_kernel` | [action](simulation-action.md) |
| analyze / QSVT good-subspace vector | `sim_utils.transform` | simulation analysis; `cudaq.get_state` | concrete QSVT | [transform](simulation-transform.md) |
| evolve / Trotter statevector | `sim_utils.evolve` | simulation analysis; `cudaq.get_state` | concrete Trotter | [evolve](simulation-evolve.md) |

Shared source is `python/cudaq_algorithms/sim_utils.py`. The module depends on
`cudaq.get_state` except for the pure slicing helper. Current public source and
tests are authoritative and must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.

## Workflow

1. Identify the requested object: raw operator action, success probability,
   conditional state/observable, or the system state after discarding ancillas.
   Select the helper above for that object; obtain the full dilation or its
   equivalent channel when an unconditioned reduced state is needed.
2. Normalize physical input states and declare basis/register order. Retain the
   actual encoded scale: `H/alpha` and `H` have different branch probabilities.
   The [good-subspace contract](simulation-good-subspace.md) identifies the
   packaged zero-ancilla slice; other layouts require their own indexing.
3. Keep the raw branch `b` and `p = b^dagger b`. At `p > 0`, a conditional expectation
   is `b^dagger O b / p`; a conditional projector probability is `b^dagger P b / p` and its
   joint success-and-outcome probability is `b^dagger P b`. At `p = 0` the conditional
   state is undefined. Immediate normalization loses successful-yield information.
4. Discarding ancillas means tracing over every outcome. For the standard
   PREPARE/SELECT/UNPREPARE Pauli LCU of `H = sum_j c_j P_j`, the same reduced
   state is `sum_j |c_j|/alpha * P_j rho P_j^dagger`. UNPREPARE acts only on discarded
   ancillas, so it cannot change that reduced state. Coefficient signs cancel
   here but still interfere in the selected branch. This identity does not
   characterize an arbitrary dilation specified only by its encoded block.
5. Use equivalent operator/channel calculations for physical predictions when
   a circuit is not requested. Label what was calculated or simulated; array
   access is not a shot experiment or a measured hardware success rate.

## Verification

Check a raw action against an independent dense operator before using its norm
as a probability. For reduced states, check trace, Hermiticity, positivity and
purity within numerical precision. Agreement on one observable does not prove
state equality. Separate floating-point error, approximation error and sampling
uncertainty; choose tolerances for the requested output and active precision.
For phase-sensitive vector comparisons, declare any phase convention explicitly.

Save this fp64 checkpoint as a `.py` file with the package importable. It compares
an executed dilation with an independent Pauli channel and raw dense action;
the partial trace follows system-first, little-endian register allocation.

```python
import cudaq
import numpy as np
from cudaq_algorithms import PauliLCU, state_from, sim_utils as sim

cudaq.set_target("qpp-cpu", precision="fp64")
I = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1., -1.])
enc = PauliLCU([(0.18, "I"), (0.46, "X"), (-0.31, "Z")])
H = 0.18*I + 0.46*X - 0.31*Z
psi = np.array([1., 0.7 + 0.4j])
psi /= np.linalg.norm(psi)
full = np.asarray(cudaq.get_state(enc.encode_kernel(), state_from(psi)))
b = sim.good_subspace(enc, full)
reference = H @ psi / enc.alpha
p = float(np.vdot(b, b).real)
blocks = full.reshape(2**enc.num_ancilla, 2**enc.num_system)
rho = blocks.T @ blocks.conj()  # sum_a |b_a><b_a|
rho_in = np.outer(psi, psi.conj())
channel = (0.18*rho_in + 0.46*X @ rho_in @ X
           + 0.31*Z @ rho_in @ Z)/enc.alpha
raw_error = np.linalg.norm(b - reference)
rho_error = np.linalg.norm(rho - channel)
assert max(raw_error, rho_error) < 1e-10
assert abs(p - np.vdot(reference, reference).real) < 1e-10
assert abs(np.trace(rho) - 1.) < 1e-10
assert np.linalg.norm(rho - rho.conj().T) < 1e-10
assert np.linalg.eigvalsh(rho).min() >= -1e-10
assert 0. < p <= 1. + 1e-10
conditional = float(np.vdot(b, Z @ b).real/p)
unconditioned = float(np.trace(rho @ Z).real)
purity = float(np.trace(rho @ rho).real)
assert 0.5 - 1e-10 <= purity <= 1. + 1e-10
print("raw/channel errors:", raw_error, rho_error)
print("success, conditional/discarded Z, discarded purity:",
      p, conditional, unconditioned, purity)
```

Source and further checkpoints: `python/cudaq_algorithms/sim_utils.py`,
`python/cudaq_algorithms/pauli_lcu.py`, `tests/python/test_pauli_lcu.py`, and
`docs/sphinx/examples/python/pauli_lcu_demo.py` (raw action and full state).
