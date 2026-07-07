# Suzuki-Trotter Hamiltonian simulation

Product-formula time evolution for Hamiltonians expressed as sums of Pauli
strings, implemented as a pure-Python peer of the LCU/QSVT primitives:
term extraction, host-side planning and ordering, resource estimation, and
the circuit primitive itself. Requires only the `cudaq` Python package.

```
python/cudaq_algorithms/trotter.py     term extraction, plans, resources,
                                       apply_trotter kernel
python/cudaq_algorithms/sim_utils.py   shared simulation-only helpers
                                       (evolve lives here)
tests/python/test_trotter.py           dense-reference test suite
examples/hamiltonian_simulation/
  trotter_chemistry.py                 chemistry-style end-to-end example
```

## Using the primitive

```python
from cudaq_algorithms import trotter

# Flexible Hamiltonian input: SpinOperator, single spin term,
# {"XZI...": coeff} mapping, or (coeff, word) pairs.
plan = trotter.make_trotter_plan(
    hamiltonian, time=0.8, steps=4, order=2,
    ordering=trotter.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING)

plan.kernel()       # ready @cudaq.kernel(): |0...0> -> evolved state
plan.resources()    # TrotterResourceEstimate (rotations, CNOT proxy, ...)
plan.num_terms, plan.identity_coefficient, plan.words, plan.coefficients
```

`make_trotter_plan` extracts and validates the Pauli terms on the host
(dropping identity terms into `identity_coefficient`), applies the
requested term ordering, and returns a frozen `TrotterPlan` whose
`kernel()` factory captures the flattened coefficient/word arrays.

### Product-formula orders

- `order=1` — first-order Trotter: one `exp(-i c_i t/steps P_i)` sweep per
  step; error O(t^2 / steps).
- `order=2` (default) — symmetric second-order (Strang) splitting: a
  half-angle forward sweep followed by a half-angle reverse sweep; error
  O(t^3 / steps^2).
- `order=4` — Forest-Ruth fourth-order formula: three symmetric
  second-order sub-steps with time fractions `FOREST_RUTH_W1`,
  `FOREST_RUTH_W0`, `FOREST_RUTH_W1`; error O(t^5 / steps^4).

The Forest-Ruth weights (`w1 = 1/(2 - 2^(1/3))`, `w0 = 1 - 2*w1`) are
precomputed module constants: CUDA-Q kernels cannot call host-only math
such as cube roots, so the kernel consumes the constants directly.

### Composing inside user kernels

The flattened primitive is the escape hatch for composition with state
preparation or measurement in a custom kernel:

```python
import cudaq
from cudaq_algorithms import trotter

@cudaq.kernel
def my_kernel(coeffs: list[float], words: list[cudaq.pauli_word],
              t: float, steps: int, order: int):
    q = cudaq.qvector(4)
    # ... state preparation ...
    trotter.apply_trotter(coeffs, words, t, steps, order, q)
```

`plan.coefficients` / `plan.words` supply the flattened arrays.

### Identity terms and the global phase

For `H = c I + H'`, the circuit applies the product formula for `H'` only;
`exp(-i c t)` cannot be realized as a circuit on the evolved register. The
phase is an unobservable global phase for a single unconditioned evolution
but a real relative phase for controlled or interference-based algorithms —
`plan.identity_coefficient` reports it so callers can account for it.

## Simulation-only helper

Statevector conveniences live in the shared `cudaq_algorithms.sim_utils`
module (simulation-only: it uses `cudaq.get_state`, which does not exist on
hardware targets). The Trotter-specific helper is `evolve`:

```python
from cudaq_algorithms import sim_utils

evolved = sim_utils.evolve(plan, ket)   # approximates exp(-i H t)|ket>,
                                        # identity phase included
```

Unlike the circuit primitive, `evolve` reintroduces the identity phase by
default (`include_identity_phase=True`), so the result approximates the
full `exp(-i H t)|ket>`.

## Testing

`tests/python/test_trotter.py` pins correctness against independent dense
references: exact matrix exponentials via diagonalization, and an explicit
Pauli-rotation simulator for per-order product formulas. Coverage includes
kernel interop with flattened arguments, invalid-input handling, per-order
error thresholds, asymptotic error-scaling slope fits (order-p error ~
dt^p), exactness for commuting Hamiltonians, a 14-term 4-qubit
chemistry-style Hamiltonian, identity/global-phase handling, and every
accepted input form of the term-extraction front end.

## Known CUDA-Q Python constraints

Two upstream compiler behaviors shape the implementation:

- `return` inside a Python kernel is silently ignored
  ([cuda-quantum#4845](https://github.com/NVIDIA/cuda-quantum/issues/4845));
  the `apply_trotter` body is a single positively-guarded block instead of
  early-return guards.
- Captured empty lists cannot be marshaled
  ([cuda-quantum#4847](https://github.com/NVIDIA/cuda-quantum/issues/4847));
  identity-only plans special-case their kernel factory.
