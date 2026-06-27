# Hamiltonian Simulation

CUDA-Q Algorithms provides Hamiltonian simulation primitives rather than a
complete quantum chemistry workflow. Users are expected to provide a qubit
Hamiltonian, usually after generating molecular integrals and mapping the
fermionic Hamiltonian to Pauli operators.

The current Suzuki-Trotter support is centered on two steps:

1. Flatten a `cudaq::spin_op` on the host.
2. Apply the product-formula circuit inside a CUDA-Q kernel.

## Python Workflow

```python
from cudaq import spin
from cudaq_algorithms import hamiltonian_simulation

H = 0.7 * spin.x(0) + 0.4 * spin.z(1) + 0.2 * spin.x(0) * spin.z(1)

plan = hamiltonian_simulation.make_trotter_plan(
    H, time=0.8, steps=4, order=2
)
resources = hamiltonian_simulation.estimate_trotter_resources(plan)
```

The plan contains:

- `coefficients`: real Pauli coefficients
- `words`: padded `cudaq.pauli_word` values
- `identity_coefficient`: coefficient of the identity term
- `num_qubits`: qubit extent of the Hamiltonian
- `time`, `steps`, `order`, and `ordering`

Inside a kernel:

```python
@cudaq.kernel
def evolve(coefficients: list[float], words: list[cudaq.pauli_word],
           time: float, steps: int, order: int):
    q = cudaq.qvector(2)
    h(q[0])
    hamiltonian_simulation.apply_trotter(coefficients, words, time, steps,
                                         order, q)
```

## Controlled Evolution Status

Controlled Trotter evolution is still future work. It is needed for phase
estimation, Hadamard tests, Krylov moments, and related interference-based
workflows. The current CUDA-Q kernel-lowering path does not yet provide a clean
generic way to apply controlled `exp_pauli()` to a runtime-sized `qview<>`.
Until that exists, callers should build controlled evolutions directly in
problem-specific kernels.

## Orders

Supported orders are:

- `1`: first-order Lie-Trotter
- `2`: second-order symmetric Suzuki-Trotter
- `4`: fourth-order Forest-Ruth composition

The default order in the Python planning helper is `2`.

## Term Ordering

The planning helper currently supports:

- `preserve_input`
- `coefficient_magnitude_descending`

More advanced commuting-group planning is intentionally left for future work.

## Invalid Kernel Arguments

`apply_trotter()` is a QPU-facing primitive. Invalid runtime inputs are treated
as no-ops: `steps == 0`, mismatched coefficient/Pauli-word lengths, or an order
other than `1`, `2`, or `4` leaves the register unchanged. Prefer constructing
inputs with `make_trotter_plan()`, which validates these options on the host.

## Simulation Validation

Examples and tests may use `cudaq.get_state()` to compare against exact
diagonalization. Library APIs do not depend on `get_state()`; it is a simulator
validation tool, not an algorithmic primitive.
