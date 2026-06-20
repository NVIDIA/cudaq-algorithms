# Suzuki-Trotter Progress And Future Work

This note summarizes the current Suzuki-Trotter work on the
`add_suzuki_trotter` branch and the remaining work needed to make the feature
more useful for quantum chemistry algorithms.

## Current Status

The branch provides a CUDA-Q kernel primitive for applying product-formula
Hamiltonian simulation to a live quantum register:

```python
hamiltonian_simulation.apply_trotter(coefficients, words, time, steps, order, q)
```

The intended workflow is:

1. Build or receive a qubit Hamiltonian as a `cudaq.SpinOperator`.
2. Flatten it on the host with `make_trotter_terms()` or `make_trotter_plan()`.
3. Pass the flattened coefficients and Pauli words into a CUDA-Q kernel.
4. Apply the Trotterized evolution to the system register that holds `|psi>`.

The supported product formulas are:

- first-order Lie-Trotter
- second-order symmetric Suzuki-Trotter
- fourth-order Forest-Ruth composition

The Python planning helper defaults to second order.

## Added In This Pass

### Host-Side Planning

The Python layer now includes:

```python
TrotterOrdering
TrotterPlan
TrotterResourceEstimate
make_trotter_plan(...)
estimate_trotter_resources(...)
```

`make_trotter_plan()` wraps `make_trotter_terms()` and records the simulation
parameters alongside the flattened Hamiltonian data.

### Term Ordering

The planning helper supports:

- `preserve_input`
- `coefficient_magnitude_descending`

This keeps the first planning layer simple while leaving room for more
chemistry-aware term grouping later.

### Resource Estimates

`estimate_trotter_resources()` reports:

- number of non-identity Pauli terms
- number of Trotter steps
- product-formula order
- number of Pauli rotations
- approximate CNOT count
- identity coefficient

The CNOT estimate is a decomposition proxy based on two CNOTs per additional
non-identity Pauli in each Pauli rotation. It is intended for rough comparison,
not backend-accurate compilation accounting.

### Chemistry-Facing Example

A new example demonstrates a 4-qubit chemistry-style Pauli Hamiltonian:

```text
examples/hamiltonian_simulation/trotter_chemistry.py
```

The example:

- hard-codes molecular-style Pauli terms
- builds a Trotter plan
- reports resource estimates
- prepares a nontrivial initial state
- applies `apply_trotter()` inside a CUDA-Q kernel
- compares the evolved state to exact diagonalization with NumPy

### Documentation

The Hamiltonian simulation docs now explain:
