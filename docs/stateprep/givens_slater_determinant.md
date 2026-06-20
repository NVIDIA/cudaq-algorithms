# Givens Slater determinant state preparation

This state-preparation path prepares a particle-number-conserving Slater determinant from an occupied-orbital coefficient matrix. The input matrix has shape `num_orbitals x num_electrons`; each column is one occupied spin orbital expressed in the computational spin-orbital basis. Columns must be orthonormal.

The host API builds a validated plan:

```python
plan = algorithms.stateprep.make_slater_determinant_plan(occupied_orbitals)
resources = algorithms.stateprep.estimate_givens_stateprep_resources(plan)
```

The plan contains the flattened data expected by CUDA-Q kernels:

- `num_orbitals`
- `num_electrons`
- `is_complex`
- `orbital_indices`
- `angles`
- `phases`
- `final_phases`

For real inputs, call `prepare_slater_determinant` from a kernel. For complex inputs, call `prepare_complex_slater_determinant`.

```python
@cudaq.kernel
def kernel(indices: list[int], angles: list[float]):
    q = cudaq.qvector(plan.num_orbitals)
    algorithms.stateprep.prepare_slater_determinant(
        q, indices, angles, plan.num_electrons)
```

## Conventions

Orbital `i` maps to qubit `i`. The prepared computational-basis determinant starts from the first `num_electrons` qubits occupied, then applies the inverse Givens-elimination schedule. Basis amplitudes follow the usual determinant sign convention: the amplitude for an occupied basis set is the determinant of the corresponding rows of the occupied-orbital matrix.

The implementation assumes a Jordan-Wigner-style occupation ordering. It does not currently perform a fermion-to-qubit mapping step; callers should provide the occupied-orbital matrix in the spin-orbital/qubit ordering they intend to use.

Complex schedules include per-rotation phases and final phases on the initially occupied orbitals. The device helper implements the phase-aware rotation using the real Givens rotation plus `rz(phase)`, which is equivalent to `exp(i * phase * n)` up to a global phase.

## Validation and kernel assumptions

The plan builder validates that the occupied-orbital matrix is rectangular, normalized, orthogonal, non-empty, and has no more electrons than orbitals. `validate_slater_determinant_plan(plan)` additionally checks that flattened schedules have consistent lengths, in-range orbital indices, and adjacent rotations.

The lower-level device kernels are intentionally minimal and may return without applying anything when flattened inputs are malformed. Users should normally enter through `make_slater_determinant_plan`, then pass the plan fields to the kernel.

## Resource estimates

`estimate_givens_stateprep_resources(plan)` reports:

- number of Givens rotations
- number of two-qubit `exp_pauli` calls
- number of phase rotations for complex state preparation
- a two-qubit gate-count proxy
- a simple primitive-depth proxy

These are implementation-level estimates, not hardware-calibrated resource estimates.

## Current scope

This feature supports particle-number-conserving Slater determinant preparation. It is not yet a full fermionic Gaussian-state or Bogoliubov-transform implementation with particle-hole mixing. Chemistry-package integration is intentionally kept outside the core API; a typical workflow is to generate molecular orbitals or occupied-orbital coefficients in Python with an external chemistry package, then pass the resulting array into this state-preparation primitive.

See `examples/stateprep/givens_slater_determinant.py` for a simulator validation example that compares CUDA-Q state preparation against a NumPy determinant reference.
