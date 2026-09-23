# Low-level Trotter apply kernel

Operation + object: **apply** a **Suzuki–Trotter product formula
to a caller-owned live qubit register**.

## Identity and classification

- Public symbol: `cudaq_algorithms.trotter.apply_trotter` (not package-root
  exported).
- Exact device signature:

  ```python
  @cudaq.kernel
  def apply_trotter(
      coefficients: list[float],
      words: list[cudaq.pauli_word],
      time: float,
      steps: int,
      order: int,
      qubits: cudaq.qview,
  ):
      ...
  ```

- Source/tests: `python/cudaq_algorithms/trotter.py`,
  `tests/python/test_trotter.py`.
- Kind/role/layer: quantum operation, computational leaf, device kernel.
- Parameterization: runtime. It mutates `qubits` and returns no host value.
- Source provenance: [Source lookup](../source-provenance.md) gives shared current-source paths.

## Scientific contract

For caller-supplied non-identity terms `H' = sum_i c_i P_i`, let
`dt = time / steps` and `R_i(tau) = exp(-i * tau * c_i * P_i)`. In the
execution-ordered sequences below, the leftmost rotation runs first:

```text
S1(tau) = R_0(tau) ... R_(n-1)(tau)
S2(tau) = R_0(tau/2) ... R_(n-1)(tau/2)
          R_(n-1)(tau/2) ... R_0(tau/2)

order 1: [S1(dt)]^steps
order 2: [S2(dt)]^steps
order 4: [S2(w1*dt) S2(w0*dt) S2(w1*dt)]^steps
```

Here `w1 = 1 / (2 - 2**(1/3))` and `w0 = 1 - 2*w1`; the fourth-order branch is
the Forest–Ruth composition. The negative angle passed to CUDA-Q `exp_pauli`
therefore implements the `exp(-i H' time)` convention. The result is an
approximate product formula unless the relevant terms commute or another
case-specific argument establishes exactness. No error bound or step selector
is provided.

## Inputs, validation, and rejection behavior

Canonical callers obtain `coefficients` and common-width `words` from
[`make_trotter_terms`](trotter-planning.md), preserve their order, allocate
`qubits` at that returned width, and supply finite real `time` and coefficients,
a positive integral `steps`, and `order` in `{1,2,4}`. The planned words contain
only `I/X/Y/Z`, exclude identity terms, and are parallel to the retained
nonzero coefficients.

This device kernel performs no general host validation. Its entire body is
guarded only by:

```text
steps > 0 and len(coefficients) == len(words) and order in {1,2,4}
```

If any guard condition is false, the kernel silently applies no gates and
leaves `qubits` unchanged; it does not raise. Equal empty lists pass the guard
but likewise execute zero rotations. The kernel does not validate finite
`time`, coefficient values, word characters or common width, non-identity
content, or agreement between word width and `qubits.size()`. Direct callers
must validate those properties on the host before launch. Prefer the
host-validating `Trotter.kernel`/`Trotter.state_kernel` factories unless the live
register must be composed inside another kernel.

## Identity, composition, and resources

The signature has no `identity_coefficient` input. Canonical planning removes
identity terms because `c I` contributes the phase `exp(-i*c*time)`, not a
device Pauli rotation. Keep that coefficient separately when the phase matters;
do not pass an all-identity word to this kernel as a substitute. The omitted
phase is observable in controlled or interference-based use, and this package
does not document a controlled-Trotter contract.

No resource estimator is returned by this kernel. For valid lists of length
`n`, it emits exactly `n * steps * {1:1, 2:2, 4:6}[order]` logical
`exp_pauli` calls. This is not decomposed gate count, depth, runtime, memory, or
measured hardware cost. The separate
[raw estimator](trotter-resources-raw.md) reproduces that logical rotation count
and adds a word-weight CNOT proxy without inspecting the emitted circuit.

No reusable capability ID applies: this leaf consumes a concrete live
`cudaq.qview` plus flattened lists. It is not a kernel factory, simulation-only
statevector helper, or resource-estimator interface.

## Oracle and runnable evidence

Independently apply the Pauli rotations to a dense statevector and compare every
supported order, sign, list order, and step count against the kernel output.
Check error scaling on a noncommuting Hamiltonian and exactness for commuting
terms. Separately assert the unchanged input state for each of the three false
guard conditions and the equal-empty-list case.

Run the authoritative cases from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_trotter.py \
  -k 'apply_trotter_kernel'
```
