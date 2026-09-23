# Application composition

Use this guide after routing each requested operation/object pair through
[the catalog](catalog.md). It maps application goals to primitive chains; it
does not define new primitives or promise pairwise compatibility.

## Composition procedure

1. Write the desired mathematical map from input to output.
2. Select one record for each independently selectable step.
3. Match capability IDs where present, then verify concrete signatures,
   representations, widths, ordering, normalization, signs, phases, execution
   layer, and unsupported conditions.
4. Inspect the cited source test or self-checking example for every selected
   seam before adapting code.
5. Keep simulation-only extraction and classical post-processing visible.
6. Choose an independent end-to-end oracle and tolerance before execution.
7. Report component and end-to-end evidence separately.

## Common application chains

| Goal | Primitive chain | Critical boundary |
| --- | --- | --- |
| Apply or inspect a Pauli Hamiltonian block | state preparation (optional) -> `PauliLCU` -> `action` or `good_subspace` for simulation analysis | The flagged block is `H/alpha`; the returned good block is unnormalized. |
| Chebyshev moments | state preparation or `cudaq.State` -> block encoding -> `Walk` -> `moment`/`moments` | Walk circuits encode `-H/alpha`, while the measurement API reports the documented positive `T_k(H/alpha)` convention. Odd moments require `select_observable`. |
| Polynomial spectral transformation | state preparation or `cudaq.State` -> block encoding -> `PhaseSequence` -> `QSVT` -> optional `transform` | QSP phases are doubled to projector phases; account for `exp(i sum(phi))`. |
| Real-time evolution by QSVT | real state -> block encoding -> cosine QSVT and sine QSVT -> two good-subspace vectors -> recovery | Both components are required; the packaged recovery helper is documented only for real Hamiltonians and real inputs. |
| Product-formula evolution | state preparation or statevector -> `Trotter` -> kernel or simulation `evolve` | Identity terms are global phase and absent from device circuits; the simulation helper can restore them. |
| Chemistry Hamiltonian for Pauli algorithms | integral loader -> either (A) `chemistry.qubit_hamiltonian` or (B) `chemistry.spin_orbital_tensors` -> `fermion.jordan_wigner` / `fermion.bravyi_kitaev` -> `PauliLCU`, `Walk`, QSVT, or `Trotter` | `qubit_hamiltonian` consumes spatial tensors and performs spin expansion plus Jordan-Wigner internally. Do not feed it spin-expanded tensors. Pass the loader's scalar offset to `qubit_hamiltonian` or the selected explicit transform. |
| Compressed chemistry approximation | integral loader -> compressed double factorization -> reconstruction/error analysis -> qubit Hamiltonian | `DoubleFactorization` is host data, not a packaged quantum kernel; approximation and optimizer status remain visible. |

## Example adaptation rules

- Start from a repository example or test named in
  [source-provenance.md](source-provenance.md); confirm its selected symbols
  against the current checkout, and do not invent an API from a roadmap
  concept or from a similarly named external package.
- Replace only the scientific inputs needed by the task. Preserve convention
  conversions, validation, target/precision setup, oracle, and tolerance until
  each change has been justified.
- Do not copy example-only classes into the answer as if they were installed
  symbols. If adapting one is requested, label it application code.
- Keep optional dependencies optional. PySCF, Psi4, QSPPACK, and CuPy are not
  unconditional package requirements merely because an example or provider can
  use them.
- A custom object that structurally satisfies `BlockEncoding` still requires
  scientific validation of its `H/alpha` block and each member a consumer uses.
- A one-argument preparation kernel must be checked against each concrete
  consumer. `BlockEncoding` conformance alone does not promise injection.

For implementation, use the shared [workflow](workflow.md#advice-or-implementation).
State input shapes, dtypes, units, ordering, output signatures, and register
allocation explicitly; report component and end-to-end evidence with the target,
precision, version, and tolerance actually used.

## Source-grounded example pointers

- State preparation and injection:
  `docs/sphinx/examples/python/05_state_prep_and_injection.py`.
- Pauli LCU quick start:
  `docs/sphinx/examples/python/01_quickstart_block_encoding.py`.
- Custom block encoding:
  `docs/sphinx/examples/python/06_bring_your_own_encoding.py`.
- QSVT Hamiltonian simulation and matrix inversion:
  `02_hamiltonian_simulation.py`, `07_matrix_inversion_qsvt.py`, and
  `hamiltonian_simulation_qsvt.py` under `docs/sphinx/examples/python/`.
- Double-factorized encoding:
  `04_double_factorization_and_the_protocol.py`, `df_compression_to_qsvt.py`,
  `df_encoding.py`, and `df_block_encoding.py` under that directory; the
  encoding classes are application examples, not packaged providers.
- Trotter end-to-end behavior: the executable reference cases in
  `tests/python/test_trotter.py`, plus `03_chemistry_to_ground_state.py` and
  `trotter_chemistry.py`.
