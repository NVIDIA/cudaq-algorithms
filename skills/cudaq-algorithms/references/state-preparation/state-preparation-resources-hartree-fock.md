# Canonical Hartree-Fock resource estimate

Operation + object: **estimate** a **logical `X`-gate description
for a canonical Hartree-Fock occupation**.

## Identity and classification

- Public function:
  `cudaq_algorithms.stateprep.estimate_hartree_fock_resources(num_qubits,
  num_electrons, spin=0)`.
- Result: frozen
  `cudaq_algorithms.stateprep.HartreeFockResourceEstimate`.
- Source/tests: `python/cudaq_algorithms/stateprep/_hartree_fock.py`,
  `tests/python/test_stateprep_hf_ucc.py`.
- Kind/role/layer: resource estimator, computational leaf, host.
## Input and rejection contract

The function obtains the occupation through
`make_hartree_fock_occupation(num_qubits, num_electrons, spin)` and then calls
the explicit-occupation estimator. All three inputs must be non-negative integer
counts; booleans, negative values, and fractional values raise `ValueError`.
`num_electrons > num_qubits` also raises `ValueError`.

For `spin == 0`, the occupation is the contiguous list
`[0, ..., num_electrons - 1]`; no even-width or electron-parity restriction is
added. For `spin > 0`, the function additionally requires:

- even `num_qubits`;
- `spin <= num_electrons`;
- even `num_electrons - spin`, because `spin` is `2*S_z`; and
- `n_occ_alpha <= num_qubits // 2`, so the requested open-shell occupation fits.

Failure of any condition raises `ValueError`. The `(0, 0, 0)` input is accepted
and yields an empty occupation. Negative spin is not supported by this interface.

## Result formulas and status

For every accepted input, the generated occupation has exactly
`num_electrons` entries. The result is:

| Field | Formula and meaning |
| --- | --- |
| `num_qubits` | validated `num_qubits`; input echo |
| `num_electrons` | validated `num_electrons`; generated occupation length |
| `num_x_gates` | `num_electrons`; exact number of logical occupation-setting `X` operations |

`spin` changes which orbitals are occupied but not these counts, and the result
does not return the occupation itself. The fields are documented as logical
pre-transpilation counts, not bounds or hardware measurements. The function
does not report depth, connectivity, native gates, runtime, memory, or state
accuracy.

## Boundaries and validation

Use
[`estimate_hartree_fock_occupation_resources`](state-preparation-resources-hartree-fock-occupation.md)
when explicit orbital indices are the owned input. Exact integer equality with
the formulas above is the oracle; separately compare the open-shell occupation
with `make_hartree_fock_occupation` when validating spin semantics.

Run the focused source cases from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_stateprep_hf_ucc.py \
  -k 'hartree_fock_host_helpers or hartree_fock_open_shell_occupation'
```
