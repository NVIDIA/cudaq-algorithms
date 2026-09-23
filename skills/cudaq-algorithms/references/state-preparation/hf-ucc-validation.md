# Hartree-Fock/UCC validation

## Host validation and rejection

Validation and rejection behavior: `validate_fixed_parameter_ucc` runs first,
then the occupation guards; every rejection is a `ValueError`
(`_hartree_fock.py:94-118`, `:196-221`, `:258-279`).

| # | Rejected condition |
| --- | --- |
| 1 | unequal outer lengths of `parameters`, `pauli_words`, `coefficients` |
| 2 | a group whose word count differs from its coefficient count |
| 3 | a `str` word whose width is not `num_qubits`, or containing a character outside `IXYZ`. A `cudaq.pauli_word` exposes no accessor and is trusted (`assumed`) |
| 4 | not exactly one of `num_electrons` / `occupied_orbitals` — `ValueError("provide exactly one of num_electrons or occupied_orbitals")` |
| 5 | `spin != 0` supplied together with `occupied_orbitals`; encode an open-shell reference in `occupied_orbitals` directly |
| 6 | an occupied index that is non-integral or negative (validated before any integer coercion, so `2.5` is rejected rather than truncated to `2`), `>= num_qubits`, or duplicated |
| 7 | from `make_hartree_fock_occupation`, which runs inside the factory: `num_electrons > num_qubits`; odd `num_qubits` when `spin > 0`; `spin > num_electrons`; **`(num_electrons - spin)` odd when `spin > 0`**; an alpha count exceeding the spatial-orbital count `num_qubits // 2` |

The parity guard in row 7 is the easiest of these to miss, because `spin` reads
like a multiplicity: it is `2 * S_z`, so `(num_electrons - spin)` must be even.
The source states that without the guard the beta floor
`(num_electrons - spin) // 2` would silently realize `spin + 1` — `(8, 4, 1)`
would return the same occupation as `(8, 4, 2)` — and the rejection is asserted
at `test_stateprep_hf_ucc.py:149-150` (`_hartree_fock.py:104-111`).

## Independent validation

| Field | Evidence |
| --- | --- |
| Independent oracles | two, both committed in the repository and cited rather than executed: a dense pool exponential built with NumPy/SciPy `expm` in pool order (`tests/python/test_stateprep_hf_ucc.py:65-76`, applied at `:239-264`), and an independent fermionic-generator bijection (`tests/python/test_operator_pools.py:172-231`). The seam itself is covered by `tests/python/test_state_prep_injection.py` — injected kernel versus `cudaq.State`-fed twin agreement, zero-argument sampleability, a no-op preparation on an all-zero register, and non-cross-contamination of two kernels minted by one factory |
| Invariants | the open-shell Hartree-Fock occupation equals the determinant implied by `get_uccsd_excitations` (`test_stateprep_hf_ucc.py:125-135`) |
| Representative cases | the `(num_qubits, num_electrons, spin)` triples in the non-interchangeability table, and `(8, 4, spin=2)` occupying `{0, 1, 2, 4}` |
| Predeclared tolerances | they differ by suite, so read the suite. `test_stateprep_hf_ucc.py:59-63` and `test_stateprep_kernels.py:127-131` select `1e-12` (fp64) or `5e-5` (fp32) from the active simulator precision, gating on `np.dtype(cudaq.complex()) == np.complex64`. `tests/python/test_state_prep_injection.py` does **not** gate on precision: it hard-codes `atol=1e-12` for the statevector comparisons and uses `abs=1e-10` / `atol=1e-10` for the `Walk.moment` / `Walk.moments` path, and `tests/python/conftest.py` honors `CUDAQ_DEFAULT_SIMULATOR` with a `qpp-cpu` fallback rather than forcing fp64. So do not tell a caller that "the repository's tolerances adapt to simulator precision" — that holds for the provider suites, not for the injection suite |
| Expected failure/adversarial cases | the factory rejections — seven `pytest.raises` blocks over six distinct messages (`test_stateprep_hf_ucc.py:335-379`) — and the occupation guards, including the `(num_electrons - spin)` parity rejection (`test_stateprep_hf_ucc.py:140-150`) |
| Reference results | none; each oracle is constructed inside the cited test |
| Evidence status per claim | `derived` from the cited source or committed test assertion, unless labeled `assumed` or `unverified` in place. Nothing here is `measured` |

See the [HF/UCC factory contract](state-preparation-hf-ucc.md) for the factory input representation,
and [UCC parameterization](ucc-parameterization.md) for the separately scoped tested cases.
