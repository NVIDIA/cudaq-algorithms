# UCC parameterization and non-interchangeability

## Parameterization boundary and non-interchangeability

A parameterized kernel *is* a state-preparation primitive: the operation is
`parameters + register -> prepared state`. Choosing or optimizing the parameters
is a consumer workflow, outside this library and this skill. But parameterization
is not interchangeability:

- `uccsd(qubits, thetas, num_electrons, spin)` and
  `fixed_parameter_ucc(qubits, thetas, words, coeffs)` are multi-argument device
  kernels, so **neither is directly injectable** as `state_prep`. Only the
  [HF/UCC factory](state-preparation-hf-ucc.md), the Givens factory, or an explicitly caller-written one-argument
  wrapper kernel is. `uccsd` performs no input validation, so `thetas` must hold
  exactly `get_num_uccsd_parameters(num_qubits, num_electrons, spin)` entries in
  the `get_uccsd_excitations` order.
- Substituting `make_uccsd_operator_pool` plus a group-consuming kernel for
  `uccsd` at the same `thetas` does not reproduce the same state. Three distinct
  committed assertions establish the parts of that statement, and they are
  **not** the same test or the same cases. In the table, `scale` is the
  multiplier `s` in the dense reference `exp(i * s * theta * c * P)` used at
  `tests/python/test_stateprep_kernels.py:105-112`, with `theta` the
  per-excitation amplitude, `c` the real pool coefficient of a term, and `P` its
  Pauli word.

| Claim | Cited assertion | Cases |
| --- | --- | --- |
| the group-consuming `fixed_parameter_ucc` path realizes `scale = +1` against the pool generators | `test_fixed_parameter_ucc_matches_dense_pool_exponential`, `tests/python/test_stateprep_hf_ucc.py:239-264` | `(4,2,0)`, `(6,3,1)`, `(8,4,2)` |
| the other group-consuming kernels also realize `scale = +1` | `tests/python/test_stateprep_kernels.py:211-255` | `uccgsd` at `num_qubits` 4, 6; `upccgsd` at 4, 8; `ceo` at `num_orbitals` 2, 3, 4 |
| the `uccsd` CNOT-ladder circuit instead comes out as `exp(-i * (theta / 2) * c * P)`, i.e. `scale = -1/2`, and negates theta for the double-excitation index patterns `(p < q and r > s)` and `(p > q and r < s)` while the pool operators carry no such sign | `test_uccsd_kernel_matches_dense_exponential` with `_uccsd_circuit_signs`, `tests/python/test_stateprep_kernels.py:139-182` | `(4,2,0)`, `(6,3,1)`, `(8,4,0)`, `(10,5,1)`, `(8,4,2)` |

`fixed_parameter_ucc` is **not** exercised in `test_stateprep_kernels.py`; its
evidence is the `test_stateprep_hf_ucc.py` case in the first row. That is tested
evidence of **non-interchangeability for the listed cases only**: do not promote
it into a general amplitude-conversion rule, and do not supply substitution code
that silently changes the prepared state.

**Parameterized-UCC routing.** Runtime-parameterized UCC constructions remain
state-preparation primitives but are separate from the injectable factory
seam. Read the focused contracts for
[`uccsd`](state-preparation-kernel-uccsd.md) and
[`fixed_parameter_ucc`](state-preparation-kernel-fixed-parameter-ucc.md), and
route `uccgsd`, `upccgsd`, and `ceo` through the
[state-preparation family selector](state-preparation.md). Their
lower-level exported building blocks are
[`single_excitation`](state-preparation-kernel-single-excitation.md) and
[`double_excitation`](state-preparation-kernel-double-excitation.md). None of
these multi-argument kernels directly satisfies the one-register `state_prep`
representation.
