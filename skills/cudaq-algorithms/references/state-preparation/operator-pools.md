# Operator pools — family front door

Status: draft. This file routes among independently selectable operator-pool
constructors and owns the shared pool representation. It does not define a
combined construction contract.

## Selection table

| Scientific need | Focused primitive | Governing input and distinguishing boundary |
| --- | --- | --- |
| Occupied-to-virtual UCC singles and doubles for a fixed electron count and spin | [`stateprep.make_uccsd_operator_pool`](operator-pool-uccsd.md) | `num_qubits`, `num_electrons`, `spin`; interleaved spin orbitals; order follows the five UCCSD excitation groups |
| Generalized singles and/or doubles over all qubits | [`stateprep.make_uccgsd_operator_pool`](operator-pool-uccgsd.md) | `num_qubits`; no occupied/virtual partition; mixed-spin singles are included |
| Spin-preserving singles plus paired doubles | [`stateprep.make_upccgsd_operator_pool`](operator-pool-upccgsd.md) | even `num_qubits`; complete spatial-orbital pairs; no singles-only mode |
| Coupled-exchange operators without Jordan–Wigner parity strings | [`stateprep.make_ceo_operator_pool`](operator-pool-ceo.md) | `num_orbitals` counts spatial orbitals, so the pool acts on `2 * num_orbitals` qubits |

All four are exact, deterministic host preprocessors. They return ordered
operator data; they do not select important excitations, optimize amplitudes,
prepare a state, emit a quantum kernel, or estimate gates. ADAPT-VQE-style
selection and VQE-style optimization are caller workflows outside these
primitive contracts. The consumer-side fixed-parameter convention and the
non-equivalence of that path to the `uccsd` device kernel live in
[state-preparation-hf-ucc.md](state-preparation-hf-ucc.md).

## Shared provenance and evidence boundary

- Public namespace: `cudaq_algorithms.stateprep`; the package root exports the
  `stateprep` module, not each pool symbol directly.
- Implementation: `python/cudaq_algorithms/stateprep/_pools.py`; exports:
  `python/cudaq_algorithms/stateprep/__init__.py:43-47`.
- Authoritative tests: `tests/python/test_operator_pools.py`,
  `tests/python/test_stateprep.py`, and
  `tests/python/test_stateprep_kernels.py`.
- Runnable documentation: `docs/sphinx/guide/state_prep.rst`; the UCCSD path
  also has `docs/sphinx/examples/python/hartree_fock_ucc.py`.
- Source provenance and version status: current public source/tests are
  authoritative and must be checked at use time. The records were historically
  source-reviewed, with review provenance recorded in
  [Source provenance](../source-provenance.md) and cite committed tests as
  `derived` evidence from that review; package/CUDA-Q execution and evaluator
  runs remain unverified unless a focused record says otherwise.
- The historically reviewed source docstring called this a port of C++ files
  that were absent from the reviewed source. Cross-implementation agreement is
  therefore unverified; check current public source and the named tests at use
  time.

## Representation Record — operator pool

- **Object and structural form:** the ordered Python `list` returned by a
  `make_*_operator_pool` constructor. Each element is accepted by
  `cudaq.SpinOperator(op)` and is iterable as Pauli terms exposing
  `get_pauli_word(width)` and `evaluate_coefficient()`
  (`_pools.py:364-376`). There is no named pool class.
- **Mathematical meaning:** each element is one Hermitian Pauli-sum generator.
  For the three UCC families it is `G = (+/- i) * (T - T^dagger)` for the
  provider's fermionic excitation `T`. CEO is deliberately non-fermionic and
  is not covered by that Jordan–Wigner statement.
- **Shape, layout, order, dtype, and units:** list position is semantically
  significant because consumers match one amplitude to each element in order.
  Packaged coefficients are real dyadic rationals. Qubit indices use qubit 0 as
  the least-significant qubit and use interleaved alpha-even/beta-odd spin
  orbitals where the provider imposes a spin layout. The CEO constructor's
  input unit is spatial orbitals.
- **Sign and phase:** there is no normalization step. The overall sign relative
  to `i(T-T^dagger)` is a construction convention. Absolute word/coefficient
  pins exist for UCCSD `(4,2,0)` and CEO `num_orbitals=2`; the generalized-pool
  fermionic oracle is deliberately agnostic to the `+/- i` choice.
- **Metadata not carried:** the list carries no construction width, provider,
  spin, electron count, excitation label, or units. `op.qubit_count` can be
  smaller than the construction width when the highest qubit is untouched.
- **Producers:** `make_uccsd_operator_pool`, `make_uccgsd_operator_pool`,
  `make_upccgsd_operator_pool`, and `make_ceo_operator_pool`.
- **Consumers:** `get_fixed_parameter_ucc_pauli_lists` accepts an arbitrary
  iterable pool; the three provider-specific `get_*_pauli_lists` helpers build
  and convert their matching pools internally. The fixed-parameter factory and
  [device kernels](state-preparation-device-kernels.md) consume grouped Pauli
  data rather than this raw list.
- **Composition preconditions:** the caller must retain the construction width,
  preserve list order, provide exactly one consumer parameter per element, and
  prepare the required reference determinant before a pool-derived UCC product.
  CEO requires the explicit conversion `num_qubits = 2 * num_orbitals`.
- **Concrete boundary, not a capability:** no stable capability ID is minted.
  This is an exchanged object shape, not evidence of a reusable semantic
  capability. Width/provider compatibility remains the caller's responsibility.
- **Unsupported or ambiguous forms:** mixing providers or widths is not
  rejected; empty pools are legal and produce a no-op product; complex
  coefficients are outside the demonstrated packaged constructors. The
  provider-specific converters take `coefficient.real` without rejecting an
  imaginary part, whereas the generic fixed-parameter converter rejects one
  above its tolerance and can prune negligible real terms.
- **Observable failure symptoms:** reordering or re-widthing can yield a
  normalized but scientifically wrong state without an exception; treating
  CEO's spatial count as a qubit count doubles the intended register; treating
  a Hermitian pool element as anti-Hermitian introduces an erroneous factor of
  `i` in dense comparisons.
