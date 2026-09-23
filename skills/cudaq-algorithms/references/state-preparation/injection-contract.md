# State-preparation injection contract

## Family scope and source provenance

- Owner: CUDA-Q Algorithms Team.
- Family scope: prepare or transform a many-fermion state through either a
  factory-emitted one-argument kernel or a documented raw device kernel. The
  Capability Record below applies only to the one-argument injection seam.
  Choosing or optimizing amplitudes is a consumer workflow, out of scope
  (`../../SKILL.md`).
- Source paths: `python/cudaq_algorithms/stateprep/` (`_givens.py`,
  `_hartree_fock.py`, `_kernels.py`, `_pools.py`); the four consumer modules
  tabulated in the Capability Record below;
  `python/cudaq_algorithms/block_encoding.py`, cited below for a non-implication
  and **not** a consumer of this seam.
- Authoritative tests: `tests/python/test_state_prep_injection.py` for the seam,
  plus the per-provider suites each primitive record lists.
- Authoritative documentation: `docs/sphinx/guide/state_prep.rst`,
  `docs/sphinx/conventions.rst`; example
  `docs/sphinx/examples/python/05_state_prep_and_injection.py`.
- Current public source/tests are authoritative and must be checked at use time.
  Cited repository assertions provide derived evidence until executed in the current task.

---

## Representation Record — the injectable preparation kernel

- Object name and canonical symbol: the one-argument preparation kernel, passed
  as the `state_prep` argument of a consumer factory.
- Public type or structural form, and source path: a compiled CUDA-Q kernel
  whose only parameter is `(qubits: cudaq.qview)`. There is no named public
  type; the form is established by the two factories
  (`python/cudaq_algorithms/stateprep/_givens.py:367`,
  `_hartree_fock.py:240-320`) and the consumers' `state_prep` parameters.
- Mathematical meaning: a unitary acting on the register it is handed. Applied
  to `|0...0>` it produces the provider's target state; a degenerate
  `pass`-bodied provider leaves the register in `|0...0>`.
- Shape, layout, ordering, dtype, and units: width is baked in at factory time
  (`schedule.num_spin_orbitals` or `num_qubits`); Jordan-Wigner layout with
  qubit 0 least significant and interleaved alpha-even / beta-odd spin orbitals
  (`../conventions.md`). The object carries no dtype or units of its own.
- Normalization, sign, and phase convention: unitary, so norm-preserving. The
  seam fixes no phase; each primitive record states its own phase contract.
- Required mathematical properties (applicability preconditions): a register
  of exactly the kernel's width, in `|0...0>` per both providers' contract.
- Producers: the packaged factories
  `slater_determinant_kernel` and `hartree_fock_ucc_kernel`, plus caller-written
  one-argument kernels such as `product_prep` and `noop_prep` in
  `tests/python/test_state_prep_injection.py:24-33`.
- Consumers: the four modules tabulated in the Capability Record below.
- Invariants preserved across the boundary: all problem data is captured inside
  the kernel at factory time — the *data erasure at the kernel boundary*
  contract of `python/cudaq_algorithms/block_encoding.py` — so the signature
  carries registers only and no status channel exists.
- Observable symptom of a misinterpretation: a multi-argument kernel offered
  here is not the one-register injection representation. A wrong-width kernel
  may fail, no-op on part of the register, or prepare a plausible wrong state;
  the outcome is provider/consumer-dependent and unverified.
- Unsupported or ambiguous forms: multi-argument device kernels — a different
  representation routed through the
  [state-preparation family selector](state-preparation.md),
  not a variant of this one; kernels that allocate ancillas,
  measure, or reset (none is packaged, and the capability is unitary-only).
- Source paths, tests, docs, and provenance: "Family scope and source
  provenance" above.

---

## Capability Record — unitary state preparation

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1` (documentation identifier).

- Contract type: documentation/taxonomy only — not a Python `Protocol`, not an
  ABC, not a public symbol. Extraction is justified because **two packaged
  providers and four independent consumer modules** already exchange it.
- Boundary representation and exact signature: the Representation Record
  above — a kernel with the single parameter `(qubits: cudaq.qview)`.
- Semantic invariants, `derived` from the `state_prep` call sites and the
  documentation:
  1. **Fresh register in `|0...0>`, allocated by the calling consumer**, handed
     to `state_prep(system)` before any ancilla, signal, or control register
     exists, so preparation precedes the consumer operation on the same
     register and receives the system register only.
  2. **Exact width match required, unenforced at factory time.** The
     preparation acts only on the register handed to it, of width `num_qubits`
     (Trotter) or `num_system` (encodings). No general factory-time check exists.
     A mismatch can fail or silently prepare the wrong state depending on which
     indices the provider touches; no uniform runtime behavior is verified.
  3. **Zero-argument consumer mode.** Given `state_prep`, the consumer
     factories return kernels taking no arguments; without it, the
     encoding/Walk/QSVT factories return a kernel taking one `cudaq.State`.
     `Trotter.kernel` is zero-argument either way — it evolves `|0...0>` with
     no preparation — and its `cudaq.State` twin is `Trotter.state_kernel`.
  4. **Unitary only.** No packaged provider kernel allocates an ancilla,
     measures, or resets. A `pass`-bodied kernel is a legal degenerate provider.
  5. **Controlled consumers run the preparation once, uncontrolled**, before
     the control register exists — a consumer sequencing choice, not evidence
     of a controlled-preparation capability.
- Register geometry and ownership: the current consumer allocates the fresh system
  register and requires exactly its own system width (invariants 1 and 2). This
  source-derived behavior does not establish a universal or future ownership policy.
  Recheck current public source before relying on it.
- Convention requirements: the qubit-ordering, Pauli-word, and interleaved
  spin-orbital conventions of `../conventions.md`.
- Host/device/simulation boundary: providers validate and flatten data on the
  host at factory time; the emitted kernel is pure device code, with no
  simulation-only dependency.
- Providers, with source paths: `slater_determinant_kernel`
  (`python/cudaq_algorithms/stateprep/_givens.py`, contract in
  [state-preparation-slater-determinant-kernel.md](state-preparation-slater-determinant-kernel.md));
  `hartree_fock_ucc_kernel`
  (`python/cudaq_algorithms/stateprep/_hartree_fock.py`, contract in
  [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md)); caller-written
  one-argument kernels.
- Consumers, with source paths — **four modules**, twelve public methods:

| Consumer module | Symbols |
| --- | --- |
| `python/cudaq_algorithms/pauli_lcu.py` | `PauliLCU.encode_kernel`, `PauliLCU.walk_kernel` |
| `python/cudaq_algorithms/qubitization.py` | `Walk.kernel`, `Walk.adjoint_kernel`, `Walk.roundtrip_kernel`, `Walk.controlled_kernel`, `Walk.controlled_roundtrip_kernel`; `Walk.moment`, `Walk.moments` (measurement/readout) |
| `python/cudaq_algorithms/qsvt.py` | `QSVT.kernel`, `QSVT.controlled_kernel` |
| `python/cudaq_algorithms/trotter.py` | `Trotter.kernel` |

  `Walk.moment` / `Walk.moments` are the classical-readout consumers: they
  return Chebyshev moments of `H/alpha` rather than a kernel, and raise
  `ValueError("provide exactly one of ket or state_prep")` when both or neither
  input mode is given, so measurement and classical post-processing stay visible
  here instead of folding into the kernel table. Two asymmetries, reported
  rather than smoothed over: `Trotter.kernel` types the parameter `Any | None`
  while the others use `Kernel | None`, and its no-preparation mode differs per
  invariant 3.
- Unsupported and unverified conditions: the shared boundaries below, plus each
  primitive record's own unsupported inputs and limitations.

## Shared unsupported and unverified boundaries

Read each label literally — unsupported, absent, and unverified are three
different statements. These hold for the seam and for both providers;
provider-specific limitations stay in the provider records.

| Shared boundary | Label | Why the label |
| --- | --- | --- |
| Controlled preparation | unverified | no packaged provider documents, wraps, or performs it, and `cudaq.control` never appears in the stateprep sources |
| Inverse/adjoint preparation | unverified | `cudaq.adjoint` never appears in the stateprep sources and no inverse-preparation symbol exists |
| Global phase under control | unverified | the Givens contract holds only up to a global phase, and no source characterizes a preparation later placed under control |
| Measurement-assisted preparation | absent | no measurement, reset, feed-forward, repeat-until-success, or success-probability construct exists in the family, so there is no library success probability to report |
| Dirty (non-zero) input register | unverified | both providers document an all-zero input register; behavior on any other input state is unspecified |
| Width-mismatch behavior | unverified | exact width is required, but no source or test establishes one uniform outcome; the HF-only path touches fixed indices and may remain in range on a differently sized register |
| Foreign consumer injection | unverified | a caller's own encoding or consumer is not covered by the consumer table above; check its source |

**`BlockEncoding` conformance does not imply injection support.** It is a
structural protocol type in `python/cudaq_algorithms/block_encoding.py`, and
**no member of it mentions `state_prep`**, so conformance promises nothing about
injection. Injection support comes from the *consumer* (`Walk`, `QSVT`), which
accepts `state_prep` generically for any conforming encoding, or from an
encoding's own non-protocol factories. Check support per consumer against source
or documentation; never infer it for an unchecked or caller-supplied consumer.

Do not hand-roll an alternative for any of these, and do not present an
out-of-contract experiment as a library capability.
