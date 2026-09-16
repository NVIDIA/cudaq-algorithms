# State preparation — family front door

Status: draft. Operation + object: **prepare** a **quantum state**.

This file is the family front door and **not a primitive record**. It carries
only two shared schemas of `../../assets/primitive-record-template.md` — a
**Representation Record** for the injectable one-register preparation kernel
and a **Capability Record** for unitary state preparation — plus routing links
to the focused factory, runtime device-kernel, and resource-estimator records.
`../conventions.md` owns the cross-cutting layout, ownership, and validation
conventions the providers depend on.

## Concrete primitive records

| Primitive record | The scientific contract it owns |
| --- | --- |
| [state-preparation-givens-schedule.md](state-preparation-givens-schedule.md) | Host planning of a Givens schedule from an orthonormal orbital-coefficient matrix |
| [state-preparation-slater-determinant-kernel.md](state-preparation-slater-determinant-kernel.md) | Injectable Slater-determinant kernel emitted from a validated Givens schedule |
| [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md) | Hartree-Fock reference occupation, optionally followed by a fixed-parameter UCC product at caller-supplied amplitudes |

## Routing front doors

| Front door | What it routes |
| --- | --- |
| [state-preparation-givens.md](state-preparation-givens.md) | Schedule planning, injectable preparation, raw Givens/Slater kernels, and logical-resource estimation |
| [state-preparation-device-kernels.md](state-preparation-device-kernels.md) | Independently callable, runtime-parameterized Hartree-Fock, excitation, UCC, Givens, and flattened Slater-determinant device kernels |
| [state-preparation-resources.md](state-preparation-resources.md) | Four independently selectable logical-resource estimators |
| [operator-pools.md](operator-pools.md) | UCCSD, UCCGSD, UpCCGSD, and CEO operator-pool providers |

The schedule planner returns host data; the Slater-determinant and HF/UCC
factories return one-register kernels. Inputs, outputs, validation, and
provenance live in their focused records. The coupled Givens pipeline is also
summarized by its [routing front door](state-preparation-givens.md).

The raw device kernels are a different representation from the injectable
one-register seam below: they take runtime scientific data alongside a live
register and are not directly accepted as `state_prep`. Route by exact public
symbol through [state-preparation-device-kernels.md](state-preparation-device-kernels.md).

## Resource estimator routing

Resource estimation is independently selectable because its inputs, results,
validation, and interpretation differ from state preparation. Route an exact
`cudaq_algorithms.stateprep` estimator symbol through
[state-preparation-resources.md](state-preparation-resources.md), then read only
the focused estimator record it selects. Do not infer one estimator's validation
or bound status from either preparation record.

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
- Source provenance: current public source/tests are authoritative and must be
  checked at use time. This record's historical source review is recorded in
  [Source provenance](../source-provenance.md).
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`). No
  compatible-range family verification was completed. The narrower task-local
  smoke run on an out-of-range CUDA-Q is recorded centrally in
  [EVAL.md](../../evals/EVAL.md).
- Lifecycle: draft, for this file and the focused records.
- **Evidence rule for this family.** Every test named in this file or a linked
  preparation/resource record is *cited repository evidence* — a committed
  assertion inspected during the historical last review, labeled `derived` —
  unless that record states a fresh execution separately. A test pass is never
  a hardware measurement.

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
- Producers (>=2 required to justify this record): the packaged factories
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
  representation routed through
  [state-preparation-device-kernels.md](state-preparation-device-kernels.md),
  not a variant of this one; kernels that allocate ancillas,
  measure, or reset (none is packaged, and the capability is unitary-only).
- Source paths, tests, docs, and provenance: "Family scope and source
  provenance" above.

---

## Capability Record — unitary state preparation

- Stable ID: `cudaq-algorithms.state-preparation.unitary.v1`. Architecture
  explicitly permits a dotted capability name. This is the adopted
  documentation identifier; capability status remains provisional.
- Status: **provisional.** Source shows multiple independent providers and
  consumers of this boundary, so it is past `candidate`, but it is not a public
  protocol and its stability across future providers is not established.
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
- Register or shape geometry and ownership: geometry is invariant 2. At the
  historical last review, ownership was **caller-owned in the reviewed source
  only**: the consumer factory that called the preparation kernel allocated the
  register, handed it over fresh in `|0...0>`, and expected exactly its own system
  width. That is a `derived` historical description, **explicitly not a universal
  or future policy.** Check current public source at use time. Whether the
  caller or the primitive should own system and ancilla registers in general is
  an **open** question in the
  taxonomy design record outside this package (`../conventions.md` repeats the
  scope limit); do not present the reviewed behavior as a library-wide guarantee,
  nor the open question as settled.
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
- Promotion criteria to a public protocol or compiler IR operation, and the
  explicit decision still required: (a) a provider outside the two historically
  reviewed providers that exercises the same boundary without widening it, (b)
  a resolved register-ownership policy, (c) characterized behavior for the conditions
  marked `unverified` below, and (d) an explicit team decision recorded with an
  owner. None of the four held at the historical last review; check current
  public source and team records at use time. This record proposes no promotion.

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

## Shared evaluation coverage

| Case id in `../../evals/evals.json` | What to load |
| --- | --- |
| `state-preparation-provider-selection` | this file (scope, Representation Record, Capability Record), [Givens schedule planning](state-preparation-givens-schedule.md), the [Slater-determinant kernel](state-preparation-slater-determinant-kernel.md), and the [HF/UCC kernel](state-preparation-hf-ucc.md) |
| `state-preparation-ucc-parameterization-boundary` | [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md) (Scientific contract; Accuracy and limitations → Parameterization boundary), plus this file's Representation Record for the one-argument seam |
| `state-preparation-reference-excitation-device-boundary` | [device-kernel routing](state-preparation-device-kernels.md), then the focused Hartree-Fock, explicit-occupation, single-excitation, and double-excitation records |
| `state-preparation-uccsd-open-shell-parity` | the focused [UCCSD device-kernel record](state-preparation-kernel-uccsd.md), [UCCSD pool record](operator-pool-uccsd.md), and the HF/UCC host contract's occupation validation |
| `state-preparation-grouped-ucc-device-boundary` | [device-kernel routing](state-preparation-device-kernels.md), then the focused UCCGSD, UpCCGSD, CEO, and fixed-parameter-UCC records |
| `state-preparation-givens-device-boundary` | [device-kernel routing](state-preparation-device-kernels.md), then the focused real/phase Givens and real/complex flattened-Slater records |
| `state-preparation-injection-composition` | this file only: Capability Record invariants, the four-module consumer table, and the `BlockEncoding` non-implication |
| `state-preparation-controlled-adjoint-unknown` | this file only: Shared unsupported and unverified boundaries |
| `state-preparation-measurement-assisted-negative` | this file only: Shared unsupported and unverified boundaries, plus Capability Record invariant 4 (unitary only) |
| `state-preparation-width-mismatch-unknown` | this file's exact-width invariant plus [the HF/UCC provider limitation](state-preparation-hf-ucc.md) |

Each leaf is reachable directly from the catalog and through at most one family
front door; no deeper directory or routing layer is required. One manual
with-skill smoke attempt passed for
`state-preparation-uccsd-open-shell-parity` and
`state-preparation-givens-device-boundary` on 2026-09-10. No baseline or formal
repeated arm has run, and all other rows remain authored coverage only;
`../../evals/EVAL.md` owns the procedure.
