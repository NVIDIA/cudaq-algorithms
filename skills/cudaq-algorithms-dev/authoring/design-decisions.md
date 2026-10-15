# Design decisions and open questions

## State-preparation ownership and capability promotion

- Register or shape geometry and ownership: geometry is invariant 2 in the
  [injection contract](../../cudaq-algorithms/references/state-preparation/injection-contract.md#capability-record--unitary-state-preparation).
  At the
  historical last review, ownership was **caller-owned in the reviewed source
  only**: the consumer factory that called the preparation kernel allocated the
  register, handed it over fresh in `|0...0>`, and expected exactly its own system
  width. That is a `derived` historical description, **explicitly not a universal
  or future policy.** Check current public source at use time. Whether the
  caller or the primitive should own system and ancilla registers in general is
  an **open** question in the
  taxonomy design record outside this package (the
  [kernel-boundary convention](../../cudaq-algorithms/references/conventions/kernel-boundaries.md)
  repeats the scope limit); do not present the reviewed behavior as a library-wide guarantee,
  nor the open question as settled.

- Promotion criteria to a public protocol or compiler IR operation, and the
  explicit decision still required: (a) a provider outside the two historically
  reviewed providers that exercises the same boundary without widening it, (b)
  a resolved register-ownership policy, (c) characterized behavior for the conditions
  marked `unverified` in the
  [injection boundaries](../../cudaq-algorithms/references/state-preparation/injection-contract.md#shared-unsupported-and-unverified-boundaries),
  and (d) an explicit team decision recorded with an
  owner. None of the four held at the historical last review; check current
  public source and team records at use time. This record proposes no promotion.

- **Scope limit.** This is a `derived` description of the injection seam found
  during the last source review. Recheck the cited call sites in the current
  checkout before relying on it. It is **not** a decided capability-level policy.
  Whether the caller or the primitive should own system and ancilla registers
  in general, and the exact input-state, width, ancilla, inverse, control, and
  failure semantics of state preparation, remain **open** questions in the
  taxonomy design record, which lives outside this skill package. Do not
  present the current behavior as a library-wide guarantee for future
  capabilities, and do not present the open questions as settled.

## Capability status

The adopted state-preparation, zero-flagged block-encoding, and chemistry-integral
documentation capability IDs remain provisional; they have not been promoted
to stable taxonomy contracts. The injection boundary has two packaged providers
and four independent consumer modules, but is not a public protocol, and stability
across future providers is not established.
