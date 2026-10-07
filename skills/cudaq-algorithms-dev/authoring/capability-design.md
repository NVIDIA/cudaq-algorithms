# Capability Design

## Capability composition

Use identifiers of the form
`cudaq-algorithms.<dotted-capability-name>.v<major>`. Dotted capability names
are valid. The major version changes only for an incompatible semantic change.

The currently adopted documentation identifiers are:

- `cudaq-algorithms.state-preparation.unitary.v1`;
- `cudaq-algorithms.block-encoding.zero-flagged.v1`;
- `cudaq-algorithms.chemistry-integrals.v1`.

Their status remains `provisional`; the identifier is resolved even though the
boundary has not been promoted to a stable taxonomy contract.

Every capability record states its ID, status, owner, direction, boundary
representation, exact signature, semantic invariants, geometry, conventions,
execution boundary, providers, consumers, and unsupported conditions.
Composition requires the same ID and compatible major version, plus every
consumer invariant. Similar names and structural member presence are
insufficient.
