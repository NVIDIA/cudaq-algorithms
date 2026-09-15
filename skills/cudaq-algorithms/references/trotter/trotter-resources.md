# Trotter resource estimation — family front door

Status: draft.

Two independently selectable interfaces return the same result type but have
different input guarantees:

| Operation + object | Public entry point | Record |
| --- | --- | --- |
| estimate / resources for a validated, pruned, ordered Trotter plan | `cudaq_algorithms.Trotter.resources` | [planned-resource estimate](trotter-resources-planned.md) |
| estimate / resources for caller-supplied flattened lists | `cudaq_algorithms.trotter.estimate_trotter_resources` | [raw resource estimate](trotter-resources-raw.md) |

Both return the root-exported immutable dataclass
`cudaq_algorithms.TrotterResourceEstimate`. Its fields are `num_terms`, `steps`,
`order`, `pauli_rotations`, `estimated_cx_count`, and
`identity_coefficient`. The origin and guarantees of `num_terms` and
`identity_coefficient` differ between the two interfaces; read the selected
record rather than transferring the wrapper's canonicalization guarantees to
the raw helper.

For both interfaces, `steps` and `order` are required and host-validated. The
exact logical Pauli-rotation count and the CNOT decomposition proxy are not
hardware depth, runtime, measured gates, T/Toffoli count, memory, or a numerical
Trotter-error estimate.

Shared source is `python/cudaq_algorithms/trotter.py`; authoritative behavior is
in `tests/python/test_trotter.py`. Current public source and tests are
authoritative and must be rechecked at use time.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context. Resource claims also require rechecking at use time; runtime
execution and hardware measurement remain unverified.
