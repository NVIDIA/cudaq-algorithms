# Suzuki–Trotter evolution — family front door

Status: draft.

| Operation + object | Record |
| --- | --- |
| preprocess / Pauli Hamiltonian into ordered product-formula terms | [trotter-planning.md](trotter-planning.md) |
| evolve / newly allocated all-zero or injected-preparation state | [`Trotter.kernel` zero-argument factory](trotter-kernel-factory.md) |
| evolve / caller-supplied `cudaq.State` | [`Trotter.state_kernel` state-input factory](trotter-state-kernel-factory.md) |
| apply / Suzuki–Trotter product formula to a live qubit register | [low-level apply kernel](trotter-apply-kernel.md) |
| estimate / logical resources from a validated `Trotter` plan | [planned-resource estimate](trotter-resources-planned.md) |
| estimate / logical resources from caller-supplied flattened lists | [raw resource estimate](trotter-resources-raw.md) |

The shared result type and the boundary between those two independently
selectable estimators are in the
[resource-estimation front door](trotter-resources.md).

The two stored-plan evolution factories are also compared by the
[evolution routing front door](trotter-evolution.md); they share formula
semantics but not their input representation or emitted signature.

Package-root exports are `cudaq_algorithms.Trotter`,
`cudaq_algorithms.TrotterOrdering`, and
`cudaq_algorithms.TrotterResourceEstimate`. The planning helper, low-level
device kernel, raw estimator, and numeric order constants are not re-exported
from the package root; use `cudaq_algorithms.trotter.make_trotter_terms`,
`cudaq_algorithms.trotter.apply_trotter`,
`cudaq_algorithms.trotter.estimate_trotter_resources`,
`cudaq_algorithms.trotter.FIRST_ORDER_TROTTER`,
`cudaq_algorithms.trotter.SECOND_ORDER_TROTTER`, and
`cudaq_algorithms.trotter.FOURTH_ORDER_TROTTER`.

Simulation-only statevector execution is documented separately in
[simulation-evolve.md](../simulation/simulation-evolve.md). Shared source is
`python/cudaq_algorithms/trotter.py`; authoritative behavior is in
`tests/python/test_trotter.py`. Current public source and tests are authoritative
and must be rechecked at use time.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.
