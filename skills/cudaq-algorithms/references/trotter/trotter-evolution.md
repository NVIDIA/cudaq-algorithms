# Planned Suzuki–Trotter evolution — routing front door

Status: draft. This file routes two independently selectable stored-plan kernel
factories; it is not a primitive record.

| Operation + input object | Public entry point | Emitted signature | Focused record |
| --- | --- | --- | --- |
| evolve / newly allocated all-zero or injected-preparation state | `Trotter.kernel` | `() -> None` | [zero-argument factory](trotter-kernel-factory.md) |
| evolve / caller-supplied `cudaq.State` | `Trotter.state_kernel` | `(state: cudaq.State) -> None` | [state-input factory](trotter-state-kernel-factory.md) |

Both consume the validated, pruned, ordered terms stored by `Trotter` and share
finite-time, positive-step, and order-in-`{1,2,4}` validation. They differ in
input representation, preparation support, register construction, and emitted
signature, so do not substitute one for the other.

Direct application to a caller-owned live `cudaq.qview` is the separate
[`apply_trotter` contract](trotter-apply-kernel.md). Formula-level resource
estimation is routed through [trotter-resources.md](trotter-resources.md), and
statevector-returning execution through [simulation-evolve.md](../simulation/simulation-evolve.md).

Shared source is `python/cudaq_algorithms/trotter.py`; authoritative behavior is
in `tests/python/test_trotter.py`. Current public source and tests are
authoritative and must be rechecked at use time.
[source-provenance.md](../source-provenance.md) records
historical last-review audit context.
