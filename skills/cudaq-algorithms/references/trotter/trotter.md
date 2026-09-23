# Suzuki–Trotter evolution — family front door

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| preprocess / Pauli Hamiltonian into ordered product-formula terms | `make_trotter_terms`, `TrotterOrdering` | classical transformation; host | none | [Trotter planning](trotter-planning.md) |
| evolve / newly allocated all-zero or injected-preparation state by a stored Suzuki–Trotter plan | `cudaq_algorithms.Trotter.kernel` | quantum operation; kernel factory/device kernel | optionally consumes `cudaq-algorithms.state-preparation.unitary.v1` | [zero-argument Trotter factory](trotter-kernel-factory.md) |
| evolve / caller-supplied `cudaq.State` by a stored Suzuki–Trotter plan | `cudaq_algorithms.Trotter.state_kernel` | quantum operation; kernel factory/device kernel | concrete state input; no state-preparation capability | [state-input Trotter factory](trotter-state-kernel-factory.md) |
| apply / Suzuki–Trotter formula to a live qubit register | `cudaq_algorithms.trotter.apply_trotter` | quantum operation; device kernel | concrete flattened lists and caller-owned `cudaq.qview`; no capability ID | [low-level Trotter apply kernel](trotter-apply-kernel.md) |
| estimate / logical resources for a validated, pruned Trotter plan | `Trotter.resources` | resource estimator; host | consumes stored `Trotter` plan | [planned Trotter resources](trotter-resources-planned.md) |
| estimate / logical resources from caller-supplied flattened lists | `trotter.estimate_trotter_resources` | resource estimator; host | raw lists; no Hamiltonian capability inferred | [raw Trotter resources](trotter-resources-raw.md) |

The two stored-plan evolution factories consume the validated, pruned, ordered
terms stored by `Trotter` and share finite-time, positive-step, and
order-in-`{1,2,4}` validation. `Trotter.kernel` emits `() -> None` for a newly
allocated all-zero or injected-preparation state; `Trotter.state_kernel` emits
`(state: cudaq.State) -> None` for caller-supplied state. They differ in input
representation, preparation support, register construction, and emitted
signature, so do not substitute one for the other. Direct application to a
caller-owned live `cudaq.qview` is the separate `apply_trotter` contract.

The two resource estimators return the same root-exported immutable dataclass,
`cudaq_algorithms.TrotterResourceEstimate`, whose fields are `num_terms`,
`steps`, `order`, `pauli_rotations`, `estimated_cx_count`, and
`identity_coefficient`. `Trotter.resources` consumes a validated, pruned,
ordered plan; `trotter.estimate_trotter_resources` consumes caller-supplied
flattened lists. The origin and guarantees of `num_terms` and
`identity_coefficient` therefore differ; do not transfer the wrapper's
canonicalization guarantees to the raw helper. Both require and host-validate
`steps` and `order`. Their exact logical Pauli-rotation count and CNOT
decomposition proxy are not hardware depth, runtime, measured gates,
T/Toffoli count, memory, or a numerical Trotter-error estimate.

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
[Source lookup](../source-provenance.md) gives shared current-source paths.
Resource claims also require rechecking at use time; runtime execution and
hardware measurement remain unverified.

## Premise check (claim → verdict)

Test every claim the request makes or assumes against this table before answering; a matching row is
the answer to give, cited by the record path in its last cell, and the records named there are read
with the `cat` command shown.

| If the request claims or assumes | Verdict | Say (demanded words first) |
| --- | --- | --- |
| `estimate_trotter_resources` separates identity words or derives `identity_coefficient` from the lists | false | **The raw estimator takes `identity_coefficient` as a pass-through float (default `0.0`) and does not detect identity words in `words`; only the stored `Trotter` plan (`make_trotter_terms`) accumulates identity terms into `identity_coefficient` and prunes them, so `Trotter.resources` and the raw helper differ in the origin of `num_terms` and `identity_coefficient`.** Records: `cat <this skill's directory>/references/trotter/trotter-resources-raw.md`, `cat <this skill's directory>/references/trotter/trotter-resources-planned.md` |
| The identity term is applied as a device rotation, or the kernel includes its phase | false | **Identity terms are never emitted as gates; they contribute only the global phase `exp(-i * identity_coefficient * time)`, which the simulation-only helper `sim_utils.evolve(..., include_identity_phase=True)` restores for statevector output; `Trotter.kernel`, `Trotter.state_kernel`, and `apply_trotter` do not.** Records: `cat <this skill's directory>/references/simulation/simulation-evolve.md`, `cat <this skill's directory>/references/trotter/trotter-kernel-factory.md` |
| `order` and `steps` are exact settings, or fourth order is a different algorithm | false | **`order` in `{1, 2, 4}` and `steps` are the approximation controls of one product-formula family; order 4 is the Forest–Ruth composition of symmetric second-order steps; the resource estimate reports rotation and CX proxies, not the Trotter error, which must be measured against an independent dense reference with a predeclared tolerance.** Records: `cat <this skill's directory>/references/trotter/trotter-planning.md`, `cat <this skill's directory>/references/trotter/trotter-resources-planned.md` |
| The raw kernel raises on zero steps or an unsupported order | false | **`apply_trotter` is a silent no-op (register unchanged, no exception) for `steps == 0`, mismatched coefficient/word lengths, or `order` not in `{1, 2, 4}`; `Trotter.kernel`, `Trotter.state_kernel`, and `sim_utils.evolve` validate on the host and raise `ValueError`.** Record: `cat <this skill's directory>/references/trotter/trotter-apply-kernel.md` |
| `Trotter.resources` gives hardware depth, runtime, or measured gate counts | false | **`pauli_rotations` and `estimated_cx_count` are formula-level logical proxies (exact rotation count and a CNOT decomposition proxy), not transpiled depth, runtime, T count, memory, or a numerical error estimate.** Record: `cat <this skill's directory>/references/trotter/trotter-resources-planned.md` |
| Controlled or inverse Trotter evolution is a packaged contract | absent | **No controlled-Trotter factory is documented and the injected preparation is not controlled; `cudaq.control` or `cudaq.adjoint` over these kernels is a generic CUDA-Q experiment outside the tested contract and must be labeled unverified.** Record: `cat <this skill's directory>/references/trotter/trotter-kernel-factory.md` |
| `Trotter.kernel` and `Trotter.state_kernel` are interchangeable, or `sim_utils.evolve` is a third device path | false | **`Trotter.kernel` emits `() -> None` for a newly allocated all-zero or injected-preparation register; `Trotter.state_kernel` emits `(state: cudaq.State) -> None` for a caller-supplied state and takes no `state_prep`; `sim_utils.evolve` is a simulation-only statevector helper that delegates to `state_kernel` and is not a hardware path.** Records: `cat <this skill's directory>/references/trotter/trotter-kernel-factory.md`, `cat <this skill's directory>/references/trotter/trotter-state-kernel-factory.md`, `cat <this skill's directory>/references/simulation/simulation-evolve.md` |
