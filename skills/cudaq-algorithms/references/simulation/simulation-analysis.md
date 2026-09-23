# Simulation analysis — family front door

These helpers are packaged but statevector-oriented; none is a
QPU substitute for the hardware-shaped kernel or observable contracts.

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| extract / zero-ancilla amplitude block | `sim_utils.good_subspace` | simulation analysis; host array slicing | concrete geometry | [good subspace](simulation-good-subspace.md) |
| analyze / `(H/alpha)|ket>` | `sim_utils.action` | simulation analysis; `cudaq.get_state` | concrete `PauliLCU.encode_kernel` | [action](simulation-action.md) |
| analyze / QSVT good-subspace vector | `sim_utils.transform` | simulation analysis; `cudaq.get_state` | concrete QSVT | [transform](simulation-transform.md) |
| evolve / Trotter statevector | `sim_utils.evolve` | simulation analysis; `cudaq.get_state` | concrete Trotter | [evolve](simulation-evolve.md) |

Shared source is `python/cudaq_algorithms/sim_utils.py`. The module depends on
`cudaq.get_state` except for the pure slicing helper. Current public source and
tests are authoritative and must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
