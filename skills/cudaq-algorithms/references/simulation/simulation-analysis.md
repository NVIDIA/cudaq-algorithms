# Simulation analysis — family front door

Status: draft. These helpers are packaged but statevector-oriented; none is a
QPU substitute for the hardware-shaped kernel or observable contracts.

| Operation + object | Record |
| --- | --- |
| extract / zero-ancilla good block from a simulated statevector | [simulation-good-subspace.md](simulation-good-subspace.md) |
| analyze / `(H/alpha)|ket>` through a Pauli LCU simulation | [simulation-action.md](simulation-action.md) |
| analyze / QSVT-transformed good-subspace statevector | [simulation-transform.md](simulation-transform.md) |
| evolve / statevector with a Trotter simulation helper | [simulation-evolve.md](simulation-evolve.md) |

Shared source is `python/cudaq_algorithms/sim_utils.py`. The module depends on
`cudaq.get_state` except for the pure slicing helper. Current public source and
tests are authoritative and must be rechecked at use time.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.
