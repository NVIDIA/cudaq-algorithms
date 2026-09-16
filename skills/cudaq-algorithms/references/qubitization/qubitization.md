# Qubitization — family front door

Status: draft.

Qubitization exposes two independently selectable contracts:

| Operation + object | Record |
| --- | --- |
| evolve / state by a qubitization walk circuit | [qubitization-walk.md](qubitization-walk.md) |
| measure / Chebyshev spectral moment | [qubitization-moments.md](qubitization-moments.md) |

Both consume the provisional
`cudaq-algorithms.block-encoding.zero-flagged.v1` capability. They are not
interchangeable: walk methods emit kernels, while moment methods call
`cudaq.observe` and return classical numbers.

Shared source is `python/cudaq_algorithms/qubitization.py`; shared tests are
`test_qubitization.py` and `test_walk_qsvt_orchestration.py`. Current public
source and tests are authoritative and must be rechecked at use time.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.
