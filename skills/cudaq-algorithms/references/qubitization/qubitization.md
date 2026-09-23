# Qubitization — family front door

Qubitization exposes two independently selectable contracts:

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| evolve / state by a qubitization walk | `Walk.kernel`, adjoint/controlled variants | quantum operation; kernel factory/device kernel | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; optional state preparation | [walk kernels](qubitization-walk.md) |
| measure / Chebyshev moment | `Walk.moment`, `Walk.moments` | measurement/readout; kernel + observable + `cudaq.observe` | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; odd orders also require `select_observable` | [moments](qubitization-moments.md) |

Both consume the
`cudaq-algorithms.block-encoding.zero-flagged.v1` capability. They are not
interchangeable: walk methods emit kernels, while moment methods call
`cudaq.observe` and return classical numbers.

Shared source is `python/cudaq_algorithms/qubitization.py`; shared tests are
`test_qubitization.py` and `test_walk_qsvt_orchestration.py`. Current public
source and tests are authoritative and must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
