# QSP and QSVT — family front door

Quantum signal processing (QSP) and quantum singular value transformation
(QSVT): route supplied phase angles, QSPPACK conventions, and polynomial
transformations of encoded operators to the sequence record; route cosine/sine
good-block reconstruction of real-time evolution to recovery.

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| transform / encoded spectrum with a phase sequence | `PhaseSequence`, `QSVT.kernel`, `QSVT.controlled_kernel` | quantum driver; host validation + kernel factory/device kernel | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; optional state preparation | [QSVT sequence](qsvt-sequence.md) |
| reconstruct / real-time state from cosine and sine QSVT blocks | `recover_real_time_evolution` | classical transformation; host | concrete QSVT/good-block inputs | [QSVT recovery](qsvt-recovery.md) |

Simulation-only execution and good-subspace extraction are separate records
under [simulation analysis](../simulation/simulation-analysis.md). Phase generation and
polynomial-degree selection are absent from the installed library.

Shared source is `python/cudaq_algorithms/qsvt.py`; shared tests are
`test_qsvt.py`, `test_walk_qsvt_orchestration.py`, and `test_df_encoding.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [Source lookup](../source-provenance.md) gives shared current-source paths.
