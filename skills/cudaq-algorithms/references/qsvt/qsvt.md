# QSP and QSVT — family front door

Status: draft.

| Operation + object | Record |
| --- | --- |
| transform / encoded spectrum with a phase sequence | [qsvt-sequence.md](qsvt-sequence.md) |
| reconstruct / real-time-evolved state from cosine/sine good blocks | [qsvt-recovery.md](qsvt-recovery.md) |

Simulation-only execution and good-subspace extraction are separate records
under [simulation analysis](../simulation/simulation-analysis.md). Phase generation and
polynomial-degree selection are absent from the installed library.

Shared source is `python/cudaq_algorithms/qsvt.py`; shared tests are
`test_qsvt.py`, `test_walk_qsvt_orchestration.py`, and `test_df_encoding.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [source-provenance.md](../source-provenance.md) records historical
last-review audit context.
