# Givens determinant preparation — routing front door

Status: draft. This file routes two independently selectable contracts; it is
not a primitive record.

| Operation + object | Public entry point | Focused record |
| --- | --- | --- |
| preprocess / orthonormal orbital-coefficient matrix into a Givens schedule | `stateprep.make_givens_rotation_schedule` | [schedule planning](state-preparation-givens-schedule.md) |
| prepare / Slater-determinant state from a Givens schedule | `stateprep.slater_determinant_kernel` | [injectable kernel](state-preparation-slater-determinant-kernel.md) |
| apply / adjacent real Givens rotation | `stateprep.givens_rotation` | [raw device kernel](state-preparation-kernel-givens-rotation.md) |
| apply / adjacent phase-aware Givens rotation | `stateprep.phase_givens_rotation` | [raw device kernel](state-preparation-kernel-phase-givens-rotation.md) |
| prepare / real Slater determinant from a flattened schedule | `stateprep.slater_determinant` | [raw device kernel](state-preparation-kernel-slater-determinant.md) |
| prepare / complex Slater determinant from a flattened schedule | `stateprep.complex_slater_determinant` | [raw device kernel](state-preparation-kernel-complex-slater-determinant.md) |

The schedule is host data, not a kernel. It is also consumed independently by
[`estimate_givens_resources`](state-preparation-resources-givens.md). The
one-register injection capability is owned by
[state-preparation.md](state-preparation.md). The four raw forms accept live
register and schedule data at runtime, do not perform the factory's host
validation, and are also indexed by the
[device-kernel family front door](state-preparation-device-kernels.md).

Shared source is `python/cudaq_algorithms/stateprep/_givens.py`; authoritative
behavior is in `tests/python/test_stateprep_givens.py`, with injection coverage
in `tests/python/test_state_prep_injection.py`. Current public source/tests are
authoritative and must be checked at use time. Claims are derived from source
and tests inspected during the historical last review recorded in
[Source provenance](../source-provenance.md) and remain unexecuted unless a
current task records stronger evidence.
