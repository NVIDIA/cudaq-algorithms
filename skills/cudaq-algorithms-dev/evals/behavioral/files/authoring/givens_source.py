# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
@dataclass(frozen=True)
class GivensResourceEstimate:
    num_spin_orbitals: int
    num_electrons: int
    num_givens_rotations: int
    num_exp_pauli_calls: int
    num_phase_rotations: int
    two_qubit_gate_count_proxy: int
    depth_proxy: int


def estimate_givens_resources(
        schedule: GivensRotationSchedule) -> GivensResourceEstimate:
    """Resource estimate for preparing a schedule's Slater determinant."""
    validate_givens_rotation_schedule(schedule)
    num_rotations = len(schedule.rotations)
    num_exp_pauli_calls = 2 * num_rotations
    num_phase_rotations = num_rotations + schedule.num_electrons if schedule.is_complex else 0
    return GivensResourceEstimate(schedule.num_spin_orbitals,
                                  schedule.num_electrons, num_rotations,
                                  num_exp_pauli_calls, num_phase_rotations,
                                  num_exp_pauli_calls,
                                  num_exp_pauli_calls + num_phase_rotations)
