# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
resources = stateprep.estimate_givens_resources(schedule)
assert resources.num_givens_rotations == len(angles)
assert resources.num_exp_pauli_calls == 2 * len(angles)
assert resources.num_phase_rotations == (len(angles) + schedule.num_electrons)
assert resources.two_qubit_gate_count_proxy == resources.num_exp_pauli_calls
assert resources.depth_proxy == resources.num_exp_pauli_calls + resources.num_phase_rotations
