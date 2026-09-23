# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def make_hartree_fock_occupation(num_qubits, num_electrons, spin=0):
    num_qubits = _as_count(num_qubits, "num_qubits")
    num_electrons = _as_count(num_electrons, "num_electrons")
    spin_number = _as_count(spin, "spin")
    if num_electrons > num_qubits:
        raise ValueError("num_electrons cannot exceed num_qubits")
    if spin_number == 0:
        return list(range(num_electrons))
    if num_qubits % 2 != 0:
        raise ValueError("num_qubits must be even when spin > 0")
    if spin_number > num_electrons:
        raise ValueError("spin cannot exceed num_electrons")
    if (num_electrons - spin_number) % 2 != 0:
        raise ValueError(
            "(num_electrons - spin) must be even when spin > 0 (spin is 2*S_z)"
        )
    num_spatial = num_qubits // 2
    n_occ_beta = (num_electrons - spin_number) // 2
    n_occ_alpha = num_electrons - n_occ_beta
    if n_occ_alpha > num_spatial:
        raise ValueError("requested spin/electron occupation does not fit")
    occupied = [2 * orbital for orbital in range(n_occ_alpha)]
    occupied += [2 * orbital + 1 for orbital in range(n_occ_beta)]
    return sorted(occupied)
