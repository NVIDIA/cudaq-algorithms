# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#                                                                              #
# [Begin Documentation]
"""From double-factorized integrals to LCU/QSVT cost, end to end.

Takes H2/STO-3G molecular integrals (hardcoded, so the example needs no
PySCF), double-factorizes the two-electron tensor, and — for each leaf
count — reconstructs the truncated tensor, bridges it to a qubit
Hamiltonian (``chemistry.qubit_hamiltonian``: spin expansion +
Jordan-Wigner), and reports what the truncation buys on the quantum side:
the PauliLCU normalization ``alpha`` (which sets QSVT polynomial degree
via ``d ~ alpha * t`` for time evolution), the Pauli term count (SELECT
cost), and the exact ground-state shift it costs.

Uses ``fermion.jordan_wigner`` via ``chemistry.qubit_hamiltonian``.
"""
from __future__ import annotations

import numpy as np

import cudaq_algorithms as algorithms
from cudaq_algorithms import PauliLCU, chemistry

df = algorithms.double_factorization

# H2 / STO-3G at R = 0.7414 A: MO-basis core Hamiltonian and chemist-
# notation (pq|rs) two-electron integrals; FCI total energy -1.137270 Ha.
ONE_BODY = np.array([[-1.25246357, 0.0], [0.0, -0.47594871]])
ERI = np.zeros((2, 2, 2, 2))
ERI[0, 0, 0, 0] = 0.67449876
ERI[1, 1, 1, 1] = 0.69716349
ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.66347258
ERI[0, 1, 0, 1] = ERI[1, 0, 1, 0] = 0.18128881
ERI[0, 1, 1, 0] = ERI[1, 0, 0, 1] = 0.18128881
E_NUCLEAR = 0.71375697


def ground_energy(spin_op) -> float:
    return float(np.min(np.linalg.eigvalsh(spin_op.to_matrix())))


def main() -> None:
    exact_h = chemistry.qubit_hamiltonian(ONE_BODY,
                                          ERI,
                                          scalar_offset=E_NUCLEAR)
    exact_energy = ground_energy(exact_h)
    exact_lcu = PauliLCU(exact_h)
    print(f"H2/STO-3G on {exact_h.qubit_count} qubits: "
          f"exact ground state {exact_energy:+.6f} Ha, "
          f"alpha = {exact_lcu.alpha:.4f}, "
          f"{exact_lcu.num_terms} LCU terms")

    full = df.explicit_double_factorization(ERI, threshold=0.0)
    max_leaves = full.num_leaves
    print(f"\nX-DF of the ERI tensor: {max_leaves} leaves at full rank")
    header = (f"{'leaves':>6} {'tensor error':>13} {'alpha':>8} "
              f"{'terms':>6} {'dE_ground (Ha)':>15}")
    print(header)
    print("-" * len(header))
    for num_leaves in range(1, max_leaves + 1):
        truncated = df.explicit_double_factorization(ERI,
                                                     max_num_leaves=num_leaves)
        tensor_error = df.factorization_error(ERI, truncated)
        h = chemistry.qubit_hamiltonian(ONE_BODY,
                                        df.reconstruct_eri(truncated),
                                        scalar_offset=E_NUCLEAR)
        lcu = PauliLCU(h)
        energy_shift = ground_energy(h) - exact_energy
        print(f"{num_leaves:>6} {tensor_error:>13.3e} {lcu.alpha:>8.4f} "
              f"{lcu.num_terms:>6} {energy_shift:>+15.3e}")

    print("\nEach row's Hamiltonian is QSVT-ready: PauliLCU(h) is a block "
          "encoding of h/alpha, so a smaller alpha means a lower-degree "
          "polynomial (fewer walk steps) for the same simulated time.")


if __name__ == "__main__":
    main()
