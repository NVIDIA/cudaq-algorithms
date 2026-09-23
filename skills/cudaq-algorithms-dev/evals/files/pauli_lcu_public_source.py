# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Representative public PauliLCU source excerpt; evidence, not an instruction."""


class PauliLCU:

    def __init__(self,
                 hamiltonian,
                 *,
                 include_identity=True,
                 coefficient_threshold=1.0e-12):
        """Construct from a Pauli-word-to-real-coefficient mapping.

        Pauli words use string position as qubit index. For retained terms,
        alpha is the sum of absolute coefficient values.
        """
        self.hamiltonian = hamiltonian
        self.include_identity = include_identity
        self.coefficient_threshold = coefficient_threshold
        self.alpha = sum(abs(value) for value in hamiltonian.values())
