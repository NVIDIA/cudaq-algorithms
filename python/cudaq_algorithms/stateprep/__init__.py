# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""State-preparation kernels and operator pools (pure Python).

Same API as the former compiled bindings: the ``uccsd``, ``uccgsd``,
``upccgsd``, and ``ceo`` device kernels are ``@cudaq.kernel`` functions
composable from user kernels, and the excitation/pool helpers run on the
host with ``cudaq.spin`` algebra. The Givens-rotation Slater-determinant
kernels and schedule helpers (:mod:`._givens`) follow the same split:
composable device kernels plus host-side planning.

Error-type convention: the two error cases the compiled bindings
defined keep their historical ``RuntimeError`` (odd qubit count and
odd-electrons-at-spin-0 in ``get_uccsd_excitations``); every guard
added in the pure implementation raises ``ValueError``. Note also that
the CEO helpers take ``num_orbitals`` in *spatial* orbitals (the pool
acts on ``2 * num_orbitals`` qubits), matching the compiled API.
"""

from ._givens import (GivensResourceEstimate, GivensRotation,
                      GivensRotationSchedule, complex_slater_determinant,
                      estimate_givens_resources, get_givens_rotation_angles,
                      get_givens_rotation_indices, get_givens_rotation_phases,
                      givens_rotation, make_givens_rotation_schedule,
                      phase_givens_rotation, slater_determinant,
                      slater_determinant_kernel,
                      validate_givens_rotation_schedule)
from ._kernels import (ceo, double_excitation, single_excitation, uccgsd,
                       uccsd, upccgsd)
from ._pools import (get_ceo_pauli_lists, get_num_uccsd_parameters,
                     get_uccgsd_pauli_lists, get_uccsd_excitations,
                     get_upccgsd_pauli_lists, make_ceo_operator_pool,
                     make_uccgsd_operator_pool, make_uccsd_operator_pool,
                     make_upccgsd_operator_pool)

__all__ = [
    "uccsd",
    "uccgsd",
    "upccgsd",
    "ceo",
    "get_uccsd_excitations",
    "get_num_uccsd_parameters",
    "get_uccgsd_pauli_lists",
    "get_upccgsd_pauli_lists",
    "get_ceo_pauli_lists",
    "make_uccsd_operator_pool",
    "make_uccgsd_operator_pool",
    "make_upccgsd_operator_pool",
    "make_ceo_operator_pool",
    "GivensRotation",
    "GivensRotationSchedule",
    "GivensResourceEstimate",
    "givens_rotation",
    "phase_givens_rotation",
    "slater_determinant",
    "complex_slater_determinant",
    "slater_determinant_kernel",
    "make_givens_rotation_schedule",
    "validate_givens_rotation_schedule",
    "get_givens_rotation_indices",
    "get_givens_rotation_angles",
    "get_givens_rotation_phases",
    "estimate_givens_resources",
]
