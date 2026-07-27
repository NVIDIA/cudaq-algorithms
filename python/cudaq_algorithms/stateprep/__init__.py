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
host with ``cudaq.spin`` algebra. ``hartree_fock`` /
``hartree_fock_occupation`` prepare the reference determinant (closed
shell, or open shell via ``make_hartree_fock_occupation``), and
``fixed_parameter_ucc`` / ``hartree_fock_ucc_kernel`` apply an arbitrary
operator pool at fixed, non-variational amplitudes on top of it.

Error-type convention: the two error cases the compiled bindings
defined keep their historical ``RuntimeError`` (odd qubit count and
odd-electrons-at-spin-0 in ``get_uccsd_excitations``); every guard
added in the pure implementation raises ``ValueError``. Note also that
the CEO helpers take ``num_orbitals`` in *spatial* orbitals (the pool
acts on ``2 * num_orbitals`` qubits), matching the compiled API.
"""

from ._hartree_fock import (
    FixedParameterUccResourceEstimate, HartreeFockResourceEstimate,
    estimate_fixed_parameter_ucc_resources,
    estimate_hartree_fock_occupation_resources,
    estimate_hartree_fock_resources, get_fixed_parameter_ucc_pauli_lists,
    hartree_fock_ucc_kernel, make_hartree_fock_occupation,
    validate_fixed_parameter_ucc, validate_hartree_fock_occupation)
from ._kernels import (ceo, double_excitation, fixed_parameter_ucc,
                       hartree_fock, hartree_fock_occupation,
                       single_excitation, uccgsd, uccsd, upccgsd)
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
    "hartree_fock",
    "hartree_fock_occupation",
    "fixed_parameter_ucc",
    "get_uccsd_excitations",
    "get_num_uccsd_parameters",
    "get_uccgsd_pauli_lists",
    "get_upccgsd_pauli_lists",
    "get_ceo_pauli_lists",
    "get_fixed_parameter_ucc_pauli_lists",
    "make_uccsd_operator_pool",
    "make_uccgsd_operator_pool",
    "make_upccgsd_operator_pool",
    "make_ceo_operator_pool",
    "make_hartree_fock_occupation",
    "validate_hartree_fock_occupation",
    "validate_fixed_parameter_ucc",
    "estimate_hartree_fock_resources",
    "estimate_hartree_fock_occupation_resources",
    "estimate_fixed_parameter_ucc_resources",
    "hartree_fock_ucc_kernel",
    "HartreeFockResourceEstimate",
    "FixedParameterUccResourceEstimate",
]
