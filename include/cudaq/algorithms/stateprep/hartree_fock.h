/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"

#include <cstddef>
#include <vector>

namespace cudaq::algorithms::stateprep {

struct hartree_fock_resource_estimate {
  std::size_t num_qubits = 0;
  std::size_t num_electrons = 0;
  std::size_t num_x_gates = 0;
};

std::vector<std::size_t>
make_hartree_fock_occupation(std::size_t num_qubits,
                             std::size_t num_electrons);

void validate_hartree_fock_occupation(
    std::size_t num_qubits, const std::vector<std::size_t> &occupied_orbitals);

hartree_fock_resource_estimate
estimate_hartree_fock_resources(std::size_t num_qubits,
                                std::size_t num_electrons);

hartree_fock_resource_estimate estimate_hartree_fock_resources(
    std::size_t num_qubits, const std::vector<std::size_t> &occupied_orbitals);

/// \pure_device_kernel
///
/// @brief Prepare the canonical Hartree-Fock occupation by filling the first
/// num_electrons spin orbitals.
__qpu__ void hartree_fock(cudaq::qview<> qubits, std::size_t num_electrons);

/// \pure_device_kernel
///
/// @brief Prepare a Hartree-Fock determinant from explicit occupied spin-orbital
/// indices.
__qpu__ void hartree_fock_occupation(
    cudaq::qview<> qubits, const std::vector<std::size_t> &occupied_orbitals);

} // namespace cudaq::algorithms::stateprep
