/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/hartree_fock.h"

#include <stdexcept>

namespace cudaq::algorithms::stateprep {

std::vector<std::size_t>
make_hartree_fock_occupation(std::size_t num_qubits,
                             std::size_t num_electrons) {
  if (num_electrons > num_qubits)
    throw std::invalid_argument("hartree_fock error - num_electrons cannot "
                                "exceed num_qubits.");

  std::vector<std::size_t> occupied_orbitals;
  occupied_orbitals.reserve(num_electrons);
  for (std::size_t i = 0; i < num_electrons; ++i)
    occupied_orbitals.push_back(i);
  return occupied_orbitals;
}

void validate_hartree_fock_occupation(
    std::size_t num_qubits, const std::vector<std::size_t> &occupied_orbitals) {
  std::vector<bool> seen(num_qubits, false);
  for (const auto orbital : occupied_orbitals) {
    if (orbital >= num_qubits)
      throw std::invalid_argument("hartree_fock error - occupied orbital index "
                                  "exceeds num_qubits.");
    if (seen[orbital])
      throw std::invalid_argument("hartree_fock error - occupied orbital "
                                  "indices must be unique.");
    seen[orbital] = true;
  }
}

hartree_fock_resource_estimate
estimate_hartree_fock_resources(std::size_t num_qubits,
                                std::size_t num_electrons) {
  return estimate_hartree_fock_resources(
      num_qubits, make_hartree_fock_occupation(num_qubits, num_electrons));
}

hartree_fock_resource_estimate estimate_hartree_fock_resources(
    std::size_t num_qubits, const std::vector<std::size_t> &occupied_orbitals) {
  validate_hartree_fock_occupation(num_qubits, occupied_orbitals);
  return {num_qubits, occupied_orbitals.size(), occupied_orbitals.size()};
}

} // namespace cudaq::algorithms::stateprep
