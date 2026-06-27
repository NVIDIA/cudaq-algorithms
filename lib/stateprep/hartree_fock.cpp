/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/hartree_fock.h"

#include <algorithm>
#include <stdexcept>

namespace cudaq::algorithms::stateprep {

std::vector<std::size_t>
make_hartree_fock_occupation(std::size_t num_qubits, std::size_t num_electrons,
                             std::size_t spin) {
  if (num_electrons > num_qubits)
    throw std::invalid_argument("hartree_fock error - num_electrons cannot "
                                "exceed num_qubits.");

  std::vector<std::size_t> occupied_orbitals;
  occupied_orbitals.reserve(num_electrons);

  if (spin == 0) {
    // Closed-shell reference: the lowest num_electrons spin orbitals. Under the
    // interleaved Jordan-Wigner convention this is exactly {0, 1, ..., n-1}.
    for (std::size_t i = 0; i < num_electrons; ++i)
      occupied_orbitals.push_back(i);
    return occupied_orbitals;
  }

  // Open-shell reference: alpha electrons on even spin orbitals, beta on odd,
  // matching the convention in get_uccsd_excitations so the prepared
  // determinant lines up with a fixed-parameter UCCSD plan built at the same
  // spin. (A contiguous {0..n-1} occupation is WRONG here -- e.g. num_qubits=8,
  // num_electrons=4, spin=2 occupies {0,1,2,4}, not {0,1,2,3}.)
  if (num_qubits % 2 != 0)
    throw std::invalid_argument(
        "hartree_fock error - num_qubits must be even for spin > 0.");
  if (spin > num_electrons)
    throw std::invalid_argument(
        "hartree_fock error - spin cannot exceed num_electrons.");

  const std::size_t num_spatial_orbitals = num_qubits / 2;
  const std::size_t num_occupied_beta = (num_electrons - spin) / 2;
  const std::size_t num_occupied_alpha = num_electrons - num_occupied_beta;
  if (num_occupied_alpha > num_spatial_orbitals ||
      num_occupied_beta > num_spatial_orbitals)
    throw std::invalid_argument(
        "hartree_fock error - the requested (num_electrons, spin) does not fit "
        "in num_qubits spin orbitals.");

  for (std::size_t i = 0; i < num_occupied_alpha; ++i)
    occupied_orbitals.push_back(i * 2);
  for (std::size_t i = 0; i < num_occupied_beta; ++i)
    occupied_orbitals.push_back(i * 2 + 1);
  std::sort(occupied_orbitals.begin(), occupied_orbitals.end());
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
