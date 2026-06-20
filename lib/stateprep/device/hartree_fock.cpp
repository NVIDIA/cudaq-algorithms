/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/hartree_fock.h"

namespace cudaq::algorithms::stateprep {

__qpu__ void hartree_fock(cudaq::qview<> qubits,
                          std::size_t num_electrons) {
  for (std::size_t i = 0; i < num_electrons; ++i)
    x(qubits[i]);
}

__qpu__ void hartree_fock_occupation(
    cudaq::qview<> qubits, const std::vector<std::size_t> &occupied_orbitals) {
  for (std::size_t i = 0; i < occupied_orbitals.size(); ++i)
    x(qubits[occupied_orbitals[i]]);
}

} // namespace cudaq::algorithms::stateprep
