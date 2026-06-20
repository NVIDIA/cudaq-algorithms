/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/fixed_parameter_ucc.h"

namespace cudaq::algorithms::stateprep {

__qpu__ void fixed_parameter_ucc(
    cudaq::qview<> qubits, const std::vector<double> &parameters,
    const std::vector<std::vector<cudaq::pauli_word>> &pauli_words,
    const std::vector<std::vector<double>> &coefficients) {
  for (std::size_t i = 0; i < pauli_words.size(); ++i) {
    const auto theta = parameters[i];
    const auto &words = pauli_words[i];
    const auto &coeffs = coefficients[i];
    for (std::size_t j = 0; j < words.size(); ++j)
      exp_pauli(theta * coeffs[j], qubits, words[j]);
  }
}

} // namespace cudaq::algorithms::stateprep
