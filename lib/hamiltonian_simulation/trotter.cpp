/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/hamiltonian_simulation/trotter.h"

#include <cmath>
#include <stdexcept>

namespace cudaq::algorithms::hamiltonian_simulation {

trotter_terms make_trotter_terms(const cudaq::spin_op &hamiltonian,
                                 double coefficient_tolerance) {
  if (coefficient_tolerance < 0.0)
    throw std::invalid_argument(
        "trotter error - coefficient tolerance must be non-negative.");

  trotter_terms terms;
  terms.num_qubits = hamiltonian.num_qubits();

  for (const auto &term : hamiltonian) {
    const auto coefficient = term.evaluate_coefficient();
    if (std::abs(coefficient.imag()) > coefficient_tolerance)
      throw std::invalid_argument(
          "trotter error - only real Hamiltonian coefficients are supported.");

    const auto real_coefficient = coefficient.real();
    if (term.is_identity()) {
      // Preserve identity contributions for callers that need the omitted
      // exp(-i c t) phase in controlled or interference-based algorithms.
      terms.identity_coefficient += real_coefficient;
      continue;
    }

    terms.coefficients.push_back(real_coefficient);
    terms.words.push_back(term.get_pauli_word(terms.num_qubits));
  }

  return terms;
}

} // namespace cudaq::algorithms::hamiltonian_simulation
