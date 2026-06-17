/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/hamiltonian_simulation/trotter.h"

namespace cudaq::algorithms::hamiltonian_simulation {

__qpu__ void apply_trotter(const std::vector<double> &coefficients,
                           const std::vector<cudaq::pauli_word> &words,
                           double time, std::size_t steps, int order,
                           cudaq::qview<> qubits) {
  if (steps == 0 || coefficients.size() != words.size())
    return;

  if (order != 1 && order != 2 && order != 4)
    return;

  const double dt = time / static_cast<double>(steps);
  for (std::size_t step = 0; step < steps; ++step) {
    if (order == 1) {
      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-dt * coefficients[i], qubits, words[i]);
    } else if (order == 4) {
      for (std::size_t i = 0; i < words.size(); ++i) {
        // double forest_ruth_w1 = 1.3512071919596578;
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i], qubits,
                  words[i]);
      }
      for (std::size_t i = words.size(); i > 0; --i) {
        // double forest_ruth_w1 = 1.3512071919596578;
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i - 1], qubits,
                  words[i - 1]);
      }

      for (std::size_t i = 0; i < words.size(); ++i) {
        // double forest_ruth_w0 = -1.7024143839193153;
        exp_pauli(-0.5 * -1.7024143839193153 * dt * coefficients[i], qubits,
                  words[i]);
      }
      for (std::size_t i = words.size(); i > 0; --i) {
        // double forest_ruth_w0 = -1.7024143839193153;
        exp_pauli(-0.5 * -1.7024143839193153 * dt * coefficients[i - 1],
                  qubits, words[i - 1]);
      }

      for (std::size_t i = 0; i < words.size(); ++i) {
        // double forest_ruth_w1 = 1.3512071919596578;
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i], qubits,
                  words[i]);
      }
      for (std::size_t i = words.size(); i > 0; --i) {
        // double forest_ruth_w1 = 1.3512071919596578;
        exp_pauli(-0.5 * 1.3512071919596578 * dt * coefficients[i - 1], qubits,
                  words[i - 1]);
      }
    } else {
      for (std::size_t i = 0; i < words.size(); ++i)
        exp_pauli(-0.5 * dt * coefficients[i], qubits, words[i]);
      for (std::size_t i = words.size(); i > 0; --i)
        exp_pauli(-0.5 * dt * coefficients[i - 1], qubits, words[i - 1]);
    }
  }
}

} // namespace cudaq::algorithms::hamiltonian_simulation
