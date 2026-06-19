/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/givens.h"

namespace cudaq_algorithms::stateprep {

__qpu__ void apply_givens_rotation(cudaq::qview<> qubits, double theta,
                                   std::size_t first_orbital,
                                   std::size_t second_orbital) {
  if (first_orbital + 1 == second_orbital) {
    // CUDA-Q's built-in Givens convention maps |10> to
    // cos(theta)|10> - sin(theta)|01>. The state-preparation convention here
    // uses the opposite sign, so this inlines cudaq::givens_rotation(-theta).
    exp_pauli(0.5 * theta, "YX", qubits[first_orbital],
              qubits[second_orbital]);
    exp_pauli(-0.5 * theta, "XY", qubits[first_orbital],
              qubits[second_orbital]);
  } else if (second_orbital + 1 == first_orbital) {
    exp_pauli(-0.5 * theta, "YX", qubits[second_orbital],
              qubits[first_orbital]);
    exp_pauli(0.5 * theta, "XY", qubits[second_orbital],
              qubits[first_orbital]);
  }
}

__qpu__ void apply_phase_givens_rotation(cudaq::qview<> qubits, double theta,
                                         double phase,
                                         std::size_t first_orbital,
                                         std::size_t second_orbital) {
  apply_givens_rotation(qubits, theta, first_orbital, second_orbital);
  // rz(phase) is equivalent to exp(i * phase * n) up to global phase.
  rz(phase, qubits[second_orbital]);
}

__qpu__ void prepare_slater_determinant(
    cudaq::qview<> qubits, const std::vector<std::size_t> &orbital_indices,
    const std::vector<double> &angles, std::size_t num_electrons) {
  if (orbital_indices.size() != 2 * angles.size())
    return;

  for (std::size_t i = 0; i < num_electrons; ++i)
    x(qubits[i]);

  for (std::size_t i = 0; i < angles.size(); ++i)
    apply_givens_rotation(qubits, angles[i], orbital_indices[2 * i],
                          orbital_indices[2 * i + 1]);
}

__qpu__ void prepare_complex_slater_determinant(
    cudaq::qview<> qubits, const std::vector<std::size_t> &orbital_indices,
    const std::vector<double> &angles, const std::vector<double> &phases,
    const std::vector<double> &final_phases, std::size_t num_electrons) {
  if (orbital_indices.size() != 2 * angles.size() ||
      phases.size() != angles.size() || final_phases.size() < num_electrons)
    return;

  for (std::size_t i = 0; i < num_electrons; ++i)
    x(qubits[i]);

  for (std::size_t i = 0; i < num_electrons; ++i)
    rz(final_phases[i], qubits[i]);

  for (std::size_t i = 0; i < angles.size(); ++i)
    apply_phase_givens_rotation(qubits, angles[i], phases[i],
                                orbital_indices[2 * i],
                                orbital_indices[2 * i + 1]);
}

} // namespace cudaq_algorithms::stateprep
