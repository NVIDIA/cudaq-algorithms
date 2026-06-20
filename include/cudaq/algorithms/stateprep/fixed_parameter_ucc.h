/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/spin_op.h"

#include <cstddef>
#include <vector>

namespace cudaq::algorithms::stateprep {

struct fixed_parameter_ucc_plan {
  std::size_t num_qubits = 0;
  std::vector<double> parameters;
  std::vector<std::vector<cudaq::pauli_word>> pauli_words;
  std::vector<std::vector<double>> coefficients;
};

struct fixed_parameter_ucc_resource_estimate {
  std::size_t num_qubits = 0;
  std::size_t num_excitations = 0;
  std::size_t num_pauli_rotations = 0;
  std::size_t max_pauli_rotations_per_excitation = 0;
};

void validate_fixed_parameter_ucc_plan(
    const fixed_parameter_ucc_plan &plan,
    double coefficient_tolerance = 1.0e-12);

fixed_parameter_ucc_plan make_fixed_parameter_ucc_plan(
    const std::vector<cudaq::spin_op> &operator_pool,
    const std::vector<double> &parameters, std::size_t num_qubits,
    double coefficient_tolerance = 1.0e-12);

fixed_parameter_ucc_plan make_fixed_parameter_ucc_plan(
    const std::vector<std::vector<cudaq::pauli_word>> &pauli_words,
    const std::vector<std::vector<double>> &coefficients,
    const std::vector<double> &parameters, std::size_t num_qubits,
    double coefficient_tolerance = 1.0e-12);

fixed_parameter_ucc_plan make_fixed_parameter_uccsd_plan(
    std::size_t num_qubits, std::size_t num_electrons,
    const std::vector<double> &parameters, std::size_t spin = 0,
    double coefficient_tolerance = 1.0e-12);

fixed_parameter_ucc_plan make_fixed_parameter_uccgsd_plan(
    std::size_t num_qubits, const std::vector<double> &parameters,
    bool only_singles = false, bool only_doubles = false,
    double coefficient_tolerance = 1.0e-12);

fixed_parameter_ucc_plan make_fixed_parameter_upccgsd_plan(
    std::size_t num_spin_orbitals, const std::vector<double> &parameters,
    bool only_doubles = false, double coefficient_tolerance = 1.0e-12);

fixed_parameter_ucc_resource_estimate estimate_fixed_parameter_ucc_resources(
    const fixed_parameter_ucc_plan &plan);

/// \pure_device_kernel
///
/// @brief Apply a fixed-parameter UCC-style product over grouped Pauli terms.
/// @details This primitive intentionally has no optimizer or variational loop.
/// It applies one supplied parameter per excitation group, where each group is
/// represented by Pauli words and real coefficients generated on the host.
__qpu__ void fixed_parameter_ucc(
    cudaq::qview<> qubits, const std::vector<double> &parameters,
    const std::vector<std::vector<cudaq::pauli_word>> &pauli_words,
    const std::vector<std::vector<double>> &coefficients);

} // namespace cudaq::algorithms::stateprep
