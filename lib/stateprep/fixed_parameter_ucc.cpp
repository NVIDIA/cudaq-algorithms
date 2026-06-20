/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/fixed_parameter_ucc.h"

#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/uccgsd.h"
#include "cudaq/algorithms/stateprep/upccgsd.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cudaq::algorithms::stateprep {

namespace {

std::size_t infer_num_qubits(
    const std::vector<std::vector<cudaq::pauli_word>> &pauli_words) {
  std::size_t num_qubits = 0;
  for (const auto &group : pauli_words)
    for (const auto &word : group)
      num_qubits = std::max(num_qubits, word.str().size());
  return num_qubits;
}

void validate_coefficients_real(const cudaq::spin_op &op,
                                double coefficient_tolerance) {
  for (const auto &term : op) {
    const auto coefficient = term.evaluate_coefficient();
    if (std::abs(coefficient.imag()) > coefficient_tolerance)
      throw std::invalid_argument("fixed_parameter_ucc error - only real "
                                  "operator-pool coefficients are supported.");
  }
}

} // namespace

void validate_fixed_parameter_ucc_plan(
    const fixed_parameter_ucc_plan &plan, double coefficient_tolerance) {
  if (coefficient_tolerance < 0.0)
    throw std::invalid_argument("fixed_parameter_ucc error - coefficient "
                                "tolerance must be non-negative.");

  if (plan.parameters.size() != plan.pauli_words.size() ||
      plan.pauli_words.size() != plan.coefficients.size())
    throw std::invalid_argument("fixed_parameter_ucc error - parameters, "
                                "Pauli-word groups, and coefficient groups "
                                "must have the same length.");

  for (std::size_t i = 0; i < plan.pauli_words.size(); ++i) {
    if (plan.pauli_words[i].size() != plan.coefficients[i].size())
      throw std::invalid_argument("fixed_parameter_ucc error - each Pauli-word "
                                  "group must match its coefficient group.");
    for (const auto &word : plan.pauli_words[i])
      if (word.str().size() > plan.num_qubits)
        throw std::invalid_argument("fixed_parameter_ucc error - Pauli word "
                                    "exceeds plan.num_qubits.");
  }
}

fixed_parameter_ucc_plan make_fixed_parameter_ucc_plan(
    const std::vector<cudaq::spin_op> &operator_pool,
    const std::vector<double> &parameters, std::size_t num_qubits,
    double coefficient_tolerance) {
  if (operator_pool.size() != parameters.size())
    throw std::invalid_argument("fixed_parameter_ucc error - operator pool and "
                                "parameter vector must have the same length.");
  if (coefficient_tolerance < 0.0)
    throw std::invalid_argument("fixed_parameter_ucc error - coefficient "
                                "tolerance must be non-negative.");

  fixed_parameter_ucc_plan plan;
  plan.num_qubits = num_qubits;
  plan.parameters = parameters;
  plan.pauli_words.reserve(operator_pool.size());
  plan.coefficients.reserve(operator_pool.size());

  for (const auto &op : operator_pool) {
    validate_coefficients_real(op, coefficient_tolerance);

    std::vector<cudaq::pauli_word> words;
    std::vector<double> coefficients;
    for (const auto &term : op) {
      const auto coefficient = term.evaluate_coefficient().real();
      if (std::abs(coefficient) <= coefficient_tolerance)
        continue;
      words.push_back(term.get_pauli_word(num_qubits));
      coefficients.push_back(coefficient);
    }
    plan.pauli_words.push_back(std::move(words));
    plan.coefficients.push_back(std::move(coefficients));
  }

  validate_fixed_parameter_ucc_plan(plan, coefficient_tolerance);
  return plan;
}

fixed_parameter_ucc_plan make_fixed_parameter_ucc_plan(
    const std::vector<std::vector<cudaq::pauli_word>> &pauli_words,
    const std::vector<std::vector<double>> &coefficients,
    const std::vector<double> &parameters, std::size_t num_qubits,
    double coefficient_tolerance) {
  fixed_parameter_ucc_plan plan;
  plan.num_qubits = num_qubits == 0 ? infer_num_qubits(pauli_words) : num_qubits;
  plan.parameters = parameters;
  plan.pauli_words = pauli_words;
  plan.coefficients = coefficients;
  validate_fixed_parameter_ucc_plan(plan, coefficient_tolerance);
  return plan;
}

fixed_parameter_ucc_plan make_fixed_parameter_uccsd_plan(
    std::size_t num_qubits, std::size_t num_electrons,
    const std::vector<double> &parameters, std::size_t spin,
    double coefficient_tolerance) {
  return make_fixed_parameter_ucc_plan(
      make_uccsd_operator_pool(num_qubits, num_electrons, spin), parameters,
      num_qubits, coefficient_tolerance);
}

fixed_parameter_ucc_plan make_fixed_parameter_uccgsd_plan(
    std::size_t num_qubits, const std::vector<double> &parameters,
    bool only_singles, bool only_doubles, double coefficient_tolerance) {
  auto [words, coefficients] =
      get_uccgsd_pauli_lists(num_qubits, only_singles, only_doubles);
  return make_fixed_parameter_ucc_plan(words, coefficients, parameters,
                                       num_qubits, coefficient_tolerance);
}

fixed_parameter_ucc_plan make_fixed_parameter_upccgsd_plan(
    std::size_t num_spin_orbitals, const std::vector<double> &parameters,
    bool only_doubles, double coefficient_tolerance) {
  auto [words, coefficients] =
      get_upccgsd_pauli_lists(num_spin_orbitals, only_doubles);
  return make_fixed_parameter_ucc_plan(words, coefficients, parameters,
                                       num_spin_orbitals,
                                       coefficient_tolerance);
}

fixed_parameter_ucc_resource_estimate estimate_fixed_parameter_ucc_resources(
    const fixed_parameter_ucc_plan &plan) {
  fixed_parameter_ucc_resource_estimate estimate;
  estimate.num_qubits = plan.num_qubits;
  estimate.num_excitations = plan.pauli_words.size();
  for (const auto &group : plan.pauli_words) {
    estimate.num_pauli_rotations += group.size();
    estimate.max_pauli_rotations_per_excitation = std::max(
        estimate.max_pauli_rotations_per_excitation, group.size());
  }
  return estimate;
}

} // namespace cudaq::algorithms::stateprep
