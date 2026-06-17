/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/block_encoding/pauli_lcu.h"
#include "cudaq/algorithms/block_encoding/kernels.h"
#include "cudaq/algorithms/detail/qpu_dispatch.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::algorithms {

// ============================================================================
// QUANTUM KERNELS (Stateless functors for CUDA-Q compiler)
// ============================================================================

__qpu__ void controlled_pauli_x(cudaq::qubit &control, cudaq::qview<> anc,
                                cudaq::qubit &target) {
  int n_anc = anc.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(x, control, anc, target,
                                                  n_anc);
}

__qpu__ void controlled_pauli_y(cudaq::qubit &control, cudaq::qview<> anc,
                                cudaq::qubit &target) {
  int n_anc = anc.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(y, control, anc, target,
                                                  n_anc);
}

__qpu__ void controlled_pauli_z(cudaq::qubit &control, cudaq::qview<> anc,
                                cudaq::qubit &target) {
  int n_anc = anc.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(z, control, anc, target,
                                                  n_anc);
}

__qpu__ void controlled_sign_correction(cudaq::qubit &control,
                                        cudaq::qview<> anc) {
  int n_anc = anc.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_Z_BY_ARITY(control, anc, n_anc);
}

/// @brief Kernel for externally controlled SELECT operation.
struct controlled_pauli_select_kernel {
  void operator()(cudaq::qubit &control, cudaq::qview<> anc, cudaq::qview<> sys,
                  const std::vector<int> &ctrls, const std::vector<int> &ops,
                  const std::vector<int> &lens,
                  const std::vector<int> &signs) const __qpu__ {
    int ptr_ctrl = 0;
    int ptr_op = 0;
    int n_anc = anc.size();

    for (std::size_t i = 0; i < lens.size(); ++i) {
      int n_ops = lens[i];
      int sign = signs[i];

      for (int b = 0; b < n_anc; ++b) {
        int bit_val = ctrls[ptr_ctrl++];
        if (bit_val == 0)
          x(anc[b]);
      }

      for (int k = 0; k < n_ops; ++k) {
        int code = ops[ptr_op++];
        int q_idx = ops[ptr_op++];

        if (code == 1)
          controlled_pauli_x(control, anc, sys[q_idx]);
        else if (code == 2)
          controlled_pauli_y(control, anc, sys[q_idx]);
        else if (code == 3)
          controlled_pauli_z(control, anc, sys[q_idx]);
      }

      if (sign < 0)
        controlled_sign_correction(control, anc);

      int back_ptr = ptr_ctrl - 1;
      for (int b_rev = 0; b_rev < n_anc; ++b_rev) {
        int anc_idx = (n_anc - 1) - b_rev;
        int bit_val = ctrls[back_ptr--];
        if (bit_val == 0)
          x(anc[anc_idx]);
      }
    }
  }
};

// ============================================================================
// CLASSICAL UTILITIES
// ============================================================================

namespace {

std::size_t ceil_log2(std::size_t value) {
  if (value <= 1)
    return 0;

  std::size_t power = 1;
  std::size_t log = 0;
  while (power < value) {
    power <<= 1;
    ++log;
  }
  return log;
}

bool is_identity_word(const cudaq::pauli_word &word) {
  const auto word_str = word.str();
  for (char pauli : word_str)
    if (pauli != 'I')
      return false;
  return true;
}

std::vector<double> compute_prepare_angles(const std::vector<double> &probs) {
  if (probs.empty())
    return {};

  std::size_t n_leaves = probs.size();
  if ((n_leaves & (n_leaves - 1)) != 0)
    throw std::runtime_error(
        "compute_prepare_angles: probability vector size must be a power of 2");

  int n_qubits = std::log2(n_leaves);
  std::vector<double> angles;

  // Build tree layer by layer
  for (int layer = 0; layer < n_qubits; ++layer) {
    int n_nodes = 1 << layer; // Number of nodes at this layer
    for (int node = 0; node < n_nodes; ++node) {
      int step = n_leaves / (1 << (layer + 1));
      int start_idx = node * step * 2;

      // Sum probabilities for current subtree
      double total_p = 0.0;
      for (int k = 0; k < 2 * step; ++k)
        total_p += probs[start_idx + k];

      if (total_p < 1e-12) {
        angles.push_back(0.0);
      } else {
        // Sum of right branch
        double p_1 = 0.0;
        for (int k = step; k < 2 * step; ++k)
          p_1 += probs[start_idx + k];

        // Compute rotation angle: theta = 2*arcsin(sqrt(p_right / p_total))
        angles.push_back(2.0 * std::asin(std::sqrt(p_1 / total_p)));
      }
    }
  }

  return angles;
}

} // namespace

lcu_decomposition decompose_lcu(const cudaq::spin_op &hamiltonian,
                                std::size_t num_qubits,
                                double coefficient_threshold) {
  if (coefficient_threshold < 0.0)
    throw std::runtime_error(
        "decompose_lcu: coefficient threshold must be non-negative");

  lcu_decomposition lcu;
  lcu.num_system_qubits = num_qubits;
  lcu.coefficient_threshold = coefficient_threshold;

  for (const auto &term : hamiltonian) {
    auto coeff = term.evaluate_coefficient();
    double abs_coeff = std::abs(coeff);

    if (abs_coeff < coefficient_threshold)
      continue;

    if (std::abs(coeff.imag()) > 1e-10)
      throw std::runtime_error(
          "decompose_lcu: complex Hamiltonians not yet supported");

    auto word = term.get_pauli_word(num_qubits);
    double real_coeff = coeff.real();
    bool identity_term = is_identity_word(word);

    lcu.coefficients.push_back(real_coeff);
    lcu.absolute_coefficients.push_back(abs_coeff);
    lcu.signs.push_back((real_coeff < 0.0) ? -1 : 1);
    lcu.identity_terms.push_back(identity_term ? 1 : 0);
    lcu.pauli_words.push_back(word);
    lcu.normalization += abs_coeff;
    if (identity_term)
      lcu.constant_term += real_coeff;
  }

  lcu.num_terms = lcu.coefficients.size();
  if (lcu.num_terms == 0)
    throw std::runtime_error(
        "decompose_lcu: Hamiltonian has no retained terms");

  lcu.num_ancilla_qubits = ceil_log2(lcu.num_terms);
  if (lcu.num_ancilla_qubits >
      cudaq::algorithms::detail::max_lcu_ancilla_qubits)
    throw std::runtime_error(
        "decompose_lcu: PauliLCU currently supports at most " +
        std::to_string(cudaq::algorithms::detail::max_lcu_ancilla_qubits) +
        " ancilla qubits (" +
        std::to_string(1ULL
                       << cudaq::algorithms::detail::max_lcu_ancilla_qubits) +
        " LCU terms)");

  lcu.padded_num_terms = 1ULL << lcu.num_ancilla_qubits;

  for (double abs_coeff : lcu.absolute_coefficients)
    lcu.probabilities.push_back(abs_coeff / lcu.normalization);

  return lcu;
}

pauli_lcu_kernel_data make_pauli_lcu_kernel_data(const lcu_decomposition &lcu) {
  if (lcu.num_terms == 0)
    throw std::runtime_error(
        "make_pauli_lcu_kernel_data: LCU decomposition has no terms");
  if (lcu.coefficients.size() != lcu.num_terms ||
      lcu.pauli_words.size() != lcu.num_terms ||
      lcu.probabilities.size() != lcu.num_terms ||
      lcu.signs.size() != lcu.num_terms)
    throw std::runtime_error(
        "make_pauli_lcu_kernel_data: inconsistent LCU decomposition metadata");

  if (lcu.num_ancilla_qubits >
      cudaq::algorithms::detail::max_lcu_ancilla_qubits)
    throw std::runtime_error(
        "make_pauli_lcu_kernel_data: PauliLCU kernel data currently supports "
        "at most " +
        std::to_string(cudaq::algorithms::detail::max_lcu_ancilla_qubits) +
        " ancilla qubits");

  pauli_lcu_kernel_data data;
  data.num_system_qubits = lcu.num_system_qubits;
  data.num_terms = lcu.num_terms;
  data.padded_num_terms = lcu.padded_num_terms;
  data.num_ancilla_qubits = lcu.num_ancilla_qubits;

  auto padded_probabilities = lcu.probabilities;
  while (padded_probabilities.size() < lcu.padded_num_terms)
    padded_probabilities.push_back(0.0);

  data.state_prep_angles = compute_prepare_angles(padded_probabilities);

  for (std::size_t idx = 0; idx < lcu.num_terms; ++idx) {
    // Binary representation of index for control pattern
    for (std::size_t b = 0; b < data.num_ancilla_qubits; ++b) {
      // MSB to LSB ordering
      int bit_val = (idx >> (data.num_ancilla_qubits - 1 - b)) & 1;
      data.term_controls.push_back(bit_val);
    }

    // Parse Pauli word string: format is "XYZII..." where position i is qubit i
    std::string word_str = lcu.pauli_words[idx].str();
    std::vector<std::pair<int, int>> ops_for_term; // (code, qubit_idx)

    // Each character position corresponds to a qubit
    for (std::size_t q_idx = 0; q_idx < word_str.size(); ++q_idx) {
      char pauli_char = word_str[q_idx];

      if (pauli_char == 'I') {
        // Identity - skip
        continue;
      } else if (pauli_char == 'X') {
        ops_for_term.push_back({1, static_cast<int>(q_idx)});
      } else if (pauli_char == 'Y') {
        ops_for_term.push_back({2, static_cast<int>(q_idx)});
      } else if (pauli_char == 'Z') {
        ops_for_term.push_back({3, static_cast<int>(q_idx)});
      }
      // Other characters are ignored
    }

    // Flatten ops
    for (const auto &[code, q_idx] : ops_for_term) {
      data.term_ops.push_back(code);
      data.term_ops.push_back(q_idx);
    }

    data.term_lengths.push_back(static_cast<int>(ops_for_term.size()));
    data.term_signs.push_back(lcu.signs[idx]);
  }

  return data;
}

std::vector<double>
pauli_lcu::compute_angles(const std::vector<double> &probs) {
  return compute_prepare_angles(probs);
}

// ============================================================================
// PAULI LCU IMPLEMENTATION
// ============================================================================

pauli_lcu::pauli_lcu(const cudaq::spin_op &hamiltonian, std::size_t num_qubits)
    : pauli_lcu(decompose_lcu(hamiltonian, num_qubits)) {}

pauli_lcu::pauli_lcu(const lcu_decomposition &lcu)
    : kernel_data(make_pauli_lcu_kernel_data(lcu)), decomposition(lcu) {
  n_sys = kernel_data.num_system_qubits;
  n_anc = kernel_data.num_ancilla_qubits;
  alpha = decomposition.normalization;
}

// ============================================================================
// QUANTUM OPERATION DISPATCHERS
// ============================================================================

void pauli_lcu::prepare(cudaq::qview<> ancilla) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error("pauli_lcu::prepare: ancilla size mismatch");
  block_encoding::prepare(ancilla, kernel_data.state_prep_angles);
}

void pauli_lcu::unprepare(cudaq::qview<> ancilla) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error("pauli_lcu::unprepare: ancilla size mismatch");
  block_encoding::unprepare(ancilla, kernel_data.state_prep_angles);
}

void pauli_lcu::select(cudaq::qview<> ancilla, cudaq::qview<> system) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error("pauli_lcu::select: ancilla size mismatch");
  if (system.size() != n_sys)
    throw std::runtime_error("pauli_lcu::select: system size mismatch");

  block_encoding::select(ancilla, system, kernel_data.term_controls,
                         kernel_data.term_ops, kernel_data.term_lengths,
                         kernel_data.term_signs);
}

void pauli_lcu::controlled_select(cudaq::qubit &control, cudaq::qview<> ancilla,
                                  cudaq::qview<> system) const {
  if (ancilla.size() != n_anc)
    throw std::runtime_error(
        "pauli_lcu::controlled_select: ancilla size mismatch");
  if (system.size() != n_sys)
    throw std::runtime_error(
        "pauli_lcu::controlled_select: system size mismatch");

  controlled_pauli_select_kernel{}(
      control, ancilla, system, kernel_data.term_controls, kernel_data.term_ops,
      kernel_data.term_lengths, kernel_data.term_signs);
}

} // namespace cudaq::algorithms
