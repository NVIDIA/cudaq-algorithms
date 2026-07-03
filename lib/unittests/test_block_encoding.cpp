/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cudaq.h"
#include "cudaq/algorithms/block_encoding/pauli_lcu.h"
#include "cudaq/algorithms/detail/qpu_dispatch.h"
#include "cudaq/algorithms/get_state.h"

namespace {

struct reference_pauli_term {
  double coefficient;
  std::vector<std::pair<std::size_t, char>> paulis;
};

cudaq::spin_op
make_spin_term(const std::vector<std::pair<std::size_t, char>> &paulis) {
  using namespace cudaq::spin;

  if (paulis.empty())
    throw std::runtime_error("Test Pauli term must not be empty.");

  cudaq::spin_op term;
  bool has_pauli = false;

  for (const auto &[qubit, pauli] : paulis) {
    cudaq::spin_op next;
    switch (pauli) {
    case 'X':
      next = x(qubit);
      break;
    case 'Y':
      next = y(qubit);
      break;
    case 'Z':
      next = z(qubit);
      break;
    default:
      throw std::runtime_error("Unsupported Pauli in test Hamiltonian.");
    }

    term = has_pauli ? term * next : next;
    has_pauli = true;
  }

  return term;
}

cudaq::spin_op
make_spin_hamiltonian(const std::vector<reference_pauli_term> &terms) {
  cudaq::spin_op h;
  for (const auto &term : terms)
    h += term.coefficient * make_spin_term(term.paulis);
  return h;
}

std::vector<std::complex<double>>
make_normalized_random_ket(std::size_t num_qubits) {
  std::mt19937_64 rng(1337);
  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<std::complex<double>> ket(1ULL << num_qubits);
  double norm_squared = 0.0;
  for (auto &amplitude : ket) {
    amplitude = {normal(rng), normal(rng)};
    norm_squared += std::norm(amplitude);
  }

  const auto inverse_norm = 1.0 / std::sqrt(norm_squared);
  for (auto &amplitude : ket)
    amplitude *= inverse_norm;

  return ket;
}

std::vector<std::complex<double>>
apply_pauli_sum_to_ket(const std::vector<reference_pauli_term> &terms,
                       const std::vector<std::complex<double>> &ket) {
  const std::complex<double> imaginary{0.0, 1.0};
  std::vector<std::complex<double>> out(ket.size(), 0.0);

  for (const auto &term : terms) {
    for (std::size_t column = 0; column < ket.size(); ++column) {
      auto row = column;
      std::complex<double> phase = term.coefficient;

      for (const auto &[qubit, pauli] : term.paulis) {
        const auto bit = (column >> qubit) & 1ULL;
        switch (pauli) {
        case 'X':
          row ^= 1ULL << qubit;
          break;
        case 'Y':
          row ^= 1ULL << qubit;
          phase *= bit == 0 ? imaginary : -imaginary;
          break;
        case 'Z':
          phase *= bit == 0 ? 1.0 : -1.0;
          break;
        default:
          throw std::runtime_error("Unsupported Pauli in dense reference.");
        }
      }

      out[row] += phase * ket[column];
    }
  }

  return out;
}

std::vector<reference_pauli_term> make_nontrivial_4q_hamiltonian_terms() {
  return {{0.70, {{0, 'Z'}}},
          {-0.43, {{1, 'Z'}}},
          {0.31, {{2, 'Z'}}},
          {-0.22, {{3, 'Z'}}},
          {0.19, {{0, 'X'}, {1, 'X'}}},
          {-0.17, {{1, 'Y'}, {2, 'Y'}}},
          {0.13, {{1, 'Z'}, {2, 'Z'}, {3, 'X'}}},
          {0.11, {{0, 'X'}, {1, 'Y'}, {2, 'Y'}, {3, 'X'}}},
          {-0.09, {{0, 'Z'}, {2, 'X'}}},
          {0.07, {{0, 'Y'}, {2, 'Z'}, {3, 'Y'}}}};
}

cudaq::spin_op make_lcu_limit_test_hamiltonian(std::size_t num_terms,
                                               std::size_t num_qubits) {
  using namespace cudaq::spin;

  cudaq::spin_op h;
  for (std::size_t term_idx = 0; term_idx < num_terms; ++term_idx) {
    std::size_t code = term_idx;
    cudaq::spin_op term;

    for (std::size_t q = 0; q < num_qubits; ++q) {
      cudaq::spin_op pauli;
      const auto digit = code % 3;
      if (digit == 0)
        pauli = x(q);
      else if (digit == 1)
        pauli = y(q);
      else
        pauli = z(q);

      term = (q == 0) ? pauli : term * pauli;
      code /= 3;
    }

    h += (1.0 + 1e-6 * static_cast<double>(term_idx)) * term;
  }

  return h;
}

} // namespace

TEST(BlockEncodingTester, checkPauliLCU_H2) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // H2 Hamiltonian (simplified, 4 qubits)
  cudaq::spin_op h2 = -1.0523732 + 0.39793742 * z(0) - 0.39793742 * z(1) -
                      0.01128010 * z(2) + 0.01128010 * z(3) +
                      0.18093120 * x(0) * x(1) * y(2) * y(3);

  // Create block encoding
  pauli_lcu encoding(h2, 4);

  // Check basic properties
  EXPECT_GT(encoding.num_ancilla(), 0);
  EXPECT_EQ(encoding.num_system(), 4);
  EXPECT_GT(encoding.normalization(), 0.0);

  // Check that we have the right number of ancilla qubits
  // log2(6 terms) = 3 ancilla qubits needed
  EXPECT_EQ(encoding.num_ancilla(), 3);

  // Normalization is the 1-norm (sum of |coefficient|) over all six terms,
  // including the identity/constant term:
  //   |−1.0523732| + |0.39793742| + |−0.39793742| + |−0.01128010|
  //   + |0.01128010| + |0.18093120| = 2.05173944.
  EXPECT_NEAR(encoding.normalization(), 2.05173944, 1e-7);

  // Check that angles are computed (2^3 - 1 = 7 angles for 3-qubit tree)
  EXPECT_EQ(encoding.get_angles().size(), 7);

  // Check term data structures have correct sizes
  EXPECT_GT(encoding.get_term_controls().size(), 0);
  EXPECT_GT(encoding.get_term_lengths().size(), 0);
  EXPECT_EQ(encoding.get_term_signs().size(),
            encoding.get_term_lengths().size());
}

TEST(BlockEncodingTester, checkPauliLCU_SimpleXYZ) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // Simple 2-qubit Hamiltonian: H = 0.5*X0 + 0.3*Y1 - 0.2*Z0Z1
  cudaq::spin_op h = 0.5 * x(0) + 0.3 * y(1) - 0.2 * z(0) * z(1);

  // Create block encoding
  pauli_lcu encoding(h, 2);

  // Check properties
  EXPECT_EQ(encoding.num_system(), 2);
  EXPECT_EQ(encoding.num_ancilla(), 2); // log2(3) = 2 ancilla

  // Normalization should be |0.5| + |0.3| + |-0.2| = 1.0
  EXPECT_NEAR(encoding.normalization(), 1.0, 1e-10);

  // Should have 3 terms with correct signs
  EXPECT_EQ(encoding.get_term_signs().size(), 3);
  EXPECT_EQ(encoding.get_term_lengths().size(), 3);

  // Each term should have the right number of Paulis
  auto lengths = encoding.get_term_lengths();
  EXPECT_EQ(lengths[0], 1); // X0 has 1 Pauli
  EXPECT_EQ(lengths[1], 1); // Y1 has 1 Pauli
  EXPECT_EQ(lengths[2], 2); // Z0Z1 has 2 Paulis
}

TEST(BlockEncodingTester, checkLCUDecompositionMetadata) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 2.0 * i(0) + 0.5 * x(0) - 0.25 * z(0);

  auto lcu = decompose_lcu(h, 1);
  EXPECT_EQ(lcu.num_system_qubits, 1);
  EXPECT_EQ(lcu.num_terms, 3);
  EXPECT_EQ(lcu.padded_num_terms, 4);
  EXPECT_EQ(lcu.num_ancilla_qubits, 2);
  EXPECT_NEAR(lcu.normalization, 2.75, 1e-10);
  EXPECT_NEAR(lcu.constant_term, 2.0, 1e-10);

  ASSERT_EQ(lcu.absolute_coefficients.size(), 3);
  ASSERT_EQ(lcu.probabilities.size(), 3);
  ASSERT_EQ(lcu.signs.size(), 3);
  ASSERT_EQ(lcu.identity_terms.size(), 3);
  EXPECT_NEAR(lcu.absolute_coefficients[0], 2.0, 1e-10);
  EXPECT_NEAR(lcu.probabilities[0], 2.0 / 2.75, 1e-10);
  EXPECT_EQ(lcu.signs[0], 1);
  EXPECT_EQ(lcu.signs[2], -1);
  EXPECT_EQ(lcu.identity_terms[0], 1);
  EXPECT_EQ(lcu.identity_terms[1], 0);

  auto kernel_data = make_pauli_lcu_kernel_data(lcu);
  EXPECT_EQ(kernel_data.num_system_qubits, 1);
  EXPECT_EQ(kernel_data.num_terms, 3);
  EXPECT_EQ(kernel_data.padded_num_terms, 4);
  EXPECT_EQ(kernel_data.num_ancilla_qubits, 2);
  EXPECT_EQ(kernel_data.state_prep_angles.size(), 3);
  EXPECT_EQ(kernel_data.term_lengths.size(), 3);
  EXPECT_EQ(kernel_data.term_signs.size(), 3);
  EXPECT_EQ(kernel_data.term_signs[2], -1);

  pauli_lcu encoding(lcu);
  EXPECT_EQ(encoding.num_system(), 1);
  EXPECT_EQ(encoding.num_ancilla(), 2);
  EXPECT_EQ(encoding.term_count(), 3);
  EXPECT_EQ(encoding.padded_term_count(), 4);
  EXPECT_NEAR(encoding.normalization(), 2.75, 1e-10);
  EXPECT_NEAR(encoding.constant_term(), 2.0, 1e-10);
  EXPECT_EQ(encoding.get_kernel_data().term_signs[2], -1);

  auto metadata = encoding.metadata();
  EXPECT_EQ(metadata.num_system_qubits, 1);
  EXPECT_EQ(metadata.num_ancilla_qubits, 2);
  EXPECT_EQ(metadata.num_terms, 3);
  EXPECT_EQ(metadata.padded_num_terms, 4);
  EXPECT_NEAR(metadata.normalization, 2.75, 1e-10);
  EXPECT_NEAR(metadata.constant_term, 2.0, 1e-10);
  EXPECT_NEAR(metadata.coefficient_threshold, lcu.coefficient_threshold, 1e-16);
}

TEST(BlockEncodingTester, checkLCUAncillaLimitValidation) {
  using namespace cudaq::algorithms;

  const auto max_terms = 1ULL
                         << cudaq::algorithms::detail::max_lcu_ancilla_qubits;
  auto boundary_hamiltonian = make_lcu_limit_test_hamiltonian(max_terms, 7);
  auto boundary_lcu = decompose_lcu(boundary_hamiltonian, 7);
  EXPECT_EQ(boundary_lcu.num_ancilla_qubits,
            cudaq::algorithms::detail::max_lcu_ancilla_qubits);
  EXPECT_NO_THROW(pauli_lcu encoding(boundary_lcu));

  auto too_large_hamiltonian =
      make_lcu_limit_test_hamiltonian(max_terms + 1, 7);
  EXPECT_THROW(decompose_lcu(too_large_hamiltonian, 7), std::runtime_error);
}

TEST(BlockEncodingTester, checkLCUDecompositionThreshold) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 1e-14 * x(0) + 0.5 * z(0);

  auto lcu = decompose_lcu(h, 1);
  EXPECT_EQ(lcu.num_terms, 1);
  EXPECT_EQ(lcu.padded_num_terms, 1);
  EXPECT_EQ(lcu.num_ancilla_qubits, 0);
  EXPECT_NEAR(lcu.normalization, 0.5, 1e-10);
  ASSERT_EQ(lcu.probabilities.size(), 1);
  EXPECT_NEAR(lcu.probabilities[0], 1.0, 1e-10);

  auto kernel_data = make_pauli_lcu_kernel_data(lcu);
  EXPECT_EQ(kernel_data.state_prep_angles.size(), 0);
  EXPECT_EQ(kernel_data.term_controls.size(), 0);
  EXPECT_EQ(kernel_data.term_lengths.size(), 1);
}

TEST(BlockEncodingTester, checkKernelExecution) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // Simple Hamiltonian
  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);

  // Create block encoding
  pauli_lcu encoding(h, 1);

  // Test kernel: Apply PREPARE on ancilla qubits
  auto prepare_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    encoding.prepare(anc);
    // The state should be prepared now
    // In a real test, we'd measure and check probabilities
  };

  // This should compile and run without error
  EXPECT_NO_THROW(prepare_test());

  // Test SELECT kernel
  auto select_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.select(anc, sys);
  };

  EXPECT_NO_THROW(select_test());

  // Test full block encoding
  auto full_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.apply(anc, sys);
  };

  EXPECT_NO_THROW(full_test());
}

TEST(BlockEncodingTester, checkBlockEncodingMatchesDenseHamiltonianAction) {
  using namespace cudaq::algorithms;

  constexpr std::size_t num_qubits = 4;
  const auto reference_terms = make_nontrivial_4q_hamiltonian_terms();
  auto h = make_spin_hamiltonian(reference_terms);
  pauli_lcu encoding(h, num_qubits);

  ASSERT_EQ(encoding.num_system(), num_qubits);
  ASSERT_EQ(encoding.term_count(), reference_terms.size());
  ASSERT_EQ(encoding.num_ancilla(), 4);

  auto ket = make_normalized_random_ket(num_qubits);
  cudaq::state input_state(ket);
  const auto expected = apply_pauli_sum_to_ket(reference_terms, ket);

  auto apply_block_encoding = [&encoding](cudaq::state state) __qpu__ {
    cudaq::qvector<> system(state);
    cudaq::qvector<> ancilla(encoding.num_ancilla());
    encoding.apply(ancilla, system);
  };

  auto encoded_state = cudaq::get_state(apply_block_encoding, input_state);
  const auto system_dimension = 1ULL << num_qubits;
  const auto normalization = encoding.normalization();

  auto reverse_bits = [](std::size_t value, std::size_t width) {
    std::size_t reversed = 0;
    for (std::size_t bit = 0; bit < width; ++bit)
      if ((value >> bit) & 1ULL)
        reversed |= 1ULL << (width - 1 - bit);
    return reversed;
  };

  double l2_error = 0.0;
  double expected_probability = 0.0;
  double good_probability = 0.0;

  for (std::size_t i = 0; i < system_dimension; ++i) {
    // The initialized system register is returned in big-endian basis order,
    // and the all-zero LCU ancilla subspace occupies the low ancilla bits.
    const auto output_index = reverse_bits(i, num_qubits)
                              << encoding.num_ancilla();
    const auto expected_amplitude = expected[i] / normalization;
    const auto actual_amplitude = encoded_state[output_index];
    l2_error += std::norm(actual_amplitude - expected_amplitude);
    expected_probability += std::norm(expected_amplitude);
    good_probability += std::norm(actual_amplitude);
  }

  EXPECT_NEAR(std::sqrt(l2_error), 0.0, 1e-10);
  EXPECT_NEAR(good_probability, expected_probability, 1e-10);
}

// Regression test: a single-term LCU has zero ancilla qubits, so a negative
// coefficient cannot be carried by the multi-controlled-Z sign correction and
// must be applied explicitly. Without that, -c * P silently encodes +c * P.
TEST(BlockEncodingTester, checkSingleTermNegativeCoefficientEncodesSign) {
  using namespace cudaq::algorithms;

  constexpr std::size_t num_qubits = 2;
  const std::vector<reference_pauli_term> reference_terms = {
      {-0.5, {{0, 'X'}, {1, 'Z'}}}};
  auto h = make_spin_hamiltonian(reference_terms);
  pauli_lcu encoding(h, num_qubits);

  ASSERT_EQ(encoding.num_system(), num_qubits);
  ASSERT_EQ(encoding.term_count(), 1u);
  ASSERT_EQ(encoding.num_ancilla(), 0u);

  auto ket = make_normalized_random_ket(num_qubits);
  cudaq::state input_state(ket);
  const auto expected = apply_pauli_sum_to_ket(reference_terms, ket);

  auto apply_block_encoding = [&encoding](cudaq::state state) __qpu__ {
    cudaq::qvector<> system(state);
    cudaq::qvector<> ancilla(encoding.num_ancilla());
    encoding.apply(ancilla, system);
  };

  auto encoded_state = cudaq::get_state(apply_block_encoding, input_state);
  const auto system_dimension = 1ULL << num_qubits;
  const auto normalization = encoding.normalization();

  auto reverse_bits = [](std::size_t value, std::size_t width) {
    std::size_t reversed = 0;
    for (std::size_t bit = 0; bit < width; ++bit)
      if ((value >> bit) & 1ULL)
        reversed |= 1ULL << (width - 1 - bit);
    return reversed;
  };

  double l2_error = 0.0;
  for (std::size_t i = 0; i < system_dimension; ++i) {
    const auto output_index = reverse_bits(i, num_qubits);
    const auto expected_amplitude = expected[i] / normalization;
    const auto actual_amplitude = encoded_state[output_index];
    l2_error += std::norm(actual_amplitude - expected_amplitude);
  }

  EXPECT_NEAR(std::sqrt(l2_error), 0.0, 1e-10);
}

TEST(BlockEncodingTester, checkIdentityTerm) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // Hamiltonian with constant term
  cudaq::spin_op h = 2.0 * i(0) + 0.5 * x(0);

  // Create block encoding - should handle identity term
  pauli_lcu encoding(h, 1);

  EXPECT_EQ(encoding.num_system(), 1);
  EXPECT_NEAR(encoding.normalization(), 2.5, 1e-10);
}

TEST(BlockEncodingTester, checkLargeHamiltonian) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // Build a larger Hamiltonian (8 qubits, multiple terms)
  cudaq::spin_op h;
  for (int i = 0; i < 7; ++i) {
    h += 0.1 * z(i) * z(i + 1);
  }
  for (int i = 0; i < 8; ++i) {
    h += 0.05 * x(i);
  }

  // Create block encoding
  pauli_lcu encoding(h, 8);

  // Should need ceil(log2(15)) = 4 ancilla qubits
  EXPECT_EQ(encoding.num_ancilla(), 4);
  EXPECT_EQ(encoding.num_system(), 8);

  // Normalization should be sum of absolute coefficients
  // 7 * 0.1 + 8 * 0.05 = 0.7 + 0.4 = 1.1
  EXPECT_NEAR(encoding.normalization(), 1.1, 1e-10);
}
