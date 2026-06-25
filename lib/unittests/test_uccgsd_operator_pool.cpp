/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/uccgsd.h"

#include <cctype>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace stateprep = cudaq::algorithms::stateprep;

namespace {

std::size_t expected_uccgsd_singles(std::size_t n_qubits) {
  return n_qubits * (n_qubits - 1) / 2;
}

std::size_t expected_uccgsd_doubles(std::size_t n_qubits) {
  if (n_qubits < 4)
    return 0;
  return n_qubits * (n_qubits - 1) * (n_qubits - 2) * (n_qubits - 3) / 8;
}

std::string canonical_string(cudaq::spin_op op) {
  op.canonicalize();
  return op.to_string();
}

bool has_expected_coefficients(const cudaq::spin_op &op) {
  for (const auto &term : op) {
    const auto coefficient = term.evaluate_coefficient();
    const auto abs_real = std::abs(coefficient.real());
    if (std::abs(coefficient.imag()) > 1e-12)
      return false;
    if (std::abs(abs_real - 0.5) > 1e-12 && std::abs(abs_real - 0.125) > 1e-12)
      return false;
  }
  return true;
}

/// Classify each operator as a single or double excitation by counting the
/// number of single-qubit X/Y Pauli factors it carries. Singles act on two
/// qubits (a handful of X/Y factors per term); doubles act on four.
std::pair<std::size_t, std::size_t>
count_singles_and_doubles(const std::vector<cudaq::spin_op> &ops) {
  std::size_t singles = 0, doubles = 0;
  for (const auto &op : ops) {
    const std::string op_str = op.to_string();
    std::size_t xy_count = 0;
    for (std::size_t i = 0; i + 1 < op_str.length(); ++i)
      if ((op_str[i] == 'X' || op_str[i] == 'Y') &&
          std::isdigit(static_cast<unsigned char>(op_str[i + 1])))
        ++xy_count;
    if (xy_count <= 10)
      ++singles;
    else
      ++doubles;
  }
  return {singles, doubles};
}

/// Verify that the Hermitian generator G has an anti-Hermitian iG, i.e.
/// (iG)^dagger = -iG, which is what makes exp(theta * iG) unitary.
bool is_anti_hermitian_generator(const cudaq::spin_op &op) {
  const auto g = op.to_matrix();
  // G itself must be Hermitian: G(row,col) == conj(G(col,row)).
  for (std::size_t row = 0; row < g.rows(); ++row)
    for (std::size_t col = 0; col < g.cols(); ++col)
      if (std::abs(std::conj(g(col, row)) - g(row, col)) > 1e-10)
        return false;
  // Then (iG)^dagger + iG == 0 reduces to the same Hermiticity condition on G,
  // so a Hermitian G guarantees an anti-Hermitian iG.
  return true;
}

} // namespace

TEST(UCCGSDOperatorPool, CorrectNumberOfOperators) {
  for (auto n_qubits : {2ul, 4ul, 8ul}) {
    auto operators = stateprep::make_uccgsd_operator_pool(n_qubits);
    EXPECT_EQ(operators.size(), expected_uccgsd_singles(n_qubits) +
                                    expected_uccgsd_doubles(n_qubits));
  }

  EXPECT_EQ(stateprep::make_uccgsd_operator_pool(4, true, false).size(), 6);
  EXPECT_EQ(stateprep::make_uccgsd_operator_pool(4, false, true).size(), 3);
}

TEST(UCCGSDOperatorPool, NoDuplicateOperators) {
  auto operators = stateprep::make_uccgsd_operator_pool(6);
  std::set<std::string> unique_operators;

  for (auto op : operators)
    unique_operators.insert(canonical_string(op));

  EXPECT_EQ(unique_operators.size(), operators.size());
}

TEST(UCCGSDOperatorPool, AllOperatorsNonEmpty) {
  auto operators = stateprep::make_uccgsd_operator_pool(4);

  for (const auto &op : operators) {
    EXPECT_GT(op.num_terms(), 0);
    for (const auto &term : op)
      EXPECT_FALSE(term.is_identity());
  }
}

TEST(UCCGSDOperatorPool, VerifyOperatorCoefficients) {
  auto operators = stateprep::make_uccgsd_operator_pool(4);

  for (const auto &op : operators)
    EXPECT_TRUE(has_expected_coefficients(op));
}

TEST(UCCGSDOperatorPool, ConsistentWithStateprepPauliLists) {
  auto operators = stateprep::make_uccgsd_operator_pool(4);
  auto [pauli_lists, coefficients] =
      stateprep::get_uccgsd_pauli_lists(4, false, false);

  ASSERT_EQ(operators.size(), pauli_lists.size());
  ASSERT_EQ(operators.size(), coefficients.size());

  for (std::size_t i = 0; i < operators.size(); ++i) {
    EXPECT_EQ(operators[i].num_terms(), pauli_lists[i].size());
    EXPECT_EQ(operators[i].num_terms(), coefficients[i].size());
  }
}

TEST(UCCGSDOperatorPool, ScalingBehavior) {
  for (auto n_qubits : {4ul, 6ul, 8ul}) {
    auto operators = stateprep::make_uccgsd_operator_pool(n_qubits);
    EXPECT_EQ(operators.size(), expected_uccgsd_singles(n_qubits) +
                                    expected_uccgsd_doubles(n_qubits));
  }
}

TEST(UCCGSDOperatorPool, CorrectSinglesAndDoublesCount) {
  auto operators = stateprep::make_uccgsd_operator_pool(4);
  auto [singles, doubles] = count_singles_and_doubles(operators);
  EXPECT_EQ(singles, expected_uccgsd_singles(4));
  EXPECT_EQ(doubles, expected_uccgsd_doubles(4));
}

// Regression test for an ordering bug in the double-excitation generation.
// The original code used an over-restrictive index ordering
// (p > q && q > r && r > s) that produced only a single double excitation for
// 4 qubits; the fix (p > q && r > s) yields all three unique pairings.
TEST(UCCGSDOperatorPool, RegressionTestForOrderingBug) {
  auto operators = stateprep::make_uccgsd_operator_pool(4);
  auto [singles, doubles] = count_singles_and_doubles(operators);
  EXPECT_GT(doubles, 1u)
      << "Regression: the ordering bug generated only one double excitation";
  EXPECT_EQ(doubles, 3u)
      << "Should generate all three unique double excitations for 4 qubits";
}

TEST(UCCGSDOperatorPool, SinglesOnlyGeneration) {
  auto operators = stateprep::make_uccgsd_operator_pool(4, true, false);
  EXPECT_EQ(operators.size(), expected_uccgsd_singles(4));
  auto [singles, doubles] = count_singles_and_doubles(operators);
  EXPECT_EQ(singles, expected_uccgsd_singles(4));
  EXPECT_EQ(doubles, 0u);
}

TEST(UCCGSDOperatorPool, DoublesOnlyGeneration) {
  auto operators = stateprep::make_uccgsd_operator_pool(4, false, true);
  EXPECT_EQ(operators.size(), expected_uccgsd_doubles(4));
  auto [singles, doubles] = count_singles_and_doubles(operators);
  EXPECT_EQ(singles, 0u);
  EXPECT_EQ(doubles, expected_uccgsd_doubles(4));
}

// UCCGSD operators G are Hermitian generators, so iG is anti-Hermitian and
// exp(theta * iG) is unitary.
TEST(UCCGSDOperatorPool, OperatorsAreAntiHermitian) {
  auto operators = stateprep::make_uccgsd_operator_pool(4);
  for (std::size_t i = 0; i < operators.size(); ++i)
    EXPECT_TRUE(is_anti_hermitian_generator(operators[i]))
        << "Operator " << i << " (G) is not a Hermitian generator";
}
