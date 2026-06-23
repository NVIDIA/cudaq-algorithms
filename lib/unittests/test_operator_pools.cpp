/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/excitations.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <set>
#include <vector>

namespace stateprep = cudaq::algorithms::stateprep;

namespace {

void expect_max_degree_less_than(const std::vector<cudaq::spin_op> &operators,
                                 std::size_t max_degree) {
  for (const auto &op : operators)
    EXPECT_LT(op.max_degree(), max_degree);
}

bool is_expected_uccsd_coefficient(std::complex<double> coefficient) {
  const auto real = coefficient.real();
  return std::abs(coefficient.imag()) < 1e-12 &&
         (std::abs(std::abs(real) - 0.5) < 1e-12 ||
          std::abs(std::abs(real) - 0.125) < 1e-12);
}

} // namespace

TEST(UCCSDOperatorPool, GenerateWithDefaultConfig) {
  auto operators = stateprep::make_uccsd_operator_pool(4, 2);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 3);
  expect_max_degree_less_than(operators, 4);
}

TEST(UCCSDOperatorPool, GenerateWithCustomCoefficients) {
  auto operators = stateprep::make_uccsd_operator_pool(4, 2);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 3);

  for (const auto &op : operators) {
    EXPECT_LT(op.max_degree(), 4);
    for (const auto &term : op)
      EXPECT_TRUE(is_expected_uccsd_coefficient(term.evaluate_coefficient()));
  }
}

TEST(UCCSDOperatorPool, GenerateWithOddElectrons) {
  auto operators = stateprep::make_uccsd_operator_pool(6, 3, 1);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 8);
  expect_max_degree_less_than(operators, 6);
}

TEST(UCCSDOperatorPool, GenerateWithLargeSystem) {
  auto operators = stateprep::make_uccsd_operator_pool(20, 10);

  ASSERT_FALSE(operators.empty());
  EXPECT_EQ(operators.size(), 875);
  expect_max_degree_less_than(operators, 20);
}

TEST(UCCSDOperatorPool, GeneratesCorrectOperators) {
  auto operators = stateprep::make_uccsd_operator_pool(4, 2);

  std::set<std::size_t> all_qubits{0, 1, 2, 3};
  for (auto &op : operators)
    op.canonicalize(all_qubits);

  std::vector<cudaq::spin_op> expected;
  expected.emplace_back(-0.500 * cudaq::spin_op::from_word("XZYI") +
                        +0.500 * cudaq::spin_op::from_word("YZXI"));
  expected.emplace_back(-0.500 * cudaq::spin_op::from_word("IXZY") +
                        +0.500 * cudaq::spin_op::from_word("IYZX"));
  expected.emplace_back(-0.125 * cudaq::spin_op::from_word("YYYX") +
                        -0.125 * cudaq::spin_op::from_word("YXXX") +
                        +0.125 * cudaq::spin_op::from_word("XXYX") +
                        -0.125 * cudaq::spin_op::from_word("YYXY") +
                        +0.125 * cudaq::spin_op::from_word("XYYY") +
                        +0.125 * cudaq::spin_op::from_word("XXXY") +
                        +0.125 * cudaq::spin_op::from_word("YXYY") +
                        -0.125 * cudaq::spin_op::from_word("XYXX"));

  ASSERT_EQ(expected.size(), operators.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(expected[i], operators[i]) << "Mismatch at index " << i;
}
