/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/ceo.h"
#include "cudaq/algorithms/stateprep/excitations.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

namespace stateprep = cudaq::algorithms::stateprep;

namespace {

std::string canonical_string(cudaq::spin_op op) {
  op.canonicalize();
  return op.to_string();
}

bool has_expected_ceo_coefficients(const cudaq::spin_op &op) {
  for (const auto &term : op) {
    const auto coefficient = term.evaluate_coefficient();
    const auto abs_real = std::abs(coefficient.real());
    if (std::abs(coefficient.imag()) > 1e-12)
      return false;
    if (std::abs(abs_real - 0.5) > 1e-12 && std::abs(abs_real - 0.25) > 1e-12)
      return false;
  }
  return true;
}

/// CEO pool operators are Hermitian generators: G(row,col) == conj(G(col,row)).
bool is_hermitian_generator(const cudaq::spin_op &op) {
  const auto g = op.to_matrix();
  for (std::size_t row = 0; row < g.rows(); ++row)
    for (std::size_t col = 0; col < g.cols(); ++col)
      if (std::abs(std::conj(g(col, row)) - g(row, col)) > 1e-10)
        return false;
  return true;
}

} // namespace

TEST(CEOOperatorPool, CorrectNumberOfOperators) {
  EXPECT_EQ(stateprep::make_ceo_operator_pool(2).size(), 4);
  EXPECT_EQ(stateprep::make_ceo_operator_pool(3).size(), 24);
  EXPECT_EQ(stateprep::make_ceo_operator_pool(4).size(), 96);
}

TEST(CEOOperatorPool, NoDuplicateOperators) {
  auto operators = stateprep::make_ceo_operator_pool(4);
  std::set<std::string> unique_operators;

  for (auto op : operators)
    unique_operators.insert(canonical_string(op));

  EXPECT_EQ(unique_operators.size(), operators.size());
}

TEST(CEOOperatorPool, AllOperatorsNonEmpty) {
  auto operators = stateprep::make_ceo_operator_pool(2);

  for (const auto &op : operators) {
    EXPECT_GT(op.num_terms(), 0);
    for (const auto &term : op)
      EXPECT_FALSE(term.is_identity());
  }
}

TEST(CEOOperatorPool, VerifyOperatorCoefficients) {
  auto operators = stateprep::make_ceo_operator_pool(4);

  for (const auto &op : operators)
    EXPECT_TRUE(has_expected_ceo_coefficients(op));
}

TEST(CEOOperatorPool, ConsistentWithStateprepPauliLists) {
  auto operators = stateprep::make_ceo_operator_pool(4);
  auto [pauli_lists, coefficients] = stateprep::get_ceo_pauli_lists(4);

  ASSERT_EQ(operators.size(), pauli_lists.size());
  ASSERT_EQ(operators.size(), coefficients.size());

  for (std::size_t i = 0; i < operators.size(); ++i) {
    EXPECT_EQ(operators[i].num_terms(), pauli_lists[i].size());
    EXPECT_EQ(operators[i].num_terms(), coefficients[i].size());
  }
}

TEST(CEOOperatorPool, OperatorsAreHermitianGenerators) {
  auto operators = stateprep::make_ceo_operator_pool(2);
  for (std::size_t i = 0; i < operators.size(); ++i)
    EXPECT_TRUE(is_hermitian_generator(operators[i]))
        << "Operator " << i << " (G) is not Hermitian";
}
