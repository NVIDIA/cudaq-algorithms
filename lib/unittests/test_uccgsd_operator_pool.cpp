/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/uccgsd.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <set>
#include <string>
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
