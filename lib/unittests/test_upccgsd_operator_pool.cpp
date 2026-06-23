/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/upccgsd.h"

#include <cmath>
#include <gtest/gtest.h>
#include <set>
#include <string>

namespace stateprep = cudaq::algorithms::stateprep;

namespace {

std::size_t expected_upccgsd_singles(std::size_t n_spin_orbitals) {
  const auto n_orbitals = n_spin_orbitals / 2;
  return n_orbitals * (n_orbitals - 1);
}

std::size_t expected_upccgsd_doubles(std::size_t n_spin_orbitals) {
  const auto n_orbitals = n_spin_orbitals / 2;
  return n_orbitals * (n_orbitals - 1) / 2;
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

TEST(UPCCGSDOperatorPool, CorrectNumberOfOperators) {
  for (auto n_spin_orbitals : {4ul, 8ul, 20ul}) {
    auto operators = stateprep::make_upccgsd_operator_pool(n_spin_orbitals);
    auto doubles = stateprep::make_upccgsd_operator_pool(n_spin_orbitals, true);

    EXPECT_EQ(operators.size(), expected_upccgsd_singles(n_spin_orbitals) +
                                    expected_upccgsd_doubles(n_spin_orbitals));
    EXPECT_EQ(doubles.size(), expected_upccgsd_doubles(n_spin_orbitals));
  }
}

TEST(UPCCGSDOperatorPool, NoDuplicateOperators) {
  auto operators = stateprep::make_upccgsd_operator_pool(8);
  std::set<std::string> unique_operators;

  for (auto op : operators)
    unique_operators.insert(canonical_string(op));

  EXPECT_EQ(unique_operators.size(), operators.size());
}

TEST(UPCCGSDOperatorPool, AllOperatorsNonEmpty) {
  auto operators = stateprep::make_upccgsd_operator_pool(4);

  for (const auto &op : operators) {
    EXPECT_GT(op.num_terms(), 0);
    for (const auto &term : op)
      EXPECT_FALSE(term.is_identity());
  }
}

TEST(UPCCGSDOperatorPool, VerifyOperatorCoefficients) {
  auto operators = stateprep::make_upccgsd_operator_pool(8);

  for (const auto &op : operators)
    EXPECT_TRUE(has_expected_coefficients(op));
}

TEST(UPCCGSDOperatorPool, ConsistentWithStateprepPauliLists) {
  auto operators = stateprep::make_upccgsd_operator_pool(8);
  auto [pauli_lists, coefficients] =
      stateprep::get_upccgsd_pauli_lists(8, false);

  ASSERT_EQ(operators.size(), pauli_lists.size());
  ASSERT_EQ(operators.size(), coefficients.size());

  for (std::size_t i = 0; i < operators.size(); ++i) {
    EXPECT_EQ(operators[i].num_terms(), pauli_lists[i].size());
    EXPECT_EQ(operators[i].num_terms(), coefficients[i].size());
  }
}

TEST(UPCCGSDOperatorPool, MinimalSystem) {
  EXPECT_TRUE(stateprep::make_upccgsd_operator_pool(2).empty());
  EXPECT_EQ(stateprep::make_upccgsd_operator_pool(4).size(), 3);
}
