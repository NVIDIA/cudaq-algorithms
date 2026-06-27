/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/fixed_parameter_ucc.h"
#include "cudaq/algorithms/stateprep/hartree_fock.h"

#include <gtest/gtest.h>

namespace stateprep = cudaq::algorithms::stateprep;

TEST(HartreeFockStatePrep, BuildsCanonicalOccupation) {
  const auto occupation = stateprep::make_hartree_fock_occupation(6, 4);
  EXPECT_EQ(occupation, (std::vector<std::size_t>{0, 1, 2, 3}));

  const auto resources = stateprep::estimate_hartree_fock_resources(6, 4);
  EXPECT_EQ(resources.num_qubits, 6);
  EXPECT_EQ(resources.num_electrons, 4);
  EXPECT_EQ(resources.num_x_gates, 4);
}

TEST(HartreeFockStatePrep, BuildsOpenShellOccupation) {
  // spin == 0 (closed shell) stays contiguous.
  EXPECT_EQ(stateprep::make_hartree_fock_occupation(8, 4, 0),
            (std::vector<std::size_t>{0, 1, 2, 3}));
  // spin > 0 uses the interleaved alpha/beta convention: {0, 1, 2, 4}.
  EXPECT_EQ(stateprep::make_hartree_fock_occupation(8, 4, 2),
            (std::vector<std::size_t>{0, 1, 2, 4}));
  // Odd num_qubits is invalid for spin > 0.
  EXPECT_THROW(stateprep::make_hartree_fock_occupation(5, 3, 1),
               std::invalid_argument);
}

TEST(HartreeFockStatePrep, ValidatesExplicitOccupation) {
  EXPECT_NO_THROW(
      stateprep::validate_hartree_fock_occupation(5, {0, 2, 4}));
  EXPECT_THROW(stateprep::validate_hartree_fock_occupation(5, {0, 5}),
               std::invalid_argument);
  EXPECT_THROW(stateprep::validate_hartree_fock_occupation(5, {0, 2, 2}),
               std::invalid_argument);
}

TEST(FixedParameterUCCStatePrep, BuildsUCCSDPlan) {
  const std::vector<double> parameters = {0.1, -0.2, 0.3};
  const auto plan = stateprep::make_fixed_parameter_uccsd_plan(4, 2, parameters);

  EXPECT_EQ(plan.num_qubits, 4);
  EXPECT_EQ(plan.parameters, parameters);
  EXPECT_EQ(plan.pauli_words.size(), parameters.size());
  EXPECT_EQ(plan.coefficients.size(), parameters.size());

  const auto resources = stateprep::estimate_fixed_parameter_ucc_resources(plan);
  EXPECT_EQ(resources.num_qubits, 4);
  EXPECT_EQ(resources.num_excitations, 3);
  EXPECT_GT(resources.num_pauli_rotations, 0);
}

TEST(FixedParameterUCCStatePrep, BuildsGeneralizedPlans) {
  const std::vector<double> uccgsd_parameters(9, 0.1);
  const auto uccgsd_plan =
      stateprep::make_fixed_parameter_uccgsd_plan(4, uccgsd_parameters);
  EXPECT_EQ(uccgsd_plan.pauli_words.size(), uccgsd_parameters.size());

  const std::vector<double> upccgsd_parameters(3, 0.1);
  const auto upccgsd_plan =
      stateprep::make_fixed_parameter_upccgsd_plan(4, upccgsd_parameters);
  EXPECT_EQ(upccgsd_plan.pauli_words.size(), upccgsd_parameters.size());
}

TEST(FixedParameterUCCStatePrep, RejectsMalformedPlans) {
  EXPECT_THROW(stateprep::make_fixed_parameter_uccsd_plan(4, 2, {0.1, 0.2}),
               std::invalid_argument);

  stateprep::fixed_parameter_ucc_plan plan;
  plan.num_qubits = 1;
  plan.parameters = {0.1};
  plan.pauli_words = {{cudaq::pauli_word("XI")}};
  plan.coefficients = {{0.5}};
  EXPECT_THROW(stateprep::validate_fixed_parameter_ucc_plan(plan),
               std::invalid_argument);
}
