/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/uccsd.h"

#include <gtest/gtest.h>
#include <vector>

namespace stateprep = cudaq::algorithms::stateprep;

TEST(UCCSD, H2Excitations) {
  auto [singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
        doubles_beta] = stateprep::get_uccsd_excitations(4, 2, 0);

  ASSERT_EQ(singles_alpha.size(), 1);
  ASSERT_EQ(singles_beta.size(), 1);
  ASSERT_EQ(doubles_mixed.size(), 1);
  EXPECT_TRUE(doubles_alpha.empty());
  EXPECT_TRUE(doubles_beta.empty());

  EXPECT_EQ(singles_alpha[0], (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(singles_beta[0], (std::vector<std::size_t>{1, 3}));
  EXPECT_EQ(doubles_mixed[0], (std::vector<std::size_t>{0, 1, 3, 2}));
  EXPECT_EQ(stateprep::get_num_uccsd_parameters(4, 2, 0), 3);
}

TEST(UCCSD, ActiveSpaceExcitations) {
  auto [singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
        doubles_beta] = stateprep::get_uccsd_excitations(8, 4, 0);

  EXPECT_EQ(singles_alpha,
            (stateprep::excitation_list{{0, 4}, {0, 6}, {2, 4}, {2, 6}}));
  EXPECT_EQ(singles_beta,
            (stateprep::excitation_list{{1, 5}, {1, 7}, {3, 5}, {3, 7}}));
  EXPECT_EQ(doubles_alpha, (stateprep::excitation_list{{0, 2, 4, 6}}));
  EXPECT_EQ(doubles_beta, (stateprep::excitation_list{{1, 3, 5, 7}}));
  EXPECT_EQ(doubles_mixed.size(), 16);
  EXPECT_EQ(stateprep::get_num_uccsd_parameters(8, 4, 0), 26);
}

TEST(UCCSD, OpenShellExcitationCounts) {
  auto [singles_alpha, singles_beta, doubles_mixed, doubles_alpha,
        doubles_beta] = stateprep::get_uccsd_excitations(6, 3, 1);

  EXPECT_EQ(singles_alpha.size(), 2);
  EXPECT_EQ(singles_beta.size(), 2);
  EXPECT_EQ(doubles_mixed.size(), 4);
  EXPECT_TRUE(doubles_alpha.empty());
  EXPECT_TRUE(doubles_beta.empty());
  EXPECT_EQ(stateprep::get_num_uccsd_parameters(6, 3, 1), 8);
}
