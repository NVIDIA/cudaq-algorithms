/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/givens.h"

#include <cmath>
#include <gtest/gtest.h>

namespace stateprep = cudaq::algorithms::stateprep;

TEST(GivensStatePrep, BuildsTwoOrbitalSchedule) {
  const double theta = 0.37;
  const std::vector<std::vector<double>> occupied_orbitals = {
      {std::cos(theta)}, {std::sin(theta)}};

  auto schedule =
      stateprep::make_givens_rotation_schedule(occupied_orbitals);

  ASSERT_EQ(schedule.num_orbitals, 2);
  ASSERT_EQ(schedule.num_electrons, 1);
  ASSERT_EQ(schedule.rotations.size(), 1);
  EXPECT_EQ(schedule.rotations[0].first_orbital, 0);
  EXPECT_EQ(schedule.rotations[0].second_orbital, 1);
  EXPECT_NEAR(schedule.rotations[0].theta, theta, 1.0e-12);

  auto indices = stateprep::get_givens_rotation_indices(schedule);
  auto angles = stateprep::get_givens_rotation_angles(schedule);
  ASSERT_EQ(indices, (std::vector<std::size_t>{0, 1}));
  ASSERT_EQ(angles.size(), 1);
  EXPECT_NEAR(angles[0], theta, 1.0e-12);
}

TEST(GivensStatePrep, RejectsNonOrthonormalInputs) {
  const std::vector<std::vector<double>> occupied_orbitals = {{1.0, 0.0},
                                                             {1.0, 0.0}};

  EXPECT_THROW(stateprep::make_givens_rotation_schedule(occupied_orbitals),
               std::invalid_argument);
}
