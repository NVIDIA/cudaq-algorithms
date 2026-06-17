/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/algorithms/krylov/krylov.h"

TEST(KrylovTester, checkRequiredChebyshevMoments) {
  using namespace cudaq::algorithms::krylov;

  EXPECT_EQ(required_chebyshev_moments(0), 0);
  EXPECT_EQ(required_chebyshev_moments(1), 2);
  EXPECT_EQ(required_chebyshev_moments(4), 8);
}

TEST(KrylovTester, checkChebyshevMatrixConstruction) {
  using namespace cudaq::algorithms::krylov;

  const std::vector<double> moments = {1.0, 0.2, -0.1, 0.05};
  auto matrices = build_chebyshev_matrices(moments, 2);

  ASSERT_EQ(matrices.dimension, 2);
  ASSERT_EQ(matrices.overlap_matrix.size(), 4);
  ASSERT_EQ(matrices.hamiltonian_matrix.size(), 4);

  EXPECT_NEAR(matrices.overlap_matrix[0], 1.0, 1e-12);
  EXPECT_NEAR(matrices.overlap_matrix[1], 0.2, 1e-12);
  EXPECT_NEAR(matrices.overlap_matrix[2], 0.2, 1e-12);
  EXPECT_NEAR(matrices.overlap_matrix[3], 0.45, 1e-12);

  EXPECT_NEAR(matrices.hamiltonian_matrix[0], 0.2, 1e-12);
  EXPECT_NEAR(matrices.hamiltonian_matrix[1], 0.45, 1e-12);
  EXPECT_NEAR(matrices.hamiltonian_matrix[2], 0.45, 1e-12);
  EXPECT_NEAR(matrices.hamiltonian_matrix[3], 0.1625, 1e-12);
}

TEST(KrylovTester, checkChebyshevMatrixValidation) {
  using namespace cudaq::algorithms::krylov;

  EXPECT_THROW(build_chebyshev_matrices({1.0, 0.0}, 0), std::runtime_error);
  EXPECT_THROW(build_chebyshev_matrices({1.0, 0.0, 0.0}, 2),
               std::runtime_error);
}
