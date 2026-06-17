/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/krylov/krylov.h"

#include <stdexcept>
#include <string>

namespace cudaq::algorithms::krylov {

namespace {

std::size_t abs_index(std::ptrdiff_t value) {
  return static_cast<std::size_t>(value < 0 ? -value : value);
}

void validate_inputs(const std::vector<double> &moments,
                     std::size_t dimension) {
  if (dimension == 0)
    throw std::runtime_error(
        "build_chebyshev_matrices: dimension must be positive");

  const auto required = required_chebyshev_moments(dimension);
  if (moments.size() < required)
    throw std::runtime_error("build_chebyshev_matrices: expected at least " +
                             std::to_string(required) +
                             " moments for dimension " +
                             std::to_string(dimension));
}

} // namespace

std::size_t required_chebyshev_moments(std::size_t dimension) {
  if (dimension == 0)
    return 0;
  return 2 * dimension;
}

chebyshev_krylov_matrices
build_chebyshev_matrices(const std::vector<double> &moments,
                         std::size_t dimension) {
  validate_inputs(moments, dimension);

  chebyshev_krylov_matrices result;
  result.dimension = dimension;
  result.hamiltonian_matrix.resize(dimension * dimension);
  result.overlap_matrix.resize(dimension * dimension);

  for (std::size_t i = 0; i < dimension; ++i) {
    for (std::size_t j = 0; j < dimension; ++j) {
      const auto row_major = i * dimension + j;
      const auto signed_i = static_cast<std::ptrdiff_t>(i);
      const auto signed_j = static_cast<std::ptrdiff_t>(j);

      result.overlap_matrix[row_major] =
          0.5 * (moments[i + j] + moments[abs_index(signed_i - signed_j)]);
      result.hamiltonian_matrix[row_major] =
          0.25 *
          (moments[i + j + 1] + moments[abs_index(signed_i + signed_j - 1)] +
           moments[abs_index(signed_i - signed_j + 1)] +
           moments[abs_index(signed_i - signed_j - 1)]);
    }
  }

  return result;
}

} // namespace cudaq::algorithms::krylov
