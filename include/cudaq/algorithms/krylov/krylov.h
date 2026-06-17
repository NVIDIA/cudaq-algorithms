/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <vector>

namespace cudaq::algorithms::krylov {

/// @brief Dense matrices for a Chebyshev Krylov subspace.
/// @details Matrices are stored in row-major order with dimension x dimension
/// entries. They can be consumed by a generalized eigensolver outside this
/// library, e.g. scipy.linalg.eigh(H, S).
struct chebyshev_krylov_matrices {
  std::vector<double> hamiltonian_matrix;
  std::vector<double> overlap_matrix;
  std::size_t dimension = 0;
};

/// @brief Return the number of Chebyshev moments required for a Krylov basis.
/// @details A dimension-d basis needs moments mu_0 through mu_{2d-1}.
std::size_t required_chebyshev_moments(std::size_t dimension);

/// @brief Build Chebyshev Krylov Hamiltonian and overlap matrices.
/// @details Given moments mu_k = <psi|T_k(A)|psi>, construct matrices in the
/// Chebyshev Krylov basis using product-to-sum identities:
///
/// S_ij = 1/2 (mu_{i+j} + mu_{|i-j|})
/// H_ij = 1/4 (mu_{i+j+1} + mu_{|i+j-1|} +
///             mu_{|i-j+1|} + mu_{|i-j-1|})
///
/// This is a generic post-processing primitive used by QEL-style workflows. It
/// intentionally does not run a QEL application or prescribe how moments are
/// measured.
chebyshev_krylov_matrices
build_chebyshev_matrices(const std::vector<double> &moments,
                         std::size_t dimension);

} // namespace cudaq::algorithms::krylov
