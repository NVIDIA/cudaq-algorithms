#pragma once

#include "cudaq/spin_op.h"

#include <complex>
#include <cstddef>
#include <vector>

namespace cudaq::algorithms::fermion {

cudaq::spin_op jordan_wigner(const std::vector<std::complex<double>> &one_body,
                             const std::vector<std::complex<double>> &two_body,
                             std::size_t num_spin_orbitals,
                             double scalar_offset = 0.0,
                             double tolerance = 1e-15);

cudaq::spin_op jordan_wigner(const std::vector<std::complex<double>> &one_body,
                             std::size_t num_spin_orbitals,
                             double scalar_offset = 0.0,
                             double tolerance = 1e-15);

} // namespace cudaq::algorithms::fermion
