#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cudaq::algorithms::fermion {

inline void
validate_integrals(const std::vector<std::complex<double>> &one_body,
                   const std::vector<std::complex<double>> &two_body,
                   std::size_t num_spin_orbitals) {
  const auto n = num_spin_orbitals;
  if (!one_body.empty() && one_body.size() != n * n)
    throw std::invalid_argument("one_body must have shape (n, n).");
  if (!two_body.empty() && two_body.size() != n * n * n * n)
    throw std::invalid_argument("two_body must have shape (n, n, n, n).");
}

inline std::complex<double>
one_body_at(const std::vector<std::complex<double>> &data, std::size_t n,
            std::size_t i, std::size_t j) {
  if (data.empty())
    return 0.0;
  return data[i * n + j];
}

inline std::complex<double>
two_body_at(const std::vector<std::complex<double>> &data, std::size_t n,
            std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  if (data.empty())
    return 0.0;
  return data[((i * n + j) * n + k) * n + l];
}

} // namespace cudaq::algorithms::fermion
