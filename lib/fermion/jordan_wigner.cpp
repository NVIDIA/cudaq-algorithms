#include "cudaq/algorithms/fermion/jordan_wigner.h"

#include "integrals.h"

#include <cassert>
#include <complex>

namespace cudaq::algorithms::fermion {

cudaq::spin_op jordan_wigner(const std::vector<std::complex<double>> &one_body,
                             const std::vector<std::complex<double>> &two_body,
                             std::size_t num_spin_orbitals,
                             double scalar_offset, double tolerance) {
  validate_integrals(one_body, two_body, num_spin_orbitals);

  const std::size_t nqubit = num_spin_orbitals;
  cudaq::spin_op spin_hamiltonian = scalar_offset * cudaq::spin_op_term();

  auto is_complex_zero = [tolerance](const std::complex<double> &z) {
    return std::abs(z.real()) < tolerance && std::abs(z.imag()) < tolerance;
  };

  auto adag = [](std::size_t, std::size_t j) {
    cudaq::spin_op_term zprod;
    for (std::size_t k = 0; k < j; k++)
      zprod *= cudaq::spin::z(k);
    return 0.5 * zprod *
           (cudaq::spin::x(j) - std::complex<double>{0, 1} * cudaq::spin::y(j));
  };

  auto a = [](std::size_t, std::size_t j) {
    cudaq::spin_op_term zprod;
    for (std::size_t k = 0; k < j; k++)
      zprod *= cudaq::spin::z(k);
    return 0.5 * zprod *
           (cudaq::spin::x(j) + std::complex<double>{0, 1} * cudaq::spin::y(j));
  };

  for (std::size_t i = 0; i < nqubit; i++) {
    for (std::size_t j = 0; j < nqubit; j++) {
      auto coefficient = one_body_at(one_body, nqubit, i, j);
      if (!is_complex_zero(coefficient))
        spin_hamiltonian += coefficient * adag(nqubit, i) * a(nqubit, j);
    }
  }

  for (std::size_t i = 0; i < nqubit; i++) {
    for (std::size_t j = 0; j < nqubit; j++) {
      for (std::size_t k = 0; k < nqubit; k++) {
        for (std::size_t l = 0; l < nqubit; l++) {
          auto coefficient = two_body_at(two_body, nqubit, i, j, k, l);
          if (!is_complex_zero(coefficient))
            spin_hamiltonian += coefficient * adag(nqubit, i) *
                                adag(nqubit, j) * a(nqubit, k) * a(nqubit, l);
        }
      }
    }
  }

  return spin_hamiltonian.canonicalize().trim(tolerance);
}

cudaq::spin_op jordan_wigner(const std::vector<std::complex<double>> &one_body,
                             std::size_t num_spin_orbitals,
                             double scalar_offset, double tolerance) {
  return jordan_wigner(one_body, {}, num_spin_orbitals, scalar_offset,
                       tolerance);
}

} // namespace cudaq::algorithms::fermion
