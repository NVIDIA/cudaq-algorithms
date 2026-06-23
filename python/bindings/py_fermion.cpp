#include "cudaq_algorithms.h"
#include "type_casters.h"

#include "cudaq/algorithms/fermion/bravyi_kitaev.h"
#include "cudaq/algorithms/fermion/jordan_wigner.h"

#include <complex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace cudaq::algorithms {

namespace {

using complex = std::complex<double>;
using array = nb::ndarray<nb::numpy, complex>;

std::vector<complex> copy_array(const array &input) {
  const auto *data = static_cast<const complex *>(input.data());
  std::size_t size = 1;
  for (std::size_t i = 0; i < input.ndim(); ++i)
    size *= input.shape(i);
  return {data, data + size};
}

std::size_t validate_square_rank(const array &input, std::size_t rank,
                                 const char *name) {
  if (input.ndim() != rank)
    throw std::runtime_error(std::string(name) + " has the wrong rank.");
  const std::size_t n = input.shape(0);
  for (std::size_t i = 1; i < rank; ++i)
    if (input.shape(i) != n)
      throw std::runtime_error(std::string(name) + " dimensions must match.");
  return n;
}

template <typename Transform>
cudaq::spin_op apply_transform(const array &first, std::optional<array> second,
                               double scalar_offset, double tolerance,
                               Transform transform) {
  if (first.ndim() == 2) {
    const auto n = validate_square_rank(first, 2, "one_body");
    std::vector<complex> one_body = copy_array(first);
    std::vector<complex> two_body;
    if (second) {
      const auto n_two = validate_square_rank(*second, 4, "two_body");
      if (n_two != n)
        throw std::runtime_error("one_body and two_body dimensions differ.");
      two_body = copy_array(*second);
    }
    return transform(one_body, two_body, n, scalar_offset, tolerance);
  }

  if (first.ndim() == 4) {
    if (second)
      throw std::runtime_error(
          "second tensor is invalid when first is rank 4.");
    const auto n = validate_square_rank(first, 4, "two_body");
    return transform({}, copy_array(first), n, scalar_offset, tolerance);
  }

  throw std::runtime_error("expected rank-2 one_body or rank-4 two_body.");
}

} // namespace

void bind_fermion(nb::module_ &mod) {
  auto fermion = mod.def_submodule("fermion");

  fermion.def(
      "jordan_wigner",
      [](const array &first, std::optional<array> second, double scalar_offset,
         double tolerance) {
        return apply_transform(
            first, second, scalar_offset, tolerance,
            [](const std::vector<complex> &one_body,
               const std::vector<complex> &two_body, std::size_t n,
               double offset, double tol) {
              return cudaq::algorithms::fermion::jordan_wigner(
                  one_body, two_body, n, offset, tol);
            });
      },
      nb::arg("one_body_or_two_body"), nb::arg("two_body") = nb::none(),
      nb::arg("scalar_offset") = 0.0, nb::arg("tolerance") = 1e-15);

  fermion.def(
      "bravyi_kitaev",
      [](const array &first, std::optional<array> second, double scalar_offset,
         double tolerance) {
        return apply_transform(
            first, second, scalar_offset, tolerance,
            [](const std::vector<complex> &one_body,
               const std::vector<complex> &two_body, std::size_t n,
               double offset, double tol) {
              return cudaq::algorithms::fermion::bravyi_kitaev(
                  one_body, two_body, n, offset, tol);
            });
      },
      nb::arg("one_body_or_two_body"), nb::arg("two_body") = nb::none(),
      nb::arg("scalar_offset") = 0.0, nb::arg("tolerance") = 1e-15);
}

} // namespace cudaq::algorithms
