/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_algorithms.h"

#include "cudaq/algorithms/krylov/krylov.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <sstream>

namespace nb = nanobind;

namespace {

nb::object numpy_matrix(const std::vector<double> &values,
                        std::size_t dimension) {
  auto np = nb::module_::import_("numpy");
  return np.attr("array")(values).attr("reshape")(
      nb::make_tuple(dimension, dimension));
}

} // namespace

namespace cudaq::algorithms {

void bind_krylov(nb::module_ &mod) {
  auto krylov = mod.def_submodule(
      "krylov",
      "Krylov-subspace post-processing utilities for moment-based algorithms.");

  nb::class_<cudaq::algorithms::krylov::chebyshev_krylov_matrices>(
      krylov, "ChebyshevKrylovMatrices",
      R"(Dense Chebyshev Krylov Hamiltonian and overlap matrices.)")
      .def_rw("hamiltonian_data",
              &cudaq::algorithms::krylov::chebyshev_krylov_matrices::
                  hamiltonian_matrix,
              "Flattened row-major Hamiltonian matrix data.")
      .def_rw(
          "overlap_data",
          &cudaq::algorithms::krylov::chebyshev_krylov_matrices::overlap_matrix,
          "Flattened row-major overlap matrix data.")
      .def_rw("dimension",
              &cudaq::algorithms::krylov::chebyshev_krylov_matrices::dimension,
              "Krylov matrix dimension.")
      .def(
          "hamiltonian_matrix",
          [](const cudaq::algorithms::krylov::chebyshev_krylov_matrices &self) {
            return numpy_matrix(self.hamiltonian_matrix, self.dimension);
          })
      .def(
          "overlap_matrix",
          [](const cudaq::algorithms::krylov::chebyshev_krylov_matrices &self) {
            return numpy_matrix(self.overlap_matrix, self.dimension);
          })
      .def(
          "__repr__",
          [](const cudaq::algorithms::krylov::chebyshev_krylov_matrices &self) {
            std::ostringstream oss;
            oss << "ChebyshevKrylovMatrices(dimension=" << self.dimension
                << ")";
            return oss.str();
          });

  krylov.def("required_chebyshev_moments",
             &cudaq::algorithms::krylov::required_chebyshev_moments,
             nb::arg("dimension"));

  krylov.def("build_chebyshev_matrices",
             &cudaq::algorithms::krylov::build_chebyshev_matrices,
             nb::arg("moments"), nb::arg("dimension"));
}

} // namespace cudaq::algorithms
