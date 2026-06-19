/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "cudaq/algorithms/block_encoding/kernels.h"
#include "cudaq/algorithms/block_encoding/pauli_lcu.h"
#include "cudaq/algorithms/qsvt/qsvt.h"
#include "cudaq/algorithms/qubitization/qubitization.h"
#include "cudaq/python/PythonCppInterop.h"
#include "type_casters.h"

namespace nb = nanobind;

namespace {
template <typename T> nb::object numpy_array(const std::vector<T> &values) {
  auto np = nb::module_::import_("numpy");
  return np.attr("array")(values);
}

nb::object numpy_matrix(const std::vector<double> &values, int dim) {
  auto np = nb::module_::import_("numpy");
  return np.attr("array")(values).attr("reshape")(nb::make_tuple(dim, dim));
}
} // namespace

namespace cudaq::algorithms {

void bind_block_encoding(nb::module_ &mod) {
  auto kernel_data_tuple = [](const pauli_lcu_kernel_data &self) {
    return nb::make_tuple(
        std::vector<double>(self.state_prep_angles),
        std::vector<int>(self.term_controls), std::vector<int>(self.term_ops),
        std::vector<int>(self.term_lengths), std::vector<int>(self.term_signs));
  };

  // ============================================================================
  // PAULI LCU BLOCK ENCODING
  // ============================================================================

  nb::class_<pauli_lcu_metadata>(
      mod, "PauliLCUMetadata",
      R"(Scalar metadata for a Pauli LCU block encoding.)")
      .def(nb::init<>())
      .def_rw("num_system_qubits", &pauli_lcu_metadata::num_system_qubits)
      .def_rw("num_ancilla_qubits", &pauli_lcu_metadata::num_ancilla_qubits)
      .def_rw("num_terms", &pauli_lcu_metadata::num_terms)
      .def_rw("padded_num_terms", &pauli_lcu_metadata::padded_num_terms)
      .def_rw("normalization", &pauli_lcu_metadata::normalization)
      .def_rw("constant_term", &pauli_lcu_metadata::constant_term)
      .def_rw("coefficient_threshold",
              &pauli_lcu_metadata::coefficient_threshold)
      .def("__repr__", [](const pauli_lcu_metadata &self) {
        std::ostringstream oss;
        oss << "PauliLCUMetadata(num_system_qubits=" << self.num_system_qubits
            << ", num_ancilla_qubits=" << self.num_ancilla_qubits
            << ", num_terms=" << self.num_terms
            << ", padded_num_terms=" << self.padded_num_terms
            << ", normalization=" << self.normalization
            << ", constant_term=" << self.constant_term << ")";
        return oss.str();
      });

  nb::class_<pauli_lcu_kernel_data>(
      mod, "PauliLCUKernelData",
      R"(Flattened Pauli LCU layout consumed by CUDA-Q kernels.

This object packages the state-preparation angles and SELECT-table data that
the Python device interop helpers need. The block encoding represents H / alpha,
where alpha is the Pauli LCU normalization. Identity terms are retained in the
encoded operator; PauliLCU.constant_term exposes their sum for algorithms that
want to handle scalar shifts separately.)")
      .def_prop_ro(
          "angles",
          [](const pauli_lcu_kernel_data &self) {
            return std::vector<double>(self.state_prep_angles);
          },
          "State-preparation rotation angles.")
      .def_prop_ro(
          "state_prep_angles",
          [](const pauli_lcu_kernel_data &self) {
            return std::vector<double>(self.state_prep_angles);
          },
          "Alias for angles.")
      .def_prop_ro(
          "term_controls",
          [](const pauli_lcu_kernel_data &self) {
            return std::vector<int>(self.term_controls);
          },
          "Flattened ancilla control patterns for SELECT.")
      .def_prop_ro(
          "term_ops",
          [](const pauli_lcu_kernel_data &self) {
            return std::vector<int>(self.term_ops);
          },
          "Flattened Pauli operation codes for SELECT.")
      .def_prop_ro(
          "term_lengths",
          [](const pauli_lcu_kernel_data &self) {
            return std::vector<int>(self.term_lengths);
          },
          "Number of Pauli operations in each retained term.")
      .def_prop_ro(
          "term_signs",
          [](const pauli_lcu_kernel_data &self) {
            return std::vector<int>(self.term_signs);
          },
          "Sign of each retained LCU coefficient.")
      .def_prop_ro("num_system_qubits",
                   [](const pauli_lcu_kernel_data &self) {
                     return self.num_system_qubits;
                   })
      .def_prop_ro(
          "num_terms",
          [](const pauli_lcu_kernel_data &self) { return self.num_terms; })
      .def_prop_ro("padded_num_terms",
                   [](const pauli_lcu_kernel_data &self) {
                     return self.padded_num_terms;
                   })
      .def_prop_ro("num_ancilla_qubits",
                   [](const pauli_lcu_kernel_data &self) {
                     return self.num_ancilla_qubits;
                   })
      .def(
          "as_tuple", kernel_data_tuple,
          "Return (angles, term_controls, term_ops, term_lengths, term_signs).")
      .def("unpack", kernel_data_tuple,
           "Alias for as_tuple(), intended for kernel interop.")
      .def("__repr__", [](const pauli_lcu_kernel_data &self) {
        std::ostringstream oss;
        oss << "PauliLCUKernelData(num_system_qubits=" << self.num_system_qubits
            << ", num_ancilla_qubits=" << self.num_ancilla_qubits
            << ", num_terms=" << self.num_terms
            << ", padded_num_terms=" << self.padded_num_terms << ")";
        return oss.str();
      });

  nb::class_<pauli_lcu>(
      mod, "PauliLCU",
      R"(Block encoding using Pauli Linear Combination of Unitaries.

This implementation is optimized for Hamiltonians expressed as sums of Pauli
strings (e.g., molecular Hamiltonians from quantum chemistry). It uses:
  - PREPARE: State preparation tree with controlled rotations
  - SELECT: Controlled Pauli operations indexed by ancilla state
  - kernel_data(): Packaged flattened data for Python device interop helpers

The encoding uses log₂(# terms) ancilla qubits and achieves α = ||H||₁.
The good block of the unitary is H / α. Identity terms are included in the
encoded operator; constant_term reports their retained coefficient sum for
workflows that choose to handle scalar shifts outside the block encoding.

Example:
    >>> from cudaq import spin
    >>> import cudaq_algorithms as solvers
    >>> 
    >>> # Define Hamiltonian
    >>> h = 0.5 * spin.x(0) + 0.3 * spin.z(0)
    >>> 
    >>> # Create block encoding
    >>> encoding = algorithms.PauliLCU(h, num_qubits=1)
    >>> 
    >>> print(f"Ancilla qubits needed: {encoding.num_ancilla}")
    >>> print(f"Normalization: {encoding.normalization}")
    >>> 
    >>> # Use in quantum kernel
    >>> @cudaq.kernel
    >>> def my_circuit():
    >>>     anc = cudaq.qvector(encoding.num_ancilla)
    >>>     sys = cudaq.qvector(encoding.num_system)
    >>>     encoding.apply(anc, sys))")
      .def(nb::init<const cudaq::spin_op &, std::size_t>(),
           nb::arg("hamiltonian"), nb::arg("num_qubits"),
           R"(Initialize Pauli LCU block encoding.

Args:
    hamiltonian: Target Hamiltonian as a SpinOperator
    num_qubits: Number of system qubits

Raises:
    RuntimeError: If Hamiltonian contains complex coefficients
    RuntimeError: If Hamiltonian has no terms)")
      .def_prop_ro("num_ancilla", &pauli_lcu::num_ancilla,
                   "Number of ancilla qubits: ⌈log₂(# terms)⌉")
      .def_prop_ro("num_system", &pauli_lcu::num_system,
                   "Number of system qubits")
      .def_prop_ro("normalization", &pauli_lcu::normalization,
                   "Normalization constant: α = ||H||₁ (1-norm)")
      .def_prop_ro("constant_term", &pauli_lcu::constant_term,
                   "Constant identity component retained in the encoding")
      .def_prop_ro("term_count", &pauli_lcu::term_count,
                   "Number of retained LCU terms before padding")
      .def_prop_ro("padded_term_count", &pauli_lcu::padded_term_count,
                   "Number of LCU leaves after power-of-two padding")
      .def("metadata", &pauli_lcu::metadata,
           "Return scalar metadata for transform setup")
      .def(
          "kernel_data",
          [](const pauli_lcu &self) {
            return pauli_lcu_kernel_data(self.get_kernel_data());
          },
          "Return packaged flattened data consumed by Python CUDA-Q kernels.")
      .def("prepare", &pauli_lcu::prepare, nb::arg("ancilla"),
           R"(Apply the PREPARE operation to ancilla qubits.
          
Prepares a superposition state on the ancilla qubits that
encodes the coefficients of the Hamiltonian terms.

Args:
    ancilla: View of ancilla qubits)")
      .def("unprepare", &pauli_lcu::unprepare, nb::arg("ancilla"),
           R"(Apply the PREPARE† (adjoint/uncomputation) operation.

Args:
    ancilla: View of ancilla qubits)")
      .def("select", &pauli_lcu::select, nb::arg("ancilla"), nb::arg("system"),
           R"(Apply the SELECT operation.
          
Applies the appropriate Hamiltonian term conditioned on the
ancilla register state.

Args:
    ancilla: View of ancilla qubits (control register)
    system: View of system qubits (target register))")
      .def("controlled_select", &pauli_lcu::controlled_select,
           nb::arg("control"), nb::arg("ancilla"), nb::arg("system"),
           R"(Apply SELECT controlled by an additional qubit.)")
      .def("apply", &pauli_lcu::apply, nb::arg("ancilla"), nb::arg("system"),
           R"(Apply the full block encoding: PREPARE → SELECT → PREPARE†.

Args:
    ancilla: View of ancilla qubits
    system: View of system qubits)")
      .def(
          "get_angles",
          [](const pauli_lcu &self) { return numpy_array(self.get_angles()); },
          "Get state preparation angles as NumPy array (for debugging)")
      .def(
          "get_term_controls",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_controls());
          },
          "Get binary control patterns as NumPy array (for debugging)")
      .def(
          "get_term_ops",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_ops());
          },
          "Get flattened Pauli operations as NumPy array (for debugging)")
      .def(
          "get_term_lengths",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_lengths());
          },
          "Get number of operators per term as NumPy array (for debugging)")
      .def(
          "get_term_signs",
          [](const pauli_lcu &self) {
            return numpy_array(self.get_term_signs());
          },
          "Get sign of each coefficient as NumPy array (for debugging)");

  cudaq::python::addDeviceKernelInterop<cudaq::qview<>,
                                        const std::vector<double> &>(
      mod, "block_encoding", "prepare",
      "Apply PauliLCU PREPARE inside a CUDA-Q Python kernel.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>,
                                        const std::vector<double> &>(
      mod, "block_encoding", "unprepare",
      "Apply PauliLCU PREPARE dagger inside a CUDA-Q Python kernel.");
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, cudaq::qview<>, const std::vector<int> &,
      const std::vector<int> &, const std::vector<int> &,
      const std::vector<int> &>(
      mod, "block_encoding", "select",
      "Apply PauliLCU SELECT inside a CUDA-Q Python kernel.");
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, cudaq::qview<>, const std::vector<double> &,
      const std::vector<int> &, const std::vector<int> &,
      const std::vector<int> &, const std::vector<int> &>(
      mod, "block_encoding", "apply",
      "Apply a full PauliLCU block encoding inside a CUDA-Q Python kernel.");

  cudaq::python::addDeviceKernelInterop<cudaq::qview<>>(
      mod, "qubitization", "reflect_about_zero",
      "Reflect about the all-zero ancilla state inside a CUDA-Q Python "
      "kernel.");
  cudaq::python::addDeviceKernelInterop<cudaq::qview<>,
                                        const std::vector<double> &>(
      mod, "qubitization", "reflect_about_prepare",
      "Reflect about the PauliLCU PREPARE state inside a CUDA-Q Python "
      "kernel.");
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, cudaq::qview<>, const std::vector<double> &,
      const std::vector<int> &, const std::vector<int> &,
      const std::vector<int> &, const std::vector<int> &>(
      mod, "qubitization", "apply_walk",
      "Apply one PauliLCU qubitization walk step inside a CUDA-Q Python "
      "kernel.");

  auto qubitization_obj = mod.attr("qubitization");
  auto qubitization = nb::borrow<nb::module_>(qubitization_obj.ptr());
  qubitization.def("build_ancilla_zero_projector",
                   &build_ancilla_zero_projector, nb::arg("num_ancilla"),
                   "Build the |0...0><0...0| ancilla projector observable.");
  qubitization.def(
      "build_qubitization_reflection_observable",
      &build_qubitization_reflection_observable, nb::arg("num_ancilla"),
      "Build the 2|0...0><0...0| - I qubitization reflection observable.");
  qubitization.def(
      "build_lcu_select_observable", &build_lcu_select_observable,
      nb::arg("encoding"),
      "Build the observable corresponding to a PauliLCU SELECT operator over "
      "the combined ancilla-system register.");

  cudaq::python::addDeviceKernelInterop<cudaq::qview<>, double>(
      mod, "qsvt", "apply_signal_phase",
      "Apply a QSVT projector phase to the all-zero signal state inside a "
      "CUDA-Q Python kernel.");
  cudaq::python::addDeviceKernelInterop<
      cudaq::qview<>, cudaq::qview<>, const std::vector<double> &,
      const std::vector<int> &, const std::vector<double> &,
      const std::vector<int> &, const std::vector<int> &,
      const std::vector<int> &, const std::vector<int> &>(
      mod, "qsvt", "apply_phase_sequence",
      "Apply a flattened PauliLCU QSVT phase/walk sequence inside a CUDA-Q "
      "Python kernel.");

  // ============================================================================
  // QSVT HOST-SIDE PRIMITIVES
  // ============================================================================

  nb::enum_<qsvt_phase_convention>(
      mod, "QSVTPhaseConvention",
      R"(Convention used to interpret QSP/QSVT phases.)")
      .value("qsvt", qsvt_phase_convention::qsvt)
      .value("qsp", qsvt_phase_convention::qsp);

  nb::class_<qsvt_response_error>(
      mod, "_QSVTResponseError",
      R"(Sampled QSVT response approximation error.)")
      .def(nb::init<>())
      .def_rw("max_abs_error", &qsvt_response_error::max_abs_error)
      .def_rw("rms_error", &qsvt_response_error::rms_error)
      .def_rw("max_error_x", &qsvt_response_error::max_error_x)
      .def_rw("num_samples", &qsvt_response_error::num_samples);

  auto qsvt_obj = mod.attr("qsvt");
  auto qsvt = nb::borrow<nb::module_>(qsvt_obj.ptr());
  qsvt.def(
      "phases_to_poly",
      [](std::vector<double> phases, qsvt_phase_convention convention) {
        return nb::cpp_function(
            [phases = std::move(phases), convention](double x) {
              return evaluate_qsvt_response(phases, x, convention).value;
            });
      },
      nb::arg("phases"), nb::arg("convention") = qsvt_phase_convention::qsvt,
      R"(Construct a host-side polynomial response from QSVT/QSP phases.)");
  qsvt.def(
      "estimate_poly_error",
      [](const std::function<std::complex<double>(double)> &poly,
         const std::function<std::complex<double>(double)> &target,
         nb::tuple domain, std::size_t num_points) {
        if (nb::len(domain) != 2)
          throw std::invalid_argument("domain must contain exactly two values");
        auto min_x = nb::cast<double>(domain[0]);
        auto max_x = nb::cast<double>(domain[1]);
        if (!std::isfinite(min_x) || !std::isfinite(max_x))
          throw std::invalid_argument("domain values must be finite");

        auto sample_points =
            make_uniform_qsvt_sample_points(min_x, max_x, num_points);
        qsvt_response_error error;
        error.num_samples = sample_points.size();

        double sum_squared_error = 0.0;
        for (double x : sample_points) {
          const auto delta = poly(x) - target(x);
          const auto abs_error = std::abs(delta);
          if (!std::isfinite(abs_error))
            throw std::invalid_argument(
                "polynomial and target must produce finite values");
          sum_squared_error += abs_error * abs_error;
          if (abs_error > error.max_abs_error) {
            error.max_abs_error = abs_error;
            error.max_error_x = x;
          }
        }

        error.rms_error = std::sqrt(sum_squared_error / sample_points.size());
        return error;
      },
      nb::arg("poly"), nb::arg("target"),
      nb::arg("domain") = nb::make_tuple(-1.0, 1.0),
      nb::arg("num_points") = 101,
      R"(Estimate a host-side polynomial approximation error on a domain.)");

  mod.def("make_uniform_qsvt_sample_points", &make_uniform_qsvt_sample_points,
          nb::arg("min_x"), nb::arg("max_x"), nb::arg("num_points"));
  mod.def("make_chebyshev_qsvt_sample_points",
          &make_chebyshev_qsvt_sample_points, nb::arg("min_x"),
          nb::arg("max_x"), nb::arg("num_points"));
}

} // namespace cudaq::algorithms
