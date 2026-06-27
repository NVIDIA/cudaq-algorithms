#include "cudaq_algorithms.h"
#include "type_casters.h"

#include "cudaq/algorithms/stateprep/ceo.h"
#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/fixed_parameter_ucc.h"
#include "cudaq/algorithms/stateprep/hartree_fock.h"
#include "cudaq/algorithms/stateprep/uccgsd.h"
#include "cudaq/algorithms/stateprep/uccsd.h"
#include "cudaq/algorithms/stateprep/upccgsd.h"
#include "cudaq/python/PythonCppInterop.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <string>

namespace nb = nanobind;

namespace cudaq::algorithms {

namespace {

template <typename... Signature>
void add_device_kernel_interop(nb::module_ &mod, const std::string &mod_name,
                               const std::string &kernel_name,
                               const std::string &docstring) {
  nb::module_ sub = nb::hasattr(mod, mod_name.c_str())
                        ? nb::cast<nb::module_>(mod.attr(mod_name.c_str()))
                        : mod.def_submodule(mod_name.c_str());

  sub.def(kernel_name.c_str(), [](const nb::args &) {}, docstring.c_str());
  const auto mangled_args = cudaq::python::getMangledArgsString<Signature...>();
  const auto private_module_name = nb::cast<std::string>(sub.attr("__name__"));
  cudaq::python::registerDeviceKernel(private_module_name, kernel_name,
                                      mangled_args);

  const auto public_module_name = std::string("cudaq_algorithms.") + mod_name;
  if (public_module_name != private_module_name)
    cudaq::python::registerDeviceKernel(public_module_name, kernel_name,
                                        mangled_args);
}

} // namespace

void bind_stateprep(nb::module_ &mod) {
  add_device_kernel_interop<cudaq::qview<>, const std::vector<double> &,
                            std::size_t, std::size_t>(
      mod, "stateprep", "uccsd",
      "Unitary Coupled Cluster Singles Doubles state-preparation circuit.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<double> &,
                            const std::vector<std::vector<cudaq::pauli_word>> &,
                            const std::vector<std::vector<double>> &>(
      mod, "stateprep", "uccgsd",
      "Unitary Coupled Cluster Generalized Singles Doubles state-preparation "
      "circuit.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<double> &,
                            const std::vector<std::vector<cudaq::pauli_word>> &,
                            const std::vector<std::vector<double>> &>(
      mod, "stateprep", "upccgsd",
      "Unitary Coupled Cluster Generalized Singles and Paired Doubles "
      "circuit.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<double> &,
                            const std::vector<std::vector<cudaq::pauli_word>> &,
                            const std::vector<std::vector<double>> &>(
      mod, "stateprep", "ceo",
      "Coupled Exchange Operator state-preparation circuit.");

  add_device_kernel_interop<cudaq::qview<>, std::size_t>(
      mod, "stateprep", "hartree_fock",
      "Canonical Hartree-Fock occupation circuit.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<std::size_t> &>(
      mod, "stateprep", "hartree_fock_occupation",
      "Hartree-Fock occupation circuit from explicit occupied orbitals.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<double> &,
                            const std::vector<std::vector<cudaq::pauli_word>> &,
                            const std::vector<std::vector<double>> &>(
      mod, "stateprep", "fixed_parameter_ucc",
      "Fixed-parameter UCC-style product over grouped Pauli terms.");

  auto stateprep = nb::cast<nb::module_>(mod.attr("stateprep"));

  nb::class_<cudaq::algorithms::stateprep::hartree_fock_resource_estimate>(
      stateprep, "HartreeFockResources")
      .def(nb::init<>())
      .def_rw("num_qubits", &cudaq::algorithms::stateprep::
                                hartree_fock_resource_estimate::num_qubits)
      .def_rw("num_electrons",
              &cudaq::algorithms::stateprep::hartree_fock_resource_estimate::
                  num_electrons)
      .def_rw("num_x_gates", &cudaq::algorithms::stateprep::
                                 hartree_fock_resource_estimate::num_x_gates);

  nb::class_<cudaq::algorithms::stateprep::fixed_parameter_ucc_plan>(
      stateprep, "FixedParameterUCCPlan")
      .def(nb::init<>())
      .def_rw(
          "num_qubits",
          &cudaq::algorithms::stateprep::fixed_parameter_ucc_plan::num_qubits)
      .def_rw(
          "parameters",
          &cudaq::algorithms::stateprep::fixed_parameter_ucc_plan::parameters)
      .def_rw(
          "pauli_words",
          &cudaq::algorithms::stateprep::fixed_parameter_ucc_plan::pauli_words)
      .def_rw("coefficients", &cudaq::algorithms::stateprep::
                                  fixed_parameter_ucc_plan::coefficients);

  nb::class_<
      cudaq::algorithms::stateprep::fixed_parameter_ucc_resource_estimate>(
      stateprep, "FixedParameterUCCResources")
      .def(nb::init<>())
      .def_rw("num_qubits",
              &cudaq::algorithms::stateprep::
                  fixed_parameter_ucc_resource_estimate::num_qubits)
      .def_rw("num_excitations",
              &cudaq::algorithms::stateprep::
                  fixed_parameter_ucc_resource_estimate::num_excitations)
      .def_rw("num_pauli_rotations",
              &cudaq::algorithms::stateprep::
                  fixed_parameter_ucc_resource_estimate::num_pauli_rotations)
      .def_rw(
          "max_pauli_rotations_per_excitation",
          &cudaq::algorithms::stateprep::fixed_parameter_ucc_resource_estimate::
              max_pauli_rotations_per_excitation);

  stateprep.def("make_hartree_fock_occupation",
                &cudaq::algorithms::stateprep::make_hartree_fock_occupation,
                nb::arg("num_qubits"), nb::arg("num_electrons"),
                nb::arg("spin") = 0);
  stateprep.def("validate_hartree_fock_occupation",
                &cudaq::algorithms::stateprep::validate_hartree_fock_occupation,
                nb::arg("num_qubits"), nb::arg("occupied_orbitals"));
  stateprep.def(
      "estimate_hartree_fock_resources",
      static_cast<cudaq::algorithms::stateprep::hartree_fock_resource_estimate (
              *)(std::size_t, std::size_t)>(
          &cudaq::algorithms::stateprep::estimate_hartree_fock_resources),
      nb::arg("num_qubits"), nb::arg("num_electrons"));
  stateprep.def(
      "estimate_hartree_fock_occupation_resources",
      static_cast<cudaq::algorithms::stateprep::hartree_fock_resource_estimate (
              *)(std::size_t, const std::vector<std::size_t> &)>(
          &cudaq::algorithms::stateprep::estimate_hartree_fock_resources),
      nb::arg("num_qubits"), nb::arg("occupied_orbitals"));

  stateprep.def(
      "make_fixed_parameter_ucc_plan",
      static_cast<cudaq::algorithms::stateprep::fixed_parameter_ucc_plan (*)(
          const std::vector<std::vector<cudaq::pauli_word>> &,
          const std::vector<std::vector<double>> &, const std::vector<double> &,
          std::size_t, double)>(
          &cudaq::algorithms::stateprep::make_fixed_parameter_ucc_plan),
      nb::arg("pauli_words"), nb::arg("coefficients"), nb::arg("parameters"),
      nb::arg("num_qubits") = 0, nb::arg("coefficient_tolerance") = 1.0e-12);
  stateprep.def("make_fixed_parameter_uccsd_plan",
                &cudaq::algorithms::stateprep::make_fixed_parameter_uccsd_plan,
                nb::arg("num_qubits"), nb::arg("num_electrons"),
                nb::arg("parameters"), nb::arg("spin") = 0,
                nb::arg("coefficient_tolerance") = 1.0e-12);
  stateprep.def("make_fixed_parameter_uccgsd_plan",
                &cudaq::algorithms::stateprep::make_fixed_parameter_uccgsd_plan,
                nb::arg("num_qubits"), nb::arg("parameters"),
                nb::arg("only_singles") = false,
                nb::arg("only_doubles") = false,
                nb::arg("coefficient_tolerance") = 1.0e-12);
  stateprep.def(
      "make_fixed_parameter_upccgsd_plan",
      &cudaq::algorithms::stateprep::make_fixed_parameter_upccgsd_plan,
      nb::arg("num_spin_orbitals"), nb::arg("parameters"),
      nb::arg("only_doubles") = false,
      nb::arg("coefficient_tolerance") = 1.0e-12);
  stateprep.def(
      "validate_fixed_parameter_ucc_plan",
      &cudaq::algorithms::stateprep::validate_fixed_parameter_ucc_plan,
      nb::arg("plan"), nb::arg("coefficient_tolerance") = 1.0e-12);
  stateprep.def(
      "estimate_fixed_parameter_ucc_resources",
      &cudaq::algorithms::stateprep::estimate_fixed_parameter_ucc_resources,
      nb::arg("plan"));

  stateprep.def("get_uccsd_excitations",
                &cudaq::algorithms::stateprep::get_uccsd_excitations,
                nb::arg("num_qubits"), nb::arg("num_electrons"),
                nb::arg("spin") = 0);
  stateprep.def("get_num_uccsd_parameters",
                &cudaq::algorithms::stateprep::get_num_uccsd_parameters,
                nb::arg("num_qubits"), nb::arg("num_electrons"),
                nb::arg("spin") = 0);
  stateprep.def("get_uccgsd_pauli_lists",
                &cudaq::algorithms::stateprep::get_uccgsd_pauli_lists,
                nb::arg("num_qubits"), nb::arg("only_singles") = false,
                nb::arg("only_doubles") = false);
  stateprep.def("get_upccgsd_pauli_lists",
                &cudaq::algorithms::stateprep::get_upccgsd_pauli_lists,
                nb::arg("num_qubits"), nb::arg("only_doubles") = false);
  stateprep.def("get_ceo_pauli_lists",
                &cudaq::algorithms::stateprep::get_ceo_pauli_lists,
                nb::arg("num_orbitals"));

  stateprep.def("make_uccsd_operator_pool",
                &cudaq::algorithms::stateprep::make_uccsd_operator_pool,
                nb::arg("num_qubits"), nb::arg("num_electrons"),
                nb::arg("spin") = 0);
  stateprep.def("make_uccgsd_operator_pool",
                &cudaq::algorithms::stateprep::make_uccgsd_operator_pool,
                nb::arg("num_qubits"), nb::arg("only_singles") = false,
                nb::arg("only_doubles") = false);
  stateprep.def("make_upccgsd_operator_pool",
                &cudaq::algorithms::stateprep::make_upccgsd_operator_pool,
                nb::arg("num_qubits"), nb::arg("only_doubles") = false);
  stateprep.def("make_ceo_operator_pool",
                &cudaq::algorithms::stateprep::make_ceo_operator_pool,
                nb::arg("num_orbitals"));
}

} // namespace cudaq::algorithms
