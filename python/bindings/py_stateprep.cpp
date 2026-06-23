#include "cudaq_algorithms.h"
#include "type_casters.h"

#include "cudaq/algorithms/stateprep/ceo.h"
#include "cudaq/algorithms/stateprep/excitations.h"
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
  cudaq::python::registerDeviceKernel(
      nb::cast<std::string>(sub.attr("__name__")), kernel_name,
      cudaq::python::getMangledArgsString<Signature...>());
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

  auto stateprep = nb::cast<nb::module_>(mod.attr("stateprep"));
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
