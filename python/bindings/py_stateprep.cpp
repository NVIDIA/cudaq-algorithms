#include "cudaq_algorithms.h"
#include "type_casters.h"

#include "cudaq/algorithms/stateprep/ceo.h"
#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/givens.h"
#include "cudaq/algorithms/stateprep/uccgsd.h"
#include "cudaq/algorithms/stateprep/uccsd.h"
#include "cudaq/algorithms/stateprep/upccgsd.h"
#include "cudaq/python/PythonCppInterop.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
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

  add_device_kernel_interop<cudaq::qview<>, double, std::size_t, std::size_t>(
      mod, "stateprep", "apply_givens_rotation",
      "Adjacent real fermionic Givens rotation.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<std::size_t> &,
                            const std::vector<double> &, std::size_t>(
      mod, "stateprep", "prepare_slater_determinant",
      "Slater determinant state preparation from a flattened Givens schedule.");

  add_device_kernel_interop<cudaq::qview<>, double, double, std::size_t,
                            std::size_t>(
      mod, "stateprep", "apply_phase_givens_rotation",
      "Adjacent phase-aware fermionic Givens rotation.");

  add_device_kernel_interop<cudaq::qview<>, const std::vector<std::size_t> &,
                            const std::vector<double> &,
                            const std::vector<double> &,
                            const std::vector<double> &, std::size_t>(
      mod, "stateprep", "prepare_complex_slater_determinant",
      "Complex Slater determinant state preparation from a flattened Givens "
      "schedule.");

  auto stateprep = nb::cast<nb::module_>(mod.attr("stateprep"));
  nb::class_<cudaq::algorithms::stateprep::givens_rotation>(stateprep,
                                                            "GivensRotation")
      .def(nb::init<>())
      .def_rw("first_orbital",
              &cudaq::algorithms::stateprep::givens_rotation::first_orbital)
      .def_rw("second_orbital",
              &cudaq::algorithms::stateprep::givens_rotation::second_orbital)
      .def_rw("theta", &cudaq::algorithms::stateprep::givens_rotation::theta)
      .def_rw("phase", &cudaq::algorithms::stateprep::givens_rotation::phase);

  nb::class_<cudaq::algorithms::stateprep::givens_rotation_schedule>(
      stateprep, "GivensRotationSchedule")
      .def(nb::init<>())
      .def_rw(
          "num_orbitals",
          &cudaq::algorithms::stateprep::givens_rotation_schedule::num_orbitals)
      .def_rw("num_electrons", &cudaq::algorithms::stateprep::
                                   givens_rotation_schedule::num_electrons)
      .def_rw(
          "rotations",
          &cudaq::algorithms::stateprep::givens_rotation_schedule::rotations)
      .def_rw("final_phases", &cudaq::algorithms::stateprep::
                                  givens_rotation_schedule::final_phases);

  stateprep.def(
      "make_givens_rotation_schedule",
      static_cast<cudaq::algorithms::stateprep::givens_rotation_schedule (*)(
          const std::vector<std::vector<double>> &, double)>(
          &cudaq::algorithms::stateprep::make_givens_rotation_schedule),
      nb::arg("occupied_orbitals"), nb::arg("tolerance") = 1.0e-12);
  stateprep.def(
      "make_complex_givens_rotation_schedule",
      static_cast<cudaq::algorithms::stateprep::givens_rotation_schedule (*)(
          const std::vector<std::vector<std::complex<double>>> &, double)>(
          &cudaq::algorithms::stateprep::make_givens_rotation_schedule),
      nb::arg("occupied_orbitals"), nb::arg("tolerance") = 1.0e-12);
  stateprep.def(
      "make_givens_rotation_schedule",
      static_cast<cudaq::algorithms::stateprep::givens_rotation_schedule (*)(
          const std::vector<std::vector<std::complex<double>>> &, double)>(
          &cudaq::algorithms::stateprep::make_givens_rotation_schedule),
      nb::arg("occupied_orbitals"), nb::arg("tolerance") = 1.0e-12);
  stateprep.def("get_givens_rotation_indices",
                &cudaq::algorithms::stateprep::get_givens_rotation_indices,
                nb::arg("schedule"));
  stateprep.def("get_givens_rotation_angles",
                &cudaq::algorithms::stateprep::get_givens_rotation_angles,
                nb::arg("schedule"));
  stateprep.def("get_givens_rotation_phases",
                &cudaq::algorithms::stateprep::get_givens_rotation_phases,
                nb::arg("schedule"));

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
