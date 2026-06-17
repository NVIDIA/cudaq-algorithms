#include "cudaq_algorithms.h"

#include "cudaq/algorithms/version.h"

#include <nanobind/nanobind.h>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;

NB_MODULE(_pycudaq_algorithms, mod) {
  nb::module_::import_("cudaq");

  try {
    cudaq::algorithms::bind_fermion(mod);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("bind_fermion failed: ") + e.what());
  }

  try {
    cudaq::algorithms::bind_stateprep(mod);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("bind_stateprep failed: ") + e.what());
  }

  try {
    cudaq::algorithms::bind_hamiltonian_simulation(mod);
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("bind_hamiltonian_simulation failed: ") + e.what());
  }

  try {
    std::stringstream ss;
    ss << "CUDA-Q Algorithms " << cudaq::algorithms::get_version() << " ("
       << cudaq::algorithms::get_full_repository_version() << ")";
    mod.attr("__version__") = nb::str(ss.str().c_str());
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("version binding failed: ") +
                             e.what());
  }

  nb::set_leak_warnings(false);
}
