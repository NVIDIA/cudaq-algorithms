#pragma once

#include <nanobind/nanobind.h>

namespace cudaq::algorithms {

void bind_fermion(nanobind::module_ &mod);
void bind_stateprep(nanobind::module_ &mod);
void bind_hamiltonian_simulation(nanobind::module_ &mod);

} // namespace cudaq::algorithms
