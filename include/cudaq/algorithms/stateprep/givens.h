/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"

#include <complex>
#include <cstddef>
#include <vector>

namespace cudaq_algorithms::stateprep {

struct givens_rotation {
  std::size_t first_orbital = 0;
  std::size_t second_orbital = 0;
  double theta = 0.0;
  double phase = 0.0;
};

struct givens_rotation_schedule {
  std::size_t num_orbitals = 0;
  std::size_t num_electrons = 0;
  std::vector<givens_rotation> rotations;
  std::vector<double> final_phases;
};

givens_rotation_schedule make_givens_rotation_schedule(
    const std::vector<std::vector<double>> &occupied_orbitals,
    double tolerance = 1.0e-12);

givens_rotation_schedule make_givens_rotation_schedule(
    const std::vector<std::vector<std::complex<double>>> &occupied_orbitals,
    double tolerance = 1.0e-12);

std::vector<std::size_t>
get_givens_rotation_indices(const givens_rotation_schedule &schedule);

std::vector<double>
get_givens_rotation_angles(const givens_rotation_schedule &schedule);

std::vector<double>
get_givens_rotation_phases(const givens_rotation_schedule &schedule);

/// \pure_device_kernel
///
/// @brief Apply an adjacent real fermionic Givens rotation.
__qpu__ void apply_givens_rotation(cudaq::qview<> qubits, double theta,
                                   std::size_t first_orbital,
                                   std::size_t second_orbital);

/// \pure_device_kernel
///
/// @brief Apply an adjacent phase-aware fermionic Givens rotation.
__qpu__ void apply_phase_givens_rotation(cudaq::qview<> qubits, double theta,
                                         double phase,
                                         std::size_t first_orbital,
                                         std::size_t second_orbital);

/// \pure_device_kernel
///
/// @brief Prepare a Slater determinant from a flattened Givens schedule.
__qpu__ void prepare_slater_determinant(
    cudaq::qview<> qubits, const std::vector<std::size_t> &orbital_indices,
    const std::vector<double> &angles, std::size_t num_electrons);

/// \pure_device_kernel
///
/// @brief Prepare a complex Slater determinant from a flattened Givens schedule.
__qpu__ void prepare_complex_slater_determinant(
    cudaq::qview<> qubits, const std::vector<std::size_t> &orbital_indices,
    const std::vector<double> &angles, const std::vector<double> &phases,
    const std::vector<double> &final_phases, std::size_t num_electrons);

} // namespace cudaq_algorithms::stateprep

namespace cudaq::algorithms::stateprep {
using ::cudaq_algorithms::stateprep::apply_givens_rotation;
using ::cudaq_algorithms::stateprep::givens_rotation;
using ::cudaq_algorithms::stateprep::givens_rotation_schedule;
using ::cudaq_algorithms::stateprep::get_givens_rotation_angles;
using ::cudaq_algorithms::stateprep::get_givens_rotation_indices;
using ::cudaq_algorithms::stateprep::get_givens_rotation_phases;
using ::cudaq_algorithms::stateprep::apply_phase_givens_rotation;
using ::cudaq_algorithms::stateprep::make_givens_rotation_schedule;
using ::cudaq_algorithms::stateprep::prepare_complex_slater_determinant;
using ::cudaq_algorithms::stateprep::prepare_slater_determinant;
} // namespace cudaq::algorithms::stateprep
