/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include <tuple>
#include <vector>

// NOTE: Unlike the sibling stateprep kernels (ceo/uccgsd/upccgsd), the UCCSD
// kernels are declared in the flat `cudaq_algorithms::stateprep` namespace and
// then re-exported into `cudaq::algorithms::stateprep` via the `using`
// declarations at the bottom of this header. This is a deliberate workaround:
// declaring the `uccsd`/`single_excitation`/`double_excitation` `__qpu__`
// kernels directly in the nested `cudaq::algorithms::stateprep` namespace
// crashes the nvq++ quake bridge (QuakeBridgeVisitor::VisitCallExpr) when the
// `uccsd` kernel calls its sibling excitation kernels. The flat namespace
// avoids the faulty mangling path. Keep this structure until the nvq++ bug is
// fixed; the public API remains `cudaq::algorithms::stateprep::*`.
namespace cudaq_algorithms::stateprep {

/// @brief Represents a list of excitations
using excitation_list = std::vector<std::vector<std::size_t>>;

/// @brief Get UCCSD excitations for a given system
/// @param num_qubits Number of qubits in the system
/// @param num_electrons Number of electrons in the system
/// @param spin Spin of the system
/// @return Tuple containing five excitation lists
std::tuple<excitation_list, excitation_list, excitation_list, excitation_list,
           excitation_list>
get_uccsd_excitations(std::size_t num_qubits, std::size_t num_electrons,
                      std::size_t spin = 0);

/// @brief Calculate the number of UCCSD parameters
/// @param num_qubits Number of qubits in the system
/// @param num_electrons Number of electrons in the system
/// @param spin Spin of the system (default 0)
/// @return Number of UCCSD parameters
std::size_t get_num_uccsd_parameters(std::size_t num_qubits,
                                     std::size_t num_electrons,
                                     std::size_t spin = 0);

/// \pure_device_kernel
///
/// @brief Perform a single excitation operation
/// @param qubits Qubit register
/// @param theta Rotation angle
/// @param p_occ Occupied orbital index
/// @param q_virt Virtual orbital index
__qpu__ void single_excitation(cudaq::qview<> qubits, double theta,
                               std::size_t p_occ, std::size_t q_virt);

/// \pure_device_kernel
///
/// @brief Perform a double excitation operation
/// @param qubits Qubit register
/// @param theta Rotation angle
/// @param p_occ First occupied orbital index
/// @param q_occ Second occupied orbital index
/// @param r_virt First virtual orbital index
/// @param s_virt Second virtual orbital index
__qpu__ void double_excitation(cudaq::qview<> qubits, double theta,
                               std::size_t p_occ, std::size_t q_occ,
                               std::size_t r_virt, std::size_t s_virt);

/// \pure_device_kernel
///
/// @brief Apply UCCSD ansatz to a qubit register
/// @param qubits Qubit register
/// @param thetas Vector of rotation angles
/// @param numElectrons Number of electrons in the system
/// @param spin Spin of the system
__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons, std::size_t spin);

/// \pure_device_kernel
///
/// @brief Apply UCCSD ansatz to a qubit vector
/// @param qubits Qubit vector
/// @param thetas Vector of rotation angles
/// @param numElectrons Number of electrons in the system
__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons);

} // namespace cudaq_algorithms::stateprep

namespace cudaq::algorithms::stateprep {
using ::cudaq_algorithms::stateprep::double_excitation;
using ::cudaq_algorithms::stateprep::excitation_list;
using ::cudaq_algorithms::stateprep::get_num_uccsd_parameters;
using ::cudaq_algorithms::stateprep::get_uccsd_excitations;
using ::cudaq_algorithms::stateprep::single_excitation;
using ::cudaq_algorithms::stateprep::uccsd;
} // namespace cudaq::algorithms::stateprep
