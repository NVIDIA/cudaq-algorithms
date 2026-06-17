/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under  *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/
#pragma once

#include "cudaq.h"

#include <vector>

namespace cudaq_algorithms::block_encoding {

/// \pure_device_kernel
///
/// @brief Apply a PauliLCU PREPARE circuit from flattened kernel data.
__qpu__ void prepare(cudaq::qview<> ancilla,
                     const std::vector<double> &state_prep_angles);

/// \pure_device_kernel
///
/// @brief Apply the adjoint of a PauliLCU PREPARE circuit.
__qpu__ void unprepare(cudaq::qview<> ancilla,
                       const std::vector<double> &state_prep_angles);

/// \pure_device_kernel
///
/// @brief Apply a PauliLCU SELECT circuit from flattened kernel data.
__qpu__ void select(cudaq::qview<> ancilla, cudaq::qview<> system,
                    const std::vector<int> &term_controls,
                    const std::vector<int> &term_ops,
                    const std::vector<int> &term_lengths,
                    const std::vector<int> &term_signs);

/// \pure_device_kernel
///
/// @brief Apply a full PauliLCU block encoding from flattened kernel data.
__qpu__ void apply(cudaq::qview<> ancilla, cudaq::qview<> system,
                   const std::vector<double> &state_prep_angles,
                   const std::vector<int> &term_controls,
                   const std::vector<int> &term_ops,
                   const std::vector<int> &term_lengths,
                   const std::vector<int> &term_signs);

} // namespace cudaq_algorithms::block_encoding

namespace cudaq_algorithms::qubitization {

/// \pure_device_kernel
///
/// @brief Reflect about the all-zero state on an ancilla register.
__qpu__ void reflect_about_zero(cudaq::qview<> ancilla);

/// \pure_device_kernel
///
/// @brief Reflect about the PauliLCU PREPARE state from flattened kernel data.
__qpu__ void
reflect_about_prepare(cudaq::qview<> ancilla,
                      const std::vector<double> &state_prep_angles);

/// \pure_device_kernel
///
/// @brief Apply one PauliLCU qubitization walk step from flattened kernel data.
__qpu__ void apply_walk(cudaq::qview<> ancilla, cudaq::qview<> system,
                        const std::vector<double> &state_prep_angles,
                        const std::vector<int> &term_controls,
                        const std::vector<int> &term_ops,
                        const std::vector<int> &term_lengths,
                        const std::vector<int> &term_signs);

/// \pure_device_kernel
///
/// @brief Apply one adjoint PauliLCU qubitization walk step from flattened
/// data.
__qpu__ void apply_adjoint_walk(cudaq::qview<> ancilla, cudaq::qview<> system,
                                const std::vector<double> &state_prep_angles,
                                const std::vector<int> &term_controls,
                                const std::vector<int> &term_ops,
                                const std::vector<int> &term_lengths,
                                const std::vector<int> &term_signs);

/// \pure_device_kernel
///
/// @brief Apply repeated PauliLCU qubitization walk steps from flattened data.
__qpu__ void apply_walk_power(cudaq::qview<> ancilla, cudaq::qview<> system,
                              const std::vector<double> &state_prep_angles,
                              const std::vector<int> &term_controls,
                              const std::vector<int> &term_ops,
                              const std::vector<int> &term_lengths,
                              const std::vector<int> &term_signs, int power);

/// \pure_device_kernel
///
/// @brief Apply repeated adjoint PauliLCU qubitization walk steps.
__qpu__ void
apply_adjoint_walk_power(cudaq::qview<> ancilla, cudaq::qview<> system,
                         const std::vector<double> &state_prep_angles,
                         const std::vector<int> &term_controls,
                         const std::vector<int> &term_ops,
                         const std::vector<int> &term_lengths,
                         const std::vector<int> &term_signs, int power);

} // namespace cudaq_algorithms::qubitization

namespace cudaq_algorithms::qsvt_primitives {

/// \pure_device_kernel
///
/// @brief Apply a QSVT projector phase to the all-zero signal state.
__qpu__ void apply_signal_phase(cudaq::qview<> signal, double phase);

/// \pure_device_kernel
///
/// @brief Apply a QSP-style signal phase to the all-zero signal state.
__qpu__ void apply_qsp_signal_phase(cudaq::qview<> signal, double phase);

/// \pure_device_kernel
///
/// @brief Apply a flattened PauliLCU QSVT phase/walk sequence.
__qpu__ void apply_sequence(cudaq::qview<> signal, cudaq::qview<> system,
                            const std::vector<double> &phases,
                            const std::vector<int> &walk_directions,
                            const std::vector<double> &state_prep_angles,
                            const std::vector<int> &term_controls,
                            const std::vector<int> &term_ops,
                            const std::vector<int> &term_lengths,
                            const std::vector<int> &term_signs);

/// \pure_device_kernel
///
/// @brief Apply a flattened PauliLCU QSP phase/walk sequence.
__qpu__ void apply_qsp_sequence(cudaq::qview<> signal, cudaq::qview<> system,
                                const std::vector<double> &phases,
                                const std::vector<int> &walk_directions,
                                const std::vector<double> &state_prep_angles,
                                const std::vector<int> &term_controls,
                                const std::vector<int> &term_ops,
                                const std::vector<int> &term_lengths,
                                const std::vector<int> &term_signs);

} // namespace cudaq_algorithms::qsvt_primitives

namespace cudaq_algorithms::qsvt {

/// \pure_device_kernel
///
/// @brief Apply a QSVT projector phase to the all-zero signal state.
__qpu__ void apply_signal_phase(cudaq::qview<> signal, double phase);

/// \pure_device_kernel
///
/// @brief Apply a flattened PauliLCU QSVT phase/walk sequence.
///
/// @details This is the Python-facing QSVT execution primitive. Phases from
/// QSPPACK or other QSP generators should be converted to this projector-phase
/// convention before calling this helper.
__qpu__ void apply_phase_sequence(cudaq::qview<> signal, cudaq::qview<> system,
                                  const std::vector<double> &phases,
                                  const std::vector<int> &walk_directions,
                                  const std::vector<double> &state_prep_angles,
                                  const std::vector<int> &term_controls,
                                  const std::vector<int> &term_ops,
                                  const std::vector<int> &term_lengths,
                                  const std::vector<int> &term_signs);

} // namespace cudaq_algorithms::qsvt

namespace cudaq::algorithms::block_encoding {
using ::cudaq_algorithms::block_encoding::apply;
using ::cudaq_algorithms::block_encoding::prepare;
using ::cudaq_algorithms::block_encoding::select;
using ::cudaq_algorithms::block_encoding::unprepare;
} // namespace cudaq::algorithms::block_encoding

namespace cudaq::algorithms::qubitization {
using ::cudaq_algorithms::qubitization::apply_adjoint_walk;
using ::cudaq_algorithms::qubitization::apply_adjoint_walk_power;
using ::cudaq_algorithms::qubitization::apply_walk;
using ::cudaq_algorithms::qubitization::apply_walk_power;
using ::cudaq_algorithms::qubitization::reflect_about_prepare;
using ::cudaq_algorithms::qubitization::reflect_about_zero;
} // namespace cudaq::algorithms::qubitization

namespace cudaq::algorithms::qsvt_primitives {
using ::cudaq_algorithms::qsvt_primitives::apply_qsp_sequence;
using ::cudaq_algorithms::qsvt_primitives::apply_qsp_signal_phase;
using ::cudaq_algorithms::qsvt_primitives::apply_sequence;
using ::cudaq_algorithms::qsvt_primitives::apply_signal_phase;
} // namespace cudaq::algorithms::qsvt_primitives

namespace cudaq::algorithms::qsvt {
using ::cudaq_algorithms::qsvt::apply_phase_sequence;
using ::cudaq_algorithms::qsvt::apply_signal_phase;
} // namespace cudaq::algorithms::qsvt
