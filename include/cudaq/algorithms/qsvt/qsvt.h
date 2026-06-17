/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/algorithms/qubitization/qubitization.h"

#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace cudaq::algorithms {

/// @brief Direction of the qubitization walk used between QSVT phases.
enum class qsvt_walk_direction { forward, adjoint };

/// @brief Host-side convention used to interpret a QSP/QSVT phase sequence.
enum class qsvt_phase_convention { qsvt, qsp };

/// @brief Host-side matrix function transform represented by QSVT metadata.
enum class qsvt_transform_kind {
  custom,
  linear_solve,
  real_time_hamiltonian_simulation,
  imaginary_time_hamiltonian_simulation
};

/// @brief Primitive walk direction codes passed across the QPU boundary.
/// @details Host-side policy objects expand to these integer codes before a
/// kernel invocation so kernels do not need to consume richer C++ objects.
inline constexpr int qsvt_forward_walk = 0;
inline constexpr int qsvt_adjoint_walk = 1;

/// @brief Host-side response of a QSP/QSVT phase sequence at a scalar x.
/// @details The value is the upper-left matrix element of the abstract
/// two-dimensional signal model. magnitude and probability are convenience
/// diagnostics derived from that complex value.
struct qsvt_response {
  std::complex<double> value;
  double magnitude = 0.0;
  double probability = 0.0;
};

/// @brief Host-side error summary for a sampled QSVT response approximation.
struct qsvt_response_error {
  double max_abs_error = 0.0;
  double rms_error = 0.0;
  double max_error_x = 0.0;
  std::size_t num_samples = 0;
};

/// @brief QSVT host/device API boundary.
///
/// qsvt_phase_sequence, qsvt_sequence_policy, and qsvt_plan are host-side
/// validation and metadata types. QPU-facing helpers consume qview objects,
/// pauli_lcu encodings, primitive numeric values, plain std::vector<double>
/// phase data, and plain std::vector<int> walk direction data extracted from a
/// validated host-side sequence or plan. Do not pass qsvt_plan directly into
/// __qpu__ kernels.

/// @brief Apply a phase to the all-zero signal/ancilla state.
/// @details Implements exp(i phase |0><0|) on the signal register by mapping
/// |0...0> to |1...1>, applying a multi-controlled r1 phase, and unmapping.
/// This helper intentionally uses explicit arities to match CUDA-Q controlled
/// gate lowering constraints used by the Pauli LCU primitives.
__qpu__ inline void apply_qsvt_signal_phase(cudaq::qview<> signal,
                                            double phase) {
  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);

  std::size_t num_signal = signal.size();
  if (num_signal == 0) {
    return;
  } else if (num_signal == 1) {
    r1(phase, signal[0]);
  } else if (num_signal == 2) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1]);
  } else if (num_signal == 3) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2]);
  } else if (num_signal == 4) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3]);
  } else if (num_signal == 5) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4]);
  } else if (num_signal == 6) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5]);
  } else if (num_signal == 7) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6]);
  } else if (num_signal == 8) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7]);
  } else if (num_signal == 9) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7], signal[8]);
  } else if (num_signal == 10) {
    r1<cudaq::ctrl>(phase, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7], signal[8],
                    signal[9]);
  }

  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);
}

/// @brief Kernel functor wrapper for a QSVT signal phase.
struct qsvt_signal_phase {
  void operator()(cudaq::qview<> signal, double phase) const __qpu__ {
    apply_qsvt_signal_phase(signal, phase);
  }
};

/// @brief Apply a QSP Z-rotation-style signal phase.
/// @details QSPPACK and many QSP references use phases
/// diag(exp(i phi), exp(-i phi)) in the two-dimensional signal model. Up to an
/// unobservable global phase, this is equivalent to applying
/// diag(exp(2 i phi), 1). The implementation therefore reuses the projector
/// phase primitive with a doubled phase angle.
__qpu__ inline void apply_qsp_signal_phase(cudaq::qview<> signal,
                                           double phase) {
  apply_qsvt_signal_phase(signal, 2.0 * phase);
}

/// @brief Kernel functor wrapper for a QSP signal phase.
struct qsp_signal_phase {
  void operator()(cudaq::qview<> signal, double phase) const __qpu__ {
    apply_qsp_signal_phase(signal, phase);
  }
};

/// @brief Apply a controlled phase to the all-zero signal/ancilla state.
/// @details Applies the signal phase only when @p control is in the |1> state.
__qpu__ inline void apply_controlled_qsvt_signal_phase(cudaq::qubit &control,
                                                       cudaq::qview<> signal,
                                                       double phase) {
  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);

  std::size_t num_signal = signal.size();
  if (num_signal == 0) {
    return;
  } else if (num_signal == 1) {
    r1<cudaq::ctrl>(phase, control, signal[0]);
  } else if (num_signal == 2) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1]);
  } else if (num_signal == 3) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2]);
  } else if (num_signal == 4) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3]);
  } else if (num_signal == 5) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3],
                    signal[4]);
  } else if (num_signal == 6) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5]);
  } else if (num_signal == 7) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6]);
  } else if (num_signal == 8) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7]);
  } else if (num_signal == 9) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7], signal[8]);
  } else if (num_signal == 10) {
    r1<cudaq::ctrl>(phase, control, signal[0], signal[1], signal[2], signal[3],
                    signal[4], signal[5], signal[6], signal[7], signal[8],
                    signal[9]);
  }

  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);
}

/// @brief Kernel functor wrapper for a controlled QSVT signal phase.
struct controlled_qsvt_signal_phase {
  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  double phase) const __qpu__ {
    apply_controlled_qsvt_signal_phase(control, signal, phase);
  }
};

/// @brief Apply a controlled QSP Z-rotation-style signal phase.
/// @details See apply_qsp_signal_phase for the phase-convention relation.
__qpu__ inline void apply_controlled_qsp_signal_phase(cudaq::qubit &control,
                                                      cudaq::qview<> signal,
                                                      double phase) {
  apply_controlled_qsvt_signal_phase(control, signal, 2.0 * phase);
}

/// @brief Kernel functor wrapper for a controlled QSP signal phase.
struct controlled_qsp_signal_phase {
  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  double phase) const __qpu__ {
    apply_controlled_qsp_signal_phase(control, signal, phase);
  }
};

/// @brief Apply a QSVT-style phase/walk sequence for a Pauli LCU encoding.
/// @details Uses the convention phase[0], then repeats a qubitization walk
/// followed by phase[j] for j = 1..degree. This helper does not validate phases
/// in device code; callers should construct or verify a qsvt_phase_sequence on
/// the host before invoking it.
__qpu__ inline void apply_qsvt_sequence(cudaq::qview<> signal,
                                        cudaq::qview<> system,
                                        const pauli_lcu &encoding,
                                        const std::vector<double> &phases,
                                        qsvt_walk_direction direction) {
  if (phases.empty())
    return;

  apply_qsvt_signal_phase(signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (direction == qsvt_walk_direction::forward)
      apply_qubitization_walk(signal, system, encoding);
    else
      apply_adjoint_qubitization_walk(signal, system, encoding);
    apply_qsvt_signal_phase(signal, phases[i]);
  }
}

/// @brief Apply a QSVT-style phase/walk sequence with per-step directions.
/// @details walk_directions contains one primitive direction code per walk,
/// so its host-validated size should be phases.size() - 1. This QPU helper
/// intentionally does not validate policy shape in device code.
__qpu__ inline void
apply_qsvt_sequence(cudaq::qview<> signal, cudaq::qview<> system,
                    const pauli_lcu &encoding,
                    const std::vector<double> &phases,
                    const std::vector<int> &walk_directions) {
  if (phases.empty())
    return;

  apply_qsvt_signal_phase(signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (walk_directions[i - 1] == qsvt_adjoint_walk)
      apply_adjoint_qubitization_walk(signal, system, encoding);
    else
      apply_qubitization_walk(signal, system, encoding);
    apply_qsvt_signal_phase(signal, phases[i]);
  }
}

/// @brief Apply a QSVT-style phase/walk sequence with forward walks.
__qpu__ inline void apply_qsvt_sequence(cudaq::qview<> signal,
                                        cudaq::qview<> system,
                                        const pauli_lcu &encoding,
                                        const std::vector<double> &phases) {
  apply_qsvt_sequence(signal, system, encoding, phases,
                      qsvt_walk_direction::forward);
}

/// @brief Apply a QSP-style phase/walk sequence for a Pauli LCU encoding.
/// @details This matches the QSP phase convention used by QSPPACK: each signal
/// phase is diag(exp(i phi), exp(-i phi)) in the abstract signal model. The
/// qubitization walk convention and phase/walk ordering match
/// apply_qsvt_sequence.
__qpu__ inline void apply_qsp_sequence(cudaq::qview<> signal,
                                       cudaq::qview<> system,
                                       const pauli_lcu &encoding,
                                       const std::vector<double> &phases,
                                       qsvt_walk_direction direction) {
  if (phases.empty())
    return;

  apply_qsp_signal_phase(signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (direction == qsvt_walk_direction::forward)
      apply_qubitization_walk(signal, system, encoding);
    else
      apply_adjoint_qubitization_walk(signal, system, encoding);
    apply_qsp_signal_phase(signal, phases[i]);
  }
}

/// @brief Apply a QSP-style phase/walk sequence with per-step directions.
__qpu__ inline void
apply_qsp_sequence(cudaq::qview<> signal, cudaq::qview<> system,
                   const pauli_lcu &encoding, const std::vector<double> &phases,
                   const std::vector<int> &walk_directions) {
  if (phases.empty())
    return;

  apply_qsp_signal_phase(signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (walk_directions[i - 1] == qsvt_adjoint_walk)
      apply_adjoint_qubitization_walk(signal, system, encoding);
    else
      apply_qubitization_walk(signal, system, encoding);
    apply_qsp_signal_phase(signal, phases[i]);
  }
}

/// @brief Apply a QSP-style phase/walk sequence with forward walks.
__qpu__ inline void apply_qsp_sequence(cudaq::qview<> signal,
                                       cudaq::qview<> system,
                                       const pauli_lcu &encoding,
                                       const std::vector<double> &phases) {
  apply_qsp_sequence(signal, system, encoding, phases,
                     qsvt_walk_direction::forward);
}

/// @brief Apply a controlled QSVT-style phase/walk sequence.
/// @details Uses the same phase/walk convention as apply_qsvt_sequence, with
/// both signal phases and qubitization walks controlled by @p control.
__qpu__ inline void
apply_controlled_qsvt_sequence(cudaq::qubit &control, cudaq::qview<> signal,
                               cudaq::qview<> system, const pauli_lcu &encoding,
                               const std::vector<double> &phases,
                               qsvt_walk_direction direction) {
  if (phases.empty())
    return;

  apply_controlled_qsvt_signal_phase(control, signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (direction == qsvt_walk_direction::forward)
      apply_controlled_qubitization_walk(control, signal, system, encoding);
    else
      apply_controlled_adjoint_qubitization_walk(control, signal, system,
                                                 encoding);
    apply_controlled_qsvt_signal_phase(control, signal, phases[i]);
  }
}

/// @brief Apply a controlled QSVT-style sequence with per-step directions.
__qpu__ inline void
apply_controlled_qsvt_sequence(cudaq::qubit &control, cudaq::qview<> signal,
                               cudaq::qview<> system, const pauli_lcu &encoding,
                               const std::vector<double> &phases,
                               const std::vector<int> &walk_directions) {
  if (phases.empty())
    return;

  apply_controlled_qsvt_signal_phase(control, signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (walk_directions[i - 1] == qsvt_adjoint_walk)
      apply_controlled_adjoint_qubitization_walk(control, signal, system,
                                                 encoding);
    else
      apply_controlled_qubitization_walk(control, signal, system, encoding);
    apply_controlled_qsvt_signal_phase(control, signal, phases[i]);
  }
}

/// @brief Apply a controlled QSVT-style phase/walk sequence with forward walks.
__qpu__ inline void
apply_controlled_qsvt_sequence(cudaq::qubit &control, cudaq::qview<> signal,
                               cudaq::qview<> system, const pauli_lcu &encoding,
                               const std::vector<double> &phases) {
  apply_controlled_qsvt_sequence(control, signal, system, encoding, phases,
                                 qsvt_walk_direction::forward);
}

/// @brief Apply a controlled QSP-style phase/walk sequence.
__qpu__ inline void
apply_controlled_qsp_sequence(cudaq::qubit &control, cudaq::qview<> signal,
                              cudaq::qview<> system, const pauli_lcu &encoding,
                              const std::vector<double> &phases,
                              qsvt_walk_direction direction) {
  if (phases.empty())
    return;

  apply_controlled_qsp_signal_phase(control, signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (direction == qsvt_walk_direction::forward)
      apply_controlled_qubitization_walk(control, signal, system, encoding);
    else
      apply_controlled_adjoint_qubitization_walk(control, signal, system,
                                                 encoding);
    apply_controlled_qsp_signal_phase(control, signal, phases[i]);
  }
}

/// @brief Apply a controlled QSP-style sequence with per-step directions.
__qpu__ inline void
apply_controlled_qsp_sequence(cudaq::qubit &control, cudaq::qview<> signal,
                              cudaq::qview<> system, const pauli_lcu &encoding,
                              const std::vector<double> &phases,
                              const std::vector<int> &walk_directions) {
  if (phases.empty())
    return;

  apply_controlled_qsp_signal_phase(control, signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (walk_directions[i - 1] == qsvt_adjoint_walk)
      apply_controlled_adjoint_qubitization_walk(control, signal, system,
                                                 encoding);
    else
      apply_controlled_qubitization_walk(control, signal, system, encoding);
    apply_controlled_qsp_signal_phase(control, signal, phases[i]);
  }
}

/// @brief Apply a controlled QSP-style phase/walk sequence with forward walks.
__qpu__ inline void
apply_controlled_qsp_sequence(cudaq::qubit &control, cudaq::qview<> signal,
                              cudaq::qview<> system, const pauli_lcu &encoding,
                              const std::vector<double> &phases) {
  apply_controlled_qsp_sequence(control, signal, system, encoding, phases,
                                qsvt_walk_direction::forward);
}

/// @brief Kernel functor wrapper for controlled QSP-style sequences.
struct controlled_qsp_sequence {
  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  const std::vector<double> &phases) const __qpu__ {
    apply_controlled_qsp_sequence(control, signal, system, encoding, phases);
  }

  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  const std::vector<double> &phases,
                  qsvt_walk_direction direction) const __qpu__ {
    apply_controlled_qsp_sequence(control, signal, system, encoding, phases,
                                  direction);
  }

  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  const std::vector<double> &phases,
                  const std::vector<int> &walk_directions) const __qpu__ {
    apply_controlled_qsp_sequence(control, signal, system, encoding, phases,
                                  walk_directions);
  }
};

/// @brief Kernel functor wrapper for controlled QSVT-style sequences.
struct controlled_qsvt_sequence {
  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  const std::vector<double> &phases) const __qpu__ {
    apply_controlled_qsvt_sequence(control, signal, system, encoding, phases);
  }

  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  const std::vector<double> &phases,
                  qsvt_walk_direction direction) const __qpu__ {
    apply_controlled_qsvt_sequence(control, signal, system, encoding, phases,
                                   direction);
  }

  void operator()(cudaq::qubit &control, cudaq::qview<> signal,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  const std::vector<double> &phases,
                  const std::vector<int> &walk_directions) const __qpu__ {
    apply_controlled_qsvt_sequence(control, signal, system, encoding, phases,
                                   walk_directions);
  }
};

/// @brief Kernel functor wrapper for applying a QSVT-style phase/walk sequence.
struct qsvt_sequence {
  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding,
                  const std::vector<double> &phases) const __qpu__ {
    apply_qsvt_sequence(signal, system, encoding, phases);
  }

  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding, const std::vector<double> &phases,
                  qsvt_walk_direction direction) const __qpu__ {
    apply_qsvt_sequence(signal, system, encoding, phases, direction);
  }

  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding, const std::vector<double> &phases,
                  const std::vector<int> &walk_directions) const __qpu__ {
    apply_qsvt_sequence(signal, system, encoding, phases, walk_directions);
  }
};

/// @brief Kernel functor wrapper for applying a QSP-style phase/walk sequence.
struct qsp_sequence {
  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding,
                  const std::vector<double> &phases) const __qpu__ {
    apply_qsp_sequence(signal, system, encoding, phases);
  }

  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding, const std::vector<double> &phases,
                  qsvt_walk_direction direction) const __qpu__ {
    apply_qsp_sequence(signal, system, encoding, phases, direction);
  }

  void operator()(cudaq::qview<> signal, cudaq::qview<> system,
                  const pauli_lcu &encoding, const std::vector<double> &phases,
                  const std::vector<int> &walk_directions) const __qpu__ {
    apply_qsp_sequence(signal, system, encoding, phases, walk_directions);
  }
};

/// @brief Plain QPU-facing data extracted from a host-side QSVT plan.
/// @details This is a convenience view for host code. Kernels should still take
/// the individual phase and walk-direction vectors as separate arguments.
struct qsvt_kernel_data {
  const std::vector<double> &phases;
  const std::vector<int> &walk_directions;
};

/// @brief Host-side QSVT phase sequence.
///
/// QSVT and QSP routines are parameterized by a sequence of phase angles. A
/// degree-d polynomial conventionally uses d + 1 phases. This type keeps that
/// convention explicit and validates the sequence before it is consumed by
/// device-side QSVT kernels.
struct qsvt_phase_sequence {
  std::vector<double> phases;

  qsvt_phase_sequence() = default;
  explicit qsvt_phase_sequence(std::vector<double> input_phases);

  bool empty() const { return phases.empty(); }
  std::size_t size() const { return phases.size(); }
  std::size_t degree() const;
  const std::vector<double> &data() const { return phases; }
  double operator[](std::size_t index) const { return phases[index]; }
};

/// @brief Host-side QSVT walk sequence policy.
/// @details Stores one primitive walk direction code for each walk between
/// phases. A degree-d QSVT sequence has d walks and d + 1 phases. Kernels
/// consume walk_direction_data() rather than this host-side policy object.
struct qsvt_sequence_policy {
  std::vector<int> walk_directions;

  qsvt_sequence_policy() = default;
  explicit qsvt_sequence_policy(std::vector<int> input_walk_directions);

  bool empty() const { return walk_directions.empty(); }
  std::size_t size() const { return walk_directions.size(); }
  std::size_t degree() const { return walk_directions.size(); }
  const std::vector<int> &walk_direction_data() const {
    return walk_directions;
  }
  int operator[](std::size_t index) const { return walk_directions[index]; }
};

/// @brief Host-side metadata for a target QSVT matrix-function transform.
/// @details This describes the transform a future phase-generation routine
/// should approximate. It does not synthesize QSP/QSVT phases.
struct qsvt_transform_descriptor {
  qsvt_transform_kind kind = qsvt_transform_kind::custom;
  qsvt_phase_convention phase_convention = qsvt_phase_convention::qsvt;
  double evolution_time = 0.0;
  double condition_number = 0.0;
  double target_error = 0.0;
  double normalization = 1.0;
  std::size_t degree_hint = 0;
};

/// @brief Host-side plan for applying a QSVT phase/walk sequence.
///
/// The plan is intentionally a host-side object. CUDA-Q kernels should consume
/// plain data extracted from the plan, such as phase_data() and
/// walk_direction_data(), rather than taking qsvt_plan directly as a kernel
/// argument.
struct qsvt_plan {
  qsvt_phase_sequence phase_sequence;
  qsvt_sequence_policy sequence_policy;

  explicit qsvt_plan(qsvt_phase_sequence input_phases);
  qsvt_plan(qsvt_phase_sequence input_phases,
            qsvt_sequence_policy input_policy);
  explicit qsvt_plan(std::vector<double> input_phases);
  qsvt_plan(std::vector<double> input_phases,
            qsvt_sequence_policy input_policy);

  std::size_t num_phases() const { return phase_sequence.size(); }
  std::size_t degree() const { return phase_sequence.degree(); }
  const qsvt_phase_sequence &phases() const { return phase_sequence; }
  const qsvt_sequence_policy &policy() const { return sequence_policy; }
  const std::vector<double> &phase_data() const {
    return phase_sequence.data();
  }
  const std::vector<int> &walk_direction_data() const {
    return sequence_policy.walk_direction_data();
  }
  qsvt_kernel_data kernel_data() const {
    return {phase_data(), walk_direction_data()};
  }
};

/// @brief Host-side QSVT plan associated with a target matrix transform.
/// @details This retains transform metadata for bookkeeping while exposing the
/// same QPU-facing phase and walk-direction data as qsvt_plan. Kernels should
/// consume kernel_data(), phase_data(), or walk_direction_data(), not this
/// host-side object directly.
struct qsvt_transform_plan {
  qsvt_transform_descriptor transform_descriptor;
  qsvt_plan sequence_plan;

  qsvt_transform_plan(qsvt_transform_descriptor input_descriptor,
                      qsvt_plan input_plan);

  const qsvt_transform_descriptor &descriptor() const {
    return transform_descriptor;
  }
  const qsvt_plan &plan() const { return sequence_plan; }
  std::size_t num_phases() const { return sequence_plan.num_phases(); }
  std::size_t degree() const { return sequence_plan.degree(); }
  const std::vector<double> &phase_data() const {
    return sequence_plan.phase_data();
  }
  const std::vector<int> &walk_direction_data() const {
    return sequence_plan.walk_direction_data();
  }
  qsvt_kernel_data kernel_data() const { return sequence_plan.kernel_data(); }
};

/// @brief Return true if a QSVT phase sequence is non-empty and finite.
bool is_valid_qsvt_phase_sequence(const std::vector<double> &phases);

/// @brief Validate a QSVT phase sequence.
/// @throws std::invalid_argument if the sequence is empty or contains a
/// non-finite phase.
void validate_qsvt_phase_sequence(const std::vector<double> &phases);

/// @brief Return the polynomial degree represented by num_phases phases.
/// @details A degree-d QSVT polynomial is represented by d + 1 phases.
/// @throws std::invalid_argument if num_phases is zero.
std::size_t qsvt_polynomial_degree(std::size_t num_phases);

/// @brief Return the primitive code for a QSVT walk direction.
int qsvt_walk_direction_code(qsvt_walk_direction direction);

/// @brief Return true if the sequence policy matches a requested degree.
bool is_valid_qsvt_sequence_policy(std::size_t degree,
                                   const qsvt_sequence_policy &policy);

/// @brief Validate a QSVT sequence policy against a requested degree.
/// @throws std::invalid_argument if the policy length does not match degree or
/// contains an unknown primitive walk direction code.
void validate_qsvt_sequence_policy(std::size_t degree,
                                   const qsvt_sequence_policy &policy);

/// @brief Construct and validate a QSVT phase sequence.
qsvt_phase_sequence make_qsvt_phase_sequence(std::vector<double> phases);

/// @brief Construct a uniform QSVT walk sequence policy.
qsvt_sequence_policy make_qsvt_sequence_policy(
    std::size_t degree,
    qsvt_walk_direction direction = qsvt_walk_direction::forward);

/// @brief Construct a custom QSVT walk sequence policy.
qsvt_sequence_policy
make_qsvt_sequence_policy(std::vector<qsvt_walk_direction> directions);

/// @brief Construct an alternating QSVT walk sequence policy.
qsvt_sequence_policy make_alternating_qsvt_sequence_policy(
    std::size_t degree,
    qsvt_walk_direction first_direction = qsvt_walk_direction::forward);

/// @brief Construct a host-side QSVT plan from a phase sequence.
qsvt_plan make_qsvt_plan(std::vector<double> phases);

/// @brief Return true if the transform descriptor is finite and meaningful.
bool is_valid_qsvt_transform_descriptor(
    const qsvt_transform_descriptor &descriptor);

/// @brief Validate a transform descriptor.
/// @throws std::invalid_argument if metadata is not finite or is inconsistent
/// with the transform kind.
void validate_qsvt_transform_descriptor(
    const qsvt_transform_descriptor &descriptor);

/// @brief Evaluate the abstract QSVT/QSP scalar response for explicit phases.
/// @details This is a host-side diagnostic and validation helper. It evaluates
/// the upper-left matrix element of the two-dimensional signal model at
/// @p x in [-1, 1]. The qsvt convention uses projector phases
/// diag(exp(i phi), 1), matching apply_qsvt_signal_phase. The qsp convention
/// uses Z-rotation phases diag(exp(i phi), exp(-i phi)).
qsvt_response evaluate_qsvt_response(
    const std::vector<double> &phases, double x,
    qsvt_phase_convention convention = qsvt_phase_convention::qsvt);

/// @brief Evaluate the scalar response for a validated phase sequence.
qsvt_response evaluate_qsvt_response(
    const qsvt_phase_sequence &phases, double x,
    qsvt_phase_convention convention = qsvt_phase_convention::qsvt);

/// @brief Evaluate the scalar response for a QSVT sequence plan.
qsvt_response evaluate_qsvt_response(
    const qsvt_plan &plan, double x,
    qsvt_phase_convention convention = qsvt_phase_convention::qsvt);

/// @brief Evaluate the scalar response using a transform plan convention.
qsvt_response evaluate_qsvt_response(const qsvt_transform_plan &plan, double x);

/// @brief Estimate response error against a target function on sample points.
/// @details This is a host-side validation helper for user-supplied or future
/// generated phases. The target callable maps scalar x values in [-1, 1] to
/// the desired complex response at that point.
qsvt_response_error estimate_qsvt_response_error(
    const std::vector<double> &phases,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points,
    qsvt_phase_convention convention = qsvt_phase_convention::qsvt);

/// @brief Estimate response error for a validated phase sequence.
qsvt_response_error estimate_qsvt_response_error(
    const qsvt_phase_sequence &phases,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points,
    qsvt_phase_convention convention = qsvt_phase_convention::qsvt);

/// @brief Estimate response error for a QSVT sequence plan.
qsvt_response_error estimate_qsvt_response_error(
    const qsvt_plan &plan,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points,
    qsvt_phase_convention convention = qsvt_phase_convention::qsvt);

/// @brief Estimate response error using a transform plan convention.
qsvt_response_error estimate_qsvt_response_error(
    const qsvt_transform_plan &plan,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points);

/// @brief Construct uniformly spaced sample points over [min_x, max_x].
/// @details Multi-point grids include both endpoints. A single-point grid uses
/// the interval midpoint.
std::vector<double> make_uniform_qsvt_sample_points(double min_x, double max_x,
                                                    std::size_t num_points);

/// @brief Construct Chebyshev extrema sample points over [min_x, max_x].
/// @details Multi-point grids are ordered from min_x to max_x and include both
/// endpoints. A single-point grid uses the interval midpoint.
std::vector<double> make_chebyshev_qsvt_sample_points(double min_x,
                                                      double max_x,
                                                      std::size_t num_points);

/// @brief Validate explicit phases against a transform descriptor.
/// @details This validates descriptor metadata, phase finiteness, and the
/// descriptor degree hint when one is provided. It does not synthesize phases.
void validate_qsvt_transform_phase_sequence(
    const qsvt_transform_descriptor &descriptor,
    const std::vector<double> &phases);

/// @brief Construct a host-side QSVT plan from phases and a walk policy.
qsvt_plan make_qsvt_plan(std::vector<double> phases,
                         qsvt_sequence_policy policy);

/// @brief Construct a QSVT plan for a described transform from explicit phases.
/// @details The transform descriptor is host-side metadata for validation and
/// bookkeeping. Phase synthesis is intentionally out of scope; callers provide
/// an explicit phase sequence.
qsvt_transform_plan
make_qsvt_transform_plan(const qsvt_transform_descriptor &descriptor,
                         std::vector<double> phases);

/// @brief Construct a QSVT transform plan with an explicit walk policy.
qsvt_transform_plan
make_qsvt_transform_plan(const qsvt_transform_descriptor &descriptor,
                         std::vector<double> phases,
                         qsvt_sequence_policy policy);

/// @brief Describe the inverse transform used by QSVT-based linear solve.
qsvt_transform_descriptor
make_linear_solve_qsvt_transform(double condition_number, double target_error,
                                 std::size_t degree_hint = 0,
                                 double normalization = 1.0);

/// @brief Describe the exp(-i H t) transform for Hamiltonian simulation.
qsvt_transform_descriptor make_real_time_hamiltonian_simulation_qsvt_transform(
    double evolution_time, double target_error, std::size_t degree_hint = 0,
    double normalization = 1.0);

/// @brief Describe the exp(-H t) transform for imaginary-time evolution.
qsvt_transform_descriptor
make_imaginary_time_hamiltonian_simulation_qsvt_transform(
    double evolution_time, double target_error, std::size_t degree_hint = 0,
    double normalization = 1.0);

} // namespace cudaq::algorithms
