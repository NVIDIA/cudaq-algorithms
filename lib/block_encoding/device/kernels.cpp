/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under  *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "cudaq/algorithms/block_encoding/kernels.h"
#include "cudaq/algorithms/detail/qpu_dispatch.h"

namespace cudaq_algorithms::block_encoding {

__qpu__ void prepare(cudaq::qview<> ancilla,
                     const std::vector<double> &state_prep_angles) {
  if (ancilla.size() == 0)
    return;

  ry(state_prep_angles[0], ancilla[0]);

  int angle_idx = 1;
  for (std::size_t layer = 1; layer < ancilla.size(); ++layer) {
    int num_branches = 1 << static_cast<int>(layer);

    for (int i = 0; i < num_branches; ++i) {
      for (int bit = 0; bit < static_cast<int>(layer); ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }

      CUDAQ_ALGORITHMS_APPLY_CONTROLLED_RY_BY_ARITY(
          state_prep_angles[angle_idx], ancilla, layer);
      angle_idx++;

      for (int bit = 0; bit < static_cast<int>(layer); ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }
    }
  }
}

__qpu__ void unprepare(cudaq::qview<> ancilla,
                       const std::vector<double> &state_prep_angles) {
  if (ancilla.size() == 0)
    return;

  int n_ancilla = static_cast<int>(ancilla.size());
  int angle_idx = static_cast<int>(state_prep_angles.size()) - 1;

  for (int layer = n_ancilla - 1; layer >= 1; --layer) {
    int num_branches = 1 << layer;

    for (int i = num_branches - 1; i >= 0; --i) {
      for (int bit = 0; bit < layer; ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }

      CUDAQ_ALGORITHMS_APPLY_CONTROLLED_RY_BY_ARITY(
          -state_prep_angles[angle_idx], ancilla, layer);
      angle_idx--;

      for (int bit = 0; bit < layer; ++bit) {
        if (!((i >> bit) & 1))
          x(ancilla[layer - 1 - bit]);
      }
    }
  }

  ry(-state_prep_angles[0], ancilla[0]);
}

__qpu__ void controlled_pauli_x(cudaq::qubit &control, cudaq::qview<> ancilla,
                                cudaq::qubit &target) {
  int n_ancilla = ancilla.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(x, control, ancilla, target,
                                                  n_ancilla);
}

__qpu__ void controlled_pauli_y(cudaq::qubit &control, cudaq::qview<> ancilla,
                                cudaq::qubit &target) {
  int n_ancilla = ancilla.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(y, control, ancilla, target,
                                                  n_ancilla);
}

__qpu__ void controlled_pauli_z(cudaq::qubit &control, cudaq::qview<> ancilla,
                                cudaq::qubit &target) {
  int n_ancilla = ancilla.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(z, control, ancilla, target,
                                                  n_ancilla);
}

__qpu__ void select(cudaq::qview<> ancilla, cudaq::qview<> system,
                    const std::vector<int> &term_controls,
                    const std::vector<int> &term_ops,
                    const std::vector<int> &term_lengths,
                    const std::vector<int> &term_signs) {
  int ptr_ctrl = 0;
  int ptr_op = 0;
  int n_ancilla = ancilla.size();

  for (std::size_t i = 0; i < term_lengths.size(); ++i) {
    int n_ops = term_lengths[i];
    int sign = term_signs[i];

    for (int b = 0; b < n_ancilla; ++b) {
      int bit_val = term_controls[ptr_ctrl++];
      if (bit_val == 0)
        x(ancilla[b]);
    }

    for (int k = 0; k < n_ops; ++k) {
      int code = term_ops[ptr_op++];
      int q_idx = term_ops[ptr_op++];

      if (code == 1)
        x<cudaq::ctrl>(ancilla, system[q_idx]);
      else if (code == 2)
        y<cudaq::ctrl>(ancilla, system[q_idx]);
      else if (code == 3)
        z<cudaq::ctrl>(ancilla, system[q_idx]);
    }

    if (sign < 0) {
      if (n_ancilla == 0)
        // Single-term LCU: with no ancilla there is no projected block, so
        // dropping the -1 would encode -H instead of H. rz(2*pi) is exactly
        // the -I matrix, so the sign also survives control synthesis.
        rz(2.0 * M_PI, system[0]);
      else
        CUDAQ_ALGORITHMS_APPLY_Z_BY_ARITY(ancilla, n_ancilla);
    }

    int back_ptr = ptr_ctrl - 1;
    for (int b_rev = 0; b_rev < n_ancilla; ++b_rev) {
      int anc_idx = (n_ancilla - 1) - b_rev;
      int bit_val = term_controls[back_ptr--];
      if (bit_val == 0)
        x(ancilla[anc_idx]);
    }
  }
}

__qpu__ void controlled_select(cudaq::qubit &control, cudaq::qview<> ancilla,
                               cudaq::qview<> system,
                               const std::vector<int> &term_controls,
                               const std::vector<int> &term_ops,
                               const std::vector<int> &term_lengths,
                               const std::vector<int> &term_signs) {
  int ptr_ctrl = 0;
  int ptr_op = 0;
  int n_ancilla = ancilla.size();

  for (std::size_t i = 0; i < term_lengths.size(); ++i) {
    int n_ops = term_lengths[i];
    int sign = term_signs[i];

    for (int b = 0; b < n_ancilla; ++b) {
      int bit_val = term_controls[ptr_ctrl++];
      if (bit_val == 0)
        x(ancilla[b]);
    }

    for (int k = 0; k < n_ops; ++k) {
      int code = term_ops[ptr_op++];
      int q_idx = term_ops[ptr_op++];

      if (code == 1)
        controlled_pauli_x(control, ancilla, system[q_idx]);
      else if (code == 2)
        controlled_pauli_y(control, ancilla, system[q_idx]);
      else if (code == 3)
        controlled_pauli_z(control, ancilla, system[q_idx]);
    }

    if (sign < 0)
      CUDAQ_ALGORITHMS_APPLY_CONTROLLED_Z_BY_ARITY(control, ancilla, n_ancilla);

    int back_ptr = ptr_ctrl - 1;
    for (int b_rev = 0; b_rev < n_ancilla; ++b_rev) {
      int anc_idx = (n_ancilla - 1) - b_rev;
      int bit_val = term_controls[back_ptr--];
      if (bit_val == 0)
        x(ancilla[anc_idx]);
    }
  }
}

__qpu__ void apply(cudaq::qview<> ancilla, cudaq::qview<> system,
                   const std::vector<double> &state_prep_angles,
                   const std::vector<int> &term_controls,
                   const std::vector<int> &term_ops,
                   const std::vector<int> &term_lengths,
                   const std::vector<int> &term_signs) {
  cudaq_algorithms::block_encoding::prepare(ancilla, state_prep_angles);
  cudaq_algorithms::block_encoding::select(ancilla, system, term_controls,
                                           term_ops, term_lengths, term_signs);
  cudaq_algorithms::block_encoding::unprepare(ancilla, state_prep_angles);
}

} // namespace cudaq_algorithms::block_encoding

namespace cudaq_algorithms::qubitization {

__qpu__ void reflect_about_zero(cudaq::qview<> ancilla) {
  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);

  std::size_t num_ancilla = ancilla.size();
  CUDAQ_ALGORITHMS_APPLY_Z_BY_ARITY(ancilla, num_ancilla);

  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);
}

__qpu__ void controlled_reflect_about_zero(cudaq::qubit &control,
                                           cudaq::qview<> ancilla) {
  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);

  std::size_t num_ancilla = ancilla.size();
  CUDAQ_ALGORITHMS_APPLY_CONTROLLED_Z_BY_ARITY(control, ancilla, num_ancilla);

  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);
}

__qpu__ void
reflect_about_prepare(cudaq::qview<> ancilla,
                      const std::vector<double> &state_prep_angles) {
  block_encoding::unprepare(ancilla, state_prep_angles);
  cudaq_algorithms::qubitization::reflect_about_zero(ancilla);
  block_encoding::prepare(ancilla, state_prep_angles);
}

__qpu__ void
controlled_reflect_about_prepare(cudaq::qubit &control, cudaq::qview<> ancilla,
                                 const std::vector<double> &state_prep_angles) {
  block_encoding::unprepare(ancilla, state_prep_angles);
  cudaq_algorithms::qubitization::controlled_reflect_about_zero(control,
                                                                ancilla);
  block_encoding::prepare(ancilla, state_prep_angles);
}

__qpu__ void apply_walk(cudaq::qview<> ancilla, cudaq::qview<> system,
                        const std::vector<double> &state_prep_angles,
                        const std::vector<int> &term_controls,
                        const std::vector<int> &term_ops,
                        const std::vector<int> &term_lengths,
                        const std::vector<int> &term_signs) {
  block_encoding::select(ancilla, system, term_controls, term_ops, term_lengths,
                         term_signs);
  reflect_about_prepare(ancilla, state_prep_angles);
}

__qpu__ void apply_adjoint_walk(cudaq::qview<> ancilla, cudaq::qview<> system,
                                const std::vector<double> &state_prep_angles,
                                const std::vector<int> &term_controls,
                                const std::vector<int> &term_ops,
                                const std::vector<int> &term_lengths,
                                const std::vector<int> &term_signs) {
  reflect_about_prepare(ancilla, state_prep_angles);
  block_encoding::select(ancilla, system, term_controls, term_ops, term_lengths,
                         term_signs);
}

__qpu__ void controlled_apply_walk(
    cudaq::qubit &control, cudaq::qview<> ancilla, cudaq::qview<> system,
    const std::vector<double> &state_prep_angles,
    const std::vector<int> &term_controls, const std::vector<int> &term_ops,
    const std::vector<int> &term_lengths, const std::vector<int> &term_signs) {
  block_encoding::controlled_select(control, ancilla, system, term_controls,
                                    term_ops, term_lengths, term_signs);
  controlled_reflect_about_prepare(control, ancilla, state_prep_angles);
}

__qpu__ void controlled_apply_adjoint_walk(
    cudaq::qubit &control, cudaq::qview<> ancilla, cudaq::qview<> system,
    const std::vector<double> &state_prep_angles,
    const std::vector<int> &term_controls, const std::vector<int> &term_ops,
    const std::vector<int> &term_lengths, const std::vector<int> &term_signs) {
  controlled_reflect_about_prepare(control, ancilla, state_prep_angles);
  block_encoding::controlled_select(control, ancilla, system, term_controls,
                                    term_ops, term_lengths, term_signs);
}

__qpu__ void apply_walk_power(cudaq::qview<> ancilla, cudaq::qview<> system,
                              const std::vector<double> &state_prep_angles,
                              const std::vector<int> &term_controls,
                              const std::vector<int> &term_ops,
                              const std::vector<int> &term_lengths,
                              const std::vector<int> &term_signs, int power) {
  for (int i = 0; i < power; ++i)
    cudaq_algorithms::qubitization::apply_walk(
        ancilla, system, state_prep_angles, term_controls, term_ops,
        term_lengths, term_signs);
}

__qpu__ void
apply_adjoint_walk_power(cudaq::qview<> ancilla, cudaq::qview<> system,
                         const std::vector<double> &state_prep_angles,
                         const std::vector<int> &term_controls,
                         const std::vector<int> &term_ops,
                         const std::vector<int> &term_lengths,
                         const std::vector<int> &term_signs, int power) {
  for (int i = 0; i < power; ++i)
    cudaq_algorithms::qubitization::apply_adjoint_walk(
        ancilla, system, state_prep_angles, term_controls, term_ops,
        term_lengths, term_signs);
}

} // namespace cudaq_algorithms::qubitization

namespace cudaq_algorithms::qsvt_primitives {

__qpu__ void apply_signal_phase(cudaq::qview<> signal, double phase) {
  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);

  std::size_t num_signal = signal.size();
  CUDAQ_ALGORITHMS_APPLY_R1_BY_ARITY(phase, signal, num_signal);

  for (std::size_t i = 0; i < signal.size(); ++i)
    x(signal[i]);
}

__qpu__ void apply_qsp_signal_phase(cudaq::qview<> signal, double phase) {
  cudaq_algorithms::qsvt_primitives::apply_signal_phase(signal, 2.0 * phase);
}

} // namespace cudaq_algorithms::qsvt_primitives

namespace cudaq_algorithms::qsvt {

__qpu__ void apply_signal_phase(cudaq::qview<> signal, double phase) {
  qsvt_primitives::apply_signal_phase(signal, phase);
}

__qpu__ void apply_phase_sequence(cudaq::qview<> signal, cudaq::qview<> system,
                                  const std::vector<double> &phases,
                                  const std::vector<int> &walk_directions,
                                  const std::vector<double> &state_prep_angles,
                                  const std::vector<int> &term_controls,
                                  const std::vector<int> &term_ops,
                                  const std::vector<int> &term_lengths,
                                  const std::vector<int> &term_signs) {
  if (phases.empty())
    return;

  cudaq_algorithms::qsvt_primitives::apply_signal_phase(signal, phases[0]);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    if (walk_directions[i - 1] == 1) {
      qubitization::reflect_about_zero(signal);
      block_encoding::apply(signal, system, state_prep_angles, term_controls,
                            term_ops, term_lengths, term_signs);
    } else {
      block_encoding::apply(signal, system, state_prep_angles, term_controls,
                            term_ops, term_lengths, term_signs);
      qubitization::reflect_about_zero(signal);
    }
    cudaq_algorithms::qsvt_primitives::apply_signal_phase(signal, phases[i]);
  }
}

} // namespace cudaq_algorithms::qsvt
