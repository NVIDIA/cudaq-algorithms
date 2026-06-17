/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/qubitization/qubitization.h"

namespace cudaq::algorithms {

cudaq::spin_op build_ancilla_zero_projector(std::size_t num_ancilla) {
  cudaq::spin_op projector = 0.5 * (cudaq::spin::i(0) + cudaq::spin::z(0));
  for (std::size_t q = 1; q < num_ancilla; ++q)
    projector *= 0.5 * (cudaq::spin::i(q) + cudaq::spin::z(q));

  return projector;
}

cudaq::spin_op
build_qubitization_reflection_observable(std::size_t num_ancilla) {
  auto projector = build_ancilla_zero_projector(num_ancilla);
  return 2.0 * projector - cudaq::spin::i(0);
}

cudaq::spin_op build_lcu_select_observable(const pauli_lcu &encoding) {
  std::size_t num_ancilla = encoding.num_ancilla();

  const auto &controls = encoding.get_term_controls();
  const auto &ops = encoding.get_term_ops();
  const auto &lengths = encoding.get_term_lengths();
  const auto &signs = encoding.get_term_signs();

  cudaq::spin_op select_observable;
  bool first_term = true;

  int ctrl_ptr = 0;
  int ops_ptr = 0;

  for (std::size_t term_idx = 0; term_idx < lengths.size(); ++term_idx) {
    cudaq::spin_op ancilla_projector;
    bool first_ancilla = true;

    for (std::size_t b = 0; b < num_ancilla; ++b) {
      int bit_val = controls[ctrl_ptr++];
      cudaq::spin_op projector_bit =
          (bit_val == 0) ? 0.5 * (cudaq::spin::i(b) + cudaq::spin::z(b))
                         : 0.5 * (cudaq::spin::i(b) - cudaq::spin::z(b));

      if (first_ancilla) {
        ancilla_projector = projector_bit;
        first_ancilla = false;
      } else {
        ancilla_projector = ancilla_projector * projector_bit;
      }
    }

    cudaq::spin_op system_pauli;
    bool first_pauli = true;

    int num_ops = lengths[term_idx];
    for (int k = 0; k < num_ops; ++k) {
      int code = ops[ops_ptr++];
      int qubit = ops[ops_ptr++] + num_ancilla;

      cudaq::spin_op pauli_op;
      if (code == 1)
        pauli_op = cudaq::spin::x(qubit);
      else if (code == 2)
        pauli_op = cudaq::spin::y(qubit);
      else if (code == 3)
        pauli_op = cudaq::spin::z(qubit);

      if (first_pauli) {
        system_pauli = pauli_op;
        first_pauli = false;
      } else {
        system_pauli = system_pauli * pauli_op;
      }
    }

    if (first_pauli)
      system_pauli = cudaq::spin::i(num_ancilla);

    if (first_ancilla)
      ancilla_projector = cudaq::spin::i(0);

    double sign = signs[term_idx];
    cudaq::spin_op term = sign * ancilla_projector * system_pauli;

    if (first_term) {
      select_observable = term;
      first_term = false;
    } else {
      select_observable = select_observable + term;
    }
  }

  return select_observable;
}

} // namespace cudaq::algorithms
