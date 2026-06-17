/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/algorithms/block_encoding/pauli_lcu.h"
#include <cstddef>

namespace cudaq::algorithms {

/// @brief Pauli LCU qubitization primitive conventions.
///
/// These helpers use snake_case names and expose concrete primitives rather
/// than an inheritance hierarchy. For a Pauli LCU encoding, the walk primitive
/// applies SELECT first and then reflects about the PREPARE state. In operator
/// order this is W = R_prepare SELECT, matching the circuit order:
///
///   encoding.select(ancilla, system);
///   reflect_about_prepare(ancilla, encoding);
///
/// The adjoint walk uses the opposite circuit order, reflection followed by
/// SELECT, since SELECT and the PREPARE-state reflection are self-adjoint. The
/// walk helpers do not prepare the ancilla register. Callers should invoke
/// encoding.prepare(ancilla) before the first walk step when the algorithm
/// requires the PREPARE state as the starting point.

/// @brief Reflect about the all-zero state on an ancilla register.
/// @details Applies X on all ancillas, a multi-controlled Z, then X again.
/// This stays inline because it is used inside generated CUDA-Q kernels.
__qpu__ inline void reflect_about_zero(cudaq::qview<> ancilla) {
  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);

  std::size_t num_ancilla = ancilla.size();
  if (num_ancilla == 0) {
    return;
  } else if (num_ancilla == 1) {
    z(ancilla[0]);
  } else if (num_ancilla == 2) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1]);
  } else if (num_ancilla == 3) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2]);
  } else if (num_ancilla == 4) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3]);
  } else if (num_ancilla == 5) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3], ancilla[4]);
  } else if (num_ancilla == 6) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3], ancilla[4],
                   ancilla[5]);
  } else if (num_ancilla == 7) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3], ancilla[4],
                   ancilla[5], ancilla[6]);
  } else if (num_ancilla == 8) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3], ancilla[4],
                   ancilla[5], ancilla[6], ancilla[7]);
  } else if (num_ancilla == 9) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3], ancilla[4],
                   ancilla[5], ancilla[6], ancilla[7], ancilla[8]);
  } else if (num_ancilla == 10) {
    z<cudaq::ctrl>(ancilla[0], ancilla[1], ancilla[2], ancilla[3], ancilla[4],
                   ancilla[5], ancilla[6], ancilla[7], ancilla[8], ancilla[9]);
  }

  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);
}

/// @brief Reflect about the all-zero ancilla state controlled by another qubit.
/// @details Applies the zero-state reflection only when @p control is |1>.
__qpu__ inline void controlled_reflect_about_zero(cudaq::qubit &control,
                                                  cudaq::qview<> ancilla) {
  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);

  std::size_t num_ancilla = ancilla.size();
  if (num_ancilla == 0) {
    return;
  } else if (num_ancilla == 1) {
    z<cudaq::ctrl>(control, ancilla[0]);
  } else if (num_ancilla == 2) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1]);
  } else if (num_ancilla == 3) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2]);
  } else if (num_ancilla == 4) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3]);
  } else if (num_ancilla == 5) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                   ancilla[4]);
  } else if (num_ancilla == 6) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                   ancilla[4], ancilla[5]);
  } else if (num_ancilla == 7) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                   ancilla[4], ancilla[5], ancilla[6]);
  } else if (num_ancilla == 8) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                   ancilla[4], ancilla[5], ancilla[6], ancilla[7]);
  } else if (num_ancilla == 9) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                   ancilla[4], ancilla[5], ancilla[6], ancilla[7], ancilla[8]);
  } else if (num_ancilla == 10) {
    z<cudaq::ctrl>(control, ancilla[0], ancilla[1], ancilla[2], ancilla[3],
                   ancilla[4], ancilla[5], ancilla[6], ancilla[7], ancilla[8],
                   ancilla[9]);
  }

  for (std::size_t i = 0; i < ancilla.size(); ++i)
    x(ancilla[i]);
}

/// @brief Reflect about the state prepared by a Pauli LCU PREPARE circuit.
/// @details Implements PREPARE dagger, zero-state reflection, PREPARE.
__qpu__ inline void reflect_about_prepare(cudaq::qview<> ancilla,
                                          const pauli_lcu &encoding) {
  encoding.unprepare(ancilla);
  reflect_about_zero(ancilla);
  encoding.prepare(ancilla);
}

/// @brief Kernel functor wrapper for reflect_about_prepare.
/// @details This makes the primitive convenient to pass into existing CUDA-Q
/// kernel call sites that use functor-style kernels.
struct prepare_reflection {
  void operator()(cudaq::qview<> ancilla,
                  const pauli_lcu &encoding) const __qpu__ {
    reflect_about_prepare(ancilla, encoding);
  }
};

/// @brief Reflect about the PREPARE state controlled by another qubit.
/// @details Implements PREPARE dagger, controlled zero-state reflection,
/// PREPARE. When the control is |0>, the uncontrolled PREPARE operations
/// cancel.
__qpu__ inline void
controlled_reflect_about_prepare(cudaq::qubit &control, cudaq::qview<> ancilla,
                                 const pauli_lcu &encoding) {
  encoding.unprepare(ancilla);
  controlled_reflect_about_zero(control, ancilla);
  encoding.prepare(ancilla);
}

/// @brief Kernel functor wrapper for controlled_reflect_about_prepare.
struct controlled_prepare_reflection {
  void operator()(cudaq::qubit &control, cudaq::qview<> ancilla,
                  const pauli_lcu &encoding) const __qpu__ {
    controlled_reflect_about_prepare(control, ancilla, encoding);
  }
};

/// @brief Apply one qubitization walk step for a Pauli LCU encoding.
/// @details Applies SELECT followed by reflection about the PREPARE state.
/// The caller is responsible for preparing the ancilla register before the
/// first walk step when the algorithm requires it.
__qpu__ inline void apply_qubitization_walk(cudaq::qview<> ancilla,
                                            cudaq::qview<> system,
                                            const pauli_lcu &encoding) {
  encoding.select(ancilla, system);
  reflect_about_prepare(ancilla, encoding);
}

/// @brief Kernel functor wrapper for one qubitization walk step.
struct qubitization_walk {
  void operator()(cudaq::qview<> ancilla, cudaq::qview<> system,
                  const pauli_lcu &encoding) const __qpu__ {
    apply_qubitization_walk(ancilla, system, encoding);
  }
};

/// @brief Apply one adjoint qubitization walk step for a Pauli LCU encoding.
/// @details Applies reflection about the PREPARE state followed by SELECT. With
/// the convention W = R_prepare SELECT, this circuit order implements the
/// adjoint walk W^{-1} = SELECT R_prepare.
__qpu__ inline void apply_adjoint_qubitization_walk(cudaq::qview<> ancilla,
                                                    cudaq::qview<> system,
                                                    const pauli_lcu &encoding) {
  reflect_about_prepare(ancilla, encoding);
  encoding.select(ancilla, system);
}

/// @brief Kernel functor wrapper for one adjoint qubitization walk step.
struct adjoint_qubitization_walk {
  void operator()(cudaq::qview<> ancilla, cudaq::qview<> system,
                  const pauli_lcu &encoding) const __qpu__ {
    apply_adjoint_qubitization_walk(ancilla, system, encoding);
  }
};

/// @brief Apply one externally controlled qubitization walk step.
/// @details Applies controlled SELECT followed by controlled reflection about
/// the PREPARE state. The caller is responsible for preparing the ancilla
/// register before the first controlled walk step when needed.
__qpu__ inline void apply_controlled_qubitization_walk(
    cudaq::qubit &control, cudaq::qview<> ancilla, cudaq::qview<> system,
    const pauli_lcu &encoding) {
  encoding.controlled_select(control, ancilla, system);
  controlled_reflect_about_prepare(control, ancilla, encoding);
}

/// @brief Kernel functor wrapper for one controlled qubitization walk step.
struct controlled_qubitization_walk {
  void operator()(cudaq::qubit &control, cudaq::qview<> ancilla,
                  cudaq::qview<> system,
                  const pauli_lcu &encoding) const __qpu__ {
    apply_controlled_qubitization_walk(control, ancilla, system, encoding);
  }
};

/// @brief Apply one externally controlled adjoint qubitization walk step.
/// @details Applies controlled reflection about PREPARE followed by controlled
/// SELECT, matching the adjoint walk ordering.
__qpu__ inline void apply_controlled_adjoint_qubitization_walk(
    cudaq::qubit &control, cudaq::qview<> ancilla, cudaq::qview<> system,
    const pauli_lcu &encoding) {
  controlled_reflect_about_prepare(control, ancilla, encoding);
  encoding.controlled_select(control, ancilla, system);
}

/// @brief Kernel functor wrapper for one controlled adjoint walk step.
struct controlled_adjoint_qubitization_walk {
  void operator()(cudaq::qubit &control, cudaq::qview<> ancilla,
                  cudaq::qview<> system,
                  const pauli_lcu &encoding) const __qpu__ {
    apply_controlled_adjoint_qubitization_walk(control, ancilla, system,
                                               encoding);
  }
};

/// @brief Apply repeated controlled qubitization walk steps.
/// @details Applies the controlled walk primitive power times. The caller is
/// responsible for preparing the ancilla register before the first controlled
/// walk step when needed.
__qpu__ inline void apply_controlled_qubitization_walk_power(
    cudaq::qubit &control, cudaq::qview<> ancilla, cudaq::qview<> system,
    const pauli_lcu &encoding, int power) {
  for (int i = 0; i < power; ++i)
    apply_controlled_qubitization_walk(control, ancilla, system, encoding);
}

/// @brief Kernel functor wrapper for repeated controlled walk steps.
struct controlled_qubitization_walk_power {
  void operator()(cudaq::qubit &control, cudaq::qview<> ancilla,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  int power) const __qpu__ {
    apply_controlled_qubitization_walk_power(control, ancilla, system, encoding,
                                             power);
  }
};

/// @brief Apply repeated controlled adjoint qubitization walk steps.
/// @details Applies the controlled adjoint walk primitive power times. The
/// caller is responsible for preparing the ancilla register before the first
/// controlled walk step when needed.
__qpu__ inline void apply_controlled_adjoint_qubitization_walk_power(
    cudaq::qubit &control, cudaq::qview<> ancilla, cudaq::qview<> system,
    const pauli_lcu &encoding, int power) {
  for (int i = 0; i < power; ++i)
    apply_controlled_adjoint_qubitization_walk(control, ancilla, system,
                                               encoding);
}

/// @brief Kernel functor wrapper for repeated controlled adjoint walk steps.
struct controlled_adjoint_qubitization_walk_power {
  void operator()(cudaq::qubit &control, cudaq::qview<> ancilla,
                  cudaq::qview<> system, const pauli_lcu &encoding,
                  int power) const __qpu__ {
    apply_controlled_adjoint_qubitization_walk_power(control, ancilla, system,
                                                     encoding, power);
  }
};

/// @brief Apply repeated qubitization walk steps for a Pauli LCU encoding.
/// @details Applies the walk primitive power times. The caller is responsible
/// for preparing the ancilla register before the first walk step when needed.
__qpu__ inline void apply_qubitization_walk_power(cudaq::qview<> ancilla,
                                                  cudaq::qview<> system,
                                                  const pauli_lcu &encoding,
                                                  int power) {
  for (int i = 0; i < power; ++i)
    apply_qubitization_walk(ancilla, system, encoding);
}

/// @brief Kernel functor wrapper for repeated qubitization walk steps.
struct qubitization_walk_power {
  void operator()(cudaq::qview<> ancilla, cudaq::qview<> system,
                  const pauli_lcu &encoding, int power) const __qpu__ {
    apply_qubitization_walk_power(ancilla, system, encoding, power);
  }
};

/// @brief Apply repeated adjoint qubitization walk steps.
/// @details Applies the adjoint walk primitive power times. The caller is
/// responsible for preparing the ancilla register before the first walk step
/// when needed.
__qpu__ inline void
apply_adjoint_qubitization_walk_power(cudaq::qview<> ancilla,
                                      cudaq::qview<> system,
                                      const pauli_lcu &encoding, int power) {
  for (int i = 0; i < power; ++i)
    apply_adjoint_qubitization_walk(ancilla, system, encoding);
}

/// @brief Kernel functor wrapper for repeated adjoint qubitization walk steps.
struct adjoint_qubitization_walk_power {
  void operator()(cudaq::qview<> ancilla, cudaq::qview<> system,
                  const pauli_lcu &encoding, int power) const __qpu__ {
    apply_adjoint_qubitization_walk_power(ancilla, system, encoding, power);
  }
};

/// @brief Build the projector |0><0| on an ancilla register.
/// @param num_ancilla Number of ancilla qubits in the projector register.
cudaq::spin_op build_ancilla_zero_projector(std::size_t num_ancilla);

/// @brief Build the reflection observable R = 2|0><0| - I.
/// @param num_ancilla Number of ancilla qubits in the projector register.
cudaq::spin_op
build_qubitization_reflection_observable(std::size_t num_ancilla);

/// @brief Build the observable corresponding to the Pauli LCU SELECT operator.
/// @details The observable is expressed over the combined ancilla-system
/// register, with system qubit indices offset by encoding.num_ancilla(). It is
/// used by QEL odd-moment estimation and is a reusable inspection/measurement
/// primitive for Pauli LCU encodings.
cudaq::spin_op build_lcu_select_observable(const pauli_lcu &encoding);

} // namespace cudaq::algorithms
