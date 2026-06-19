/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq.h"
#include "cudaq/qis/pauli_word.h"
#include <cstddef>
#include <vector>

namespace cudaq::algorithms {

///  Host-side data needed to build a Pauli LCU block encoding.
///
/// This type separates extraction and normalization of Hamiltonian terms from
/// QPU kernel generation. It intentionally does not impose an inheritance
/// hierarchy on block encodings; it is just the concrete data model consumed by
/// pauli_lcu.
struct lcu_decomposition {
  std::vector<double> coefficients;
  std::vector<double> absolute_coefficients;
  std::vector<double> probabilities;
  std::vector<int> signs;
  std::vector<int> identity_terms;
  std::vector<cudaq::pauli_word> pauli_words;
  std::size_t num_system_qubits = 0;
  std::size_t num_terms = 0;
  std::size_t padded_num_terms = 0;
  std::size_t num_ancilla_qubits = 0;
  double normalization = 0.0;

  /// @brief Sum of retained identity-only terms.
  ///
  /// By default pauli_lcu includes identity terms in the encoded operator so
  /// existing behavior is unchanged. When include_identity is false, identity
  /// terms are excluded from the encoded operator and are only reported here
  /// for algorithms that account for scalar shifts outside the LCU.
  double constant_term = 0.0;

  double coefficient_threshold = 1e-12;
  bool include_identity = true;
};

/// @brief Host-side metadata for a Pauli LCU block encoding.
/// @details This is the lightweight inspection view that downstream transform
/// descriptors and examples should use. It intentionally contains only scalar
/// metadata, not QPU kernel layout vectors.
struct pauli_lcu_metadata {
  std::size_t num_system_qubits = 0;
  std::size_t num_ancilla_qubits = 0;
  std::size_t num_terms = 0;
  std::size_t padded_num_terms = 0;
  double normalization = 0.0;
  double constant_term = 0.0;
  double coefficient_threshold = 0.0;
  bool include_identity = true;
};

/// @brief Host-side layout consumed by the Pauli LCU PREPARE/SELECT kernels.
struct pauli_lcu_kernel_data {
  std::vector<double> state_prep_angles;
  std::vector<int> term_controls;
  std::vector<int> term_ops;
  std::vector<int> term_lengths;
  std::vector<int> term_signs;
  std::size_t num_system_qubits = 0;
  std::size_t num_terms = 0;
  std::size_t padded_num_terms = 0;
  std::size_t num_ancilla_qubits = 0;
};

/// @brief Decompose a spin_op into host-side Pauli LCU data.
lcu_decomposition decompose_lcu(const cudaq::spin_op &hamiltonian,
                                std::size_t num_qubits,
                                double coefficient_threshold = 1e-12,
                                bool include_identity = true);

/// @brief Build the flattened Pauli LCU kernel layout from a decomposition.
pauli_lcu_kernel_data make_pauli_lcu_kernel_data(const lcu_decomposition &lcu);

/// @brief Block encoding implementation using Pauli LCU (Linear Combination of
/// Unitaries)
///
/// A block encoding represents a quantum subroutine that encodes a target
/// operator (typically a Hamiltonian) into a unitary operator acting on
/// an extended Hilbert space with ancilla qubits. This is a fundamental
/// building block for algorithms like Quantum Eigenvalue Learning (QEL),
/// Quantum Singular Value Transformation (QSVT), and Hamiltonian simulation.
///
/// The block encoding U satisfies: (⟨0|_anc ⊗ I_sys) U (|0⟩_anc ⊗ I_sys) = H/α
/// where α is the normalization constant and H is the target operator.
///
/// This implementation is optimized for Hamiltonians expressed as sums of Pauli
/// strings (e.g., molecular Hamiltonians from quantum chemistry). It uses:
/// - PREPARE: State preparation tree with controlled rotations
/// - SELECT: Controlled Pauli operations indexed by ancilla state
///
/// The encoding uses log₂(# terms) ancilla qubits and achieves α = ||H||₁.
class pauli_lcu {
private:
  // Flattened data for GPU kernels
  pauli_lcu_kernel_data kernel_data;

  // Metadata
  std::size_t n_anc; // Number of ancilla qubits
  std::size_t n_sys; // Number of system qubits
  double alpha;      // Normalization (1-norm)

  lcu_decomposition decomposition;

  /// @brief Compute rotation angles for state preparation tree
  /// @param probs Probability distribution (must be power of 2 length)
  /// @return Vector of rotation angles for each node in the tree
  static std::vector<double> compute_angles(const std::vector<double> &probs);

public:
  /// @brief Construct a Pauli LCU block encoding from a spin operator
  /// @param hamiltonian The target Hamiltonian as a spin_op
  /// @param num_qubits Number of system qubits (must match Hamiltonian support)
  explicit pauli_lcu(const cudaq::spin_op &hamiltonian, std::size_t num_qubits);
  /// @brief Construct a Pauli LCU block encoding from a spin operator.
  /// @param include_identity Whether identity terms are retained in H / alpha.
  explicit pauli_lcu(const cudaq::spin_op &hamiltonian, std::size_t num_qubits,
                     bool include_identity);

  ///  Construct a Pauli LCU block encoding from host-side LCU data
  explicit pauli_lcu(const lcu_decomposition &lcu);

  /// @brief Get the number of ancilla qubits
  /// @return Number of ancilla qubits
  std::size_t num_ancilla() const { return n_anc; }

  /// @brief Get the number of system qubits
  /// @return Number of system qubits
  std::size_t num_system() const { return n_sys; }

  /// @brief Get the normalization constant (1-norm of coefficients)
  /// @details The block encoding satisfies ||H|| ≤ α = ||H||₁
  /// @return Normalization constant (1-norm)
  double normalization() const { return alpha; }

  /// @brief Get scalar metadata for this block encoding.
  /// @details Downstream QSVT/QSP transform setup should use this host-side
  /// view to recover the encoded operator scale H / alpha and related layout
  /// metadata without depending on the flattened kernel representation.
  pauli_lcu_metadata metadata() const {
    return {n_sys,
            n_anc,
            decomposition.num_terms,
            decomposition.padded_num_terms,
            alpha,
            decomposition.constant_term,
            decomposition.coefficient_threshold,
            decomposition.include_identity};
  }

  ///  Get the constant identity component detected during decomposition
  double constant_term() const { return decomposition.constant_term; }

  ///  Return whether identity terms are retained in the encoded operator
  bool include_identity() const { return decomposition.include_identity; }

  ///  Get the number of retained LCU terms before padding
  std::size_t term_count() const { return decomposition.num_terms; }

  ///  Get the number of LCU leaves after power-of-two padding
  std::size_t padded_term_count() const {
    return decomposition.padded_num_terms;
  }

  /// @brief Apply PREPARE: encode coefficients into ancilla superposition
  /// @details Prepares a superposition state on the ancilla qubits that
  /// encodes the coefficients of the Hamiltonian terms
  /// @param ancilla View of ancilla qubits
  void prepare(cudaq::qview<> ancilla) const;

  /// @brief Apply PREPARE†: uncompute the ancilla superposition
  /// @details Adjoint of the PREPARE operation
  /// @param ancilla View of ancilla qubits
  void unprepare(cudaq::qview<> ancilla) const;

  /// @brief Apply SELECT: controlled Pauli operations
  /// @details Applies the appropriate Hamiltonian term conditioned on the
  /// ancilla register state
  /// @param ancilla View of ancilla qubits (control register)
  /// @param system View of system qubits (target register)
  void select(cudaq::qview<> ancilla, cudaq::qview<> system) const;

  /// @brief Apply SELECT controlled by an additional external qubit.
  /// @details Applies SELECT only when @p control is in the |1> state. The
  /// ancilla register still selects the Pauli LCU term.
  void controlled_select(cudaq::qubit &control, cudaq::qview<> ancilla,
                         cudaq::qview<> system) const;

  /// @brief Apply the full block encoding: PREPARE → SELECT → PREPARE†
  /// @details Applies the complete block encoding unitary operation
  /// @param ancilla View of ancilla qubits
  /// @param system View of system qubits
  void apply(cudaq::qview<> ancilla, cudaq::qview<> system) const {
    prepare(ancilla);
    select(ancilla, system);
    unprepare(ancilla);
  }

  /// @brief Get the state preparation angles (for debugging/testing)
  const std::vector<double> &get_angles() const {
    return kernel_data.state_prep_angles;
  }

  /// @brief Get the flattened term controls (for debugging/testing)
  const std::vector<int> &get_term_controls() const {
    return kernel_data.term_controls;
  }

  /// @brief Get the flattened term ops (for debugging/testing)
  const std::vector<int> &get_term_ops() const { return kernel_data.term_ops; }

  /// @brief Get the term lengths (for debugging/testing)
  const std::vector<int> &get_term_lengths() const {
    return kernel_data.term_lengths;
  }

  /// @brief Get the term signs (for debugging/testing)
  const std::vector<int> &get_term_signs() const {
    return kernel_data.term_signs;
  }

  /// @brief Get the host-side LCU decomposition (for inspection/testing)
  const lcu_decomposition &get_decomposition() const { return decomposition; }

  /// @brief Get the flattened Pauli LCU kernel layout (for inspection/testing)
  const pauli_lcu_kernel_data &get_kernel_data() const { return kernel_data; }
};

} // namespace cudaq::algorithms
