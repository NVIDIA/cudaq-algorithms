#pragma once

#include "cudaq/spin_op.h"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaq::algorithms::stateprep {

using uccgsd_double_excitation = std::pair<std::pair<std::size_t, std::size_t>,
                                           std::pair<std::size_t, std::size_t>>;

std::vector<std::pair<std::size_t, std::size_t>>
generate_uccgsd_singles(std::size_t num_qubits);

std::vector<uccgsd_double_excitation>
generate_uccgsd_doubles(std::size_t num_qubits);

void add_uccgsd_single_excitation(std::vector<cudaq::spin_op> &ops,
                                  std::size_t p, std::size_t q);

void add_uccgsd_double_excitation(std::vector<cudaq::spin_op> &ops,
                                  std::size_t p, std::size_t q, std::size_t r,
                                  std::size_t s);

std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_alpha_singles(std::size_t num_orbitals);

std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_beta_singles(std::size_t num_orbitals);

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_alpha_doubles(std::size_t num_orbitals);

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_beta_doubles(std::size_t num_orbitals);

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_mixed_doubles(std::size_t num_orbitals);

void add_ceo_single_excitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                               std::size_t q);

void add_ceo_double_excitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                               std::size_t q, std::size_t r, std::size_t s);

std::vector<cudaq::spin_op> make_uccsd_operator_pool(std::size_t num_qubits,
                                                     std::size_t num_electrons,
                                                     std::size_t spin = 0);

std::vector<cudaq::spin_op>
make_uccgsd_operator_pool(std::size_t num_qubits, bool only_singles = false,
                          bool only_doubles = false);

std::vector<cudaq::spin_op>
make_upccgsd_operator_pool(std::size_t num_spin_orbitals,
                           bool only_doubles = false);

std::vector<cudaq::spin_op> make_ceo_operator_pool(std::size_t num_orbitals);

} // namespace cudaq::algorithms::stateprep
