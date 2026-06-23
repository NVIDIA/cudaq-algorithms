#include "cudaq/algorithms/stateprep/upccgsd.h"
#include "cudaq/algorithms/stateprep/excitations.h"

namespace cudaq::algorithms::stateprep {

std::pair<std::vector<std::vector<cudaq::pauli_word>>,
          std::vector<std::vector<double>>>
get_upccgsd_pauli_lists(std::size_t norbitals, bool only_doubles) {
  auto ops = make_upccgsd_operator_pool(norbitals, only_doubles);

  std::vector<std::vector<cudaq::pauli_word>> pauli_words_list;
  std::vector<std::vector<double>> coefficients_list;
  for (const auto &op : ops) {
    std::vector<cudaq::pauli_word> words;
    std::vector<double> coefficients;
    for (const auto &term : op) {
      words.push_back(term.get_pauli_word(norbitals));
      coefficients.push_back(term.evaluate_coefficient().real());
    }
    pauli_words_list.push_back(std::move(words));
    coefficients_list.push_back(std::move(coefficients));
  }
  return {pauli_words_list, coefficients_list};
}

__qpu__ void
upccgsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
        const std::vector<std::vector<cudaq::pauli_word>> &pauli_words_list,
        const std::vector<std::vector<double>> &coefficients_list) {
  for (std::size_t i = 0; i < pauli_words_list.size(); ++i) {
    double theta = thetas[i];
    const auto &words = pauli_words_list[i];
    const auto &coefficients = coefficients_list[i];
    for (std::size_t j = 0; j < words.size(); ++j)
      exp_pauli(theta * coefficients[j], qubits, words[j]);
  }
}

} // namespace cudaq::algorithms::stateprep
