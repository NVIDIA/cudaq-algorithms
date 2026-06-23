#include "cudaq/algorithms/stateprep/excitations.h"
#include "cudaq/algorithms/stateprep/uccsd.h"

#include <algorithm>
#include <array>
#include <complex>
#include <set>
#include <stdexcept>

namespace cudaq::algorithms::stateprep {

std::vector<std::pair<std::size_t, std::size_t>>
generate_uccgsd_singles(std::size_t numQubits) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  singles.reserve(numQubits * (numQubits - 1) / 2);
  for (std::size_t p = 1; p < numQubits; ++p)
    for (std::size_t q = 0; q < p; ++q)
      singles.emplace_back(p, q);
  return singles;
}

std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                      std::pair<std::size_t, std::size_t>>>
generate_uccgsd_doubles(std::size_t numQubits) {
  std::set<std::pair<std::pair<std::size_t, std::size_t>,
                     std::pair<std::size_t, std::size_t>>>
      doubles;

  // Iterate over all combinations of 4 distinct qubits
  for (std::size_t a = 0; a < numQubits; ++a)
    for (std::size_t b = a + 1; b < numQubits; ++b)
      for (std::size_t c = b + 1; c < numQubits; ++c)
        for (std::size_t d = c + 1; d < numQubits; ++d) {
          std::array<std::size_t, 4> arr = {a, b, c, d};

          // Generate all 3 unique pairings of the 4 indices
          std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                                std::pair<std::size_t, std::size_t>>>
              pairings = {{{arr[0], arr[1]}, {arr[2], arr[3]}},
                          {{arr[0], arr[2]}, {arr[1], arr[3]}},
                          {{arr[0], arr[3]}, {arr[1], arr[2]}}};

          // Normalize and deduplicate each pairing
          for (auto &pairing : pairings) {
            auto p1 = pairing.first, p2 = pairing.second;

            // Ensure within each pair: first > second
            if (p1.first < p1.second)
              std::swap(p1.first, p1.second);
            if (p2.first < p2.second)
              std::swap(p2.first, p2.second);

            // Order the two pairs
            auto sorted_pairing = std::minmax(p1, p2);
            doubles.insert({sorted_pairing.first, sorted_pairing.second});
          }
        }

  return std::vector<std::pair<std::pair<std::size_t, std::size_t>,
                               std::pair<std::size_t, std::size_t>>>(
      doubles.begin(), doubles.end());
}

void add_uccgsd_single_excitation(std::vector<cudaq::spin_op> &ops,
                                  std::size_t p, std::size_t q) {
  if (p > q) {
    // Compute parity string (Z operators between q and p)
    cudaq::spin_op_term parity;
    for (std::size_t i = q + 1; i < p; ++i)
      parity *= cudaq::spin::z(i);

    std::complex<double> c = {0.5, 0.0};

    // Single excitation: Y_q * Z_parity * X_p - X_q * Z_parity * Y_p
    ops.emplace_back(c * cudaq::spin::y(q) * parity * cudaq::spin::x(p) -
                     c * cudaq::spin::x(q) * parity * cudaq::spin::y(p));
  }
}

void add_uccgsd_double_excitation(std::vector<cudaq::spin_op> &ops,
                                  std::size_t p, std::size_t q, std::size_t r,
                                  std::size_t s) {
  if (p > q && r > s) {
    // Compute parity strings
    cudaq::spin_op_term parity_a, parity_b;
    for (std::size_t i = q + 1; i < p; ++i)
      parity_a *= cudaq::spin::z(i);
    for (std::size_t i = s + 1; i < r; ++i)
      parity_b *= cudaq::spin::z(i);

    std::complex<double> c = {0.125, 0.0};

    // Build the 8-term double excitation operator
    cudaq::spin_op temp_op;

    // Positive terms
    temp_op = c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) *
              cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
    temp_op += c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::x(q) * parity_a * cudaq::spin::x(p);
    temp_op += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
    temp_op += c * cudaq::spin::y(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);

    // Negative terms
    temp_op -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::x(p);
    temp_op -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::x(r) *
               cudaq::spin::x(q) * parity_a * cudaq::spin::y(p);
    temp_op -= c * cudaq::spin::x(s) * parity_b * cudaq::spin::y(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);
    temp_op -= c * cudaq::spin::y(s) * parity_b * cudaq::spin::x(r) *
               cudaq::spin::y(q) * parity_a * cudaq::spin::y(p);

    ops.emplace_back(temp_op);
  }
}

std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_alpha_singles(std::size_t numOrbitals) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  // Alpha spin orbitals are at even indices: 0, 2, 4, ..., 2*(numOrbitals-1)
  singles.reserve(numOrbitals * (numOrbitals - 1) / 2);
  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      std::size_t p = 2 * i; // Higher alpha index
      std::size_t q = 2 * j; // Lower alpha index
      singles.emplace_back(p, q);
    }
  }
  return singles;
}

std::vector<std::pair<std::size_t, std::size_t>>
generate_ceo_beta_singles(std::size_t numOrbitals) {
  std::vector<std::pair<std::size_t, std::size_t>> singles;
  // Beta spin orbitals are at odd indices: 1, 3, 5, ..., 2*numOrbitals-1
  singles.reserve(numOrbitals * (numOrbitals - 1) / 2);
  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      std::size_t p = 2 * i + 1; // Higher beta index
      std::size_t q = 2 * j + 1; // Lower beta index
      singles.emplace_back(p, q);
    }
  }
  return singles;
}

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_alpha_doubles(std::size_t numOrbitals) {
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
      doubles;
  // Alpha spin orbitals are at even indices
  // For p > q > r > s, generate three pairings:
  // (p,q)->(r,s), (p,r)->(q,s), (q,p)->(r,s)
  // For CEO, these correspond to different excitations (following
  // the paper conventions (https://arxiv.org/abs/2407.08696)).

  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        for (std::size_t l = 0; l < k; ++l) {
          // i > j > k > l in spatial orbital indices
          std::size_t p = 2 * i;
          std::size_t q = 2 * j;
          std::size_t r = 2 * k;
          std::size_t s = 2 * l;

          // Pairing 1: (p,q)->(r,s)
          doubles.emplace_back(p, q, r, s);
          // Pairing 2: (p,r)->(q,s)
          doubles.emplace_back(p, r, q, s);
          // Pairing 3: (q,p)->(r,s)
          doubles.emplace_back(q, p, r, s);
        }
      }
    }
  }
  return doubles;
}

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_beta_doubles(std::size_t numOrbitals) {
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
      doubles;
  // Beta spin orbitals are at odd indices
  // For p > q > r > s, generate three pairings:
  // (p,q)->(r,s), (p,r)->(q,s), (q,p)->(r,s)
  // For CEO, these correspond to different excitations (following
  // the paper conventions (https://arxiv.org/abs/2407.08696)).

  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        for (std::size_t l = 0; l < k; ++l) {
          // i > j > k > l in spatial orbital indices
          std::size_t p = 2 * i + 1;
          std::size_t q = 2 * j + 1;
          std::size_t r = 2 * k + 1;
          std::size_t s = 2 * l + 1;

          // Pairing 1: (p,q)->(r,s)
          doubles.emplace_back(p, q, r, s);
          // Pairing 2: (p,r)->(q,s)
          doubles.emplace_back(p, r, q, s);
          // Pairing 3: (q,p)->(r,s)
          doubles.emplace_back(q, p, r, s);
        }
      }
    }
  }
  return doubles;
}

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
generate_ceo_mixed_doubles(std::size_t numOrbitals) {
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
      doubles;
  // Mixed doubles: following the CEO convention
  // (https://arxiv.org/abs/2407.08696), the excitation operator for p,q,r,s
  // excites pair (p,q) to (r,s), so the spins should be (alpha, beta) ->
  // (alpha, beta), and we want p > r and q > s.

  for (std::size_t i = 0; i < numOrbitals; ++i) {
    for (std::size_t j = 0; j < numOrbitals; ++j) {
      for (std::size_t k = 0; k < i; ++k) {
        for (std::size_t l = 0; l < j; ++l) {
          std::size_t p = 2 * i;
          std::size_t q = 2 * j + 1;
          std::size_t r = 2 * k;
          std::size_t s = 2 * l + 1;
          doubles.emplace_back(p, q, r, s);
        }
      }
    }
  }
  return doubles;
}

void add_ceo_single_excitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                               std::size_t q) {
  // CEO single excitation: 0.5 * (Y_q X_p - X_q Y_p)
  // No Z parity string
  std::complex<double> c = {0.5, 0.0};

  ops.emplace_back(c * cudaq::spin::y(q) * cudaq::spin::x(p) -
                   c * cudaq::spin::x(q) * cudaq::spin::y(p));
}

void add_ceo_double_excitation(std::vector<cudaq::spin_op> &ops, std::size_t p,
                               std::size_t q, std::size_t r, std::size_t s) {
  // CEO double excitation generates TWO operators for indices (p, q, r, s)
  std::complex<double> c = {0.25, 0.0};

  // Operator A: 0.25 * (X_r X_p X_s Y_q - X_r X_p Y_s X_q + Y_r Y_p X_s Y_q -
  // Y_r Y_p Y_s X_q)
  cudaq::spin_op op_a;
  op_a = c * cudaq::spin::x(r) * cudaq::spin::x(p) * cudaq::spin::x(s) *
         cudaq::spin::y(q);
  op_a -= c * cudaq::spin::x(r) * cudaq::spin::x(p) * cudaq::spin::y(s) *
          cudaq::spin::x(q);
  op_a += c * cudaq::spin::y(r) * cudaq::spin::y(p) * cudaq::spin::x(s) *
          cudaq::spin::y(q);
  op_a -= c * cudaq::spin::y(r) * cudaq::spin::y(p) * cudaq::spin::y(s) *
          cudaq::spin::x(q);
  ops.emplace_back(op_a);

  // Operator B: 0.25 * (X_r Y_p X_s X_q + X_r Y_p Y_s Y_q - Y_r X_p X_s X_q -
  // Y_r X_p Y_s Y_q)
  cudaq::spin_op op_b;
  op_b = c * cudaq::spin::x(r) * cudaq::spin::y(p) * cudaq::spin::x(s) *
         cudaq::spin::x(q);
  op_b += c * cudaq::spin::x(r) * cudaq::spin::y(p) * cudaq::spin::y(s) *
          cudaq::spin::y(q);
  op_b -= c * cudaq::spin::y(r) * cudaq::spin::x(p) * cudaq::spin::x(s) *
          cudaq::spin::x(q);
  op_b -= c * cudaq::spin::y(r) * cudaq::spin::x(p) * cudaq::spin::y(s) *
          cudaq::spin::y(q);
  ops.emplace_back(op_b);
}

std::vector<cudaq::spin_op> make_uccsd_operator_pool(std::size_t num_qubits,
                                                     std::size_t num_electrons,
                                                     std::size_t spin) {
  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(num_qubits, num_electrons, spin);

  std::vector<cudaq::spin_op> ops;

  auto addSinglesExcitation = [num_qubits](std::vector<cudaq::spin_op> &ops,
                                           std::size_t p, std::size_t q) {
    double parity = 1.0;

    cudaq::spin_op_term o;
    for (std::size_t i = p + 1; i < q; i++)
      o *= cudaq::spin::z(i);
    std::complex<double> c = {0.5, 0};
    ops.emplace_back(c * cudaq::spin::y(p) * o * cudaq::spin::x(q) -
                     c * cudaq::spin::x(p) * o * cudaq::spin::y(q));
  };

  auto addDoublesExcitation = [num_qubits](std::vector<cudaq::spin_op> &ops,
                                           std::size_t p, std::size_t q,
                                           std::size_t r, std::size_t s) {
    cudaq::spin_op_term parity_a;
    cudaq::spin_op_term parity_b;
    std::size_t i_occ = 0, j_occ = 0, a_virt = 0, b_virt = 0;
    if (p < q && r < s) {
      i_occ = p;
      j_occ = q;
      a_virt = r;
      b_virt = s;
    }

    else if (p > q && r > s) {
      i_occ = q;
      j_occ = p;
      a_virt = s;
      b_virt = r;
    } else if (p < q && r > s) {
      i_occ = p;
      j_occ = q;
      a_virt = s;
      b_virt = r;
    } else if (p > q && r < s) {
      i_occ = q;
      j_occ = p;
      a_virt = r;
      b_virt = s;
    }
    for (std::size_t i = i_occ + 1; i < j_occ; i++)
      parity_a *= cudaq::spin::z(i);

    for (std::size_t i = a_virt + 1; i < b_virt; i++)
      parity_b *= cudaq::spin::z(i);

    cudaq::spin_op op_term_temp =
        cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
        cudaq::spin::x(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp += cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::x(b_virt);
    op_term_temp += cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp += cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp -= cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::x(a_virt) * parity_b * cudaq::spin::x(b_virt);
    op_term_temp -= cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                    cudaq::spin::x(a_virt) * parity_b * cudaq::spin::x(b_virt);
    op_term_temp -= cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::x(a_virt) * parity_b * cudaq::spin::y(b_virt);
    op_term_temp -= cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                    cudaq::spin::y(a_virt) * parity_b * cudaq::spin::x(b_virt);

    std::complex<double> c = {0.125, 0};
    ops.emplace_back(c * op_term_temp);
  };

  for (auto &sa : singlesAlpha)
    addSinglesExcitation(ops, sa[0], sa[1]);
  for (auto &sa : singlesBeta)
    addSinglesExcitation(ops, sa[0], sa[1]);

  for (auto &d : doublesMixed)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);
  for (auto &d : doublesAlpha)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);
  for (auto &d : doublesBeta)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);

  return ops;
}

std::vector<cudaq::spin_op> make_uccgsd_operator_pool(std::size_t num_qubits,
                                                      bool only_singles,
                                                      bool only_doubles) {
  std::vector<cudaq::spin_op> ops;
  if (!only_doubles)
    for (auto [p, q] : generate_uccgsd_singles(num_qubits))
      add_uccgsd_single_excitation(ops, p, q);

  if (!only_singles)
    for (auto [pq, rs] : generate_uccgsd_doubles(num_qubits))
      add_uccgsd_double_excitation(ops, pq.first, pq.second, rs.first,
                                   rs.second);
  return ops;
}

static std::vector<std::pair<std::size_t, std::size_t>>
generate_upccgsd_singles(std::size_t num_spin_orbitals) {
  auto all_singles = generate_uccgsd_singles(num_spin_orbitals);
  std::vector<std::pair<std::size_t, std::size_t>> filtered;
  for (auto [p, q] : all_singles)
    if ((p % 2) == (q % 2))
      filtered.emplace_back(p, q);
  return filtered;
}

static std::vector<uccgsd_double_excitation>
generate_upccgsd_doubles(std::size_t num_spin_orbitals) {
  if (num_spin_orbitals % 2 != 0)
    throw std::invalid_argument(
        "make_upccgsd_operator_pool expects an even number of spin orbitals.");

  const std::size_t num_spatial_orbitals = num_spin_orbitals / 2;
  std::vector<uccgsd_double_excitation> doubles;
  for (std::size_t p = 0; p < num_spatial_orbitals; ++p) {
    for (std::size_t q = p + 1; q < num_spatial_orbitals; ++q) {
      const std::size_t p_alpha = 2 * p;
      const std::size_t p_beta = 2 * p + 1;
      const std::size_t q_alpha = 2 * q;
      const std::size_t q_beta = 2 * q + 1;
      doubles.push_back({{q_beta, q_alpha}, {p_beta, p_alpha}});
    }
  }
  return doubles;
}

std::vector<cudaq::spin_op>
make_upccgsd_operator_pool(std::size_t num_spin_orbitals, bool only_doubles) {
  std::vector<cudaq::spin_op> ops;
  if (!only_doubles)
    for (auto [p, q] : generate_upccgsd_singles(num_spin_orbitals))
      add_uccgsd_single_excitation(ops, p, q);

  for (auto [pq, rs] : generate_upccgsd_doubles(num_spin_orbitals))
    add_uccgsd_double_excitation(ops, pq.first, pq.second, rs.first, rs.second);
  return ops;
}

std::vector<cudaq::spin_op> make_ceo_operator_pool(std::size_t num_orbitals) {
  std::vector<cudaq::spin_op> ops;
  for (auto [p, q] : generate_ceo_alpha_singles(num_orbitals))
    add_ceo_single_excitation(ops, p, q);
  for (auto [p, q] : generate_ceo_beta_singles(num_orbitals))
    add_ceo_single_excitation(ops, p, q);
  for (auto [p, q, r, s] : generate_ceo_alpha_doubles(num_orbitals))
    add_ceo_double_excitation(ops, p, q, r, s);
  for (auto [p, q, r, s] : generate_ceo_beta_doubles(num_orbitals))
    add_ceo_double_excitation(ops, p, q, r, s);
  for (auto [p, q, r, s] : generate_ceo_mixed_doubles(num_orbitals))
    add_ceo_double_excitation(ops, p, q, r, s);
  return ops;
}

} // namespace cudaq::algorithms::stateprep
