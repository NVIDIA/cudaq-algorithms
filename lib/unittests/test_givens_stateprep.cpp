/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithm.h"
#include "cudaq/algorithms/stateprep/givens.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

namespace stateprep = cudaq::algorithms::stateprep;

TEST(GivensStatePrep, BuildsTwoOrbitalSchedule) {
  const double theta = 0.37;
  const std::vector<std::vector<double>> occupied_orbitals = {
      {std::cos(theta)}, {std::sin(theta)}};

  auto schedule = stateprep::make_givens_rotation_schedule(occupied_orbitals);

  ASSERT_EQ(schedule.num_orbitals, 2);
  ASSERT_EQ(schedule.num_electrons, 1);
  ASSERT_EQ(schedule.rotations.size(), 1);
  EXPECT_EQ(schedule.rotations[0].first_orbital, 0);
  EXPECT_EQ(schedule.rotations[0].second_orbital, 1);
  EXPECT_NEAR(schedule.rotations[0].theta, theta, 1.0e-12);

  auto indices = stateprep::get_givens_rotation_indices(schedule);
  auto angles = stateprep::get_givens_rotation_angles(schedule);
  ASSERT_EQ(indices, (std::vector<std::size_t>{0, 1}));
  ASSERT_EQ(angles.size(), 1);
  EXPECT_NEAR(angles[0], theta, 1.0e-12);
}

TEST(GivensStatePrep, BuildsComplexTwoOrbitalSchedule) {
  const double theta = 0.37;
  const double phase = std::acos(-1.0) / 2.0;
  const std::complex<double> imaginary = {0.0, 1.0};
  const std::vector<std::vector<std::complex<double>>> occupied_orbitals = {
      {std::cos(theta)}, {imaginary * std::sin(theta)}};

  auto schedule = stateprep::make_givens_rotation_schedule(occupied_orbitals);

  ASSERT_EQ(schedule.rotations.size(), 1);
  EXPECT_EQ(schedule.rotations[0].first_orbital, 0);
  EXPECT_EQ(schedule.rotations[0].second_orbital, 1);
  EXPECT_NEAR(schedule.rotations[0].theta, theta, 1.0e-12);
  EXPECT_NEAR(schedule.rotations[0].phase, phase, 1.0e-12);
  ASSERT_EQ(schedule.final_phases.size(), 1);
  EXPECT_NEAR(schedule.final_phases[0], 0.0, 1.0e-12);
}

TEST(GivensStatePrep, RejectsNonOrthonormalInputs) {
  const std::vector<std::vector<double>> occupied_orbitals = {{1.0, 0.0},
                                                              {1.0, 0.0}};

  EXPECT_THROW(stateprep::make_givens_rotation_schedule(occupied_orbitals),
               std::invalid_argument);
}

namespace {

using complex = std::complex<double>;

std::size_t count_bits(std::size_t value) {
  std::size_t count = 0;
  while (value != 0) {
    count += value & 1;
    value >>= 1;
  }
  return count;
}

complex determinant(std::vector<std::vector<complex>> matrix) {
  const auto size = matrix.size();
  complex result = 1.0;

  for (std::size_t col = 0; col < size; ++col) {
    std::size_t pivot = col;
    for (std::size_t row = col + 1; row < size; ++row)
      if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col]))
        pivot = row;

    if (std::abs(matrix[pivot][col]) < 1.0e-14)
      return 0.0;

    if (pivot != col) {
      std::swap(matrix[pivot], matrix[col]);
      result *= -1.0;
    }

    const auto pivot_value = matrix[col][col];
    result *= pivot_value;
    for (std::size_t row = col + 1; row < size; ++row) {
      const auto factor = matrix[row][col] / pivot_value;
      for (std::size_t entry = col + 1; entry < size; ++entry)
        matrix[row][entry] -= factor * matrix[col][entry];
    }
  }

  return result;
}

std::vector<complex> reference_slater_state(
    const std::vector<std::vector<complex>> &occupied_orbitals) {
  const auto num_orbitals = occupied_orbitals.size();
  const auto num_electrons = occupied_orbitals.front().size();
  std::vector<complex> state(1ULL << num_orbitals, 0.0);

  for (std::size_t basis = 0; basis < state.size(); ++basis) {
    if (count_bits(basis) != num_electrons)
      continue;

    std::vector<std::size_t> occupied;
    for (std::size_t orbital = 0; orbital < num_orbitals; ++orbital)
      if ((basis >> orbital) & 1ULL)
        occupied.push_back(orbital);

    std::vector<std::vector<complex>> minor(
        num_electrons, std::vector<complex>(num_electrons));
    for (std::size_t row = 0; row < num_electrons; ++row)
      for (std::size_t col = 0; col < num_electrons; ++col)
        minor[row][col] = occupied_orbitals[occupied[row]][col];

    state[basis] = determinant(minor);
  }

  return state;
}

std::size_t reverse_bits(std::size_t value, std::size_t num_bits) {
  std::size_t reversed = 0;
  for (std::size_t bit = 0; bit < num_bits; ++bit)
    if ((value >> bit) & 1ULL)
      reversed |= 1ULL << (num_bits - bit - 1);
  return reversed;
}

void expect_state_close_up_to_global_phase(const cudaq::state &actual,
                                           const std::vector<complex> &expected,
                                           double tolerance) {
  std::size_t num_qubits = 0;
  while ((1ULL << num_qubits) < expected.size())
    ++num_qubits;

  std::size_t pivot = 0;
  for (std::size_t i = 1; i < expected.size(); ++i)
    if (std::abs(expected[i]) > std::abs(expected[pivot]))
      pivot = i;

  complex phase = 1.0;
  if (std::abs(expected[pivot]) > tolerance) {
    phase = actual[reverse_bits(pivot, num_qubits)] / expected[pivot];
    phase /= std::abs(phase);
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    // The determinant helper labels orbital i as bit i; cudaq::state indexes
    // amplitudes with the opposite bit significance in these C++ tests.
    const auto actual_index = reverse_bits(i, num_qubits);
    const auto aligned = phase * expected[i];
    EXPECT_NEAR(actual[actual_index].real(), aligned.real(), tolerance)
        << "basis " << i;
    EXPECT_NEAR(actual[actual_index].imag(), aligned.imag(), tolerance)
        << "basis " << i;
  }
}

std::vector<std::vector<complex>>
make_complex_orthonormal_matrix(std::size_t num_orbitals,
                                std::size_t num_electrons) {
  std::vector<std::vector<complex>> matrix(num_orbitals,
                                           std::vector<complex>(num_electrons));

  for (std::size_t row = 0; row < num_orbitals; ++row)
    for (std::size_t col = 0; col < num_electrons; ++col)
      matrix[row][col] = {std::sin((row + 1) * (col + 2) * 0.37),
                          std::cos((row + 2) * (col + 1) * 0.21)};

  for (std::size_t col = 0; col < num_electrons; ++col) {
    for (std::size_t prev = 0; prev < col; ++prev) {
      complex overlap = 0.0;
      for (std::size_t row = 0; row < num_orbitals; ++row)
        overlap += std::conj(matrix[row][prev]) * matrix[row][col];
      for (std::size_t row = 0; row < num_orbitals; ++row)
        matrix[row][col] -= overlap * matrix[row][prev];
    }

    double norm = 0.0;
    for (std::size_t row = 0; row < num_orbitals; ++row)
      norm += std::norm(matrix[row][col]);
    norm = std::sqrt(norm);
    for (std::size_t row = 0; row < num_orbitals; ++row)
      matrix[row][col] /= norm;
  }

  return matrix;
}

struct real_slater_stateprep_kernel {
  void operator()(std::vector<std::size_t> orbital_indices,
                  std::vector<double> angles) __qpu__ {
    cudaq::qvector qubits(4);
    cudaq::algorithms::stateprep::prepare_slater_determinant(
        qubits, orbital_indices, angles, 2);
  }
};

struct complex_slater_stateprep_kernel {
  void operator()(std::vector<std::size_t> orbital_indices,
                  std::vector<double> angles, std::vector<double> phases,
                  std::vector<double> final_phases) __qpu__ {
    cudaq::qvector qubits(6);
    cudaq::algorithms::stateprep::prepare_complex_slater_determinant(
        qubits, orbital_indices, angles, phases, final_phases, 3);
  }
};

} // namespace

TEST(GivensStatePrep, BuildsRealSlaterDeterminantPlan) {
  const double theta0 = 0.31;
  const double theta1 = 0.47;
  const std::vector<std::vector<double>> occupied_orbitals = {
      {std::cos(theta0), 0.0},
      {0.0, std::cos(theta1)},
      {std::sin(theta0), 0.0},
      {0.0, std::sin(theta1)}};

  auto plan = stateprep::make_slater_determinant_plan(occupied_orbitals);

  EXPECT_EQ(plan.num_orbitals, 4);
  EXPECT_EQ(plan.num_electrons, 2);
  EXPECT_FALSE(plan.is_complex);
  EXPECT_EQ(plan.orbital_indices.size(), 2 * plan.angles.size());
  EXPECT_EQ(plan.phases.size(), plan.angles.size());
  EXPECT_EQ(plan.final_phases.size(), plan.num_electrons);

  auto resources = stateprep::estimate_givens_stateprep_resources(plan);
  EXPECT_EQ(resources.num_givens_rotations, plan.angles.size());
  EXPECT_EQ(resources.num_exp_pauli_calls, 2 * plan.angles.size());
  EXPECT_EQ(resources.num_phase_rotations, 0);

  auto actual = cudaq::get_state(real_slater_stateprep_kernel{},
                                 plan.orbital_indices, plan.angles);
  std::vector<std::vector<complex>> complex_orbitals;
  for (const auto &row : occupied_orbitals)
    complex_orbitals.push_back({row[0], row[1]});
  expect_state_close_up_to_global_phase(
      actual, reference_slater_state(complex_orbitals), 1.0e-6);
}

TEST(GivensStatePrep, BuildsComplexSlaterDeterminantPlan) {
  auto occupied_orbitals = make_complex_orthonormal_matrix(6, 3);
  auto plan = stateprep::make_slater_determinant_plan(occupied_orbitals);

  EXPECT_EQ(plan.num_orbitals, 6);
  EXPECT_EQ(plan.num_electrons, 3);
  EXPECT_TRUE(plan.is_complex);
  EXPECT_EQ(plan.orbital_indices.size(), 2 * plan.angles.size());
  EXPECT_EQ(plan.phases.size(), plan.angles.size());
  EXPECT_EQ(plan.final_phases.size(), plan.num_electrons);

  auto resources = stateprep::estimate_givens_stateprep_resources(plan);
  EXPECT_EQ(resources.num_givens_rotations, plan.angles.size());
  EXPECT_EQ(resources.num_exp_pauli_calls, 2 * plan.angles.size());
  EXPECT_EQ(resources.num_phase_rotations,
            plan.angles.size() + plan.num_electrons);

  auto actual =
      cudaq::get_state(complex_slater_stateprep_kernel{}, plan.orbital_indices,
                       plan.angles, plan.phases, plan.final_phases);
  expect_state_close_up_to_global_phase(
      actual, reference_slater_state(occupied_orbitals), 1.0e-6);
}

TEST(GivensStatePrep, RejectsMalformedSlaterDeterminantPlan) {
  stateprep::slater_determinant_plan plan;
  plan.num_orbitals = 3;
  plan.num_electrons = 1;
  plan.orbital_indices = {0, 2};
  plan.angles.push_back(0.25);
  plan.phases.push_back(0.0);
  plan.final_phases.push_back(0.0);

  EXPECT_THROW(stateprep::validate_slater_determinant_plan(plan),
               std::invalid_argument);
}
