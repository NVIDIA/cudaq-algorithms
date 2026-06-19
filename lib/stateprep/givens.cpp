/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/stateprep/givens.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace cudaq_algorithms::stateprep {

namespace {

void validate_occupied_orbitals(
    const std::vector<std::vector<double>> &occupied_orbitals,
    double tolerance) {
  if (occupied_orbitals.empty())
    throw std::invalid_argument("occupied_orbitals must not be empty");

  const auto num_electrons = occupied_orbitals.front().size();
  if (num_electrons == 0)
    throw std::invalid_argument(
        "occupied_orbitals must contain at least one occupied orbital");

  if (num_electrons > occupied_orbitals.size())
    throw std::invalid_argument(
        "number of occupied orbitals cannot exceed number of spin orbitals");

  for (const auto &row : occupied_orbitals)
    if (row.size() != num_electrons)
      throw std::invalid_argument(
          "occupied_orbitals must be a rectangular matrix");

  for (std::size_t col = 0; col < num_electrons; ++col) {
    double norm = 0.0;
    for (const auto &row : occupied_orbitals)
      norm += row[col] * row[col];
    if (std::abs(norm - 1.0) > 100.0 * tolerance)
      throw std::invalid_argument(
          "occupied_orbitals columns must be normalized");

    for (std::size_t other = col + 1; other < num_electrons; ++other) {
      double overlap = 0.0;
      for (const auto &row : occupied_orbitals)
        overlap += row[col] * row[other];
      if (std::abs(overlap) > 100.0 * tolerance)
        throw std::invalid_argument(
            "occupied_orbitals columns must be orthogonal");
    }
  }
}

void validate_occupied_orbitals(
    const std::vector<std::vector<std::complex<double>>> &occupied_orbitals,
    double tolerance) {
  if (occupied_orbitals.empty())
    throw std::invalid_argument("occupied_orbitals must not be empty");

  const auto num_electrons = occupied_orbitals.front().size();
  if (num_electrons == 0)
    throw std::invalid_argument(
        "occupied_orbitals must contain at least one occupied orbital");

  if (num_electrons > occupied_orbitals.size())
    throw std::invalid_argument(
        "number of occupied orbitals cannot exceed number of spin orbitals");

  for (const auto &row : occupied_orbitals)
    if (row.size() != num_electrons)
      throw std::invalid_argument(
          "occupied_orbitals must be a rectangular matrix");

  for (std::size_t col = 0; col < num_electrons; ++col) {
    double norm = 0.0;
    for (const auto &row : occupied_orbitals)
      norm += std::norm(row[col]);
    if (std::abs(norm - 1.0) > 100.0 * tolerance)
      throw std::invalid_argument(
          "occupied_orbitals columns must be normalized");

    for (std::size_t other = col + 1; other < num_electrons; ++other) {
      std::complex<double> overlap = 0.0;
      for (const auto &row : occupied_orbitals)
        overlap += std::conj(row[col]) * row[other];
      if (std::abs(overlap) > 100.0 * tolerance)
        throw std::invalid_argument(
            "occupied_orbitals columns must be orthogonal");
    }
  }
}

double argument_or_zero(const std::complex<double> &value, double tolerance) {
  if (std::abs(value) <= tolerance)
    return 0.0;
  return std::arg(value);
}

} // namespace

givens_rotation_schedule make_givens_rotation_schedule(
    const std::vector<std::vector<double>> &occupied_orbitals,
    double tolerance) {
  validate_occupied_orbitals(occupied_orbitals, tolerance);

  const auto num_orbitals = occupied_orbitals.size();
  const auto num_electrons = occupied_orbitals.front().size();
  auto work = occupied_orbitals;
  std::vector<givens_rotation> elimination_rotations;

  for (std::size_t col = 0; col < num_electrons; ++col) {
    for (std::size_t row = num_orbitals - 1; row > col; --row) {
      const auto upper_row = row - 1;
      const double upper = work[upper_row][col];
      const double lower = work[row][col];

      if (std::abs(lower) <= tolerance)
        continue;

      const double radius = std::hypot(upper, lower);
      if (radius <= tolerance)
        throw std::runtime_error("failed to construct Givens rotation");

      const double cosine = upper / radius;
      const double sine = lower / radius;
      const double theta = std::atan2(sine, cosine);

      for (std::size_t k = 0; k < num_electrons; ++k) {
        const double upper_value = work[upper_row][k];
        const double lower_value = work[row][k];
        work[upper_row][k] = cosine * upper_value + sine * lower_value;
        work[row][k] = -sine * upper_value + cosine * lower_value;
      }

      elimination_rotations.push_back({upper_row, row, theta});
    }
  }

  givens_rotation_schedule schedule;
  schedule.num_orbitals = num_orbitals;
  schedule.num_electrons = num_electrons;
  schedule.final_phases.assign(num_electrons, 0.0);
  schedule.rotations.reserve(elimination_rotations.size());

  // State preparation applies the inverse of the row rotations that reduce the
  // occupied-orbital matrix to the computational-basis determinant.
  for (auto iter = elimination_rotations.rbegin();
       iter != elimination_rotations.rend(); ++iter)
    schedule.rotations.push_back(*iter);

  return schedule;
}

givens_rotation_schedule make_givens_rotation_schedule(
    const std::vector<std::vector<std::complex<double>>> &occupied_orbitals,
    double tolerance) {
  validate_occupied_orbitals(occupied_orbitals, tolerance);

  const auto num_orbitals = occupied_orbitals.size();
  const auto num_electrons = occupied_orbitals.front().size();
  auto work = occupied_orbitals;
  std::vector<givens_rotation> elimination_rotations;

  for (std::size_t col = 0; col < num_electrons; ++col) {
    for (std::size_t row = num_orbitals - 1; row > col; --row) {
      const auto upper_row = row - 1;
      const auto upper = work[upper_row][col];
      const auto lower = work[row][col];

      if (std::abs(lower) <= tolerance)
        continue;

      const double upper_magnitude = std::abs(upper);
      const double lower_magnitude = std::abs(lower);
      const double radius = std::hypot(upper_magnitude, lower_magnitude);
      if (radius <= tolerance)
        throw std::runtime_error("failed to construct Givens rotation");

      const double cosine = upper_magnitude / radius;
      const double sine = lower_magnitude / radius;
      const double theta = std::atan2(sine, cosine);
      const double phase = argument_or_zero(lower, tolerance) -
                           argument_or_zero(upper, tolerance);
      const auto lower_phase =
          std::exp(std::complex<double>{0.0, -phase});

      for (std::size_t k = 0; k < num_electrons; ++k) {
        const auto upper_value = work[upper_row][k];
        const auto lower_value = lower_phase * work[row][k];
        work[upper_row][k] = cosine * upper_value + sine * lower_value;
        work[row][k] = -sine * upper_value + cosine * lower_value;
      }

      elimination_rotations.push_back({upper_row, row, theta, phase});
    }
  }

  givens_rotation_schedule schedule;
  schedule.num_orbitals = num_orbitals;
  schedule.num_electrons = num_electrons;
  schedule.final_phases.reserve(num_electrons);
  for (std::size_t col = 0; col < num_electrons; ++col)
    schedule.final_phases.push_back(
        argument_or_zero(work[col][col], tolerance));

  schedule.rotations.reserve(elimination_rotations.size());
  for (auto iter = elimination_rotations.rbegin();
       iter != elimination_rotations.rend(); ++iter)
    schedule.rotations.push_back(*iter);

  return schedule;
}

std::vector<std::size_t>
get_givens_rotation_indices(const givens_rotation_schedule &schedule) {
  std::vector<std::size_t> indices;
  indices.reserve(2 * schedule.rotations.size());
  for (const auto &rotation : schedule.rotations) {
    indices.push_back(rotation.first_orbital);
    indices.push_back(rotation.second_orbital);
  }
  return indices;
}

std::vector<double>
get_givens_rotation_angles(const givens_rotation_schedule &schedule) {
  std::vector<double> angles;
  angles.reserve(schedule.rotations.size());
  for (const auto &rotation : schedule.rotations)
    angles.push_back(rotation.theta);
  return angles;
}

std::vector<double>
get_givens_rotation_phases(const givens_rotation_schedule &schedule) {
  std::vector<double> phases;
  phases.reserve(schedule.rotations.size());
  for (const auto &rotation : schedule.rotations)
    phases.push_back(rotation.phase);
  return phases;
}

} // namespace cudaq_algorithms::stateprep
