/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/qsvt/qsvt.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>

namespace cudaq::algorithms {

namespace {

using complex = std::complex<double>;

struct qsvt_response_matrix {
  complex a00 = 1.0;
  complex a01 = 0.0;
  complex a10 = 0.0;
  complex a11 = 1.0;
};

qsvt_response_matrix multiply(const qsvt_response_matrix &lhs,
                              const qsvt_response_matrix &rhs) {
  return {lhs.a00 * rhs.a00 + lhs.a01 * rhs.a10,
          lhs.a00 * rhs.a01 + lhs.a01 * rhs.a11,
          lhs.a10 * rhs.a00 + lhs.a11 * rhs.a10,
          lhs.a10 * rhs.a01 + lhs.a11 * rhs.a11};
}

qsvt_response_matrix phase_response_matrix(double phase,
                                           qsvt_phase_convention convention) {
  const complex positive_phase = std::polar(1.0, phase);
  if (convention == qsvt_phase_convention::qsp)
    return {positive_phase, 0.0, 0.0, std::polar(1.0, -phase)};

  return {positive_phase, 0.0, 0.0, 1.0};
}

// Two-dimensional invariant-subspace block of the *ideal* qubitization walk as
// a function of the block-encoded singular value x.
//
// CONVENTION NOTE: the device walk in this library composes the zero-state
// reflection `reflect_about_zero` (which realizes `I - 2|0><0|`, see
// qubitization.h) with the block encoding, so the device walk's good-subspace
// block is evaluated at `-x` (i.e. -H/alpha) relative to this matrix. Callers
// predicting the *device* response from a phase list via evaluate_qsvt_response
// / phases_to_poly must therefore evaluate at -singular_value. See the QSPPACK
// end-to-end test, which negates the eigenvalue accordingly. This was verified
// numerically (device good-block == response evaluated at -x).
qsvt_response_matrix walk_response_matrix(double x) {
  const double y = std::sqrt(std::max(0.0, 1.0 - x * x));
  const complex off_diagonal(0.0, y);
  return {x, off_diagonal, off_diagonal, x};
}

bool is_finite_complex(const std::complex<double> &value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void validate_qsvt_response_x(double x) {
  if (!std::isfinite(x) || x < -1.0 || x > 1.0)
    throw std::invalid_argument("QSVT response x value must be in [-1, 1].");
}

void validate_qsvt_response_input(const std::vector<double> &phases, double x) {
  validate_qsvt_phase_sequence(phases);
  validate_qsvt_response_x(x);
}

void validate_qsvt_response_samples(const std::vector<double> &sample_points) {
  if (sample_points.empty())
    throw std::invalid_argument(
        "QSVT response error estimation requires at least one sample point.");

  for (double x : sample_points)
    validate_qsvt_response_x(x);
}

void validate_qsvt_sample_range(double min_x, double max_x,
                                std::size_t num_points) {
  validate_qsvt_response_x(min_x);
  validate_qsvt_response_x(max_x);

  if (min_x > max_x)
    throw std::invalid_argument(
        "QSVT sample point range must satisfy min_x <= max_x.");
  if (num_points == 0)
    throw std::invalid_argument(
        "QSVT sample point generation requires at least one point.");
}

double qsvt_sample_midpoint(double min_x, double max_x) {
  return 0.5 * (min_x + max_x);
}

} // namespace

qsvt_phase_sequence::qsvt_phase_sequence(std::vector<double> input_phases)
    : phases(std::move(input_phases)) {
  validate_qsvt_phase_sequence(phases);
}

std::size_t qsvt_phase_sequence::degree() const {
  return qsvt_polynomial_degree(phases.size());
}

qsvt_sequence_policy::qsvt_sequence_policy(
    std::vector<int> input_walk_directions)
    : walk_directions(std::move(input_walk_directions)) {
  validate_qsvt_sequence_policy(walk_directions.size(), *this);
}

qsvt_plan::qsvt_plan(qsvt_phase_sequence input_phases)
    : phase_sequence(std::move(input_phases)),
      sequence_policy(make_qsvt_sequence_policy(phase_sequence.degree())) {
  validate_qsvt_phase_sequence(phase_sequence.data());
  validate_qsvt_sequence_policy(phase_sequence.degree(), sequence_policy);
}

qsvt_plan::qsvt_plan(qsvt_phase_sequence input_phases,
                     qsvt_sequence_policy input_policy)
    : phase_sequence(std::move(input_phases)),
      sequence_policy(std::move(input_policy)) {
  validate_qsvt_phase_sequence(phase_sequence.data());
  validate_qsvt_sequence_policy(phase_sequence.degree(), sequence_policy);
}

qsvt_plan::qsvt_plan(std::vector<double> input_phases)
    : qsvt_plan(qsvt_phase_sequence(std::move(input_phases))) {}

qsvt_plan::qsvt_plan(std::vector<double> input_phases,
                     qsvt_sequence_policy input_policy)
    : qsvt_plan(qsvt_phase_sequence(std::move(input_phases)),
                std::move(input_policy)) {}

qsvt_transform_plan::qsvt_transform_plan(
    qsvt_transform_descriptor input_descriptor, qsvt_plan input_plan)
    : transform_descriptor(std::move(input_descriptor)),
      sequence_plan(std::move(input_plan)) {
  validate_qsvt_transform_phase_sequence(transform_descriptor,
                                         sequence_plan.phase_data());
}

bool is_valid_qsvt_phase_sequence(const std::vector<double> &phases) {
  if (phases.empty())
    return false;

  for (double phase : phases) {
    if (!std::isfinite(phase))
      return false;
  }

  return true;
}

void validate_qsvt_phase_sequence(const std::vector<double> &phases) {
  if (phases.empty())
    throw std::invalid_argument("QSVT phase sequence must not be empty.");

  for (std::size_t i = 0; i < phases.size(); ++i) {
    if (!std::isfinite(phases[i]))
      throw std::invalid_argument(
          "QSVT phase sequence contains a non-finite phase.");
  }
}

std::size_t qsvt_polynomial_degree(std::size_t num_phases) {
  if (num_phases == 0)
    throw std::invalid_argument(
        "QSVT polynomial degree requires at least one phase.");

  return num_phases - 1;
}

int qsvt_walk_direction_code(qsvt_walk_direction direction) {
  return direction == qsvt_walk_direction::adjoint ? qsvt_adjoint_walk
                                                   : qsvt_forward_walk;
}

bool is_valid_qsvt_sequence_policy(std::size_t degree,
                                   const qsvt_sequence_policy &policy) {
  if (policy.size() != degree)
    return false;

  for (int direction : policy.walk_direction_data()) {
    if (direction != qsvt_forward_walk && direction != qsvt_adjoint_walk)
      return false;
  }

  return true;
}

void validate_qsvt_sequence_policy(std::size_t degree,
                                   const qsvt_sequence_policy &policy) {
  if (policy.size() != degree)
    throw std::invalid_argument(
        "QSVT sequence policy length must match the polynomial degree.");

  for (int direction : policy.walk_direction_data()) {
    if (direction != qsvt_forward_walk && direction != qsvt_adjoint_walk)
      throw std::invalid_argument(
          "QSVT sequence policy contains an unknown walk direction.");
  }
}

qsvt_phase_sequence make_qsvt_phase_sequence(std::vector<double> phases) {
  return qsvt_phase_sequence(std::move(phases));
}

qsvt_sequence_policy make_qsvt_sequence_policy(std::size_t degree,
                                               qsvt_walk_direction direction) {
  std::vector<int> walk_directions(degree, qsvt_walk_direction_code(direction));
  return qsvt_sequence_policy(std::move(walk_directions));
}

qsvt_sequence_policy
make_qsvt_sequence_policy(std::vector<qsvt_walk_direction> directions) {
  std::vector<int> walk_directions;
  walk_directions.reserve(directions.size());
  for (auto direction : directions)
    walk_directions.push_back(qsvt_walk_direction_code(direction));

  return qsvt_sequence_policy(std::move(walk_directions));
}

qsvt_sequence_policy
make_alternating_qsvt_sequence_policy(std::size_t degree,
                                      qsvt_walk_direction first_direction) {
  std::vector<int> walk_directions;
  walk_directions.reserve(degree);

  auto direction = first_direction;
  for (std::size_t i = 0; i < degree; ++i) {
    walk_directions.push_back(qsvt_walk_direction_code(direction));
    direction = direction == qsvt_walk_direction::forward
                    ? qsvt_walk_direction::adjoint
                    : qsvt_walk_direction::forward;
  }

  return qsvt_sequence_policy(std::move(walk_directions));
}

qsvt_plan make_qsvt_plan(std::vector<double> phases) {
  return qsvt_plan(std::move(phases));
}

qsvt_plan make_qsvt_plan(std::vector<double> phases,
                         qsvt_sequence_policy policy) {
  return qsvt_plan(std::move(phases), std::move(policy));
}

bool is_valid_qsvt_transform_descriptor(
    const qsvt_transform_descriptor &descriptor) {
  if (!std::isfinite(descriptor.evolution_time) ||
      !std::isfinite(descriptor.condition_number) ||
      !std::isfinite(descriptor.target_error) ||
      !std::isfinite(descriptor.normalization))
    return false;

  if (descriptor.target_error < 0.0 || descriptor.normalization <= 0.0)
    return false;

  switch (descriptor.kind) {
  case qsvt_transform_kind::linear_solve:
    return descriptor.condition_number >= 1.0;
  case qsvt_transform_kind::real_time_hamiltonian_simulation:
  case qsvt_transform_kind::imaginary_time_hamiltonian_simulation:
    return descriptor.evolution_time >= 0.0;
  case qsvt_transform_kind::custom:
    return true;
  }

  return false;
}

void validate_qsvt_transform_descriptor(
    const qsvt_transform_descriptor &descriptor) {
  if (!is_valid_qsvt_transform_descriptor(descriptor))
    throw std::invalid_argument("Invalid QSVT transform descriptor.");
}

qsvt_response evaluate_qsvt_response(const std::vector<double> &phases,
                                     double x,
                                     qsvt_phase_convention convention) {
  validate_qsvt_response_input(phases, x);

  qsvt_response_matrix response = phase_response_matrix(phases[0], convention);
  const qsvt_response_matrix walk = walk_response_matrix(x);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    response = multiply(walk, response);
    response = multiply(phase_response_matrix(phases[i], convention), response);
  }

  qsvt_response result;
  result.value = response.a00;
  result.magnitude = std::abs(result.value);
  result.probability = std::norm(result.value);
  return result;
}

qsvt_response evaluate_qsvt_response(const qsvt_phase_sequence &phases,
                                     double x,
                                     qsvt_phase_convention convention) {
  return evaluate_qsvt_response(phases.data(), x, convention);
}

qsvt_response evaluate_qsvt_response(const qsvt_plan &plan, double x,
                                     qsvt_phase_convention convention) {
  return evaluate_qsvt_response(plan.phase_data(), x, convention);
}

qsvt_response evaluate_qsvt_response(const qsvt_transform_plan &plan,
                                     double x) {
  return evaluate_qsvt_response(plan.phase_data(), x,
                                plan.descriptor().phase_convention);
}

qsvt_response_error estimate_qsvt_response_error(
    const std::vector<double> &phases,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points,
    qsvt_phase_convention convention) {
  validate_qsvt_phase_sequence(phases);
  validate_qsvt_response_samples(sample_points);

  qsvt_response_error summary;
  summary.num_samples = sample_points.size();

  double sum_squared_error = 0.0;
  for (double x : sample_points) {
    const auto response = evaluate_qsvt_response(phases, x, convention);
    const auto target_value = target(x);
    if (!is_finite_complex(target_value))
      throw std::invalid_argument(
          "QSVT response error target returned a non-finite value.");

    const double error = std::abs(response.value - target_value);
    sum_squared_error += error * error;
    if (error > summary.max_abs_error) {
      summary.max_abs_error = error;
      summary.max_error_x = x;
    }
  }

  summary.rms_error =
      std::sqrt(sum_squared_error / static_cast<double>(summary.num_samples));
  return summary;
}

qsvt_response_error estimate_qsvt_response_error(
    const qsvt_phase_sequence &phases,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points,
    qsvt_phase_convention convention) {
  return estimate_qsvt_response_error(phases.data(), target, sample_points,
                                      convention);
}

qsvt_response_error estimate_qsvt_response_error(
    const qsvt_plan &plan,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points,
    qsvt_phase_convention convention) {
  return estimate_qsvt_response_error(plan.phase_data(), target, sample_points,
                                      convention);
}

qsvt_response_error estimate_qsvt_response_error(
    const qsvt_transform_plan &plan,
    const std::function<std::complex<double>(double)> &target,
    const std::vector<double> &sample_points) {
  return estimate_qsvt_response_error(plan.phase_data(), target, sample_points,
                                      plan.descriptor().phase_convention);
}

std::vector<double> make_uniform_qsvt_sample_points(double min_x, double max_x,
                                                    std::size_t num_points) {
  validate_qsvt_sample_range(min_x, max_x, num_points);

  if (num_points == 1)
    return {qsvt_sample_midpoint(min_x, max_x)};

  std::vector<double> points;
  points.reserve(num_points);
  const double step = (max_x - min_x) / static_cast<double>(num_points - 1);
  for (std::size_t i = 0; i < num_points; ++i)
    points.push_back(min_x + step * static_cast<double>(i));

  points.back() = max_x;
  return points;
}

std::vector<double> make_chebyshev_qsvt_sample_points(double min_x,
                                                      double max_x,
                                                      std::size_t num_points) {
  validate_qsvt_sample_range(min_x, max_x, num_points);

  if (num_points == 1)
    return {qsvt_sample_midpoint(min_x, max_x)};

  std::vector<double> points;
  points.reserve(num_points);
  const double midpoint = qsvt_sample_midpoint(min_x, max_x);
  const double half_width = 0.5 * (max_x - min_x);
  constexpr double pi = 3.141592653589793238462643383279502884;
  for (std::size_t i = 0; i < num_points; ++i) {
    const double theta = pi * static_cast<double>(num_points - 1 - i) /
                         static_cast<double>(num_points - 1);
    points.push_back(midpoint + half_width * std::cos(theta));
  }

  points.front() = min_x;
  points.back() = max_x;
  return points;
}

void validate_qsvt_transform_phase_sequence(
    const qsvt_transform_descriptor &descriptor,
    const std::vector<double> &phases) {
  validate_qsvt_transform_descriptor(descriptor);
  validate_qsvt_phase_sequence(phases);

  if (descriptor.degree_hint != 0 &&
      qsvt_polynomial_degree(phases.size()) != descriptor.degree_hint)
    throw std::invalid_argument(
        "QSVT transform phase sequence degree does not match the descriptor "
        "degree hint.");
}

qsvt_transform_plan
make_qsvt_transform_plan(const qsvt_transform_descriptor &descriptor,
                         std::vector<double> phases) {
  validate_qsvt_transform_phase_sequence(descriptor, phases);
  return qsvt_transform_plan(descriptor, qsvt_plan(std::move(phases)));
}

qsvt_transform_plan
make_qsvt_transform_plan(const qsvt_transform_descriptor &descriptor,
                         std::vector<double> phases,
                         qsvt_sequence_policy policy) {
  validate_qsvt_transform_phase_sequence(descriptor, phases);
  return qsvt_transform_plan(descriptor,
                             qsvt_plan(std::move(phases), std::move(policy)));
}

qsvt_transform_descriptor
make_linear_solve_qsvt_transform(double condition_number, double target_error,
                                 std::size_t degree_hint,
                                 double normalization) {
  qsvt_transform_descriptor descriptor;
  descriptor.kind = qsvt_transform_kind::linear_solve;
  descriptor.condition_number = condition_number;
  descriptor.target_error = target_error;
  descriptor.normalization = normalization;
  descriptor.degree_hint = degree_hint;
  validate_qsvt_transform_descriptor(descriptor);
  return descriptor;
}

qsvt_transform_descriptor make_real_time_hamiltonian_simulation_qsvt_transform(
    double evolution_time, double target_error, std::size_t degree_hint,
    double normalization) {
  qsvt_transform_descriptor descriptor;
  descriptor.kind = qsvt_transform_kind::real_time_hamiltonian_simulation;
  descriptor.evolution_time = evolution_time;
  descriptor.target_error = target_error;
  descriptor.normalization = normalization;
  descriptor.degree_hint = degree_hint;
  validate_qsvt_transform_descriptor(descriptor);
  return descriptor;
}

qsvt_transform_descriptor
make_imaginary_time_hamiltonian_simulation_qsvt_transform(
    double evolution_time, double target_error, std::size_t degree_hint,
    double normalization) {
  qsvt_transform_descriptor descriptor;
  descriptor.kind = qsvt_transform_kind::imaginary_time_hamiltonian_simulation;
  descriptor.evolution_time = evolution_time;
  descriptor.target_error = target_error;
  descriptor.normalization = normalization;
  descriptor.degree_hint = degree_hint;
  validate_qsvt_transform_descriptor(descriptor);
  return descriptor;
}

} // namespace cudaq::algorithms
