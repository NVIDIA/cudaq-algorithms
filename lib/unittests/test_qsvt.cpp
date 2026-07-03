/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "cudaq/algorithms/block_encoding/kernels.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/algorithms/qsvt/qsvt.h"
#include "cudaq/algorithms/qubitization/qubitization.h"

namespace {

void expect_basis_state(const cudaq::state &state, std::size_t index) {
  EXPECT_NEAR(std::norm(state[index]), 1.0, 1e-10);
}

void expect_states_equal(const cudaq::state &actual,
                         const cudaq::state &expected, std::size_t dimension,
                         double tolerance = 1e-10) {
  for (std::size_t i = 0; i < dimension; ++i)
    EXPECT_NEAR(std::abs(std::complex<double>(actual[i]) -
                         std::complex<double>(expected[i])),
                0.0, tolerance)
        << "amplitude mismatch at index " << i;
}

} // namespace

// Test purpose: verify QSVT signal phase kernels compile in CUDA-Q kernels.
TEST(QSVTTester, signal_phase_kernels_compile) {
  using namespace cudaq::algorithms;

  auto qsvt_signal_phase_test = []() __qpu__ {
    cudaq::qvector<> one_signal(1);
    apply_qsvt_signal_phase(one_signal, 0.25);

    cudaq::qvector<> three_signal(3);
    qsvt_signal_phase{}(three_signal, -0.5);
  };
  EXPECT_NO_THROW(qsvt_signal_phase_test());

  auto controlled_qsvt_signal_phase_test = []() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(2);
    x(control);
    apply_controlled_qsvt_signal_phase(control, signal, 0.25);
    controlled_qsvt_signal_phase{}(control, signal, -0.5);
  };
  EXPECT_NO_THROW(controlled_qsvt_signal_phase_test());
}

// Test purpose: verify QSP-convention signal phase kernels compile.
TEST(QSVTTester, qsp_signal_phase_kernels_compile) {
  using namespace cudaq::algorithms;

  auto qsp_signal_phase_test = []() __qpu__ {
    cudaq::qvector<> one_signal(1);
    apply_qsp_signal_phase(one_signal, 0.25);

    cudaq::qvector<> three_signal(3);
    qsp_signal_phase{}(three_signal, -0.5);
  };
  EXPECT_NO_THROW(qsp_signal_phase_test());

  auto controlled_qsp_signal_phase_test = []() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(2);
    x(control);
    apply_controlled_qsp_signal_phase(control, signal, 0.25);
    controlled_qsp_signal_phase{}(control, signal, -0.5);
  };
  EXPECT_NO_THROW(controlled_qsp_signal_phase_test());
}

// Test purpose: verify QSVT/QSP sequence kernels compile with walk policies.
TEST(QSVTTester, qsvt_sequence_kernels_compile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto qsvt_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    apply_qsvt_sequence(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(qsvt_sequence_test());

  auto qsvt_adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(qsvt_adjoint_sequence_test());

  auto qsvt_policy_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
  };
  EXPECT_NO_THROW(qsvt_policy_sequence_test());

  auto qsvt_sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    qsvt_sequence{}(signal, system, encoding, phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(qsvt_sequence_functor_test());

  auto qsp_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    apply_qsp_sequence(signal, system, encoding, phase_data,
                       walk_direction_data);
  };
  EXPECT_NO_THROW(qsp_sequence_test());

  auto qsp_sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    qsp_sequence{}(signal, system, encoding, phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(qsp_sequence_functor_test());
}

// Test purpose: verify controlled QSVT/QSP sequence kernels compile.
TEST(QSVTTester, controlled_sequence_kernels_compile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto controlled_qsvt_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data);
  };
  EXPECT_NO_THROW(controlled_qsvt_sequence_test());

  auto controlled_qsvt_adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(controlled_qsvt_adjoint_sequence_test());

  auto controlled_qsvt_policy_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsvt_policy_sequence_test());

  auto controlled_qsvt_sequence_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    controlled_qsvt_sequence{}(control, signal, system, encoding, phase_data,
                               walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsvt_sequence_functor_test());

  auto controlled_qsp_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    apply_controlled_qsp_sequence(control, signal, system, encoding, phase_data,
                                  walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsp_sequence_test());

  auto controlled_qsp_sequence_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    controlled_qsp_sequence{}(control, signal, system, encoding, phase_data,
                              walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsp_sequence_functor_test());
}

// Test purpose: verify QSVT sequences reproduce one and two qubitization walks.
TEST(QSVTTester, qsvt_sequence_executes_expected_walk_powers) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);

  auto walk_once = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qubitization_walk(signal, system, encoding);
  };
  expect_basis_state(cudaq::get_state(walk_once), 1);

  auto one_walk_plan = make_qsvt_plan({0.0, 0.0});
  auto one_walk_kernel_data = one_walk_plan.kernel_data();
  auto one_walk_phases = one_walk_kernel_data.phases;
  auto one_walk_directions = one_walk_kernel_data.walk_directions;

  auto qsvt_one_walk = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    apply_qsvt_sequence(signal, system, encoding, one_walk_phases,
                        one_walk_directions);
  };
  expect_basis_state(cudaq::get_state(qsvt_one_walk), 1);

  auto two_walk_plan =
      make_qsvt_plan({0.0, 0.0, 0.0}, make_alternating_qsvt_sequence_policy(2));
  auto two_walk_kernel_data = two_walk_plan.kernel_data();
  auto two_walk_phases = two_walk_kernel_data.phases;
  auto two_walk_directions = two_walk_kernel_data.walk_directions;

  auto qsvt_two_walks = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    apply_qsvt_sequence(signal, system, encoding, two_walk_phases,
                        two_walk_directions);
  };
  expect_basis_state(cudaq::get_state(qsvt_two_walks), 0);
}

// Test purpose: verify the QSVT sequence reproduces the host response for a
// non-degenerate (2-term, 1-ancilla) Hamiltonian on an eigenstate.
TEST(QSVTTester, qsvt_sequence_matches_host_response_for_two_term_hamiltonian) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // h = 0.5 X + 0.3 Z has eigenvalues +/- lambda and ry(theta)|0> is the
  // +lambda eigenvector.
  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  ASSERT_EQ(encoding.num_ancilla(), 1);
  const double alpha = encoding.normalization();
  EXPECT_NEAR(alpha, 0.8, 1e-12);
  const double lambda = std::sqrt(0.34);
  const double theta = std::atan2(0.5, 0.3);

  // Odd degrees keep the -H/alpha walk sign observable.
  const std::vector<std::vector<double>> phase_sets = {
      {0.3, -0.4}, {0.2, -0.5, 0.1, 0.4}};

  for (const auto &phases : phase_sets) {
    auto qsvt_on_eigenstate = [&]() __qpu__ {
      cudaq::qvector<> signal(encoding.num_ancilla());
      cudaq::qvector<> system(encoding.num_system());
      ry(theta, system[0]);
      apply_qsvt_sequence(signal, system, encoding, phases);
    };
    auto state = cudaq::get_state(qsvt_on_eigenstate);

    // The signal register is allocated first, so it occupies the high bit and
    // the good (signal = 0) subspace is the first 2^num_system amplitudes.
    // The device walk realizes the response at -eigenvalue / alpha.
    const auto response = evaluate_qsvt_response(phases, -lambda / alpha);
    const std::complex<double> expected0 =
        response.value * std::cos(0.5 * theta);
    const std::complex<double> expected1 =
        response.value * std::sin(0.5 * theta);

    EXPECT_NEAR(std::abs(std::complex<double>(state[0]) - expected0), 0.0,
                1e-10);
    EXPECT_NEAR(std::abs(std::complex<double>(state[1]) - expected1), 0.0,
                1e-10);

    // Guard against a silent sign-convention flip: the +lambda/alpha
    // prediction must differ for odd-degree sequences.
    const auto flipped = evaluate_qsvt_response(phases, lambda / alpha);
    EXPECT_GT(std::abs(response.value - flipped.value), 1e-6);
  }
}

// Test purpose: verify QSP phases phi execute exactly as projector phases
// 2*phi for a non-degenerate encoding.
TEST(QSVTTester, qsp_sequence_matches_doubled_projector_phases) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  const double theta = 0.7;

  std::vector<double> qsp_phases{0.15, -0.3, 0.45};
  std::vector<double> projector_phases{0.3, -0.6, 0.9};

  auto qsp_kernel = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    ry(theta, system[0]);
    apply_qsp_sequence(signal, system, encoding, qsp_phases);
  };
  auto qsvt_kernel = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    ry(theta, system[0]);
    apply_qsvt_sequence(signal, system, encoding, projector_phases);
  };

  const auto dimension =
      1ULL << (encoding.num_ancilla() + encoding.num_system());
  expect_states_equal(cudaq::get_state(qsp_kernel),
                      cudaq::get_state(qsvt_kernel), dimension);
}

// Test purpose: verify controlled QSVT sequences respect the control state for
// a non-degenerate encoding.
TEST(QSVTTester, controlled_qsvt_sequence_respects_control) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  const double theta = 0.7;
  std::vector<double> phases{0.3, -0.4, 0.25};

  auto uncontrolled = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    ry(theta, system[0]);
    apply_qsvt_sequence(signal, system, encoding, phases);
  };
  auto controlled_on = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    ry(theta, system[0]);
    apply_controlled_qsvt_sequence(control, signal, system, encoding, phases);
  };
  auto controlled_off = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    ry(theta, system[0]);
    apply_controlled_qsvt_sequence(control, signal, system, encoding, phases);
  };

  const auto dimension =
      1ULL << (encoding.num_ancilla() + encoding.num_system());
  auto uncontrolled_state = cudaq::get_state(uncontrolled);
  auto on_state = cudaq::get_state(controlled_on);
  auto off_state = cudaq::get_state(controlled_off);

  // The control qubit is allocated first (highest bit). With control |1>, the
  // control = 1 half must reproduce the uncontrolled state and the control = 0
  // half must be empty.
  for (std::size_t i = 0; i < dimension; ++i) {
    EXPECT_NEAR(std::abs(std::complex<double>(on_state[dimension + i]) -
                         std::complex<double>(uncontrolled_state[i])),
                0.0, 1e-10);
    EXPECT_NEAR(std::abs(std::complex<double>(on_state[i])), 0.0, 1e-10);
  }

  // With control |0> the sequence must collapse to the identity, leaving only
  // the ry state preparation.
  EXPECT_NEAR(std::abs(std::complex<double>(off_state[0]) -
                       std::cos(0.5 * theta)),
              0.0, 1e-10);
  EXPECT_NEAR(std::abs(std::complex<double>(off_state[1]) -
                       std::sin(0.5 * theta)),
              0.0, 1e-10);
  for (std::size_t i = 2; i < 2 * dimension; ++i)
    EXPECT_NEAR(std::abs(std::complex<double>(off_state[i])), 0.0, 1e-10);
}

// Test purpose: verify the header sequence family matches the Python-facing
// device kernel qsvt::apply_phase_sequence bit for bit.
TEST(QSVTTester, qsvt_sequence_matches_device_phase_sequence_kernel) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  const double theta = 0.7;

  std::vector<double> phases{0.3, -0.4, 0.25};
  // Exercise both the forward and adjoint step paths.
  std::vector<int> directions{qsvt_forward_walk, qsvt_adjoint_walk};

  const auto angles = encoding.get_angles();
  const auto term_controls = encoding.get_term_controls();
  const auto term_ops = encoding.get_term_ops();
  const auto term_lengths = encoding.get_term_lengths();
  const auto term_signs = encoding.get_term_signs();

  auto header_kernel = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    ry(theta, system[0]);
    apply_qsvt_sequence(signal, system, encoding, phases, directions);
  };
  auto device_kernel = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    ry(theta, system[0]);
    cudaq_algorithms::qsvt::apply_phase_sequence(
        signal, system, phases, directions, angles, term_controls, term_ops,
        term_lengths, term_signs);
  };

  const auto dimension =
      1ULL << (encoding.num_ancilla() + encoding.num_system());
  expect_states_equal(cudaq::get_state(header_kernel),
                      cudaq::get_state(device_kernel), dimension);
}

// Test purpose: verify response evaluation and QSVT/QSP phase conventions.
TEST(QSVTTester, qsvt_response_conventions) {
  using namespace cudaq::algorithms;

  auto one_walk = evaluate_qsvt_response({0.0, 0.0}, 0.25);
  EXPECT_NEAR(0.25, one_walk.value.real(), 1e-12);
  EXPECT_NEAR(0.0, one_walk.value.imag(), 1e-12);
  EXPECT_NEAR(0.25, one_walk.magnitude, 1e-12);
  EXPECT_NEAR(0.0625, one_walk.probability, 1e-12);

  auto two_walks = evaluate_qsvt_response({0.0, 0.0, 0.0}, 0.25);
  EXPECT_NEAR(2.0 * 0.25 * 0.25 - 1.0, two_walks.value.real(), 1e-12);
  EXPECT_NEAR(0.0, two_walks.value.imag(), 1e-12);

  std::vector<double> phases{0.2, -0.3, 0.4};
  auto qsvt_response =
      evaluate_qsvt_response(phases, 0.5, qsvt_phase_convention::qsvt);
  auto qsp_response =
      evaluate_qsvt_response(phases, 0.5, qsvt_phase_convention::qsp);
  EXPECT_GT(std::abs(qsvt_response.value - qsp_response.value), 1e-6);

  // QSPPACK's full phase factors use the QSP Z-rotation convention.
  std::vector<double> qsppack_cosine_phases{
      0.78539811199339948,    1.1393905344921082e-05, -0.0013479778846395907,
      0.062500795316736538,   -0.39587833857675897,   0.062500795316736538,
      -0.0013479778846395907, 1.1393905344921082e-05, 0.78539811199339948};
  auto qsppack_response = evaluate_qsvt_response(qsppack_cosine_phases, 0.5,
                                                 qsvt_phase_convention::qsp);
  EXPECT_NEAR(0.5 * std::cos(0.5), qsppack_response.value.real(), 1e-8);
}
