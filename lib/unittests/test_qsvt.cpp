/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#include "cudaq/algorithms/get_state.h"
#include "cudaq/algorithms/qsvt/qsvt.h"
#include "cudaq/algorithms/qubitization/qubitization.h"

namespace {

void expect_basis_state(const cudaq::state &state, std::size_t index) {
  EXPECT_NEAR(std::norm(state[index]), 1.0, 1e-10);
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
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(qsvt_sequence_test());

  auto qsvt_adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(qsvt_adjoint_sequence_test());

  auto qsvt_policy_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
  };
  EXPECT_NO_THROW(qsvt_policy_sequence_test());

  auto qsvt_sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(qsvt_sequence_functor_test());

  auto qsp_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsp_sequence(signal, system, encoding, phase_data,
                       walk_direction_data);
  };
  EXPECT_NO_THROW(qsp_sequence_test());

  auto qsp_sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
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
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data);
  };
  EXPECT_NO_THROW(controlled_qsvt_sequence_test());

  auto controlled_qsvt_adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(controlled_qsvt_adjoint_sequence_test());

  auto controlled_qsvt_policy_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsvt_policy_sequence_test());

  auto controlled_qsvt_sequence_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    controlled_qsvt_sequence{}(control, signal, system, encoding, phase_data,
                               walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsvt_sequence_functor_test());

  auto controlled_qsp_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsp_sequence(control, signal, system, encoding, phase_data,
                                  walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_qsp_sequence_test());

  auto controlled_qsp_sequence_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
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
    encoding.prepare(signal);
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
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, two_walk_phases,
                        two_walk_directions);
  };
  expect_basis_state(cudaq::get_state(qsvt_two_walks), 0);
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
