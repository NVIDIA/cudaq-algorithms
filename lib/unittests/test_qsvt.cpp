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
#include <limits>
#include <stdexcept>
#include <vector>

#include "cudaq/algorithms/qsvt/qsvt.h"
#include "cudaq/algorithms/qubitization/qubitization.h"

TEST(QSVTTester, checkSignalPhaseKernelCompile) {
  using namespace cudaq::algorithms;

  auto one_signal_test = []() __qpu__ {
    cudaq::qvector<> signal(1);
    apply_qsvt_signal_phase(signal, 0.25);
  };
  EXPECT_NO_THROW(one_signal_test());

  auto three_signal_test = []() __qpu__ {
    cudaq::qvector<> signal(3);
    qsvt_signal_phase{}(signal, -0.5);
  };
  EXPECT_NO_THROW(three_signal_test());
}

TEST(QSVTTester, checkControlledSignalPhaseKernelCompile) {
  using namespace cudaq::algorithms;

  auto controlled_phase_test = []() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(2);
    x(control);
    apply_controlled_qsvt_signal_phase(control, signal, 0.25);
  };
  EXPECT_NO_THROW(controlled_phase_test());

  auto controlled_phase_functor_test = []() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(3);
    controlled_qsvt_signal_phase{}(control, signal, -0.5);
  };
  EXPECT_NO_THROW(controlled_phase_functor_test());
}

TEST(QSVTTester, checkQSPSignalPhaseKernelCompile) {
  using namespace cudaq::algorithms;

  auto one_signal_test = []() __qpu__ {
    cudaq::qvector<> signal(1);
    apply_qsp_signal_phase(signal, 0.25);
  };
  EXPECT_NO_THROW(one_signal_test());

  auto three_signal_test = []() __qpu__ {
    cudaq::qvector<> signal(3);
    qsp_signal_phase{}(signal, -0.5);
  };
  EXPECT_NO_THROW(three_signal_test());

  auto controlled_phase_test = []() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(2);
    x(control);
    apply_controlled_qsp_signal_phase(control, signal, 0.25);
  };
  EXPECT_NO_THROW(controlled_phase_test());

  auto controlled_phase_functor_test = []() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(3);
    controlled_qsp_signal_phase{}(control, signal, -0.5);
  };
  EXPECT_NO_THROW(controlled_phase_functor_test());
}

TEST(QSVTTester, checkSequenceKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto phases = make_qsvt_phase_sequence({0.1, -0.2, 0.3});
  qsvt_plan plan(phases);
  auto phase_data = plan.phase_data();

  auto sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(sequence_test());

  auto sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(sequence_functor_test());
}

TEST(QSVTTester, checkQSPSequenceKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsp_sequence(signal, system, encoding, phase_data);
  };
  EXPECT_NO_THROW(sequence_test());

  auto adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsp_sequence(signal, system, encoding, phase_data,
                       qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(adjoint_sequence_test());

  auto policy_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsp_sequence(signal, system, encoding, phase_data,
                       walk_direction_data);
  };
  EXPECT_NO_THROW(policy_sequence_test());

  auto sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsp_sequence{}(signal, system, encoding, phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(sequence_functor_test());
}

TEST(QSVTTester, checkSequenceWalkDirectionKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto phase_data = plan.phase_data();

  auto adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(adjoint_sequence_test());

  auto adjoint_sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data,
                    qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(adjoint_sequence_functor_test());
}

TEST(QSVTTester, checkQubitizationAndQSVTExecution) {
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
  auto walk_counts = cudaq::sample(100, walk_once);
  EXPECT_FLOAT_EQ(1.0, walk_counts.probability("1"));

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
  auto one_walk_counts = cudaq::sample(100, qsvt_one_walk);
  EXPECT_FLOAT_EQ(1.0, one_walk_counts.probability("1"));

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
  auto two_walk_counts = cudaq::sample(100, qsvt_two_walks);
  EXPECT_FLOAT_EQ(1.0, two_walk_counts.probability("0"));
}

TEST(QSVTTester, checkControlledSequenceKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto controlled_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data);
  };
  EXPECT_NO_THROW(controlled_sequence_test());

  auto controlled_adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(controlled_adjoint_sequence_test());

  auto controlled_policy_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_policy_sequence_test());

  auto controlled_sequence_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    controlled_qsvt_sequence{}(control, signal, system, encoding, phase_data,
                               walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_sequence_functor_test());
}

TEST(QSVTTester, checkControlledQSPSequenceKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3});
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto controlled_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsp_sequence(control, signal, system, encoding,
                                  phase_data);
  };
  EXPECT_NO_THROW(controlled_sequence_test());

  auto controlled_adjoint_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsp_sequence(control, signal, system, encoding, phase_data,
                                  qsvt_walk_direction::adjoint);
  };
  EXPECT_NO_THROW(controlled_adjoint_sequence_test());

  auto controlled_policy_sequence_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsp_sequence(control, signal, system, encoding, phase_data,
                                  walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_policy_sequence_test());

  auto controlled_sequence_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    controlled_qsp_sequence{}(control, signal, system, encoding, phase_data,
                              walk_direction_data);
  };
  EXPECT_NO_THROW(controlled_sequence_functor_test());
}

TEST(QSVTTester, checkControlledQSVTExecution) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);
  auto plan = make_qsvt_plan({0.0, 0.0});
  auto kernel_data = plan.kernel_data();
  auto phase_data = kernel_data.phases;
  auto walk_direction_data = kernel_data.walk_directions;

  auto control_off = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, walk_direction_data);
  };
  auto off_counts = cudaq::sample(100, control_off);
  EXPECT_FLOAT_EQ(1.0, off_counts.probability("00"));

  auto control_on = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    x(control);
    encoding.prepare(signal);
    apply_controlled_qsvt_sequence(control, signal, system, encoding,
                                   phase_data, walk_direction_data);
  };
  auto on_counts = cudaq::sample(100, control_on);
  EXPECT_FLOAT_EQ(1.0, on_counts.probability("11"));
}

TEST(QSVTTester, checkSequencePolicyKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  auto policy =
      make_alternating_qsvt_sequence_policy(3, qsvt_walk_direction::adjoint);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3, -0.4}, policy);
  auto phase_data = plan.phase_data();
  auto walk_direction_data = plan.walk_direction_data();

  auto sequence_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    apply_qsvt_sequence(signal, system, encoding, phase_data,
                        walk_direction_data);
  };
  EXPECT_NO_THROW(sequence_test());

  auto sequence_functor_test = [&]() __qpu__ {
    cudaq::qvector<> signal(encoding.num_ancilla());
    cudaq::qvector<> system(encoding.num_system());
    encoding.prepare(signal);
    qsvt_sequence{}(signal, system, encoding, phase_data, walk_direction_data);
  };
  EXPECT_NO_THROW(sequence_functor_test());
}

TEST(QSVTTester, checkPlanMetadata) {
  using namespace cudaq::algorithms;

  qsvt_plan plan(std::vector<double>{0.1, -0.2, 0.3});

  EXPECT_EQ(plan.num_phases(), 3);
  EXPECT_EQ(plan.degree(), 2);
  EXPECT_EQ(plan.phases().size(), 3);
  EXPECT_DOUBLE_EQ(plan.phase_data()[1], -0.2);
  EXPECT_EQ(plan.policy().size(), 2);
  EXPECT_EQ(plan.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(plan.walk_direction_data()[1], qsvt_forward_walk);

  auto kernel_data = plan.kernel_data();
  EXPECT_EQ(kernel_data.phases.size(), 3);
  EXPECT_EQ(kernel_data.walk_directions.size(), 2);
  EXPECT_DOUBLE_EQ(kernel_data.phases[1], -0.2);
  EXPECT_EQ(kernel_data.walk_directions[0], qsvt_forward_walk);
}

TEST(QSVTTester, checkPlanPolicyMetadata) {
  using namespace cudaq::algorithms;

  auto policy = make_alternating_qsvt_sequence_policy(3);
  auto plan = make_qsvt_plan({0.1, -0.2, 0.3, -0.4}, policy);

  EXPECT_EQ(plan.degree(), 3);
  EXPECT_EQ(plan.policy().degree(), 3);
  EXPECT_EQ(plan.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(plan.walk_direction_data()[1], qsvt_adjoint_walk);
  EXPECT_EQ(plan.walk_direction_data()[2], qsvt_forward_walk);
}

TEST(QSVTTester, checkPlanFactoryAndValidation) {
  using namespace cudaq::algorithms;

  auto plan = make_qsvt_plan({0.25});
  EXPECT_EQ(plan.num_phases(), 1);
  EXPECT_EQ(plan.degree(), 0);

  EXPECT_THROW(qsvt_plan(qsvt_phase_sequence{}), std::invalid_argument);
  EXPECT_THROW(make_qsvt_plan({0.1, 0.2}, make_qsvt_sequence_policy(2)),
               std::invalid_argument);
  EXPECT_NO_THROW(make_qsvt_plan(
      {0.1, 0.2, 0.3},
      make_qsvt_sequence_policy(2, qsvt_walk_direction::adjoint)));
}

TEST(QSVTTester, checkPhaseSequenceMetadata) {
  using namespace cudaq::algorithms;

  qsvt_phase_sequence phases({0.1, -0.2, 0.3, 0.4});

  EXPECT_FALSE(phases.empty());
  EXPECT_EQ(phases.size(), 4);
  EXPECT_EQ(phases.degree(), 3);
  EXPECT_DOUBLE_EQ(phases[0], 0.1);
  EXPECT_DOUBLE_EQ(phases.data()[2], 0.3);
}

TEST(QSVTTester, checkPhaseSequenceFactory) {
  using namespace cudaq::algorithms;

  auto phases = make_qsvt_phase_sequence({0.0, 1.0});

  EXPECT_EQ(phases.size(), 2);
  EXPECT_EQ(phases.degree(), 1);
}

TEST(QSVTTester, checkPhaseSequenceValidation) {
  using namespace cudaq::algorithms;

  EXPECT_TRUE(is_valid_qsvt_phase_sequence({0.0}));
  EXPECT_TRUE(is_valid_qsvt_phase_sequence({0.0, 1.0, -1.0}));
  EXPECT_FALSE(is_valid_qsvt_phase_sequence({}));
  EXPECT_FALSE(is_valid_qsvt_phase_sequence(
      {0.0, std::numeric_limits<double>::quiet_NaN()}));
  EXPECT_FALSE(is_valid_qsvt_phase_sequence(
      {0.0, std::numeric_limits<double>::infinity()}));

  EXPECT_NO_THROW(validate_qsvt_phase_sequence({0.0, 1.0}));
  EXPECT_THROW(validate_qsvt_phase_sequence({}), std::invalid_argument);
  EXPECT_THROW(
      validate_qsvt_phase_sequence({std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
}

TEST(QSVTTester, checkSequencePolicyFactoryAndValidation) {
  using namespace cudaq::algorithms;

  auto adjoint_policy =
      make_qsvt_sequence_policy(3, qsvt_walk_direction::adjoint);
  EXPECT_EQ(adjoint_policy.size(), 3);
  EXPECT_EQ(adjoint_policy.walk_direction_data()[0], qsvt_adjoint_walk);
  EXPECT_EQ(adjoint_policy.walk_direction_data()[2], qsvt_adjoint_walk);

  auto custom_policy = make_qsvt_sequence_policy(
      {qsvt_walk_direction::forward, qsvt_walk_direction::adjoint});
  EXPECT_EQ(custom_policy.size(), 2);
  EXPECT_EQ(custom_policy.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(custom_policy.walk_direction_data()[1], qsvt_adjoint_walk);

  auto alternating_policy =
      make_alternating_qsvt_sequence_policy(4, qsvt_walk_direction::adjoint);
  EXPECT_EQ(alternating_policy.walk_direction_data()[0], qsvt_adjoint_walk);
  EXPECT_EQ(alternating_policy.walk_direction_data()[1], qsvt_forward_walk);
  EXPECT_EQ(alternating_policy.walk_direction_data()[2], qsvt_adjoint_walk);
  EXPECT_EQ(alternating_policy.walk_direction_data()[3], qsvt_forward_walk);

  EXPECT_TRUE(is_valid_qsvt_sequence_policy(2, custom_policy));
  EXPECT_FALSE(is_valid_qsvt_sequence_policy(3, custom_policy));
  EXPECT_THROW(validate_qsvt_sequence_policy(
                   1, qsvt_sequence_policy(std::vector<int>{2})),
               std::invalid_argument);
}

TEST(QSVTTester, checkTransformDescriptorFactories) {
  using namespace cudaq::algorithms;

  auto linear_solve = make_linear_solve_qsvt_transform(12.0, 1e-3, 27, 2.0);
  EXPECT_EQ(linear_solve.kind, qsvt_transform_kind::linear_solve);
  EXPECT_EQ(linear_solve.phase_convention, qsvt_phase_convention::qsvt);
  EXPECT_DOUBLE_EQ(linear_solve.condition_number, 12.0);
  EXPECT_DOUBLE_EQ(linear_solve.target_error, 1e-3);
  EXPECT_DOUBLE_EQ(linear_solve.normalization, 2.0);
  EXPECT_EQ(linear_solve.degree_hint, 27);
  EXPECT_TRUE(is_valid_qsvt_transform_descriptor(linear_solve));

  auto real_time =
      make_real_time_hamiltonian_simulation_qsvt_transform(0.75, 1e-4, 32, 1.5);
  EXPECT_EQ(real_time.kind,
            qsvt_transform_kind::real_time_hamiltonian_simulation);
  EXPECT_DOUBLE_EQ(real_time.evolution_time, 0.75);

  auto imaginary_time =
      make_imaginary_time_hamiltonian_simulation_qsvt_transform(1.25, 1e-5, 48);
  EXPECT_EQ(imaginary_time.kind,
            qsvt_transform_kind::imaginary_time_hamiltonian_simulation);
  EXPECT_DOUBLE_EQ(imaginary_time.evolution_time, 1.25);

  qsvt_transform_descriptor invalid_normalization;
  invalid_normalization.normalization = 0.0;
  EXPECT_FALSE(is_valid_qsvt_transform_descriptor(invalid_normalization));

  EXPECT_THROW(make_linear_solve_qsvt_transform(0.5, 1e-3),
               std::invalid_argument);
  EXPECT_THROW(make_real_time_hamiltonian_simulation_qsvt_transform(-0.1, 1e-3),
               std::invalid_argument);
  EXPECT_THROW(
      make_imaginary_time_hamiltonian_simulation_qsvt_transform(1.0, -1e-3),
      std::invalid_argument);
}

TEST(QSVTTester, checkResponseEvaluator) {
  using namespace cudaq::algorithms;

  auto phase_only = evaluate_qsvt_response({0.25}, 0.3);
  EXPECT_NEAR(std::cos(0.25), phase_only.value.real(), 1e-12);
  EXPECT_NEAR(std::sin(0.25), phase_only.value.imag(), 1e-12);
  EXPECT_NEAR(1.0, phase_only.magnitude, 1e-12);
  EXPECT_NEAR(1.0, phase_only.probability, 1e-12);

  auto one_walk = evaluate_qsvt_response({0.0, 0.0}, 0.25);
  EXPECT_NEAR(0.25, one_walk.value.real(), 1e-12);
  EXPECT_NEAR(0.0, one_walk.value.imag(), 1e-12);
  EXPECT_NEAR(0.25, one_walk.magnitude, 1e-12);
  EXPECT_NEAR(0.0625, one_walk.probability, 1e-12);

  auto two_walks = evaluate_qsvt_response({0.0, 0.0, 0.0}, 0.25);
  EXPECT_NEAR(2.0 * 0.25 * 0.25 - 1.0, two_walks.value.real(), 1e-12);
  EXPECT_NEAR(0.0, two_walks.value.imag(), 1e-12);

  qsvt_phase_sequence phases({0.0, 0.0});
  EXPECT_NEAR(0.5, evaluate_qsvt_response(phases, 0.5).value.real(), 1e-12);

  auto plan = make_qsvt_plan({0.0, 0.0, 0.0});
  EXPECT_NEAR(-0.5, evaluate_qsvt_response(plan, 0.5).value.real(), 1e-12);

  EXPECT_THROW(evaluate_qsvt_response({0.0}, 1.1), std::invalid_argument);
  EXPECT_THROW(
      evaluate_qsvt_response({0.0}, std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);
}

TEST(QSVTTester, checkSamplePointFactories) {
  using namespace cudaq::algorithms;

  auto uniform = make_uniform_qsvt_sample_points(-1.0, 1.0, 5);
  ASSERT_EQ(uniform.size(), 5);
  EXPECT_NEAR(-1.0, uniform.front(), 1e-12);
  EXPECT_NEAR(-0.5, uniform[1], 1e-12);
  EXPECT_NEAR(0.0, uniform[2], 1e-12);
  EXPECT_NEAR(1.0, uniform.back(), 1e-12);

  auto uniform_midpoint = make_uniform_qsvt_sample_points(-0.25, 0.75, 1);
  ASSERT_EQ(uniform_midpoint.size(), 1);
  EXPECT_NEAR(0.25, uniform_midpoint.front(), 1e-12);

  auto chebyshev = make_chebyshev_qsvt_sample_points(-1.0, 1.0, 5);
  ASSERT_EQ(chebyshev.size(), 5);
  EXPECT_NEAR(-1.0, chebyshev.front(), 1e-12);
  EXPECT_NEAR(-std::sqrt(0.5), chebyshev[1], 1e-12);
  EXPECT_NEAR(0.0, chebyshev[2], 1e-12);
  EXPECT_NEAR(std::sqrt(0.5), chebyshev[3], 1e-12);
  EXPECT_NEAR(1.0, chebyshev.back(), 1e-12);

  auto shifted_chebyshev = make_chebyshev_qsvt_sample_points(0.0, 1.0, 3);
  ASSERT_EQ(shifted_chebyshev.size(), 3);
  EXPECT_NEAR(0.0, shifted_chebyshev.front(), 1e-12);
  EXPECT_NEAR(0.5, shifted_chebyshev[1], 1e-12);
  EXPECT_NEAR(1.0, shifted_chebyshev.back(), 1e-12);

  EXPECT_THROW(make_uniform_qsvt_sample_points(0.5, -0.5, 4),
               std::invalid_argument);
  EXPECT_THROW(make_uniform_qsvt_sample_points(-1.1, 1.0, 4),
               std::invalid_argument);
  EXPECT_THROW(make_chebyshev_qsvt_sample_points(-1.0, 1.0, 0),
               std::invalid_argument);
}

TEST(QSVTTester, checkResponseErrorEstimator) {
  using namespace cudaq::algorithms;

  auto sample_points = make_uniform_qsvt_sample_points(-1.0, 1.0, 5);
  auto identity_target = [](double x) { return std::complex<double>(x, 0.0); };
  auto zero_target = [](double) { return std::complex<double>(0.0, 0.0); };

  auto exact_error =
      estimate_qsvt_response_error({0.0, 0.0}, identity_target, sample_points);
  EXPECT_EQ(exact_error.num_samples, sample_points.size());
  EXPECT_NEAR(0.0, exact_error.max_abs_error, 1e-12);
  EXPECT_NEAR(0.0, exact_error.rms_error, 1e-12);

  auto mismatch_error =
      estimate_qsvt_response_error({0.0, 0.0}, zero_target, sample_points);
  EXPECT_NEAR(1.0, mismatch_error.max_abs_error, 1e-12);
  EXPECT_NEAR(std::sqrt(0.5), mismatch_error.rms_error, 1e-12);
  EXPECT_NEAR(-1.0, mismatch_error.max_error_x, 1e-12);

  qsvt_phase_sequence phases({0.0, 0.0});
  EXPECT_NEAR(
      0.0,
      estimate_qsvt_response_error(phases, identity_target, sample_points)
          .max_abs_error,
      1e-12);

  auto plan = make_qsvt_plan({0.0, 0.0});
  EXPECT_NEAR(0.0,
              estimate_qsvt_response_error(plan, identity_target, sample_points)
                  .max_abs_error,
              1e-12);
}

TEST(QSVTTester, checkResponseErrorEstimatorValidation) {
  using namespace cudaq::algorithms;

  auto target = [](double x) { return std::complex<double>(x, 0.0); };
  auto non_finite_target = [](double) {
    return std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0);
  };

  EXPECT_THROW(estimate_qsvt_response_error({0.0}, target, {}),
               std::invalid_argument);
  EXPECT_THROW(estimate_qsvt_response_error({0.0}, target, {1.1}),
               std::invalid_argument);
  EXPECT_THROW(estimate_qsvt_response_error({0.0}, non_finite_target, {0.0}),
               std::invalid_argument);
}

TEST(QSVTTester, checkResponseEvaluatorConventions) {
  using namespace cudaq::algorithms;

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

  auto descriptor = make_real_time_hamiltonian_simulation_qsvt_transform(
      0.75, 1e-4, phases.size() - 1);
  descriptor.phase_convention = qsvt_phase_convention::qsp;
  auto transform_plan = make_qsvt_transform_plan(descriptor, phases);
  auto transform_response = evaluate_qsvt_response(transform_plan, 0.5);

  EXPECT_NEAR(qsp_response.value.real(), transform_response.value.real(),
              1e-12);
  EXPECT_NEAR(qsp_response.value.imag(), transform_response.value.imag(),
              1e-12);
}

TEST(QSVTTester, checkTransformPlanFactories) {
  using namespace cudaq::algorithms;

  auto real_time =
      make_real_time_hamiltonian_simulation_qsvt_transform(0.75, 1e-4, 2);
  auto transform_plan = make_qsvt_transform_plan(real_time, {0.1, -0.2, 0.3});

  EXPECT_EQ(transform_plan.descriptor().kind,
            qsvt_transform_kind::real_time_hamiltonian_simulation);
  EXPECT_EQ(transform_plan.degree(), 2);
  EXPECT_EQ(transform_plan.num_phases(), 3);
  EXPECT_DOUBLE_EQ(transform_plan.phase_data()[1], -0.2);
  EXPECT_EQ(transform_plan.walk_direction_data()[0], qsvt_forward_walk);
  EXPECT_EQ(transform_plan.plan().degree(), 2);
  EXPECT_EQ(transform_plan.kernel_data().walk_directions.size(), 2);

  auto policy =
      make_alternating_qsvt_sequence_policy(2, qsvt_walk_direction::adjoint);
  auto policy_plan =
      make_qsvt_transform_plan(real_time, {0.1, -0.2, 0.3}, policy);
  EXPECT_EQ(policy_plan.walk_direction_data()[0], qsvt_adjoint_walk);
  EXPECT_EQ(policy_plan.walk_direction_data()[1], qsvt_forward_walk);

  qsvt_transform_descriptor custom;
  EXPECT_NO_THROW(validate_qsvt_transform_phase_sequence(custom, {0.25}));
  EXPECT_THROW(make_qsvt_transform_plan(real_time, {0.1, -0.2}),
               std::invalid_argument);

  qsvt_transform_descriptor invalid;
  invalid.normalization = 0.0;
  EXPECT_THROW(validate_qsvt_transform_phase_sequence(invalid, {0.25}),
               std::invalid_argument);
}

TEST(QSVTTester, checkPolynomialDegreeConvention) {
  using namespace cudaq::algorithms;

  EXPECT_EQ(qsvt_polynomial_degree(1), 0);
  EXPECT_EQ(qsvt_polynomial_degree(4), 3);
  EXPECT_THROW(qsvt_polynomial_degree(0), std::invalid_argument);
}
