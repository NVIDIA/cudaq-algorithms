/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include <complex>

#include "cudaq.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/algorithms/qubitization/qubitization.h"

namespace {

void expect_basis_state(const cudaq::state &state, std::size_t index) {
  EXPECT_NEAR(std::norm(state[index]), 1.0, 1e-10);
}

} // namespace

// Test purpose: verify zero-state and PREPARE-state reflection kernels compile.
TEST(QubitizationTester, checkReflectionKernelsCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto zero_reflection_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    reflect_about_zero(anc);
  };
  EXPECT_NO_THROW(zero_reflection_test());

  auto prepared_reflection_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    reflect_about_prepare(anc, encoding);
  };
  EXPECT_NO_THROW(prepared_reflection_test());
}

// Test purpose: verify forward qubitization walk kernels compile.
TEST(QubitizationTester, checkWalkKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto walk_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_qubitization_walk(anc, sys, encoding);
  };
  EXPECT_NO_THROW(walk_test());

  auto walk_functor_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    qubitization_walk{}(anc, sys, encoding);
  };
  EXPECT_NO_THROW(walk_functor_test());
}

// Test purpose: verify adjoint qubitization walk kernels compile.
TEST(QubitizationTester, checkAdjointWalkKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto adjoint_walk_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_adjoint_qubitization_walk(anc, sys, encoding);
  };
  EXPECT_NO_THROW(adjoint_walk_test());

  auto adjoint_walk_functor_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    adjoint_qubitization_walk{}(anc, sys, encoding);
  };
  EXPECT_NO_THROW(adjoint_walk_functor_test());
}

// Test purpose: verify repeated forward qubitization walk kernels compile.
TEST(QubitizationTester, checkWalkPowerKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto walk_power_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_qubitization_walk_power(anc, sys, encoding, 2);
  };
  EXPECT_NO_THROW(walk_power_test());

  auto walk_power_functor_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    qubitization_walk_power{}(anc, sys, encoding, 2);
  };
  EXPECT_NO_THROW(walk_power_functor_test());
}

// Test purpose: verify repeated adjoint qubitization walk kernels compile.
TEST(QubitizationTester, checkAdjointWalkPowerKernelCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto adjoint_walk_power_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_adjoint_qubitization_walk_power(anc, sys, encoding, 2);
  };
  EXPECT_NO_THROW(adjoint_walk_power_test());

  auto adjoint_walk_power_functor_test = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    adjoint_qubitization_walk_power{}(anc, sys, encoding, 2);
  };
  EXPECT_NO_THROW(adjoint_walk_power_functor_test());
}

// Test purpose: verify controlled SELECT, reflection, and walk kernels compile.
TEST(QubitizationTester, checkControlledSelectAndWalkKernelsCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto controlled_select_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    encoding.controlled_select(control, anc, sys);
  };
  EXPECT_NO_THROW(controlled_select_test());

  auto controlled_reflection_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    encoding.prepare(anc);
    controlled_reflect_about_prepare(control, anc, encoding);
  };
  EXPECT_NO_THROW(controlled_reflection_test());

  auto controlled_walk_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_controlled_qubitization_walk(control, anc, sys, encoding);
  };
  EXPECT_NO_THROW(controlled_walk_test());

  auto controlled_walk_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    controlled_qubitization_walk{}(control, anc, sys, encoding);
  };
  EXPECT_NO_THROW(controlled_walk_functor_test());

  auto controlled_adjoint_walk_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_controlled_adjoint_qubitization_walk(control, anc, sys, encoding);
  };
  EXPECT_NO_THROW(controlled_adjoint_walk_test());

  auto controlled_adjoint_walk_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    controlled_adjoint_qubitization_walk{}(control, anc, sys, encoding);
  };
  EXPECT_NO_THROW(controlled_adjoint_walk_functor_test());
}

// Test purpose: verify controlled repeated walk kernels compile.
TEST(QubitizationTester, checkControlledWalkPowerKernelsCompile) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);

  auto controlled_walk_power_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_controlled_qubitization_walk_power(control, anc, sys, encoding, 2);
  };
  EXPECT_NO_THROW(controlled_walk_power_test());

  auto controlled_walk_power_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    controlled_qubitization_walk_power{}(control, anc, sys, encoding, 2);
  };
  EXPECT_NO_THROW(controlled_walk_power_functor_test());

  auto controlled_adjoint_walk_power_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_controlled_adjoint_qubitization_walk_power(control, anc, sys,
                                                     encoding, 2);
  };
  EXPECT_NO_THROW(controlled_adjoint_walk_power_test());

  auto controlled_adjoint_walk_power_functor_test = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    controlled_adjoint_qubitization_walk_power{}(control, anc, sys, encoding,
                                                 2);
  };
  EXPECT_NO_THROW(controlled_adjoint_walk_power_functor_test());
}

// Test purpose: verify controlled walk execution respects the control state.
TEST(QubitizationTester, checkControlledWalkExecution) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);

  auto control_off = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_controlled_qubitization_walk(control, anc, sys, encoding);
  };
  expect_basis_state(cudaq::get_state(control_off), 0);

  auto control_on = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    x(control);
    encoding.prepare(anc);
    apply_controlled_qubitization_walk(control, anc, sys, encoding);
  };
  const auto control_one_index =
      1ULL << (encoding.num_system() + encoding.num_ancilla());
  const auto system_one_index = 1ULL;
  expect_basis_state(cudaq::get_state(control_on),
                     control_one_index + system_one_index);
}

// Test purpose: verify controlled walk powers execute expected X-walk behavior.
TEST(QubitizationTester, checkControlledWalkPowerExecution) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = x(0);
  pauli_lcu encoding(h, 1);

  auto control_off_power_one = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    encoding.prepare(anc);
    apply_controlled_qubitization_walk_power(control, anc, sys, encoding, 1);
  };
  expect_basis_state(cudaq::get_state(control_off_power_one), 0);

  auto control_on_power_one = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    x(control);
    encoding.prepare(anc);
    apply_controlled_qubitization_walk_power(control, anc, sys, encoding, 1);
  };
  const auto control_one_index =
      1ULL << (encoding.num_system() + encoding.num_ancilla());
  const auto system_one_index = 1ULL;
  expect_basis_state(cudaq::get_state(control_on_power_one),
                     control_one_index + system_one_index);

  auto control_on_power_two = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    x(control);
    encoding.prepare(anc);
    apply_controlled_qubitization_walk_power(control, anc, sys, encoding, 2);
  };
  expect_basis_state(cudaq::get_state(control_on_power_two), control_one_index);
}
