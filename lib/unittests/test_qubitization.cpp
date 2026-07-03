/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>

#include "cudaq.h"
#include "cudaq/algorithms/get_state.h"
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

// Chebyshev polynomial of the first kind on [-1, 1].
double chebyshev(int order, double value) {
  return std::cos(static_cast<double>(order) * std::acos(value));
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

// Test purpose: verify PREPARE + walk^k + UNPREPARE reproduces the Chebyshev
// moments <T_{2k}(H/alpha)> through the ancilla reflection observable for a
// non-degenerate (3-term, 2-ancilla) Hamiltonian.
TEST(QubitizationTester, checkWalkPowerReproducesChebyshevMoments) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  // h = 0.2 I + 0.5 X + 0.3 Z has the asymmetric spectrum 0.2 +/- lambda, so
  // the moments depend on both eigenvalues, not just |lambda|.
  cudaq::spin_op h = 0.2 * i(0) + 0.5 * x(0) + 0.3 * z(0);
  pauli_lcu encoding(h, 1);
  ASSERT_EQ(encoding.term_count(), 3u);
  ASSERT_EQ(encoding.num_ancilla(), 2u);
  const double alpha = encoding.normalization();
  EXPECT_NEAR(alpha, 1.0, 1e-12);

  const double lambda = std::sqrt(0.34);
  const double theta = std::atan2(0.5, 0.3);
  const double prep_angle = 0.7;
  // ry(prep_angle)|0> = cos(delta/2) |E+> + sin(delta/2) |E->.
  const double delta = prep_angle - theta;
  const double plus_weight = std::cos(0.5 * delta) * std::cos(0.5 * delta);
  const double minus_weight = std::sin(0.5 * delta) * std::sin(0.5 * delta);
  const double eigenvalue_plus = (0.2 + lambda) / alpha;
  const double eigenvalue_minus = (0.2 - lambda) / alpha;

  for (int k = 1; k <= 3; ++k) {
    auto moment_kernel = [&]() __qpu__ {
      cudaq::qvector<> anc(encoding.num_ancilla());
      cudaq::qvector<> sys(encoding.num_system());
      ry(prep_angle, sys[0]);
      encoding.prepare(anc);
      apply_qubitization_walk_power(anc, sys, encoding, k);
      encoding.unprepare(anc);
    };
    auto state = cudaq::get_state(moment_kernel);

    // The ancillas are allocated first (high bits), so the all-zero ancilla
    // subspace is the first 2^num_system amplitudes.
    double zero_probability = 0.0;
    for (std::size_t i = 0; i < (1ULL << encoding.num_system()); ++i)
      zero_probability += std::norm(state[i]);
    const double reflection_expectation = 2.0 * zero_probability - 1.0;

    // <R> after k walks is <T_{2k}(H/alpha)>; T_{2k} is even, so the walk's
    // -H/alpha sign convention drops out.
    const double expected_moment =
        plus_weight * chebyshev(2 * k, eigenvalue_plus) +
        minus_weight * chebyshev(2 * k, eigenvalue_minus);
    EXPECT_NEAR(reflection_expectation, expected_moment, 1e-10)
        << "moment mismatch at k = " << k;
  }
}

// Test purpose: verify adjoint walk powers invert forward walk powers for a
// non-degenerate multi-ancilla encoding.
TEST(QubitizationTester, checkAdjointWalkPowerInvertsWalkPower) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0) + 0.2 * z(0) * z(1);
  pauli_lcu encoding(h, 2);
  ASSERT_EQ(encoding.num_ancilla(), 2u);

  auto reference = [&]() __qpu__ {
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    ry(0.4, sys[0]);
    ry(1.1, sys[1]);
  };
  auto reference_state = cudaq::get_state(reference);
  const auto dimension =
      1ULL << (encoding.num_ancilla() + encoding.num_system());

  for (int power = 2; power <= 3; ++power) {
    auto round_trip = [&]() __qpu__ {
      cudaq::qvector<> anc(encoding.num_ancilla());
      cudaq::qvector<> sys(encoding.num_system());
      ry(0.4, sys[0]);
      ry(1.1, sys[1]);
      encoding.prepare(anc);
      apply_qubitization_walk_power(anc, sys, encoding, power);
      apply_adjoint_qubitization_walk_power(anc, sys, encoding, power);
      encoding.unprepare(anc);
    };
    expect_states_equal(cudaq::get_state(round_trip), reference_state,
                        dimension);
  }
}

// Test purpose: verify controlled adjoint walk powers invert controlled walk
// powers for a non-degenerate multi-ancilla encoding.
TEST(QubitizationTester, checkControlledAdjointWalkPowerInvertsWalkPower) {
  using namespace cudaq::spin;
  using namespace cudaq::algorithms;

  cudaq::spin_op h = 0.5 * x(0) + 0.3 * z(0) + 0.2 * z(0) * z(1);
  pauli_lcu encoding(h, 2);
  ASSERT_EQ(encoding.num_ancilla(), 2u);

  auto reference_on = [&]() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector<> anc(encoding.num_ancilla());
    cudaq::qvector<> sys(encoding.num_system());
    x(control);
    ry(0.4, sys[0]);
    ry(1.1, sys[1]);
  };
  auto reference_on_state = cudaq::get_state(reference_on);
  const auto dimension =
      1ULL << (1 + encoding.num_ancilla() + encoding.num_system());

  for (int power = 2; power <= 3; ++power) {
    auto round_trip_on = [&]() __qpu__ {
      cudaq::qubit control;
      cudaq::qvector<> anc(encoding.num_ancilla());
      cudaq::qvector<> sys(encoding.num_system());
      x(control);
      ry(0.4, sys[0]);
      ry(1.1, sys[1]);
      encoding.prepare(anc);
      apply_controlled_qubitization_walk_power(control, anc, sys, encoding,
                                               power);
      apply_controlled_adjoint_qubitization_walk_power(control, anc, sys,
                                                       encoding, power);
      encoding.unprepare(anc);
    };
    expect_states_equal(cudaq::get_state(round_trip_on), reference_on_state,
                        dimension);
  }
}
