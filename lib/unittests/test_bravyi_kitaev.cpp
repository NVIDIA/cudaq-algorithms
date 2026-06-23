/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/fermion/bravyi_kitaev.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

namespace fermion = cudaq::algorithms::fermion;

namespace {

using complex = std::complex<double>;

std::size_t one_body_index(std::size_t n, std::size_t p, std::size_t q) {
  return p * n + q;
}

std::size_t two_body_index(std::size_t n, std::size_t p, std::size_t q,
                           std::size_t r, std::size_t s) {
  return ((p * n + q) * n + r) * n + s;
}

void expect_spin_ops_near(const cudaq::spin_op &result,
                          const cudaq::spin_op &expected,
                          double tolerance = 1e-4) {
  auto residuals = result - expected;
  for (const auto &term : residuals)
    EXPECT_NEAR(std::abs(term.evaluate_coefficient()), 0.0, tolerance);
}

} // namespace

TEST(BravyiKitaev, H2Hamiltonian) {
  using namespace cudaq::spin;

  constexpr std::size_t n = 4;
  std::vector<complex> one_body(n * n);
  std::vector<complex> two_body(n * n * n * n);

  auto hpq = [&](std::size_t p, std::size_t q) -> complex & {
    return one_body[one_body_index(n, p, q)];
  };
  auto hpqrs = [&](std::size_t p, std::size_t q, std::size_t r,
                   std::size_t s) -> complex & {
    return two_body[two_body_index(n, p, q, r, s)];
  };

  const double scalar_offset = 0.7080240981000804;
  hpq(0, 0) = -1.2488;
  hpq(1, 1) = -1.2488;
  hpq(2, 2) = -.47967;
  hpq(3, 3) = -.47967;
  hpqrs(0, 0, 0, 0) = 0.3366719725032414;
  hpqrs(0, 0, 2, 2) = 0.0908126657382825;
  hpqrs(0, 1, 1, 0) = 0.3366719725032414;
  hpqrs(0, 1, 3, 2) = 0.0908126657382825;
  hpqrs(0, 2, 0, 2) = 0.09081266573828267;
  hpqrs(0, 2, 2, 0) = 0.33121364716348484;
  hpqrs(0, 3, 1, 2) = 0.09081266573828267;
  hpqrs(0, 3, 3, 0) = 0.33121364716348484;
  hpqrs(1, 0, 0, 1) = 0.3366719725032414;
  hpqrs(1, 0, 2, 3) = 0.0908126657382825;
  hpqrs(1, 1, 1, 1) = 0.3366719725032414;
  hpqrs(1, 1, 3, 3) = 0.0908126657382825;
  hpqrs(1, 2, 0, 3) = 0.09081266573828267;
  hpqrs(1, 2, 2, 1) = 0.33121364716348484;
  hpqrs(1, 3, 1, 3) = 0.09081266573828267;
  hpqrs(1, 3, 3, 1) = 0.33121364716348484;
  hpqrs(2, 0, 0, 2) = 0.3312136471634851;
  hpqrs(2, 0, 2, 0) = 0.09081266573828246;
  hpqrs(2, 1, 1, 2) = 0.3312136471634851;
  hpqrs(2, 1, 3, 0) = 0.09081266573828246;
  hpqrs(2, 2, 0, 0) = 0.09081266573828264;
  hpqrs(2, 2, 2, 2) = 0.34814578499360427;
  hpqrs(2, 3, 1, 0) = 0.09081266573828264;
  hpqrs(2, 3, 3, 2) = 0.34814578499360427;
  hpqrs(3, 0, 0, 3) = 0.3312136471634851;
  hpqrs(3, 0, 2, 1) = 0.09081266573828246;
  hpqrs(3, 1, 1, 3) = 0.3312136471634851;
  hpqrs(3, 1, 3, 1) = 0.09081266573828246;
  hpqrs(3, 2, 0, 1) = 0.09081266573828264;
  hpqrs(3, 2, 2, 3) = 0.34814578499360427;
  hpqrs(3, 3, 1, 1) = 0.09081266573828264;
  hpqrs(3, 3, 3, 3) = 0.34814578499360427;

  auto result = fermion::bravyi_kitaev(one_body, two_body, n, scalar_offset);
  cudaq::spin_op expected =
      -0.1064770114930045 + 0.04540633286914125 * x(0) * z(1) * x(2) +
      0.04540633286914125 * x(0) * z(1) * x(2) * z(3) +
      0.04540633286914125 * y(0) * z(1) * y(2) +
      0.04540633286914125 * y(0) * z(1) * y(2) * z(3) +
      0.17028010135220506 * z(0) + 0.1702801013522051 * z(0) * z(1) +
      0.16560682358174256 * z(0) * z(1) * z(2) +
      0.16560682358174256 * z(0) * z(1) * z(2) * z(3) +
      0.12020049071260128 * z(0) * z(2) +
      0.12020049071260128 * z(0) * z(2) * z(3) + 0.1683359862516207 * z(1) -
      0.22004130022421792 * z(1) * z(2) * z(3) +
      0.17407289249680227 * z(1) * z(3) - 0.22004130022421792 * z(2);

  expect_spin_ops_near(result, expected);
}

TEST(BravyiKitaev, SeeleyRichardLoveCases) {
  using namespace cudaq::spin;

  expect_spin_ops_near(fermion::seeley_richard_love(2, 2, 4.0, 20),
                       (complex(-2.0, 0.0) * i(0) * i(1) * z(2) +
                        complex(2.0, 0.0) * i(0) * i(1) * i(2))
                           .canonicalize());

  expect_spin_ops_near(
      fermion::seeley_richard_love(2, 6, 4.0, 20),
      (complex(1.0, 0.0) * i(0) * z(1) * x(2) * y(3) * i(4) * z(5) * y(6) +
       complex(-1.0, 0.0) * i(0) * z(1) * y(2) * y(3) * i(4) * z(5) * x(6) +
       complex(0.0, -1.0) * i(0) * z(1) * x(2) * y(3) * i(4) * z(5) * x(6) +
       complex(0.0, -1.0) * i(0) * z(1) * y(2) * y(3) * i(4) * z(5) * y(6))
          .canonicalize());

  expect_spin_ops_near(fermion::seeley_richard_love(5, 2, 4.0, 20),
                       (complex(-1.0, 0.0) * z(1) * y(2) * y(3) * z(4) * x(5) +
                        complex(0.0, 1.0) * z(1) * x(2) * y(3) * z(4) * x(5) +
                        complex(1.0, 0.0) * z(1) * x(2) * y(3) * y(5) +
                        complex(0.0, 1.0) * z(1) * y(2) * y(3) * y(5))
                           .canonicalize());

  expect_spin_ops_near(fermion::seeley_richard_love(1, 2, 4.0, 20),
                       (complex(1.0, 0.0) * z(0) * y(1) * y(2) +
                        complex(0.0, -1.0) * z(0) * y(1) * x(2) +
                        complex(1.0, 0.0) * i(0) * x(1) * x(2) +
                        complex(0.0, 1.0) * i(0) * x(1) * y(2))
                           .canonicalize());

  expect_spin_ops_near(fermion::seeley_richard_love(0, 7, 4.0, 20),
                       (complex(-1.0, 0.0) * y(0) * x(1) * y(3) * z(5) * z(6) +
                        complex(0.0, -1.0) * x(0) * x(1) * y(3) * z(5) * z(6) +
                        complex(0.0, 1.0) * y(0) * x(1) * x(3) * z(7) +
                        complex(-1.0, 0.0) * x(0) * x(1) * x(3) * z(7))
                           .canonicalize());
}
