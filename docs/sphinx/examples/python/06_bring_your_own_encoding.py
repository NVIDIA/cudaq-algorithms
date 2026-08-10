#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 6 — Bring your own block encoding.

`Walk` and `QSVT` are generic over the `BlockEncoding` protocol: any
object exposing the right members works, with no inheritance and no
changes to the primitives. This example implements the protocol from
scratch for the simplest nontrivial case -- a two-term LCU of a single
qubit, H = a*X + b*Z -- and shows the qubitization `Walk` measuring
correct Chebyshev moments on it.

The protocol (see docs and `block_encoding.py`): three sizes
(`num_system`, `num_ancilla`, `alpha`) plus kernel factories. The walk
step convention, mirrored below, is: SELECT, then reflect about the
PREPARE state. Get that right and everything else is inherited.

Run: python3 06_bring_your_own_encoding.py
"""

from __future__ import annotations

import math
import os

import cudaq
import numpy as np

from cudaq_algorithms import BlockEncoding, Walk
from cudaq_algorithms.common_kernels import reflect_about_zero


class TwoTermLCU:
    """Block encoding of H = a*X + b*Z (a, b > 0) on one system qubit.

    PREPARE puts the single ancilla into (sqrt(a), sqrt(b)) / sqrt(alpha);
    SELECT applies X when the ancilla is |0> and Z when it is |1>. Then
    <0|B^dag SELECT B|0> = (a*X + b*Z) / alpha = H / alpha.
    """

    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)
        self.num_system = 1
        self.num_ancilla = 1
        self.alpha = self.a + self.b
        # PREPARE angle: ry(theta)|0> = (cos, sin) = (sqrt(a), sqrt(b))/sqrt(alpha)
        self._theta = 2.0 * math.atan2(math.sqrt(self.b), math.sqrt(self.a))

    # -- sizes -----------------------------------------------------------
    # num_system, num_ancilla, alpha are plain attributes (set above).

    # -- kernel factories (data captured at factory time) ----------------
    def prepare_kernel(self):
        theta = self._theta

        @cudaq.kernel
        def prepare(ancilla: cudaq.qview):
            ry(theta, ancilla[0])

        return prepare

    def unprepare_kernel(self):
        theta = self._theta

        @cudaq.kernel
        def unprepare(ancilla: cudaq.qview):
            ry(-theta, ancilla[0])

        return unprepare

    def _select_kernel(self):

        @cudaq.kernel
        def select(ancilla: cudaq.qview, system: cudaq.qview):
            # |0><0| (x) X : X on system, controlled on ancilla == 0
            x(ancilla[0])
            x.ctrl(ancilla[0], system[0])
            x(ancilla[0])
            # |1><1| (x) Z : Z on system, controlled on ancilla == 1
            z.ctrl(ancilla[0], system[0])

        return select

    def walk_step_kernel(self):
        theta = self._theta
        select = self._select_kernel()

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            select(ancilla, system)  # SELECT
            ry(-theta, ancilla[0])  # reflect about PREPARE state:
            reflect_about_zero(ancilla)  #   B (I - 2|0><0|) B^dag
            ry(theta, ancilla[0])

        return walk_step

    def adjoint_walk_step_kernel(self):
        theta = self._theta
        select = self._select_kernel()

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            ry(-theta, ancilla[0])  # reflection first (self-adjoint),
            reflect_about_zero(ancilla)
            ry(theta, ancilla[0])
            select(ancilla, system)  # then SELECT (self-adjoint)

        return adjoint_walk_step

    def apply_kernel(self):
        theta = self._theta
        select = self._select_kernel()

        @cudaq.kernel
        def apply(ancilla: cudaq.qview, system: cudaq.qview):
            ry(theta, ancilla[0])
            select(ancilla, system)
            ry(-theta, ancilla[0])

        return apply

    # -- hooks left to the reader ---------------------------------------
    # The controlled variants (for QPE) follow the same pattern with the
    # control on qubit 0 of a combined register; select_observable enables
    # odd moments. Omitted here -- even moments below need none of them.
    def controlled_apply_kernel(self):
        raise NotImplementedError("exercise: controlled U_A for QPE")

    def controlled_walk_step_kernel(self):
        raise NotImplementedError("exercise: controlled W for QPE")

    def controlled_adjoint_walk_step_kernel(self):
        raise NotImplementedError("exercise: controlled W-dagger")

    def select_observable(self):
        raise NotImplementedError("exercise: enables odd Chebyshev moments")


def dense_moment(a, b, k, state):
    """Reference <T_k(H/alpha)> for H = a*X + b*Z."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    x = (a * X + b * Z) / (a + b)
    previous, current = np.eye(2, dtype=complex), x
    if k == 0:
        current = np.eye(2, dtype=complex)
    for _ in range(2, k + 1):
        previous, current = current, 2 * x @ current - previous
    return float(np.real(state.conj() @ current @ state))


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    encoding = TwoTermLCU(a=0.6, b=0.9)

    # It satisfies the protocol -- structurally, no base class involved.
    print(f"isinstance(encoding, BlockEncoding) : "
          f"{isinstance(encoding, BlockEncoding)}")
    print(f"num_system / num_ancilla / alpha    : "
          f"{encoding.num_system} / {encoding.num_ancilla} / {encoding.alpha}")

    # Walk consumes it with no changes. Measure even Chebyshev moments
    # (even orders use the geometry-derived reflection observable, so they
    # need no encoding-specific observable hook).
    walk = Walk(encoding)
    state = np.array([0.8, 0.6], dtype=np.complex128)  # a chosen |psi>

    print("\n k   Walk.moment    dense <T_k(H/alpha)>")
    ok = True
    for k in (0, 2, 4, 6):
        measured = walk.moment(state, k)
        reference = dense_moment(encoding.a, encoding.b, k, state)
        flag = "ok" if abs(measured - reference) < 1e-10 else "MISMATCH"
        ok = ok and flag == "ok"
        print(f" {k}   {measured:+.8f}    {reference:+.8f}   {flag}")

    assert ok
    print("\nOK — a from-scratch encoding drops into Walk unchanged. That is "
          "the extension point: bring an encoding, inherit the primitives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
