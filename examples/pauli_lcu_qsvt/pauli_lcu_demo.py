#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Walkthrough of the PauliLCU block-encoding API.

Run with:  python3 pauli_lcu_demo.py
"""

import os

import numpy as np

import cudaq
from cudaq import spin

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import PauliLCU, prepare, select, state_from, unprepare


def banner(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main():
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    banner("1. Construct from a plain dict (word -> coefficient)")
    enc = PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
    print(enc)
    print(f"terms:         {enc.terms}")
    print(f"alpha:         {enc.alpha}")
    print(f"constant term: {enc.constant_term}")

    banner("2. Or from a cudaq.SpinOperator — same encoding")
    h = (0.70 * spin.z(0) - 0.43 * spin.z(1) + 0.19 * spin.x(0) * spin.x(1) +
         0.11 * spin.y(0) * spin.z(1))
    print(PauliLCU(h, num_qubits=2))

    banner("3. One call: (H/alpha) |psi> via encode + postselect")
    rng = np.random.default_rng(7)
    psi = rng.normal(size=4).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    good = sim.action(enc, psi)
    print(f"|| action(psi) ||           = {np.linalg.norm(good):.6f}")
    print(f"success probability         = {np.linalg.norm(good)**2:.6f}")

    banner("4. Kernel factories for sampling/observing workflows")
    kernel = enc.encode_kernel()
    state = cudaq.get_state(kernel, state_from(psi))
    print(f"full statevector dimension  = {len(np.asarray(state))}")
    print(f"good-subspace dimension     = "
          f"{len(sim.good_subspace(enc, state))}")

    banner("5. Qubitization walks (Chebyshev moments)")
    moment_enc = PauliLCU({"I": 0.2, "X": 0.5, "Z": 0.3})
    ket = np.array([np.cos(0.35), np.sin(0.35)], dtype=np.complex128)
    for k in (1, 2, 3):
        walked = cudaq.get_state(moment_enc.walk_kernel(power=k),
                                 state_from(ket))
        p0 = float(np.sum(np.abs(sim.good_subspace(moment_enc, walked))**2))
        print(f"<T_{2*k}(H/alpha)> from the circuit = {2 * p0 - 1:+.10f}")

    banner("6. Compose the module-level kernels inside your own kernel")
    angles, controls, ops, lengths, signs = enc.kernel_args
    n_anc = enc.num_ancilla

    @cudaq.kernel
    def custom(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        prepare(ancilla, angles)
        select(ancilla, system, controls, ops, lengths, signs)
        unprepare(ancilla, angles)

    manual = sim.good_subspace(enc, cudaq.get_state(custom, state_from(psi)))
    print(f"manual composition matches action(): "
          f"{np.allclose(manual, good, atol=1e-12)}")

    banner("7. Single-term encodings (normalized to one ancilla)")
    negative_single = PauliLCU({"XZ": -0.5})
    positive_single = PauliLCU({"XZ": +0.5})
    print(negative_single)
    opposite = np.allclose(sim.action(negative_single, psi),
                           -sim.action(positive_single, psi),
                           atol=1e-12)
    print(f"-0.5*XZ encodes the opposite state of +0.5*XZ: {opposite}")
    print(f"action norm = "
          f"{np.linalg.norm(sim.action(negative_single, psi)):.6f} "
          f"(single unitary Pauli word: exactly 1)")

    print()


if __name__ == "__main__":
    main()
