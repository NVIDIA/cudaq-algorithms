#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 5 — State preparation and injection.

Every primitive factory (Walk, QSVT, Trotter) takes a `state_prep`
argument: a `(qubits: qview)` CUDA-Q kernel that prepares the register.
Pass one and the factory returns a *zero-argument* circuit -- no
`cudaq.State` argument, no statevector crossing the API boundary -- that
you can sample directly
and hand to synthesis. This is the seam that makes the primitives
hardware-ready, and the seam an MPS/tensor-network state-prep compiler
would plug into.

Two things this shows:
  A. The injection contract: an injected walk is a zero-argument kernel,
     and its measured moment matches the data-path (cudaq.State) form.
  B. Any `(qubits: qview)` kernel qualifies -- including a Givens
     Slater-determinant prep produced from an orbital-coefficient matrix.

Run: python3 05_state_prep_and_injection.py
"""

from __future__ import annotations

import os

import cudaq
import numpy as np

from cudaq_algorithms import PauliLCU, Walk, stateprep


# A state-prep kernel is just a `(qubits: qview)` device kernel. This one
# prepares |...01> -- one electron in the lowest orbital (a 2-qubit
# Hartree-Fock reference).
@cudaq.kernel
def hartree_fock_prep(qubits: cudaq.qview):
    x(qubits[0])


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    encoding = PauliLCU({"ZZ": 0.5, "XI": 0.3, "IX": 0.3, "ZI": 0.2})
    walk = Walk(encoding)

    # -- A. Injection contract -------------------------------------------
    # With state_prep, walk.kernel returns a ZERO-ARGUMENT circuit: prepare
    # the reference, then apply W^3. Sample it like any kernel.
    injected = walk.kernel(power=3, state_prep=hartree_fock_prep)
    counts = cudaq.sample(injected, shots_count=2000)
    print("A. injected walk is a zero-argument, directly sampleable circuit")
    print(f"   sampled {len(counts)} bitstrings, e.g. "
          f"{dict(list(counts.items())[:3])}")

    # The observable path also accepts state_prep. It must agree with the
    # data path, where we hand in the same reference as a statevector.
    reference = np.zeros(1 << encoding.num_system, dtype=np.complex128)
    reference[1] = 1.0  # |01>, matching x(qubits[0])
    moment_injected = walk.moment(None, 3, state_prep=hartree_fock_prep)
    moment_data = walk.moment(reference, 3)
    print(f"   moment via injection : {moment_injected:+.10f}")
    print(f"   moment via data path : {moment_data:+.10f}")
    assert abs(moment_injected - moment_data) < 1e-10
    print("   -> identical: injection changes how the state enters, not the "
          "physics")

    # -- B. Any (qubits: qview) kernel qualifies -------------------------
    # A Givens Slater determinant of an orbital-coefficient matrix Q is
    # also just a (qubits: qview) kernel -- built from Q, injectable the
    # same way. Its computational-basis amplitudes are the minors of Q.
    orbital_coefficients = np.array([[0.6, 0.0], [0.8, 0.0], [0.0, 0.6],
                                     [0.0, 0.8]])
    schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
    slater_prep = stateprep.slater_determinant_kernel(schedule)

    @cudaq.kernel
    def prepare_slater():
        qubits = cudaq.qvector(4)
        slater_prep(qubits)

    slater_counts = cudaq.sample(prepare_slater, shots_count=4000)
    print("\nB. a Givens Slater-determinant prep is the same kind of kernel")
    print(f"   4 spin-orbitals, 2 electrons -> occupied-pair bitstrings:")
    for bits, n in sorted(slater_counts.items(), key=lambda kv: -kv[1])[:4]:
        print(f"     |{bits}>  {n/4000:.3f}")
    print("   (these frequencies are |det Q[S,:]|^2 -- an entangled "
          "reference, injectable exactly like the HF one)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
