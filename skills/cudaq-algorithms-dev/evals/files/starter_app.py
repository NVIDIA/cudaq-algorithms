# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cudaq_algorithms import PauliLCU, Walk

HAMILTONIAN = {"Z": 0.75, "X": -0.25}


def third_moment(ket):
    """Return the third Chebyshev moment for H / alpha."""
    encoding = PauliLCU(HAMILTONIAN)
    walk = Walk(encoding)
    # Incorrect starter: this returns a compiled kernel, not a classical
    # moment, and asks for T_3(-H/alpha) through a circuit block.
    return walk.kernel(power=3)
