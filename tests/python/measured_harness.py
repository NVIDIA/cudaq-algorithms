# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared compiler harness for the measured-primitive tests."""

import cudaq


def resources_of(kernel, *widths):
    """Compile a harness allocating ``widths`` fresh registers in order,
    call ``kernel`` on them and return the ``cudaq.estimate_resources``
    result (the measured tests read Toffoli, ``mz`` and conditional-CZ
    counts from it). Three widths mean ``(address, ladder, output)``; four
    mean a leading one-qubit control view."""
    padded = list(widths) + [1] * (4 - len(widths))
    w0, w1, w2, w3 = padded  # scalars only: kernels cannot capture lists

    if len(widths) == 3:

        @cudaq.kernel
        def harness():
            r0 = cudaq.qvector(w0)
            r1 = cudaq.qvector(w1)
            r2 = cudaq.qvector(w2)
            kernel(r0, r1, r2)
    else:

        @cudaq.kernel
        def harness():
            r0 = cudaq.qvector(w0)
            r1 = cudaq.qvector(w1)
            r2 = cudaq.qvector(w2)
            r3 = cudaq.qvector(w3)
            kernel(r0, r1, r2, r3)

    return cudaq.estimate_resources(harness)
