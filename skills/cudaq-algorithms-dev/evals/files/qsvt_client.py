# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application code written against the recorded QSVT contract."""

from cudaq_algorithms import QSVT


def build_qsp_kernel(transformer: QSVT, phases):
    """Build a QSVT kernel from raw QSPPACK phases."""
    return transformer.kernel(phases, convention="qsp")
