# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the Python test suite."""

import os

import cudaq
import pytest

# Simulation backend for tests that execute kernels. Honors CUDA-Q's standard
# selection variable (e.g. CUDAQ_DEFAULT_SIMULATOR=nvidia-fp64 for the GPU
# statevector simulator) and falls back to the CPU statevector simulator.
SIMULATION_TARGET = os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")


@pytest.fixture(autouse=True)
def simulation_target():
    cudaq.set_target(SIMULATION_TARGET)
    yield
    cudaq.reset_target()
