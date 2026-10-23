# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the Python test suite."""

import os
import re

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


def cudaq_version() -> tuple:
    """The running CUDA-Q as ``(major, minor, patch)``."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", cudaq.__version__)
    if match is None:
        # Fail loudly rather than fall back to (0, 0, 0), which would skip the
        # version-gated tests silently and permanently.
        raise RuntimeError(f"could not parse a version from cudaq.__version__="
                           f"{cudaq.__version__!r}")
    return tuple(int(part) for part in match.groups())


# ``cudaq.control`` over a kernel that calls other kernels is supported from
# CUDA-Q 0.16.0, the package minimum. Tests marked ``controlled_composition``
# pin that behaviour: they skip only on an older CUDA-Q and FAIL on a 0.16.x
# that regresses it (a capability probe would have skipped there instead).
CONTROLLED_COMPOSITION_OK = cudaq_version() >= (0, 16, 0)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "controlled_composition: cudaq.control over a kernel that "
        "calls other kernels (CUDA-Q >= 0.16.0)")


def pytest_collection_modifyitems(config, items):
    if CONTROLLED_COMPOSITION_OK:
        return
    skip = pytest.mark.skip(
        reason="cudaq.control over composed kernels needs CUDA-Q >= 0.16.0 "
        f"(running {cudaq.__version__.split('(')[0].strip()})")
    for item in items:
        if "controlled_composition" in item.keywords:
            item.add_marker(skip)
