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


def cudaq_version() -> tuple | None:
    """The running CUDA-Q as ``(major, minor, patch)``, or ``None`` when
    ``cudaq.__version__`` carries no parseable version."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", cudaq.__version__)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


# ``cudaq.control`` over a kernel that calls other kernels is supported from
# CUDA-Q 0.16.0, the package minimum. Tests marked ``controlled_composition``
# pin that behaviour: they skip only on an older CUDA-Q and FAIL on a 0.16.x
# that regresses it (a capability probe would have skipped there instead).
# An unparseable version fails those tests too, and only those: nothing is
# decided at import time, so the rest of the suite still collects and runs.
_VERSION_ERROR = (None if cudaq_version() is not None else
                  "could not parse a version from cudaq.__version__="
                  f"{cudaq.__version__!r}; the controlled_composition gate "
                  "cannot decide whether this CUDA-Q is supported")
CONTROLLED_COMPOSITION_OK = (_VERSION_ERROR is None
                             and cudaq_version() >= (0, 16, 0))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "controlled_composition: cudaq.control over a kernel that "
        "calls other kernels (CUDA-Q >= 0.16.0)")


def pytest_collection_modifyitems(config, items):
    if CONTROLLED_COMPOSITION_OK or _VERSION_ERROR is not None:
        return
    skip = pytest.mark.skip(
        reason="cudaq.control over composed kernels needs CUDA-Q >= 0.16.0 "
        f"(running {cudaq.__version__.split('(')[0].strip()})")
    for item in items:
        if "controlled_composition" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _controlled_composition_gate(request):
    # Fail the gated tests, and only them, when the version is unparseable.
    if (_VERSION_ERROR is not None
            and request.node.get_closest_marker("controlled_composition")):
        pytest.fail(_VERSION_ERROR)
