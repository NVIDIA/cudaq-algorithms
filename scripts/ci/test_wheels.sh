#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

python_version=${1:-}
platform=${2:-}
cuda_version=${3:-}
cudaq_version=${4:-}
algorithms_version=${5:-}
device=${6:-gpu}

if [[ -z "$python_version" || -z "$platform" || -z "$cuda_version" || -z "$cudaq_version" || -z "$algorithms_version" ]]; then
    echo "Usage: $0 <python-version> <platform> <cuda-version> <cudaq-version|SKIP> <algorithms-version> [cpu|gpu]" >&2
    exit 1
fi

python=python${python_version}
cuda_major=$(echo "$cuda_version" | cut -d . -f 1)

$python -m pip install --no-cache-dir pytest scipy numpy pyscf qsppack

if [[ "$cudaq_version" != "SKIP" && -d /cudaq-wheels ]]; then
    # Custom mode tests the wheel against the from-source (unreleased)
    # cuda-quantum; --no-deps bypasses the `cudaq` resolver (only needed for
    # released installs) since cuda-quantum + numpy + scipy are already
    # provided.
    $python -m pip install --find-links /cudaq-wheels "cuda-quantum-cu${cuda_major}==${cudaq_version}"
    $python -m pip install --no-deps --find-links /wheels "cudaq-algorithms==${algorithms_version}"
else
    # PyPI mode installs the wheel with full deps so `cudaq` resolves a
    # released cuda-quantum. (A cudart pre-install used to nudge the
    # resolver toward the lane's CUDA major on CPU hosts; cudaq >= 0.15.1
    # ignores it and picks its own default when no driver is present, so
    # the nudge is gone and the CPU-lane check below accepts whichever
    # variant the resolver chose.)
    $python -m pip install --find-links /wheels --extra-index-url https://pypi.nvidia.com/ "cudaq-algorithms==${algorithms_version}"
fi

$python -c "import cudaq_algorithms"

# Use `pip show` (exits 0/1) rather than piping `pip list` into grep -q: a
# quitting grep closes the pipe, pip takes SIGPIPE, and pipefail would turn a
# successful match into a false "not installed".
if ! $python -m pip show cudaq-algorithms >/dev/null 2>&1; then
    echo "::error cudaq-algorithms is not installed." >&2
    exit 1
fi
# In Custom mode we installed a specific cuda-quantum-cuNN; on GPU lanes
# the resolver's detection must match the host's CUDA major. Both get the
# exact check. On CPU lanes in PyPI mode there is no driver to detect, and
# which variant the `cudaq` metapackage defaults to is its policy, not this
# wheel's contract (0.15.1 switched the no-driver default to the newest
# CUDA): require exactly one cuda-quantum-cuNN, whichever it is.
if [[ "$cudaq_version" != "SKIP" || "$device" == "gpu" ]]; then
    if ! $python -m pip show "cuda-quantum-cu${cuda_major}" >/dev/null 2>&1; then
        echo "::error Expected cuda-quantum-cu${cuda_major} to be installed." >&2
        exit 1
    fi
else
    resolved=$($python -m pip list --format=freeze 2>/dev/null | grep -i '^cuda-quantum-cu' | cut -d= -f1)
    variant_count=$(echo "$resolved" | grep -c . || true)
    if [[ "$variant_count" != "1" ]]; then
        echo "::error Expected exactly one cuda-quantum-cuNN, found: ${resolved:-none}." >&2
        exit 1
    fi
    echo "cudaq resolved ${resolved} on the CPU lane."
fi

$python -m pytest -q tests/python
