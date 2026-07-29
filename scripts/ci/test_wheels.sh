#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -euo pipefail

python_version=${1:-}
platform=${2:-}
cuda_version=${3:-}
cudaq_version=${4:-}
algorithms_version=${5:-}

if [[ -z "$python_version" || -z "$platform" || -z "$cuda_version" || -z "$cudaq_version" || -z "$algorithms_version" ]]; then
    echo "Usage: $0 <python-version> <platform> <cuda-version> <cudaq-version|SKIP> <algorithms-version>" >&2
    exit 1
fi

python=python${python_version}
cuda_major=$(echo "$cuda_version" | cut -d . -f 1)

$python -m pip install --no-cache-dir pytest scipy numpy

if [[ "$cudaq_version" != "SKIP" && -d /cudaq-wheels ]]; then
    # Custom mode tests the wheel against the from-source (unreleased)
    # cuda-quantum; --no-deps bypasses the `cudaq` resolver (only needed for
    # released installs) since cuda-quantum + numpy + scipy are already
    # provided.
    $python -m pip install --find-links /cudaq-wheels "cuda-quantum-cu${cuda_major}==${cudaq_version}"
    $python -m pip install --no-deps --find-links /wheels "cudaq-algorithms==${algorithms_version}"
else
    # Help CUDA detection in CPU-only validation jobs.
    $python -m pip install --extra-index-url https://pypi.nvidia.com/ "cuda_toolkit[cudart]==${cuda_version}.*" || true
    # PyPI mode installs the wheel with full deps so `cudaq` resolves a
    # released cuda-quantum.
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
if ! $python -m pip show "cuda-quantum-cu${cuda_major}" >/dev/null 2>&1; then
    echo "::error Expected cuda-quantum-cu${cuda_major} to be installed." >&2
    exit 1
fi

$python -m pytest -q tests/python
