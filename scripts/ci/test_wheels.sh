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

$python -m pip install --no-cache-dir pytest scipy pyscf

find_links=(--find-links /wheels --find-links /metapackages)
if [[ -d /cudaq-wheels ]]; then
    find_links+=(--find-links /cudaq-wheels)
    if [[ "$cudaq_version" != "SKIP" ]]; then
        $python -m pip install "${find_links[@]}" "cuda-quantum-cu${cuda_major}==${cudaq_version}"
    fi
fi

# Help the metapackage detect CUDA major version in CPU-only validation jobs.
$python -m pip install --extra-index-url https://pypi.nvidia.com/ "cuda_toolkit[cudart]==${cuda_version}.*" || true

$python -m pip install "${find_links[@]}" "cudaq-algorithms==${algorithms_version}"
$python -c "import cudaq_algorithms"

package_installed=$($python -m pip list | awk '/cudaq-algorithms-cu/ {print $1; exit}')
package_expected=cudaq-algorithms-cu${cuda_major}
if [[ "$package_installed" != "$package_expected" ]]; then
    echo "::error Expected installation of $package_expected package, but got $package_installed." >&2
    exit 1
fi

$python -m pytest -q tests/python
