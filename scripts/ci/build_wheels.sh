#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -euo pipefail

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build-type      Build type (default: Release)"
    echo "  --cuda-version    CUDA version or major version to build for (default: 12)"
    echo "  --cudaq-prefix    CUDA-Q install prefix (default: \$HOME/.cudaq)"
    echo "  --python-version  Python version to build wheel for (default: 3.11)"
    echo "  --devdeps         Build wheels suitable for internal testing"
    echo "  --version         Version of wheels to produce (default: 0.0.0)"
}

build_type=Release
cuda_version=12
cudaq_prefix="$HOME/.cudaq"
python_version=3.11
devdeps=false
wheels_version=0.0.0

while (( $# > 0 )); do
    case "$1" in
        --build-type)
            build_type="$2"
            shift 2
            ;;
        --cuda-version)
            cuda_version="$2"
            shift 2
            ;;
        --cudaq-prefix)
            cudaq_prefix="$2"
            shift 2
            ;;
        --python-version)
            python_version="$2"
            shift 2
            ;;
        --devdeps)
            devdeps=true
            shift
            ;;
        --version)
            wheels_version="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1" >&2
            show_help
            exit 1
            ;;
    esac
done

cuda_major=$(echo "$cuda_version" | cut -d . -f 1)
if [[ "$cuda_major" != "12" && "$cuda_major" != "13" ]]; then
    echo "Error: unsupported CUDA major version '$cuda_major'" >&2
    exit 1
fi

python=python${python_version}
arch=$(uname -m)
plat_args=()

if $devdeps; then
    if [[ "$arch" == "x86_64" ]]; then
        plat_args=(--plat manylinux_2_34_x86_64)
    elif [[ "$arch" == "aarch64" ]]; then
        plat_args=(--plat manylinux_2_34_aarch64)
    fi
elif [[ -f /opt/rh/gcc-toolset-12/enable ]]; then
    source /opt/rh/gcc-toolset-12/enable
fi

export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export SETUPTOOLS_SCM_PRETEND_VERSION=$wheels_version
export CUDAQ_ALGORITHMS_VERSION=$wheels_version

if [[ ! -d "$cudaq_prefix/lib/cmake/cudaq" ]]; then
    echo "Error: CUDA-Q CMake package not found at $cudaq_prefix/lib/cmake/cudaq" >&2
    exit 1
fi

if [[ ! -f "pyproject.toml.cu${cuda_major}" ]]; then
    echo "Error: pyproject.toml.cu${cuda_major} not found" >&2
    exit 1
fi

rm -rf dist _skbuild pyproject.toml
cp "pyproject.toml.cu${cuda_major}" pyproject.toml

skbuild_args="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq;-DCMAKE_BUILD_TYPE=$build_type"
toolchain_dir="/opt/rh/gcc-toolset-12/root/usr/lib/gcc/${arch}-redhat-linux/12/"
if ! $devdeps && [[ -d "$toolchain_dir" ]]; then
    skbuild_args="$skbuild_args;-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=$toolchain_dir"
fi
export SKBUILD_CMAKE_ARGS=$skbuild_args

echo "Building cudaq-algorithms-cu${cuda_major} $wheels_version for Python $python_version"
$python -m pip install --no-cache-dir build auditwheel
$python -m build --wheel

exclude_args=()
if [[ -d "$cudaq_prefix/lib" ]]; then
    while IFS= read -r lib; do
        exclude_args+=(--exclude "$lib")
    done < <(find "$cudaq_prefix/lib" -name "*.so" -printf "%P\n" | sort)
fi

mkdir -p /wheels
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/*.whl \
    "${exclude_args[@]}" \
    --wheel-dir /wheels \
    "${plat_args[@]}"

ls -la /wheels
