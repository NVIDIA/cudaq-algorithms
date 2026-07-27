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

# The package is pure Python: no compiler toolchain, CUDA-Q CMake
# package, or auditwheel repair is needed; --devdeps and --cudaq-prefix
# are accepted for interface compatibility but unused.

export SETUPTOOLS_SCM_PRETEND_VERSION=$wheels_version

if [[ ! -f "pyproject.toml.cu${cuda_major}" ]]; then
    echo "Error: pyproject.toml.cu${cuda_major} not found" >&2
    exit 1
fi

# `build` is also removed: a stale setuptools staging directory would be
# merged into the wheel.
rm -rf dist build _skbuild pyproject.toml
cp "pyproject.toml.cu${cuda_major}" pyproject.toml

echo "Building cudaq-algorithms-cu${cuda_major} $wheels_version (pure Python)"
$python -m pip install --no-cache-dir build
$python -m build --wheel

mkdir -p /wheels
cp dist/*.whl /wheels

ls -la /wheels
