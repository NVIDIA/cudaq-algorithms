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
    echo "  -h, --help           Show this help message"
    echo "  -i, --install        Install after build/test"
    echo "  --cudaq-prefix PATH  CUDA-Q install prefix"
    echo "                       Defaults to /usr/local/cudaq when present, otherwise \$HOME/.cudaq"
    echo "  --cudaq-dir PATH     CUDA-Q CMake package directory"
    echo "                       Defaults to <cudaq-prefix>/lib/cmake/cudaq"
    echo "  --build-dir PATH     Build directory (default: build)"
    echo "  --install-prefix PATH"
    echo "                       Install prefix (default: <cudaq-prefix>)"
}

install=0
build_dir="build"

if [[ -d /usr/local/cudaq/lib/cmake/cudaq ]]; then
    cudaq_prefix="/usr/local/cudaq"
else
    cudaq_prefix="$HOME/.cudaq"
fi

while (( $# > 0 )); do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--install)
            install=1
            shift
            ;;
        --cudaq-prefix)
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                cudaq_prefix="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --cudaq-dir)
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                cudaq_dir="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --build-dir)
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                build_dir="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --install-prefix)
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                install_prefix="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

cudaq_dir=${cudaq_dir:-"$cudaq_prefix/lib/cmake/cudaq"}
install_prefix=${install_prefix:-"$cudaq_prefix"}
repo_root=$(git rev-parse --show-toplevel)

if [[ "$build_dir" = /* ]]; then
    build_path="$build_dir"
else
    build_path="$repo_root/$build_dir"
fi

cd "$repo_root"

cmake -S . -B "$build_path" \
  -DCUDAQ_DIR="$cudaq_dir" \
  -DCMAKE_INSTALL_PREFIX="$install_prefix" \
  -DCUDAQ_ALGORITHMS_INCLUDE_TESTS=ON \
  -DCUDAQ_ALGORITHMS_BINDINGS_PYTHON=ON

cmake --build "$build_path" -j
ctest --test-dir "$build_path" --output-on-failure

PYTHONPATH="$build_path/python:$cudaq_prefix:${PYTHONPATH:-}" \
  python3 -m pytest -q tests/python

if [[ "$install" -eq 1 ]]; then
  cmake --build "$build_path" --target install
fi
