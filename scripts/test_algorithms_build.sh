#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the Python test suite against the source tree. The package is pure
# Python; the only external requirement is a CUDA-Q install providing the
# `cudaq` Python package.

set -euo pipefail

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  --cudaq-prefix PATH  CUDA-Q install prefix"
    echo "                       Defaults to /usr/local/cudaq when present, otherwise \$HOME/.cudaq"
    echo "  --pip-cudaq          Use the pip-installed cudaq package (no install"
    echo "                       prefix on PYTHONPATH; a container-baked /usr/local/cudaq"
    echo "                       would otherwise shadow it)"
}

if [[ -d /usr/local/cudaq ]]; then
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
        --pip-cudaq)
            cudaq_prefix=""
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
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

python=${PYTHON:-python3}

if [[ -n "$cudaq_prefix" ]]; then
    pythonpath="$repo_root/python:$cudaq_prefix:${PYTHONPATH:-}"
else
    pythonpath="$repo_root/python:${PYTHONPATH:-}"
fi

PYTHONPATH="$pythonpath" "$python" -m pytest -q tests/python
