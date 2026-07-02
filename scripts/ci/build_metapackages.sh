#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -euo pipefail

# Relies on this script being stored at scripts/ci, two levels below the repo root
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>" >&2
    exit 1
fi

version=$1
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+ ]]; then
    echo "Error: Version $version is not a valid Python package version" >&2
    exit 1
fi

metapackage_dir=$repo_root/python/metapackages
cd "$metapackage_dir"

rm -rf dist *.egg-info _version.txt pyproject.toml.cu12 pyproject.toml.cu13
cp "$repo_root"/pyproject.toml.cu12 .
cp "$repo_root"/pyproject.toml.cu13 .
echo "$version" > _version.txt

CUDAQ_META_WHEEL_BUILD=1 python3 -m build . --sdist

expected_file="dist/cudaq_algorithms-${version}.tar.gz"
if [[ ! -f "$expected_file" ]]; then
    echo "Error: Expected file $expected_file not found" >&2
    exit 1
fi

if ! tar -tzf "$expected_file" >/dev/null; then
    echo "Error: Invalid tarball created: $expected_file" >&2
    exit 1
fi

rm pyproject.toml.cu12 pyproject.toml.cu13 _version.txt
