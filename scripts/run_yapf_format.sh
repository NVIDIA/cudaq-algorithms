#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if ! command -v yapf >/dev/null; then
  echo "Error: yapf executable not found" >&2
  exit 1
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

mapfile -t files < <(git ls-files -- '*.py')
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No Python files found for yapf."
  exit 0
fi

yapf -i "${files[@]}"
