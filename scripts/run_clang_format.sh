#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

show_help() {
  echo "Usage: $0 [-p /path/to/clang-format]"
}

clang_format_executable=${CLANG_FORMAT:-clang-format}

while getopts ":hp:" opt; do
  case "$opt" in
    h)
      show_help
      exit 0
      ;;
    p)
      clang_format_executable="$OPTARG"
      ;;
    \?)
      echo "Invalid command line option -$OPTARG" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

if ! command -v "$clang_format_executable" >/dev/null; then
  echo "Error: clang-format executable '$clang_format_executable' not found" >&2
  exit 1
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

mapfile -t files < <(git ls-files -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' '*.cu' '*.cuh')
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No C/C++ files found for clang-format."
  exit 0
fi

"$clang_format_executable" -i "${files[@]}"
