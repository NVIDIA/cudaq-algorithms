# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_import():
    import cudaq_algorithms

    assert "CUDA-Q Algorithms" in cudaq_algorithms.__version__
