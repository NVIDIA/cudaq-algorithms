# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_import():
    import cudaq_algorithms

    assert "CUDA-Q Algorithms" in cudaq_algorithms.__version__


# test_import above also passes on the "(source)" fallback, so an installed
# wheel that cannot resolve its own name goes unnoticed. Pin to metadata.
def test_version_reports_the_installed_distribution():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("cudaq-algorithms")
    except PackageNotFoundError:
        # A source tree has no metadata; "(source)" is honest there.
        pytest.skip("cudaq-algorithms is not installed as a distribution")

    import cudaq_algorithms

    # Equality, not containment: the "(source)" fallback must not pass.
    assert cudaq_algorithms.__version__ == f"CUDA-Q Algorithms {installed}"
