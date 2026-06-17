# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq_algorithms as algorithms


def test_required_chebyshev_moments():
    assert algorithms.krylov.required_chebyshev_moments(0) == 0
    assert algorithms.krylov.required_chebyshev_moments(1) == 2
    assert algorithms.krylov.required_chebyshev_moments(4) == 8


def test_build_chebyshev_matrices():
    matrices = algorithms.krylov.build_chebyshev_matrices(
        [1.0, 0.2, -0.1, 0.05], 2)

    assert matrices.dimension == 2
    assert np.allclose(matrices.overlap_matrix(), [[1.0, 0.2], [0.2, 0.45]])
    assert np.allclose(matrices.hamiltonian_matrix(),
                       [[0.2, 0.45], [0.45, 0.1625]])
    assert matrices.overlap_data == pytest.approx([1.0, 0.2, 0.2, 0.45])
    assert matrices.hamiltonian_data == pytest.approx(
        [0.2, 0.45, 0.45, 0.1625])


def test_build_chebyshev_matrices_validation():
    with pytest.raises(RuntimeError):
        algorithms.krylov.build_chebyshev_matrices([1.0, 0.0], 0)
    with pytest.raises(RuntimeError):
        algorithms.krylov.build_chebyshev_matrices([1.0, 0.0, 0.0], 2)
