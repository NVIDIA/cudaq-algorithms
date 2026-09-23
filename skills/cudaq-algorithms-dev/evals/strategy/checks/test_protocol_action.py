# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private E03 numerical oracle, excluded from every worker source projection.

Run only in the frozen CUDA-Q 0.15.1 qpp-cpu environment. A captured worker
implementation must execute in a separate secret-free, network-disabled,
resource-bounded container. A missing runtime is an error, never a skip/pass.

Dense matrices and normalization below are literal references, independent of
the implementation's Pauli conversion, alpha, slicing, or state utilities.
The foreign object exposes the ORIGINAL structural contract and deliberately
has no encode_kernel. The controller also runs frozen original Walk/QSVT test
files separately, never tests supplied or modified by the worker.
"""

from importlib.metadata import version

import cudaq
import numpy as np
import pytest

from cudaq_algorithms import PauliLCU
from cudaq_algorithms import sim_utils as sim

ATOL = 1e-11
RTOL = 1e-11


@pytest.fixture(autouse=True)
def frozen_cpu_runtime():
    assert version(
        "cudaq"
    ) == "0.15.1", "independent checks require the frozen real CUDA-Q runtime"
    cudaq.set_target("qpp-cpu")
    assert "fp64" in str(
        cudaq.get_target()), "independent checks require double precision"
    yield
    cudaq.reset_target()


class ForeignEncoding:
    """An original-contract implementation with only minted circuit handles.

    No PauliLCU inheritance, underlying object, terms, kernel_args, or
    encode_kernel is available to the consumer under test.
    """

    __slots__ = ("num_system", "num_ancilla", "alpha", "_prepare",
                 "_unprepare", "_apply", "_controlled_apply", "_walk_step",
                 "_adjoint_walk_step", "_controlled_walk_step",
                 "_controlled_adjoint_walk_step", "_select_observable")

    def __init__(self, inner):
        self.num_system = inner.num_system
        self.num_ancilla = inner.num_ancilla
        self.alpha = inner.alpha
        self._prepare = inner.prepare_kernel()
        self._unprepare = inner.unprepare_kernel()
        self._apply = inner.apply_kernel()
        self._controlled_apply = inner.controlled_apply_kernel()
        self._walk_step = inner.walk_step_kernel()
        self._adjoint_walk_step = inner.adjoint_walk_step_kernel()
        self._controlled_walk_step = inner.controlled_walk_step_kernel()
        self._controlled_adjoint_walk_step = inner.controlled_adjoint_walk_step_kernel(
        )
        self._select_observable = inner.select_observable()

    def prepare_kernel(self):
        return self._prepare

    def unprepare_kernel(self):
        return self._unprepare

    def apply_kernel(self):
        return self._apply

    def controlled_apply_kernel(self):
        return self._controlled_apply

    def walk_step_kernel(self):
        return self._walk_step

    def adjoint_walk_step_kernel(self):
        return self._adjoint_walk_step

    def controlled_walk_step_kernel(self):
        return self._controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self):
        return self._controlled_adjoint_walk_step

    def select_observable(self):
        return self._select_observable


# Words index qubit zero first: e.g. IY = Y tensor I in dense ordering.
# Deliberately unequal coefficients and Y terms expose ordering and phase bugs.
CASES = [
    pytest.param({
        "I": 0.23,
        "X": -0.51,
        "Y": 0.37,
        "Z": 0.69
    }, [[0.92, -0.51 - 0.37j], [-0.51 + 0.37j, -0.46]],
                 1.80,
                 id="one-qubit-four-terms"),
    pytest.param({
        "II": 0.17,
        "XI": -0.31,
        "IY": 0.43,
        "ZZ": -0.59,
        "ZX": 0.23
    }, [[-0.42, -0.31, 0.23 - 0.43j, 0.0], [-0.31, 0.76, 0.0, -0.23 - 0.43j],
        [0.23 + 0.43j, 0.0, 0.76, -0.31], [0.0, -0.23 + 0.43j, -0.31, -0.42]],
                 1.73,
                 id="two-qubit-five-terms"),
    pytest.param({"Y": -0.71}, [[0.0, 0.71j], [-0.71j, 0.0]],
                 0.71,
                 id="one-qubit-negative-single-term"),
    pytest.param({"XZ": -0.53},
                 [[0.0, -0.53, 0.0, 0.0], [-0.53, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.53], [0.0, 0.0, 0.53, 0.0]],
                 0.53,
                 id="two-qubit-negative-single-term"),
]


def _ket(dimension, kind):
    if kind == "real":
        return np.array([0.6, 0.8]) if dimension == 2 else np.array(
            [1.0, -2.0, 3.0, 4.0]) / np.sqrt(30.0)
    if kind == "complex":
        return (np.array([1.0 + 2.0j, -2.0 + 0.5j]) /
                np.sqrt(9.25) if dimension == 2 else
                np.array([0.2 + 1.0j, -0.7 + 0.4j, 0.8 - 0.3j, 1.2 - 0.2j]) /
                np.sqrt(3.90))
    result = np.zeros(dimension, dtype=np.complex128)
    result[0 if kind == "first-basis" else -1] = 1.0
    return result


@pytest.mark.parametrize("terms,dense,alpha", CASES)
@pytest.mark.parametrize("state_kind",
                         ["real", "complex", "first-basis", "last-basis"])
@pytest.mark.parametrize("encoding_kind", ["builtin", "foreign"])
def test_action_matches_independent_dense_reference(terms, dense, alpha,
                                                    state_kind, encoding_kind):
    encoding = PauliLCU(terms)
    if encoding_kind == "foreign":
        encoding = ForeignEncoding(encoding)
        assert not hasattr(encoding, "encode_kernel")
        assert not hasattr(encoding, "kernel_args")
        assert not isinstance(encoding, PauliLCU)
    matrix = np.asarray(dense, dtype=np.complex128)
    ket = _ket(len(matrix), state_kind)
    assert encoding.alpha == pytest.approx(alpha, abs=1e-14)
    assert encoding.num_system == (1 if len(matrix) == 2 else 2)
    actual = np.asarray(sim.action(encoding, ket))
    assert actual.shape == ket.shape
    assert np.iscomplexobj(actual)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,
                               matrix @ ket / alpha,
                               atol=ATOL,
                               rtol=RTOL)
