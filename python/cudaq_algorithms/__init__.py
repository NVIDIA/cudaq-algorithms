# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-Q Algorithms.

A pure-Python package: the fermion-to-qubit transforms
(:mod:`.fermion`), the state-preparation kernels and operator pools
(:mod:`.stateprep`), the quantum primitives (:mod:`.pauli_lcu`,
:mod:`.qubitization`, :mod:`.qsvt`, :mod:`.trotter`,
:mod:`.sim_utils`), and the classical double-factorization
preprocessing (:mod:`.double_factorization`, NumPy/SciPy with optional
CuPy GPU acceleration) are implemented as CUDA-Q Python kernels and
host-side helpers. The only runtime requirements are the ``cudaq``
Python package plus NumPy/SciPy.
"""


def _resolve_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    # The shipped wheel is CUDA-agnostic; the -cuNN names are fallbacks.
    for distribution in ("cudaq-algorithms", "cudaq-algorithms-cu12",
                         "cudaq-algorithms-cu13"):
        try:
            return f"CUDA-Q Algorithms {version(distribution)}"
        except PackageNotFoundError:
            continue
    return "CUDA-Q Algorithms (source)"


__version__ = _resolve_version()
del _resolve_version

from . import (block_encoding, chemistry, common_kernels, double_factorization,
               fermion, pauli_lcu, qsvt, qubitization, sim_utils, stateprep,
               trotter)
from .block_encoding import BlockEncoding
from .common_kernels import state_from
from .pauli_lcu import PauliLCU, select_observable
from .qsvt import (ADJOINT, FORWARD, PhaseSequence, QSVT,
                   recover_real_time_evolution)
from .qubitization import Walk, reflection_observable
from .trotter import Trotter, TrotterOrdering, TrotterResourceEstimate

# The composable device kernels keep their module namespaces
# (cudaq_algorithms.pauli_lcu.prepare, .select, .apply_phase_sequence, ...;
# cudaq_algorithms.common_kernels.reflect_about_zero, .signal_phase, ...):
# their names are too generic to export from the package root.
#
# A worked, full-scale implementation of the BlockEncoding protocol — the
# double-factorized encoding — ships as a runnable example at
# examples/bring_your_own_encoding/df_encoding.py (dense-validated by
# tests/python/test_df_encoding.py).
