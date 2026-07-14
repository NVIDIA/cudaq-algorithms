# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q Algorithms.

The package has two kinds of components:

- Compiled bindings (``_pycudaq_algorithms``): C++-backed APIs such as the
  state-preparation utilities. These require the native extension built
  against CUDA-Q.
- Pure-Python modules (:mod:`.fermion`, :mod:`.pauli_lcu`,
  :mod:`.qubitization`, :mod:`.qsvt`, :mod:`.trotter`,
  :mod:`.sim_utils`): the fermion-to-qubit transforms and the quantum
  primitives implemented as CUDA-Q Python kernels. These only require
  the ``cudaq`` Python package.
- Pure-Python classical preprocessing (:mod:`.double_factorization`):
  chemistry-oriented numerics on NumPy/SciPy with optional CuPy GPU
  acceleration. The subpackage itself imports neither ``cudaq`` nor the
  compiled extension, but it is reached through this package, whose
  import does load ``cudaq`` alongside it.

The native extension is optional: if it is not present (for example in a
source checkout without a build), the pure-Python APIs below still import
and work. Code that needs the compiled APIs will raise ``ImportError`` at
the point of use instead of at package import.
"""

# The absence of the compiled extension is tolerated (source checkouts
# without a build); a PRESENT-but-broken extension is not — an extension
# that fails to load (ABI mismatch, missing shared-library dependency)
# re-raises with the loader's message instead of silently degrading.
_NATIVE_IMPORT_ERROR = None
try:
    from ._pycudaq_algorithms import *
    from ._pycudaq_algorithms import __version__
except ModuleNotFoundError as exc:
    if exc.name != __name__ + "._pycudaq_algorithms":
        raise
    _NATIVE_IMPORT_ERROR = exc
    __version__ = "CUDA-Q Algorithms (compiled extension not built)"

# Pure-Python classical preprocessing (no compiled-extension dependency).
from . import double_factorization

# Pure-Python quantum primitives (no compiled-extension dependency).
# ``fermion`` is imported explicitly so the pure package deterministically
# owns the ``cudaq_algorithms.fermion`` attribute even when the compiled
# extension's ``import *`` above also exposed a submodule of that name.
from . import (block_encoding, common_kernels, fermion, pauli_lcu, qsvt,
               qubitization, sim_utils, trotter)
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
