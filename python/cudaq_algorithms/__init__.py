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
  fermion and state-preparation utilities. These require the native
  extension built against CUDA-Q.
- Pure-Python modules (:mod:`.pauli_lcu`, :mod:`.qubitization`, :mod:`.qsvt`,
  :mod:`.sim_utils`): quantum primitives implemented as CUDA-Q Python
  kernels. These only require the ``cudaq`` Python package.

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

# Pure-Python quantum primitives (no compiled-extension dependency).
from . import (block_encoding, common_kernels, pauli_lcu, qsvt, qubitization,
               sim_utils)
from .block_encoding import BlockEncoding
from .common_kernels import (controlled_reflect_about_zero,
                             controlled_signal_phase, reflect_about_zero,
                             signal_phase)
from .pauli_lcu import (PauliLCU, adjoint_walk, apply,
                        apply_controlled_phase_sequence, apply_phase_sequence,
                        controlled_adjoint_walk,
                        controlled_reflect_about_prepare, controlled_select,
                        controlled_walk, prepare, reflect_about_prepare,
                        select, select_observable, state_from, unprepare, walk)
from .qsvt import (ADJOINT, FORWARD, PhaseSequence, QSVT,
                   recover_real_time_evolution)
from .qubitization import Walk, reflection_observable
