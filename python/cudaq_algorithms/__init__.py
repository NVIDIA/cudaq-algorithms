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

try:
    from ._pycudaq_algorithms import *
    from ._pycudaq_algorithms import __version__
except ImportError:
    # Pure-Python-only environment (no compiled extension). The C++-backed
    # APIs are unavailable, but everything imported below still works.
    __version__ = "CUDA-Q Algorithms (pure Python; native extension not built)"

# Pure-Python quantum primitives (no compiled-extension dependency).
from . import block_encoding, pauli_lcu, qsvt, qubitization, sim_utils
from .block_encoding import BlockEncoding
from .pauli_lcu import (PauliLCU, apply, controlled_select, prepare,
                        reflect_about_zero, select, state_from, unprepare,
                        walk)
from .qsvt import (ADJOINT, FORWARD, PhaseSequence, QSVT,
                   apply_controlled_phase_sequence, apply_phase_sequence,
                   controlled_signal_phase, recover_real_time_evolution,
                   signal_phase)
from .qubitization import (Walk, adjoint_walk, controlled_adjoint_walk,
                           controlled_reflect_about_prepare,
                           controlled_reflect_about_zero, controlled_walk,
                           reflect_about_prepare, reflection_observable,
                           select_observable)
