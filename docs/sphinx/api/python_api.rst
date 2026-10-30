********************
Python API Reference
********************

The object surface (``PauliLCU``, ``Walk``, ``QSVT``, ``Trotter``, ...)
is re-exported from the package root
(``cudaq_algorithms.PauliLCU``); the composable device kernels stay in their
module namespaces. Symbols are documented below at their defining module.

Block encodings
===============

.. autoclass:: cudaq_algorithms.pauli_lcu.PauliLCU
   :members:

.. autofunction:: cudaq_algorithms.pauli_lcu.select_observable

.. autoclass:: cudaq_algorithms.lcu.LCUBlockEncoding
   :members:

.. autoclass:: cudaq_algorithms.block_encoding.BlockEncoding
   :members:

.. automodule:: cudaq_algorithms.combinators
   :members:

Sparse-access block encodings
=============================

.. autofunction:: cudaq_algorithms.sparse.encode_sparse

.. autoclass:: cudaq_algorithms.sparse.SparseOracleEncoding
   :members:

.. autoclass:: cudaq_algorithms.sparse.OracleKernels
   :members:

.. autofunction:: cudaq_algorithms.sparse.banded_oracles

.. autofunction:: cudaq_algorithms.sparse.qrom_oracles

.. autofunction:: cudaq_algorithms.sparse.quantized_angle

Qubitization and QSVT
=====================

.. autoclass:: cudaq_algorithms.qubitization.Walk
   :members:

.. autofunction:: cudaq_algorithms.qubitization.reflection_observable

.. autoclass:: cudaq_algorithms.qsvt.QSVT
   :members:

.. autoclass:: cudaq_algorithms.qsvt.PhaseSequence
   :members:

.. autofunction:: cudaq_algorithms.qsvt.recover_real_time_evolution

.. autodata:: cudaq_algorithms.qsvt.FORWARD

.. autodata:: cudaq_algorithms.qsvt.ADJOINT

Trotter
=======

.. automodule:: cudaq_algorithms.trotter
   :members:

Circuit primitives
==================

.. automodule:: cudaq_algorithms.primitives
   :members:

Preprocessing
=============

.. automodule:: cudaq_algorithms.chemistry
   :members:

.. automodule:: cudaq_algorithms.fermion
   :members:

.. automodule:: cudaq_algorithms.double_factorization
   :members:

State preparation
=================

.. automodule:: cudaq_algorithms.stateprep
   :members:

Simulation utilities
====================

.. automodule:: cudaq_algorithms.sim_utils
   :members:

.. autofunction:: cudaq_algorithms.common_kernels.state_from
