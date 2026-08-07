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

.. autoclass:: cudaq_algorithms.block_encoding.BlockEncoding
   :members:

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

Trotter
=======

.. automodule:: cudaq_algorithms.trotter
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
