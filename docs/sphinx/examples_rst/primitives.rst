Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Alias sampling and an LCU
-------------------------

Prepare a weighted index marginal, inspect garbage and discretization, then
compose a mixed-sign non-Pauli LCU and validate its Walk/QSVT consumers.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/alias_sampling_lcu.py
   :language: python
   :start-after: [Begin Documentation]
