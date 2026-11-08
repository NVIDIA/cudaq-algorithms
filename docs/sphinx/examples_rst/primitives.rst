Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Reversible modular arithmetic
-----------------------------

Compare CDKM and QFT arithmetic, use unsigned comparison flags, and restore
all work registers. The example checks full complex amplitudes and overflow.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/reversible_arithmetic.py
   :language: python
   :start-after: [Begin Documentation]
