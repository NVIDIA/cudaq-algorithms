Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Reversible modular arithmetic
-----------------------------

Compare CDKM and QFT arithmetic, use unsigned comparison flags, and restore
all work registers. The example checks full complex amplitudes and overflow.

It starts with an explicit threshold oracle: prepare ``(|2> + |5>)/sqrt(2)``,
compute ``x >= 3``, mark its flag with Z, and uncompute. The printed states
show the relative minus sign on x=5 and the restored flag and workspace.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/reversible_arithmetic.py
   :language: python
   :start-after: [Begin Documentation]
