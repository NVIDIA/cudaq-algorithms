Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Coherent QROM lookup
--------------------

Read a table coherently, use a loaded bit, and unlookup. The example checks
SELECT and SELECT-SWAP, native and wrapped control, and clean workspace.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/qrom_lookup.py
   :language: python
   :start-after: [Begin Documentation]
