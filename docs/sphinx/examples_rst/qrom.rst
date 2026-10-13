QROM examples
=============

Coherent table lookups used as binary controls.

.. seealso::

   :doc:`../guide/qrom` — the lookup constructions, their costs and the
   chain contract this example relies on.

Controlled rotations from a fused lookup chain
----------------------------------------------

Loads three rotation words from one address register, uses them as controls
on a target qubit, and unloads them again through XOR-difference lookups,
checking the complete complex state against a dense oracle and comparing the
fused chain's Toffoli count with the separate-lookup reference.

.. literalinclude:: ../examples/python/qrom_controlled_rotations.py
   :language: python
   :start-after: [Begin Documentation]
