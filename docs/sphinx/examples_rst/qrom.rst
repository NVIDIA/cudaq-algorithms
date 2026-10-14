QROM examples
=============

Coherent and measured table lookups, with independent complex-vector checks.

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

Measured SELECT-SWAP cache cleanup
----------------------------------

Compare cleanup choices, validate the arbitrary-output XOR map, and return a
classical bit using ``cudaq.run``. See :doc:`../guide/measured_qrom` for the
backend and register contracts.

.. literalinclude:: ../examples/python/measured_qrom.py
   :language: python
   :start-after: [Begin Documentation]

Measured packet lookup and image erasure
-----------------------------------------

Use fixed-width SELECT-COPY packets and an entangled borrowed bank. The load
accepts arbitrary output; the separate eraser consumes only a known table
image, using its own correctly sized clean-prefix/borrowed-suffix ladder.

.. literalinclude:: ../examples/python/qrom_select_copy_measured.py
   :language: python
   :start-after: [Begin Documentation]

Rotations during a retained-bank epoch
--------------------------------------

Run begin, every difference-table step, and end around three noncommuting
rotations on separate data. An independent dense oracle checks the full
state, including the borrowed bank, external reference, and clean output.

.. literalinclude:: ../examples/python/qrom_retained_rotations.py
   :language: python
   :start-after: [Begin Documentation]
