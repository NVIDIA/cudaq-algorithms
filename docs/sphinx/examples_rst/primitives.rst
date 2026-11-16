Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Coherent QROM lookup
--------------------

A quantum algorithm may need different classical data for different branches
of a quantum state. This example shows how a quantum address can look up a
table entry while preserving the superposition and its relative phases. It
then uses the loaded data to change the state and reverses the lookup,
demonstrating how table data can influence a calculation while its temporary
storage is returned to zero.

You should come away able to identify the address, output, and scratch
registers and understand why their initial values and cleanup matter. The same
lookup is built with SELECT and SELECT-SWAP strategies and with an external
enable, making their qubit and gate tradeoffs visible. Full-state comparisons
with an independently constructed reference check that the chosen entry, the
phase change, and workspace restoration all agree.

Individual validation checks are shown with ``--verbose``. All checks still
run by default, and failures are always reported.

Read a table coherently, use a loaded bit, and unlookup. The example checks
SELECT and SELECT-SWAP, native and wrapped control, and clean workspace.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/qrom_lookup.py
   :language: python
   :start-after: [Begin Documentation]
