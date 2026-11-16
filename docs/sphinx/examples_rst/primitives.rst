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

Reversible modular arithmetic
-----------------------------

Compare CDKM and QFT arithmetic, use unsigned comparison flags, and restore
all work registers. The example checks full complex amplitudes and overflow.

It starts with an explicit threshold oracle: prepare ``(|2> + |5>)/sqrt(2)``,
compute ``x >= 3``, mark its flag with Z, and uncompute. The printed states
show the relative minus sign on x=5 and the restored flag and workspace.

All checks run in the default mode, which prints the demonstrations and a
short validation summary. Add ``--verbose`` to print each check's numerical
error. A failed check always reports its details and exits with an error.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/reversible_arithmetic.py
   :language: python
   :start-after: [Begin Documentation]

Alias sampling and an LCU
-------------------------

Individual validation checks are shown with ``--verbose``. All checks still
run by default, and failures are always reported.

Prepare a weighted index marginal, inspect garbage and discretization, then
compose a mixed-sign non-Pauli LCU and validate its Walk/QSVT consumers.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/alias_sampling_lcu.py
   :language: python
   :start-after: [Begin Documentation]
