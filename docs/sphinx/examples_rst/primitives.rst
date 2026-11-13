Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Reversible modular arithmetic
-----------------------------

Many quantum algorithms make decisions from numbers stored in qubits: add an
offset, compare a value with a threshold, or mark the values that satisfy a
condition. This example builds those operations as reversible circuits. Its
threshold demonstration computes a yes/no flag, uses that flag to place a
relative minus sign on the matching input, and then undoes the comparison so
the flag and scratch qubits can be reused.

The lessons are to track what every register means, account for wraparound in
fixed-width arithmetic, and undo temporary calculations after using their
result. Addition and subtraction, two constant-arithmetic constructions, and
unsigned comparisons are checked against classical predictions on
superpositions. You should be able to distinguish a correct numerical output
from a complete reversible operation that also preserves phases and restores
its workspace, and read the resource counts as the cost of those requirements.

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
