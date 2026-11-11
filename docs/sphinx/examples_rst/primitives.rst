Circuit primitive examples
==========================

These standalone examples use public APIs and fail if an independent
numerical reference or workspace check disagrees with the circuit.

Alias sampling and an LCU
-------------------------

Quantum models are often written as weighted sums of simple operations, and
later algorithms need a circuit representation of that sum. This example
builds one from an X gate that flips a data qubit and a signed controlled-Z
operation that changes selected amplitudes' signs. An extra index register
acts as a coherent selector, with alias sampling supplying the relative
weights. Preparing the selector, applying its chosen operations, and reversing
the preparation places the weighted sum in the part of the larger circuit
where all its extra qubits are zero.

You should learn why the sampling weights stay positive while a negative
coefficient is supplied by a phase in the selected operation, and why the
sampler's temporary information can remain entangled with the index until
preparation is reversed. The example also distinguishes scratch that must
return to zero from amplitude that legitimately remains outside the encoded
block after selection. Independent NumPy checks compare the requested weights,
their finite-precision replacements, and the resulting matrix before the same
encoding is used by Walk and QSVT.

Prepare a weighted index marginal, inspect garbage and discretization, then
compose a mixed-sign non-Pauli LCU and validate its Walk/QSVT consumers.

.. seealso::

   :doc:`../guide/primitives` for register contracts and interpretation.

.. literalinclude:: ../examples/python/alias_sampling_lcu.py
   :language: python
   :start-after: [Begin Documentation]
