Block-encoding examples
===========================

Two ways to hide a Hamiltonian inside a larger unitary. These examples walk the
flat `PauliLCU` API and put the structured double-factorized encoding head to
head with it.

Walkthrough of the PauliLCU block-encoding API
----------------------------------------------

A guided tour of the `PauliLCU` block-encoding API: build the encoding from a
spin operator, inspect its sizes and normalization, and exercise the device
kernels (`pauli_lcu.prepare`, the select, and the walk) it exposes.

.. literalinclude:: ../examples/python/pauli_lcu_demo.py
   :language: python
   :start-after: [Begin Documentation]

The double-factorized block encoding vs a flat PauliLCU
-------------------------------------------------------

Builds a random two-orbital electronic-structure system, block-encodes it both
with the structured double-factorized encoding and with a flat `PauliLCU`, and
compares them on ancilla count and normalization `alpha` -- showing where the
DF structure pays off.

.. literalinclude:: ../examples/python/df_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]
