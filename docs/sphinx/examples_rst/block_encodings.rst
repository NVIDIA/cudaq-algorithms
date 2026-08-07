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

Runs the worked double-factorized encoding (the bring-your-own-encoding
example ``df_encoding.py``, next to this script) head to head with a flat
`PauliLCU` on real molecules from PySCF -- comparing alpha and term counts,
sweeping the truncation dial, re-optimizing kept leaves with RC-DF, and
verifying the encoded blocks against a sparse Jordan-Wigner reference.

.. literalinclude:: ../examples/python/df_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]
