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

How the example encoding works, in brief: the factorized Hamiltonian is
regrouped so every term is diagonal in some rotated orbital basis -- the
one-body part in the eigenbasis of the corrected one-body matrix, each
factorization leaf in its own eigenbasis. SELECT walks through these
frames with *uncontrolled* Givens-rotation networks (only the Z words and
sign phases carry ancilla controls, so unselected frames telescope to the
identity), and the encoding's normalization reproduces the published LCU
one-norm of ``double_factorization_one_norm(..., "lcu")`` exactly, up to
the identity term. Compressing the factorization (fewer leaves) shrinks
the term count and, typically though not monotonically, the
normalization -- the knob a flat Pauli expansion does not have. The full
construction is documented in the example module's docstrings.

.. literalinclude:: ../examples/python/df_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]
