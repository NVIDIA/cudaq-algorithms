Preprocessing examples
==========================

Classical preprocessing that shapes an electronic-structure Hamiltonian before
it ever reaches a quantum circuit. These examples double-factorize the
two-electron integrals and trace the result through to block-encoding cost.

Double factorization of the two-electron integrals
--------------------------------------------------

Explicit (X-DF) and compressed (C-DF) double factorization of the two-electron
integrals, following Cohn, Motta, and Parrish. Generates restricted
Hartree-Fock molecular-orbital integrals for H2O/STO-3G with PySCF, then
factorizes the ERI tensor both the exact and the least-squares-optimized way.

.. literalinclude:: ../examples/python/double_factorization.py
   :language: python
   :start-after: [Begin Documentation]

From double-factorized integrals to LCU/QSVT cost
-------------------------------------------------

Takes H2/STO-3G integrals, double-factorizes the two-electron tensor, and for
each leaf count reconstructs the truncated tensor and bridges it to a qubit
`PauliLCU` block encoding. The result is an end-to-end view of how DF
compression drives the downstream LCU and `QSVT` cost.

.. literalinclude:: ../examples/python/df_compression_to_qsvt.py
   :language: python
   :start-after: [Begin Documentation]
