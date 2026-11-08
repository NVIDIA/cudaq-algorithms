Preprocessing examples
==========================

Classical preprocessing that shapes an electronic-structure Hamiltonian before
it ever reaches a quantum circuit. These examples double-factorize the
two-electron integrals and trace the result through to block-encoding cost.

.. seealso::

   :doc:`../guide/preprocessing` — the chemistry bridges and double
   factorization these examples build on.

Fermion mappings on an interacting lattice
-------------------------------------------

Compare JW, BK, original superfast, and binary-tree generalized superfast
mappings on one four-site spinless ring. An independent occupation-basis
NumPy Hamiltonian checks coefficients and signs. Each superfast parity
sector is restricted to the loop stabilizers' joint +1 eigenspace before
all eight physical eigenvalues are compared, including multiplicities.
The host-side tutorial reports Pauli weights, qubits, and code dimensions;
it requires no chemistry or phase-generation package.

.. seealso::

   :doc:`../guide/preprocessing` for tensor and parity contracts;
   example 09 on :doc:`getting_started` for a related lattice application.

.. literalinclude:: ../examples/python/fermion_mappings_lattice.py
   :language: python
   :start-after: [Begin Documentation]

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
