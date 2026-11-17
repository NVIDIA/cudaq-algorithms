Preprocessing examples
==========================

Classical preprocessing turns model coefficients or electronic-structure
integrals into quantum-ready operators. These examples compare fermion
mappings, double-factorize two-electron integrals, and trace the results
through to block-encoding cost.

.. seealso::

   :doc:`../guide/preprocessing` — the chemistry bridges and double
   factorization these examples build on.

Fermion mappings on an interacting lattice
-------------------------------------------

Before simulating electrons or other fermions, we must translate their
creation, annihilation, and interaction terms into operations on qubits. This
example applies several translations to the same lattice model, where
particles can hop between sites and interact when neighboring sites are
occupied. The aim is to keep the physical model fixed while examining how its
qubit representation changes.

You should learn how to check that different mappings preserve the model's
energies and how to interpret differences in Pauli terms, operator weights,
and qubit counts. Some representations include states that do not belong to
the intended physical problem; their parity and stabilizer constraints
identify the subspace to compare. An independent occupation-basis Hamiltonian
supplies the reference, making the signs and physical-subspace checks explicit
before any resource comparison is interpreted.

Individual validation checks are shown with ``--verbose``. All checks still
run by default, and failures are always reported.

Compare JW, BK, original superfast, and binary-tree generalized superfast
mappings of one spinless lattice model. On a four-site ring an independent
occupation-basis NumPy Hamiltonian checks coefficients and signs. Each
superfast parity sector is restricted to the loop stabilizers' joint +1
eigenspace before all eight physical eigenvalues are compared, including
multiplicities.

A second table repeats the comparison on open square lattices from 4x4 to
20x20, up to 400 modes, where the mappings separate. The JW maximum weight
grows with the lattice side and the BK weight with the logarithm of the mode
count: BK is the heavier of the two up to 10x10, they are equal at 14x14,
and BK is lighter at 20x20. Both superfast forms keep a fixed weight at the
cost of more qubits and one stabilizer per independent cycle. Nothing is simulated at
those sizes: Pauli weights, qubit counts, and stabilizer counts are read
from the mapped operators, and the checks are exact statements about their
Pauli words. The host-side tutorial requires no chemistry or
phase-generation package.

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
