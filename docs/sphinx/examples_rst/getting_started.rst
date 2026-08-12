Getting-started walkthrough
===============================

An ordered learning path through the library. Each example builds on the one
before it, from the single core idea -- block-encode a Hamiltonian and walk it
-- up to bringing your own encoding. Every example verifies its own claims
against an independent dense reference, so each script runs to completion or
fails loudly.

.. seealso::

   The :doc:`user guide <../guide/block_encodings>` pages cover each
   primitive in depth; :doc:`../getting_started/index` sets up the
   installation these scripts assume.

Quickstart: block-encode a Hamiltonian and walk it
---------------------------------------------------

The five-minute tour of the core idea. A Hamiltonian is not a circuit, so you
block-encode it: `PauliLCU` hides `H / alpha` inside a larger unitary on a few
ancilla qubits, and the qubitization `Walk` turns powers of that unitary into
Chebyshev moments checked against a dense matrix.

.. literalinclude:: ../examples/python/01_quickstart_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]

Hamiltonian simulation, two independent ways
--------------------------------------------

Compute the time evolution ``exp(-iHt)|psi>`` with two different primitives and
check both against dense linear algebra: `QSVT` with Jacobi-Anger phase factors,
and `Trotter` product formulas with an error-vs-steps trade-off. Because the two
constructions are completely different, their agreement is a strong test.

.. literalinclude:: ../examples/python/02_hamiltonian_simulation.py
   :language: python
   :start-after: [Begin Documentation]

From a molecule to its ground-state energy
------------------------------------------

The full classical to quantum to classical loop, end to end: PySCF mean field,
Jordan-Wigner qubit Hamiltonian, `PauliLCU` block encoding, `Walk` moments, and
a classical Krylov solve for the ground-state energy checked against FCI.

.. literalinclude:: ../examples/python/03_chemistry_to_ground_state.py
   :language: python
   :start-after: [Begin Documentation]

Double factorization and the BlockEncoding protocol
---------------------------------------------------

Double factorization rewrites the two-electron tensor as a sum of low-rank
leaves, giving a compression dial you can watch trade accuracy for a cheaper
block encoding. The same pipeline scales to larger molecules and plugs into the
structural `BlockEncoding` protocol shared by every consumer.

.. literalinclude:: ../examples/python/04_double_factorization_and_the_protocol.py
   :language: python
   :start-after: [Begin Documentation]

State preparation and injection
-------------------------------

Every primitive factory (`Walk`, `QSVT`, `Trotter`) accepts a `state_prep`
kernel argument. Pass one and the factory returns a zero-argument, hardware-shaped
circuit with no statevector crossing the API boundary -- the seam that makes the primitives
hardware-ready and the seam a tensor-network state-prep compiler would plug into.

.. literalinclude:: ../examples/python/05_state_prep_and_injection.py
   :language: python
   :start-after: [Begin Documentation]

Bring your own block encoding
-----------------------------

`Walk` and `QSVT` are generic over the `BlockEncoding` protocol: any object
exposing the right members works, with no inheritance. This example implements
the protocol from scratch for a two-term single-qubit LCU and watches the walk
measure correct Chebyshev moments on it.

.. literalinclude:: ../examples/python/06_bring_your_own_encoding.py
   :language: python
   :start-after: [Begin Documentation]

Solve a linear system with QSVT
-------------------------------

The payoff of QSVT's "pick a polynomial" design: example 2 approximated
``exp(-ixt)`` for time evolution; swap the phase sequence for one
approximating ``1/x`` and the *same* `PauliLCU` + `QSVT` machinery becomes a
quantum linear solver -- the QSVT reading of HHL. A 5x5 symmetric
positive-definite system is padded to three qubits, block-encoded, and
inverted with a Childs-Kothari-Somma polynomial (degree set by the condition
number); the good-subspace state is proportional to ``A^-1 b``, with the
scale recovered classically. Verified against ``numpy.linalg.solve``.
Requires ``qsppack`` for the phase factors, like example 2.

.. literalinclude:: ../examples/python/07_matrix_inversion_qsvt.py
   :language: python
   :start-after: [Begin Documentation]
