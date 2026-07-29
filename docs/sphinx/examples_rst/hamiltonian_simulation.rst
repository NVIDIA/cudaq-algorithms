Hamiltonian-simulation examples
===================================

Realizing time evolution ``exp(-iHt)`` with two contrasting primitives: the
ancilla-based `QSVT` path and the ancilla-free `Trotter` product formula. Both
are checked against exact diagonalization.

Hamiltonian simulation with PauliLCU and QSVT
---------------------------------------------

Evolves a 4-qubit Pauli Hamiltonian by block-encoding it with `PauliLCU` and
applying a `QSVT` polynomial built from QSPPACK-generated phases. The
time-evolved state is checked against exact diagonalization.

.. literalinclude:: ../examples/python/hamiltonian_simulation_qsvt.py
   :language: python
   :start-after: [Begin Documentation]

Suzuki-Trotter simulation of a chemistry-style Hamiltonian
----------------------------------------------------------

Simulates a chemistry-style Hamiltonian, hard-coded as Pauli terms, with the
`Trotter` product-formula primitive. It shows two ways to run the same evolution
and the error-vs-steps trade-off, checked against dense linear algebra.

.. literalinclude:: ../examples/python/trotter_chemistry.py
   :language: python
   :start-after: [Begin Documentation]
