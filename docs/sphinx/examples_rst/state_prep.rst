State-preparation examples
==============================

Preparing the input register a primitive will consume. These examples build
chemistry-style states with the composable `stateprep` device kernels, each one
checked against a dense reference.

.. seealso::

   :doc:`../guide/state_prep` — the ansatz kernels, operator pools, and
   injection contract behind these examples.

Hartree-Fock and fixed-parameter UCCSD state preparation
--------------------------------------------------------

Prepares a 4-qubit / 2-electron UCCSD ansatz state at fixed amplitudes
(H2-style). It shows how the Hartree-Fock reference and the fixed-parameter
UCCSD singles and doubles compose into a single `state_prep` kernel.

.. literalinclude:: ../examples/python/hartree_fock_ucc.py
   :language: python
   :start-after: [Begin Documentation]

Givens-rotation Slater determinant preparation
----------------------------------------------

Prepares a real 4-orbital / 2-electron determinant and a complex 5-orbital /
3-electron determinant with the composable Givens-rotation `stateprep` kernels.
Both are validated against their dense Slater-determinant references.

.. literalinclude:: ../examples/python/givens_slater_determinant.py
   :language: python
   :start-after: [Begin Documentation]

Fermionic basis changes with explicit QROM
------------------------------------------

Applies a fixed two-mode basis change, prepares one orbital from a
reference occupation, enables the fixed change with an external control,
and then selects between two transformations with a quantum address: a
`QROM` lookup loads the angle words, the programmable plan consumes them,
and the inverse lookup clears them. The last check reflects about the
selected orbital. Each step is compared with its exact state. See
:doc:`../guide/fermion_basis_change`.

.. literalinclude:: ../examples/python/fermion_basis_change.py
   :language: python
   :start-after: [Begin Documentation]
