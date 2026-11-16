Chemistry-primitive examples
============================

Fermionic transformations applied to an existing register, each checked
against its exact state.

.. seealso::

   :doc:`../guide/chemistry_primitives` — the primitives behind these
   examples.

Fermionic basis changes with explicit QROM
------------------------------------------

Applies a fixed two-mode basis change, enables it with an external
control, and then selects between two transformations with a quantum
address: a `QROM` lookup loads the angle words, the programmable plan
consumes them, and the inverse lookup clears them. See
:doc:`../guide/fermion_basis_change`.

.. literalinclude:: ../examples/python/fermion_basis_change.py
   :language: python
   :start-after: [Begin Documentation]
