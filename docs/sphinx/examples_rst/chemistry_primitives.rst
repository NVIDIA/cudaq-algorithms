Chemistry-primitive examples
============================

Fermionic transformations applied to an existing register, each checked
against its exact state.

.. seealso::

   :doc:`../guide/chemistry_primitives` — the primitives behind these
   examples.

Fermionic basis changes with explicit QROM
------------------------------------------

Fermionic simulation often becomes simpler after mixing the orbital modes into
a different basis. A Givens rotation performs one such mixing between a pair
of modes while preserving particle number. This example first applies a fixed
basis change, then shows how a quantum address can load angle words from QROM
so one programmable circuit performs the transformation selected by that
address.

You should learn the difference between enabling a fixed operation and
supplying its parameters as quantum data. A selected-orbital reflection then
transforms into the chosen orbital basis, marks occupation with a sign, and
transforms back, providing a useful building block for chemistry encodings.
Complete-state NumPy checks follow the address phases, particle sectors, and
cleared angle/scratch registers. Comparing two angle precisions shows how a
represented transformation can match its circuit closely while still
approximating the desired transformation.

Individual validation checks are shown with ``--verbose``. All checks still
run by default, and failures are always reported.

Applies a fixed two-mode basis change, enables it with an external
control, and then selects between two transformations with a quantum
address: a `QROM` lookup loads the angle words, the programmable plan
consumes them, and the inverse lookup clears them. See
:doc:`../guide/fermion_basis_change`.

The same script uses a partial program for coherent selected orbital
reflections, controls only their middle occupation phase, and compares
three- and four-bit off-grid programs with independent NumPy and
Jordan-Wigner references. Complete-state checks include angle and ladder
restoration and phase between particle sectors.

.. literalinclude:: ../examples/python/fermion_basis_change.py
   :language: python
   :start-after: [Begin Documentation]
