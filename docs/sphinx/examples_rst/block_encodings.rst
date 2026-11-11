Block-encoding examples
===========================

Two ways to hide a Hamiltonian inside a larger unitary. These examples walk the
flat `PauliLCU` API and put the structured double-factorized encoding head to
head with it.

.. seealso::

   :doc:`../guide/block_encodings` — the ``PauliLCU`` API and the
   ``BlockEncoding`` protocol these examples exercise.

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

A molecular Hamiltonian contains many interaction coefficients, and the way
those coefficients are organized affects the quantum circuit built from them.
This example uses double factorization to express the interactions through
simpler contributions in rotated orbital bases. Starting from real molecular
integrals, it compares that construction with a Pauli expansion and asks how
keeping fewer factors or refining them changes the encoded Hamiltonian and its
resource requirements.

You should learn how the choice of orbital basis turns the selected
interactions into simple phase operations, and how QROM-loaded angle words let
a quantum index choose that basis coherently. The fixed-angle and programmable
constructions share the same block-encoding interface, allowing the same Walk
and QSVT consumers to use them. Independent references separate factorization
and coefficient errors from angle rounding and circuit error, so readers can
evaluate accuracy together with normalization, qubit counts, and the cost of
loading, using, and clearing the rotation data.

Runs the worked double-factorized encoding (the bring-your-own-encoding
example ``df_encoding.py``, next to this script) head to head with a flat
`PauliLCU` on real molecules from PySCF -- comparing alpha and term counts,
sweeping the truncation dial, re-optimizing kept leaves with RC-DF, and
verifying the encoded blocks against a sparse Jordan-Wigner reference.

How the example encoding works, in brief: the factorized Hamiltonian is
regrouped so every term is diagonal in some rotated orbital basis -- the
one-body part in the eigenbasis of the corrected one-body matrix, each
factorization leaf in its own eigenbasis. SELECT walks through these
frames in its fixed backend with *uncontrolled* Givens networks (only Z words and
sign phases carry ancilla controls, so unselected frames telescope to the
identity), and the encoding's normalization reproduces the published LCU
one-norm of ``double_factorization_one_norm(..., "lcu")`` exactly, up to
the identity term. Compressing the factorization (fewer leaves) shrinks
the term count and, typically though not monotonically, the
normalization -- the knob a flat Pauli expansion does not have. The full
construction is documented in the example module's docstrings.

The default H2 run also selects one real spatial frame coherently with
QROM-loaded angle words. It compares fixed and represented operators,
clears angle/ladder workspace, and executes Walk and QSVT against NumPy
references. Use ``--rotation-backend {fixed,qrom,both}`` and
``--angle-bits 4``. A host-only four/six/eight-bit sweep separates angle
rounding from circuit error; larger QROM cases are resource-only.
See the guide's real spatial frame section for register conventions and
the distinction from the papers' complete DF algorithms.

.. literalinclude:: ../examples/python/df_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]
