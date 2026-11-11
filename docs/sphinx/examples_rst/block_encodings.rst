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

A Hamiltonian is a weighted sum of simple operations. To use it in quantum
algorithms, we need a larger circuit that contains a scaled copy of that sum
in the part where its extra qubits are zero. This is a block encoding. This
example follows one small Hamiltonian from its coefficients to an executable
block encoding, then uses that encoding in a quantum walk and a polynomial
transformation.

The comparison asks what changes when the same Hamiltonian uses exact state
preparation or finite-precision alias sampling. You should learn how
preparation weights, the selected operations, and the normalization determine
the encoded matrix; how extra qubits contribute to cost; and how to compare
accuracy across implementations. The NumPy checks separate errors caused by
rounded coefficients from errors in executing the circuit, so the printed
resource savings or costs can be interpreted alongside the approximation.

A guided tour of the `PauliLCU` block-encoding API: build the encoding from a
spin operator, inspect its sizes and normalization, and exercise the device
kernels (`pauli_lcu.prepare`, the select, and the walk) it exposes.

The final sections compare exact and alias PREPARE/SELECT at ``mu=2``,
``3`` and ``4`` using the same Hamiltonian and kernel-factory consumers.
The output summarizes Hamiltonian approximation error, executed algorithms,
and resource cost. On this fixture ``mu=3`` has the same rounded weights as
``mu=2``, while ``mu=4`` reduces the error by about 2.6 times. The resource
table accounts for alias workspace and lists every gate family. A separate
small fixture prints odd/even measured moments. Independent NumPy checks
remain active and are quiet on success; pass ``--verbose`` for the detailed
diagnostics. See :doc:`../guide/block_encodings` for the accuracy/resource
tradeoff and the distinction between rounding and circuit error.

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
